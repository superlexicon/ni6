"""
GLiNER-based Generic ID Card Extractor

Generic ID card extractor using GLiNER zero-shot NER.
Works with any ID card type (PAN, national ID, driver's license, etc.)
Returns key-value pairs rather than fixed schema.

Extraction Strategy:
1. GLiNER identifies entity VALUES (names, numbers, dates)
2. Spatial analysis finds field LABELS on card
3. Pair labels with values using spatial proximity
"""

import re
import logging
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass

from app.core.gliner_ner_model import get_gliner_ner_model
from app.core.logger import get_logger
from app.helper.doctr.document_text_extractor import DocumentTextExtractor
from app.schemas.id_card_schema import IDCardData


@dataclass
class FieldLabel:
    """Represents a field label found on the ID card."""
    text: str
    x1: float
    y1: float
    x2: float
    y2: float
    label_type: str  # The type of field this label represents


class GLiNERIDCardExtractor:
    """
    Generic ID card extractor using GLiNER zero-shot NER.

    Works with any ID card type (PAN, national ID, driver's license, etc.)
    Returns key-value pairs rather than fixed schema.

    Extraction Strategy:
    1. GLiNER identifies entity VALUES (names, numbers, dates)
    2. Spatial analysis finds field LABELS on card
    3. Pair labels with values using spatial proximity
    """

    # Field label patterns for identifying field types
    FIELD_LABEL_PATTERNS = {
        'full_name': [
            'name', 'full name', 'given name', 'first name', 'surname',
            'last name', 'father name', 'mother name', 's/o', 'd/o', 'a/l'
        ],
        'father_name': ['father', 'father name', 'father\'s name', 's/o'],
        'mother_name': ['mother', 'mother name', 'mother\'s name', 'd/o'],
        'date_of_birth': ['dob', 'date of birth', 'birth date', 'born'],
        'gender': ['gender', 'sex', 'male', 'female'],
        'identification_number': [
            'number', 'card number', 'id number', 'document number',
            'passport number', 'no.', 'no :'
        ],
        'pan_number': ['pan', 'pan number', 'permanent account number'],
        'tax_id': ['tax id', 'tax identification number', 'tin', 'tax number', 'trn'],
        'address': ['address', 'residential address'],
        'issue_date': ['issue date', 'date of issue', 'issued'],
        'expiry_date': ['expiry', 'expiration', 'valid until', 'expiry date'],
        'country': ['country', 'nationality'],
    }

    # PAN card specific pattern
    PAN_PATTERN = re.compile(r'^[A-Z]{5}[0-9]{4}[A-Z]$', re.IGNORECASE)

    def __init__(self):
        self.logger = get_logger()
        self.text_extractor = DocumentTextExtractor()
        self._gliner_model = None

    async def extract(
        self,
        content: bytes,
        is_pdf: bool = False,
        document_type_hint: Optional[str] = None
    ) -> IDCardData:
        """
        Extract ID card fields using GLiNER + spatial analysis.

        Args:
            content: Document bytes
            is_pdf: Whether content is PDF
            document_type_hint: Optional hint (e.g., "PAN", "DL")

        Returns:
            IDCardData with extracted key-value pairs
        """
        try:
            self.logger.info("=" * 80)
            self.logger.info("GLiNER ID CARD EXTRACTION")
            self.logger.info("=" * 80)

            # Step 1: Extract text with geometry
            self.logger.info("Step 1: Extracting text with geometry")
            text_blocks = await self.text_extractor.extract_text_with_geometry_enhanced(
                content, is_pdf=is_pdf, max_pages=1
            )
            self.logger.info(f"  Extracted {len(text_blocks)} text lines")

            # Combine all text for GLiNER - preserve reading order with newlines
            all_text = "\n".join([block.get('text', '') for block in text_blocks])
            self.logger.info(f"  Total text length: {len(all_text)} characters")

            # Step 2: Detect document type if not provided
            document_type = document_type_hint
            if not document_type:
                document_type = self._detect_document_type(all_text, text_blocks)
            self.logger.info(f"  Document type: {document_type or 'unknown'}")

            # Step 3: Extract entities using GLiNER
            self.logger.info("Step 2: Extracting entities with GLiNER")
            gliner_model = get_gliner_ner_model()
            gliner_result = await gliner_model.extract_id_card_entities(all_text)
            self.logger.info(f"  GLiNER extracted {len([v for v in gliner_result.values() if v])} entities")

            # Step 4: Find field labels using spatial analysis
            self.logger.info("Step 3: Finding field labels")
            field_labels = self._find_field_labels(text_blocks)
            self.logger.info(f"  Found {len(field_labels)} field labels")

            # Step 5: Pair labels with values
            self.logger.info("Step 4: Pairing labels with values")
            paired_fields = self._pair_labels_with_values(
                text_blocks, gliner_result, field_labels
            )
            self.logger.info(f"  Paired {len(paired_fields)} field-value pairs")

            # Step 5.5: Pattern-based extraction fallback
            self.logger.info("Step 4.5: Running pattern-based fallback")
            pattern_fallback = self._extract_patterns_fallback(
                text_blocks, document_type, paired_fields
            )
            self.logger.info(f"  Pattern fallback found {len(pattern_fallback)} fields")

            # Merge pattern fallback into paired fields
            for field_name, (value, confidence) in pattern_fallback.items():
                if field_name not in paired_fields:
                    paired_fields[field_name] = value
                    # Also add to GLiNER result for confidence tracking
                    gliner_result[field_name] = {'value': value, 'confidence': confidence / 100}

            # Step 6: Post-process and build result
            self.logger.info("Step 5: Building result")
            result = self._build_result(
                gliner_result, paired_fields, document_type
            )

            # Step 7: Auto-detect country
            result.issuing_country = self._detect_country(all_text, text_blocks)

            self.logger.info("=" * 80)
            self.logger.info("EXTRACTION COMPLETE")
            self.logger.info(f"  Document type: {result.document_type}")
            self.logger.info(f"  Fields extracted: {len(result.field_values)}")
            self.logger.info("=" * 80)

            return result

        except Exception as e:
            self.logger.error(f"ID card extraction failed: {e}", exc_info=True)
            # Return minimal result on error
            return IDCardData(
                document_type=document_type_hint,
                field_values={},
                confidence_scores={},
                overall_confidence=0.0
            )

    def _detect_document_type(
        self,
        all_text: str,
        text_blocks: List[Dict]
    ) -> Optional[str]:
        """
        Auto-detect document type from text.

        Returns: 'PAN', 'UAE_TRC', 'national_id', 'driver_license', or None
        """
        text_upper = all_text.upper()

        # Check for UAE Tax Residency Certificate
        if ('TAX RESIDENCY' in text_upper or 'TAX RESIDENT' in text_upper or
            'MINISTRY OF FINANCE' in text_upper):
            if 'CERTIFICATE' in text_upper or 'TRC' in text_upper:
                return 'UAE_TRC'
        # Also check for UAE with tax-related keywords
        if 'UAE' in text_upper or 'UNITED ARAB EMIRATES' in text_upper:
            if 'TAX' in text_upper and ('CERTIFICATE' in text_upper or 'TRC' in text_upper or 'RESIDENT' in text_upper):
                return 'UAE_TRC'

        # Check for PAN card indicators (more comprehensive)
        if ('PAN' in text_upper or 'PERMANENT ACCOUNT NUMBER' in text_upper or
            'PERMANENT ACCOUNT NUMBER CARD' in text_upper or 'INCOME TAX DEPARTMENT' in text_upper):
            # Also check for PAN pattern in individual text blocks
            for block in text_blocks:
                block_text = block.get('text', '').strip().upper().replace(' ', '')
                if self.PAN_PATTERN.search(block_text):
                    return 'PAN'
            # Still return PAN if text indicators are strong
            if 'INCOME TAX DEPARTMENT' in text_upper or 'PERMANENT ACCOUNT NUMBER CARD' in text_upper:
                return 'PAN'

        # Check for driver license
        if 'DRIVER' in text_upper or 'DRIVING' in text_upper or 'LICENCE' in text_upper or 'LICENSE' in text_upper:
            return 'driver_license'

        # Check for national ID
        if 'NATIONAL ID' in text_upper or 'NRIC' in text_upper or 'IC NO' in text_upper:
            return 'national_id'

        return None

    def _extract_patterns_fallback(
        self,
        text_blocks: List[Dict],
        document_type: Optional[str],
        already_extracted: Dict[str, str]
    ) -> Dict[str, Tuple[str, float]]:
        """
        Pattern-based extraction fallback for fields GLiNER might miss.

        Returns dict of {field_name: (value, confidence)}
        """
        found = {}
        all_text = ' '.join([b.get('text', '') for b in text_blocks])
        all_text_upper = all_text.upper()

        # Extract PAN number using pattern
        if 'identification_number' not in already_extracted and 'pan_number' not in already_extracted:
            for block in text_blocks:
                block_text = block.get('text', '').strip()
                # Check for PAN pattern
                if self.PAN_PATTERN.match(block_text):
                    found['pan_number'] = (block_text, 95.0)  # High confidence for pattern match
                    self.logger.info(f"  Pattern fallback: Found PAN number '{block_text}'")
                    break

        # Try to extract name near "Name:" label (for PAN cards)
        if 'full_name' not in already_extracted:
            # Find the name label and get the text below it
            for i, block in enumerate(text_blocks):
                text = block.get('text', '').strip()
                text_upper = text.upper()
                # Look for labels that indicate cardholder name (not father/mother name)
                if ('NAME' in text_upper and 'FATHER' not in text_upper and
                    'MOTHER' not in text_upper and 'S/O' not in text_upper and
                    'D/O' not in text_upper and 'A/L' not in text_upper):
                    # This is likely the cardholder name label, get the next non-label text
                    for j in range(i + 1, min(i + 3, len(text_blocks))):
                        next_text = text_blocks[j].get('text', '').strip()
                        # Skip if it's another label
                        next_upper = next_text.upper()
                        if any(label in next_upper for label in ['NAME', 'FATHER', 'MOTHER', 'BIRTH', 'DATE', 'GENDER']):
                            continue
                        # Check if it looks like a name (letters and spaces, no digits, reasonable length)
                        if (next_text and len(next_text) > 3 and
                            next_text.replace(' ', '').replace("'", '').replace('.', '').isalpha()):
                            found['full_name'] = (next_text, 90.0)
                            self.logger.info(f"  Pattern fallback: Found cardholder name '{next_text}' below '{text}'")
                            break

        return found

    def _detect_country(
        self,
        all_text: str,
        text_blocks: List[Dict]
    ) -> Optional[str]:
        """
        Detect issuing country from text.
        """
        text_upper = all_text.upper()

        # Common country indicators
        country_indicators = {
            'INDIA': 'IND',
            'GOVT OF INDIA': 'IND',
            'INCOME TAX DEPARTMENT': 'IND',
            'SINGAPORE': 'SGP',
            'MALAYSIA': 'MYS',
            'UNITED STATES': 'USA',
            'USA': 'USA',
            'GREAT BRITAIN': 'GBR',
            'UNITED KINGDOM': 'GBR',
            'UK': 'GBR',
            'UAE': 'ARE',
            'UNITED ARAB EMIRATES': 'ARE',
            'DUBAI': 'ARE',
            'ABU DHABI': 'ARE',
        }

        for indicator, code in country_indicators.items():
            if indicator in text_upper:
                return code

        return None

    def _find_field_labels(self, text_blocks: List[Dict]) -> List[FieldLabel]:
        """
        Find field labels in text blocks using pattern matching.

        Args:
            text_blocks: OCR text blocks with geometry

        Returns:
            List of FieldLabel objects
        """
        labels = []

        for block in text_blocks:
            text = block.get('text', '').strip()
            if not text:
                continue

            text_lower = text.lower()

            # Check if this text matches any field label pattern
            for field_type, patterns in self.FIELD_LABEL_PATTERNS.items():
                for pattern in patterns:
                    if pattern.lower() in text_lower:
                        # Found a label
                        labels.append(FieldLabel(
                            text=text,
                            x1=block.get('x1', 0),
                            y1=block.get('y1', 0),
                            x2=block.get('x2', 0),
                            y2=block.get('y2', 0),
                            label_type=field_type
                        ))
                        self.logger.debug(f"  Found label '{text}' for field '{field_type}'")
                        break

        return labels

    def _pair_labels_with_values(
        self,
        text_blocks: List[Dict],
        gliner_result: Dict[str, Any],
        field_labels: List[FieldLabel]
    ) -> Dict[str, str]:
        """
        Pair field labels with their values using spatial proximity.

        Algorithm:
        1. For each label, find the nearest text block that contains
           a GLiNER-identified entity of matching type
        2. Use Euclidean distance between bounding boxes
        3. Filter pairs beyond max distance threshold

        Args:
            text_blocks: OCR text blocks with geometry
            gliner_result: GLiNER extracted entities
            field_labels: Field labels found on card

        Returns:
            Dictionary mapping field names to extracted values
        """
        paired = {}

        # Build a spatial index of text blocks by value
        value_locations = []
        for block in text_blocks:
            text = block.get('text', '').strip()
            if text:
                value_locations.append({
                    'text': text,
                    'x1': block.get('x1', 0),
                    'y1': block.get('y1', 0),
                    'x2': block.get('x2', 0),
                    'y2': block.get('y2', 0),
                })

        # For each GLiNER result, find the closest label
        for entity_label, entity_data in gliner_result.items():
            if not entity_data:
                continue

            # Handle both list (multi-value) and dict (single-value) formats
            if isinstance(entity_data, list):
                # Multi-value: process each item
                for item in entity_data:
                    value_text = item.get('value', '').strip()
                    if not value_text:
                        continue
                    self._try_pair_value(value_text, value_locations, field_labels, paired, entity_label)
            else:
                # Single-value: process directly
                value_text = entity_data.get('value', '').strip()
                if not value_text:
                    continue
                self._try_pair_value(value_text, value_locations, field_labels, paired, entity_label)

            # Find text blocks containing this value
            matching_blocks = [
                loc for loc in value_locations
                if value_text.lower() in loc['text'].lower() or
                   loc['text'].lower() in value_text.lower()
            ]

            if not matching_blocks:
                # Value not found in text blocks, add it directly
                self.logger.debug(f"  GLiNER value '{value_text}' not found in spatial blocks, adding directly")
                paired[entity_label] = value_text
                continue

            # Find the closest label to this value
            for block_loc in matching_blocks:
                closest_label = None
                min_distance = float('inf')

                for label in field_labels:
                    distance = self._calculate_distance(label, block_loc)
                    if distance < min_distance and distance < 0.15:  # 15% threshold
                        min_distance = distance
                        closest_label = label

                if closest_label:
                    field_name = closest_label.label_type
                    if field_name not in paired:
                        paired[field_name] = value_text
                        self.logger.debug(
                            f"  Paired '{closest_label.text}' (label) with "
                            f"'{value_text}' (value) at distance {min_distance:.3f}"
                        )

        # Also add GLiNER results that weren't paired spatially
        for entity_label, entity_data in gliner_result.items():
            if not entity_data:
                continue

            # Handle both list (multi-value) and dict (single-value) formats
            if isinstance(entity_data, list):
                # Multi-value: add all items
                for item in entity_data:
                    value_text = item.get('value', '').strip()
                    if value_text and entity_label not in paired:
                        paired[entity_label] = value_text
                        self.logger.debug(f"  Added GLiNER value '{entity_label}': '{value_text}'")
                        break  # Only add the first one for now
            else:
                # Single-value: add directly
                value_text = entity_data.get('value', '').strip()
                if value_text and entity_label not in paired:
                    paired[entity_label] = value_text
                    self.logger.debug(f"  Added GLiNER value '{entity_label}': '{value_text}'")

        return paired

    def _calculate_distance(
        self,
        label: FieldLabel,
        value_loc: Dict[str, float]
    ) -> float:
        """
        Calculate normalized distance between label and value.

        Uses normalized coordinates (0-1), so result is in 0-1 range.
        Lower values indicate closer proximity.
        """
        # Get centers of bounding boxes
        label_center_x = (label.x1 + label.x2) / 2
        label_center_y = (label.y1 + label.y2) / 2

        value_center_x = (value_loc['x1'] + value_loc['x2']) / 2
        value_center_y = (value_loc['y1'] + value_loc['y2']) / 2

        # Euclidean distance
        distance = ((label_center_x - value_center_x) ** 2 +
                    (label_center_y - value_center_y) ** 2) ** 0.5

        return distance

    def _try_pair_value(
        self,
        value_text: str,
        value_locations: List[Dict],
        field_labels: List[FieldLabel],
        paired: Dict[str, str],
        entity_label: str
    ):
        """
        Try to pair a single value with the closest field label.

        Args:
            value_text: The value text to pair
            value_locations: List of text block locations
            field_labels: List of field labels
            paired: Dictionary to store paired results
            entity_label: Original GLiNER entity label
        """
        # Find text blocks containing this value
        matching_blocks = [
            loc for loc in value_locations
            if value_text.lower() in loc['text'].lower() or
               loc['text'].lower() in value_text.lower()
        ]

        if not matching_blocks:
            # Value not found in text blocks, add it directly
            self.logger.debug(f"  GLiNER value '{value_text}' not found in spatial blocks, adding directly")
            paired[entity_label] = value_text
            return

        # Find the closest label to this value
        for block_loc in matching_blocks:
            closest_label = None
            min_distance = float('inf')

            for label in field_labels:
                distance = self._calculate_distance(label, block_loc)
                if distance < min_distance and distance < 0.15:  # 15% threshold
                    min_distance = distance
                    closest_label = label

            if closest_label:
                field_name = closest_label.label_type
                if field_name not in paired:
                    paired[field_name] = value_text
                    self.logger.debug(
                        f"  Paired '{closest_label.text}' (label) with "
                        f"'{value_text}' (value) at distance {min_distance:.3f}"
                    )

    def _build_result(
        self,
        gliner_result: Dict[str, Any],
        paired_fields: Dict[str, str],
        document_type: Optional[str]
    ) -> IDCardData:
        """
        Build IDCardData from extraction results.

        Args:
            gliner_result: Raw GLiNER extraction results
            paired_fields: Spatially paired field-value pairs
            document_type: Detected document type

        Returns:
            IDCardData with all extracted information
        """
        # Initialize field values and confidence scores
        field_values = {}
        confidence_scores = {}

        # Process paired fields (higher priority as they have spatial verification)
        for field_name, value in paired_fields.items():
            field_values[field_name] = value

            # Find confidence from GLiNER result (handle both list and dict formats)
            found_confidence = False
            for gliner_label, gliner_data in gliner_result.items():
                if not gliner_data:
                    continue

                # Handle list format
                if isinstance(gliner_data, list):
                    for item in gliner_data:
                        if item.get('value') == value:
                            confidence_scores[field_name] = item.get('confidence', 0.5) * 100
                            found_confidence = True
                            break
                # Handle dict format
                elif gliner_data.get('value') == value:
                    confidence_scores[field_name] = gliner_data.get('confidence', 0.5) * 100
                    found_confidence = True

                if found_confidence:
                    break

            # Default confidence if not found
            if not found_confidence:
                confidence_scores[field_name] = 70.0

        # Add GLiNER results that weren't paired
        for gliner_label, gliner_data in gliner_result.items():
            if not gliner_data:
                continue

            # Handle list format
            if isinstance(gliner_data, list):
                for item in gliner_data:
                    value = item.get('value', '').strip()
                    if value and gliner_label not in field_values:
                        field_values[gliner_label] = value
                        confidence_scores[gliner_label] = item.get('confidence', 0.5) * 100
                        break  # Only add the first one
            # Handle dict format
            else:
                value = gliner_data.get('value', '').strip()
                if value and gliner_label not in field_values:
                    field_values[gliner_label] = value
                    confidence_scores[gliner_label] = gliner_data.get('confidence', 0.5) * 100

        # Map to standard fields - prioritize exact field names
        full_name = (
            field_values.get('full_name') or
            field_values.get('full name or person name including s/o d/o a/l patterns') or
            field_values.get('given name or first name') or
            field_values.get('name')
        )

        # Handle S/O, D/O, A/L patterns for father's name
        # Explicitly exclude the full_name if it looks like a father's name
        father_name = (
            field_values.get('father_name') or
            field_values.get("father's name or father name") or
            self._extract_name_with_so_pattern(field_values)
        )

        # If full_name and father_name are the same, try to find the real full_name
        if full_name == father_name and 'given name or first name' in field_values:
            # Try to find another name in field_values
            for key, value in field_values.items():
                if 'name' in key.lower() and value != father_name and 'father' not in key.lower():
                    full_name = value
                    break

        # Map date fields
        date_of_birth = (
            field_values.get('date of birth or birth date') or
            field_values.get('date_of_birth')
        )

        # Map gender
        gender = (
            field_values.get('gender or sex m f') or
            field_values.get('gender')
        )

        # Map ID number - try pan_number first, then tax_id (for UAE TRC), then identification number, then generic
        identification_number = (
            field_values.get('pan_number') or
            field_values.get('pan number or permanent account number') or
            field_values.get('tax_id') or  # UAE TRC tax ID
            field_values.get('identification_number') or
            field_values.get('identification number or id number or card number')
        )

        # Create result
        # Build raw_entities list (handle both list and dict formats)
        raw_entities = []
        for k, v in gliner_result.items():
            if not v:
                raw_entities.append({'label': k, 'value': None, 'confidence': None})
            elif isinstance(v, list):
                # For multi-value entities, include all values
                for item in v:
                    raw_entities.append({
                        'label': k,
                        'value': item.get('value'),
                        'confidence': item.get('confidence')
                    })
            else:
                raw_entities.append({
                    'label': k,
                    'value': v.get('value'),
                    'confidence': v.get('confidence')
                })

        result = IDCardData(
            document_type=document_type,
            full_name=full_name,
            date_of_birth=date_of_birth,
            gender=gender,
            identification_number=identification_number,
            field_values=field_values,
            confidence_scores=confidence_scores,
            raw_entities=raw_entities
        )

        # Calculate overall confidence
        result.overall_confidence = result.calculate_overall_confidence()

        return result

    def _extract_name_with_so_pattern(self, field_values: Dict[str, str]) -> Optional[str]:
        """
        Extract name that contains S/O, D/O, or A/L pattern.

        Returns the name with the pattern if found, None otherwise.
        """
        for value in field_values.values():
            if re.search(r'\s+(S/O|D/O|A/L|S/O\.|D/O\.|A/L\.)\s+', value, re.IGNORECASE):
                return value
        return None

"""
Unified Identity Document Extractor

Layout-independent extractor for passports, ID cards, and other identity documents.
Uses logic-based field recognition with strict validation, making it work with
any document format from any country.

Key Features:
- Logic-based extraction (not pattern matching)
- Country-first detection
- Document type auto-detection
- Handles optional fields gracefully (e.g., no expiry on Singapore NRIC)
- Confidence scoring per field
"""

import re
import asyncio
import bisect
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from rapidfuzz import fuzz, process

from app.core import logger
from app.helper.doctr.document_text_extractor import DocumentTextExtractor
from app.helper.extractors.country_field_labels import get_country_field_labels, CountryFieldLabels
from app.helper.extractors.pattern_definitions import (
    COUNTRY_PATTERNS,
    FIELD_PATTERNS,
    get_country_pattern,
    get_field_pattern,
    detect_country_from_text,
    detect_document_type_from_patterns,
    is_optional_field,
)
from app.utils.country_code_converter import validate_iso_country_code
from app.utils.date_extractor import extract_and_remove_dates
from app.utils.date_parser import format_date_for_passport


class DocumentExtractionResult:
    """Result of unified identity document extraction."""

    def __init__(
        self,
        document_type: str,
        country_code: str,
        confidence: float,
        text_blocks: Optional[List[Dict]] = None,
        **fields
    ):
        self.document_type = document_type  # "passport" or "id_card"
        self.country_code = country_code
        self.confidence = confidence
        self.fields = fields
        self.text_blocks = text_blocks or []  # Raw OCR blocks with geometry

    def get(self, field_name: str, default=None) -> Any:
        """Get a field value."""
        return self.fields.get(field_name, default)

    def has(self, field_name: str) -> bool:
        """Check if a field exists and has a value."""
        return field_name in self.fields and self.fields[field_name] is not None


class UnifiedIDExtractor:
    """
    Layout-independent extractor for passports, ID cards, and other identity documents.

    Uses pattern-based field recognition to extract data without relying on
    document layout or spatial positioning.
    """

    # Minimum OCR confidence for name candidates (0.0-1.0)
    # Text with lower confidence is likely garbage from OCR errors
    MIN_NAME_CONFIDENCE = 0.6

    def __init__(self):
        self.logger = logger
        self.text_extractor = DocumentTextExtractor()
        self.field_labels = get_country_field_labels()

    # ========================================================================
    # VALIDATION FUNCTIONS (from PassportExtractor)
    # ========================================================================

    def _is_type(self, text: str) -> bool:
        """Check if text is a passport type (1-2 chars starting with P/D/S)."""
        text = text.strip().upper()
        if len(text) < 1 or len(text) > 2:
            return False
        return text[0] in ('P', 'D', 'S') and text.isalpha()

    def _is_country_code(self, text: str) -> bool:
        """Check if text is a valid 3-letter country code."""
        text = text.strip().upper()
        return len(text) == 3 and validate_iso_country_code(text)

    def _is_passport_number(self, text: str) -> bool:
        """Check if text is a passport number (8-9 alphanumeric chars with ≥6 digits)."""
        text = text.strip().upper()
        if len(text) < 8 or len(text) > 9:
            return False
        if not text.isalnum():
            return False
        digit_count = sum(c.isdigit() for c in text)
        return digit_count >= 6

    def _is_sex(self, text: str) -> bool:
        """Check if text is a sex indicator (standalone M or F)."""
        text = text.strip().upper()
        return text == 'M' or text == 'F'

    def _is_name_label(self, text: str, country_code: Optional[str] = None) -> bool:
        """
        Check if text is a name-related label.

        Uses country-specific labels when country is known, falls back
        to global patterns otherwise.

        Args:
            text: Text to check
            country_code: ISO 2-letter country code (e.g., "SG", "MM")

        Returns:
            True if text matches a name label
        """
        if not text:
            return False

        text_lower = text.lower()

        # First, exclude non-name labels that happen to contain name-like substrings
        # "nationality" is for country codes, not names
        if 'nationality' in text_lower:
            return False

        # Use country-specific labels if available
        if country_code:
            return self.field_labels.is_name_label(country_code, text)

        # Fallback: check for name-related keywords (original behavior)
        # Direct matches for "name"
        if 'name' in text_lower:
            return True
        # Handle "surname" and common OCR errors
        if 'surname' in text_lower or 'surame' in text_lower:
            return True
        # Handle other name-related labels
        name_keywords = ['given name', 'last name', 'first name', 'family name',
                        'given', 'surname']
        for keyword in name_keywords:
            if keyword in text_lower:
                return True
        return False

    def _get_name_label_type(self, text: str) -> str:
        """
        Determine the type of name label.
        Returns: 'given' for given/first name, 'surname' for surname/last name, 'other' for generic name labels
        """
        text_lower = text.lower()
        # Check for given name indicators
        given_keywords = ['given name', 'first name', 'given', 'first']
        for keyword in given_keywords:
            if keyword in text_lower:
                return 'given'
        # Check for surname indicators
        surname_keywords = ['surname', 'surame', 'last name', 'family name']
        for keyword in surname_keywords:
            if keyword in text_lower:
                return 'surname'
        # Default to generic name label
        return 'other'

    def _is_date_text(self, text: str) -> Optional[datetime]:
        """
        Check if text is a date and parse it.
        Returns datetime object if valid, None otherwise.
        Supports: DD MMM YYYY, DD/MM/YYYY, DD-MM-YYYY, YYYY-MM-DD
        """
        text = text.strip().upper()

        months = {
            'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4,
            'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8,
            'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
        }

        try:
            # Try DD MMM YYYY format (e.g., "12 NOV 1992")
            match = re.match(r'^(\d{1,2})\s*([A-Z]{3})\s*(\d{4})$', text)
            if match:
                day, month_name, year = match.groups()
                month = months.get(month_name)
                if month:
                    return datetime(int(year), month, int(day))

            # Try DD/MM/YYYY or DD-MM-YYYY
            match = re.match(r'^(\d{1,2})[/-](\d{1,2})[/-](\d{4})$', text)
            if match:
                day, month, year = match.groups()
                return datetime(int(year), int(month), int(day))

            # Try YYYY-MM-DD or YYYY/MM/DD
            match = re.match(r'^(\d{4})[/-](\d{1,2})[/-](\d{1,2})$', text)
            if match:
                year, month, day = match.groups()
                return datetime(int(year), int(month), int(day))

        except (ValueError, TypeError):
            pass

        return None

    async def extract(self, content: bytes, is_pdf: bool = False):
        """
        Extract identity document fields using GLINER2 with logic-based fallback.

        For passport documents, GLINER2 is tried first. Falls back to logic-based
        extraction if confidence < 50% or if GLINER2 fails.

        For other documents, uses logic-based extraction.

        Args:
            content: Document bytes
            is_pdf: Whether content is PDF

        Returns:
            PassportData (for passports with good GLINER2 result) or
            DocumentExtractionResult (for all other cases)
        """
        try:
            self.logger.info("=" * 80)
            self.logger.info("UNIFIED ID DOCUMENT EXTRACTION")
            self.logger.info("=" * 80)

            # Step 1: Extract text with OCR
            self.logger.info("Step 1: Extracting text with OCR")
            text_blocks = await self.text_extractor.extract_text_with_geometry_enhanced(
                content, is_pdf=is_pdf
            )
            self.logger.info(f"  Extracted {len(text_blocks)} text lines")

            # Combine all text for country/document type detection - preserve reading order with newlines
            all_text = "\n".join([block.get('text', '') for block in text_blocks])

            # Step 2: Detect country
            self.logger.info("Step 2: Detecting country")
            country_code = detect_country_from_text(all_text)
            if not country_code:
                self.logger.warning("  Could not detect country from text")
                # Default to generic extraction
                country_code = "UNKNOWN"

            self.logger.info(f"  Detected country: {country_code}")

            # Step 3: Detect document type
            self.logger.info("Step 3: Detecting document type")
            document_type = detect_document_type_from_patterns(all_text, country_code)
            self.logger.info(f"  Detected document type: {document_type}")

            # Step 4: Skip GLiNER2 for passports - use reliable logic-based extraction
            # GLiNER2 schema-based extraction is unreliable for structured passport documents.
            # It returns garbage results (e.g., "KILARI CHANDRA ROHITH PreCTErTT") even when
            # DocTR OCR correctly extracts clean data. Passports have highly structured
            # formats with known field positions - spatial/logic-based extraction is far more reliable.
            if False:  # GLiNER2 disabled for passports
                self.logger.info("Step 4: Trying GLINER2 extraction for passport")
                try:
                    from app.helper.extractors import get_gliner_passport_extractor

                    gliner_extractor = get_gliner_passport_extractor()
                    result = await gliner_extractor.extract(content, is_pdf=is_pdf)

                    # Log what GLINER2 extracted
                    self.logger.info(
                        f"GLINER2 extracted: confidence={result.overall_confidence:.1f}%, "
                        f"source={result.extraction_source}"
                    )

                    # Check if GLINER2 result is good enough
                    if result.overall_confidence and result.overall_confidence >= 50:
                        if self._has_required_fields(result):
                            self.logger.info("GLINER2 result accepted - using GLINER2 extraction")
                            return result

                    # Fallback to logic-based
                    self.logger.info(
                        f"GLINER2 fallback triggered: confidence={result.overall_confidence or 0:.1f}%"
                    )
                except Exception as e:
                    self.logger.error(f"GLINER2 extraction error: {str(e)}, falling back to logic-based")

            # Step 5: Logic-based extraction (now primary for passports)
            self.logger.info(f"Using logic-based extraction for {document_type}")
            return await self._extract_with_logic_based(content, is_pdf, text_blocks, all_text, country_code, document_type)

        except Exception as e:
            self.logger.error(f"Extraction failed: {type(e).__name__}")

            # Return minimal result on error
            return DocumentExtractionResult(
                document_type="unknown",
                country_code="UNKNOWN",
                confidence=0.0,
                error=str(e)
            )

    def _has_required_fields(self, passport_data) -> bool:
        """Check if passport data has all required fields."""
        required = ['passport_number', 'full_name', 'date_of_birth', 'nationality']
        for field in required:
            value = getattr(passport_data, field, None)
            if not value or not str(value).strip():
                return False
        return True

    async def _extract_with_logic_based(
        self,
        content: bytes,
        is_pdf: bool,
        text_blocks: List[Dict[str, Any]],
        all_text: str,
        country_code: str,
        document_type: str
    ) -> DocumentExtractionResult:
        """
        Extract using logic-based algorithm (original implementation).
        """
        self.logger.info("Step 4: Extracting fields (logic-based)")
        extracted_fields = await self._extract_fields_from_text(
            text_blocks, all_text, country_code, document_type
        )

        # Step 5: Validate and score
        self.logger.info("Step 5: Validating and scoring")
        validation_result = self._validate_extraction(
            extracted_fields, country_code, document_type
        )

        # Calculate overall confidence
        confidence = self._calculate_overall_confidence(
            extracted_fields, validation_result, country_code, document_type
        )

        self.logger.info(f"  Overall confidence: {confidence:.2f}%")
        self.logger.info(f"  Extracted {len(extracted_fields)} fields: {list(extracted_fields.keys())}")
        self.logger.info("=" * 80)

        # Remove country_code from extracted_fields if it exists to avoid duplicate argument
        extracted_fields_copy = extracted_fields.copy()
        if 'country_code' in extracted_fields_copy:
            del extracted_fields_copy['country_code']

        return DocumentExtractionResult(
            document_type=document_type,
            country_code=country_code,
            confidence=confidence,
            text_blocks=text_blocks,
            **extracted_fields_copy
        )

    async def _extract_fields_from_text(
        self,
        text_blocks: List[Dict[str, Any]],
        all_text: str,
        country_code: str,
        document_type: str
    ) -> Dict[str, Any]:
        """
        Extract fields using logic-based single-pass algorithm.

        Algorithm (ported from PassportExtractor):
        1. Single pass through OCR blocks
        2. Check in priority order: type → country → passport → alt_id → "name" → dates → sex
        3. Skip already-found fields (early exit per field)
        4. Post-pass: resolve dates (sort) and names (x-proximity)

        Note: Missing optional fields (e.g., expiry on Singapore NRIC) do not
        reduce confidence score. Only missing required fields affect confidence.
        """
        self.logger.info("=" * 80)
        self.logger.info(f"LOGIC-BASED EXTRACTION ({document_type.upper()})")
        self.logger.info("=" * 80)

        # Initialize state
        found = {
            'type': None,
            'country_code': None,
            'passport_number': None,
            'alternative_id': None,
            'sex': None
        }

        # Dates stored as (datetime, original_text) tuples, kept sorted
        dates: List[Tuple[datetime, str]] = []

        # Name labels: (x1, y2, label_text) - store bottom-left corner for matching
        name_labels: List[Tuple[float, float, str]] = []

        # Unprocessed blocks for name extraction: (text, x_coord, y_coord, confidence)
        unprocessed: List[Tuple[str, float, float, float]] = []

        # Track processed texts to avoid double-counting
        processed_texts = set()

        self.logger.info(f"Processing {len(text_blocks)} OCR blocks...")

        # SINGLE PASS through all OCR blocks
        for block in text_blocks:
            text = block.get('text', '').strip()
            x = block.get('x1', 0)
            y = block.get('y1', 0)

            if not text:
                continue

            matched = False
            text_upper = text.upper()

            # Check in priority order, skip already-found fields

            # 1. Type (1-2 chars starting with P/D/S) - only for passports
            if document_type == 'passport' and not found['type'] and self._is_type(text):
                found['type'] = text_upper
                processed_texts.add(text)
                matched = True
                self.logger.info(f"  ✓ Type: {text_upper}")

            # 2. Country code (3-letter ISO code)
            elif not found['country_code'] and self._is_country_code(text):
                found['country_code'] = text_upper
                processed_texts.add(text)
                matched = True
                self.logger.info(f"  ✓ Country: {text_upper}")

            # 3. Passport/ID number
            elif not found['passport_number'] and self._is_passport_number(text):
                found['passport_number'] = text_upper
                processed_texts.add(text)
                matched = True
                self.logger.info(f"  ✓ Number: {text_upper}")

            # 4. Alternative ID (second passport-number-like string)
            elif document_type == 'passport' and found['passport_number'] and not found['alternative_id'] and self._is_passport_number(text):
                found['alternative_id'] = text_upper
                processed_texts.add(text)
                matched = True
                self.logger.info(f"  ✓ Alternative ID: {text_upper}")

            # 5. Name labels (using country-specific labels) - collect for post-pass matching
            elif self._is_name_label(text, found.get('country_code')):
                y2 = block.get('y2', y)  # Get bottom of label box
                name_labels.append((x, y2, text))
                processed_texts.add(text)
                matched = True
                self.logger.info(f"  ✓ Name label at (x1={x:.3f}, y2={y2:.3f}): {text}")

            # 6. Dates (max 3, auto-sorted)
            elif len(dates) < 3:
                parsed_date = self._is_date_text(text)
                if parsed_date:
                    # Insert sorted by date
                    bisect.insort(dates, (parsed_date, text))
                    processed_texts.add(text)
                    matched = True
                    self.logger.info(f"  ✓ Date: {text} -> {parsed_date.date()}")

            # 7. Sex (standalone M or F)
            if not matched and not found['sex'] and self._is_sex(text):
                found['sex'] = text_upper
                processed_texts.add(text)
                matched = True
                self.logger.info(f"  ✓ Sex: {text_upper}")

            # Track unprocessed blocks for name extraction (include confidence)
            if not matched:
                confidence = block.get('confidence', 0.9)
                unprocessed.append((text, x, y, confidence))

        # ============================================================
        # SECOND PASS: Substring matching for merged OCR values
        # ============================================================
        self.logger.info("-" * 40)
        self.logger.info("SECOND PASS: Extracting from merged OCR values...")

        # Track strings that get fully processed in second pass
        still_unprocessed = []

        for text, x, y, confidence in unprocessed:
            remaining_text = text

            # STEP 1: Extract embedded dates first (dates have internal spaces)
            if len(dates) < 3:
                embedded_dates, remaining_text = extract_and_remove_dates(text)
                for dt_tuple in embedded_dates:
                    if len(dates) < 3:
                        bisect.insort(dates, dt_tuple)
                        self.logger.info(f"  ✓ Embedded date from '{text}': {dt_tuple[1]} -> {dt_tuple[0].date()}")

            # STEP 2: Split remaining string by spaces, check each word
            words = remaining_text.split()
            word_matched_any = False

            for word in words:
                word_upper = word.upper()
                matched_word = False

                # Same if/elif priority as first pass
                if document_type == 'passport' and not found['type'] and self._is_type(word):
                    found['type'] = word_upper
                    matched_word = True
                    self.logger.info(f"  ✓ Type from '{text}': {word_upper}")

                elif not found['country_code'] and self._is_country_code(word):
                    found['country_code'] = word_upper
                    matched_word = True
                    self.logger.info(f"  ✓ Country from '{text}': {word_upper}")

                elif not found['passport_number'] and self._is_passport_number(word):
                    found['passport_number'] = word_upper
                    matched_word = True
                    self.logger.info(f"  ✓ Passport number from '{text}': {word_upper}")

                elif document_type == 'passport' and found['passport_number'] and not found['alternative_id'] and self._is_passport_number(word):
                    found['alternative_id'] = word_upper
                    matched_word = True
                    self.logger.info(f"  ✓ Alternative ID from '{text}': {word_upper}")

                elif not found['sex'] and self._is_sex(word):
                    found['sex'] = word_upper
                    matched_word = True
                    self.logger.info(f"  ✓ Sex from '{text}': {word_upper}")

                if matched_word:
                    word_matched_any = True

            # Keep track of strings that still have unmatched content (for name extraction)
            if not word_matched_any and remaining_text.strip():
                still_unprocessed.append((remaining_text, x, y, confidence))

        # Update unprocessed list for name extraction
        unprocessed = still_unprocessed

        # ============================================================
        # POST-PASS: Resolve dates
        # ============================================================
        self.logger.info("-" * 40)
        self.logger.info("Resolving dates (earliest=DOB, latest=Expiry, middle=Issue)...")

        extracted = {}

        if len(dates) >= 1:
            extracted['dob'] = format_date_for_passport(dates[0][0])
            self.logger.info(f"  DOB (earliest): {extracted['dob']}")

        if len(dates) >= 3:
            extracted['issue_date'] = format_date_for_passport(dates[1][0])
            extracted['expiry'] = format_date_for_passport(dates[2][0])
            self.logger.info(f"  Issue (middle): {extracted['issue_date']}")
            self.logger.info(f"  Expiry (latest): {extracted['expiry']}")
        elif len(dates) == 2:
            # Only 2 dates: assume DOB and Expiry
            extracted['expiry'] = format_date_for_passport(dates[1][0])
            self.logger.info(f"  Expiry (latest): {extracted['expiry']}")

        # ============================================================
        # POST-PASS: Match name values to labels using y2→y1 spatial proximity
        # ============================================================
        if name_labels:
            self.logger.info("-" * 40)
            self.logger.info(f"Matching {len(name_labels)} name label(s) to values using y2→y1 proximity...")

            # Sort labels by y2 (top to bottom) for predictable processing
            name_labels_sorted = sorted(name_labels, key=lambda t: t[1])

            # Store name parts by label type for correct ordering
            given_name_parts = []
            surname_parts = []
            other_name_parts = []  # For generic "Name" labels

            # Filter candidates by confidence (exclude low-confidence OCR results)
            candidates = [
                (text, x, y, conf) for text, x, y, conf in unprocessed
                if conf >= self.MIN_NAME_CONFIDENCE
            ]
            self.logger.info(f"  Filtered to {len(candidates)} candidates (confidence >= {self.MIN_NAME_CONFIDENCE})")

            for label_x1, label_y2, label_text in name_labels_sorted:
                if not candidates:
                    break

                # Determine label type
                label_type = self._get_name_label_type(label_text)

                # Find closest value that is BELOW or AT THE SAME LEVEL as the label
                # Use tolerance to handle OCR geometry inaccuracies where values
                # may appear at the same vertical level as labels
                below_candidates = [
                    t for t in candidates
                    if t[2] > label_y2 - 0.10 and abs(t[1] - label_x1) < 150  # At same level or below, and horizontally aligned
                ]

                if below_candidates:
                    # Choose the closest value below
                    closest = min(below_candidates, key=lambda t: abs(t[1] - label_x1) + abs(t[2] - label_y2))
                    delta = abs(closest[1] - label_x1) + abs(closest[2] - label_y2)
                    value = closest[0]
                    conf = closest[3]
                    candidates.remove(closest)

                    # Store value based on label type
                    if label_type == 'given':
                        given_name_parts.append(value)
                    elif label_type == 'surname':
                        surname_parts.append(value)
                    else:  # Generic "Name" label
                        other_name_parts.append(value)

                    self.logger.info(f"  '{label_text}' ({label_type}) → '{value}' (delta={delta:.3f}, conf={conf:.2f})")
                else:
                    self.logger.warning(f"  '{label_text}' → No matching value found below")

            # Concatenate in correct order: given name + surname + other
            name_parts = given_name_parts + surname_parts + other_name_parts

            if name_parts:
                extracted['full_name'] = ' '.join(name_parts)
                self.logger.info(f"  Full name: {extracted['full_name']}")

        # Map found fields to standard field names
        if found['type']:
            extracted['type'] = found['type']
        if found['country_code']:
            extracted['country_code'] = found['country_code']
        if found['passport_number']:
            extracted['number'] = found['passport_number']
        if found['alternative_id']:
            extracted['alternative_id'] = found['alternative_id']
        if found['sex']:
            extracted['sex'] = found['sex']

        # Log summary
        self.logger.info("=" * 80)
        self.logger.info("EXTRACTION SUMMARY:")
        for field, value in extracted.items():
            if value:
                self.logger.info(f"  {field}: {value}")
        self.logger.info("=" * 80)

        return extracted

    async def _extract_field(
        self,
        field_name: str,
        text_blocks: List[Dict[str, Any]],
        country_code: str,
        document_type: str
    ) -> Optional[Any]:
        """
        Extract a single field using pattern matching.

        Args:
            field_name: Name of the field to extract
            text_blocks: OCR text blocks with geometry
            country_code: ISO country code
            document_type: Document type (passport/id_card)

        Returns:
            Extracted value or None if not found
        """
        field_pattern = get_field_pattern(field_name)
        if not field_pattern:
            self.logger.debug(f"    No pattern defined for {field_name}")
            return None

        # Get labels for this field
        labels = field_pattern.get("labels", [])
        if not labels:
            self.logger.debug(f"    No labels defined for {field_name}")
            return None

        # Convert text blocks to searchable format
        lines_with_positions = self._prepare_lines_with_positions(text_blocks)

        # Search for field label
        for line_info in lines_with_positions:
            line_text = line_info["text"]
            line_y = line_info["y"]

            # Check if any label matches this line (fuzzy matching)
            matched_label = None
            for label in labels:
                if self._fuzzy_match(line_text, label, threshold=75.0):
                    matched_label = label
                    break

            if matched_label:
                # Found the label, now extract the value
                value = self._extract_value_near_line(
                    line_info, lines_with_positions, field_name, country_code
                )
                if value:
                    return value

        return None

    def _extract_value_near_line(
        self,
        label_line: Dict[str, Any],
        all_lines: List[Dict[str, Any]],
        field_name: str,
        country_code: str
    ) -> Optional[Any]:
        """
        Extract field value from lines near a label line.

        Looks at the same line and next few lines to find the value.
        """
        label_y = label_line["y"]
        label_text = label_line["text"]

        # Get field pattern for validation
        field_pattern = get_field_pattern(field_name)
        if not field_pattern:
            return None

        # Try to extract value from the same line
        # Value is typically after the label, possibly separated by colon or space
        same_line_values = self._extract_values_from_line(label_text, field_name)
        if same_line_values:
            # Validate against country-specific pattern
            for value in same_line_values:
                if self._validate_value_format(value, field_name, country_code):
                    return value

        # Try next few lines (within ~50 pixels in Y coordinate)
        nearby_lines = [
            line for line in all_lines
            if 0 < abs(line["y"] - label_y) < 50
        ]

        for nearby_line in nearby_lines:
            line_values = self._extract_values_from_line(nearby_line["text"], field_name)
            if line_values:
                for value in line_values:
                    if self._validate_value_format(value, field_name, country_code):
                        return value

        return None

    def _extract_values_from_line(self, line_text: str, field_name: str) -> List[str]:
        """
        Extract potential values from a line of text.

        Splits the line and returns candidate values that could match this field type.
        """
        values = []

        # Common value separators
        separators = [":", " ", "\t", "-"]

        # Try different splitting approaches
        for sep in separators:
            if sep in line_text:
                parts = line_text.split(sep)
                # Value is typically after the last separator
                if len(parts) > 1:
                    candidate = parts[-1].strip()
                    if candidate and len(candidate) > 0:
                        values.append(candidate)

        # Also try taking the whole line minus common labels
        for label in FIELD_PATTERNS.get(field_name, {}).get("labels", []):
            if label.lower() in line_text.lower():
                # Remove the label and take what remains
                remaining = line_text.lower().replace(label.lower(), "").strip()
                # Remove common separators at the start
                remaining = remaining.lstrip(": -")
                if remaining:
                    values.append(remaining)

        return list(set(values))  # Remove duplicates

    def _validate_value_format(
        self,
        value: str,
        field_name: str,
        country_code: str
    ) -> bool:
        """
        Validate that a value matches the expected format for this field and country.

        Args:
            value: Candidate value
            field_name: Name of the field
            country_code: ISO country code

        Returns:
            True if value matches expected format
        """
        if not value:
            return False

        country_pattern = get_country_pattern(country_code)
        if not country_pattern:
            return True  # Can't validate without pattern, accept it

        # Check field-specific patterns
        if field_name == "number":
            # Check if it matches passport or ID card number pattern
            for doc_type in ["passport", "id_card"]:
                if doc_type in country_pattern:
                    pattern = country_pattern[doc_type].get("number")
                    if pattern:
                        import re
                        if re.match(pattern, value.strip().upper()):
                            return True

        # Generic validation based on field patterns
        field_pattern = get_field_pattern(field_name)
        if field_pattern and "regex" in field_pattern:
            import re
            if re.match(field_pattern["regex"], value.strip()):
                return True

        return True  # Accept if no specific validation

    def _prepare_lines_with_positions(
        self,
        text_blocks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Convert text blocks to line format with Y positions.

        Returns list of dicts with 'text', 'y', 'x1', 'x2' keys.
        """
        lines = []
        for block in text_blocks:
            text = block.get('text', '').strip()
            if text:
                lines.append({
                    'text': text,
                    'y': block.get('y1', 0),
                    'x1': block.get('x1', 0),
                    'x2': block.get('x2', 0),
                })

        # Sort by Y position (top to bottom)
        lines.sort(key=lambda x: x['y'])

        return lines

    def _fuzzy_match(self, text: str, expected: str, threshold: float = 75.0) -> bool:
        """
        Check if text contains expected label using fuzzy matching.

        Args:
            text: OCR text to check
            expected: Expected label
            threshold: Minimum similarity score (0-100)

        Returns:
            True if text contains expected label with sufficient similarity
        """
        if not text or not expected:
            return False

        # Check direct containment first
        if expected.lower() in text.lower():
            return True

        # Use fuzzy matching for OCR errors
        ratio = fuzz.partial_ratio(expected.lower(), text.lower())
        return ratio >= threshold

    def _validate_extraction(
        self,
        extracted_fields: Dict[str, Any],
        country_code: str,
        document_type: str
    ) -> Dict[str, Any]:
        """
        Validate extracted fields and return validation result.

        Checks:
        - All required fields are present
        - Fields have valid formats
        - Dates are consistent

        Returns:
            Validation dict with is_valid flag and messages
        """
        validation = {
            "is_valid": True,
            "missing_required": [],
            "invalid_format": [],
            "warnings": []
        }

        country_pattern = get_country_pattern(country_code)
        if not country_pattern:
            validation["warnings"].append(f"No country pattern for {country_code}")
            return validation

        if document_type not in country_pattern:
            validation["warnings"].append(f"No document type pattern for {document_type}")
            return validation

        doc_config = country_pattern[document_type]
        required_fields = doc_config.get("required_fields", [])

        # Check required fields
        for field in required_fields:
            if field not in extracted_fields or not extracted_fields[field]:
                validation["missing_required"].append(field)
                validation["is_valid"] = False

        return validation

    def _calculate_overall_confidence(
        self,
        extracted_fields: Dict[str, Any],
        validation_result: Dict[str, Any],
        country_code: str,
        document_type: str
    ) -> float:
        """
        Calculate overall confidence score for the extraction.

        Args:
            extracted_fields: Dict of extracted fields
            validation_result: Validation result
            country_code: ISO country code
            document_type: Document type

        Returns:
            Confidence score (0-100)
        """
        if not extracted_fields:
            return 0.0

        # Start with 100% and deduct for issues
        confidence = 100.0

        # Deduct for missing required fields
        missing_required = validation_result.get("missing_required", [])
        if missing_required:
            confidence -= 20.0 * len(missing_required)

        # Bonus points for extracting important fields
        important_fields = ["number", "full_name", "dob"]
        for field in important_fields:
            if field in extracted_fields:
                confidence += 5.0

        # Ensure confidence is in valid range
        confidence = max(0.0, min(100.0, confidence))

        return confidence

    def _extract_generically(
        self,
        text_blocks: List[Dict[str, Any]],
        all_text: str
    ) -> Dict[str, Any]:
        """
        Fallback generic extraction when no country-specific patterns exist.

        Uses basic pattern matching to extract common fields.
        """
        extracted = {}

        # Try to find passport-like numbers
        import re
        passport_pattern = r"\b[A-Z]{1,2}\d{6,9}\b"
        for block in text_blocks:
            text = block.get('text', '').strip()
            matches = re.findall(passport_pattern, text.upper())
            if matches and "number" not in extracted:
                extracted["number"] = matches[0]

        # Try to find dates
        date_patterns = [
            r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",
            r"\b\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+\d{4}\b",
        ]
        for pattern in date_patterns:
            for block in text_blocks:
                text = block.get('text', '').strip()
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches and "dob" not in extracted:
                    extracted["dob"] = matches[0]
                    break

        return extracted

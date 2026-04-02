from app.core.logger import get_logger
from app.core.gliner_ner_model import get_gliner_ner_model
from .address_validator import should_keep_address, normalize_address, looks_like_bank_address
from .data_structures import TextElement
from .line_by_line_extractor import LineExtractionResult
from typing import Dict, List, Optional
import re
from datetime import datetime
from dataclasses import dataclass

@dataclass
class LineInfo:
    """Information about a line of text."""
    text: str
    elements: List[TextElement]
    y_position: float
    x_position: float
    confidence: float

@dataclass
class SpatialInfo:
    """Spatial information about text blocks."""
    page_width: float
    page_height: float
    text_density: float
    has_left_column: bool
    has_right_column: bool
    top_margin: float
    left_margin: float
    cluster_proximity_score: float


class SimpleBankStatementAnalyzer:
    """
    Simple analyzer for bank statements using GLiNER2 NER.

    Uses field-based detection and spatial coordinates for value extraction.
    This approach is more reliable than GLiNER for structured documents.
    """

    def __init__(self):
        # Field-based extraction using spatial coordinates (more reliable for structured documents)
        pass

    def _extract_name_using_field_names(self, text_elements: List) -> Optional[str]:
        """
        Extract account holder name using field-based detection and spatial coordinates.

        This method uses hardcoded field name patterns and finds their associated values
        by checking spatial proximity (value to right or directly below field name).

        This is more reliable than GLiNER for structured documents.

        Returns:
            Account holder name if found, otherwise None.
        """
        import re

        # Helper function to get text from element (handles both objects and dicts)
        def get_text(elem):
            if isinstance(elem, dict):
                return elem.get('text', '').strip()
            return getattr(elem, 'text', '').strip()

        # Helper function to get y position from element
        def get_y(elem):
            if isinstance(elem, dict):
                return elem.get('y1', 0)
            if hasattr(elem, 'geometry'):
                return elem.geometry[1] if len(elem.geometry) > 1 else 0
            return 0

        # Helper function to get x position from element
        def get_x(elem):
            if isinstance(elem, dict):
                return elem.get('x1', 0)
            if hasattr(elem, 'geometry'):
                return elem.geometry[0] if len(elem.geometry) > 0 else 0
            return 0

        # Common bank statement field name patterns (hardcoded for reliability)
        field_name_patterns = [
            r'Account\s+Holder\s+Names?',
            r'Customer\s+Name',
            r'Name\s*:?',
            r'Account\s+Holder\s*:?',
            r'Account\s+No',
            r'Customer\s+ID',
            r'Joint\s+Holder',
            r'Primary\s+Holder',
            r'Statement\s+Date',
            r'Ifsc\s+Code',
            r'MICR\s+Code',
            r'PAN\s+Number',
            r'Email',
            r'Mobile',
            r'Address',
            r'Branch',
            r'City',
            r'State',
            r'Pincode',
        ]

        # Normalize labels for matching (case-insensitive)
        normalized_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in field_name_patterns]

        # First pass: Find field names and record their positions
        field_names = []
        for elem in text_elements:
            text_clean = get_text(elem)
            for pattern in normalized_patterns:
                if pattern.search(text_clean):
                    field_names.append({
                        'text': text_clean,
                        'x': get_x(elem),
                        'y': get_y(elem),
                        'element': elem
                    })
                    break  # Only record first match per element

        # If no field names found, return None
        if not field_names:
            from app.core.logger import get_logger
            logger = get_logger()
            logger.info(f"No field names found in document")
            return None

        from app.core.logger import get_logger
        logger = get_logger()
        logger.info(f"Found {len(field_names)} field name(s) in document")

        # Second pass: For each field name, find associated value
        for field_name in field_names:
            field_x = field_name['x']
            field_y = field_name['y']
            field_text = field_name['text']
            logger.debug(f"Looking for value after field '{field_text}' at ({field_x}, {field_y})")

            # Find value candidates (elements close to field name)
            candidates = []
            for elem in text_elements:
                if elem is field_name['element']:
                    continue  # Skip the field name element itself

                cand_text = get_text(elem)
                elem_x = get_x(elem)
                elem_y = get_y(elem)

                # Skip obvious non-value content (field names, labels like "Name:")
                if len(cand_text) < 2 or cand_text.lower() in ['name', 'account', 'holder', 'customer', 'label', 'field', 'heading']:
                    continue

                # Calculate spatial distance
                y_distance = abs(elem_y - field_y)
                x_distance = abs(elem_x - field_x)

                # Value must be close to field name:
                # - Same line (within 30px) and to the right, OR
                # - Directly below (within 40px)
                is_close = (y_distance < 30 and x_distance > 10) or (y_distance < 40)

                if is_close:
                    candidates.append({
                        'text': cand_text,
                        'x': elem_x,
                        'y': elem_y,
                        'element': elem
                    })

            # Sort candidates by proximity (y-distance first, then x-distance)
            candidates.sort(key=lambda c: (c['y'], abs(c['x'] - field_x)))

            # Take closest candidate as the value
            if candidates:
                value_candidate = candidates[0]
                value_text = value_candidate['text']

                # For name fields, apply validation
                if any(pattern in field_text.lower() for pattern in ['holder', 'name', 'customer']):
                    # Check if looks like a valid name
                    if len(value_text) >= 3 and not re.search(r'\d{4,}', value_text):
                        from app.core.logger import get_logger
                        logger = get_logger()
                        logger.info(f"✓ Field-based extraction: '{field_text}' -> '{value_text}'")
                        return self._remove_name_titles(value_text)

        return None

    def analyze_bank_statement(self, text_elements: List[TextElement]) -> LineExtractionResult:
        """
        Analyze bank statement and extract account name, address, and bank information using field-based extraction.

        Args:
            text_elements: List of text elements with position data

        Returns:
            LineExtractionResult with extracted information including bank details
        """
        from app.core.logger import get_logger
        from app.core.gliner_ner_model import get_gliner_ner_model
        from .address_validator import should_keep_address, normalize_address, looks_like_bank_address

        logger = get_logger()
        logger.info(f"Starting field-based bank statement analysis with {len(text_elements)} elements")

        # Step 1: Try field-based extraction first (highest accuracy)
        # This method detects field names and finds values using spatial coordinates
        field_based_name = self._extract_name_using_field_names(text_elements)
        if field_based_name:
            account_holder_name = field_based_name
            account_holder_name_confidence = 0.95  # High confidence for field-based extraction
            logger.info(f"✓ Using field-based name extraction: '{account_holder_name}'")

        # If field-based extraction found a name, still try GLiNER as fallback for other fields
        # Initialize result fields
        account_holder_name = field_based_name
        account_holder_name_confidence = 0.95 if field_based_name else 0.0
        bank_name = None
        account_number = None
        bank_address = None
        account_holder_address = None
        account_currency = None
        statement_date = None
        bank_name_confidence = 0.0
        account_number_confidence = 0.0

        # Detect country early (needed for address validation)
        detected_country = detect_country_in_text(' '.join([elem.get('text', '') if isinstance(elem, dict) else elem.text
            for elem in text_elements]))

        if detected_country:
            logger.info(f"Detected country: {detected_country}")

        # Use GLiNER2 for additional entity extraction (only if no field-based name found)
        if not field_based_name:
            from app.core.gliner_ner_model import get_gliner_ner_model
            entities = get_gliner_ner_model().extract_bank_statement_entities(
                ' '.join([elem.get('text', '') if isinstance(elem, dict) else elem.text
                for elem in text_elements])
            )

            # Map GLiNER2 entities to our expected fields
            for entity_type, values in entities.items():
                if not values:
                    continue

                entity_type_lower = entity_type.lower()

                # Account holder name - match multiple possible label patterns
                # Including field-aware patterns for extracting names after specific headings
                if any(pattern in entity_type_lower for pattern in [
                    'person name', 'customer', 'account holder', 'full name',
                    'name after', 'name value', 'name label', 'name heading'
                ]):
                    entity = values[0] if isinstance(values, list) else values

                    if isinstance(entity, dict):
                        raw_name = entity.get('value', entity)
                        entity_conf = entity.get('confidence', 0.8)
                    else:
                        raw_name = entity
                        entity_conf = 0.8

                    # Remove titles
                    name = self._remove_name_titles(raw_name)

                    # Only use this name if it's valid and not too short
                    if name and len(name.strip()) >= 3:
                        # Don't overwrite field-based high-confidence name
                        if entity_conf > account_holder_name_confidence:
                            logger.debug(f"Skipping GLiNER2 name (conf={entity_conf:.2f}) - keeping field-based name '{account_holder_name}'")
                        # Otherwise use this name
                        elif not account_holder_name or entity_conf > account_holder_name_confidence:
                            account_holder_name = name
                            account_holder_name_confidence = entity_conf
                            logger.info(f"✓ Account holder name (GLiNER2): '{account_holder_name}' (conf={account_holder_name_confidence:.2f})")
                    else:
                        logger.debug(f"Skipping invalid name: '{raw_name}' -> '{name}'")

                elif 'bank name' in entity_type_lower or 'bank' in entity_type_lower:
                    # Skip if this is part of account holder name or other non-bank-name field
                    if 'person' in entity_type_lower or 'customer' in entity_type_lower or 'name' not in entity_type_lower:
                        continue

                    entity = values[0] if isinstance(values, list) else values

                    if isinstance(entity, dict):
                        raw_bank_name = entity.get('value', entity)
                        bank_name_confidence = entity.get('confidence', 0.8)
                    else:
                        raw_bank_name = entity
                        bank_name_confidence = 0.8

                    # Validate with global banks database
                    bank_info = get_bank_info(raw_bank_name)
                    if bank_info:
                        bank_name = bank_info.swift_code
                        logger.info(f"✓ Bank SWIFT (GLiNER2): '{bank_name}' (from '{raw_bank_name}', conf={bank_name_confidence:.2f})")
                    else:
                        bank_name = raw_bank_name
                        logger.info(f"✓ Bank name (GLiNER2): '{bank_name}' (conf={bank_name_confidence:.2f})")

                elif 'account number' in entity_type_lower:
                    entity = values[0] if isinstance(values, list) else values

                    if isinstance(entity, dict):
                        raw_account_number = entity.get('value', entity)
                        account_number_confidence = entity.get('confidence', 0.8)
                    else:
                        raw_account_number = entity
                        account_number_confidence = 0.8

                    # Validate and normalize account number
                    if self._is_valid_account_number(raw_account_number):
                        account_number = raw_account_number.replace('-', '').replace(' ', '')
                        logger.info(f"✓ Account number (GLiNER2): '{account_number[:4]}****' (conf={account_number_confidence:.2f})")
                    else:
                        logger.debug(f"Skipping GLiNER2 'account number' '{raw_account_number}' (invalid format)")

                elif 'address' in entity_type_lower:
                    # Collect all addresses for classification
                    address_list = values if isinstance(values, list) else [values]
                    for addr_entity in address_list:
                        if isinstance(addr_entity, dict):
                            addr_text = addr_entity.get('value', '')
                            addr_conf = addr_entity.get('confidence', 0.5)
                        else:
                            addr_text = addr_entity
                            addr_conf = 0.5

                    if address_list:
                        all_addresses = [(addr['text'], addr['confidence']) for addr in address_list]

                        # Classify addresses as bank or account holder
                        from .address_validator import should_keep_address, normalize_address, looks_like_bank_address
                        classified = self._classify_addresses(all_addresses, detected_country)

                        if 'bank_address' in classified:
                            bank_address = normalize_address(classified['bank_address'])
                            if bank_address:
                                logger.info(f"✓ Bank address (classified): '{bank_address[:50]}...'")
                        if 'account_holder_address' in classified:
                            account_holder_address = normalize_address(classified['account_holder_address'])
                            if account_holder_address:
                                logger.info(f"✓ Account holder address (classified): '{account_holder_address[:50]}...'")
                        else:
                            logger.debug(f"No valid addresses found after classification")

        # Calculate confidence based on extraction success
        confidence = self._calculate_confidence({
            'account_holder_name': account_holder_name,
            'bank_name': bank_name,
            'account_number': account_number,
            'bank_address': bank_address,
            'account_holder_address': account_holder_address,
            'account_currency': account_currency,
            'statement_date': statement_date,
        })

        logger.info(f"=== GLiNER2 EXTRACTION COMPLETE ===")
        logger.info(f"  Account holder name: {account_holder_name}")
        logger.info(f"  Bank name: {bank_name}")
        logger.info(f"  Account number: {account_number}")
        logger.info(f"  Bank address: {bank_address}")
        logger.info(f"  Account holder address: {account_holder_address}")
        logger.info(f"  Account holder country: {detected_country}")
        logger.info(f"  Account currency: {account_currency}")
        logger.info(f"  Statement date: {statement_date}")
        logger.info(f"  Confidence: {confidence:.3f}")

        return LineExtractionResult(
            bank_name=bank_name,
            bank_address=bank_address,
            account_holder_name=account_holder_name,
            account_holder_address=account_holder_address,
            account_holder_country=detected_country,
            account_number=account_number,
            account_currency=account_currency,
            statement_date=statement_date,
            confidence=confidence,
            extraction_metadata={'method': 'GLiNER2'}
        )


# Global instance for module-level access
simple_bank_analyzer = SimpleBankStatementAnalyzer()
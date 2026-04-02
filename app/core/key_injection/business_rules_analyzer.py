"""
Business Rules Analyzer for intelligent document content extraction.
This module uses business rules and document layout patterns to extract information
that doesn't follow strict key-value patterns.
"""

import re
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

from .key_field_detector import DetectedKey
from .key_config import DocumentType
from .data_structures import TextElement


class ContentType(Enum):
    """Classification of text content types."""
    NAME = "name"
    ADDRESS = "address"
    ACCOUNT_NUMBER = "account_number"
    BANK_NAME = "bank_name"
    DATE = "date"
    BALANCE = "balance"
    STATEMENT_INFO = "statement_info"
    UNKNOWN = "unknown"


@dataclass
class ContentBlock:
    """Represents a coherent block of content extracted from the document."""
    text: str
    elements: List[TextElement]
    content_type: ContentType
    confidence: float
    geometry: Dict[str, float]  # Combined geometry of all elements
    position: str  # "top_left", "top_right", "middle", "bottom", etc.


class BusinessRulesAnalyzer:
    """
    Analyzes document content using business rules and layout patterns.
    This complements the pattern-based key detection with contextual analysis.
    """

    def __init__(self):
        # Define position zones (normalized coordinates 0-1)
        self.position_zones = {
            'top_left': {'x_max': 0.5, 'y_max': 0.3},
            'top_right': {'x_min': 0.5, 'y_max': 0.3},
            'middle_left': {'x_max': 0.5, 'y_min': 0.3, 'y_max': 0.7},
            'middle_right': {'x_min': 0.5, 'y_min': 0.3, 'y_max': 0.7},
            'bottom': {'y_min': 0.7}
        }

        # Singapore bank name variations
        self.singapore_banks = {
            'POSB', 'DBS', 'OCBC', 'UOB', 'CITIBANK', 'HSBC',
            'STANDARD CHARTERED', 'STANDARD CHARTER', 'MAYBANK',
            'CIMB', 'RHB', 'RHBBANK', 'XDBS', 'BANK OF CHINA'
        }

        # Name patterns for different cultures
        self.name_patterns = [
            # Western names
            re.compile(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+$', re.IGNORECASE),
            # Singapore/Malaysian names with S/O (Son Of) or D/O (Daughter Of)
            re.compile(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+[SD]/O\s+[A-Z\s]+$', re.IGNORECASE),
            # Asian names with spaces and possible initial
            re.compile(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+[A-Z](?:\s+[A-Z])*$', re.IGNORECASE),
            # General multi-word names (starts with capital)
            re.compile(r'^[A-Z][a-zA-Z\s\-\'\.\/]+(?:\s+[A-Z][a-zA-Z\-\'\.\/]+)*$', re.IGNORECASE)
        ]

        # Singapore address patterns (enhanced for multi-line support)
        self.address_patterns = [
            # Block + Street + Postal code
            re.compile(r'^(?:BLK|BLOCK)\s+\d+[A-Z]?\s+[A-Za-z0-9\s\-\#\.]+\s+\d{6}$', re.IGNORECASE),
            # Street address with postal code
            re.compile(r'^[A-Za-z0-9\s\-\#\.]+\s+\d{6}$', re.IGNORECASE),
            # General address pattern
            re.compile(r'^[A-Za-z0-9\s\-\#\.\/]+\s+SINGAPORE\s+\d{6}$', re.IGNORECASE),
            # Street addresses with numbers (comprehensive) - Enhanced to allow additional content
            re.compile(r'^\d+\s+[A-Za-z0-9\s\-\#\.]+(?:\s+(?:CRESCENT|AVENUE|ROAD|STREET|DRIVE|LANE|PLACE|WALK|CLOSE|GROVE|HEIGHTS|HILLS|PARK|TERRACE|VIEW|WAY))', re.IGNORECASE),
            # Enhanced comprehensive address pattern (catches complete addresses)
            re.compile(r'^\d+\s+[A-Z][A-Za-z\s]+(?:ROAD|STREET|AVENUE|CRESCENT|DRIVE|LANE|PLACE|WALK|CLOSE|GROVE|HEIGHTS|HILLS|PARK|TERRACE|VIEW|WAY)', re.IGNORECASE),
            # Individual address components
            re.compile(r'^(?:BLK|BLOCK)\s+\d+', re.IGNORECASE),  # Block numbers
            re.compile(r'^#\d+-\d+$', re.IGNORECASE),  # Unit numbers
            re.compile(r'^SINGAPORE\s+\d{6}$', re.IGNORECASE),  # Postal codes
            re.compile(r'^[A-Z]+(?:\s+(?:CRESCENT|AVENUE|ROAD|STREET|DRIVE|LANE|PLACE|WALK|CLOSE|GROVE|HEIGHTS|HILLS|PARK|TERRACE|VIEW|WAY))', re.IGNORECASE)  # Street names
        ]

        # Account number patterns (more flexible)
        self.account_number_patterns = [
            # Standard account numbers
            re.compile(r'\b\d{8,20}\b'),
            # Account numbers with prefixes
            re.compile(r'\b(?:AC|ACC|ACCOUNT)?\s*#?\s*(\d{8,20})\b', re.IGNORECASE),
            # Hybrid alphanumeric
            re.compile(r'\b[A-Z]{2,4}\d{6,12}\b')
        ]

    def _correct_ocr_errors(self, text: str) -> str:
        """
        Apply OCR error correction for common bank statement terms.

        Args:
            text: Text to correct

        Returns:
            Corrected text
        """
        if not text:
            return text

        # Common OCR error corrections for bank statements
        corrections = {
            # Bank name corrections
            'DOCBC': 'OCBC',
            'O0BC': 'OCBC',
            '0CBC': 'OCBC',
            'DB5': 'DBS',
            'D85': 'DBS',
            'UOB': 'UOB',
            'U0B': 'UOB',
            'P05B': 'POSB',
            'P0SB': 'POSB',

            # Common terms
            'prinsed': 'printed',
            'prinsed ont': 'printed on',
            'prinsed on': 'printed on',
            'ont': 'on',
            'oure': 'our',
            'cvsomentalyliendy': 'environmentally friendly',
            'al lo': 'all',
            'wil': 'will',
            'stataments': 'statements',
            'statemeni': 'statement',
            'accoont': 'account',
            'accoant': 'account',
            'balanoe': 'balance',
            'balanc': 'balance',
            'deplosii': 'deposit',
            'withdr wal': 'withdrawal',
            'transler': 'transfer',
            'transaclion': 'transaction',
            'paymeni': 'payment',
            'chrges': 'charges',
            'inleresi': 'interest',

            # Numbers and symbols
            'O': '0',  # Letter O to zero (context-dependent)
            'I': '1',  # Letter I to one (context-dependent)
            'S': '5',  # Letter S to five (context-dependent)
            ',': '.',  # Comma to period in numbers
        }

        corrected_text = text

        # Apply corrections while preserving context
        for old, new in corrections.items():
            # Use word boundaries to avoid over-correction
            if old in ['O', 'I', 'S', ',']:
                # Context-dependent corrections for individual characters
                continue  # Skip single character corrections for now

            # Apply full word/phrase corrections
            corrected_text = corrected_text.replace(old, new)

        # Handle numeric context corrections separately
        # Convert letter O to zero in numeric contexts
        corrected_text = re.sub(r'\bO(?=\d)', '0', corrected_text)
        # Convert letter I to one in numeric contexts
        corrected_text = re.sub(r'\bI(?=\d)', '1', corrected_text)
        # Convert letter S to five in numeric contexts
        corrected_text = re.sub(r'\bS(?=\d)', '5', corrected_text)

        return corrected_text

    def analyze_document_layout(self, text_elements: List[TextElement]) -> List[ContentBlock]:
        """
        Analyze document layout and identify meaningful content blocks.

        Args:
            text_elements: List of text elements with geometry from OCR

        Returns:
            List of identified content blocks with classification
        """
        from app.core.logger import get_logger
        logger = get_logger()

        logger.info(f"=== BUSINESS RULES ANALYSIS START ===")
        logger.info(f"Analyzing {len(text_elements)} text elements")

        # Apply OCR error correction to all text elements
        corrected_elements = []
        for element in text_elements:
            corrected_text = self._correct_ocr_errors(element.text)
            if corrected_text != element.text:
                logger.debug(f"OCR correction: '{element.text}' -> '{corrected_text}'")

            # Create a new element with corrected text
            corrected_element = TextElement(
                text=corrected_text,
                confidence=element.confidence,
                geometry=element.geometry
            )
            corrected_elements.append(corrected_element)

        # Sort elements by reading order (top to bottom, left to right)
        sorted_elements = sorted(corrected_elements, key=lambda e: (e.geometry['y1'], e.geometry['x1']))

        # Group nearby elements into content blocks
        content_blocks = self._group_elements_into_blocks(sorted_elements)

        # Classify each content block
        classified_blocks = []
        for block in content_blocks:
            content_type = self._classify_content_block(block)
            classified_block = ContentBlock(
                text=" ".join([e.text for e in block.elements]),
                elements=block.elements,
                content_type=content_type,
                confidence=self._calculate_block_confidence(block, content_type),
                geometry=block.geometry,
                position=self._determine_position(block.geometry)
            )
            classified_blocks.append(classified_block)

        logger.info(f"Identified {len(classified_blocks)} content blocks:")
        for i, block in enumerate(classified_blocks):
            logger.info(f"  Block {i+1}: {block.content_type.value} - '{block.text[:50]}...' "
                       f"(position: {block.position}, confidence: {block.confidence:.3f})")

        # Apply business rules to extract information
        extracted_keys = self._apply_business_rules(classified_blocks)

        logger.info(f"Business rules extracted {len(extracted_keys)} keys")
        logger.info("=== BUSINESS RULES ANALYSIS END ===")

        return extracted_keys

    def _group_elements_into_blocks(self, sorted_elements: List[TextElement]) -> List[ContentBlock]:
        """Group nearby text elements into coherent content blocks."""
        blocks = []
        current_block_elements = []

        # Define proximity thresholds
        max_y_gap = 0.05  # Maximum vertical gap between elements in same block
        max_x_gap = 0.15  # Maximum horizontal gap

        # Enhanced grouping for multi-line addresses and related content
        for i, element in enumerate(sorted_elements):
            if not current_block_elements:
                # Start new block
                current_block_elements.append(element)
            else:
                # Check proximity to last element
                last_element = current_block_elements[-1]

                y_gap = abs(element.geometry['y1'] - last_element.geometry['y1'])
                x_gap = abs(element.geometry['x1'] - last_element.geometry['x1'])

                # Check if this could be a continuation of an address or related content
                is_address_continuation = self._is_address_continuation(element, last_element, current_block_elements)
                is_related_content = self._is_related_content(element, last_element, current_block_elements)

                # Enhanced logic for grouping
                should_group = (
                    (y_gap <= max_y_gap and x_gap <= max_x_gap) or  # Standard proximity
                    is_address_continuation or  # Address continuation
                    is_related_content  # Related content (e.g., unit number after address)
                )

                if should_group:
                    current_block_elements.append(element)
                else:
                    # End current block and start new one
                    if current_block_elements:
                        blocks.append(self._create_temp_block(current_block_elements))
                    current_block_elements = [element]

        # Don't forget the last block
        if current_block_elements:
            blocks.append(self._create_temp_block(current_block_elements))

        return blocks

    def _is_address_continuation(self, current_element: TextElement, last_element: TextElement,
                                current_block: List[TextElement]) -> bool:
        """Check if current element is a continuation of an address."""
        current_text = current_element.text.strip()
        last_text = last_element.text.strip()

        # Check for Singapore address patterns
        address_patterns = [
            r'^BLK\s*\d+',  # Block number
            r'^#\d+-\d+',   # Unit number
            r'^\d+\s+[A-Z]+(?:\s+[A-Z]+)*',  # Street name
            r'^SINGAPORE\s+\d{6}',  # Postal code
            r'^[A-Z]+\s+CRESCENT|AVENUE|ROAD|STREET|DRIVE'  # Street types
        ]

        # If current text matches address pattern
        for pattern in address_patterns:
            if re.match(pattern, current_text, re.IGNORECASE):
                # Check if close in vertical position (lines below each other)
                y_gap = abs(current_element.geometry['y1'] - last_element.geometry['y1'])
                x_gap = abs(current_element.geometry['x1'] - last_element.geometry['x1'])

                # For addresses, allow larger vertical gaps but require horizontal alignment
                return (y_gap <= 0.08 and x_gap <= 0.05)  # More lenient for addresses

        # Check if this looks like unit number or postal code after an address line
        if (re.match(r'^#\d+-\d+', current_text) or re.match(r'^SINGAPORE\s+\d{6}', current_text)) and \
           len(current_block) > 0:
            # Check if previous elements look like address
            block_text = ' '.join([elem.text for elem in current_block[-2:]])
            if any(keyword in block_text.upper() for keyword in ['BLK', 'STREET', 'AVENUE', 'ROAD']):
                return True

        return False

    def _is_related_content(self, current_element: TextElement, last_element: TextElement,
                           current_block: List[TextElement]) -> bool:
        """Check if current element is related to the current block content."""
        current_text = current_element.text.strip()
        last_text = last_element.text.strip()

        # Check for unit numbers following addresses
        if re.match(r'^#\d+-\d+', current_text) and len(current_block) > 0:
            # Check if previous elements look like address components
            block_text = ' '.join([elem.text for elem in current_block])
            if any(keyword in block_text.upper() for keyword in ['BLK', 'CRESCENT', 'AVENUE', 'STREET']):
                return True

        # Check for postal codes following addresses
        if re.match(r'^SINGAPORE\s+\d{6}', current_text) and len(current_block) > 0:
            block_text = ' '.join([elem.text for elem in current_block])
            if any(keyword in block_text.upper() for keyword in ['BLK', 'CRESCENT', 'AVENUE', 'STREET', '#']):
                return True

        return False

    def _create_temp_block(self, elements: List[TextElement]) -> 'ContentBlock':
        """Create a temporary block from elements."""
        # Calculate combined geometry
        min_x = min(e.geometry['x1'] for e in elements)
        min_y = min(e.geometry['y1'] for e in elements)
        max_x = max(e.geometry['x2'] for e in elements)
        max_y = max(e.geometry['y2'] for e in elements)

        geometry = {'x1': min_x, 'y1': min_y, 'x2': max_x, 'y2': max_y}

        return ContentBlock(
            text="", elements=elements, content_type=ContentType.UNKNOWN,
            confidence=0.0, geometry=geometry, position=""
        )

    def _classify_content_block(self, block: ContentBlock) -> ContentType:
        """Classify the content type of a block based on its text and position."""
        from app.core.logger import get_logger
        logger = get_logger()

        text = block.text.strip()
        position = self._determine_position(block.geometry)

        logger.debug(f"Classifying block: '{text[:50]}...' at position {position}")

        # Check for bank names (high confidence)
        for bank in self.singapore_banks:
            if bank.lower() in text.lower():
                logger.debug(f"Classified as BANK_NAME: {text}")
                return ContentType.BANK_NAME

        # Check for account numbers
        for pattern in self.account_number_patterns:
            if re.search(pattern, text):
                logger.debug(f"Classified as ACCOUNT_NUMBER: {text}")
                return ContentType.ACCOUNT_NUMBER

        # Enhanced position-based classification for names and addresses
        if position in ['top_left', 'top_right', 'upper_left', 'upper_center', 'upper_right']:
            # First check for address patterns (more specific)
            is_address = False
            for addr_pattern in self.address_patterns:
                if re.search(addr_pattern, text.strip()):
                    is_address = True
                    break

            if is_address:
                logger.debug(f"Classified as ADDRESS: {text}")
                return ContentType.ADDRESS

            # Then check for name patterns with enhanced validation
            name_score = self._calculate_enhanced_name_score(text)
            if name_score >= 0.6:  # Higher threshold for business rules
                logger.debug(f"Classified as NAME: {text} (score: {name_score:.3f})")
                return ContentType.NAME

        # Check for statement information
        if any(keyword in text.lower() for keyword in [
            'statement', 'account summary', 'consolidated', 'as at', 'period'
        ]):
            logger.debug(f"Classified as STATEMENT_INFO: {text}")
            return ContentType.STATEMENT_INFO

        # Check for dates
        date_pattern = re.compile(r'\b\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[a-z]*\s+\d{4}\b', re.IGNORECASE)
        if date_pattern.search(text):
            logger.debug(f"Classified as DATE: {text}")
            return ContentType.DATE

        # Check for balances (currency patterns)
        currency_pattern = re.compile(r'[\$\$£\€]\s*[\d,]+\.\d{2}')
        if currency_pattern.search(text):
            logger.debug(f"Classified as BALANCE: {text}")
            return ContentType.BALANCE

        logger.debug(f"Classified as UNKNOWN: {text}")
        return ContentType.UNKNOWN

    def _calculate_block_confidence(self, block: ContentBlock, content_type: ContentType) -> float:
        """Calculate confidence score for content classification."""
        base_confidence = 0.5
        text = block.text.strip()

        if content_type == ContentType.BANK_NAME:
            # High confidence if it matches known bank names
            for bank in self.singapore_banks:
                if bank.lower() == text.lower():
                    return 0.95
                elif bank.lower() in text.lower():
                    return 0.85

        elif content_type == ContentType.NAME:
            # Check name pattern matches
            for i, pattern in enumerate(self.name_patterns):
                if re.match(pattern, text.strip()):
                    base_confidence = 0.7 + (i * 0.1)  # Higher patterns get higher confidence

                    # Apply position-based boosting for names
                    position_boost = self._calculate_name_position_boost(block.position, text)
                    final_confidence = min(base_confidence * position_boost, 1.0)

                    return final_confidence

        elif content_type == ContentType.ADDRESS:
            # Check address pattern matches
            for i, pattern in enumerate(self.address_patterns):
                if re.search(pattern, text.strip()):
                    return 0.7 + (i * 0.1)

        elif content_type == ContentType.ACCOUNT_NUMBER:
            # Check account number patterns
            for pattern in self.account_number_patterns:
                if re.search(pattern, text):
                    return 0.9

        return base_confidence

    def _determine_position(self, geometry: Dict[str, float]) -> str:
        """Determine the position zone of the content block."""
        x_center = (geometry['x1'] + geometry['x2']) / 2
        y_center = (geometry['y1'] + geometry['y2']) / 2

        for zone_name, zone_bounds in self.position_zones.items():
            if 'x_min' in zone_bounds and x_center < zone_bounds['x_min']:
                continue
            if 'x_max' in zone_bounds and x_center > zone_bounds['x_max']:
                continue
            if 'y_min' in zone_bounds and y_center < zone_bounds['y_min']:
                continue
            if 'y_max' in zone_bounds and y_center > zone_bounds['y_max']:
                continue
            return zone_name

        return 'middle'

    def _apply_business_rules(self, content_blocks: List[ContentBlock]) -> List[DetectedKey]:
        """Apply business rules to extract meaningful information from content blocks."""
        from app.core.logger import get_logger
        logger = get_logger()

        extracted_keys = []

        # Sort blocks by Y position (top to bottom) for better context analysis
        sorted_blocks = sorted(content_blocks, key=lambda b: b.geometry.get('y1', 0))

        # Pre-extract address blocks for proximity analysis
        address_blocks = [block for block in sorted_blocks if block.content_type == ContentType.ADDRESS]
        logger.debug(f"Found {len(address_blocks)} address blocks for proximity analysis")

        for block in sorted_blocks:
            if block.content_type == ContentType.NAME:
                # Use consolidated account_name detection with enhanced business rules including proximity
                account_name_key = self._analyze_account_name_from_block_with_proximity(block, address_blocks)
                if account_name_key:
                    extracted_keys.append(account_name_key)
                    logger.debug(f"Business Rules: Added NAME '{account_name_key.value_candidate}' at {block.position}")

            elif block.content_type == ContentType.ADDRESS:
                # Enhanced address analysis to distinguish from names
                address_analysis = self._analyze_address_from_block(block, sorted_blocks)
                if address_analysis:
                    extracted_keys.append(address_analysis)
                    logger.debug(f"Business Rules: Added ADDRESS '{address_analysis.value_candidate}' at {block.position}")

            elif block.content_type == ContentType.BANK_NAME:
                key = DetectedKey(
                    key_name='bank_name',
                    key_text=block.text,
                    confidence=block.confidence,
                    geometry=block.geometry,
                    value_candidate=block.text,
                    value_confidence=block.confidence,
                    document_type=DocumentType.BANK_STATEMENT
                )
                extracted_keys.append(key)

            elif block.content_type == ContentType.ACCOUNT_NUMBER:
                # Extract the actual account number from the text
                for pattern in self.account_number_patterns:
                    match = re.search(pattern, block.text)
                    if match:
                        account_number = match.group(1) if match.groups() else match.group(0)
                        key = DetectedKey(
                            key_name='account_number',
                            key_text=block.text,
                            confidence=block.confidence,
                            geometry=block.geometry,
                            value_candidate=account_number,
                            value_confidence=block.confidence,
                            document_type=DocumentType.BANK_STATEMENT
                        )
                        extracted_keys.append(key)
                        break

        return extracted_keys

    def find_spatial_key_value_pairs(self, text_elements: List[TextElement]) -> List[DetectedKey]:
        """
        Find key-value pairs based on spatial relationships.
        This handles cases where keys are near their values but not on the same line.
        """
        from app.core.logger import get_logger
        logger = get_logger()

        logger.info("=== SPATIAL KEY-VALUE ANALYSIS ===")
        extracted_keys = []

        # Common key terms in bank statements
        key_terms = {
            'account_number': ['account no', 'account number', 'acc no', 'a/c no', 'account#'],
            'account_holder': ['account holder', 'account name', 'customer name', 'name'],
            'opening_balance': ['opening balance', 'beginning balance', 'brought forward', 'balance bf'],
            'closing_balance': ['closing balance', 'ending balance', 'balance cf', 'final balance'],
            'statement_date': ['statement date', 'as at', 'date'],
            'branch': ['branch code', 'branch']
        }

        for element in text_elements:
            text_lower = element.text.lower().strip()

            # Check if this element is a key term
            for key_name, terms in key_terms.items():
                for term in terms:
                    if term in text_lower:
                        # Look for values near this key
                        value_candidate, value_confidence = self._find_nearby_value(
                            element, text_elements, key_name
                        )

                        if value_candidate:
                            detected_key = DetectedKey(
                                key_name=key_name,
                                key_text=element.text,
                                confidence=0.8,
                                geometry=element.geometry,
                                value_candidate=value_candidate,
                                value_confidence=value_confidence,
                                document_type=DocumentType.BANK_STATEMENT
                            )
                            extracted_keys.append(detected_key)
                            logger.info(f"Found spatial key-value: '{key_name}' = '{value_candidate}' "
                                       f"(confidence: {value_confidence:.3f})")
                            break

        logger.info(f"Spatial analysis found {len(extracted_keys)} key-value pairs")
        logger.info("=== END SPATIAL KEY-VALUE ANALYSIS ===")
        return extracted_keys

    def _find_nearby_value(self, key_element: TextElement, all_elements: List[TextElement],
                          key_name: str) -> Tuple[Optional[str], float]:
        """Find a value near a key element."""
        key_center_x = (key_element.geometry['x1'] + key_element.geometry['x2']) / 2
        key_center_y = (key_element.geometry['y1'] + key_element.geometry['y2']) / 2

        best_candidate = None
        best_confidence = 0.0

        for element in all_elements:
            if element == key_element:
                continue

            element_center_x = (element.geometry['x1'] + element.geometry['x2']) / 2
            element_center_y = (element.geometry['y1'] + element.geometry['y2']) / 2

            # Calculate distance
            x_diff = abs(element_center_x - key_center_x)
            y_diff = abs(element_center_y - key_center_y)
            distance = (x_diff ** 2 + y_diff ** 2) ** 0.5

            # Only consider nearby elements
            if distance > 0.2:  # Maximum distance threshold
                continue

            # Check if this looks like a value for the key type
            if self._is_valid_value_for_key(element.text, key_name):
                # Calculate confidence based on distance and content validity
                content_confidence = self._calculate_value_confidence(element.text, key_name)
                distance_confidence = max(0, 1 - distance / 0.2)  # Linear decay
                combined_confidence = (content_confidence * 0.7 + distance_confidence * 0.3)

                if combined_confidence > best_confidence:
                    best_candidate = element.text.strip()
                    best_confidence = combined_confidence

        return best_candidate, best_confidence

    def _is_valid_value_for_key(self, text: str, key_name: str) -> bool:
        """Check if text looks like a valid value for the given key."""
        text = text.strip()

        if key_name == 'account_number':
            return bool(re.match(r'^[\d\s\-A-Z]+$', text))
        elif key_name == 'opening_balance' or key_name == 'closing_balance':
            return bool(re.match(r'^[\$\£\€]?\s*[\d,]+\.\d{2}$', text))
        elif key_name == 'statement_date':
            return bool(re.search(r'\d{1,2}[\s\-\/]\d{1,2}[\s\-\/]\d{2,4}', text))
        elif key_name == 'branch':
            return len(text) <= 10 and text.replace(' ', '').isalnum()

        return True

    def _calculate_value_confidence(self, text: str, key_name: str) -> float:
        """Calculate confidence score for a value based on its format."""
        text = text.strip()

        if key_name == 'account_number':
            if re.match(r'^\d{8,16}$', text):
                return 0.95
            elif re.match(r'^\d{6,7}[\-\s]\d$', text):
                return 0.9
            else:
                return 0.7

        elif key_name in ['opening_balance', 'closing_balance']:
            if re.match(r'^\$\d{1,3}(?:,\d{3})*\.\d{2}$', text):
                return 0.95
            else:
                return 0.7

        elif key_name == 'statement_date':
            if re.search(r'\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[a-z]*\s+\d{4}', text, re.IGNORECASE):
                return 0.95
            else:
                return 0.7

        return 0.8  # Default confidence

    def extract_key_value_pairs_from_geometry(self, ocr_results: List[Dict], document_type: DocumentType) -> List[DetectedKey]:
        """
        Extract key-value pairs from OCR geometry data using business rules analysis.
        This method integrates with the spatial analyzer by accepting OCR results
        and returning DetectedKey objects.

        Args:
            ocr_results: List of OCR results with geometry from doctr
                        [{'text': 'text', 'x1': 0.1, 'y1': 0.2, 'x2': 0.3, 'y2': 0.4, 'confidence': 0.95}, ...]
            document_type: Type of document being processed

        Returns:
            List of DetectedKey objects extracted using business rules
        """
        from app.core.logger import get_logger
        logger = get_logger()

        logger.info(f"=== BUSINESS RULES ANALYSIS START ===")
        logger.info(f"Processing {len(ocr_results)} OCR results for {document_type.value}")

        # Convert OCR results to TextElement objects
        text_elements = []
        for result in ocr_results:
            if not result.get('text', '').strip():
                continue  # Skip empty text elements

            text_elements.append(TextElement(
                text=result['text'],
                confidence=result.get('confidence', 0.9),
                geometry={
                    'x1': result['x1'],
                    'y1': result['y1'],
                    'x2': result['x2'],
                    'y2': result['y2']
                }
            ))

        logger.info(f"Converted to {len(text_elements)} text elements")

        if not text_elements:
            logger.warning("No text elements to process")
            return []

        # Step 1: Layout analysis for contextual information (names, addresses)
        logger.info("Starting layout analysis...")
        content_blocks = self.analyze_document_layout(text_elements)
        layout_detected_keys = self._apply_business_rules(content_blocks)

        logger.info(f"Layout analysis found {len(layout_detected_keys)} keys:")
        for i, key in enumerate(layout_detected_keys):
            logger.info(f"  Layout Key {i+1}: '{key.key_name}' -> '{key.value_candidate}' "
                       f"(confidence: {key.confidence:.3f})")

        # Step 2: Spatial key-value pairs for labeled fields
        logger.info("Starting spatial key-value analysis...")
        spatial_detected_keys = self.find_spatial_key_value_pairs(text_elements)

        logger.info(f"Spatial analysis found {len(spatial_detected_keys)} keys:")
        for i, key in enumerate(spatial_detected_keys):
            logger.info(f"  Spatial Key {i+1}: '{key.key_name}' -> '{key.value_candidate}' "
                       f"(confidence: {key.confidence:.3f})")

        # Step 3: Combine results
        all_business_keys = layout_detected_keys + spatial_detected_keys

        # Remove duplicates within business rules results
        unique_keys = self._remove_duplicate_business_keys(all_business_keys)

        logger.info(f"After combining and removing duplicates, we have {len(unique_keys)} business rules keys:")
        for i, key in enumerate(unique_keys):
            logger.info(f"  Business Rules Key {i+1}: '{key.key_name}' -> '{key.value_candidate}' "
                       f"(confidence: {key.confidence:.3f})")

        logger.info(f"=== BUSINESS RULES ANALYSIS END ===")
        return unique_keys

    def _remove_duplicate_business_keys(self, keys: List[DetectedKey]) -> List[DetectedKey]:
        """
        Remove duplicate keys from business rules analysis, keeping the highest confidence version.
        """
        from app.core.logger import get_logger
        logger = get_logger()

        key_dict = {}

        for key in keys:
            if key.key_name not in key_dict or key.confidence > key_dict[key.key_name].confidence:
                key_dict[key.key_name] = key
                logger.debug(f"Updated key '{key.key_name}' with higher confidence version")

        return list(key_dict.values())

  
    def _calculate_name_position_boost(self, position: str, text: str) -> float:
        """
        Calculate position-based confidence boost for names based on document layout patterns.

        Args:
            position: The position zone (top_left, top_right, etc.)
            text: The text content

        Returns:
            Confidence boost multiplier (1.0 = no boost, >1.0 = boost)
        """
        from app.core.logger import get_logger
        logger = get_logger()

        # Define high-probability name zones with boost factors
        name_position_boosts = {
            'top_left': 1.3,      # Names typically appear in top-left
            'top_center': 1.2,     # Centered headers
            'top_right': 1.15,     # Sometimes names appear top-right
            'upper_left': 1.2,     # Upper area but not very top
            'upper_center': 1.1,   # Upper center area
            'upper_right': 1.05,   # Upper right area
        }

        # Default boost for other positions
        base_boost = 1.0

        # Get position-specific boost
        position_boost = name_position_boosts.get(position, base_boost)

        # Additional text-based boosts
        text_boost = 1.0

        # Boost for names that look like account holders
        if any(indicator in text.upper() for indicator in ['S/O', 'D/O', 'B/O']):
            text_boost *= 1.1  # Cultural format boost

        # Boost for properly capitalized names
        if re.match(r'^[A-Z][a-zA-Z\s\-\'\.\/]+$', text):
            text_boost *= 1.05

        # Reduce boost for generic words that might be mistakenly classified as names
        generic_words = [
            'STATEMENT', 'ACCOUNT', 'PAGE', 'DATE', 'BALANCE', 'DEPOSIT',
            'WITHDRAWAL', 'TRANSACTION', 'SUMMARY', 'TOTAL', 'BANK'
        ]
        if text.strip().upper() in generic_words:
            text_boost *= 0.5  # Penalize generic words

        # Combined boost
        final_boost = position_boost * text_boost

        # Cap the boost to prevent unrealistic confidence
        final_boost = min(final_boost, 1.5)

        logger.debug(f"Name position boost for '{text}' at {position}: {final_boost:.2f} "
                    f"(position: {position_boost:.2f}, text: {text_boost:.2f})")

        return final_boost

    def _analyze_account_name_from_block(self, block: ContentBlock) -> Optional[DetectedKey]:
        """
        Analyze a content block classified as a name and extract account name using enhanced business rules.
        Unified method that handles both personal names and company names.

        Args:
            block: Content block classified as ContentType.NAME

        Returns:
            DetectedKey with 'account_name' as key_name, or None if not a valid name
        """
        if not block or not block.text.strip():
            return None

        text = block.text.strip()
        position = block.position

        # Enhanced name validation using cultural and format patterns
        name_score = self._calculate_enhanced_name_score(text)
        if name_score < 0.4:
            return None

        # Apply position-based confidence boosting for bank statements
        position_boost = self._calculate_name_position_boost(position, text)

        # Additional business rules for bank statement name detection
        bank_statement_boost = self._calculate_bank_statement_name_boost(text, position)

        # NEW: Enhanced business statement name detection with top-30% filtering and transaction exclusion
        business_statement_boost = self._calculate_business_statement_name_boost(
            text, position, block.geometry
        )

        # NEW: Name-address proximity boost for enhanced confidence
        proximity_boost = self._calculate_name_address_proximity_boost(
            block, self._get_all_address_blocks()
        )

        # Calculate combined confidence
        base_confidence = 0.7
        combined_confidence = min(base_confidence * position_boost * bank_statement_boost * name_score * business_statement_boost * proximity_boost, 1.0)

        # Enhanced value confidence based on name characteristics
        value_confidence = self._calculate_name_value_confidence(text, position)

        return DetectedKey(
            key_name="account_name",
            key_text=text,
            confidence=combined_confidence,
            geometry=block.geometry,
            value_candidate=text,
            value_confidence=value_confidence,
            document_type=DocumentType.BANK_STATEMENT
        )

    def _analyze_account_name_from_block_with_proximity(self, block: ContentBlock, address_blocks: List[ContentBlock]) -> Optional[DetectedKey]:
        """
        Analyze a content block classified as a name with address proximity enhancement.
        Unified method that handles both personal names and company names as account_name.

        Args:
            block: Content block classified as ContentType.NAME
            address_blocks: List of address blocks for proximity analysis

        Returns:
            DetectedKey with 'account_name' as key_name, or None if not a valid name
        """
        if not block or not block.text.strip():
            return None

        text = block.text.strip()
        position = block.position

        # Enhanced name validation using cultural and format patterns
        name_score = self._calculate_enhanced_name_score(text)
        if name_score < 0.4:
            return None

        from app.core.logger import get_logger
        logger = get_logger()
        logger.debug(f"Analyzing account name '{text}' (score: {name_score:.3f})")

        # Apply position-based confidence boosting for bank statements
        position_boost = self._calculate_name_position_boost(position, text)

        # Additional business rules for bank statement name detection
        bank_statement_boost = self._calculate_bank_statement_name_boost(text, position)

        # Enhanced business statement name detection with top-30% filtering and transaction exclusion
        business_statement_boost = self._calculate_business_statement_name_boost(
            text, position, block.geometry
        )

        # Name-address proximity boost for enhanced confidence
        proximity_boost = self._calculate_name_address_proximity_boost(block, address_blocks)

        # Calculate combined confidence
        base_confidence = 0.7
        combined_confidence = min(base_confidence * position_boost * bank_statement_boost * name_score * business_statement_boost * proximity_boost, 1.0)

        # Enhanced value confidence based on name characteristics
        value_confidence = self._calculate_name_value_confidence(text, position)

        return DetectedKey(
            key_name="account_name",
            key_text=text,
            confidence=combined_confidence,
            geometry=block.geometry,
            value_candidate=text,
            value_confidence=value_confidence,
            document_type=DocumentType.BANK_STATEMENT
        )

    def _calculate_enhanced_name_score(self, text: str) -> float:
        """
        Calculate a comprehensive name score using cultural and format analysis.

        Args:
            text: Text to evaluate as a potential name

        Returns:
            Score from 0.0 to 1.0 indicating likelihood of being a name
        """
        from app.core.logger import get_logger
        logger = get_logger()

        if not text or len(text.strip()) < 2:
            return 0.0

        text = text.strip()
        base_score = 0.4

        # First, filter out obvious non-name patterns
        if self._is_likely_not_name(text):
            logger.debug(f"Text filtered as non-name: '{text}'")
            return 0.0

        # Cultural format detection (high confidence)
        cultural_patterns = [
            # Singapore/Malaysian S/O D/O B/O patterns
            (r'^[A-Z][a-zA-Z\s\-\'\.\/]+(?:\s+[SD]/O\s+[A-Z\s]+)$', 0.95),
            (r'^[A-Z][a-zA-Z\s\-\'\.\/]+(?:\s+B/O\s+[A-Z\s]+)$', 0.90),
            # Arabic/Middle Eastern bin/binte patterns
            (r'^[A-Z][a-zA-Z\s\-\'\.\/]+\s+(?:bin|binte)\s+[A-Z\s\-\'\.\/]+$', 0.90),
            # European patronymic patterns
            (r'^[A-Z][a-zA-Z\s\-\'\.\/]+\s+(?:de|van|von|da|del|di)\s+[A-Z\s\-\'\.\/]+$', 0.85),
        ]

        for pattern, score in cultural_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                logger.debug(f"Cultural pattern match: '{text}' (score: {score})")
                return score

        # Western name patterns
        western_patterns = [
            # Standard Western names (2-4 words, proper capitalization)
            (r'^[A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-]+){1,3}$', 0.85),
            # Names with middle initials
            (r'^[A-Z][a-zA-Z\'\-]+\s+[A-Z](?:\.[A-Z])*\.?\s+(?:[A-Z][a-zA-Z\'\-]+)?$', 0.80),
            # Hyphenated surnames
            (r'^[A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-]+)*\s+[A-Z][a-zA-Z\'\-]+(?:-[A-Z][a-zA-Z\'\-]+)+$', 0.80),
        ]

        for pattern, score in western_patterns:
            if re.match(pattern, text):
                logger.debug(f"Western pattern match: '{text}' (score: {score})")
                return score

        # Company name patterns (for business accounts)
        company_patterns = [
            # Singapore/Malaysian company formats (PTE. LTD., etc.)
            (r'^[A-Z][A-Za-z\s&\.\-\']+\s+(?:PTE\.?\s*LTD\.?|LTD\.?|INC\.?|CORP\.?|SDN\.?\s*BHD\.?)$', 0.90),
            # General company names with business suffixes
            (r'^[A-Z][A-Za-z\s&\.\-\']+(?:\s+(?:CO|COMPANY|CORP|CORPORATION|ENTERPRISES|TRADING|GROUP|HOLDINGS))$', 0.85),
            # Multi-word business names (likely legitimate companies)
            (r'^[A-Z][A-Za-z\s&\.\-\']+\s+[A-Z][A-Za-z\s&\.\-\']+(?:\s+[A-Z][A-Za-z\s&\.\-\']+)?$', 0.75),
        ]

        for pattern, score in company_patterns:
            if re.match(pattern, text):
                logger.debug(f"Company pattern match: '{text}' (score: {score})")
                return score

        # General Asian name patterns (broader coverage)
        asian_patterns = [
            # Short Asian names (2-3 parts)
            (r'^[A-Z][a-zA-Z\s\-\'\.\/]{2,25}$', 0.70),
            # Multiple-word Asian names
            (r'^[A-Z][a-zA-Z\s\-\'\.\/]+\s+[A-Z][a-zA-Z\s\-\'\.\/]+$', 0.75),
        ]

        for pattern, score in asian_patterns:
            if re.match(pattern, text):
                logger.debug(f"Asian pattern match: '{text}' (score: {score})")
                return max(base_score, score)

        # Penalize patterns that look like non-names and OCR errors
        non_name_indicators = [
            r'\d',  # Contains numbers
            r'^[A-Z]{5,}$',  # All caps single words (likely OCR errors like "OSINSOTA", "ANBOSESNEM")
            r'^[A-Z]{3,}\s+[A-Z]{3,}$',  # Multiple all-caps words (likely OCR fragments)
            r'^(?:STATEMENT|ACCOUNT|PAGE|DATE|BALANCE|DEPOSIT|WITHDRAWAL|TRANSACTION|SUMMARY|TOTAL|BANK|INFORMATION|WPI)$',
            r'^(?:DOCBC|OSINSOTA|ANBOSESNEM)$',  # Known OCR errors from logs
            r'[^\w\s\-\'\.\/]',  # Special characters
        ]

        for pattern in non_name_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                logger.debug(f"Non-name pattern detected: '{text}' (penalty applied)")
                return 0.0

        # Default score for properly capitalized multi-word text
        if re.match(r'^[A-Z][a-zA-Z\s\-\'\.\/]+\s+[A-Z]', text):
            logger.debug(f"Default capitalized pattern: '{text}' (score: {base_score})")
            return base_score

        logger.debug(f"No strong name pattern match: '{text}' (score: 0.0)")
        return 0.0

    def _is_top_30_percent(self, geometry: Dict[str, float]) -> bool:
        """
        Check if element is in top 30% of page.

        Args:
            geometry: Element geometry with x1, y1, x2, y2 coordinates

        Returns:
            True if element is in top 30% of page
        """
        y_center = (geometry['y1'] + geometry['y2']) / 2
        return y_center <= 0.3  # Top 30% threshold

    def _detect_account_type(self, text_elements: List['TextElement']) -> str:
        """
        Detect if document is business or personal account based on content.

        Args:
            text_elements: List of text elements from OCR

        Returns:
            'business' or 'personal'
        """
        # Check for business indicators in top 30%
        business_indicators = ['PTE', 'LTD', 'INC', 'CORP', 'ENTERPRISE', 'COMPANY', 'SDN', 'BHD']

        top_30_elements = [e for e in text_elements if self._is_top_30_percent(e.geometry)]

        for element in top_30_elements:
            text_upper = element.text.upper()
            if any(indicator in text_upper for indicator in business_indicators):
                return 'business'

        return 'personal'

    def _is_transaction_area(self, geometry: Dict[str, float]) -> bool:
        """
        Check if geometry is in transaction/summary area (middle of page).

        Args:
            geometry: Element geometry with x1, y1, x2, y2 coordinates

        Returns:
            True if element is in typical transaction area
        """
        y_center = (geometry['y1'] + geometry['y2']) / 2

        # Middle 40% of page (typical transaction area)
        if 0.3 <= y_center <= 0.7:
            return True
        return False

    def _calculate_business_statement_name_boost(
        self, text: str, position: str, geometry: Dict[str, float]
    ) -> float:
        """
        Calculate enhanced bank statement name boost with business detection.

        Args:
            text: The text content
            position: The position zone
            geometry: Element geometry for spatial analysis

        Returns:
            Boost multiplier (>= 1.0)
        """
        from app.core.logger import get_logger
        logger = get_logger()

        boost = 1.0

        # Massive boost for top-30% positions (business account holder names are here)
        if self._is_top_30_percent(geometry):
            boost *= 2.0  # 100% boost for top 30%
            logger.debug(f"Applied top-30% boost for: '{text}'")

        # Position-based boosts for bank statements
        if position == 'top_left':
            boost *= 1.3  # Highest boost - typical name position
        elif position == 'top_right':
            boost *= 1.15  # Secondary position
        elif position == 'top_center':
            boost *= 1.2  # Centered headers

        # Business-specific boosts
        text_upper = text.upper()

        # Extra boost for company indicators
        business_indicators = ['PTE', 'LTD', 'INC', 'CORP', 'SDN', 'BHD', 'ENTERPRISES', 'TRADING', 'GROUP', 'HOLDINGS']
        if any(indicator in text_upper for indicator in business_indicators):
            boost *= 1.5
            logger.debug(f"Applied business indicator boost for: '{text}'")

        # Boost for cultural name formats common in Singapore/Malaysia
        if any(indicator in text_upper for indicator in [' S/O ', ' D/O ', ' B/O ']):
            boost *= 1.1

        # Boost for company names (multi-word with business structure)
        words = text.split()
        if len(words) >= 3 and any(indicator in text_upper for indicator in business_indicators):
            boost *= 1.3
            logger.debug(f"Applied multi-word company boost for: '{text}'")

        # Boost for names that look like account holders
        if len(words) >= 2 and len(words) <= 4:
            boost *= 1.05

        return boost

    def _calculate_bank_statement_name_boost(self, text: str, position: str) -> float:
        """
        Calculate bank statement specific boost for name detection.

        Args:
            text: The text content
            position: The position zone

        Returns:
            Boost multiplier (>= 1.0)
        """
        boost = 1.0

        # Position-based boosts for bank statements
        if position == 'top_left':
            boost *= 1.3  # Highest boost - typical name position
        elif position == 'top_right':
            boost *= 1.15  # Secondary position
        elif position == 'top_center':
            boost *= 1.2  # Centered headers

        # Content-based boosts
        text_upper = text.upper()

        # Boost for cultural name formats common in Singapore/Malaysia
        if any(indicator in text_upper for indicator in [' S/O ', ' D/O ', ' B/O ']):
            boost *= 1.1

        # Boost for names that look like personal account holders
        if len(text.split()) >= 2 and len(text.split()) <= 4:  # Typical name length
            boost *= 1.05

        # Reduce boost for text that looks like headers or labels
        header_words = ['ACCOUNT', 'STATEMENT', 'SUMMARY', 'CONSOLIDATED', 'PAGE']
        if any(word in text_upper for word in header_words):
            boost *= 0.3

        return max(boost, 0.5)  # Minimum boost to prevent zero confidence

    def _calculate_name_address_proximity_boost(self, name_block: ContentBlock, address_blocks: List[ContentBlock]) -> float:
        """
        Calculate confidence boost based on proximity between name and address blocks.

        In bank statements, name and address are always the closest located data points.
        This method boosts confidence for name-address pairs that are within 15% vertical distance.

        Args:
            name_block: The content block containing a name
            address_blocks: List of content blocks containing addresses

        Returns:
            Proximity boost multiplier (>= 1.0)
        """
        from app.core.logger import get_logger
        logger = get_logger()

        if not address_blocks:
            logger.debug("No address blocks found for proximity analysis")
            return 1.0  # No boost if no addresses to compare

        name_geometry = name_block.geometry
        name_center_y = (name_geometry['y1'] + name_geometry['y2']) / 2
        name_center_x = (name_geometry['x1'] + name_geometry['x2']) / 2

        # Find the closest address block
        closest_distance = float('inf')
        closest_address = None

        for address_block in address_blocks:
            addr_geometry = address_block.geometry
            addr_center_y = (addr_geometry['y1'] + addr_geometry['y2']) / 2
            addr_center_x = (addr_geometry['x1'] + addr_geometry['x2']) / 2

            # Calculate Euclidean distance between name and address centers
            y_distance = abs(name_center_y - addr_center_y)
            x_distance = abs(name_center_x - addr_center_x)

            # Weight vertical distance more heavily (names are typically above/below addresses)
            distance = (y_distance ** 2 * 2 + x_distance ** 2) ** 0.5

            if distance < closest_distance:
                closest_distance = distance
                closest_address = address_block

        # Calculate boost based on distance (within 15% vertical distance gets highest boost)
        if closest_distance <= 0.15:  # Within 15% of page height
            boost = 1.0 + (0.15 - closest_distance) * 2  # Max 1.3x boost for very close pairs
            logger.debug(f"Name-address proximity boost: {boost:.3f} (distance: {closest_distance:.3f}) "
                        f"between '{name_block.text[:20]}...' and '{closest_address.text[:20]}...'")
        else:
            boost = 1.0  # No boost for distant pairs
            logger.debug(f"No proximity boost: distance {closest_distance:.3f} > 0.15 threshold")

        return boost

    def _get_all_address_blocks(self) -> List[ContentBlock]:
        """
        Get all content blocks classified as addresses from the current analysis.

        Returns:
            List of address content blocks
        """
        # This is a placeholder - in practice, we'd need to pass the list of all blocks
        # or store them as an instance variable during the analysis phase
        # For now, return empty list - this method will need to be integrated with the main analysis flow
        return []

    def _calculate_name_value_confidence(self, text: str, position: str) -> float:
        """
        Calculate confidence specifically for the value part of name detection.

        Args:
            text: The extracted name text
            position: The position zone

        Returns:
            Confidence score from 0.0 to 1.0
        """
        base_confidence = 0.7

        # Format validation
        if re.match(r'^[A-Z][a-zA-Z\s\-\'\.\/]+$', text):
            base_confidence += 0.1  # Proper capitalization

        # Length validation (typical name lengths)
        word_count = len(text.split())
        if 2 <= word_count <= 4:  # Typical name word count
            base_confidence += 0.1
        elif word_count > 6 or word_count == 1:
            base_confidence -= 0.2

        # Position validation
        if position == 'top_left':
            base_confidence += 0.15
        elif position in ['top_right', 'top_center']:
            base_confidence += 0.1

        # Cultural format validation
        if any(indicator in text.upper() for indicator in ['S/O', 'D/O', 'B/O', 'BIN', 'BINTE']):
            base_confidence += 0.1

        return min(max(base_confidence, 0.0), 1.0)

    def _analyze_address_from_block(self, block: ContentBlock, all_blocks: List[ContentBlock]) -> Optional[DetectedKey]:
        """
        Analyze a content block as an address with enhanced discrimination from names.

        Args:
            block: Content block to analyze
            all_blocks: All content blocks for context analysis

        Returns:
            DetectedKey with address information, or None if not a valid address
        """
        from app.core.logger import get_logger
        logger = get_logger()

        text = block.text.strip()

        # Check if this is actually a name misclassified as address
        name_score = self._calculate_enhanced_name_score(text)

        # For S/O patterns and high-confidence names, prefer name classification over address
        if name_score >= 0.85:
            logger.debug(f"High-confidence name detected in address block: '{text}' (score: {name_score:.3f})")
            return None  # Let name detection handle this

        # Enhanced address validation
        address_indicators = [
            r'BLK\s+\d+',  # Block numbers
            r'#\d+-\d+',   # Unit numbers
            r'SINGAPORE\s+\d{6}',  # Singapore postal codes
            r'\d{6}',     # Standalone postal codes
            r'CRESCENT|AVENUE|ROAD|STREET|DRIVE|LANE|PLACE',  # Street types
            r'^[A-Z]+\s+(?:CRESCENT|AVENUE|ROAD|STREET)'  # Street names
        ]

        is_likely_address = any(re.search(pattern, text, re.IGNORECASE) for pattern in address_indicators)

        if not is_likely_address:
            # If no clear address indicators, be more cautious
            if name_score >= 0.4:
                logger.debug(f"Ambiguous text, preferring name detection: '{text}' (name_score: {name_score:.3f})")
                return None

        # Context analysis: check if this appears near other address components
        address_context_score = self._calculate_address_context_score(block, all_blocks)

        if address_context_score < 0.3:
            logger.debug(f"Low address context score: '{text}' (context: {address_context_score:.3f})")
            return None

        # Calculate final confidence
        base_confidence = 0.6
        if is_likely_address:
            base_confidence += 0.2

        final_confidence = min(base_confidence * address_context_score, 1.0)

        # Determine appropriate address field type
        if re.search(r'BLK\s+\d+', text, re.IGNORECASE):
            field_name = 'address'
        elif re.search(r'#\d+-\d+', text):
            field_name = 'address_unit'
        elif re.search(r'SINGAPORE\s+\d{6}', text):
            field_name = 'address_postal'
        else:
            field_name = 'address'

        return DetectedKey(
            key_name=field_name,
            key_text=text,
            confidence=final_confidence,
            geometry=block.geometry,
            value_candidate=text,
            value_confidence=final_confidence,
            document_type=DocumentType.BANK_STATEMENT
        )

    def _calculate_address_context_score(self, block: ContentBlock, all_blocks: List[ContentBlock]) -> float:
        """
        Calculate address context score based on proximity to other address components.

        Args:
            block: Current block to evaluate
            all_blocks: All content blocks for spatial analysis

        Returns:
            Context score from 0.0 to 1.0
        """
        from app.core.logger import get_logger
        logger = get_logger()

        if not all_blocks:
            return 0.5

        block_y = block.geometry.get('y1', 0)
        nearby_address_blocks = 0
        total_nearby_blocks = 0

        # Look for blocks within reasonable vertical proximity
        max_y_distance = 0.15  # Maximum vertical distance for address grouping

        for other_block in all_blocks:
            if other_block == block:
                continue

            other_y = other_block.geometry.get('y1', 0)
            y_distance = abs(block_y - other_y)

            if y_distance <= max_y_distance:
                total_nearby_blocks += 1

                # Check if other block is address-related
                other_text = other_block.text.strip()
                address_patterns = [
                    r'BLK\s+\d+', r'#\d+-\d+', r'SINGAPORE\s+\d{6}', r'\d{6}',
                    r'CRESCENT|AVENUE|ROAD|STREET|DRIVE|LANE'
                ]

                if any(re.search(pattern, other_text, re.IGNORECASE) for pattern in address_patterns):
                    nearby_address_blocks += 1

        if total_nearby_blocks == 0:
            return 0.5  # Neutral score

        context_score = nearby_address_blocks / total_nearby_blocks
        logger.debug(f"Address context for '{block.text[:30]}...': {nearby_address_blocks}/{total_nearby_blocks} = {context_score:.3f}")

        return context_score

    def _is_likely_not_name(self, text: str) -> bool:
        """
        Enhanced filtering to identify text that is definitely not a person's name.
        This is more comprehensive than the basic version and handles bank statement specifics.

        Args:
            text: Text to evaluate

        Returns:
            True if text is definitely not a name, False otherwise
        """
        if not text:
            return True

        text = text.strip()
        text_upper = text.upper()

        # Check for company name patterns first (these should always be treated as names)
        company_patterns = [
            r'(?:PTE\.?\s*LTD\.?|LTD\.?|INC\.?|CORP\.?|SDN\.?\s*BHD\.?|PRIVATE\s*LIMITED|LIMITED|LLC|LP|ENTERPRISE|TRADING|SOLUTIONS|TECHNOLOGIES|CONSULTANCY|SERVICES|HOLDINGS|GROUP|INTERNATIONAL)$',
            r'(?:CO|COMPANY|CORP|CORPORATION|ENTERPRISES|TRADING|GROUP|HOLDINGS)$'
        ]

        for pattern in company_patterns:
            if re.search(pattern, text_upper):
                return False  # This is a valid company name

        # Check for cultural name patterns first (these should always be treated as names)
        if re.search(r'\b(?:S/O|D/O|B/O)\b', text_upper):
            # Validate it's a proper cultural name format (Name S/O Name)
            cultural_pattern = r'^[A-Z][a-zA-Z\'\-\.]+(?:\s+[A-Z][a-zA-Z\'\-\.]+)*\s+(?:S/O|D/O|B/O)\s+[A-Z][a-zA-Z\'\-\.]+(?:\s+[A-Z][a-zA-Z\'\-\.]+)*$'
            if re.match(cultural_pattern, text):
                return False  # This is a valid cultural name

        # Contains numbers (strong indicator of non-name), except in cultural name formats
        if re.search(r'\d', text) and not re.search(r'S/O|D/O|B/O', text):
            return True

        # Special characters: be more nuanced about what constitutes invalid patterns
        # Count dots to distinguish between reasonable use (initials) vs excessive use (OCR errors)
        dot_count = text.count('.')
        if dot_count > 3:  # More than 3 dots suggests OCR error or formatted text, not a name
            return True

        # Reject specific problematic character combinations
        if re.search(r'[^\w\s\-\'\.\/&]', text) and not re.search(r'S/O|D/O|B/O', text):
            # Allow reasonable special characters but reject obviously problematic ones
            problematic_chars = r'[<>"{}|\\^`~\[\]()*+!?=#%@]'
            if re.search(problematic_chars, text):
                return True

        # Bank statement specific keywords and patterns
        bank_statement_keywords = [
            'STATEMENT', 'ACCOUNT', 'PAGE', 'BALANCE', 'DEPOSIT', 'WITHDRAWAL',
            'TRANSACTION', 'SUMMARY', 'TOTAL', 'BANK', 'CONSOLIDATED', 'SAVINGS',
            'CURRENT', 'CURRENCY', 'EQUIVALENT', 'BREAKDOWN', 'SUMMARY', 'CLOSING',
            'OPENING', 'DEBIT', 'CREDIT', 'TRANSFER', 'PAYMENT', 'CHARGE', 'FEE',
            'INTEREST', 'TAX', 'GST', 'VAT', 'PAGE', 'CHAPTER', 'SECTION'
        ]

        for keyword in bank_statement_keywords:
            if keyword in text_upper:
                return True

        # Single letters or very short abbreviations
        if len(text) <= 3 and not text.count(' ') >= 1:
            return True

        # All caps abbreviations (usually not names unless well-known)
        if re.match(r'^[A-Z]{3,}$', text) and text not in ['POSB', 'DBS', 'OCBC', 'UOB']:  # Keep known bank names
            return True

        # Currency codes
        if re.match(r'^[A-Z]{3}$', text) and text in ['SGD', 'USD', 'EUR', 'GBP', 'JPY']:
            return True

        # Patterns that look like account numbers, IDs, or codes
        if re.match(r'^[A-Z]{2,4}\d{6,12}$', text):  # Account numbers
            return True
        if re.match(r'^[A-Z]+\s*[:\-]\s*[A-Z0-9]+$', text):  # Code patterns
            return True

        # Generic financial terms
        financial_terms = [
            'AS AT', 'PERIOD', 'FOR THE PERIOD', 'YEAR TO DATE', 'MONTHLY',
            'QUARTERLY', 'ANNUAL', 'DAILY', 'WEEKLY'
        ]
        for term in financial_terms:
            if term in text_upper:
                return True

        # Very short single words (unlikely to be full names)
        if len(text.split()) == 1 and len(text) < 4:
            return True

        # Lines with excessive punctuation or symbols
        if text.count('-') > len(text) * 0.3:  # More than 30% dashes
            return True

        # Patterns like "XXXX - - - -" (redacted info)
        if re.match(r'^X+\s*[\-\s]*X*$', text_upper):
            return True

        return False
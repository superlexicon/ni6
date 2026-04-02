"""
Key field detector for real-time key recognition during OCR processing.
This module identifies keys in text as they are being processed by doctr.
"""

from typing import Dict, List, Tuple, Optional
import re
from dataclasses import dataclass

from ..key_injection.key_config import key_config, DocumentType, KeyPattern




@dataclass
class KeyMatch:
    """Represents a detected key match with its metadata."""
    key_name: str
    matched_text: str
    confidence: float
    start_pos: int
    end_pos: int
    pattern_used: KeyPattern
    document_type: DocumentType


@dataclass
class DetectedKey:
    """Represents a fully detected key with potential value."""
    key_name: str
    key_text: str
    confidence: float
    geometry: Dict[str, float]
    value_candidate: Optional[str] = None
    value_confidence: float = 0.0
    document_type: DocumentType = DocumentType.PASSPORT


class KeyFieldDetector:
    """Detects key fields in OCR text during processing."""

    def __init__(self):
        self.known_keys = {}
        self._compile_all_patterns()

    def _compile_all_patterns(self):
        """Pre-compile all patterns for all document types."""
        for doc_type in DocumentType:
            self.known_keys[doc_type] = key_config.get_all_keys(doc_type)

    def detect_keys_in_text(self, text: str, document_type: DocumentType) -> List[KeyMatch]:
        """
        Detect key fields in a given text for a specific document type.

        Args:
            text: Text to search for keys
            document_type: Type of document being processed

        Returns:
            List of KeyMatch objects for each detected key
        """
        from app.core.logger import get_logger
        logger = get_logger()

        if document_type not in self.known_keys:
            logger.warning(f"No key patterns found for document type: {document_type}")
            return []

        key_patterns = self.known_keys[document_type]
        matches = []

        logger.debug(f"Testing {len(key_patterns)} patterns against text: '{text[:50]}...'")

        # Check each key pattern against the text
        for key_name, key_pattern in key_patterns.items():
            match = self._check_pattern_match(text, key_name, key_pattern, document_type)
            if match:
                # Special validation for full_name to prevent false positives
                if key_name == 'full_name':
                    if self._validate_full_name_content(match.matched_text):
                        matches.append(match)
                        logger.debug(f"✓ Pattern MATCH: '{key_name}' (conf: {match.confidence:.3f}) - {key_pattern.description}")
                        logger.debug(f"✓ Full name validation passed: '{match.matched_text}'")
                    else:
                        logger.debug(f"✗ Pattern MATCH REJECTED: '{key_name}' - failed validation: '{match.matched_text}'")
                else:
                    matches.append(match)
                    logger.debug(f"✓ Pattern MATCH: '{key_name}' (conf: {match.confidence:.3f}) - {key_pattern.description}")
            else:
                logger.debug(f"✗ Pattern NO MATCH: '{key_name}' - {key_pattern.description}")

        if matches:
            logger.info(f"Found {len(matches)} key matches in text: '{text[:30]}...'")
            for i, match in enumerate(matches):
                logger.info(f"  Match {i+1}: {match.key_name}: '{match.matched_text}' (conf: {match.confidence:.3f})")

        # Sort by confidence score
        matches.sort(key=lambda x: x.confidence, reverse=True)
        return matches

    def _check_pattern_match(self, text: str, key_name: str, key_pattern: KeyPattern,
                            document_type: DocumentType) -> Optional[KeyMatch]:
        """
        Check if a specific key pattern matches the text.

        Args:
            text: Text to check
            key_name: Name of the key
            key_pattern: Pattern to check
            document_type: Document type context

        Returns:
            KeyMatch if pattern matches, None otherwise
        """
        try:
            match = key_pattern.pattern.search(text)
            if match:
                # Calculate confidence based on pattern specificity and match position
                base_confidence = key_pattern.confidence_weight

                # Boost confidence for exact matches at beginning of line
                if match.start() == 0:
                    base_confidence *= 1.1

                # Boost confidence for patterns with capture groups
                if match.groups():
                    base_confidence *= 1.05

                # Cap confidence at 1.0
                confidence = min(base_confidence, 1.0)

                return KeyMatch(
                    key_name=key_name,
                    matched_text=match.group(0),
                    confidence=confidence,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    pattern_used=key_pattern,
                    document_type=document_type
                )
        except re.error:
            # Pattern compilation error - skip this pattern
            pass

        return None

    def extract_key_value_pairs(self, text_lines: List[str], document_type: DocumentType) -> List[DetectedKey]:
        """
        Extract key-value pairs from a list of text lines.

        Args:
            text_lines: List of text lines from OCR
            document_type: Type of document being processed

        Returns:
            List of DetectedKey objects with potential values
        """
        from app.core.logger import get_logger
        logger = get_logger()

        logger.info(f"=== KEY DETECTION START ===")
        logger.info(f"Analyzing {len(text_lines)} text lines for {document_type.value}")

        detected_keys = []
        keys_per_line = {}

        for i, line in enumerate(text_lines):
            # Detect keys in this line
            key_matches = self.detect_keys_in_text(line, document_type)
            keys_per_line[i] = len(key_matches)

            for match in key_matches:
                # Try to extract the value from the same line
                value_candidate, value_confidence = self._extract_value_from_line(
                    line, match, document_type
                )

                detected_key = DetectedKey(
                    key_name=match.key_name,
                    key_text=match.matched_text,
                    confidence=match.confidence,
                    geometry={},  # Will be filled by spatial analyzer
                    value_candidate=value_candidate,
                    value_confidence=value_confidence,
                    document_type=match.document_type
                )

                detected_keys.append(detected_key)
                logger.info(f"Line {i+1}: Detected key '{match.key_name}' with value '{value_candidate}' "
                           f"(key_conf: {match.confidence:.3f}, value_conf: {value_confidence:.3f})")
                logger.info(f"  Line text: '{line}'")

        # Summary statistics
        lines_with_keys = sum(1 for count in keys_per_line.values() if count > 0)
        logger.info(f"Key detection summary:")
        logger.info(f"  Total lines processed: {len(text_lines)}")
        logger.info(f"  Lines with detected keys: {lines_with_keys}")
        logger.info(f"  Total keys detected: {len(detected_keys)}")
        logger.info(f"  Keys per line: {dict(keys_per_line)}")
        logger.info(f"=== KEY DETECTION END ===")

        return detected_keys

    def _extract_value_from_line(self, line: str, key_match: KeyMatch,
                                document_type: DocumentType) -> Tuple[Optional[str], float]:
        """
        Extract value from the same line as a detected key.

        Args:
            line: Text line containing the key
            key_match: The detected key match
            document_type: Document type for context

        Returns:
            Tuple of (extracted_value, confidence_score)
        """
        from app.core.logger import get_logger
        logger = get_logger()

        key_pattern = key_match.pattern_used

        # FIRST: Try to get value from regex capture groups (highest priority)
        try:
            full_match = key_pattern.pattern.search(line)
            if full_match and full_match.groups():
                # Take the last capture group as value
                value = full_match.groups()[-1].strip()
                logger.debug(f"Extracted value from capture group: '{value}' for pattern {key_pattern.pattern}")
                if value and self._validate_value_format(value, key_pattern):
                    logger.debug(f"Value format validated successfully: '{value}'")
                    return value, 0.9
                else:
                    logger.debug(f"Value format validation failed for: '{value}'")
        except (re.error, IndexError) as e:
            logger.debug(f"Capture group extraction failed: {e}")

        # SECOND: Try to get value from text after the matched key
        if key_match.end_pos < len(line):
            remaining_text = line[key_match.end_pos:].strip()
            logger.debug(f"Trying to extract from remaining text: '{remaining_text}'")

            # Fallback: extract text after the key
            if remaining_text and len(remaining_text) > 0:
                confidence = 0.7  # Lower confidence for fallback extraction

                # Boost confidence if value matches expected format
                if self._validate_value_format(remaining_text, key_pattern):
                    confidence = 0.85
                    logger.debug(f"Remaining text format validated: '{remaining_text}' (conf: {confidence})")

                return remaining_text, confidence

        logger.debug(f"No value extracted from line: '{line}'")
        return None, 0.0

    def _validate_value_format(self, value: str, key_pattern: KeyPattern) -> bool:
        """
        Validate if the extracted value matches the expected format.

        Args:
            value: Extracted value to validate
            key_pattern: Pattern with value format requirements

        Returns:
            True if value matches expected format, False otherwise
        """
        if not key_pattern.value_format:
            return True  # No format restriction

        try:
            return bool(key_pattern.value_format.match(value))
        except re.error:
            return True  # If pattern is invalid, assume valid

    def get_high_confidence_keys(self, detected_keys: List[DetectedKey],
                                min_confidence: float = 0.8) -> List[DetectedKey]:
        """
        Filter detected keys by confidence threshold.

        Args:
            detected_keys: List of detected keys
            min_confidence: Minimum confidence score

        Returns:
            List of high-confidence detected keys
        """
        return [key for key in detected_keys if key.confidence >= min_confidence]

    def remove_duplicate_keys(self, detected_keys: List[DetectedKey]) -> List[DetectedKey]:
        """
        Remove duplicate keys, keeping the highest confidence version.
        Enhanced to prioritize company_name over full_name for business accounts.

        Args:
            detected_keys: List of detected keys with potential duplicates

        Returns:
            List of unique detected keys
        """
        key_dict = {}

        # Group keys by type and company names
        company_names = []
        personal_names = []

        for key in detected_keys:
            if key.key_name == 'company_name':
                company_names.append(key)
            elif key.key_name == 'full_name':
                personal_names.append(key)

        # If we have company names, remove/deprioritize personal names that are likely from transactions
        if company_names:
            # Keep only the highest confidence company name
            best_company = max(company_names, key=lambda k: k.confidence)
            key_dict['company_name'] = best_company

            # Filter out personal names that are likely from transactions (lower confidence or not in top positions)
            for personal_name in personal_names:
                # Only keep personal names if they have significantly higher confidence than company names
                # or if they're in top 30% of document (likely account holder)
                is_top_position = self._is_top_30_percent(personal_name.geometry)
                has_much_higher_confidence = personal_name.confidence > best_company.confidence * 1.1

                if is_top_position and has_much_higher_confidence:
                    key_dict['full_name'] = personal_name
                else:
                    # Remove this personal name as it's likely from transactions
                    pass

        # Process all other keys normally
        for key in detected_keys:
            if key.key_name not in ['company_name', 'full_name']:
                if key.key_name not in key_dict or key.confidence > key_dict[key.key_name].confidence:
                    key_dict[key.key_name] = key

        # If no company names were found, process full_name normally
        if not company_names:
            for key in personal_names:
                if 'full_name' not in key_dict or key.confidence > key_dict['full_name'].confidence:
                    key_dict['full_name'] = key

        return list(key_dict.values())

    def _is_top_30_percent(self, geometry: Dict[str, float]) -> bool:
        """Check if a text element is in the top 30% of the document."""
        if not geometry:
            return False

        y_center = (geometry['y1'] + geometry['y2']) / 2
        return y_center <= 0.3  # Top 30% of page

    def prioritize_required_keys(self, detected_keys: List[DetectedKey],
                               document_type: DocumentType) -> List[DetectedKey]:
        """
        Prioritize required keys over optional keys.

        Args:
            detected_keys: List of detected keys
            document_type: Document type context

        Returns:
            List with required keys first
        """
        required_keys = []
        optional_keys = []

        for key in detected_keys:
            if key_config.is_required_key(document_type, key.key_name):
                required_keys.append(key)
            else:
                optional_keys.append(key)

        # Sort each group by confidence
        required_keys.sort(key=lambda x: x.confidence, reverse=True)
        optional_keys.sort(key=lambda x: x.confidence, reverse=True)

        return required_keys + optional_keys

    def _validate_full_name_content(self, text: str) -> bool:
        """
        Validate if text content is appropriate for a full_name field.
        Enhanced to reject OCR errors while accepting legitimate names and company names.

        Args:
            text: Text content to validate

        Returns:
            True if content is appropriate for a full name
        """
        if not text or len(text.strip()) < 2:
            return False

        text_clean = text.strip()
        text_upper = text_clean.upper()

        # Step 1: Reject OCR error patterns and garbage
        ocr_error_patterns = [
            r'^[A-Z]{5,}$',  # All caps single words (likely OCR errors like "OSINSOTA", "ANBOSESNEM")
            r'^[A-Z]{2,}[A-Z\s]{3,}$',  # Mixed capitalization errors
            r'^[A-Z][A-Z]{2,}[A-Z\s]+$',  # Repeated characters with spaces
            r'^[A-Z]{8,}$',  # Very long all-caps sequences (OCR garbage)
            r'^[A-Z]{1,2}\s+[A-Z]{1,2}$',  # Short fragmented caps
            r'^[A-Z]{3,}\s+[A-Z]{3,}$',  # Medium fragmented caps
            r'^(?:DOCBC|OSINSOTA|ANBOSESNEM)$',  # Known OCR errors from logs
        ]

        for pattern in ocr_error_patterns:
            if re.match(pattern, text_clean):
                return False

        # Step 2: Reject obvious non-name content
        invalid_patterns = [
            r'STATEMENT', r'ACCOUNT', r'BALANCE', r'DEPOSIT', r'WITHDRAWAL',
            r'TRANSACTION', r'SUMMARY', r'TOTAL', r'BANK', r'CONSOLIDATED',
            r'CURRENT', r'SAVINGS', r'CURRENCY', r'EQUIVALENT', r'BREAKDOWN',
            r'CLOSING', r'OPENING', r'DEBIT', r'CREDIT', r'TRANSFER',
            r'PAYMENT', r'CHARGE', r'FEE', r'INTEREST', r'TAX', r'PAGE',
            r'CHAPTER', r'SECTION', r'POSB', r'DBS', r'OCBC', r'UOB',
            r'INFORMATION', r'WPI', r'DOCBC',  # Add specific OCR errors
            # Reject excessive special characters
            r'-{2,}', r'_{2,}', r'\.{2,}'
        ]

        for pattern in invalid_patterns:
            if re.search(pattern, text_upper):
                return False

        # Step 3: Accept if it matches legitimate name patterns
        # Support personal names, cultural names, and company names
        name_patterns = [
            # Company names (PTE. LTD., LTD., INC., etc.) - Enhanced with more business suffixes
            r'^[A-Z][A-Za-z\s&\.\-\']+\s+(?:PTE\.?\s*LTD\.?|LTD\.?|INC\.?|CORP\.?|SDN\.?\s*BHD\.?|PRIVATE\s*LIMITED|LIMITED|LLC|LP|ENTERPRISE|TRADING|SOLUTIONS|TECHNOLOGIES|CONSULTANCY|SERVICES|HOLDINGS|GROUP|INTERNATIONAL)$',
            r'^[A-Z][A-Za-z\s&\.\-\']+(?:\s+(?:PTE\.?\s*LTD\.?|LTD\.?|INC\.?|CORP\.?|SDN\.?\s*BHD\.?|PRIVATE\s*LIMITED|LIMITED|LLC))?$',

            # Business names with activity indicators
            r'^[A-Z][A-Za-z\s&\.\-\']+(?:\s+(?:TRADING|SOLUTIONS|TECHNOLOGIES|CONSULTANCY|SERVICES|HOLDINGS|GROUP|INTERNATIONAL|GLOBAL|ASIA|PACIFIC|SINGAPORE))$',

            # Cultural names with S/O, D/O, B/O
            r'^[A-Z][A-Za-z\'\-]+\s+(?:S/O|D/O|B/O)\s+[A-Z][A-Za-z\'\-\.]+(?:\s+[A-Z][A-Za-z\'\-\.]+)*$',

            # Standard personal names (2-4 words, reasonable capitalization)
            r'^[A-Z][A-Za-z\'\-]+(?:\s+[A-Z][A-Za-z\'\-]+){1,3}$',

            # Names with mixed case but reasonable structure (accept some OCR imperfection)
            r'^[A-Za-z][A-Za-z\'\-]*(?:\s+[A-Za-z][A-Za-z\'\-]*){1,4}$',
        ]

        # Check if any pattern matches
        if any(re.match(pattern, text_clean) for pattern in name_patterns):
            return True

        # Step 4: Additional validation for edge cases
        # Accept names that have at least some lowercase letters (indicates real name vs OCR error)
        if re.search(r'[a-z]', text_clean) and len(text_clean.split()) >= 2:
            # Must have proper word structure (not all random caps)
            words = text_clean.split()
            valid_words = sum(1 for word in words if len(word) >= 2)
            return valid_words >= len(words) * 0.7  # At least 70% of words are valid length

        return False
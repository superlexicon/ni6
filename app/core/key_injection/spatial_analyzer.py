"""
Spatial analyzer for geometry-based key-value relationship mapping.
This module uses the geometry information from doctr to find values near detected keys.
"""

import re
from typing import Dict, List, Tuple, Optional
import math

from .key_field_detector import DetectedKey, KeyFieldDetector
from .business_rules_analyzer import BusinessRulesAnalyzer
from .data_structures import TextElement, SpatialRelationship
from .key_config import key_config, DocumentType


class SpatialAnalyzer:
    """Analyzes spatial relationships between keys and values using geometry data."""

    def __init__(self, detector: KeyFieldDetector):
        self.detector = detector
        self.business_rules_analyzer = BusinessRulesAnalyzer()
        self.spatial_weights = {
            'horizontal_right': 1.0,  # Value to the right of key (same line)
            'vertical_below': 0.9,    # Value below key (next line)
            'horizontal_left': 0.7,   # Value to the left of key
            'vertical_above': 0.6,    # Value above key
            'diagonal': 0.4           # Diagonal relationship
        }

    def analyze_spatial_relationships(self, text_elements: List[TextElement],
                                    detected_keys: List[DetectedKey],
                                    document_type: DocumentType) -> List[DetectedKey]:
        """
        Analyze spatial relationships and enhance detected keys with value candidates.

        Args:
            text_elements: List of all text elements with geometry
            detected_keys: List of keys detected by text matching
            document_type: Type of document being processed

        Returns:
            Enhanced detected keys with spatially-derived values
        """
        # Mark text elements that are keys
        self._mark_key_elements(text_elements, detected_keys)

        enhanced_keys = []

        for detected_key in detected_keys:
            # Find the key element in text_elements
            key_element = self._find_key_element(text_elements, detected_key)
            if not key_element:
                continue

            # Find potential value elements based on spatial analysis
            value_candidates = self._find_value_candidates(key_element, text_elements)

            # Select the best value candidate
            best_value, best_confidence = self._select_best_value(
                key_element, value_candidates, detected_key, document_type
            )

            # Update the detected key with spatial information
            detected_key.geometry = key_element.geometry
            detected_key.value_candidate = best_value
            detected_key.value_confidence = best_confidence

            enhanced_keys.append(detected_key)

        return enhanced_keys

    def _mark_key_elements(self, text_elements: List[TextElement], detected_keys: List[DetectedKey]):
        """Mark text elements that contain detected keys."""
        for element in text_elements:
            for detected_key in detected_keys:
                if self._text_contains_key(element.text, detected_key):
                    element.is_key = True
                    element.key_name = detected_key.key_name
                    break

    def _find_key_element(self, text_elements: List[TextElement], detected_key: DetectedKey) -> Optional[TextElement]:
        """Find the text element corresponding to a detected key."""
        for element in text_elements:
            if (element.is_key and element.key_name == detected_key.key_name and
                detected_key.key_text.lower() in element.text.lower()):
                return element
        return None

    def _find_value_candidates(self, key_element: TextElement,
                              text_elements: List[TextElement]) -> List[Tuple[TextElement, float]]:
        """Find potential value elements based on spatial relationships."""
        candidates = []

        key_center_x = (key_element.geometry['x1'] + key_element.geometry['x2']) / 2
        key_center_y = (key_element.geometry['y1'] + key_element.geometry['y2']) / 2

        for element in text_elements:
            if element.is_key or element == key_element:
                continue  # Skip keys and self

            # Calculate spatial relationship
            relationship, confidence = self._calculate_spatial_relationship(
                key_element, element, key_center_x, key_center_y
            )

            if confidence > 0.3:  # Minimum spatial confidence threshold
                candidates.append((element, confidence))

        # Sort by confidence
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates

    def _calculate_spatial_relationship(self, key_element: TextElement, value_element: TextElement,
                                      key_center_x: float, key_center_y: float) -> Tuple[str, float]:
        """
        Calculate the spatial relationship type and confidence between key and value elements.

        Returns:
            Tuple of (relationship_type, confidence_score)
        """
        value_center_x = (value_element.geometry['x1'] + value_element.geometry['x2']) / 2
        value_center_y = (value_element.geometry['y1'] + value_element.geometry['y2']) / 2

        # Calculate distances and directional differences
        x_diff = value_center_x - key_center_x
        y_diff = value_center_y - key_center_y
        distance = math.sqrt(x_diff**2 + y_diff**2)

        # Determine relationship type based on relative positions
        if abs(y_diff) < 0.05:  # Same line (vertical proximity)
            if x_diff > 0:
                return ('horizontal_right', self._calculate_distance_confidence(distance))
            else:
                return ('horizontal_left', self._calculate_distance_confidence(distance) * 0.8)
        elif abs(x_diff) < 0.1:  # Same column (horizontal proximity)
            if y_diff > 0:
                return ('vertical_below', self._calculate_distance_confidence(distance) * 0.95)
            else:
                return ('vertical_above', self._calculate_distance_confidence(distance) * 0.7)
        else:
            return ('diagonal', self._calculate_distance_confidence(distance) * 0.5)

    def _calculate_distance_confidence(self, distance: float) -> float:
        """Calculate confidence based on spatial distance."""
        # Distance is normalized (0-1), so we want smaller distances to have higher confidence
        max_expected_distance = 0.3  # Maximum expected distance for key-value pairs
        if distance > max_expected_distance:
            return 0.1

        # Linear decay of confidence with distance
        return 1.0 - (distance / max_expected_distance)

    def _select_best_value(self, key_element: TextElement, value_candidates: List[Tuple[TextElement, float]],
                         detected_key: DetectedKey, document_type: DocumentType) -> Tuple[Optional[str], float]:
        """
        Select the best value candidate from spatial analysis with enhanced logic.

        Returns:
            Tuple of (best_value_text, combined_confidence)
        """
        from app.core.logger import get_logger
        logger = get_logger()

        # Enhanced handling for self-contained patterns that already capture the complete value
        self_contained_patterns = [
            'account_name', 'business_name', 'account_holder', 'bank_name_singapore', 'serial_number',
            'address', 'address_unit', 'address_postal', 'account_number'
        ]

        # Special handling for account numbers to preserve high-confidence numeric values
        if detected_key.key_name == 'account_number':
            # If the account number already has a high-confidence numeric value, preserve it
            if (detected_key.value_candidate and
                detected_key.value_confidence >= 0.95 and
                detected_key.value_candidate.isdigit() and
                len(detected_key.value_candidate) >= 8):  # Valid account number length
                logger.debug(f"Preserving high-confidence account number: '{detected_key.value_candidate}' (conf: {detected_key.value_confidence})")
                return detected_key.value_candidate, detected_key.value_confidence

        if detected_key.key_name in self_contained_patterns:
            logger.debug(f"Self-contained pattern detected: {detected_key.key_name}")

            # For self-contained patterns, validate that the detected text is appropriate
            key_pattern = key_config.get_key_pattern(document_type, detected_key.key_name)
            detected_text = detected_key.key_text.strip()

            # Unified validation for account_name patterns to prevent false associations
            if detected_key.key_name == 'account_name':
                if self._validate_account_name_content(detected_text):
                    logger.debug(f"Valid account_name content: '{detected_text}'")
                    # Extract from capture groups if available
                    if key_pattern:
                        match = key_pattern.pattern.search(detected_key.key_text)
                        if match and match.groups():
                            value = match.groups()[-1].strip()
                            if value:
                                logger.debug(f"Extracted account name from pattern: '{value}'")
                                return value, detected_key.confidence * 0.95
                    return detected_text, detected_key.confidence * 0.9
                else:
                    logger.debug(f"Invalid account_name content: '{detected_text}'")
                    return None, 0.0  # Don't return invalid account names

            # Additional validation for address patterns
            if detected_key.key_name.startswith('address'):
                if self._validate_address_content(detected_text, detected_key.key_name):
                    logger.debug(f"Valid address content for {detected_key.key_name}: '{detected_text}'")
                    return detected_text, detected_key.confidence * 0.9
                else:
                    logger.debug(f"Invalid address content for {detected_key.key_name}: '{detected_text}'")
                    # Try to find better value from candidates
                    return self._find_best_address_value(value_candidates, detected_key, logger)

            # For other self-contained patterns
            if key_pattern:
                # Try to extract from capture groups first
                match = key_pattern.pattern.search(detected_key.key_text)
                if match and match.groups():
                    # Use the last capture group as the value
                    value = match.groups()[-1].strip()
                    if value:
                        logger.debug(f"Extracted value from pattern: '{value}'")
                        return value, detected_key.confidence * 0.95
                else:
                    # Use the entire matched text as value
                    value = detected_key.key_text.strip()
                    if value:
                        logger.debug(f"Using full matched text as value: '{value}'")
                        return value, detected_key.confidence * 0.9
            else:
                # Fallback: use the detected key text as value
                value = detected_key.key_text.strip()
                if value:
                    logger.debug(f"Using key text as value: '{value}'")
                    return value, detected_key.confidence * 0.85

        # Original logic for patterns that need separate values
        if not value_candidates:
            return None, 0.0

        best_candidate = None
        best_combined_confidence = 0.0

        for value_element, spatial_confidence in value_candidates:
            # Get format validation confidence
            format_confidence = self._validate_value_format(
                value_element.text, detected_key, document_type
            )

            # Combine confidences
            combined_confidence = self._combine_confidences(
                detected_key.confidence,  # Key detection confidence
                spatial_confidence,       # Spatial relationship confidence
                format_confidence         # Format validation confidence
            )

            if combined_confidence > best_combined_confidence:
                best_combined_confidence = combined_confidence
                best_candidate = value_element

        if best_candidate:
            logger.debug(f"Selected spatial value: '{best_candidate.text}' (confidence: {best_combined_confidence:.3f})")
            return best_candidate.text, best_combined_confidence
        return None, 0.0

    def _validate_address_content(self, text: str, address_type: str) -> bool:
        """
        Validate if text content is appropriate for the given address type.

        Args:
            text: Text content to validate
            address_type: Type of address field ('address', 'address_unit', 'address_postal')

        Returns:
            True if content is appropriate for the address type
        """
        text = text.strip().upper()

        if address_type == 'address_unit':
            return bool(re.match(r'^#\d+-\d+$', text))

        elif address_type == 'address_postal':
            return bool(re.match(r'^SINGAPORE\s+\d{6}$', text))

        elif address_type == 'address':
            # Should contain street patterns or block numbers, not just names
            address_indicators = [
                r'BLK\s+\d+',
                r'CRESCENT|AVENUE|ROAD|STREET|DRIVE|LANE|PLACE',
                r'SINGAPORE',  # If it's a full address including postal
            ]
            return any(re.search(pattern, text) for pattern in address_indicators)

        return True  # Default for other address types

    def _validate_account_name_content(self, text: str) -> bool:
        """
        Unified validation for account_name content.
        Handles both personal names and company names while rejecting
        addresses, OCR garbage, and banking terminology.

        Args:
            text: Text content to validate

        Returns:
            True if content is appropriate for an account name field
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

        # Step 2: Reject banking terminology and document artifacts
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

        # Step 3: Reject address patterns
        address_patterns = [
            r'^\d+\s+',  # Starts with number (street addresses)
            r'BLK\s+\d+',  # Block numbers
            r'#\d+-\d+',  # Unit numbers
            r'(?:ROAD|STREET|AVENUE|CRESCENT|DRIVE|LANE|PLACE|WALK|CLOSE|GROVE|HEIGHTS|HILLS|PARK|TERRACE|VIEW|WAY)$',  # Street names
            r'SINGAPORE\s+\d{6}$',  # Singapore postal codes
            r'\d{6}$',  # Just postal codes
        ]

        for pattern in address_patterns:
            if re.search(pattern, text_upper):
                return False

        # Step 4: Accept legitimate account name patterns (personal + company)
        account_name_patterns = [
            # Company names (PTE. LTD., LTD., INC., etc.)
            r'^[A-Z][A-Za-z\s&\.\-\']+\s+(?:PTE\.?\s*LTD\.?|LTD\.?|INC\.?|CORP\.?|SDN\.?\s*BHD\.?)$',
            r'^[A-Z][A-Za-z\s&\.\-\']+(?:\s+(?:PTE\.?\s*LTD\.?|LTD\.?|INC\.?))?$',

            # Cultural names with S/O, D/O, B/O
            r'^[A-Z][A-Za-z\'\-]+\s+(?:S/O|D/O|B/O)\s+[A-Z][A-Za-z\'\-\.]+(?:\s+[A-Z][A-Za-z\'\-\.]+)*$',

            # Standard personal names (2-4 words, reasonable capitalization)
            r'^[A-Z][A-Za-z\'\-]+(?:\s+[A-Z][A-Za-z\'\-]+){1,3}$',

            # Names with mixed case but reasonable structure (accept some OCR imperfection)
            r'^[A-Za-z][A-Za-z\'\-]*(?:\s+[A-Za-z][A-Za-z\'\-]*){1,4}$',
        ]

        # Check if any pattern matches
        if any(re.match(pattern, text_clean) for pattern in account_name_patterns):
            return True

        # Step 5: Additional company name patterns (business suffixes)
        company_patterns = [
            # Singapore/Malaysian company formats (PTE. LTD., etc.)
            r'(?:PTE\.?\s*LTD\.?|LTD\.?|INC\.?|CORP\.?|SDN\.?\s*BHD\.?|PRIVATE\s*LIMITED|LIMITED|LLC|LP|ENTERPRISE|TRADING|SOLUTIONS|TECHNOLOGIES|CONSULTANCY|SERVICES|HOLDINGS|GROUP|INTERNATIONAL)$',
            # General company names with business suffixes
            r'(?:CO|COMPANY|CORP|CORPORATION|ENTERPRISES|TRADING|GROUP|HOLDINGS)$',
            # Business activity indicators
            r'(?:TRADING|SOLUTIONS|TECHNOLOGIES|CONSULTANCY|SERVICES|HOLDINGS|GROUP|INTERNATIONAL|GLOBAL|ASIA|PACIFIC|SINGAPORE)$'
        ]

        for pattern in company_patterns:
            if re.search(pattern, text_upper):
                return True

        # Step 6: Multi-word business names (likely legitimate companies)
        if re.search(r'^[A-Z][A-Za-z\s&\.\-\']+\s+[A-Z][A-Za-z\s&\.\-\']+(?:\s+[A-Z][A-Za-z\s&\.\-\']+)?$', text_upper):
            # Additional check: ensure it doesn't look like an address
            if not re.search(r'\d', text_clean):  # No numbers in company names
                return True

        # Step 7: Additional validation for edge cases
        # Accept names that have at least some lowercase letters (indicates real name vs OCR error)
        if re.search(r'[a-z]', text_clean) and len(text_clean.split()) >= 2:
            # Must have proper word structure (not all random caps)
            words = text_clean.split()
            valid_words = sum(1 for word in words if len(word) >= 2)
            return valid_words >= len(words) * 0.7  # At least 70% of words are valid length

        return False

    def _find_best_address_value(self, value_candidates: List[Tuple[TextElement, float]],
                                detected_key: DetectedKey, logger) -> Tuple[Optional[str], float]:
        """
        Find the best address value from candidates when the detected pattern is invalid.

        Args:
            value_candidates: List of candidate values with their confidences
            detected_key: The detected key information
            logger: Logger instance

        Returns:
            Tuple of (best_value, confidence)
        """
        if not value_candidates:
            return None, 0.0

        # Sort by spatial confidence
        value_candidates.sort(key=lambda x: x[1], reverse=True)

        for value_element, spatial_confidence in value_candidates:
            candidate_text = value_element.text.strip()

            # Check if this candidate is appropriate for the address type
            if self._validate_address_content(candidate_text, detected_key.key_name):
                combined_confidence = spatial_confidence * 0.8  # Slightly lower for fallback
                logger.debug(f"Found better address value for {detected_key.key_name}: '{candidate_text}' (confidence: {combined_confidence:.3f})")
                return candidate_text, combined_confidence

        # If no valid candidate found, return the best candidate with a warning
        best_candidate, best_confidence = value_candidates[0]
        logger.debug(f"No valid address value found, using best candidate: '{best_candidate.text}' (confidence: {best_confidence * 0.5:.3f})")
        return best_candidate.text, best_confidence * 0.5

    def _validate_value_format(self, value_text: str, detected_key: DetectedKey,
                             document_type: DocumentType) -> float:
        """
        Validate if the value text matches expected format for the key.

        Returns:
            Format validation confidence (0.0 to 1.0)
        """
        key_pattern = key_config.get_key_pattern(document_type, detected_key.key_name)
        if not key_pattern or not key_pattern.value_format:
            return 0.8  # Default confidence if no format restriction

        try:
            if key_pattern.value_format.match(value_text.strip()):
                return 1.0
            else:
                return 0.3  # Low confidence if format doesn't match
        except re.error:
            return 0.5  # Medium confidence if regex is invalid

    def _combine_confidences(self, key_confidence: float, spatial_confidence: float,
                           format_confidence: float) -> float:
        """
        Combine multiple confidence scores into a single value.

        Uses weighted averaging with different weights for different confidence types.
        """
        weights = {
            'key': 0.4,      # Key detection is most important
            'spatial': 0.4,  # Spatial relationship is also important
            'format': 0.2    # Format validation is supporting evidence
        }

        combined = (
            key_confidence * weights['key'] +
            spatial_confidence * weights['spatial'] +
            format_confidence * weights['format']
        )

        return min(combined, 1.0)  # Cap at 1.0

    def _text_contains_key(self, text: str, detected_key: DetectedKey) -> bool:
        """Check if text contains the detected key (case-insensitive)."""
        return detected_key.key_text.lower() in text.lower()

    def extract_key_value_pairs_from_geometry(self, ocr_results: List[Dict],
                                            document_type: DocumentType) -> List[DetectedKey]:
        """
        Main method to extract key-value pairs from OCR geometry data using enhanced
        business rules analysis combined with traditional key-based detection.

        Args:
            ocr_results: List of OCR results with geometry from doctr
                        [{'text': 'text', 'x1': 0.1, 'y1': 0.2, 'x2': 0.3, 'y2': 0.4, 'confidence': 0.95}, ...]
            document_type: Type of document being processed

        Returns:
            List of DetectedKey objects with extracted values
        """
        from app.core.logger import get_logger
        logger = get_logger()

        logger.info(f"=== ENHANCED SPATIAL ANALYSIS START ===")
        logger.info(f"Processing {len(ocr_results)} OCR results for {document_type.value}")

        # Convert OCR results to TextElement objects
        text_elements = []
        for result in ocr_results:
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
        logger.info(f"Text lines sample: {text_elements[:5]}")  # Show first 5 lines

        # Step 1: Traditional key-based detection
        logger.info("=== STEP 1: TRADITIONAL KEY DETECTION ===")
        text_lines = [element.text for element in text_elements]
        pattern_detected_keys = self.detector.extract_key_value_pairs(text_lines, document_type)
        logger.info(f"Pattern-based key detection found {len(pattern_detected_keys)} keys:")
        for i, key in enumerate(pattern_detected_keys):
            logger.info(f"  Pattern Key {i+1}: '{key.key_name}' -> '{key.value_candidate}' "
                       f"(confidence: {key.confidence:.3f})")

        # Step 2: Business rules-based detection for contextual information
        logger.info("=== STEP 2: BUSINESS RULES ANALYSIS ===")
        business_rules_keys = self.business_rules_analyzer.extract_key_value_pairs_from_geometry(
            ocr_results, document_type
        )
        logger.info(f"Business rules analysis found {len(business_rules_keys)} keys:")
        for i, key in enumerate(business_rules_keys):
            logger.info(f"  Business Rules Key {i+1}: '{key.key_name}' -> '{key.value_candidate}' "
                       f"(confidence: {key.confidence:.3f})")

        # Step 3: Combine results from both approaches
        logger.info("=== STEP 3: COMBINING RESULTS ===")
        all_detected_keys = pattern_detected_keys + business_rules_keys

        # Remove duplicates (business rules take precedence for same key names)
        combined_keys = self._merge_detection_results(pattern_detected_keys, business_rules_keys)
        logger.info(f"After merging, we have {len(combined_keys)} unique keys")

        # Step 4: Enhance combined results with spatial analysis
        logger.info("=== STEP 4: SPATIAL ENHANCEMENT ===")
        enhanced_keys = self.analyze_spatial_relationships(text_elements, combined_keys, document_type)
        logger.info(f"Spatial enhancement completed for {len(enhanced_keys)} keys:")
        for i, key in enumerate(enhanced_keys):
            logger.info(f"  Final Key {i+1}: '{key.key_name}' -> '{key.value_candidate}' "
                       f"(key_conf: {key.confidence:.3f}, value_conf: {key.value_confidence:.3f})")

        logger.info(f"=== ENHANCED SPATIAL ANALYSIS END ===")
        return enhanced_keys

    def _merge_detection_results(self, pattern_keys: List[DetectedKey],
                                business_rules_keys: List[DetectedKey]) -> List[DetectedKey]:
        """
        Merge results from pattern-based and business-rules-based detection.
        Numeric account number patterns take highest precedence over business rules.
        """
        import logging
        import re
        logger = logging.getLogger(__name__)

        # Separate keys by type and priority
        numeric_account_keys = []
        other_pattern_keys = []
        business_rules_dict = {}

        # Separate pattern keys by type
        for key in pattern_keys:
            if key.key_name == 'account_number' and key.value_candidate and re.match(r'^\d{8,20}$', key.value_candidate):
                numeric_account_keys.append(key)
                logger.debug(f"Found numeric account pattern: '{key.value_candidate}' (confidence: {key.confidence})")
            else:
                other_pattern_keys.append(key)

        # Separate business rules keys by type
        for key in business_rules_keys:
            business_rules_dict[key.key_name] = key

        merged_keys = []

        # Priority 1: Add numeric account number keys (highest precedence)
        merged_keys.extend(numeric_account_keys)
        logger.debug(f"Added {len(numeric_account_keys)} numeric account keys with highest precedence")

        # Priority 2: Add business rules keys (except account_number if we have numeric one)
        for key_name, key in business_rules_dict.items():
            if key_name != 'account_number' or not numeric_account_keys:
                merged_keys.append(key)
                logger.debug(f"Added business rules key: '{key_name}'")
            else:
                logger.debug(f"Skipped business rules account_number (numeric pattern takes precedence)")

        # Priority 3: Add other pattern keys (non-conflicting)
        for key in other_pattern_keys:
            if key.key_name not in business_rules_dict:
                merged_keys.append(key)
                logger.debug(f"Added pattern key: '{key.key_name}'")
            else:
                logger.debug(f"Skipped pattern key '{key.key_name}' (conflict with business rules)")

        # Sort by confidence (highest first) but keep numeric account keys at top if they exist
        if numeric_account_keys:
            # Keep numeric account keys at top, sort others by confidence
            numeric_keys = [k for k in merged_keys if k.key_name == 'account_number' and re.match(r'^\d+$', k.value_candidate or '')]
            other_keys = [k for k in merged_keys if k not in numeric_keys]
            other_keys.sort(key=lambda k: k.confidence, reverse=True)
            merged_keys = numeric_keys + other_keys
        else:
            # No numeric account keys, sort all by confidence
            merged_keys.sort(key=lambda k: k.confidence, reverse=True)

        logger.debug(f"Merged {len(merged_keys)} keys with precedence: numeric_account > business_rules > patterns")

        return merged_keys
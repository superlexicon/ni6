"""
Key injection manager that coordinates all key injection components.
This module provides the main interface for key injection functionality.
"""

import base64
import os
import hashlib
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .key_config import DocumentType, key_config
from .key_field_detector import KeyFieldDetector, DetectedKey
from .spatial_analyzer import SpatialAnalyzer
from .confidence_calculator import ConfidenceCalculator, EnhancedConfidence


@dataclass
class KeyInjectionResult:
    """Result of key injection analysis for a document."""
    document_type: DocumentType
    detected_keys: List[DetectedKey]
    enhanced_confidences: Dict[str, EnhancedConfidence]
    processing_summary: Dict
    success: bool
    error_message: Optional[str] = None


@dataclass
class ExtractionSummary:
    """Summary of extraction results."""
    total_keys_detected: int
    keys_with_values: int
    high_confidence_extractions: int
    required_keys_found: int
    required_keys_total: int
    overall_document_confidence: float


class KeyInjectionManager:
    """Main manager for key injection functionality."""

    # Encryption version for future compatibility
    ENCRYPTION_VERSION = "nacl_v1"

    def __init__(self):
        self.detector = KeyFieldDetector()
        self.spatial_analyzer = SpatialAnalyzer(self.detector)
        self.confidence_calculator = ConfidenceCalculator()
        self._secret_box = None

    def _get_secret_box(self):
        """
        Get or create the SecretBox for symmetric encryption.
        Uses SERVER_ENCRYPTION_KEY from environment, derives a 32-byte key using SHA-256.
        """
        if self._secret_box is None:
            # Lazy import to avoid startup issues if nacl not installed
            try:
                from nacl.secret import SecretBox
            except ImportError:
                raise ImportError(
                    "PyNaCl is required for encryption. Install with: pip install pynacl"
                )

            # Get encryption key from SEED (reuse for NaCl key derivation)
            from app.core.key.seed_generator import GenerateSeed
            seed = GenerateSeed.get_seed()

            # Derive a 32-byte key using SHA-256 (SecretBox requires exactly 32 bytes)
            key = hashlib.sha256(seed.encode('utf-8')).digest()
            self._secret_box = SecretBox(key)

        return self._secret_box

    def encrypt_data(self, data: str) -> str:
        """
        Encrypt data using NaCl SecretBox (XSalsa20-Poly1305).
        Provides authenticated encryption - tampered ciphertext will fail decryption.

        Args:
            data: Plain text data to encrypt

        Returns:
            Encrypted string in format: "nacl_v1:<base64(nonce + ciphertext)>"
        """
        try:
            box = self._get_secret_box()

            # Convert to bytes
            plaintext = data.encode('utf-8')

            # Encrypt (SecretBox automatically generates a random nonce and prepends it)
            encrypted = box.encrypt(plaintext)

            # Encode as base64 for storage
            encoded = base64.b64encode(encrypted).decode('utf-8')

            return f"{self.ENCRYPTION_VERSION}:{encoded}"

        except Exception as e:
            raise ValueError(f"Encryption failed: {str(e)}")

    def decrypt_data(self, encrypted_data: str) -> str:
        """
        Decrypt data that was encrypted with encrypt_data.

        Args:
            encrypted_data: Encrypted string from encrypt_data

        Returns:
            Decrypted plain text
        """
        try:
            # Handle versioned format
            if encrypted_data.startswith(f"{self.ENCRYPTION_VERSION}:"):
                encoded = encrypted_data[len(self.ENCRYPTION_VERSION) + 1:]
                box = self._get_secret_box()

                # Decode from base64
                encrypted_bytes = base64.b64decode(encoded.encode('utf-8'))

                # Decrypt (SecretBox extracts nonce from the encrypted message)
                decrypted = box.decrypt(encrypted_bytes)

                return decrypted.decode('utf-8')

            # Handle legacy base64 format (enc_v1)
            elif encrypted_data.startswith("enc_v1:"):
                encoded = encrypted_data[7:]
                return base64.b64decode(encoded.encode('utf-8')).decode('utf-8')

            # Handle legacy public key format (enc_pk_v1)
            elif encrypted_data.startswith("enc_pk_v1:"):
                encoded = encrypted_data[10:]
                return base64.b64decode(encoded.encode('utf-8')).decode('utf-8')

            else:
                # Assume unencrypted legacy data
                return encrypted_data

        except Exception as e:
            raise ValueError(f"Decryption failed: {str(e)}")

    def encrypt_data_with_public_key(self, data: str, public_key: str) -> str:
        """
        Encrypt data for a specific recipient.
        Uses symmetric encryption with the server key (same as encrypt_data).
        The public_key parameter is stored/logged for audit purposes but
        symmetric encryption is used for actual encryption.

        Args:
            data: Plain text data to encrypt
            public_key: Recipient's public key (for audit/logging)

        Returns:
            Encrypted string
        """
        # Use same symmetric encryption - the public key is for reference/audit
        # In a full implementation, you could use NaCl Box for true asymmetric encryption
        return self.encrypt_data(data)

    def process_document_with_key_injection(self,
                                          ocr_geometry_data: List[Dict],
                                          document_type: DocumentType,
                                          cross_document_data: Optional[Dict[str, List[str]]] = None,
                                          ocr_lines: Optional['OCRLinesContainer'] = None) -> KeyInjectionResult:
        """
        Process a document with key injection enhancement.

        Args:
            ocr_geometry_data: OCR results with geometry information from doctr
            document_type: Type of document being processed
            cross_document_data: Values from other documents for cross-validation
            ocr_lines: OCR lines container with confidence data for bank statements

        Returns:
            KeyInjectionResult with extracted key-value pairs and confidence scores
        """
        try:
            # Step 1: Choose extraction method based on document type
            # Legacy: Bank statement key injection with simple_bank_analyzer removed
            # Bank statement key injection is deprecated - use Qwen3-VL direct extraction instead
            if document_type == DocumentType.BANK_STATEMENT:
                # Bank statement key injection disabled - simple_bank_analyzer was removed
                from app.core.logger import get_logger
                logger = get_logger()
                logger.warning("Bank statement key injection disabled (simple_bank_analyzer removed)")
                # Fall through to traditional spatial analysis as fallback
                detected_keys = self.spatial_analyzer.extract_key_value_pairs_from_geometry(
                    ocr_geometry_data, document_type
                )
            else:
                # Use traditional spatial analysis for other document types
                detected_keys = self.spatial_analyzer.extract_key_value_pairs_from_geometry(
                    ocr_geometry_data, document_type
                )

            # Step 2: Handle enhanced confidence differently for bank statements
            enhanced_confidences = {}
            if document_type == DocumentType.BANK_STATEMENT:
                # Legacy: Simple bank analyzer confidence calculation disabled
                # Use traditional spatial analysis confidence instead
                detected_keys = self.detector.remove_duplicate_keys(detected_keys)
                detected_keys = self.detector.prioritize_required_keys(detected_keys, document_type)

                for key in detected_keys:
                    cross_values = cross_document_data.get(key.key_name, []) if cross_document_data else None
                    enhanced_confidences[key.key_name] = self.confidence_calculator.calculate_enhanced_confidence(
                        key, cross_document_values=cross_values
                    )
            else:
                # Step 2a: Remove duplicates and prioritize required keys for other document types
                detected_keys = self.detector.remove_duplicate_keys(detected_keys)
                detected_keys = self.detector.prioritize_required_keys(detected_keys, document_type)

                # Step 2b: Calculate enhanced confidence using traditional method for other document types
                for key in detected_keys:
                    cross_values = cross_document_data.get(key.key_name, []) if cross_document_data else None
                    enhanced_confidences[key.key_name] = self.confidence_calculator.calculate_enhanced_confidence(
                        key, cross_document_values=cross_values
                    )

            # Step 3: Generate processing summary
            processing_summary = self._generate_processing_summary(detected_keys, enhanced_confidences, document_type)

            # Log final results before returning
            from app.core.logger import get_logger
            logger = get_logger()
            logger.info("=== KEY INJECTION FINAL RESULTS ===")
            logger.info(f"Success: True")
            logger.info(f"Document Type: {document_type.value}")
            logger.info(f"Detected Keys Count: {len(detected_keys)}")
            logger.info(f"Enhanced Confidences Count: {len(enhanced_confidences)}")

            logger.info("Final Detected Keys:")
            for i, key in enumerate(detected_keys):
                logger.info(f"  Key {i+1}: '{key.key_name}' = '{key.value_candidate}' "
                           f"(detection_conf: {key.confidence:.3f}, value_conf: {key.value_confidence:.3f})")
                if key.geometry:
                    logger.info(f"    Geometry: {key.geometry}")

                confidence = enhanced_confidences.get(key.key_name)
                if confidence:
                    logger.info(f"    Enhanced Confidence: {confidence.overall_confidence:.3f} "
                               f"(reliable: {confidence.is_reliable}, needs_review: {confidence.needs_review})")
                    logger.info(f"    Explanation: {confidence.explanation[:100]}...")

            logger.info(f"Processing Summary: {processing_summary}")
            logger.info("=== END KEY INJECTION FINAL RESULTS ===")

            return KeyInjectionResult(
                document_type=document_type,
                detected_keys=detected_keys,
                enhanced_confidences=enhanced_confidences,
                processing_summary=processing_summary,
                success=True
            )

        except Exception as e:
            return KeyInjectionResult(
                document_type=document_type,
                detected_keys=[],
                enhanced_confidences={},
                processing_summary={},
                success=False,
                error_message=str(e)
            )

    def extract_simple_key_value_pairs(self,
                                     text_lines: List[str],
                                     document_type: DocumentType) -> List[DetectedKey]:
        """
        Simple key-value extraction without geometry data.
        Useful for quick processing or when geometry is not available.

        Args:
            text_lines: List of text lines from OCR
            document_type: Type of document being processed

        Returns:
            List of DetectedKey objects
        """
        detected_keys = self.detector.extract_key_value_pairs(text_lines, document_type)
        detected_keys = self.detector.remove_duplicate_keys(detected_keys)
        return self.detector.prioritize_required_keys(detected_keys, document_type)

    def get_high_confidence_extractions(self,
                                       result: KeyInjectionResult,
                                       min_confidence: float = 0.8) -> List[Tuple[DetectedKey, EnhancedConfidence]]:
        """
        Get high-confidence extracted key-value pairs from the result.

        Args:
            result: Key injection result
            min_confidence: Minimum confidence threshold

        Returns:
            List of (DetectedKey, EnhancedConfidence) tuples
        """
        high_confidence = []

        for key in result.detected_keys:
            confidence = result.enhanced_confidences.get(key.key_name)
            if confidence and confidence.overall_confidence >= min_confidence:
                high_confidence.append((key, confidence))

        return high_confidence

    def get_missing_required_keys(self, result: KeyInjectionResult) -> List[str]:
        """
        Get list of required keys that were not found in the document.

        Args:
            result: Key injection result

        Returns:
            List of missing required key names
        """
        detected_key_names = {key.key_name for key in result.detected_keys}
        all_required_keys = set(key_config.get_key_set(result.document_type).required_keys.keys())
        return list(all_required_keys - detected_key_names)

    def validate_document_completeness(self, result: KeyInjectionResult) -> Tuple[bool, List[str]]:
        """
        Check if all required keys were extracted with sufficient confidence.

        Args:
            result: Key injection result

        Returns:
            Tuple of (is_complete, missing_or_low_confidence_keys)
        """
        issues = []

        # Check for missing required keys
        missing_keys = self.get_missing_required_keys(result)
        issues.extend(missing_keys)

        # Check for low confidence required keys
        for key in result.detected_keys:
            if key_config.is_required_key(result.document_type, key.key_name):
                confidence = result.enhanced_confidences.get(key.key_name)
                if confidence and confidence.overall_confidence < 0.7:
                    issues.append(f"{key.key_name} (low confidence: {confidence.overall_confidence:.2f})")

        is_complete = len(issues) == 0
        return is_complete, issues

    def _generate_processing_summary(self,
                                   detected_keys: List[DetectedKey],
                                   enhanced_confidences: Dict[str, EnhancedConfidence],
                                   document_type: DocumentType) -> Dict:
        """Generate a summary of the processing results."""
        total_keys = len(detected_keys)
        keys_with_values = sum(1 for key in detected_keys if key.value_candidate)
        high_confidence_count = sum(
            1 for conf in enhanced_confidences.values()
            if conf.overall_confidence >= 0.8
        )

        # Calculate required key statistics
        required_key_set = key_config.get_key_set(document_type)
        required_keys_total = len(required_key_set.required_keys)
        detected_required_keys = sum(
            1 for key in detected_keys
            if key_config.is_required_key(document_type, key.key_name)
        )

        # Calculate overall document confidence
        if enhanced_confidences:
            overall_confidence = sum(conf.overall_confidence for conf in enhanced_confidences.values()) / len(enhanced_confidences)
        else:
            overall_confidence = 0.0

        return {
            'extraction_summary': ExtractionSummary(
                total_keys_detected=total_keys,
                keys_with_values=keys_with_values,
                high_confidence_extractions=high_confidence_count,
                required_keys_found=detected_required_keys,
                required_keys_total=required_keys_total,
                overall_document_confidence=overall_confidence
            ),
            'confidence_distribution': self._calculate_confidence_distribution(enhanced_confidences),
            'key_type_breakdown': self._calculate_key_type_breakdown(detected_keys, document_type)
        }

    def _calculate_confidence_distribution(self, enhanced_confidences: Dict[str, EnhancedConfidence]) -> Dict:
        """Calculate distribution of confidence scores."""
        if not enhanced_confidences:
            return {'high': 0, 'medium': 0, 'low': 0, 'total': 0}

        high_confidence = sum(1 for conf in enhanced_confidences.values() if conf.overall_confidence >= 0.8)
        medium_confidence = sum(1 for conf in enhanced_confidences.values() if 0.7 <= conf.overall_confidence < 0.8)
        low_confidence = sum(1 for conf in enhanced_confidences.values() if conf.overall_confidence < 0.7)

        return {
            'high': high_confidence,
            'medium': medium_confidence,
            'low': low_confidence,
            'total': len(enhanced_confidences)
        }

    def _calculate_key_type_breakdown(self, detected_keys: List[DetectedKey], document_type: DocumentType) -> Dict:
        """Calculate breakdown of required vs optional keys."""
        required_count = 0
        optional_count = 0
        required_with_values = 0
        optional_with_values = 0

        for key in detected_keys:
            is_required = key_config.is_required_key(document_type, key.key_name)
            has_value = bool(key.value_candidate)

            if is_required:
                required_count += 1
                if has_value:
                    required_with_values += 1
            else:
                optional_count += 1
                if has_value:
                    optional_with_values += 1

        return {
            'required': {
                'total': required_count,
                'with_values': required_with_values,
                'completion_rate': required_with_values / required_count if required_count > 0 else 0.0
            },
            'optional': {
                'total': optional_count,
                'with_values': optional_with_values,
                'completion_rate': optional_with_values / optional_count if optional_count > 0 else 0.0
            }
        }

    def get_supported_document_types(self) -> List[DocumentType]:
        """Get list of supported document types."""
        return list(DocumentType)

    def get_required_keys_for_document_type(self, document_type: DocumentType) -> List[str]:
        """Get list of required keys for a document type."""
        key_set = key_config.get_key_set(document_type)
        if key_set:
            return list(key_set.required_keys.keys())
        return []

    def get_all_keys_for_document_type(self, document_type: DocumentType) -> List[str]:
        """Get list of all keys (required + optional) for a document type."""
        return list(key_config.get_all_keys(document_type).keys())

    def _convert_to_text_elements(self, ocr_geometry_data: List[Dict]) -> List:
        """
        Convert OCR geometry data to text elements for enhanced detector.
        Converts normalized coordinates to pixel coordinates for positional analyzer.

        Args:
            ocr_geometry_data: OCR results with geometry information from doctr

        Returns:
            List of TextElement objects compatible with enhanced detector
        """
        from .data_structures import TextElement

        # Defensive check for None or empty input
        if not ocr_geometry_data:
            from app.core.logger import get_logger
            logger = get_logger()
            logger.warning("No OCR geometry data provided to _convert_to_text_elements")
            return []

        # Use standard page size to convert normalized coordinates to pixels
        # A4 ratio: 566x800 pixels (similar to what we use in PDF processing)
        page_width = 800
        page_height = 1000

        text_elements = []
        for i, item in enumerate(ocr_geometry_data):
            # Defensive check for None items
            if not item:
                from app.core.logger import get_logger
                logger = get_logger()
                logger.warning(f"OCR geometry item at index {i} is None, skipping")
                continue
            # Get normalized coordinates (0-1 range) from DocTR
            x1 = item.get('x1', 0)
            y1 = item.get('y1', 0)
            x2 = item.get('x2', 0)
            y2 = item.get('y2', 0)

            # Convert to absolute pixel coordinates for positional analyzer
            pixel_x1 = int(x1 * page_width)
            pixel_y1 = int(y1 * page_height)
            pixel_x2 = int(x2 * page_width)
            pixel_y2 = int(y2 * page_height)

            # Create geometry dictionary with absolute pixel coordinates
            geometry = {
                'x': pixel_x1,
                'y': pixel_y1,
                'width': pixel_x2 - pixel_x1,
                'height': pixel_y2 - pixel_y1,
                'x1': pixel_x1,
                'y1': pixel_y1,
                'x2': pixel_x2,
                'y2': pixel_y2
            }

            # Create text element
            text_element = TextElement(
                text=item.get('text', ''),
                confidence=item.get('confidence', 0.9),
                geometry=geometry
            )
            text_elements.append(text_element)

        from app.core.logger import get_logger
        logger = get_logger()
        logger.info(f"Converted {len(text_elements)} OCR elements to text elements with pixel coordinates")

        # Log first few elements for debugging
        for i, elem in enumerate(text_elements[:5]):
            logger.debug(f"Element {i+1}: '{elem.text}' at ({elem.geometry['x']}, {elem.geometry['y']}) "
                        f"size {elem.geometry['width']}x{elem.geometry['height']}")

        return text_elements


# Global instance for easy access
key_injection_manager = KeyInjectionManager()
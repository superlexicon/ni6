"""
Sequential ID Card Service - Handles ID card processing in sequential mode.

ID cards (PAN, national ID, driver's license) use GLiNER-based extraction
and are independent documents - they don't increment verification_state.

State is tracked via verification_state column in user_identity_index.
"""

from typing import Dict, Any, Optional, Tuple, List
from datetime import date
from app.services.sequential_document_processor_base import DocumentProcessorBase
from app.helper.extractors.gliner_id_card_extractor import GLiNERIDCardExtractor


class SequentialIDCardService(DocumentProcessorBase):
    """
    Service for handling ID card processing using GLiNER-based extraction.

    ID cards are independent documents that don't increment verification_state.
    They use GLiNER zero-shot NER for flexible field extraction.
    """

    # ============================================================
    # ABSTRACT METHOD IMPLEMENTATIONS
    # ============================================================

    def get_document_type(self) -> str:
        return "id_card"

    def get_required_fields(self) -> List[str]:
        # Core fields that should be present for most ID cards
        # Not all fields are required for all card types (e.g., some ID cards have no expiry)
        return ['full_name']

    def get_name_field(self) -> str:
        """ID cards use 'full_name' as the name field."""
        return 'full_name'

    def should_increment_state(self) -> bool:
        """
        ID cards are independent documents - no state change.

        Only passport increments state (1 -> 2 -> 3).
        ID cards can be submitted at any time without affecting flow.
        """
        return False

    def should_validate_photoholmes(self) -> bool:
        """
        ID cards should have PhotoHolmes forgery detection.

        Validates authenticity of the ID card document.
        """
        return True

    def get_photoholmes_document_type(self) -> str:
        """ID cards use 'id_card' for PhotoHolmes detection."""
        return "id_card"

    async def extract_fields_from_ocr(
        self, text_blocks: list, raw_text: str, image_bytes: bytes, is_pdf: bool
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Extract ID card fields using GLiNER-based extractor.

        Returns:
            (extracted_data, confidence_data)
        """
        # Use GLiNER ID card extractor
        extractor = GLiNERIDCardExtractor()
        result = await extractor.extract(image_bytes, is_pdf)

        # Build extracted_data from IDCardData
        extracted_data = {
            "document_type": result.document_type,
            "issuing_country": result.issuing_country,
            "full_name": result.full_name,
            "date_of_birth": result.date_of_birth,
            "gender": result.gender,
            "identification_number": result.identification_number,
            # All field_values as separate fields for flexibility
            **result.field_values,
        }

        # Build confidence_data from confidence_scores
        confidence_data = {}
        for field, confidence in result.confidence_scores.items():
            # Convert to 0-1 range if needed
            conf_value = confidence / 100 if confidence > 1 else confidence
            confidence_data[field] = {
                'overall_confidence': conf_value,
                'sources': ['gliner_ner']
            }

        # Also add core fields to confidence_data
        core_fields = ['full_name', 'date_of_birth', 'gender', 'identification_number']
        for field in core_fields:
            if field in extracted_data and extracted_data[field] and field not in confidence_data:
                confidence_data[field] = {
                    'overall_confidence': 0.85,  # Default confidence for extracted fields
                    'sources': ['gliner_ner']
                }

        self.logger.info(
            f"GLiNER ID card extraction: document_type={result.document_type}, "
            f"fields_extracted={len(result.field_values)}, "
            f"overall_confidence={result.overall_confidence:.2f}%"
        )

        return extracted_data, confidence_data

    def perform_document_specific_validations(
        self, extracted_data: Dict[str, Any], user_identity: Dict[str, Any]
    ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Perform ID card specific validations.

        Validations:
        - Document type detection (should be set)
        - Identification number validation (format varies by type)

        Note: ID cards have flexible validation since different
        card types have different requirements (e.g., PAN vs national ID).
        """
        document_type = extracted_data.get('document_type')

        # Check if document type was detected
        if not document_type:
            return False, "Could not detect ID card type", {}

        additional_checks = {
            'document_type': document_type,
            'issuing_country': extracted_data.get('issuing_country'),
        }

        # PAN-specific validation (if detected)
        if document_type and document_type.upper() == 'PAN':
            pan_number = extracted_data.get('pan_number') or extracted_data.get('identification_number')
            if pan_number:
                import re
                pan_pattern = re.compile(r'^[A-Z]{5}[0-9]{4}[A-Z]$', re.IGNORECASE)
                if not pan_pattern.match(pan_number):
                    additional_checks['pan_format_valid'] = False
                    return False, f"PAN number format invalid: {pan_number}", additional_checks
                additional_checks['pan_format_valid'] = True

        # ID cards are generally valid if we got this far
        return True, None, additional_checks

    # ============================================================
    # ID CARD SPECIFIC METHODS
    # ============================================================

    # ============================================================
    # PUBLIC ENTRY POINT
    # ============================================================

    async def process_id_card(
        self, client_public_key: str, file_data: str, filename: str,
        iv: str, callback_url: Optional[str] = None
    ):
        """
        Process an ID card document (PAN, national ID, driver's license).

        ID cards are independent documents - they don't affect verification_state.

        Args:
            client_public_key: Client's public key
            file_data: Base64 encoded file data
            filename: File name
            iv: Initialization vector (for encrypted data)
            callback_url: Optional callback URL

        Returns:
            SequentialJobResponse with extraction results
        """
        return await self.process_document(
            client_public_key=client_public_key,
            file_data=file_data,
            filename=filename,
            iv=iv,
            callback_url=callback_url
        )

"""
Sequential National ID Service - Handles national ID card processing.

NEW: Uses DocumentProcessorBase for unified validation pipeline with name-first validation.

National ID cards are alternative identity documents to passports.
They can be submitted at state 1 (after selfie) and require name matching
against the stored full_name from the database (if available).
"""

from typing import Dict, Any, Optional, Tuple, List
from datetime import date
from app.services.sequential_document_processor_base import DocumentProcessorBase
from app.helper.extractors.unified_id_extractor import UnifiedIDExtractor
from app.services.face_extraction_service import FaceExtractionService
from app.repositories.face_biometrics_repository import FaceBiometricsRepository
from app.helper.deepface_helper import DeepfaceHelper


class SequentialNationalIdService(DocumentProcessorBase):
    """Service for handling national ID card processing using unified framework."""

    def __init__(self):
        super().__init__()
        self.unified_extractor = UnifiedIDExtractor()
        self.face_extraction_service = FaceExtractionService()
        self.face_biometrics_repo = FaceBiometricsRepository()

    # ============================================================
    # ABSTRACT METHOD IMPLEMENTATIONS
    # ============================================================

    def get_document_type(self) -> str:
        return "national_id"

    def get_required_fields(self) -> List[str]:
        return [
            'full_name',
            'document_number',
            'date_of_birth',
        ]

    def get_name_field(self) -> str:
        """National ID uses 'full_name' as the name field."""
        return 'full_name'

    def get_photoholmes_document_type(self) -> str:
        return "id_card"

    def extract_fields_from_ocr(
        self, text_blocks: list, raw_text: str, image_bytes: bytes, is_pdf: bool
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Extract national ID fields using UnifiedIDExtractor.

        Returns:
            (extracted_data, confidence_data)
        """
        # Use the unified ID extractor
        import asyncio

        # Run the async extractor in sync context
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            document_data = loop.run_until_complete(
                self.unified_extractor.extract(image_bytes, is_pdf)
            )
        except Exception as e:
            self.logger.warning(f"Unified ID extraction failed: {e}")
            return {}, {}

        extracted_data = self._build_extracted_data(document_data)

        # Build confidence_data based on extraction confidence
        confidence_data = {}
        for field in self.get_required_fields():
            value = extracted_data.get(field)
            if value:
                confidence_data[field] = {
                    'overall_confidence': 0.90,  # High confidence for MRZ/VIZ extraction
                    'sources': self._get_extraction_sources(document_data, field)
                }
            else:
                confidence_data[field] = {
                    'overall_confidence': 0.0,
                    'sources': []
                }

        return extracted_data, confidence_data

    def perform_document_specific_validations(
        self, extracted_data: Dict[str, Any], user_identity: Dict[str, Any]
    ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Perform national ID specific validations.

        Validations:
        - Document expiry check (if expiry date present)
        """
        # Check document expiry (some ID cards have no expiry like Singapore NRIC)
        document_expiry_valid = None
        expiry_date = extracted_data.get('date_of_expiry')

        if expiry_date:
            document_expiry_valid = self._check_document_expiry(expiry_date)
        else:
            # No expiry date = valid for life (e.g., Singapore NRIC)
            document_expiry_valid = True

        additional_checks = {
            'document_expiry_valid': document_expiry_valid,
            'document_type': 'national_id'
        }

        if document_expiry_valid is False:
            return False, "National ID card has expired or expires soon", additional_checks

        return True, None, additional_checks

    def should_increment_state(self) -> bool:
        """
        National ID is an alternative to passport but does NOT increment state.
        Only passport increments state from 1 to 2.
        """
        return False

    # ============================================================
    # NATIONAL ID SPECIFIC METHODS
    # ============================================================

    def _build_extracted_data(self, document_data) -> Dict[str, Any]:
        """Build extracted_data dict from UnifiedIDExtractor result."""
        if not document_data:
            return {}

        # Check if this is a UnifiedIDExtractor result (DocumentExtractionResult)
        if hasattr(document_data, 'document_type'):
            result = {
                "document_type": document_data.document_type,
                "country_code": document_data.country_code,
                "number": document_data.get('number'),
                "full_name": document_data.get('full_name'),
                "dob": document_data.get('dob'),
                "date_of_birth": document_data.get('dob'),
                "sex": document_data.get('sex'),
                "date_of_expiry": document_data.get('expiry'),
                "expiry": document_data.get('expiry'),
                "place_of_birth": document_data.get('place_of_birth'),
                "issuing_authority": document_data.get('issuing_authority'),
                "issuing_country": document_data.get('issuing_country'),
                "document_country": document_data.get('issuing_country') or document_data.country_code,
                "address": document_data.get('address'),
                "date_of_issue": document_data.get('issue_date'),
            }

            # Filter out None values
            return {k: v for k, v in result.items() if v is not None}

        return {}

    def _get_extraction_sources(self, document_data, field: str) -> List[str]:
        """Get list of sources used for field extraction."""
        sources = []

        if hasattr(document_data, 'extraction_details'):
            details = document_data.extraction_details
            if details.get('mrz_used'):
                sources.append('mrz')
            if details.get('viz_used'):
                sources.append('viz')

        return sources if sources else ['ocr']

    def _check_document_expiry(self, expiry_date_str: Optional[str]) -> bool:
        """Check if document has minimum required validity."""
        if not expiry_date_str:
            return False

        expiry_date = self._parse_date(expiry_date_str)
        if not expiry_date:
            return False

        # Minimum 6 months validity
        from app.config.verification_config import verification_settings
        min_validity_days = getattr(verification_settings, 'passport_min_validity_days', 180)
        days_until_expiry = (expiry_date - date.today()).days

        return days_until_expiry >= min_validity_days

    def _parse_date(self, date_str: Optional[str]) -> Optional[date]:
        """Parse date string to date object."""
        if not date_str:
            return None

        try:
            date_clean = str(date_str).strip()

            date_formats = [
                "%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d", "%d/%m/%Y", "%Y%m%d",
                "%d %b %Y",   # 04 Apr 2033
                "%d %B %Y",   # 04 April 2033
                "%d-%b-%Y",   # 04-Apr-2033
                "%d/%b/%Y",   # 04/Apr/2033
            ]

            from datetime import datetime
            for date_variant in [date_clean, date_clean.title()]:
                for fmt in date_formats:
                    try:
                        return datetime.strptime(date_variant, fmt).date()
                    except ValueError:
                        continue

            return None
        except Exception:
            return None

    # ============================================================
    # PUBLIC ENTRY POINT
    # ============================================================

    async def process_national_id(
        self, client_public_key: str, file_data: str, filename: str,
        iv: str, callback_url: Optional[str] = None
    ):
        """
        Process a national ID document.

        This is the public entry point that calls the base class pipeline.
        """
        return await self.process_document(
            client_public_key=client_public_key,
            file_data=file_data,
            filename=filename,
            iv=iv,
            callback_url=callback_url
        )

"""
Sequential Driving License Service - Handles driving license processing.

NEW: Uses DocumentProcessorBase for unified validation pipeline with name-first validation.

Driving licenses are alternative identity documents to passports.
They can be submitted at state 1 (after selfie) and require name matching
against the stored full_name from the database (if available).
"""

from typing import Dict, Any, Optional, Tuple, List
from datetime import date
from app.services.sequential_document_processor_base import DocumentProcessorBase
from app.helper.extractors.unified_id_extractor import UnifiedIDExtractor


class SequentialDrivingLicenseService(DocumentProcessorBase):
    """Service for handling driving license processing using unified framework."""

    def __init__(self):
        super().__init__()
        self.unified_extractor = UnifiedIDExtractor()

    # ============================================================
    # ABSTRACT METHOD IMPLEMENTATIONS
    # ============================================================

    def get_document_type(self) -> str:
        return "driving_license"

    def get_required_fields(self) -> List[str]:
        return [
            'full_name',
            'license_number',
            'date_of_birth',
        ]

    def get_name_field(self) -> str:
        """Driving license uses 'full_name' as the name field."""
        return 'full_name'

    def get_photoholmes_document_type(self) -> str:
        return "driving_license"

    def extract_fields_from_ocr(
        self, text_blocks: list, raw_text: str, image_bytes: bytes, is_pdf: bool
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Extract driving license fields using UnifiedIDExtractor.

        Returns:
            (extracted_data, confidence_data)
        """
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

        # Build confidence_data
        confidence_data = {}
        for field in self.get_required_fields():
            value = extracted_data.get(field)
            if value:
                confidence_data[field] = {
                    'overall_confidence': 0.85,  # Good confidence for driving license extraction
                    'sources': ['ocr', 'viz']
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
        Perform driving license specific validations.

        Validations:
        - Document expiry check
        - Vehicle classes validation (if present)
        """
        # Check document expiry
        document_expiry_valid = None
        expiry_date = extracted_data.get('date_of_expiry')

        if expiry_date:
            document_expiry_valid = self._check_document_expiry(expiry_date)
        else:
            # Some driving licenses may not have expiry
            document_expiry_valid = True

        additional_checks = {
            'document_expiry_valid': document_expiry_valid,
            'document_type': 'driving_license',
            'vehicle_classes': extracted_data.get('vehicle_classes')
        }

        if document_expiry_valid is False:
            return False, "Driving license has expired or expires soon", additional_checks

        return True, None, additional_checks

    def should_increment_state(self) -> bool:
        """
        Driving license is an alternative to passport but does NOT increment state.
        Only passport increments state from 1 to 2.
        """
        return False

    # ============================================================
    # DRIVING LICENSE SPECIFIC METHODS
    # ============================================================

    def _build_extracted_data(self, document_data) -> Dict[str, Any]:
        """Build extracted_data dict from UnifiedIDExtractor result."""
        if not document_data:
            return {}

        if hasattr(document_data, 'document_type'):
            result = {
                "document_type": document_data.document_type,
                "country_code": document_data.country_code,
                "license_number": document_data.get('number') or document_data.get('document_number'),
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
                "vehicle_classes": document_data.get('vehicle_classes'),
            }

            # Filter out None values
            return {k: v for k, v in result.items() if v is not None}

        return {}

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

    async def process_driving_license(
        self, client_public_key: str, file_data: str, filename: str,
        iv: str, callback_url: Optional[str] = None
    ):
        """
        Process a driving license document.

        This is the public entry point that calls the base class pipeline.
        """
        return await self.process_document(
            client_public_key=client_public_key,
            file_data=file_data,
            filename=filename,
            iv=iv,
            callback_url=callback_url
        )

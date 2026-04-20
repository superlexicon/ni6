"""
Sequential Tax Return Service - Handles tax return processing in sequential mode.

NEW: Uses DocumentProcessorBase for unified validation pipeline with name-first validation.

Tax returns are optional documents that can be submitted after bank statement completion.
They implement name matching against the stored full_name from passport step.
"""

from typing import Dict, Any, Optional, Tuple, List
from datetime import date
from app.services.sequential_document_processor_base import DocumentProcessorBase
from app.helper.extractors.tax_statement_extractor import TaxStatementExtractor
from app.core.key_injection.key_injection_manager import key_injection_manager
from app.core.key_injection.key_config import DocumentType


class SequentialTaxReturnService(DocumentProcessorBase):
    """Service for handling tax return processing using unified framework."""

    def __init__(self):
        super().__init__()
        self.tax_statement_extractor = TaxStatementExtractor()

    # ============================================================
    # ABSTRACT METHOD IMPLEMENTATIONS
    # ============================================================

    def get_document_type(self) -> str:
        return "tax_return"

    def get_required_fields(self) -> List[str]:
        return [
            'taxpayer_name',
            'tax_year',
            'gross_income',
        ]

    def get_name_field(self) -> str:
        """Tax return uses 'taxpayer_name' as the name field."""
        return 'taxpayer_name'

    def get_photoholmes_document_type(self) -> str:
        return "tax_statement"

    def extract_fields_from_ocr(
        self, text_blocks: list, raw_text: str, image_bytes: bytes, is_pdf: bool
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Extract tax return fields using hybrid approach: key injection + regex fallback.

        Returns:
            (extracted_data, confidence_data)
        """
        # Try key injection extraction first
        extracted_data = {}

        if text_blocks:
            try:
                key_injection_result = key_injection_manager.process_document_with_key_injection(
                    ocr_geometry_data=text_blocks,
                    document_type=DocumentType.TAX_STATEMENT
                )

                if key_injection_result.success and key_injection_result.detected_keys:
                    # Build extracted_data from key injection results
                    for key in key_injection_result.detected_keys:
                        extracted_data[key.key_name] = key.value_candidate

                    # Check if we have required fields
                    required_fields = self.get_required_fields()
                    has_required = all(extracted_data.get(f) for f in required_fields)

                    if has_required:
                        self.logger.info("Key injection extraction successful with all required fields")
                    else:
                        self.logger.info("Key injection missing required fields, falling back to regex")
                        extracted_data = self._extract_fallback(image_bytes, is_pdf)
            except Exception as e:
                self.logger.warning(f"Key injection extraction failed: {e}, falling back to regex")
                extracted_data = self._extract_fallback(image_bytes, is_pdf)
        else:
            extracted_data = self._extract_fallback(image_bytes, is_pdf)

        # Build confidence_data
        confidence_data = {}
        for field in self.get_required_fields():
            value = extracted_data.get(field)
            if value:
                confidence_data[field] = {
                    'overall_confidence': 0.80,  # Default confidence for extracted fields
                    'sources': ['ocr', 'key_injection']
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
        Perform tax return specific validations.

        Validations:
        - Tax year not too old (max 3 years from current)
        """
        tax_year = extracted_data.get('tax_year')
        additional_checks = {'tax_year': tax_year}

        if tax_year:
            try:
                tax_year_int = int(str(tax_year))
                current_year = date.today().year

                if tax_year_int < current_year - 3:
                    return False, f"Tax year too old: {tax_year_int}", additional_checks

                if tax_year_int > current_year + 1:
                    return False, f"Tax year in future: {tax_year_int}", additional_checks

            except (ValueError, TypeError):
                # If we can't parse the year, don't fail on this validation
                pass

        return True, None, additional_checks

    def should_increment_state(self) -> bool:
        """Tax return is optional - no state increment."""
        return False

    # ============================================================
    # TAX RETURN SPECIFIC METHODS
    # ============================================================

    def _extract_fallback(self, image_bytes: bytes, is_pdf: bool) -> Dict[str, Any]:
        """Fallback to regex-based extraction when key injection fails."""
        import asyncio

        # Run the async extractor in sync context
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            tax_data = loop.run_until_complete(
                self.tax_statement_extractor.extract(image_bytes, is_pdf)
            )
        except Exception as e:
            self.logger.warning(f"Tax statement extraction failed: {e}")
            return {}

        return {
            "taxpayer_name": getattr(tax_data, 'taxpayer_name', None),
            "tax_id": getattr(tax_data, 'tax_id', None),
            "social_security_number": getattr(tax_data, 'social_security_number', None),
            "address": getattr(tax_data, 'address', None),
            "tax_year": getattr(tax_data, 'tax_year', None),
            "tax_period_start": getattr(tax_data, 'tax_period_start', None),
            "tax_period_end": getattr(tax_data, 'tax_period_end', None),
            "gross_income": getattr(tax_data, 'gross_income', None),
            "net_income": getattr(tax_data, 'net_income', None),
            "taxable_income": getattr(tax_data, 'taxable_income', None),
            "tax_paid": getattr(tax_data, 'tax_paid', None),
            "tax_withheld": getattr(tax_data, 'tax_withheld', None),
            "tax_due": getattr(tax_data, 'tax_due', None),
            "tax_refund": getattr(tax_data, 'tax_refund', None),
            "filing_date": getattr(tax_data, 'filing_date', None),
            "filing_status": getattr(tax_data, 'filing_status', None),
            "tax_authority": getattr(tax_data, 'tax_authority', None),
        }

    # ============================================================
    # PUBLIC ENTRY POINT
    # ============================================================

    async def process_tax_return(
        self, client_public_key: str, file_data: str, filename: str,
        callback_url: Optional[str] = None
    ):
        """
        Process a tax return document.

        This is the public entry point that calls the base class pipeline.
        """
        return await self.process_document(
            client_public_key=client_public_key,
            file_data=file_data,
            filename=filename,
            callback_url=callback_url
        )

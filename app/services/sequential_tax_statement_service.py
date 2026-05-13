"""
Sequential Tax Statement Service - Handles tax statement processing as independent documents.

Tax statements are NOT part of the sequential verification flow (states 0-3).
They can be submitted independently at any time without state validation or increment.
"""

from typing import Dict, Any, Optional
import asyncio
import time
import base64
import uuid
from app.dto.verification_session import SequentialJobResponse
from app.services import comprehensive_photoholmes_service
from app.services.detailed_analysis_service import DetailedAnalysisService
from app.services.selfie_validation_service import SelfieValidationService
from app.core.logger import get_logger
from app.helper.doctr.document_text_extractor import DocumentTextExtractor
from app.helper.extractors.tax_statement_extractor import TaxStatementExtractor
from app.helper.tax_statement_validator import TaxStatementValidator
from app.core.key_injection.key_injection_manager import key_injection_manager
from app.core.key_injection.key_config import DocumentType
from app.repositories.user_key_repository import UserKeyRepository


class SequentialTaxStatementService:
    """Service for handling tax statement processing (independent of sequential flow)"""

    def __init__(self):
        self.logger = get_logger()
        self.user_key_repo = UserKeyRepository()
        self.detailed_analysis_service = DetailedAnalysisService()
        self.text_extractor = DocumentTextExtractor()
        self.tax_statement_extractor = TaxStatementExtractor()
        self.tax_statement_validator = TaxStatementValidator()

    async def process_tax_statement(
        self,
        client_public_key: str,
        file_data: str,
        filename: str,
        callback_url: Optional[str] = None
    ) -> SequentialJobResponse:
        """
        Process a tax statement document (independent of sequential flow).

        NO state validation - tax statements can be submitted at any time.
        NO state increment - doesn't affect verification flow.

        Returns simplified SequentialJobResponse with:
        - result: bool (all checks passed)
        - verification_state: int (unchanged from current state)
        - extracted_data: tax statement fields
        - forgery_checks: PhotoHolmes results
        - other_checks: validation_score, overall_valid
        """
        job_id = f"tax_{uuid.uuid4().hex[:12]}"
        start_time = time.time()

        # Get current verification state (will be returned unchanged)
        try:
            user_key = self.user_key_repo.get_key_by_public_key(client_public_key)
            current_state = 0
            if user_key and user_key.get('user_identity_id'):
                from app.repositories.user_identity_repository import UserIdentityRepository
                user_identity_repo = UserIdentityRepository()
                current_state = user_identity_repo.get_verification_state(user_key['user_identity_id'])
        except Exception as e:
            self.logger.warning(f"Could not get verification state: {e}")
            current_state = 0

        try:
            self.logger.info(f"Processing tax statement for: {client_public_key[:16]}...")

            # Decode image
            image_bytes = base64.b64decode(file_data)
            is_pdf = filename.lower().endswith('.pdf') or image_bytes.startswith(b'%PDF')

            # Convert PDF for PhotoHolmes
            photoholmes_image_bytes = image_bytes
            if is_pdf:
                photoholmes_image_bytes = self._convert_pdf_to_image(image_bytes)

            # PHASE 1 & 2: Run PhotoHolmes + OCR in PARALLEL for performance
            self.logger.info("Running PhotoHolmes and OCR extraction in parallel...")

            async def safe_ocr_extraction():
                """Wrapper to catch OCR extraction errors"""
                try:
                    return await self.text_extractor.extract_text_with_geometry(
                        image_bytes, is_pdf=is_pdf
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to extract text blocks: {e}")
                    return []

            # Run both tasks in parallel
            photoholmes_results, text_blocks = await asyncio.gather(
                comprehensive_photoholmes_service.run_all_methods(photoholmes_image_bytes, document_type="tax_statement"),
                safe_ocr_extraction()
            )

            # Process PhotoHolmes results
            forgery_checks = None
            if photoholmes_results:
                detailed_results = self.detailed_analysis_service.transform_photoholmes_results(photoholmes_results)
                # Build per-method forgery_checks: {"method": {"score": x, "threshold": y}}
                forgery_checks = {}
                for check in detailed_results.checks:
                    forgery_checks[check.name] = {
                        "score": round(check.raw_score, 3),
                        "threshold": check.research_threshold
                    }

                # Validate PhotoHolmes
                validation_service = SelfieValidationService()
                photoholmes_valid, photoholmes_error, photoholmes_error_code = validation_service.validate_photoholmes_results(detailed_results)
                if not photoholmes_valid:
                    return SequentialJobResponse(
                        result=False,
                        job_id=job_id,
                        verification_state=current_state,  # Unchanged
                        processing_time_seconds=round(time.time() - start_time, 2),
                        forgery_checks=forgery_checks
                    )

            # PHASE 3: Extract tax data (hybrid approach: key injection + regex fallback)
            extracted_data = await self._extract_tax_data(text_blocks, image_bytes, is_pdf)

            # Validate required fields
            if not extracted_data.get('taxpayer_name'):
                return SequentialJobResponse(
                    result=False,
                    job_id=job_id,
                    verification_state=current_state,
                    processing_time_seconds=round(time.time() - start_time, 2),
                    extracted_data=extracted_data,
                    forgery_checks=forgery_checks
                )

            # PHASE 4: Validate tax statement
            validation_result = await self._validate_tax_statement(extracted_data)

            # Build other_checks
            other_checks = {
                "validation_score": validation_result.validation_score,
                "overall_valid": validation_result.overall_valid,
                "tax_year_valid": validation_result.tax_year_valid,
                "amounts_consistent": validation_result.amounts_consistent,
                "has_required_fields": validation_result.has_required_fields,
                "findings": validation_result.findings
            }

            # Check if validation passed
            if not validation_result.overall_valid:
                return SequentialJobResponse(
                    result=False,
                    job_id=job_id,
                    verification_state=current_state,
                    processing_time_seconds=round(time.time() - start_time, 2),
                    extracted_data=extracted_data,
                    forgery_checks=forgery_checks,
                    other_checks=other_checks
                )

            # Success - return with unchanged verification state
            self.logger.info(f"Tax statement processed successfully. State unchanged: {current_state}")
            return SequentialJobResponse(
                result=True,
                job_id=job_id,
                verification_state=current_state,  # Unchanged
                processing_time_seconds=round(time.time() - start_time, 2),
                extracted_data=extracted_data,
                forgery_checks=forgery_checks,
                other_checks=other_checks
            )

        except Exception as e:
            self.logger.error(f"Error processing tax statement: {str(e)}")
            return SequentialJobResponse(
                result=False,
                job_id=job_id,
                verification_state=current_state,
                processing_time_seconds=round(time.time() - start_time, 2)
            )

    async def _extract_tax_data(self, text_blocks: list, image_bytes: bytes, is_pdf: bool) -> Dict[str, Any]:
        """
        Extract tax data using hybrid approach: key injection (primary) + regex fallback.

        Args:
            text_blocks: OCR text blocks with geometry
            image_bytes: Original image bytes for fallback extraction
            is_pdf: Whether content is PDF

        Returns:
            Dict with extracted tax fields
        """
        # Try key injection extraction first
        if text_blocks:
            try:
                key_injection_result = key_injection_manager.process_document_with_key_injection(
                    ocr_geometry_data=text_blocks,
                    document_type=DocumentType.TAX_STATEMENT
                )

                if key_injection_result.success and key_injection_result.detected_keys:
                    # Build extracted_data from key injection results
                    extracted = self._build_extracted_data_from_keys(
                        key_injection_result.detected_keys,
                        text_blocks
                    )

                    # Check if we have required fields
                    required_fields = ['taxpayer_name', 'tax_year', 'gross_income']
                    has_required = all(extracted.get(f) for f in required_fields)

                    if has_required:
                        self.logger.info("Key injection extraction successful with all required fields")
                        return extracted
                    else:
                        self.logger.info("Key injection missing required fields, falling back to regex")
            except Exception as e:
                self.logger.warning(f"Key injection extraction failed: {e}, falling back to regex")

        # Fallback to regex extraction
        self.logger.info("Using regex-based tax statement extraction")
        tax_data = await self.tax_statement_extractor.extract(image_bytes, is_pdf)
        return self._build_extracted_data_from_schema(tax_data)

    def _build_extracted_data_from_keys(self, detected_keys: list, text_blocks: list) -> Dict[str, Any]:
        """Build extracted_data dict from key injection detected keys."""
        extracted = {}
        for key in detected_keys:
            extracted[key.key_name] = key.value_candidate

        # Add raw OCR text
        extracted['raw_data'] = "\n".join([block.get('text', '') for block in text_blocks])

        return extracted

    def _build_extracted_data_from_schema(self, tax_data) -> Dict[str, Any]:
        """Build extracted_data dict from TaxStatementData schema."""
        if not tax_data:
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
            "raw_data": getattr(tax_data, 'raw_data', None),
        }

    async def _validate_tax_statement(self, extracted_data: Dict[str, Any]):
        """Validate tax statement data using TaxStatementValidator."""
        from app.schemas.tax_statement_schema import TaxStatementData

        # Convert dict to TaxStatementData
        tax_data = TaxStatementData(
            taxpayer_name=extracted_data.get('taxpayer_name'),
            tax_id=extracted_data.get('tax_id'),
            social_security_number=extracted_data.get('social_security_number'),
            address=extracted_data.get('address'),
            tax_year=extracted_data.get('tax_year'),
            tax_period_start=extracted_data.get('tax_period_start'),
            tax_period_end=extracted_data.get('tax_period_end'),
            gross_income=extracted_data.get('gross_income'),
            net_income=extracted_data.get('net_income'),
            taxable_income=extracted_data.get('taxable_income'),
            tax_paid=extracted_data.get('tax_paid'),
            tax_withheld=extracted_data.get('tax_withheld'),
            tax_due=extracted_data.get('tax_due'),
            tax_refund=extracted_data.get('tax_refund'),
            filing_date=extracted_data.get('filing_date'),
            filing_status=extracted_data.get('filing_status'),
            tax_authority=extracted_data.get('tax_authority'),
            confidence_scores={}
        )

        return await self.tax_statement_validator.validate(tax_data)

    def _convert_pdf_to_image(self, pdf_bytes: bytes) -> bytes:
        """Convert PDF first page to PNG image."""
        try:
            import fitz
            pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            if len(pdf_doc) > 0:
                page = pdf_doc[0]
                pix = page.get_pixmap(dpi=150)
                result = pix.tobytes("png")
                pdf_doc.close()
                return result
            pdf_doc.close()
            return pdf_bytes
        except Exception as e:
            self.logger.warning(f"PDF conversion failed: {e}")
            return pdf_bytes

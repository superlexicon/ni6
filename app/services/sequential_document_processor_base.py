"""
Sequential Document Processor Base - Abstract base class for unified document processing.

Implements the generalized document processing framework with name-first validation:
1. State validation
2. Parallel PhotoHolmes + OCR
3. PhotoHolmes validation
4. ORIENTATION VALIDATION ⭐ NEW
5. FIELD EXTRACTION WITH CONFIDENCE ⭐ NEW
6. Required fields validation
7. NAME MATCHING (gatekeeper) ⭐ NEW
8. Document-specific validations
9. State increment (in user_keys for per-device state)
10. Return response

Multi-Device State Management:
- user_keys: Stores per-device verification_state and sequence_no
- user_identity_index: Stores overall best state across devices
- On successful processing: Update state in user_keys for this device
- On failure: Revert state in user_keys for this device
- Document types determine state progression:
  - passport/national_id/driving_license: state 0 -> 2
  - bank_statement: state 2 -> 3
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Callable
import asyncio
import time
import base64
import uuid
from datetime import datetime, date

from app.dto.verification_session import SequentialJobResponse
from app.services.verification_state_service import VerificationStateService
from app.utils.string_matching import fuzzy_match_names, get_match_details
from app.core.logger import get_logger
from app.config.verification_config import verification_settings
from app.services import comprehensive_photoholmes_service
from app.services.detailed_analysis_service import DetailedAnalysisService
from app.services.selfie_validation_service import SelfieValidationService
from app.repositories.user_key_repository import UserKeyRepository
from app.repositories.user_identity_repository import UserIdentityRepository
from app.helper.doctr.document_text_extractor import DocumentTextExtractor
from app.utils.orientation_validator import OrientationValidator
from app.dto import DocumentErrorCode


class DocumentProcessorBase(ABC):
    """
    Base class for sequential document processing with unified validation pipeline.

    Subclasses must implement abstract methods to define document-specific behavior.
    """

    def __init__(self):
        self.logger = get_logger()
        self.state_service = VerificationStateService()
        self.user_key_repo = UserKeyRepository()
        self.user_identity_repo = UserIdentityRepository()
        self.detailed_analysis_service = DetailedAnalysisService()
        self.text_extractor = DocumentTextExtractor()
        self.orientation_validator = OrientationValidator()

    # ============================================================
    # ABSTRACT METHODS - Must be implemented by subclasses
    # ============================================================

    @abstractmethod
    def get_document_type(self) -> str:
        """
        Return the document type identifier (e.g., 'bank_statement', 'tax_return').

        Used for logging, job_id generation, and state validation.
        """
        pass

    @abstractmethod
    def get_required_fields(self) -> List[str]:
        """
        Return list of required field names for this document type.

        These fields must be present and meet confidence thresholds for processing to pass.
        """
        pass

    @abstractmethod
    def extract_fields_from_ocr(
        self, text_blocks: list, raw_text: str, image_bytes: bytes, is_pdf: bool
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Extract required fields from OCR results.

        Returns:
            (extracted_data, confidence_data) where:
            - extracted_data: Dict of field_name -> extracted_value
            - confidence_data: Dict of field_name -> confidence_info

        Example confidence_data:
        {
            'full_name': {'overall_confidence': 0.85, 'sources': ['mrz', 'viz']},
            'document_number': {'overall_confidence': 0.92, 'sources': ['mrz']}
        }
        """
        pass

    @abstractmethod
    def perform_document_specific_validations(
        self, extracted_data: Dict[str, Any], user_identity: Dict[str, Any]
    ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Perform document-specific validations beyond required fields.

        Args:
            extracted_data: All extracted field values
            user_identity: User identity record from database

        Returns:
            (is_valid, error_message, additional_other_checks)
        """
        pass

    @abstractmethod
    def should_increment_state(self) -> bool:
        """
        Return True if this document type should increment verification_state.

        Example:
        - Bank statement: True (increments from state 2 to 3)
        - Tax return: False (independent document, no state change)
        """
        pass

    # ============================================================
    # OPTIONAL HOOKS - Can be overridden by subclasses
    # ============================================================

    def get_name_field(self) -> str:
        """Return the field name that contains the person's name for matching."""
        return 'full_name'

    def get_photoholmes_document_type(self) -> str:
        """Return the document type for PhotoHolmes detection."""
        return self.get_document_type()

    def should_validate_photoholmes(self) -> bool:
        """Return True if PhotoHolmes forgery detection should be performed."""
        return True

    def should_use_pdf_conversion(self) -> bool:
        """Return True if PDF should be converted to image for PhotoHolmes."""
        return True

    def get_name_from_extracted_data(self, extracted_data: Dict[str, Any]) -> Optional[str]:
        """
        Extract the name from extracted_data for matching against stored full_name.

        Override if name extraction needs custom logic.
        """
        name_field = self.get_name_field()
        name = extracted_data.get(name_field)

        # Handle alternative name fields
        if not name:
            for alt_field in ['account_holder_name', 'taxpayer_name', 'holder_name']:
                name = extracted_data.get(alt_field)
                if name:
                    break

        return name

    # ============================================================
    # COMMON PIPELINE - Shared implementation for all documents
    # ============================================================

    async def process_document(
        self,
        client_public_key: str,
        file_data: str,
        filename: str,
        user_identity_id: Optional[str] = None,
        stored_full_name: Optional[str] = None,
        callback_url: Optional[str] = None
    ) -> SequentialJobResponse:
        """
        Main processing pipeline - implements the unified validation flow.

        Pipeline:
        1. State validation
        2. Parallel PhotoHolmes + OCR
        3. PhotoHolmes validation
        4. ORIENTATION VALIDATION ⭐
        5. FIELD EXTRACTION WITH CONFIDENCE ⭐
        6. Required fields validation
        7. NAME MATCHING (gatekeeper) ⭐
        8. Document-specific validations
        9. State increment
        10. Return response
        """
        job_id = f"{self.get_document_type()}_{uuid.uuid4().hex[:12]}"
        start_time = time.time()
        document_type = self.get_document_type()

        # ============================================================
        # PHASE 1: State validation
        # ============================================================
        is_valid, error_msg, is_resubmission = self.state_service.validate_document_submission(
            client_public_key, document_type
        )
        if not is_valid:
            self.logger.error(f"State validation failed: {error_msg}")
            current_state = self.state_service.get_verification_state(client_public_key)
            current_seq = self.state_service.get_sequence_no(client_public_key)
            return self._build_error_response(
                job_id, start_time, current_state, current_seq, error_msg
            )

        if is_resubmission:
            self.logger.info(f"Processing {document_type} resubmission for: {client_public_key[:16]}...")

        # ============================================================
        # Get user_identity_id and user data
        # ============================================================
        if not user_identity_id:
            user_key = self.user_key_repo.get_key_by_public_key(client_public_key)
            if not user_key or not user_key.get('user_identity_id'):
                current_state = self.state_service.get_verification_state(client_public_key)
                current_seq = self.state_service.get_sequence_no(client_public_key)
                return self._build_error_response(
                    job_id, start_time, current_state, current_seq,
                    "User identity not found. Complete previous steps first."
                )
            user_identity_id = user_key['user_identity_id']

        user_identity = self.user_identity_repo.get_user_by_id(user_identity_id)
        if not user_identity:
            current_state = self.state_service.get_verification_state(client_public_key)
            current_seq = self.state_service.get_sequence_no(client_public_key)
            return self._build_error_response(
                job_id, start_time, current_state, current_seq,
                "User identity record not found"
            )

        # Get stored full_name if not provided
        if not stored_full_name:
            stored_full_name = user_identity.get('full_name')

        self.logger.info(f"Processing {document_type} for user_identity: {user_identity_id}")

        try:
            # ============================================================
            # PHASE 2: Decode image and prepare for processing
            # ============================================================
            image_bytes = base64.b64decode(file_data)
            is_pdf = filename.lower().endswith('.pdf') or image_bytes.startswith(b'%PDF')

            # Convert PDF for PhotoHolmes if needed
            photoholmes_image_bytes = image_bytes
            if is_pdf and self.should_use_pdf_conversion():
                photoholmes_image_bytes = self._convert_pdf_to_image(image_bytes)

            # ============================================================
            # PHASE 3: Parallel PhotoHolmes + OCR extraction
            # ============================================================
            self.logger.info(f"Running PhotoHolmes and OCR extraction in parallel...")

            async def safe_ocr_extraction():
                """Wrapper to catch OCR extraction errors"""
                try:
                    return await self.text_extractor.extract_text_with_geometry_enhanced(
                        image_bytes, is_pdf=is_pdf, max_pages=1
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to extract text blocks: {e}")
                    return []

            # Run PhotoHolmes and OCR in parallel
            tasks = [safe_ocr_extraction()]

            if self.should_validate_photoholmes():
                tasks.append(
                    comprehensive_photoholmes_service.run_all_methods(
                        photoholmes_image_bytes,
                        document_type=self.get_photoholmes_document_type()
                    )
                )

            if self.should_validate_photoholmes():
                text_blocks, photoholmes_results = await asyncio.gather(*tasks)
            else:
                text_blocks = await tasks[0]
                photoholmes_results = None

            # Process OCR results - preserve reading order with newlines
            # Using newlines instead of spaces maintains the document structure
            # This helps GLiNER2 understand label-value relationships and multi-line addresses
            raw_text = "\n".join([block.get('text', '') for block in text_blocks])

            # Extract orientation info for logging
            is_landscape = any(
                block.get('orientation') == 'landscape'
                for block in text_blocks
            ) if text_blocks else False

            if is_landscape:
                self.logger.info(f"Processing landscape {document_type} - coordinates have been transformed")

            # ============================================================
            # PHASE 4.5: Orientation validation (after OCR, before field extraction)
            # ============================================================
            orientation_valid, orientation_error = self.orientation_validator.validate_orientation(
                text_blocks=text_blocks,
                document_type=document_type
            )

            if not orientation_valid:
                current_state = self.state_service.get_verification_state(client_public_key)
                current_seq = self.state_service.get_sequence_no(client_public_key)
                return self._build_error_response(
                    job_id, start_time, current_state, current_seq, orientation_error, DocumentErrorCode.OCR_FAILED,
                    raw_text=raw_text
                )

            # ============================================================
            # PHASE 5: PhotoHolmes validation
            # ============================================================
            forgery_checks = None
            if photoholmes_results and self.should_validate_photoholmes():
                detailed_results = self.detailed_analysis_service.transform_photoholmes_results(
                    photoholmes_results
                )

                # Build per-method forgery_checks
                forgery_checks = {}
                for check in detailed_results.checks:
                    forgery_checks[check.name] = {
                        "score": round(check.raw_score, 3),
                        "threshold": check.research_threshold
                    }

                # Validate PhotoHolmes results
                validation_service = SelfieValidationService()
                photoholmes_valid, photoholmes_error, photoholmes_error_code = validation_service.validate_photoholmes_results(
                    detailed_results
                )

                if not photoholmes_valid:
                    # Revert state if it was incremented (in user_keys for this device)
                    if user_identity_id:
                        current_state = self.state_service.get_verification_state(client_public_key)
                        # Bank statement processing happens at state 2, so revert if state went beyond 2
                        if current_state > 2:
                            # Revert state in user_keys for this device
                            self.user_key_repo.update_state_and_sequence(
                                user_public_key=client_public_key,
                                verification_state=2,
                                sequence_no=2
                            )
                            self.logger.info(f"Reverted verification state after PhotoHolmes failure: {current_state} -> 2")

                    current_state = self.state_service.get_verification_state(client_public_key)
                    current_seq = self.state_service.get_sequence_no(client_public_key)
                    return SequentialJobResponse(
                        result=False,
                        job_id=job_id,
                        verification_state=current_state,
                        sequence_no=current_seq,
                        processing_time_seconds=round(time.time() - start_time, 2),
                        forgery_checks=forgery_checks,
                        error=photoholmes_error,
                        error_code=photoholmes_error_code,
                        extracted_data={'raw_data': raw_text}
                    )

            # ============================================================
            # PHASE 6: Field extraction with confidence
            # ============================================================
            # Support both sync and async extract_fields_from_ocr implementations
            # Some services (like bank statement, ID card) use async GLiNER extractors
            extraction_result = self.extract_fields_from_ocr(
                text_blocks, raw_text, image_bytes, is_pdf
            )
            # Check if result is a coroutine (async implementation)
            if asyncio.iscoroutine(extraction_result):
                extracted_data, confidence_data = await extraction_result
            else:
                extracted_data, confidence_data = extraction_result

            # ============================================================
            # PHASE 7: Validate required fields with confidence
            # ============================================================
            required_fields_valid, missing_fields, field_validations = \
                self.validate_required_fields_with_confidence(extracted_data, confidence_data)

            if not required_fields_valid:
                error_msg = f"Required field validation failed: {', '.join(missing_fields)}"
                self.logger.error(error_msg)

                # Determine error code based on missing fields
                if any('missing' in f.lower() for f in missing_fields):
                    error_code = DocumentErrorCode.LOGICAL_EXTRACTION_INCOMPLETE
                elif any('low confidence' in f.lower() for f in missing_fields):
                    error_code = DocumentErrorCode.LOGICAL_EXTRACTION_LOW_CONFIDENCE
                else:
                    error_code = DocumentErrorCode.OCR_FAILED

                # Revert state if it was incremented (in user_keys for this device)
                if user_identity_id:
                    current_state = self.state_service.get_verification_state(client_public_key)
                    # Bank statement processing happens at state 2, so revert if state went beyond 2
                    if current_state > 2:
                        # Revert state in user_keys for this device
                        self.user_key_repo.update_state_and_sequence(
                            user_public_key=client_public_key,
                            verification_state=2,
                            sequence_no=2
                        )
                        self.logger.info(f"Reverted verification state after required fields failure: {current_state} -> 2")

                current_state = self.state_service.get_verification_state(client_public_key)
                current_seq = self.state_service.get_sequence_no(client_public_key)
                return SequentialJobResponse(
                    result=False,
                    job_id=job_id,
                    verification_state=current_state,
                    sequence_no=current_seq,
                    processing_time_seconds=round(time.time() - start_time, 2),
                    extracted_data=extracted_data,
                    forgery_checks=forgery_checks,
                    other_checks={'field_validations': field_validations},
                    error=error_msg,
                    error_code=error_code
                )

            # ============================================================
            # PHASE 8: NAME MATCHING (GATEKEEPER) ⭐
            # ============================================================
            name_match_valid, name_match_score, name_match_details = \
                self.validate_name_matching(stored_full_name, extracted_data)

            if not name_match_valid:
                error_msg = (
                    f"Name matching failed: score={name_match_score}%, "
                    f"threshold={verification_settings.name_match_threshold}%"
                )
                self.logger.error(error_msg)

                # Revert state if it was incremented (in user_keys for this device)
                if user_identity_id:
                    current_state = self.state_service.get_verification_state(client_public_key)
                    # Bank statement processing happens at state 2, so revert if state went beyond 2
                    if current_state > 2:
                        # Revert state in user_keys for this device
                        self.user_key_repo.update_state_and_sequence(
                            user_public_key=client_public_key,
                            verification_state=2,
                            sequence_no=2
                        )
                        self.logger.info(f"Reverted verification state after name matching failure: {current_state} -> 2")

                current_state = self.state_service.get_verification_state(client_public_key)
                current_seq = self.state_service.get_sequence_no(client_public_key)
                return SequentialJobResponse(
                    result=False,
                    job_id=job_id,
                    verification_state=current_state,
                    sequence_no=current_seq,
                    processing_time_seconds=round(time.time() - start_time, 2),
                    extracted_data=extracted_data,
                    forgery_checks=forgery_checks,
                    other_checks=name_match_details,
                    error=error_msg,
                    error_code=DocumentErrorCode.LOGICAL_NAME_MISMATCH
                )

            # ============================================================
            # PHASE 9: Document-specific validations
            # ============================================================
            doc_valid, doc_error, additional_other_checks = \
                self.perform_document_specific_validations(extracted_data, user_identity)

            if not doc_valid:
                self.logger.error(f"Document-specific validation failed: {doc_error}")

                # Extract error_code from additional_other_checks if provided
                doc_error_code = additional_other_checks.pop('error_code', None)

                # Revert state if it was incremented (in user_keys for this device)
                if user_identity_id:
                    current_state = self.state_service.get_verification_state(client_public_key)
                    # Bank statement processing happens at state 2, so revert if state went beyond 2
                    if current_state > 2:
                        # Revert state in user_keys for this device
                        self.user_key_repo.update_state_and_sequence(
                            user_public_key=client_public_key,
                            verification_state=2,
                            sequence_no=2
                        )
                        self.logger.info(f"Reverted verification state after document-specific validation failure: {current_state} -> 2")

                current_state = self.state_service.get_verification_state(client_public_key)
                current_seq = self.state_service.get_sequence_no(client_public_key)
                return SequentialJobResponse(
                    result=False,
                    job_id=job_id,
                    verification_state=current_state,
                    sequence_no=current_seq,
                    processing_time_seconds=round(time.time() - start_time, 2),
                    extracted_data=extracted_data,
                    forgery_checks=forgery_checks,
                    other_checks={**name_match_details, **additional_other_checks},
                    error=doc_error,
                    error_code=doc_error_code
                )

            # ============================================================
            # PHASE 10: State increment (if applicable)
            # ============================================================
            current_state = self.state_service.get_verification_state(client_public_key)
            current_seq = self.state_service.get_sequence_no(client_public_key)
            new_state = current_state
            new_seq = current_seq

            if self.should_increment_state():
                # Only increment if at the expected state (not resubmission)
                expected_state = self.state_service.EXPECTED_STATE.get(document_type)
                # Handle both single values and lists of acceptable states
                expected_states = expected_state if isinstance(expected_state, list) else [expected_state]
                if current_state in expected_states:
                    # Determine new state based on document type
                    if document_type in ('passport', 'national_id', 'driving_license'):
                        new_state = 2
                        new_seq = 2
                    elif document_type == 'bank_statement':
                        new_state = 3
                        new_seq = 3
                    else:
                        # Keep current state for other document types
                        new_state = current_state
                        new_seq = current_seq

                    # Update state in BOTH user_keys (per-device) AND user_identity_index (overall)
                    self.user_key_repo.update_state_and_sequence(
                        user_public_key=client_public_key,
                        verification_state=new_state,
                        sequence_no=new_seq
                    )
                    # Update user_identity_index to match (use SET, not INCREMENT)
                    self.user_identity_repo.set_verification_state(user_identity_id, new_state)
                    self.user_identity_repo.set_sequence_no(user_identity_id, new_seq)
                    self.logger.info(
                        f"{document_type} processing completed. "
                        f"New state: {new_state}, sequence_no: {new_seq}"
                    )
                else:
                    self.logger.info(
                        f"{document_type} resubmission (state={current_state}). "
                        f"State unchanged."
                    )
            else:
                self.logger.info(
                    f"{document_type} is independent - no state change "
                    f"(current state: {current_state})"
                )

            # ============================================================
            # PHASE 11: Build successful response
            # ============================================================
            final_other_checks = {
                **name_match_details,
                **additional_other_checks,
                'is_resubmission': is_resubmission
            }

            return SequentialJobResponse(
                result=True,
                job_id=job_id,
                verification_state=new_state,
                sequence_no=new_seq,
                processing_time_seconds=round(time.time() - start_time, 2),
                extracted_data=extracted_data,
                forgery_checks=forgery_checks,
                other_checks=final_other_checks,
                user_identity_id=user_identity_id
            )

        except Exception as e:
            self.logger.error(f"Error processing {document_type}: {str(e)}")

            # Revert state if it was incremented (in user_keys for this device)
            if user_identity_id:
                current_state = self.state_service.get_verification_state(client_public_key)
                # Bank statement processing happens at state 2, so revert if state went beyond 2
                if current_state > 2:
                    # Revert state in user_keys for this device
                    self.user_key_repo.update_state_and_sequence(
                        user_public_key=client_public_key,
                        verification_state=2,
                        sequence_no=2
                    )
                    self.logger.info(f"Reverted verification state after failure: {current_state} -> 2")

            current_state = self.state_service.get_verification_state(client_public_key)
            current_seq = self.state_service.get_sequence_no(client_public_key)
            return SequentialJobResponse(
                result=False,
                job_id=job_id,
                verification_state=current_state,
                sequence_no=current_seq,
                processing_time_seconds=round(time.time() - start_time, 2),
                error=str(e),
                error_code=DocumentErrorCode.PROCESSING_ERROR
            )

    # ============================================================
    # COMMON VALIDATION METHODS
    # ============================================================

    def validate_name_matching(
        self,
        stored_full_name: str,
        extracted_data: Dict[str, Any]
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Validate name matching using Jaro-Winkler fuzzy matching.

        Args:
            stored_full_name: Name from passport (stored in database)
            extracted_data: Extracted data from current document

        Returns:
            (is_valid, score, details)
        """
        extracted_name = self.get_name_from_extracted_data(extracted_data)

        if not stored_full_name or not extracted_name:
            return False, 0.0, {
                'error': 'Missing name for comparison',
                'stored_full_name': stored_full_name,
                'extracted_name': extracted_name
            }

        # Use Jaro-Winkler for name matching
        score = round(fuzzy_match_names(stored_full_name, extracted_name) * 100, 1)
        details = get_match_details(stored_full_name, extracted_name)

        threshold = verification_settings.name_match_threshold  # Default: 70%
        is_valid = score >= threshold

        self.logger.info(
            f"Name matching for {self.get_document_type()}: "
            f"stored='{stored_full_name}' vs extracted='{extracted_name}' "
            f"-> score={score}% (threshold={threshold}%) - "
            f"{'PASS' if is_valid else 'FAIL'}"
        )

        details.update({
            'name_match_score': score,
            'stored_full_name': stored_full_name,
            'extracted_name': extracted_name,
            'name_match_threshold': threshold
        })

        return is_valid, score, details

    def validate_required_fields_with_confidence(
        self,
        extracted_data: Dict[str, Any],
        confidence_data: Dict[str, Any]
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Validate that all required fields are present and meet confidence thresholds.

        Args:
            extracted_data: Extracted field values
            confidence_data: Confidence scores for each field

        Returns:
            (is_valid, missing_fields, field_validations)
        """
        required_fields = self.get_required_fields()
        missing_fields = []
        field_validations = {}
        all_valid = True

        # Check if confidence validation is disabled
        # NOTE: This only skips the confidence threshold check, NOT the field presence check
        skip_confidence = getattr(verification_settings, 'skip_confidence_validation', False)

        # Get confidence threshold from settings
        confidence_threshold = getattr(
            verification_settings,
            'field_confidence_threshold',
            0.70  # Reverted to original threshold
        )

        # Use overall confidence for validation (matches working version caa04d4)
        # The overall_confidence is calculated as the average of all matched entities
        # and better represents extraction quality than individual field confidences
        overall_confidence = confidence_data.get('overall', {}).get('overall_confidence', 0.0)
        use_overall_for_validation = overall_confidence > 0

        for field in required_fields:
            value = extracted_data.get(field)

            # Use overall confidence for validation if available
            # Otherwise fall back to per-field confidence
            if use_overall_for_validation:
                confidence = overall_confidence
            else:
                confidence_info = confidence_data.get(field, {})
                confidence = confidence_info.get('overall_confidence', 0.0) if confidence_info else 0.0

            # Always check field presence
            is_present = value is not None and str(value).strip() != ''

            # Check confidence only if not skipped
            is_confident = skip_confidence or (confidence >= confidence_threshold)

            field_validations[field] = {
                'value': value,
                'confidence': confidence,
                'is_present': is_present,
                'is_confident': is_confident,
                'status': 'PASS' if (is_present and is_confident) else 'FAIL'
            }

            if not is_present:
                missing_fields.append(f"{field} (missing)")
                all_valid = False
            elif not is_confident:
                missing_fields.append(f"{field} (low confidence: {confidence:.0%})")
                all_valid = False

        return all_valid, missing_fields, field_validations

    # ============================================================
    # UTILITY METHODS
    # ============================================================

    def _build_error_response(
        self,
        job_id: str,
        start_time: float,
        verification_state: int,
        sequence_no: int,
        error_message: str,
        error_code: Optional[str] = None,
        extracted_data: Optional[Dict[str, Any]] = None,
        raw_text: Optional[str] = None
    ) -> SequentialJobResponse:
        """Build a standard error response.

        Args:
            job_id: Unique job identifier
            start_time: Processing start time
            verification_state: Current verification state
            sequence_no: Current sequence number
            error_message: Error description
            error_code: Optional error code
            extracted_data: Optional extracted data dict (will be included in response)
            raw_text: Optional raw OCR text (will be added to extracted_data as raw_data)
        """
        # Include raw_data if OCR has completed and raw_text is provided
        if raw_text and 'raw_data' not in (extracted_data or {}):
            if extracted_data is None:
                extracted_data = {}
            extracted_data['raw_data'] = raw_text

        response = SequentialJobResponse(
            result=False,
            job_id=job_id,
            verification_state=verification_state,
            sequence_no=sequence_no,
            processing_time_seconds=round(time.time() - start_time, 2),
            error=error_message,
            error_code=error_code or DocumentErrorCode.PROCESSING_ERROR
        )

        # Only add extracted_data if it exists (may be None for early failures)
        if extracted_data:
            response.extracted_data = extracted_data

        return response

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

    def _parse_date(self, date_str: Optional[str]) -> Optional[date]:
        """Parse date string to date object."""
        if not date_str:
            return None

        try:
            date_clean = str(date_str).strip()

            date_formats = [
                "%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d", "%d/%m/%Y", "%Y%m%d",
                "%d %b %Y",   # 30 NOV 2025
                "%d %B %Y",   # 30 November 2025
                "%d-%b-%Y",   # 30-Nov-2025
                "%d/%b/%Y",   # 30/Nov/2025
            ]

            for date_variant in [date_clean, date_clean.title()]:
                for fmt in date_formats:
                    try:
                        return datetime.strptime(date_variant, fmt).date()
                    except ValueError:
                        continue

            return None
        except Exception:
            return None

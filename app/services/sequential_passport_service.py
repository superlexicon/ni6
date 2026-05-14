"""
Sequential Passport Service - Handles passport processing in sequential mode.

State is tracked via verification_state column in user_identity_index.

PIPELINE (v2.0 - Dynamic Region Exclusion):
1. PhotoHolmes authenticity checks
2. DocTR OCR extraction
3. Process OCR + Remove user text regions (values, MRZ)
4. Face extraction + matching + face region removal
5. Reference passport comparison (on cleaned image)
6. PEP/Criminal database checks
7. Web search + sentiment analysis
"""

from typing import Dict, Any, Optional, List, Tuple
import asyncio
import time
import base64
import uuid
import io
from datetime import datetime, date
import numpy as np
from PIL import Image

from app.dto.verification_session import SequentialJobResponse
from app.services.verification_state_service import VerificationStateService
from app.services import comprehensive_photoholmes_service
from app.services.face_extraction_service import FaceExtractionService
from app.services.detailed_analysis_service import DetailedAnalysisService
from app.core.logger import get_logger
from app.helper.deepface_helper import DeepfaceHelper
from app.helper.extractors import PassportExtractor
from app.helper.extractors.unified_id_extractor import UnifiedIDExtractor
from app.config.verification_config import verification_settings
from app.repositories.user_key_repository import UserKeyRepository
from app.repositories.user_identity_repository import UserIdentityRepository
from app.repositories.face_biometrics_repository import FaceBiometricsRepository
from app.services.osint_screening_service import osint_screening_service
from app.services.worldcheck_service import worldcheck_service
from app.config.osint_config import osint_settings
from app.dto import DocumentErrorCode


class SequentialPassportService:
    """Service for handling passport processing in sequential mode"""

    # Maximum number of passport resubmissions allowed
    MAX_PASSPORT_RESUBMISSIONS = 3

    def __init__(self):
        self.logger = get_logger()
        self.face_extraction_service = FaceExtractionService()
        self.state_service = VerificationStateService()
        self.user_key_repo = UserKeyRepository()
        self.user_identity_repo = UserIdentityRepository()
        self.face_biometrics_repo = FaceBiometricsRepository()
        self.passport_extractor = PassportExtractor()
        self.unified_extractor = UnifiedIDExtractor()
        self.detailed_analysis_service = DetailedAnalysisService()

    # =========================================================================
    # Image Manipulation Helpers for Dynamic Region Exclusion
    # =========================================================================

    def _decode_image_to_numpy(self, image_bytes: bytes) -> np.ndarray:
        """Decode image bytes to numpy array (RGB)."""
        with Image.open(io.BytesIO(image_bytes)) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return np.array(img)

    def _encode_numpy_to_bytes(self, image_np: np.ndarray) -> bytes:
        """Encode numpy array back to image bytes (PNG)."""
        img = Image.fromarray(image_np.astype('uint8'), 'RGB')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()

    def _normalize_bbox(self, bbox: List, image_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
        """
        Normalize bbox to pixel coordinates.

        Args:
            bbox: [x1, y1, x2, y2] either normalized (0-1) or pixel coordinates
            image_shape: (height, width, channels)

        Returns:
            (x1, y1, x2, y2) in pixel coordinates, clamped to image bounds
        """
        height, width = image_shape[:2]

        if len(bbox) != 4:
            return (0, 0, width, height)

        x1, y1, x2, y2 = bbox

        # If coordinates are normalized (0-1), convert to pixels
        if all(0 <= c <= 1 for c in [x1, y1, x2, y2]):
            x1 = int(x1 * width)
            y1 = int(y1 * height)
            x2 = int(x2 * width)
            y2 = int(y2 * height)
        else:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Clamp to image bounds
        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))

        return (x1, y1, x2, y2)

    def _remove_regions_from_image(
        self,
        image_np: np.ndarray,
        regions: List[Tuple[int, int, int, int]],
        fill_color: Tuple[int, int, int] = (0, 0, 0)
    ) -> np.ndarray:
        """
        Remove regions from image by filling with solid color.

        Args:
            image_np: Image as numpy array (H, W, C)
            regions: List of (x1, y1, x2, y2) pixel coordinates
            fill_color: RGB color to fill removed regions with

        Returns:
            Modified image copy with regions removed
        """
        result = image_np.copy()

        for x1, y1, x2, y2 in regions:
            # Ensure coordinates are in correct order
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            result[y1:y2, x1:x2] = fill_color

        return result

    def _combine_regions(
        self,
        regions: List[Tuple[int, int, int, int]],
        padding: int = 5
    ) -> Tuple[int, int, int, int]:
        """
        Combine multiple regions into a single bounding box.

        Args:
            regions: List of (x1, y1, x2, y2) pixel coordinates
            padding: Extra padding around combined region

        Returns:
            Combined (x1, y1, x2, y2) with padding
        """
        if not regions:
            return (0, 0, 0, 0)

        min_x = min(r[0] for r in regions) - padding
        min_y = min(r[1] for r in regions) - padding
        max_x = max(r[2] for r in regions) + padding
        max_y = max(r[3] for r in regions) + padding

        return (max(0, min_x), max(0, min_y), max_x, max_y)

    def _find_mrz_regions(self, ocr_blocks: List[Dict], image_shape: Tuple[int, int, int]) -> List[Tuple[int, int, int, int]]:
        """
        Find MRZ (Machine Readable Zone) regions from OCR blocks.

        MRZ is identified by rows containing "<<<" pattern.

        Args:
            ocr_blocks: List of OCR blocks with 'text' and 'bbox' keys
            image_shape: (height, width, channels)

        Returns:
            List of MRZ region bounding boxes in pixel coordinates
        """
        mrz_rows = []
        for block in ocr_blocks:
            text = block.get('text', '')
            if '<<<' in text:
                bbox = block.get('bbox', [])
                if len(bbox) == 4:
                    mrz_rows.append(self._normalize_bbox(bbox, image_shape))

        return mrz_rows

    async def process_passport(self, client_public_key: str, file_data: str, filename: str,
                             callback_url: Optional[str] = None,
                             document_type: str = "passport") -> SequentialJobResponse:
        """
        Process an identity document (passport or ID card) in sequential mode.

        Args:
            document_type: Type of document ("passport" or "id_card")

        Returns simplified SequentialJobResponse with:
        - result: bool (all checks passed)
        - verification_state: int (new state after this step)
        - extracted_data: document fields
        - forgery_checks: PhotoHolmes results
        - other_checks: face_match_confidence, document_expiry_valid (if applicable)
        """
        job_id = f"{document_type}_{uuid.uuid4().hex[:12]}"
        start_time = time.time()
        user_identity_id = None

        # Validate state - must be state 1 (selfie done) or resubmission (state >= 2)
        is_valid, error_msg, is_resubmission = self.state_service.validate_document_submission(client_public_key, 'passport')
        if not is_valid:
            self.logger.error(f"State validation failed: {error_msg}")
            current_state = self.state_service.get_verification_state(client_public_key)
            current_seq = self.state_service.get_sequence_no(client_public_key)
            return SequentialJobResponse(
                result=False,
                job_id=job_id,
                verification_state=current_state,
                sequence_no=current_seq,
                processing_time_seconds=round(time.time() - start_time, 2),
                error=error_msg,
                error_code=DocumentErrorCode.PROCESSING_ERROR
            )

        if is_resubmission:
            self.logger.info(f"Processing passport resubmission for: {client_public_key[:16]}...")

        try:
            # Get user_identity_id
            user_key = self.user_key_repo.get_key_by_public_key(client_public_key)
            user_identity_id = user_key['user_identity_id']
            self.logger.info(f"Processing passport for user_identity: {user_identity_id}")

            # Decode image
            image_bytes = base64.b64decode(file_data)
            is_pdf = filename.lower().endswith('.pdf') or image_bytes.startswith(b'%PDF')

            # PHASE 1, 2, 3: Run PhotoHolmes + OCR + Face extraction in PARALLEL for performance
            self.logger.info("Running PhotoHolmes, OCR, and face extraction in parallel...")

            async def safe_face_extraction():
                """Wrapper to catch face extraction errors"""
                try:
                    return await self.face_extraction_service.extract_face_embedding(
                        image_bytes=image_bytes,
                        public_key=client_public_key,
                        user_identity_id=user_identity_id,
                        document_type="passport"
                    )
                except Exception as e:
                    self.logger.warning(f"Face extraction failed: {e}")
                    return None

            # Run all three tasks in parallel
            photoholmes_results, document_data, face_biometric = await asyncio.gather(
                comprehensive_photoholmes_service.run_all_methods(image_bytes, document_type=document_type),
                self.unified_extractor.extract(image_bytes, is_pdf),
                safe_face_extraction()
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
                from app.services.selfie_validation_service import SelfieValidationService
                validation_service = SelfieValidationService()
                photoholmes_valid, photoholmes_error, photoholmes_error_code = validation_service.validate_photoholmes_results(detailed_results)

                # Build extracted_data early for PhotoHolmes failure case
                temp_extracted_data = self._build_extracted_data(document_data, document_type)

                if not photoholmes_valid:
                    # Check if we have extracted data (key fields present)
                    country_field = 'document_country' if document_type == 'id_card' else 'passport_country'
                    number_field = 'number'

                    has_extracted_data = (
                        temp_extracted_data.get(country_field) and
                        temp_extracted_data.get(number_field)
                    )

                    if has_extracted_data:
                        # Data extracted successfully but forgery checks failed
                        # Increment sequence_no to 2 so bank statement can be submitted
                        current_seq = self.state_service.get_sequence_no(client_public_key)
                        if current_seq < 2:
                            self.state_service.increment_sequence_no(user_identity_id)

                        # Build other_checks for response
                        temp_other_checks = {
                            "document_type": document_type,
                            "is_resubmission": is_resubmission
                        }

                        return SequentialJobResponse(
                            result=False,
                            job_id=job_id,
                            verification_state=1,
                            sequence_no=2,
                            processing_time_seconds=round(time.time() - start_time, 2),
                            error=photoholmes_error,
                            error_code=photoholmes_error_code,
                            extracted_data=temp_extracted_data,  # Return extracted data!
                            forgery_checks=forgery_checks,
                            other_checks=temp_other_checks
                        )
                    else:
                        # Data extraction failed - outright rejection, no increment
                        return SequentialJobResponse(
                            result=False,
                            job_id=job_id,
                            verification_state=1,
                            sequence_no=1,
                            processing_time_seconds=round(time.time() - start_time, 2),
                            error=photoholmes_error,
                            error_code=photoholmes_error_code,
                            forgery_checks=forgery_checks,
                            extracted_data=temp_extracted_data  # Include raw_data from OCR
                        )

            # Process OCR results - Build extracted_data
            extracted_data = self._build_extracted_data(document_data, document_type)

            # Validate required fields
            country_field = 'document_country' if document_type == 'id_card' else 'passport_country'
            number_field = 'number'  # Unified extractor uses 'number' for both

            if not extracted_data.get(country_field) or not extracted_data.get(number_field):
                doc_type_label = "ID card" if document_type == "id_card" else "passport"
                orientation_error = (
                    f"Document appears to be rotated. Please ensure the {doc_type_label} is "
                    f"upright (right-side up) before submitting. "
                    f"Critical fields could not be extracted."
                )
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
                    error=orientation_error,
                    error_code=DocumentErrorCode.OCR_FAILED
                )

            # Process face extraction results
            passport_face_embedding = None
            if face_biometric:
                if isinstance(face_biometric, dict):
                    passport_face_embedding = face_biometric.get('face_embedding')
                elif hasattr(face_biometric, 'face_embedding'):
                    passport_face_embedding = face_biometric.face_embedding

            # PHASE 4: Get selfie face embedding
            selfie_face_embedding = None
            selfie_embeddings = self.face_biometrics_repo.get_embeddings_by_user_identity(user_identity_id)
            if selfie_embeddings:
                selfie_face_embedding = selfie_embeddings[0]

            # PHASE 5: Face matching
            face_match_confidence = None
            if selfie_face_embedding and passport_face_embedding:
                face_match_result = await self._perform_face_matching(selfie_face_embedding, passport_face_embedding)
                face_match_confidence = face_match_result.get("face_match_confidence")

            # PHASE 6: Check document expiry (only for passports - some ID cards have no expiry)
            document_expiry_valid = None
            if document_type == "passport":
                document_expiry_valid = self._check_passport_expiry(extracted_data.get('date_of_expiry'))
            elif document_type == "id_card":
                # For ID cards, expiry is optional (e.g., Singapore NRIC has no expiry)
                # If expiry is present, validate it; otherwise, consider it valid
                expiry_date = extracted_data.get('date_of_expiry')
                if expiry_date:
                    document_expiry_valid = self._check_passport_expiry(expiry_date)
                else:
                    document_expiry_valid = True  # No expiry = valid for life

            # PHASE 6.1: Check minimum age (18 years) for passport holders
            minimum_age_valid = None
            calculated_age = None
            if document_type == "passport":
                minimum_age_valid, calculated_age = self._check_minimum_age(
                    extracted_data.get('dob') or extracted_data.get('date_of_birth'),
                    min_age=18
                )
                if not minimum_age_valid and calculated_age is not None:
                    self.logger.warning(
                        f"Passport holder is underage: {calculated_age} years old (minimum: 18)"
                    )

            # Build other_checks
            other_checks = {
                "face_match_confidence": face_match_confidence,
                "document_type": document_type,
                "document_expiry_valid": document_expiry_valid,
                "minimum_age_valid": minimum_age_valid,
                "age": calculated_age,
                "is_resubmission": is_resubmission
            }

            # PHASE 6.5: Screening (OSINT always, World-Check if API key configured)
            # This happens AFTER all validations pass but BEFORE state increment
            screening_name = extracted_data.get('full_name')
            # Mask screening name for logs (PII protection)
            masked_screening_name = f"{screening_name.split(' ')[0][0]}*** {screening_name.split(' ')[-1][0]}***" if screening_name and len(screening_name.split()) >= 2 else "*** ***"
            if len(masked_screening_name) > 30:  # Truncate if too long
                masked_screening_name = masked_screening_name[:30] + "..."

            # Normalize date of birth to ISO format for World Check (YYYY-MM-DD)
            # World Check requires ISO 8601 format for dates
            dob_date_obj = self._parse_date(extracted_data.get('date_of_birth'))
            screening_dob = dob_date_obj.strftime('%Y-%m-%d') if dob_date_obj else None

            # Always run OSINT screening
            self.logger.info(f"Performing OSINT screening for: {masked_screening_name}")
            osint_result = await osint_screening_service.screen_individual(
                full_name=screening_name,
                date_of_birth=screening_dob,
                country=extracted_data.get('passport_country'),
                gender=extracted_data.get('sex'),
                address=None,  # Not available at passport step
                user_identity_id=user_identity_id
            )

            # Store OSINT result
            self.user_identity_repo.update_osint_result(user_identity_id, osint_result)

            # Check if World-Check is available and should run for this key
            worldcheck_available = self._is_worldcheck_available()
            should_run_worldcheck = self._should_run_worldcheck_for_key(client_public_key)
            osint_failed = False
            worldcheck_failed = False
            worldcheck_result = None

            # Check OSINT risk score
            if osint_result.get('overall_risk_score', 0) >= osint_settings.risk_threshold:
                osint_failed = True
                self.logger.warning(f"OSINT risk score exceeds threshold: {osint_result['overall_risk_score']}")

            # Run World-Check if available and key is registered with Fraxn API
            if worldcheck_available and should_run_worldcheck:
                self.logger.info(f"World-Check API key configured, performing World-Check screening for: {masked_screening_name}")
                # Convert country code to ISO 3166-1 alpha-3 format required by World Check
                from app.utils.country_code_converter import convert_to_alpha3
                country_alpha3 = convert_to_alpha3(extracted_data.get('passport_country'))
                self.logger.info(f"Country code for World Check: {extracted_data.get('passport_country')} -> {country_alpha3}")
                worldcheck_result = await worldcheck_service.screen_individual(
                    full_name=screening_name,
                    date_of_birth=screening_dob,
                    country=country_alpha3,
                    gender=extracted_data.get('sex')
                )

                # Store World Check result
                self.user_identity_repo.update_worldcheck_result(user_identity_id, worldcheck_result)

                if worldcheck_result.get('error'):
                    self.logger.error(f"World-Check error: {worldcheck_result.get('error')}")
                elif worldcheck_result.get('is_match'):
                    worldcheck_failed = True
                    self.logger.warning(f"World-Check match found for: {screening_name}")
            else:
                # Log why World-Check was skipped
                if not worldcheck_available:
                    self.logger.info(f"World-Check not performed: API key not configured")
                elif not should_run_worldcheck:
                    self.logger.info(f"World-Check not performed: Public key not registered with Fraxn API")

            # Apply decision logic: FAIL if either failed
            if osint_failed or worldcheck_failed:
                failure_reasons = []
                if osint_failed:
                    failure_reasons.append(f"OSINT risk score: {osint_result['overall_risk_score']:.1f} ({osint_result['risk_category']})")
                if worldcheck_failed:
                    failure_reasons.append("World-Check watchlist match")

                # Add screening results to other_checks
                other_checks.update({
                    "osint_risk_score": osint_result.get('overall_risk_score', 0),
                    "osint_risk_category": osint_result.get('risk_category', 'UNKNOWN'),
                    "osint_result": osint_result.get('result', 'PASS'),
                    "worldcheck_match": worldcheck_result.get('is_match', False) if worldcheck_result else False,
                    "worldcheck_available": worldcheck_available
                })

                # NO STATE REVERSION - leave current state unchanged on failure

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
                    other_checks=other_checks,
                    message=f"Screening failed: {', '.join(failure_reasons)}"
                )

            # Add screening results to other_checks for successful case
            other_checks.update({
                "osint_risk_score": osint_result.get('overall_risk_score', 0),
                "osint_risk_category": osint_result.get('risk_category', 'UNKNOWN'),
                "osint_result": osint_result.get('result', 'PASS'),
                "worldcheck_match": worldcheck_result.get('is_match', False) if worldcheck_result else False,
                "worldcheck_available": worldcheck_available
            })

            # Validate face matching
            face_match_threshold = verification_settings.face_match_threshold
            if face_match_confidence is None:
                # NO STATE REVERSION - leave current state unchanged on failure

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
                    other_checks=other_checks,
                    error="Face matching failed - no confidence score",
                    error_code=DocumentErrorCode.PROCESSING_ERROR
                )

            if face_match_confidence < face_match_threshold:
                # NO STATE REVERSION - leave current state unchanged on failure

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
                    other_checks=other_checks,
                    error=f"Face match confidence {face_match_confidence}% below threshold {face_match_threshold}%",
                    error_code=DocumentErrorCode.LOGICAL_NAME_MISMATCH
                )

            # Validate document expiry (only for passports)
            if document_type == "passport" and not document_expiry_valid:
                # NO STATE REVERSION - leave current state unchanged on failure

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
                    other_checks=other_checks,
                    error="Document has expired or has insufficient validity",
                    error_code=DocumentErrorCode.LOGICAL_EXTRACTION_INCOMPLETE
                )

            # Validate minimum age (only for passports - must be 18+)
            if document_type == "passport" and minimum_age_valid is False:
                # NO STATE REVERSION - leave current state unchanged on failure

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
                    other_checks=other_checks,
                    error=f"Passport holder is underage: {calculated_age} years old (minimum: 18)",
                    error_code=DocumentErrorCode.LOGICAL_EXTRACTION_INCOMPLETE
                )

            # PHASE 7: Update user_identity_index with document data
            # This now works for BOTH first submission and resubmission
            try:
                update_success = self._update_passport_data(
                    user_identity_id=user_identity_id,
                    extracted_data=extracted_data,
                    osint_result=osint_result,
                    worldcheck_result=worldcheck_result,
                    client_public_key=client_public_key
                )

                if not update_success:
                    doc_type_label = "ID card" if document_type == "id_card" else "passport"
                    self.logger.error(f"Failed to update user_identity with passport data for {user_identity_id}")
                    current_state = self.state_service.get_verification_state(client_public_key)
                    current_seq = self.state_service.get_sequence_no(client_public_key)
                    return SequentialJobResponse(
                        result=False,
                        job_id=job_id,
                        message=f"Failed to update {doc_type_label} data in database",
                        error=DocumentErrorCode.PROCESSING_ERROR,
                        verification_state=current_state,
                        sequence_no=current_seq,
                        processing_time_seconds=round(time.time() - start_time, 2),
                        extracted_data=extracted_data,
                        forgery_checks=forgery_checks,
                        other_checks=other_checks
                    )

                self.logger.info(f"Successfully updated passport data for {user_identity_id}")
            except ValueError as e:
                # This catches duplicate passport errors
                self.logger.error(f"ValueError caught in passport processing: {e}")
                doc_type_label = "ID card" if document_type == "id_card" else "passport"
                current_state = self.state_service.get_verification_state(client_public_key)
                current_seq = self.state_service.get_sequence_no(client_public_key)
                return SequentialJobResponse(
                    result=False,
                    job_id=job_id,
                    message=str(e),  # Duplicate document error
                    error=DocumentErrorCode.PROCESSING_ERROR,
                    verification_state=current_state,
                    sequence_no=current_seq,
                    processing_time_seconds=round(time.time() - start_time, 2),
                    extracted_data=extracted_data,
                    forgery_checks=forgery_checks,
                    other_checks=other_checks
                )

            # Increment verification state (1 -> 2) only if currently at state 1
            # IMPORTANT: Only passport increments state - ID cards are independent (no state change)
            # For resubmissions (state >= 2), don't change the state
            current_state = self.state_service.get_verification_state(client_public_key)
            current_seq = self.state_service.get_sequence_no(client_public_key)
            doc_type_label = "ID card" if document_type == "id_card" else "passport"
            new_seq = current_seq

            self.logger.info(f"PHASE 8: Before state increment. current_state={current_state}, current_seq={current_seq}")
            self.logger.info(f"Before increment - current_state: {current_state}, current_seq: {current_seq}, document_type: {document_type}")

            if current_state == 1 and document_type == "passport":  # Only passport increments state
                self.logger.info(f"Setting state from 1 to 2 for user: {user_identity_id}")
                new_state = 2
                new_seq = 2
                # Update state in BOTH user_keys (per-device) AND user_identity_index (overall)
                self.user_key_repo.update_state_and_sequence(
                    user_public_key=client_public_key,
                    verification_state=new_state,
                    sequence_no=new_seq
                )
                # Update user_identity_index to match (use SET, not INCREMENT)
                self.user_identity_repo.set_verification_state(user_identity_id, new_state)
                self.user_identity_repo.set_sequence_no(user_identity_id, new_seq)
                self.logger.info(f"{doc_type_label.capitalize()} processing completed. New state: {new_state}, sequence_no: {new_seq}")
                message = f"{doc_type_label.capitalize()} processed successfully. Please submit bank statement next."
            elif document_type == "id_card":
                # ID card is independent - no state change
                new_state = current_state
                self.logger.info(f"{doc_type_label.capitalize()} processed successfully. No state change (current state: {new_state}).")
                message = f"{doc_type_label.capitalize()} processed successfully."
            else:
                new_state = current_state
                if current_state < 1:
                    self.logger.error(f"State is {current_state}, expected 1. Selfie step may not have completed properly.")
                elif current_state >= 2:
                    self.logger.info(f"{doc_type_label.capitalize()} resubmission or already complete (state: {current_state}).")
                message = f"{doc_type_label.capitalize()} updated successfully."

            return SequentialJobResponse(
                result=True,
                job_id=job_id,
                verification_state=new_state,
                sequence_no=new_seq,
                processing_time_seconds=round(time.time() - start_time, 2),
                extracted_data=extracted_data,
                forgery_checks=forgery_checks,
                other_checks=other_checks,
                user_identity_id=user_identity_id
            )

        except Exception as e:
            self.logger.error(f"Error processing {document_type}: {str(e)}")

            # NO STATE REVERSION - leave current state unchanged on exception

            current_seq = self.state_service.get_sequence_no(client_public_key)
            current_state = self.state_service.get_verification_state(client_public_key)
            return SequentialJobResponse(
                result=False,
                job_id=job_id,
                verification_state=current_state,
                sequence_no=current_seq,
                processing_time_seconds=round(time.time() - start_time, 2),
                error=str(e),
                error_code=DocumentErrorCode.PROCESSING_ERROR
            )

    def _update_passport_data(
        self,
        user_identity_id: str,
        extracted_data: Dict[str, Any],
        osint_result: Dict[str, Any],
        worldcheck_result: Optional[Dict[str, Any]],
        client_public_key: str
    ) -> bool:
        """
        Update passport data in database (works for both first submission and resubmission).

        Args:
            user_identity_id: User identity ID
            extracted_data: Extracted passport data
            osint_result: OSINT screening result
            worldcheck_result: World-Check screening result (optional)
            client_public_key: Client's public key for encryption

        Returns:
            True if updated successfully

        Raises:
            ValueError: If passport data is invalid
        """
        from app.utils.country_code_converter import convert_to_alpha3

        # Map fields based on document type
        country = extracted_data.get('document_country') or extracted_data.get('passport_country')
        number = extracted_data.get('number') or extracted_data.get('passport_number')

        if not country or not number:
            raise ValueError("Missing passport country or number")

        # Get current user data to check if this is a passport update
        user_identity = self.user_identity_repo.get_user_by_id(user_identity_id)
        if not user_identity:
            raise ValueError(f"User identity {user_identity_id} not found")

        existing_passport_expiry = user_identity.get('passport_expiry_date')
        is_passport_update = existing_passport_expiry is not None

        # Note: passport_hash removed - uniqueness enforced by face biometrics trigger.
        # Same passport can be re-encrypted for different device keys (multi-device support).

        # Update passport data
        update_success = self.user_identity_repo.update_with_passport_data(
            user_identity_id=user_identity_id,
            passport_country=country,
            passport_number=number,
            user_public_key=client_public_key,
            full_name=extracted_data.get('full_name'),
            date_of_birth=self._parse_date(extracted_data.get('dob')),
            gender=extracted_data.get('sex'),
            passport_expiry_date=self._parse_date(extracted_data.get('date_of_expiry'))
        )

        if not update_success:
            self.logger.error(f"Failed to update passport data for {user_identity_id}")
            return False

        # Update OSINT result
        self.user_identity_repo.update_osint_result(user_identity_id, osint_result)

        # Update World-Check result if available
        if worldcheck_result:
            self.user_identity_repo.update_worldcheck_result(user_identity_id, worldcheck_result)

        # Log with distinction between new passport and passport update
        country_alpha3 = convert_to_alpha3(country)
        action = "Passport updated" if is_passport_update else "Passport created"
        self.logger.info(
            f"{action} for user {user_identity_id[:16]}... from device {client_public_key[:16]}... "
            f"({country_alpha3}/{number[:4]}****)"
        )
        return True

    def _build_extracted_data(self, document_data, document_type: str = "passport") -> Dict[str, Any]:
        """
        Extract key fields from identity document data.

        Handles both UnifiedIDExtractor output (new) and PassportExtractor output (legacy).
        All dates are normalized to ISO format (YYYY-MM-DD).
        """
        if not document_data:
            return {}

        # Import the name cleaning utility
        from app.utils.string_matching import clean_name_for_storage

        # Check if this is a UnifiedIDExtractor result (DocumentExtractionResult)
        if hasattr(document_data, 'document_type'):
            # UnifiedIDExtractor result - use its fields directly
            raw_full_name = document_data.get('full_name')
            cleaned_full_name = clean_name_for_storage(raw_full_name)

            # Get date of birth for age calculation and normalize to ISO
            dob_str = document_data.get('dob')
            dob_date = self._parse_date(dob_str)
            age = self._calculate_age(dob_date)
            dob_normalized = self._normalize_date_to_iso(dob_str)

            # Normalize expiry date to ISO
            expiry_str = document_data.get('expiry')
            expiry_normalized = self._normalize_date_to_iso(expiry_str)

            # Normalize issue date to ISO
            issue_str = document_data.get('issue_date')
            issue_normalized = self._normalize_date_to_iso(issue_str)

            result = {
                "document_type": document_data.document_type,
                "country_code": document_data.country_code,
                "number": document_data.get('number'),
                "full_name": cleaned_full_name,
                "dob": dob_normalized,  # ISO format (YYYY-MM-DD)
                "age": age,  # Calculated age in years
                "sex": document_data.get('sex'),
                "expiry": expiry_normalized,  # ISO format (YYYY-MM-DD)
                "place_of_birth": document_data.get('place_of_birth'),
                "issuing_authority": document_data.get('issuing_authority'),
                "issuing_country": document_data.get('issuing_country'),
                "address": document_data.get('address'),
                "date_of_issue": issue_normalized,  # ISO format (YYYY-MM-DD)
                "nrc_number": document_data.get('nrc_number'),
                "raw_data": getattr(document_data, 'raw_data', None),
            }

            # Add backward-compatible field names for passport processing
            if document_type == "passport":
                result["passport_number"] = document_data.get('number')
                result["passport_country"] = document_data.country_code
                result["nationality"] = document_data.get('issuing_country') or document_data.country_code

            # Filter out None values
            return {k: v for k, v in result.items() if v is not None}

        # Legacy PassportExtractor result (PassportData or similar)
        raw_full_name = getattr(document_data, 'full_name', None)
        cleaned_full_name = clean_name_for_storage(raw_full_name)

        # Get date of birth for age calculation and normalize to ISO
        dob_str = getattr(document_data, 'date_of_birth', None) or getattr(document_data, 'dob', None)
        dob_date = self._parse_date(dob_str)
        age = self._calculate_age(dob_date)
        dob_normalized = self._normalize_date_to_iso(dob_str) or dob_str

        # Normalize expiry date to ISO
        expiry_str = getattr(document_data, 'date_of_expiry', None) or getattr(document_data, 'expiry', None)
        expiry_normalized = self._normalize_date_to_iso(expiry_str) or expiry_str

        # Normalize issue date to ISO
        issue_str = getattr(document_data, 'date_of_issue', None)
        issue_normalized = self._normalize_date_to_iso(issue_str) or issue_str

        return {
            "full_name": cleaned_full_name,
            "dob": dob_normalized,  # ISO format (YYYY-MM-DD)
            "age": age,  # Calculated age in years
            "sex": getattr(document_data, 'sex', None),
            "passport_number": getattr(document_data, 'passport_number', None) or getattr(document_data, 'number', None),
            "number": getattr(document_data, 'number', None) or getattr(document_data, 'passport_number', None),
            "passport_country": getattr(document_data, 'passport_country', None) or getattr(document_data, 'nationality', None),
            "document_country": getattr(document_data, 'document_country', None) or getattr(document_data, 'passport_country', None),
            "date_of_issue": issue_normalized,  # ISO format (YYYY-MM-DD)
            "expiry": expiry_normalized,  # ISO format (YYYY-MM-DD)
            "nationality": getattr(document_data, 'nationality', None),
            "place_of_birth": getattr(document_data, 'place_of_birth', None),
            "issuing_authority": getattr(document_data, 'issuing_authority', None),
            "address": getattr(document_data, 'address', None),
            "raw_data": getattr(document_data, 'raw_data', None),
        }

    def _check_passport_expiry(self, expiry_date_str: Optional[str]) -> bool:
        """Check if passport has minimum required validity."""
        if not expiry_date_str:
            return False

        expiry_date = self._parse_date(expiry_date_str)
        if not expiry_date:
            return False

        # Minimum 6 months validity
        min_validity_days = getattr(verification_settings, 'passport_min_validity_days', 180)
        days_until_expiry = (expiry_date - date.today()).days

        return days_until_expiry >= min_validity_days

    def _check_minimum_age(self, dob_str: Optional[str], min_age: int = 18) -> Tuple[bool, Optional[int]]:
        """
        Check if the person meets minimum age requirement.

        Args:
            dob_str: Date of birth string
            min_age: Minimum required age (default: 18)

        Returns:
            Tuple of (is_valid, calculated_age)
        """
        if not dob_str:
            return False, None

        dob_date = self._parse_date(dob_str)
        if not dob_date:
            return False, None

        age = self._calculate_age(dob_date)
        if age is None:
            return False, None

        return age >= min_age, age

    def _parse_date(self, date_str: Optional[str]) -> Optional[date]:
        """Parse date string to date object."""
        if not date_str:
            return None

        date_clean = str(date_str).strip()

        # Handle DDMMM YYYY format (e.g., "11AUG 1996") using regex
        # This format is common in passport OCR where spacing is tight
        import re
        from app.utils.date_extractor import MONTH_MAP

        ddmymm_pattern = re.compile(r'^(\d{1,2})\s*([A-Z]{3})\s*(\d{4})$')
        match = ddmymm_pattern.match(date_clean.upper())
        if match:
            day_str, month_name, year_str = match.groups()
            month = MONTH_MAP.get(month_name)
            if month:
                try:
                    return date(int(year_str), month, int(day_str))
                except ValueError:
                    pass  # Fall through to standard parsing

        date_formats = [
            "%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d", "%d/%m/%Y", "%Y%m%d",
            "%d %b %Y",   # 04 Apr 2033
            "%d %B %Y",   # 04 April 2033
            "%d-%b-%Y",   # 04-Apr-2033
            "%d/%b/%Y",   # 04/Apr/2033
        ]

        # Try original, then title case (for month names like "Apr")
        for date_variant in [date_clean, date_clean.title()]:
            for fmt in date_formats:
                try:
                    return datetime.strptime(date_variant, fmt).date()
                except ValueError:
                    continue

        return None

    def _calculate_age(self, dob: Optional[date]) -> Optional[int]:
        """
        Calculate age from date of birth.

        Args:
            dob: Date of birth as a date object

        Returns:
            Age in years, or None if dob is invalid
        """
        if not dob:
            return None

        today = date.today()
        age = today.year - dob.year

        # Adjust if birthday hasn't occurred yet this year
        if (today.month, today.day) < (dob.month, dob.day):
            age -= 1

        return age

    def _normalize_date_to_iso(self, date_str: Optional[str]) -> Optional[str]:
        """
        Normalize a date string to ISO format (YYYY-MM-DD).

        Args:
            date_str: Date string in any supported format

        Returns:
            Date in ISO format (YYYY-MM-DD) or None if parsing fails
        """
        parsed_date = self._parse_date(date_str)
        if parsed_date:
            return parsed_date.strftime('%Y-%m-%d')
        return None

    async def _perform_face_matching(self, selfie_embedding: list, passport_embedding: list) -> Dict[str, Any]:
        """Perform face matching between embeddings."""
        try:
            match_result = await DeepfaceHelper.compare_face_embeddings(
                selfie_embedding,
                passport_embedding,
                model_name="Facenet512",
                use_ensemble=False
            )

            confidence_percentage = round(match_result.match_confidence * 100, 2)
            self.logger.info(f"✅ Face match: passport ↔ selfie confidence={confidence_percentage}%")
            return {"face_match_confidence": confidence_percentage}

        except Exception as e:
            self.logger.error(f"Face matching error: {str(e)}")
            # Fallback to cosine similarity
            cosine_sim = self._cosine_similarity(selfie_embedding, passport_embedding)
            from app.utils.model_thresholds import get_model_threshold
            threshold = get_model_threshold("Facenet512")
            distance = 1 - cosine_sim
            confidence = max(0, 1.0 - distance) if distance < threshold else 0.0
            confidence_percentage = round(confidence * 100, 2)
            self.logger.info(f"✅ Face match (fallback): passport ↔ selfie confidence={confidence_percentage}%")
            return {"face_match_confidence": confidence_percentage}

    def _cosine_similarity(self, vec1: list, vec2: list) -> float:
        """Calculate cosine similarity."""
        import math
        if len(vec1) != len(vec2) or not vec1 or not vec2:
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = math.sqrt(sum(a * a for a in vec1))
        mag2 = math.sqrt(sum(b * b for b in vec2))

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot_product / (mag1 * mag2)

    def _is_worldcheck_available(self) -> bool:
        """Check if World-Check API key is configured and non-empty."""
        try:
            from app.config.worldcheck_config import worldcheck_settings
            return bool(worldcheck_settings.api_key and
                       worldcheck_settings.api_key.strip() and
                       worldcheck_settings.api_secret and
                       worldcheck_settings.api_secret.strip())
        except Exception:
            return False

    def _should_run_worldcheck_for_key(self, client_public_key: str) -> bool:
        """
        Check if World-Check screening should run for a given public key.

        World-Check screening is restricted to only run for passport submissions
        that come from a public key registered with api_url == 'https://api.fraxn.ai:443'.

        Args:
            client_public_key: The client's public key

        Returns:
            True if World-Check should run for this key, False otherwise
        """
        try:
            user_key = self.user_key_repo.get_key_by_public_key(client_public_key)
            if not user_key:
                self.logger.info(f"World-Check skipped: User key not found for public key: {client_public_key[:16]}...")
                return False

            api_url = user_key.get('api_url')
            if api_url == 'https://api.fraxn.ai:443':
                self.logger.info(f"World-Check enabled: Public key registered with Fraxn API URL")
                return True
            else:
                self.logger.info(
                    f"World-Check skipped: Public key registered with non-Fraxn API URL: {api_url or 'None'}"
                )
                return False
        except Exception as e:
            self.logger.error(f"Error checking if World-Check should run for key: {e}")
            return False

    # =========================================================================
    # STRICT LINEAR PIPELINE - 7 Steps, each must pass before next
    # =========================================================================

    async def process_passport_strict(
        self,
        client_public_key: str,
        file_data: str,
        filename: str,
        callback_url: Optional[str] = None,
        document_type: str = "passport"
    ) -> SequentialJobResponse:
        """
        Strict linear passport processing pipeline (v2.0 - Dynamic Region Exclusion).

        Each step must pass before proceeding to the next.
        Any failure returns immediately with failure reason.

        Pipeline Steps:
        1. PhotoHolmes authenticity checks
        2. DocTR OCR extraction
        3. Process OCR + Remove user text regions (values, MRZ)
        4. Face matching + Remove face region
        5. Reference passport comparison (on cleaned image)
        6. PEP/Criminal database checks
        7. Web search + sentiment analysis

        Returns:
            SequentialJobResponse with result and failure step if applicable
        """
        job_id = f"{document_type}_strict_{uuid.uuid4().hex[:12]}"
        start_time = time.time()
        user_identity_id = None

        # Validate state - must be state 1 (selfie done) or resubmission (state >= 2)
        is_valid, error_msg, is_resubmission = self.state_service.validate_document_submission(
            client_public_key, 'passport'
        )
        if not is_valid:
            self.logger.error(f"State validation failed: {error_msg}")
            current_state = self.state_service.get_verification_state(client_public_key)
            current_seq = self.state_service.get_sequence_no(client_public_key)
            return SequentialJobResponse(
                result=False,
                job_id=job_id,
                verification_state=current_state,
                sequence_no=current_seq,
                processing_time_seconds=round(time.time() - start_time, 2),
                error=error_msg,
                error_code=DocumentErrorCode.PROCESSING_ERROR
            )

        try:
            # Get user_identity_id and decode image
            user_key = self.user_key_repo.get_key_by_public_key(client_public_key)
            user_identity_id = user_key['user_identity_id']
            self.logger.info(f"[STRICT PIPELINE v2.0] Processing passport for user_identity: {user_identity_id}")

            # Decode image
            image_bytes = base64.b64decode(file_data)
            is_pdf = filename.lower().endswith('.pdf') or image_bytes.startswith(b'%PDF')

            # ═══════════════════════════════════════════════════════════════
            # STEP 1: PhotoHolmes Authenticity Checks
            # ═══════════════════════════════════════════════════════════════
            self.logger.info("[STEP 1/7] Running PhotoHolmes authenticity checks...")
            step1_result = await self._step1_photoholmes_check(image_bytes, document_type)
            if not step1_result['passed']:
                return self._fail_strict_result(
                    step="photoholmes",
                    reason=step1_result['reason'],
                    user_identity_id=user_identity_id,
                    job_id=job_id,
                    start_time=start_time,
                    client_public_key=client_public_key
                )

            forgery_checks = step1_result.get('forgery_checks')

            # ═══════════════════════════════════════════════════════════════
            # STEP 2: DocTR OCR Extraction
            # ═══════════════════════════════════════════════════════════════
            self.logger.info("[STEP 2/7] Running DocTR OCR extraction...")
            step2_result = await self._step2_doctr_extraction(image_bytes, is_pdf)
            ocr_result = step2_result  # Contains text_blocks, raw_text, text_regions

            # ═══════════════════════════════════════════════════════════════
            # STEP 3: Process OCR + Remove User Text Regions (NEW v2.0)
            # ═══════════════════════════════════════════════════════════════
            self.logger.info("[STEP 3/7] Processing OCR and removing user text regions...")
            step3_result = self._step3_process_ocr_and_remove_regions(
                image_bytes=image_bytes,
                ocr_result=ocr_result,
                document_type=document_type
            )
            if not step3_result['passed']:
                return self._fail_strict_result(
                    step="field_extraction",
                    reason=step3_result['reason'],
                    user_identity_id=user_identity_id,
                    job_id=job_id,
                    start_time=start_time,
                    client_public_key=client_public_key,
                    forgery_checks=forgery_checks,
                    extracted_data=step3_result.get('extracted_data')
                )

            extracted_data = step3_result['data']
            cleaned_image_np = step3_result['cleaned_image']  # Image with text/MRZ removed
            removed_regions = step3_result.get('removed_regions', [])
            self.logger.info(f"[STEP 3] Removed {len(removed_regions)} user text regions")

            # ═══════════════════════════════════════════════════════════════
            # STEP 4: Face Matching + Remove Face Region (NEW v2.0)
            # ═══════════════════════════════════════════════════════════════
            self.logger.info("[STEP 4/7] Running face matching and removing face region...")
            step4_result = await self._step4_face_match_and_remove(
                cleaned_image_np=cleaned_image_np,
                user_identity_id=user_identity_id
            )
            if not step4_result['passed']:
                return self._fail_strict_result(
                    step="face_matching",
                    reason=step4_result['reason'],
                    user_identity_id=user_identity_id,
                    job_id=job_id,
                    start_time=start_time,
                    client_public_key=client_public_key,
                    forgery_checks=forgery_checks,
                    extracted_data=extracted_data,
                    removed_regions_count=len(removed_regions)
                )

            face_match_confidence = step4_result.get('face_match_confidence')
            final_cleaned_image_np = step4_result['final_cleaned_image']  # Image with face also removed
            face_bbox = step4_result.get('face_bbox')
            self.logger.info(f"[STEP 4] Face matched (confidence: {face_match_confidence}%), face region removed")

            # ═══════════════════════════════════════════════════════════════
            # STEP 5: Reference Passport Comparison (on cleaned image)
            # ═══════════════════════════════════════════════════════════════
            self.logger.info("[STEP 5/7] Running reference passport comparison on cleaned image...")
            step5_result = await self._step5_reference_comparison_cleaned(
                cleaned_image_np=final_cleaned_image_np,
                country_code=extracted_data.get('country_code') or extracted_data.get('passport_country')
            )
            if not step5_result['passed']:
                return self._fail_strict_result(
                    step="reference_comparison",
                    reason=step5_result['reason'],
                    user_identity_id=user_identity_id,
                    job_id=job_id,
                    start_time=start_time,
                    client_public_key=client_public_key,
                    forgery_checks=forgery_checks,
                    extracted_data=extracted_data,
                    face_match_confidence=face_match_confidence,
                    removed_regions_count=len(removed_regions),
                    similarity_score=step5_result.get('similarity_score'),
                    reference_threshold=step5_result.get('threshold', 0.65)
                )

            reference_scores = step5_result.get('region_scores', {})
            similarity_score = step5_result.get('similarity_score', 0)
            reference_threshold = step5_result.get('threshold', 0.65)
            self.logger.info(f"[STEP 5] Reference comparison passed, similarity: {similarity_score:.4f}")

            # ═══════════════════════════════════════════════════════════════
            # STEP 6: PEP/Criminal Database Checks
            # ═══════════════════════════════════════════════════════════════
            self.logger.info("[STEP 6/7] Running PEP/Criminal database checks...")
            step6_result = await self._step6_pep_criminal_check(
                full_name=extracted_data.get('full_name'),
                date_of_birth=extracted_data.get('dob') or extracted_data.get('date_of_birth'),
                country=extracted_data.get('country_code') or extracted_data.get('passport_country'),
                gender=extracted_data.get('sex'),
                user_identity_id=user_identity_id
            )
            if not step6_result['passed']:
                return self._fail_strict_result(
                    step="pep_criminal_check",
                    reason=step6_result['reason'],
                    user_identity_id=user_identity_id,
                    job_id=job_id,
                    start_time=start_time,
                    client_public_key=client_public_key,
                    forgery_checks=forgery_checks,
                    extracted_data=extracted_data,
                    face_match_confidence=face_match_confidence,
                    removed_regions_count=len(removed_regions)
                )

            osint_result = step6_result.get('osint_result')
            worldcheck_result = step6_result.get('worldcheck_result')

            # ═══════════════════════════════════════════════════════════════
            # STEP 7: Web Search + Sentiment Analysis
            # ═══════════════════════════════════════════════════════════════
            self.logger.info("[STEP 7/7] Running web search + sentiment analysis...")
            step7_result = await self._step7_web_search_sentiment(
                full_name=extracted_data.get('full_name'),
                country=extracted_data.get('country_code') or extracted_data.get('passport_country'),
                user_identity_id=user_identity_id
            )
            if not step7_result['passed']:
                return self._fail_strict_result(
                    step="sentiment_analysis",
                    reason=step7_result['reason'],
                    user_identity_id=user_identity_id,
                    job_id=job_id,
                    start_time=start_time,
                    client_public_key=client_public_key,
                    forgery_checks=forgery_checks,
                    extracted_data=extracted_data,
                    face_match_confidence=face_match_confidence,
                    removed_regions_count=len(removed_regions)
                )

            # ═══════════════════════════════════════════════════════════════
            # ALL STEPS PASSED - Update verification state
            # ═══════════════════════════════════════════════════════════════
            self.logger.info("[SUCCESS] All 7 steps passed! Updating verification state...")

            # Check document expiry
            document_expiry_valid = None
            if document_type == "passport":
                document_expiry_valid = self._check_passport_expiry(
                    extracted_data.get('date_of_expiry') or extracted_data.get('expiry')
                )
            elif document_type == "id_card":
                expiry_date = extracted_data.get('date_of_expiry') or extracted_data.get('expiry')
                if expiry_date:
                    document_expiry_valid = self._check_passport_expiry(expiry_date)
                else:
                    document_expiry_valid = True  # No expiry = valid for life

            if document_type == "passport" and not document_expiry_valid:
                return self._fail_strict_result(
                    step="expiry_check",
                    reason="Document has expired or has insufficient validity",
                    user_identity_id=user_identity_id,
                    job_id=job_id,
                    start_time=start_time,
                    client_public_key=client_public_key,
                    forgery_checks=forgery_checks,
                    extracted_data=extracted_data,
                    face_match_confidence=face_match_confidence
                )

            # Update user_identity_index with document data
            try:
                update_success = self._update_passport_data(
                    user_identity_id=user_identity_id,
                    extracted_data=extracted_data,
                    osint_result=osint_result,
                    worldcheck_result=worldcheck_result,
                    client_public_key=client_public_key
                )

                if not update_success:
                    doc_type_label = "ID card" if document_type == "id_card" else "passport"
                    return self._fail_strict_result(
                        step="database_update",
                        reason=f"Failed to update {doc_type_label} data in database",
                        user_identity_id=user_identity_id,
                        job_id=job_id,
                        start_time=start_time,
                        client_public_key=client_public_key,
                        forgery_checks=forgery_checks,
                        extracted_data=extracted_data,
                        face_match_confidence=face_match_confidence
                    )

            except ValueError as e:
                # Duplicate passport error
                return self._fail_strict_result(
                    step="database_update",
                    reason=str(e),
                    user_identity_id=user_identity_id,
                    job_id=job_id,
                    start_time=start_time,
                    client_public_key=client_public_key,
                    forgery_checks=forgery_checks,
                    extracted_data=extracted_data,
                    face_match_confidence=face_match_confidence
                )

            # Increment verification state (1 -> 2) only if currently at state 1
            current_state = self.state_service.get_verification_state(client_public_key)
            current_seq = self.state_service.get_sequence_no(client_public_key)
            new_state = current_state
            new_seq = current_seq

            if current_state == 1 and document_type == "passport":
                new_state = 2
                new_seq = 2
                # Update state in BOTH user_keys (per-device) AND user_identity_index (overall)
                self.user_key_repo.update_state_and_sequence(
                    user_public_key=client_public_key,
                    verification_state=new_state,
                    sequence_no=new_seq
                )
                # Update user_identity_index to match (use SET, not INCREMENT)
                self.user_identity_repo.set_verification_state(user_identity_id, new_state)
                self.user_identity_repo.set_sequence_no(user_identity_id, new_seq)
                self.logger.info(f"State set: {current_state} -> {new_state}, seq: {current_seq} -> {new_seq}")

            # Build other_checks
            other_checks = {
                "face_match_confidence": face_match_confidence,
                "document_type": document_type,
                "document_expiry_valid": document_expiry_valid,
                "is_resubmission": is_resubmission,
                "reference_scores": reference_scores,
                "similarity_score": similarity_score,
                "reference_threshold": reference_threshold,
                "removed_regions_count": len(removed_regions),
                "osint_risk_score": osint_result.get('overall_risk_score', 0) if osint_result else 0,
                "osint_risk_category": osint_result.get('risk_category', 'UNKNOWN') if osint_result else 'UNKNOWN',
                "osint_result": osint_result.get('result', 'PASS') if osint_result else 'PASS',
                "worldcheck_match": worldcheck_result.get('is_match', False) if worldcheck_result else False,
                "worldcheck_available": self._is_worldcheck_available()
            }

            processing_time = round(time.time() - start_time, 2)
            self.logger.info(f"[STRICT PIPELINE v2.0 COMPLETE] All steps passed in {processing_time}s")

            return SequentialJobResponse(
                result=True,
                job_id=job_id,
                verification_state=new_state,
                sequence_no=new_seq,
                processing_time_seconds=processing_time,
                extracted_data=extracted_data,
                forgery_checks=forgery_checks,
                other_checks=other_checks,
                user_identity_id=user_identity_id,
                message="Passport verified successfully through strict linear pipeline"
            )

        except Exception as e:
            self.logger.error(f"[STRICT PIPELINE ERROR] {str(e)}")
            current_seq = self.state_service.get_sequence_no(client_public_key)
            current_state = self.state_service.get_verification_state(client_public_key)
            return SequentialJobResponse(
                result=False,
                job_id=job_id,
                verification_state=current_state,
                sequence_no=current_seq,
                processing_time_seconds=round(time.time() - start_time, 2),
                error=str(e),
                error_code=DocumentErrorCode.PROCESSING_ERROR
            )

    # -------------------------------------------------------------------------
    # Step Helper Methods
    # -------------------------------------------------------------------------

    async def _step1_photoholmes_check(
        self,
        image_bytes: bytes,
        document_type: str
    ) -> Dict[str, Any]:
        """
        STEP 1: PhotoHolmes authenticity checks.

        Validates that the document hasn't been digitally manipulated.
        Can be skipped via verification_settings.skip_photoholmes (for testing).
        """
        # Check if PhotoHolmes should be skipped
        if verification_settings.skip_photoholmes:
            self.logger.info("[STEP 1] PhotoHolmes SKIPPED (skip_photoholmes=True)")
            return {
                "passed": True,
                "forgery_checks": {"skipped": True, "reason": "PhotoHolmes disabled via config"}
            }

        try:
            photoholmes_results = await comprehensive_photoholmes_service.run_all_methods(
                image_bytes,
                document_type=document_type
            )

            if not photoholmes_results:
                return {
                    "passed": False,
                    "reason": "PhotoHolmes analysis failed - no results returned"
                }

            # Transform to detailed results
            detailed_results = self.detailed_analysis_service.transform_photoholmes_results(
                photoholmes_results
            )

            # Build forgery_checks for response
            forgery_checks = {}
            for check in detailed_results.checks:
                forgery_checks[check.name] = {
                    "score": round(check.raw_score, 3),
                    "threshold": check.research_threshold
                }

            # Validate against threshold
            from app.services.selfie_validation_service import SelfieValidationService
            validation_service = SelfieValidationService()
            photoholmes_valid, photoholmes_error, photoholmes_error_code = validation_service.validate_photoholmes_results(
                detailed_results
            )

            if not photoholmes_valid:
                detected_methods = [
                    check.name for check in detailed_results.checks
                    if check.detected_forgery
                ]
                return {
                    "passed": False,
                    "reason": f"Forgery detected by {len(detected_methods)} methods: {', '.join(detected_methods)}",
                    "forgery_checks": forgery_checks
                }

            return {
                "passed": True,
                "forgery_checks": forgery_checks
            }

        except Exception as e:
            self.logger.error(f"PhotoHolmes check failed: {e}")
            return {
                "passed": False,
                "reason": f"PhotoHolmes check error: {str(e)}"
            }

    async def _step2_doctr_extraction(
        self,
        image_bytes: bytes,
        is_pdf: bool
    ) -> Dict[str, Any]:
        """
        STEP 2: DocTR OCR extraction.

        Extracts text blocks with geometry info. No pass/fail here.
        """
        try:
            document_data = await self.unified_extractor.extract(image_bytes, is_pdf)

            # Get text regions from the document result's text_blocks
            text_regions = []
            self.logger.info(f"[Step 2] document_data has text_blocks: {hasattr(document_data, 'text_blocks')}")
            if hasattr(document_data, 'text_blocks'):
                self.logger.info(f"[Step 2] text_blocks count: {len(document_data.text_blocks) if document_data.text_blocks else 0}")
                if document_data.text_blocks:
                    for block in document_data.text_blocks:
                        # Convert to bbox format [x1, y1, x2, y2]
                        bbox = [
                            block.get('x1', 0),
                            block.get('y1', 0),
                            block.get('x2', 1),
                            block.get('y2', 1)
                        ]
                        text_regions.append({
                            "bbox": bbox,
                            "text": block.get('text', '')
                        })
                    self.logger.info(f"[Step 2] Created {len(text_regions)} text_regions from text_blocks")

            return {
                "document_data": document_data,
                "text_regions": text_regions
            }

        except Exception as e:
            self.logger.error(f"DocTR extraction failed: {e}")
            return {
                "document_data": None,
                "text_regions": []
            }

    def _step3_process_ocr_and_remove_regions(
        self,
        image_bytes: bytes,
        ocr_result: Dict[str, Any],
        document_type: str
    ) -> Dict[str, Any]:
        """
        STEP 3 (v2.0): Process OCR blocks to extract fields, then remove user-specific regions.

        This combines field extraction with dynamic region removal:
        1. Extract fields and validate required fields
        2. Identify value boxes (user-specific data) from OCR
        3. Find and identify MRZ regions (rows containing "<<<")
        4. Remove all these regions from the image

        Returns:
            {
                "passed": bool,
                "data": extracted_data,
                "cleaned_image": np.ndarray (image with text regions removed),
                "removed_regions": list of removed region coordinates
            }
        """
        document_data = ocr_result.get('document_data')
        text_regions = ocr_result.get('text_regions', [])

        if not document_data:
            return {
                "passed": False,
                "reason": "No document data extracted from OCR",
                "extracted_data": {},
                "cleaned_image": None,
                "removed_regions": []
            }

        # Build extracted data
        extracted_data = self._build_extracted_data(document_data, document_type)

        # Determine required fields based on document type
        country_field = 'document_country' if document_type == 'id_card' else 'passport_country'
        number_field = 'number'

        # Check required fields
        country = extracted_data.get(country_field) or extracted_data.get('country_code')
        number = extracted_data.get(number_field)

        if not country:
            return {
                "passed": False,
                "reason": "Could not extract country code from document",
                "extracted_data": extracted_data,
                "cleaned_image": None,
                "removed_regions": []
            }

        if not number:
            return {
                "passed": False,
                "reason": "Could not extract document number from document",
                "extracted_data": extracted_data,
                "cleaned_image": None,
                "removed_regions": []
            }

        # Check for name (required)
        full_name = extracted_data.get('full_name')
        if not full_name:
            return {
                "passed": False,
                "reason": "Could not extract name from document",
                "extracted_data": extracted_data,
                "cleaned_image": None,
                "removed_regions": []
            }

        # ═══════════════════════════════════════════════════════════════════
        # Now remove user-specific regions from the image
        # ═══════════════════════════════════════════════════════════════════

        try:
            # Decode image to numpy array
            image_np = self._decode_image_to_numpy(image_bytes)
            image_shape = image_np.shape

            # Collect regions to remove
            regions_to_remove = []

            # 1. Add all text regions from OCR (these are user-specific values)
            for region in text_regions:
                bbox = region.get('bbox', [])
                if len(bbox) == 4:
                    normalized_bbox = self._normalize_bbox(bbox, image_shape)
                    regions_to_remove.append(normalized_bbox)

            # 2. Find and add MRZ region (rows containing "<<<")
            mrz_regions = self._find_mrz_regions(text_regions, image_shape)
            if mrz_regions:
                # Combine all MRZ rows into a single region
                combined_mrz = self._combine_regions(mrz_regions, padding=10)
                regions_to_remove.append(combined_mrz)
                self.logger.debug(f"Found MRZ region: {combined_mrz}")

            # Remove regions by filling with black
            cleaned_image = self._remove_regions_from_image(
                image_np,
                regions_to_remove,
                fill_color=(0, 0, 0)  # Black fill
            )

            self.logger.info(f"[Step 3] Removed {len(regions_to_remove)} user text regions from image")

            return {
                "passed": True,
                "data": extracted_data,
                "cleaned_image": cleaned_image,
                "removed_regions": regions_to_remove
            }

        except Exception as e:
            self.logger.error(f"Failed to remove regions from image: {e}")
            # Still return success for field extraction, but without cleaned image
            return {
                "passed": True,
                "data": extracted_data,
                "cleaned_image": self._decode_image_to_numpy(image_bytes),  # Return original
                "removed_regions": []
            }

    async def _step4_face_match_and_remove(
        self,
        cleaned_image_np: np.ndarray,
        user_identity_id: str
    ) -> Dict[str, Any]:
        """
        STEP 4 (v2.0): Face matching and face region removal.

        1. Extract face from the cleaned passport image
        2. Match against selfie stored in face_biometrics table
        3. If match passes, remove the face region from the image

        Args:
            cleaned_image_np: Passport image with text/MRZ already removed (numpy array)
            user_identity_id: User identity ID for fetching selfie embedding

        Returns:
            {
                "passed": bool,
                "face_match_confidence": float,
                "final_cleaned_image": np.ndarray (image with face also removed),
                "face_bbox": tuple (x1, y1, x2, y2) of face region
            }
        """
        # Check if face matching should be skipped
        if verification_settings.skip_face_matching:
            self.logger.info("[Step 4] Face matching SKIPPED via config")
            return {
                "passed": True,
                "face_match_confidence": None,
                "final_cleaned_image": cleaned_image_np,
                "face_bbox": None,
                "skipped": True
            }

        face_match_threshold = verification_settings.face_match_threshold

        try:
            # Convert numpy image to bytes for face extraction service
            image_bytes = self._encode_numpy_to_bytes(cleaned_image_np)

            # Extract face from passport
            passport_face_result = await self.face_extraction_service.extract_face_embedding(
                image_bytes=image_bytes,
                public_key=None,
                user_identity_id=user_identity_id,
                document_type="passport"
            )

            passport_face_embedding = None
            face_bbox = None

            if passport_face_result:
                if isinstance(passport_face_result, dict):
                    passport_face_embedding = passport_face_result.get('face_embedding')
                    face_bbox = passport_face_result.get('face_bbox')
                elif hasattr(passport_face_result, 'face_embedding'):
                    passport_face_embedding = passport_face_result.face_embedding
                    face_bbox = getattr(passport_face_result, 'face_bbox', None)

            if not passport_face_embedding:
                return {
                    "passed": False,
                    "reason": "No face detected in passport image"
                }

        except Exception as e:
            self.logger.error(f"Passport face extraction failed: {e}")
            return {
                "passed": False,
                "reason": f"Face extraction failed: {str(e)}"
            }

        # Get selfie embedding from database
        selfie_embeddings = self.face_biometrics_repo.get_embeddings_by_user_identity_ordered(
            user_identity_id,
            limit=1
        )

        if not selfie_embeddings:
            return {
                "passed": False,
                "reason": "No selfie face embedding found - selfie step may not be complete"
            }

        selfie_face_embedding = selfie_embeddings[0].get('embedding')

        if not selfie_face_embedding:
            return {
                "passed": False,
                "reason": "Selfie face embedding is invalid"
            }

        # Compare faces
        try:
            match_result = await self._perform_face_matching(
                selfie_face_embedding,
                passport_face_embedding
            )
            face_match_confidence = match_result.get("face_match_confidence")

            if face_match_confidence is None:
                return {
                    "passed": False,
                    "reason": "Face matching failed - no confidence score"
                }

            if face_match_confidence < face_match_threshold:
                return {
                    "passed": False,
                    "reason": f"Face match confidence {face_match_confidence}% below threshold {face_match_threshold}%",
                    "face_match_confidence": face_match_confidence
                }

        except Exception as e:
            self.logger.error(f"Face matching error: {e}")
            return {
                "passed": False,
                "reason": f"Face matching error: {str(e)}"
            }

        # ═══════════════════════════════════════════════════════════════════
        # Face matched successfully - now remove face region from image
        # ═══════════════════════════════════════════════════════════════════

        try:
            if face_bbox:
                # Normalize face bbox if needed
                if isinstance(face_bbox, (list, tuple)) and len(face_bbox) == 4:
                    # If bbox is normalized (0-1), convert to pixels
                    height, width = cleaned_image_np.shape[:2]
                    x1, y1, x2, y2 = face_bbox

                    if all(0 <= c <= 1 for c in [x1, y1, x2, y2]):
                        x1 = int(x1 * width)
                        y1 = int(y1 * height)
                        x2 = int(x2 * width)
                        y2 = int(y2 * height)
                    else:
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Add padding around face for safety
                    padding = 10
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(width, x2 + padding)
                    y2 = min(height, y2 + padding)

                    face_region = (x1, y1, x2, y2)
                else:
                    face_region = None

                if face_region:
                    # Remove face region from the already-cleaned image
                    final_cleaned_image = self._remove_regions_from_image(
                        cleaned_image_np,
                        [face_region],
                        fill_color=(0, 0, 0)
                    )
                    self.logger.info(f"[Step 4] Removed face region: {face_region}")
                else:
                    final_cleaned_image = cleaned_image_np
                    face_region = (0, 0, 0, 0)
            else:
                final_cleaned_image = cleaned_image_np
                face_region = (0, 0, 0, 0)
                self.logger.warning("[Step 4] No face bbox available, skipping face removal")

            return {
                "passed": True,
                "face_match_confidence": face_match_confidence,
                "final_cleaned_image": final_cleaned_image,
                "face_bbox": face_region
            }

        except Exception as e:
            self.logger.error(f"Failed to remove face region: {e}")
            # Still return success since face matching passed
            return {
                "passed": True,
                "face_match_confidence": face_match_confidence,
                "final_cleaned_image": cleaned_image_np,
                "face_bbox": (0, 0, 0, 0)
            }

    async def _step5_reference_comparison_cleaned(
        self,
        cleaned_image_np: np.ndarray,
        country_code: Optional[str]
    ) -> Dict[str, Any]:
        """
        STEP 5 (v2.0): Reference passport comparison on cleaned image.

        Compare the cleaned image (text removed, MRZ removed, face removed)
        against the reference template using simple SSIM comparison.

        Since user-specific regions are already removed, we only compare
        the non-user-specific visual elements (guilloche, security features).

        Args:
            cleaned_image_np: Cleaned passport image (numpy array)
            country_code: ISO 3-letter country code

        Returns:
            {
                "passed": bool,
                "reason": str,
                "similarity_score": float,
                "region_scores": dict
            }
        """
        from app.services.passport_reference_checker import passport_reference_checker
        from app.config.reference_config import reference_settings

        if not reference_settings.enable_reference_check:
            self.logger.info("Reference checking disabled, skipping step 5")
            return {"passed": True, "region_scores": {}, "similarity_score": 1.0}

        if not country_code:
            self.logger.warning("No country code available for reference comparison")
            if reference_settings.skip_if_no_template:
                return {"passed": True, "region_scores": {}, "reason": "Skipped - no country code", "similarity_score": 1.0}
            return {"passed": False, "reason": "No country code available for reference comparison", "similarity_score": 0.0}

        # Check if template exists
        if not passport_reference_checker.has_template(country_code):
            self.logger.info(f"No reference template for country: {country_code}")
            if reference_settings.skip_if_no_template:
                return {
                    "passed": True,
                    "region_scores": {},
                    "reason": f"Skipped - no template for {country_code}",
                    "similarity_score": 1.0
                }
            return {
                "passed": False,
                "reason": f"No reference template available for country: {country_code}",
                "similarity_score": 0.0
            }

        # Convert numpy array to bytes for the reference checker
        cleaned_image_bytes = self._encode_numpy_to_bytes(cleaned_image_np)

        # Run comparison with empty text_regions (already removed from image)
        result = passport_reference_checker.compare(
            submitted_image=cleaned_image_bytes,
            text_regions=[],  # No need to exclude - already removed
            country_code=country_code
        )

        # If enforcement is disabled, always pass regardless of score
        if not reference_settings.enforce_passport_specimen_check:
            return {
                "passed": True,
                "reason": "Similarity threshold enforcement disabled",
                "region_scores": result.get("region_scores", {}),
                "similarity_score": result.get("similarity_score", 0),
                "threshold": result.get("threshold", 0.65)
            }

        return {
            "passed": result.get("passed", False),
            "reason": result.get("reason", "Unknown error"),
            "region_scores": result.get("region_scores", {}),
            "similarity_score": result.get("similarity_score", 0),
            "threshold": result.get("threshold", 0.65)
        }

    async def _step6_pep_criminal_check(
        self,
        full_name: Optional[str],
        date_of_birth: Optional[str],
        country: Optional[str],
        gender: Optional[str],
        user_identity_id: str
    ) -> Dict[str, Any]:
        """
        STEP 6: PEP/Criminal database checks.

        Queries pep_entries and sanctions tables for name matches.
        Fails if high-confidence match (>=65%) found.
        """
        from app.config.osint_config import osint_settings

        if not full_name:
            return {
                "passed": False,
                "reason": "No name available for screening"
            }

        # Normalize DOB for screening
        dob_date_obj = self._parse_date(date_of_birth)
        screening_dob = dob_date_obj.strftime('%Y-%m-%d') if dob_date_obj else None

        # Run OSINT screening (includes PEP and sanctions)
        self.logger.info(f"Running OSINT screening for: {full_name[:20]}...")
        osint_result = await osint_screening_service.screen_individual(
            full_name=full_name,
            date_of_birth=screening_dob,
            country=country,
            gender=None,
            address=None,
            user_identity_id=user_identity_id
        )

        # Store OSINT result
        self.user_identity_repo.update_osint_result(user_identity_id, osint_result)

        # Check if OSINT failed
        if osint_result.get('overall_risk_score', 0) >= osint_settings.risk_threshold:
            return {
                "passed": False,
                "reason": f"OSINT risk score exceeds threshold: {osint_result['overall_risk_score']:.1f} ({osint_result.get('risk_category', 'UNKNOWN')})",
                "osint_result": osint_result
            }

        # Run World-Check if available
        worldcheck_result = None
        if self._is_worldcheck_available():
            from app.utils.country_code_converter import convert_to_alpha3
            country_alpha3 = convert_to_alpha3(country)

            worldcheck_result = await worldcheck_service.screen_individual(
                full_name=full_name,
                date_of_birth=screening_dob,
                country=country_alpha3,
                gender=gender
            )

            self.user_identity_repo.update_worldcheck_result(user_identity_id, worldcheck_result)

            if worldcheck_result.get('is_match'):
                return {
                    "passed": False,
                    "reason": "World-Check watchlist match found",
                    "osint_result": osint_result,
                    "worldcheck_result": worldcheck_result
                }

        return {
            "passed": True,
            "osint_result": osint_result,
            "worldcheck_result": worldcheck_result
        }

    async def _step7_web_search_sentiment(
        self,
        full_name: Optional[str],
        country: Optional[str],
        user_identity_id: str
    ) -> Dict[str, Any]:
        """
        STEP 7: Web search + sentiment analysis.

        Performs face-verified negative news search.
        Fails if significant negative sentiment detected.
        """
        from app.config.osint_config import osint_settings

        if not full_name:
            self.logger.info("No name available for web search, skipping step 7")
            return {"passed": True}

        if not osint_settings.enable_web_search:
            self.logger.info("Web search disabled, skipping step 7")
            return {"passed": True}

        try:
            # Use the OSINT service's face-verified search
            from app.services.osint_screening_service import osint_screening_service

            web_result = await osint_screening_service._get_face_verified_negative_news(
                full_name=full_name,
                country=country,
                user_identity_id=user_identity_id
            )

            if not web_result:
                self.logger.info("No web search results found")
                return {"passed": True}

            negative_count = web_result.get('negative_news_count', 0)
            avg_sentiment = web_result.get('average_sentiment', 0)

            # Check if too many negative results
            # Threshold: more than 2 negative news OR average sentiment below -0.5
            if negative_count > 2:
                return {
                    "passed": False,
                    "reason": f"Multiple negative news articles found: {negative_count}"
                }

            if avg_sentiment < -0.5:
                return {
                    "passed": False,
                    "reason": f"Significant negative sentiment detected: {avg_sentiment:.2f}"
                }

            return {"passed": True}

        except Exception as e:
            self.logger.error(f"Web search + sentiment analysis error: {e}")
            # Don't fail on web search errors - it's an external service
            return {"passed": True, "reason": f"Web search skipped due to error: {str(e)}"}

    def _fail_strict_result(
        self,
        step: str,
        reason: str,
        user_identity_id: Optional[str],
        job_id: str,
        start_time: float,
        client_public_key: str,
        forgery_checks: Optional[Dict] = None,
        extracted_data: Optional[Dict] = None,
        face_match_confidence: Optional[float] = None,
        removed_regions_count: int = 0,
        similarity_score: Optional[float] = None,
        reference_threshold: Optional[float] = None
    ) -> SequentialJobResponse:
        """
        Create a failure response for the strict pipeline.

        NO STATE REVERSION - leave current state unchanged on failure.
        """
        self.logger.warning(f"[STRICT PIPELINE FAILED] Step: {step}, Reason: {reason}")

        current_state = self.state_service.get_verification_state(client_public_key)
        current_seq = self.state_service.get_sequence_no(client_public_key)

        other_checks = {
            "failed_step": step,
            "face_match_confidence": face_match_confidence,
            "removed_regions_count": removed_regions_count
        }

        # Add similarity score if available (from reference comparison step)
        if similarity_score is not None:
            other_checks["similarity_score"] = similarity_score
        if reference_threshold is not None:
            other_checks["reference_threshold"] = reference_threshold

        # Map step to error code
        step_to_error_code = {
            "photoholmes": DocumentErrorCode.LOGICAL_FORGERY_DETECTED,
            "field_extraction": DocumentErrorCode.OCR_FAILED,
            "face_matching": DocumentErrorCode.LOGICAL_NAME_MISMATCH,
            "reference_comparison": DocumentErrorCode.LOGICAL_FORGERY_DETECTED,
            "osint_check": DocumentErrorCode.PROCESSING_ERROR,
            "web_search": DocumentErrorCode.PROCESSING_ERROR,
        }
        error_code = step_to_error_code.get(step, DocumentErrorCode.PROCESSING_ERROR)

        return SequentialJobResponse(
            result=False,
            job_id=job_id,
            verification_state=current_state,
            sequence_no=current_seq,
            processing_time_seconds=round(time.time() - start_time, 2),
            extracted_data=extracted_data,
            forgery_checks=forgery_checks,
            other_checks=other_checks,
            error=f"Failed at step {step}: {reason}",
            error_code=error_code
        )

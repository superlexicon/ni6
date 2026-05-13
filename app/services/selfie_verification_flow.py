"""
Shared selfie verification flow used across multiple services.

This module provides a unified verification flow for:
- sequential_selfie_service.py (identity verification)
- key_recovery_service.py (key recovery / secret share retrieval)
"""

from typing import Dict, Optional, List
from dataclasses import dataclass
import asyncio
import base64
import os

from app.repositories.otp_repository import OTPRepository
from app.repositories.face_biometrics_repository import FaceBiometricsRepository
from app.services.selfie_validation_service import SelfieValidationService
from app.services.detailed_analysis_service import DetailedAnalysisService
from app.helper.extractors.selfie_otp_extractor import SelfieOTPExtractor
from app.config.verification_config import verification_settings
from app.core.logger import get_logger
from app.dto import DocumentErrorCode


@dataclass
class SelfieVerificationResult:
    """Result of selfie verification flow"""
    success: bool
    error: Optional[str] = None
    error_code: Optional[str] = None  # NEW: error code for client response

    # Extracted OTP
    extracted_otp: Optional[str] = None

    # Mobile number and identity (available on success)
    mobile_number: Optional[str] = None
    identity_id: Optional[str] = None

    # Face data (available on success)
    face_embedding: Optional[List[float]] = None
    face_image_b64: Optional[str] = None
    anti_spoofing_score: Optional[float] = None
    face_detected: bool = False
    face_confidence: Optional[float] = None
    estimated_age: Optional[int] = None
    age_confidence: Optional[float] = None

    # PhotoHolmes results (detailed per-method)
    forgery_checks: Optional[Dict[str, Dict]] = None  # {"method": {"score": x, "threshold": y}}
    photoholmes_detections: Optional[int] = None
    forgery_probability: Optional[float] = None


class SelfieVerificationFlow:
    """
    Shared selfie verification flow used across multiple services.

    Encapsulates the full verification flow:
    1. OTP extraction and validation (early, before slow analysis)
    2. PhotoHolmes forgery detection
    3. Anti-spoofing liveness check + face extraction (embedding + image)
    4. Duplicate face check (prevents same face registering twice)
    """

    def __init__(self):
        self.logger = get_logger()
        self.otp_repository = OTPRepository()
        self.validation_service = SelfieValidationService(self.otp_repository)
        self.otp_extractor = SelfieOTPExtractor()
        self.face_biometrics_repo = FaceBiometricsRepository()

    async def verify_selfie(
        self,
        selfie_bytes: bytes,
        public_key: str,
        otp_code: Optional[str] = None,
        filename: Optional[str] = None,
        require_otp: bool = True,
        photoholmes_threshold: Optional[int] = None,
        anti_spoofing_threshold: Optional[float] = None,
        skip_photoholmes: bool = False
    ) -> SelfieVerificationResult:
        """
        Complete selfie verification flow.

        Args:
            selfie_bytes: Raw bytes of the selfie image
            public_key: User's public key (for OTP lookup)
            otp_code: OTP code (if already extracted, skips extraction)
            filename: Optional filename (may contain OTP)
            require_otp: Whether OTP is mandatory (default True)
            photoholmes_threshold: Forgery detection threshold (default from config)
            anti_spoofing_threshold: Liveness threshold (default from config)
            skip_photoholmes: Skip PhotoHolmes analysis (for faster verification)

        Returns:
            SelfieVerificationResult with all checks and extracted face data
        """
        if photoholmes_threshold is None:
            photoholmes_threshold = verification_settings.forgery_detection_threshold
        if anti_spoofing_threshold is None:
            anti_spoofing_threshold = verification_settings.anti_spoofing_threshold

        try:
            # Step 1: OTP extraction (if not provided) and validation
            extracted_otp = otp_code
            mobile_number = None
            identity_id = None
            if require_otp:
                otp_result = await self._validate_otp(
                    selfie_bytes, public_key, otp_code, filename
                )
                if not otp_result['success']:
                    return SelfieVerificationResult(
                        success=False,
                        error=otp_result['error'],
                        error_code=otp_result.get('error_code')
                    )
                extracted_otp = otp_result['otp']
                mobile_number = otp_result.get('mobile_number')
                identity_id = otp_result.get('identity_id')
                self.logger.debug(f"OTP validation passed: {extracted_otp}, mobile: {mobile_number}, identity: {identity_id[:16] if identity_id else None}...")

            # Step 2 & 3: Run PhotoHolmes and Face Extraction in PARALLEL for performance
            photoholmes_detections = None
            forgery_probability = None
            forgery_checks = None

            # Create tasks for parallel execution
            face_task = self._extract_face_with_anti_spoofing(
                selfie_bytes, extracted_otp or "000000", anti_spoofing_threshold
            )

            if not skip_photoholmes:
                # Run both in parallel
                photoholmes_task = self._run_photoholmes(selfie_bytes, photoholmes_threshold)
                photoholmes_result, face_result = await asyncio.gather(
                    photoholmes_task, face_task
                )

                # Check PhotoHolmes result
                if not photoholmes_result['success']:
                    return SelfieVerificationResult(
                        success=False,
                        error=photoholmes_result['error'],
                        forgery_checks=photoholmes_result.get('forgery_checks'),
                        error_code=photoholmes_result.get('error_code')
                    )
                photoholmes_detections = photoholmes_result.get('detections')
                forgery_probability = photoholmes_result.get('forgery_probability')
                forgery_checks = photoholmes_result.get('forgery_checks')
                self.logger.info(f"PhotoHolmes passed: {photoholmes_detections} detections")
            else:
                # Only run face extraction
                face_result = await face_task

            # Check face result
            if not face_result['success']:
                return SelfieVerificationResult(
                    success=False,
                    error=face_result['error'],
                    error_code=face_result.get('error_code')
                )

            self.logger.info(
                f"Selfie verification passed all checks "
                f"(OTP: {extracted_otp}, anti-spoofing: {face_result['anti_spoofing_score']:.2f})"
            )

            # Return success with all extracted data
            return SelfieVerificationResult(
                success=True,
                extracted_otp=extracted_otp,
                mobile_number=mobile_number,
                identity_id=identity_id,
                face_embedding=face_result['face_embedding'],
                face_image_b64=face_result['face_image_b64'],
                anti_spoofing_score=face_result['anti_spoofing_score'],
                face_detected=face_result['face_detected'],
                face_confidence=face_result.get('face_confidence'),
                estimated_age=face_result.get('estimated_age'),
                age_confidence=face_result.get('age_confidence'),
                forgery_checks=forgery_checks,
                photoholmes_detections=photoholmes_detections,
                forgery_probability=forgery_probability
            )

        except Exception as e:
            self.logger.error(f"Selfie verification failed: {str(e)}")
            return SelfieVerificationResult(
                success=False,
                error=f"Selfie verification error: {str(e)}",
                error_code=DocumentErrorCode.PROCESSING_ERROR
            )

    async def _validate_otp(
        self,
        selfie_bytes: bytes,
        public_key: Optional[str],
        otp_code: Optional[str],
        filename: Optional[str]
    ) -> Dict:
        """
        Extract and validate OTP.

        For recovery mode (public_key=None), use new OTP validation flow:
        - Lookup OTP by code (not public_key)
        - Get mobile_number and identity_id for face biometrics lookup

        For normal mode (public_key provided), use existing validation against public_key.

        Returns:
            Dict with 'success', 'otp', 'mobile_number', 'identity_id', and 'error' keys
        """
        try:
            # Extract OTP if not provided
            extracted_otp = otp_code
            if not extracted_otp:
                otp_data = await self.otp_extractor.extract_otp_quick(selfie_bytes, filename)
                extracted_otp = otp_data.get('otp')

            if not extracted_otp:
                return {
                    'success': False,
                    'otp': None,
                    'mobile_number': None,
                    'identity_id': None,
                    'error': "OTP not found in selfie image or filename"
                }

            # For recovery mode (public_key is None), use new OTP validation flow
            if public_key is None:
                # Recovery mode: validate OTP by code and get mobile_number + identity_id
                otp_valid, otp_error, mobile_number, identity_id, otp_error_code = self.validation_service.validate_otp(
                    extracted_otp
                )

                if not otp_valid:
                    return {
                        'success': False,
                        'otp': None,
                        'mobile_number': None,
                        'identity_id': None,
                        'error': otp_error,
                        'error_code': otp_error_code
                    }

                self.logger.debug(f"OTP validation passed for recovery mode: {extracted_otp}, mobile: {mobile_number}, identity: {identity_id[:16]}...")
                # Note: We don't delete OTP in recovery mode since we don't have a public_key to look up

                return {
                    'success': True,
                    'otp': extracted_otp,
                    'mobile_number': mobile_number,
                    'identity_id': identity_id,
                    'error': None,
                    'error_code': None
                }
            else:
                # Normal mode: validate OTP against specific public_key
                otp_valid, otp_error, otp_error_code = self.validation_service.validate_otp_against_database(
                    extracted_otp, public_key
                )

                if not otp_valid:
                    return {
                        'success': False,
                        'otp': None,
                        'mobile_number': None,
                        'identity_id': None,
                        'error': otp_error,
                        'error_code': otp_error_code
                    }

                # Get mobile_number and identity_id from user_keys for consistency
                from app.repositories.user_key_repository import UserKeyRepository
                user_key_repo = UserKeyRepository()
                user_key = user_key_repo.get_key_by_public_key(public_key)
                mobile_number = user_key.get('mobile_number') if user_key else None
                identity_id = user_key.get('user_identity_id') if user_key else None

                # Note: OTP deletion is now handled AFTER user key creation
                # in sequential_selfie_service.py to avoid authentication gap

                return {
                    'success': True,
                    'otp': extracted_otp,
                    'mobile_number': mobile_number,
                    'identity_id': identity_id,
                    'error': None,
                    'error_code': None
                }

        except Exception as e:
            return {
                'success': False,
                'otp': None,
                'mobile_number': None,
                'identity_id': None,
                'error': f"OTP validation error: {str(e)}"
            }

    async def _run_photoholmes(
        self,
        selfie_bytes: bytes,
        threshold: int
    ) -> Dict:
        """
        Run PhotoHolmes forgery detection.

        Returns:
            Dict with 'success', 'detections', 'forgery_probability', and 'error' keys
        """
        try:
            # Use ComprehensivePhotoHolmesService directly
            from app.services.comprehensive_photoholmes_service import ComprehensivePhotoHolmesService

            photoholmes_service = ComprehensivePhotoHolmesService()
            photoholmes_results = await photoholmes_service.run_all_methods(
                selfie_bytes, document_type="selfie"
            )

            if not photoholmes_results:
                self.logger.warning("PhotoHolmes results unavailable")
                # Return success but with no results (allow proceeding)
                return {
                    'success': True,
                    'detections': None,
                    'forgery_probability': None,
                    'forgery_checks': None,
                    'error': None
                }

            # Transform and validate results
            detailed_results = DetailedAnalysisService().transform_photoholmes_results(photoholmes_results)

            # Build forgery_checks dict: {"method": {"score": x, "threshold": y}}
            forgery_checks = {}
            for check in detailed_results.checks:
                forgery_checks[check.name] = {
                    "score": round(check.raw_score, 3),
                    "threshold": check.research_threshold
                }

            photoholmes_valid, photoholmes_error, photoholmes_error_code = self.validation_service.validate_photoholmes_results(
                detailed_results, threshold=threshold
            )

            if not photoholmes_valid:
                return {
                    'success': False,
                    'detections': detailed_results.checks_with_detections,
                    'forgery_probability': detailed_results.overall_forgery_probability,
                    'forgery_checks': forgery_checks,
                    'error': photoholmes_error,
                    'error_code': photoholmes_error_code
                }

            return {
                'success': True,
                'detections': detailed_results.checks_with_detections,
                'forgery_probability': detailed_results.overall_forgery_probability,
                'forgery_checks': forgery_checks,
                'error': None,
                'error_code': None
            }

        except Exception as e:
            self.logger.error(f"PhotoHolmes analysis failed: {str(e)}")
            return {
                'success': False,
                'detections': None,
                'forgery_probability': None,
                'forgery_checks': None,
                'error': f"Forgery detection error: {str(e)}"
            }

    async def _extract_face_with_anti_spoofing(
        self,
        selfie_bytes: bytes,
        pre_extracted_otp: str,
        anti_spoofing_threshold: float
    ) -> Dict:
        """
        Extract face data and validate anti-spoofing.

        Returns:
            Dict with face data and validation result
        """
        try:
            # Use extract_face_only (skips OCR since OTP already extracted)
            selfie_data = await self.otp_extractor.extract_face_only(
                selfie_bytes, pre_extracted_otp=pre_extracted_otp
            )

            # Validate anti-spoofing score
            if selfie_data.anti_spoofing_score < anti_spoofing_threshold:
                return {
                    'success': False,
                    'error': (
                        f"Liveness check failed - anti-spoofing score {selfie_data.anti_spoofing_score:.2f} "
                        f"below required threshold {anti_spoofing_threshold}. Please submit a live selfie."
                    ),
                    'anti_spoofing_score': selfie_data.anti_spoofing_score,
                    'error_code': DocumentErrorCode.SELFIE_LIVENESS_FAILED
                }

            return {
                'success': True,
                'error': None,
                'face_embedding': getattr(selfie_data, 'face_embedding', None),
                'face_image_b64': getattr(selfie_data, 'face_image_b64', None),
                'anti_spoofing_score': selfie_data.anti_spoofing_score,
                'face_detected': selfie_data.face_detected,
                'face_confidence': selfie_data.face_confidence,
                'estimated_age': selfie_data.estimated_age,
                'age_confidence': selfie_data.age_confidence,
                'error_code': None
            }

        except Exception as e:
            self.logger.error(f"Face extraction failed: {str(e)}")
            return {
                'success': False,
                'error': f"Face extraction error: {str(e)}"
            }

    async def verify_video_selfie(
        self,
        video_bytes: bytes,
        public_key: Optional[str],
        filename: str,
        require_otp: bool = True,
        photoholmes_threshold: Optional[int] = None,
        anti_spoofing_threshold: Optional[float] = None,
        skip_photoholmes: bool = True  # Skip PhotoHolmes by default for video (faster)
    ) -> SelfieVerificationResult:
        """
        Verify video selfie with hand gesture OTP.

        Reuses components from VideoSelfieService but returns
        SelfieVerificationResult for compatibility with KeyRecoveryService.

        Flow:
        1. Validate video format
        2. Extract frames at guided timestamps
        3. Detect hand gestures in frames
        4. Extract OTP from gesture sequence
        5. Validate OTP (recovery mode)
        6. Extract face from best frame
        7. Validate anti-spoofing score
        8. Return SelfieVerificationResult

        Args:
            video_bytes: Raw bytes of the video file
            public_key: User's public key (for OTP lookup) - None for recovery mode
            filename: Video filename
            require_otp: Whether OTP is mandatory (default True)
            photoholmes_threshold: Forgery detection threshold (default from config)
            anti_spoofing_threshold: Liveness threshold (default from config)
            skip_photoholmes: Skip PhotoHolmes analysis (default True for video)

        Returns:
            SelfieVerificationResult with all checks and extracted face data
        """
        if photoholmes_threshold is None:
            photoholmes_threshold = verification_settings.forgery_detection_threshold
        if anti_spoofing_threshold is None:
            anti_spoofing_threshold = verification_settings.anti_spoofing_threshold

        try:
            # Import video utilities
            from app.utils.video_utils import (
                validate_video_format,
                get_video_metadata,
                _extract_frame_with_ffmpeg,
                VideoValidationError,
                FrameResult,
            )
            from app.helper.hand_gesture_detector import get_detector
            from app.helper.gesture_otp_extractor import GestureOTPExtractor

            # Step 1: Validate video format
            try:
                format_ext = validate_video_format(filename)
                self.logger.info(f"Video format validated: {format_ext}")
            except VideoValidationError as e:
                return SelfieVerificationResult(
                    success=False,
                    error=f"Invalid video format: {e.message}",
                    error_code=DocumentErrorCode.SELFIE_INVALID_VIDEO_FORMAT
                )

            # Step 2: Parse 6-digit OTP for timing values (format: delay1, gesture1, delay2, gesture2, delay3, gesture3)
            # Example: 255552 → delay1=2s, gesture1=5 fingers, delay2=5s, gesture2=5 fingers, delay3=5s, gesture3=2 fingers
            # For recovery mode, we need to extract OTP from filename first since we don't have it yet
            if require_otp:
                # Extract OTP from filename for recovery mode
                from app.helper.extractors.selfie_otp_extractor import SelfieOTPExtractor
                otp_extractor = SelfieOTPExtractor()
                otp_data = await otp_extractor.extract_otp_quick(video_bytes, filename)
                parsed_otp = otp_data.get('otp')

                if not parsed_otp or len(parsed_otp) != 6:
                    return SelfieVerificationResult(
                        success=False,
                        error="Video selfie requires 6-digit OTP for gesture timing (format: D1G1D2G2D3G3)",
                        error_code=DocumentErrorCode.SELFIE_INVALID_OTP
                    )
            else:
                # For testing without OTP, use default timing values
                parsed_otp = "255552"  # Default OTP format

            timings = [int(parsed_otp[0]), int(parsed_otp[2]), int(parsed_otp[4])]  # Delay values
            expected_gestures = [int(parsed_otp[1]), int(parsed_otp[3]), int(parsed_otp[5])]  # Finger counts

            self.logger.info(f"Parsed OTP {parsed_otp}: timings={timings}, expected_gestures={expected_gestures}")

            # Step 3: Calculate dynamic frame timestamps from OTP timing values
            reaction_time = 1.0  # 1 second reaction time for user to form gesture
            gesture_gap = 1.0  # 1 second gap between gestures

            # Gesture 1 display and frame extraction
            gesture1_display_time = float(timings[0])
            frame1_time = gesture1_display_time + reaction_time

            # Gesture 2 display and frame extraction
            gesture2_display_time = gesture1_display_time + reaction_time + gesture_gap + float(timings[1])
            frame2_time = gesture2_display_time + reaction_time

            # Gesture 3 display and frame extraction
            gesture3_display_time = gesture2_display_time + reaction_time + gesture_gap + float(timings[2])
            frame3_time = gesture3_display_time + reaction_time

            frame_times = [frame1_time, frame2_time, frame3_time]

            self.logger.info(
                f"Gesture display times: {gesture1_display_time}s, {gesture2_display_time}s, {gesture3_display_time}s"
            )
            self.logger.info(f"Extracting frames at {frame_times}")

            # Step 4: Extract frames at calculated timestamps using FFmpeg
            import datetime
            os.makedirs('/tmp', exist_ok=True)
            tmp_path = os.path.join('/tmp', f'temp_video_{os.getpid()}_{datetime.now().timestamp()}.mp4')
            with open(tmp_path, 'wb') as tmp:
                tmp.write(video_bytes)

            try:
                metadata = get_video_metadata(tmp_path)

                max_time = max(frame_times)
                if max_time >= metadata.duration_seconds:
                    return SelfieVerificationResult(
                        success=False,
                        error=f"Frame time {max_time}s exceeds video duration {metadata.duration_seconds}s",
                        error_code=DocumentErrorCode.SELFIE_INVALID_VIDEO_FORMAT
                    )

                frames = []
                for i, target_time in enumerate(frame_times):
                    frame = _extract_frame_with_ffmpeg(tmp_path, target_time)
                    if frame is None or frame.size == 0:
                        return SelfieVerificationResult(
                            success=False,
                            error=f"Failed to extract frame {i+1} at {target_time}s",
                            error_code=DocumentErrorCode.SELFIE_INVALID_VIDEO_FORMAT
                        )
                    frames.append(FrameResult(
                        frame=frame,
                        frame_index=i,
                        timestamp_seconds=target_time
                    ))

                self.logger.info(f"Extracted {len(frames)} frames from video at dynamic timestamps")

            except Exception as e:
                return SelfieVerificationResult(
                    success=False,
                    error=f"Frame extraction failed: {str(e)}",
                    error_code=DocumentErrorCode.SELFIE_INVALID_VIDEO_FORMAT
                )
            finally:
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

            # Step 5: Detect hand gestures
            detector = get_detector()
            finger_counts = []
            confidences = []

            self.logger.info(f"Expected gestures: {expected_gestures} (finger counts)")

            for i, frame_result in enumerate(frames):
                self.logger.debug(f"Processing frame {i+1}/3 (expected: {expected_gestures[i]} fingers)")
                gesture_result = detector.detect_gesture(frame_result.frame)

                if gesture_result.hand_detected:
                    self.logger.info(f"Frame {i+1}: Detected {gesture_result.finger_count} fingers (confidence: {gesture_result.confidence:.3f})")
                else:
                    self.logger.warning(f"Frame {i+1}: No hand detected")

                finger_counts.append(gesture_result.finger_count if gesture_result.hand_detected else -1)
                confidences.append(gesture_result.confidence if gesture_result.hand_detected else 0.0)

            self.logger.info(f"Detected gestures in {len(frames)} frames")

            # Step 6: Extract OTP using guided recording (each frame = one digit position)
            extractor = GestureOTPExtractor()
            otp_result = extractor.extract_otp_guided(finger_counts, confidences)

            if not otp_result.success:
                # Build descriptive error message
                error_msg = otp_result.error or "OTP extraction failed"
                if otp_result.hand_detection_rate > 0:
                    error_msg += f" (hand detection rate: {otp_result.hand_detection_rate:.1%})"
                return SelfieVerificationResult(
                    success=False,
                    error=error_msg,
                    error_code=DocumentErrorCode.SELFIE_OTP_EXTRACTION_FAILED
                )

            extracted_otp = otp_result.otp
            self.logger.info(f"Extracted OTP from video: {extracted_otp}, hand_detection_rate: {otp_result.hand_detection_rate:.1%}")

            # Step 7: Validate OTP
            mobile_number = None
            identity_id = None

            if require_otp:
                # For recovery mode (public_key is None), use new OTP validation flow
                if public_key is None:
                    # Recovery mode: validate OTP by code and get mobile_number + identity_id
                    otp_valid, otp_error, mobile_number, identity_id, otp_error_code = self.validation_service.validate_otp(
                        extracted_otp
                    )

                    if not otp_valid:
                        return SelfieVerificationResult(
                            success=False,
                            error=otp_error or "OTP validation failed",
                            error_code=otp_error_code
                        )

                    self.logger.debug(f"OTP validation passed for recovery mode: {extracted_otp}, mobile: {mobile_number}, identity: {identity_id[:16]}...")
                else:
                    # Normal mode: validate OTP against specific public_key
                    otp_valid, otp_error, otp_error_code = self.validation_service.validate_otp_against_database(
                        extracted_otp, public_key
                    )

                    if not otp_valid:
                        return SelfieVerificationResult(
                            success=False,
                            error=otp_error or "OTP validation failed",
                            error_code=otp_error_code
                        )

                    # Get mobile_number and identity_id from user_keys
                    from app.repositories.user_key_repository import UserKeyRepository
                    user_key_repo = UserKeyRepository()
                    user_key = user_key_repo.get_key_by_public_key(public_key)
                    mobile_number = user_key.get('mobile_number') if user_key else None
                    identity_id = user_key.get('user_identity_id') if user_key else None

                    # Note: OTP deletion is now handled by the calling service
                    # after user key creation to prevent authentication gap

            # Step 8: Extract face from best frame using gesture timestamps
            # Use gesture display times for face extraction (when user is making gestures)
            import cv2
            import tempfile

            face_result = None
            # Use gesture display times + reaction time for face extraction
            # These are the times when the user should be visible in the frame
            face_target_times = [
                gesture1_display_time + 0.5,  # Early in gesture 1
                gesture2_display_time + 0.5,  # Early in gesture 2
                gesture3_display_time + 0.5,  # Early in gesture 3
            ]

            # Create temporary file for FFmpeg
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(video_bytes)
                tmp_path = tmp.name

            try:
                # Import FFmpeg helper function
                from app.utils.video_utils import _extract_frame_with_ffmpeg

                # Try each target time until we get a successful face extraction
                for target_time in target_times:
                    # Use FFmpeg for frame-accurate extraction
                    frame = _extract_frame_with_ffmpeg(tmp_path, target_time)

                    if frame is None:
                        continue

                    # Verify frame is valid
                    if frame.size == 0:
                        continue

                    # Convert BGR to RGB and encode as JPEG
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    is_success, buffer = cv2.imencode(".jpg", frame_rgb)
                    if not is_success:
                        continue

                    frame_bytes = buffer.tobytes()

                    # Try to extract face from this frame
                    try:
                        selfie_data = await self.otp_extractor.extract_face_only(
                            content=frame_bytes,
                            pre_extracted_otp=extracted_otp
                        )

                        if selfie_data.face_detected:
                            # Validate anti-spoofing score
                            anti_spoofing_score = selfie_data.anti_spoofing_score
                            if anti_spoofing_score < anti_spoofing_threshold:
                                return SelfieVerificationResult(
                                    success=False,
                                    error=(
                                        f"Liveness check failed - anti-spoofing score {anti_spoofing_score:.2f} "
                                        f"below required threshold {anti_spoofing_threshold}. "
                                        f"Please submit a live video selfie."
                                    ),
                                    error_code=DocumentErrorCode.SELFIE_LIVENESS_FAILED
                                )

                            # Success - found face with valid anti-spoofing score
                            self.logger.info(f"Successfully extracted face from frame at {target_time}s")

                            face_result = {
                                'success': True,
                                'face_embedding': getattr(selfie_data, 'face_embedding', None),
                                'face_image_b64': getattr(selfie_data, 'face_image_b64', None),
                                'anti_spoofing_score': anti_spoofing_score,
                                'face_detected': True,
                                'face_confidence': selfie_data.face_confidence,
                                'estimated_age': getattr(selfie_data, 'estimated_age', None),
                                'age_confidence': getattr(selfie_data, 'age_confidence', None),
                            }
                            break
                    except Exception as e:
                        self.logger.warning(f"Face extraction failed at {target_time}s: {str(e)}")
                        continue

            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

            if not face_result or not face_result['success']:
                return SelfieVerificationResult(
                    success=False,
                    error=f"No face detected in video frames at {face_target_times}",
                    error_code=DocumentErrorCode.SELFIE_NO_FACE_DETECTED
                )

            self.logger.info(
                f"Video selfie verification passed all checks "
                f"(OTP: {extracted_otp}, anti-spoofing: {face_result['anti_spoofing_score']:.2f})"
            )

            # Return success with all extracted data
            return SelfieVerificationResult(
                success=True,
                extracted_otp=extracted_otp,
                mobile_number=mobile_number,
                identity_id=identity_id,
                face_embedding=face_result.get('face_embedding'),
                face_image_b64=face_result.get('face_image_b64'),
                anti_spoofing_score=face_result.get('anti_spoofing_score'),
                face_detected=face_result.get('face_detected', False),
                face_confidence=face_result.get('face_confidence'),
                estimated_age=face_result.get('estimated_age'),
                age_confidence=face_result.get('age_confidence'),
                forgery_checks=None,  # PhotoHolmes skipped for video
                photoholmes_detections=None,
                forgery_probability=None
            )

        except Exception as e:
            self.logger.error(f"Video selfie verification failed: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return SelfieVerificationResult(
                success=False,
                error=f"Video selfie verification error: {str(e)}",
                error_code=DocumentErrorCode.PROCESSING_ERROR
            )

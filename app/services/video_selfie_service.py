"""
Video Selfie Service - Main orchestration for video-based selfie verification.

Combines hand gesture detection, face anti-spoofing, and forgery detection
to verify users who gesture their OTP using finger counts (0-5).
"""

import time
import uuid
import asyncio
import os
from datetime import datetime
from typing import Optional, Dict, Any, List

import numpy as np
import cv2

from app.dto.video_selfie import (
    VideoSelfieResult,
    FrameGestureResult,
    GestureTransition,
    VideoMetadata,
)
from app.dto.verification_session import SequentialJobResponse
from app.config.video_config import video_config
from app.utils.video_utils import (
    extract_frames,
    validate_video_format,
    get_video_metadata,
    VideoValidationError,
    frame_to_rgb,
)
from app.helper.hand_gesture_detector import get_detector
from app.helper.gesture_otp_extractor import GestureOTPExtractor
from app.helper.extractors.selfie_otp_extractor import SelfieOTPExtractor
from app.services.selfie_validation_service import SelfieValidationService
from app.services.detailed_analysis_service import DetailedAnalysisService
# from app.services.comprehensive_photoholmes_service import ComprehensivePhotoHolmesService  # DISABLED
from app.services.otp_service import OTPService
from app.repositories.otp_repository import OTPRepository
from app.repositories.face_biometrics_repository import FaceBiometricsRepository
from app.services.verification_state_service import VerificationStateService
from app.repositories.user_key_repository import UserKeyRepository
from app.repositories.user_identity_repository import UserIdentityRepository
from app.core.logger import get_logger
from app.helper.face_recognition_factory import get_model_name


class VideoSelfieService:
    """
    Service for video-based selfie verification with hand gesture OTP.

    Flow:
    1. Video validation (format, duration, size)
    2. Frame extraction (2 FPS sampling)
    3. Hand gesture detection per frame
    4. OTP extraction from gesture transitions
    5. OTP validation
    6. Face anti-spoofing (from best frame)
    7. PhotoHolmes forgery detection
    8. Face extraction and embedding

    This provides a secure, liveness-assured alternative to text-based OTP selfie.
    """

    def __init__(
        self,
        otp_repository: Optional[OTPRepository] = None,
        user_key_repository: Optional[UserKeyRepository] = None,
        user_identity_repository: Optional[UserIdentityRepository] = None,
        face_biometrics_repository: Optional[FaceBiometricsRepository] = None,
    ):
        self.logger = get_logger()
        self.otp_repository = otp_repository or OTPRepository()
        self.user_key_repository = user_key_repository or UserKeyRepository()
        self.user_identity_repository = user_identity_repository or UserIdentityRepository()
        self.face_biometrics_repo = face_biometrics_repository or FaceBiometricsRepository()

        self.state_service = VerificationStateService()
        self.validation_service = SelfieValidationService(self.otp_repository)
        # Use the singleton otp_service from app.services (already properly initialized)
        from app.services import otp_service
        self.otp_service = otp_service
        self.detailed_analysis = DetailedAnalysisService()

        # PhotoHolmes service (singleton) - DISABLED
        # from app.services import comprehensive_photoholmes_service
        # self.photoholmes_service = comprehensive_photoholmes_service

        self.logger.info("VideoSelfieService initialized")

    async def process_video_selfie(
        self,
        client_public_key: str,
        video_bytes: bytes,
        filename: str,
        mobile_number: Optional[str] = None,
        country_code: Optional[str] = None,
        callback_url: Optional[str] = None,
    ) -> SequentialJobResponse:
        """
        Process a video selfie with hand gesture OTP.

        Args:
            client_public_key: Client's public key
            video_bytes: Video file as bytes
            filename: Video filename
            mobile_number: User's mobile number
            country_code: Country code for mobile
            callback_url: Optional callback URL

        Returns:
            SequentialJobResponse with verification results
        """
        job_id = f"video_selfie_{uuid.uuid4().hex[:12]}"
        start_time = time.time()

        # Step 1: Validate user and get identity
        user_key = self.user_key_repository.get_key_by_public_key(client_public_key)
        if not user_key:
            self.logger.error(f"User not found for public_key: {client_public_key[:16]}...")
            return self._error_response(
                job_id=job_id,
                error="User not found. Please request OTP first",
                start_time=start_time,
            )

        user_identity_id = user_key.get('user_identity_id')
        self.logger.info(f"Processing video selfie for user: {user_identity_id[:16]}...")

        # Step 2: Process video and extract OTP
        try:
            video_result = await self._process_video_and_extract_otp(
                video_bytes=video_bytes,
                filename=filename,
                public_key=client_public_key,
            )

            if not video_result.result:
                return self._error_response(
                    job_id=job_id,
                    error=video_result.error,
                    start_time=start_time,
                    extracted_data=video_result.extracted_data,
                    other_checks=video_result.other_checks,
                    video_metadata=video_result.video_metadata,
                    gesture_transitions=video_result.gesture_transitions,
                )

            # Step 3: Verify OTP
            otp_code = video_result.extracted_data.get("otp_number")
            otp_verification = self.otp_service.verify_otp_from_selfie(
                mobile_number=mobile_number,
                otp_code=otp_code,
                client_public_key=client_public_key,
            )

            if not otp_verification.get('valid'):
                self.logger.error(f"OTP verification failed: {otp_verification.get('message')}")
                return self._error_response(
                    job_id=job_id,
                    error=otp_verification.get('message', 'OTP verification failed'),
                    start_time=start_time,
                    extracted_data=video_result.extracted_data,
                    other_checks={
                        **video_result.other_checks,
                        "otp_verified": False,
                    },
                )

            # Step 4: Face extraction and embedding (from best frame)
            face_result = await self._extract_face_from_video(
                video_bytes=video_bytes,
                otp_code=otp_code,
            )

            if not face_result['success']:
                return self._error_response(
                    job_id=job_id,
                    error=face_result.get('error', 'Face extraction failed'),
                    start_time=start_time,
                    extracted_data=video_result.extracted_data,
                    other_checks={**video_result.other_checks, **face_result.get('other_checks', {})},
                    forgery_checks=video_result.forgery_checks,
                )

            # Step 5: Check for duplicate face
            face_embedding = face_result.get('face_embedding')
            if face_embedding:
                duplicate_check = self.face_biometrics_repo.check_duplicate_embedding(
                    face_embedding=face_embedding
                )
                if duplicate_check and duplicate_check.get('user_identity_id') != user_identity_id:
                    self.logger.error(f"Duplicate face detected for different user")
                    return self._error_response(
                        job_id=job_id,
                        error="Duplicate face detected - this face is already registered",
                        start_time=start_time,
                        extracted_data=video_result.extracted_data,
                    )

            # Step 6: Store face biometric
            if face_embedding:
                try:
                    biometric_id = self.face_biometrics_repo.create_face_biometric(
                        user_identity_id=user_identity_id,
                        face_embedding=face_embedding,
                        model_name=get_model_name()
                    )
                    if biometric_id:
                        self.logger.info(f"Stored face biometric {biometric_id}")
                except Exception as e:
                    self.logger.error(f"Failed to store face biometric: {str(e)}")
                    return self._error_response(
                        job_id=job_id,
                        error=f"Failed to store face biometric: {str(e)}",
                        start_time=start_time,
                        extracted_data=video_result.extracted_data,
                    )

            # Step 7: Increment verification state (0 -> 1) and sequence_no (0 -> 1)
            # ONLY if this is the first submission (not a resubmission)
            current_state = self.state_service.get_verification_state(client_public_key)
            current_seq = self.state_service.get_sequence_no(client_public_key)

            if current_state == 0:  # First submission only
                new_state = 1
                new_seq = 1
                # Update state in BOTH user_keys (per-device) AND user_identity_index (overall)
                self.user_key_repository.update_state_and_sequence(
                    user_public_key=client_public_key,
                    verification_state=new_state,
                    sequence_no=new_seq
                )
                # Update user_identity_index to match (use SET, not INCREMENT)
                self.user_identity_repo.set_verification_state(user_identity_id, new_state)
                self.user_identity_repo.set_sequence_no(user_identity_id, new_seq)
                self.logger.info(f"First video selfie submission - state updated to {new_state}")
            else:
                # Resubmission - keep current state and sequence
                new_state = current_state
                new_seq = current_seq
                self.logger.info(f"Video selfie resubmission (state={current_state}). State unchanged.")

            processing_time = time.time() - start_time

            self.logger.info(
                f"Video selfie verification successful for user {user_identity_id[:16]}... "
                f"(OTP: {otp_code}, time: {processing_time:.2f}s)"
            )

            # Build success response
            return SequentialJobResponse(
                result=True,
                job_id=job_id,
                verification_state=new_state,
                sequence_no=new_seq,
                processing_time_seconds=round(processing_time, 2),
                user_identity_id=user_identity_id,
                extracted_data=video_result.extracted_data,
                forgery_checks=video_result.forgery_checks,
                other_checks={
                    **video_result.other_checks,
                    **face_result.get('other_checks', {}),
                    "otp_verified": True,
                    "hand_detection_rate": video_result.other_checks.get("hand_detection_rate", 0.0),
                },
            )

        except VideoValidationError as e:
            self.logger.warning(f"Video validation failed: {e}")
            # Revert state from 1 back to 0 if it was incremented (in user_keys for this device)
            if user_identity_id:
                current_state = self.state_service.get_verification_state(client_public_key)
                if current_state == 1:
                    # Revert state in user_keys for this device
                    self.user_key_repository.update_state_and_sequence(
                        user_public_key=client_public_key,
                        verification_state=0,
                        sequence_no=0
                    )
                    self.logger.info(f"Reverted verification state from 1 to 0 after validation failure")
            return self._error_response(
                job_id=job_id,
                error=e.message if hasattr(e, 'message') else str(e),
                start_time=start_time,
            )
        except Exception as e:
            self.logger.error(f"Unexpected error in process_video_selfie: {e}")
            # Revert state from 1 back to 0 if it was incremented (in user_keys for this device)
            if user_identity_id:
                current_state = self.state_service.get_verification_state(client_public_key)
                if current_state == 1:
                    # Revert state in user_keys for this device
                    self.user_key_repository.update_state_and_sequence(
                        user_public_key=client_public_key,
                        verification_state=0,
                        sequence_no=0
                    )
                    self.logger.info(f"Reverted verification state from 1 to 0 after error")
            return self._error_response(
                job_id=job_id,
                error=f"Video selfie processing failed: {str(e)}",
                start_time=start_time,
            )

    async def _process_video_and_extract_otp(
        self,
        video_bytes: bytes,
        filename: str,
        public_key: str,
    ) -> VideoSelfieResult:
        """
        Process video and extract OTP from hand gestures.

        Args:
            video_bytes: Video file as bytes
            filename: Video filename
            public_key: User's public key

        Returns:
            VideoSelfieResult with OTP and gesture data
        """
        # Step 1: Validate format
        format_ext = validate_video_format(filename)

        # Step 2: Extract frames
        frames, metadata = extract_frames(video_bytes)

        # Step 3: Detect gestures in all frames
        detector = get_detector()
        finger_counts = []
        confidences = []
        frame_results = []

        for i, frame_result in enumerate(frames):
            gesture_result = detector.detect_gesture(frame_result.frame)

            # Save frame if hand detection FAILED for debugging
            if not gesture_result.hand_detected:
                failed_dir = "/tmp/video_frames_failed"
                os.makedirs(failed_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                failed_path = os.path.join(
                    failed_dir,
                    f"failed_digit{i+1}_t{frame_result.timestamp_seconds}s_{timestamp}.jpg"
                )
                cv2.imwrite(failed_path, frame_result.frame)
                self.logger.warning(f"Saved failed detection frame to {failed_path}")

            frame_results.append(FrameGestureResult(
                finger_count=gesture_result.finger_count,
                confidence=gesture_result.confidence,
                hand_detected=gesture_result.hand_detected,
                handedness=gesture_result.handedness,
                frame_index=i,
                timestamp_seconds=frame_result.timestamp_seconds,
            ))
            finger_counts.append(gesture_result.finger_count if gesture_result.hand_detected else -1)
            confidences.append(gesture_result.confidence if gesture_result.hand_detected else 0.0)

        # Don't close the singleton detector - it will be reused

        # Step 4: Extract OTP using guided recording (each frame = one digit position)
        extractor = GestureOTPExtractor()
        otp_result = extractor.extract_otp_guided(finger_counts, confidences)

        # Step 5: Build transitions list (from result for compatibility)
        transitions = [
            GestureTransition(
                digit=t.digit,
                frame_start=t.frame_start,
                frame_end=t.frame_end,
                duration_seconds=t.duration_seconds,
                confidence=t.confidence,
            )
            for t in otp_result.transitions
        ]

        if not otp_result.success:
            return VideoSelfieResult(
                result=False,
                status="failed",
                error=otp_result.error,
                video_metadata=VideoMetadata(
                    duration_seconds=metadata.duration_seconds,
                    frame_count=metadata.frame_count,
                    fps=metadata.fps,
                    width=metadata.width,
                    height=metadata.height,
                    format=metadata.format,
                    size_bytes=metadata.size_bytes,
                ),
                frames_processed=otp_result.frames_processed,
                gesture_transitions=transitions,
                other_checks={
                    "hand_detection_rate": otp_result.hand_detection_rate,
                },
            )

        # Step 6: Run PhotoHolmes forgery detection (on first frame) - DISABLED
        # forgery_checks = await self._run_photoholmes_on_frame(frames[0].frame)
        forgery_checks = {}

        return VideoSelfieResult(
            result=True,
            status="completed",
            video_metadata=VideoMetadata(
                duration_seconds=metadata.duration_seconds,
                frame_count=metadata.frame_count,
                fps=metadata.fps,
                width=metadata.width,
                height=metadata.height,
                format=metadata.format,
                size_bytes=metadata.size_bytes,
            ),
            frames_processed=len(frames),
            extracted_data={"otp_number": otp_result.otp},
            gesture_transitions=transitions,
            forgery_checks=forgery_checks,
            other_checks={
                "hand_detection_rate": otp_result.hand_detection_rate,
                "gesture_count": len(transitions),
            },
        )

    # DISABLED: PhotoHolmes forgery detection
    # async def _run_photoholmes_on_frame(self, frame: np.ndarray) -> Dict[str, Any]:
    #     """Run PhotoHolmes forgery detection on a single frame."""
    #     try:
    #         # Convert frame to JPEG for PhotoHolmes
    #         is_success, buffer = cv2.imencode(".jpg", frame)
    #         if not is_success:
    #             return {}
    #
    #         frame_bytes = buffer.tobytes()
    #
    #         result = await asyncio.to_thread(
    #             self.photoholmes_service.run_all_methods,
    #             frame_bytes
    #         )
    #
    #         return {
    #             "photoholmes_score": result.get("overall_score", 0.0),
    #             "photoholmes_detections": result.get("detections", 0),
    #         }
    #
    #     except Exception as e:
    #         self.logger.error(f"PhotoHolmes analysis failed: {str(e)}")
    #         return {}

    async def _extract_face_from_video(
        self,
        video_bytes: bytes,
        otp_code: str,
    ) -> Dict[str, Any]:
        """
        Extract face from frames at 4.5, 8.5, 12.5, and 16.5 seconds.

        Video timeline:
        - 0-1s: "Ready" overlay
        - 1-2s: "Set" overlay
        - 2-3s: "Go" overlay
        - 3-5s: Digit 1 displayed (sampled at 4.5s)
        - 5-7s: Gap
        - 7-9s: Digit 2 displayed (sampled at 8.5s)
        - 9-11s: Gap
        - 11-13s: Digit 3 displayed (sampled at 12.5s)
        - 13-15s: Gap
        - 15-17s: Digit 4 displayed (sampled at 16.5s)

        Uses FFmpeg for frame-accurate extraction instead of OpenCV seeking.

        Args:
            video_bytes: Video file as bytes
            otp_code: OTP code for logging

        Returns:
            Dict with face extraction results including embedding
        """
        target_times = [4.5, 8.5, 12.5, 16.5]
        otp_extractor = SelfieOTPExtractor()

        try:
            # Create temporary file for FFmpeg
            import tempfile
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
                        self.logger.warning(f"Failed to extract frame at {target_time}s using FFmpeg")
                        continue

                    # Verify frame is valid
                    if frame.size == 0:
                        self.logger.warning(f"Extracted empty frame at {target_time}s")
                        continue

                    # Convert BGR to RGB
                    import cv2
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Encode frame as JPEG for analysis
                    is_success, buffer = cv2.imencode(".jpg", frame_rgb)
                    if not is_success:
                        self.logger.warning(f"Failed to encode frame at {target_time}s")
                        continue

                    frame_bytes = buffer.tobytes()

                    # Try to extract face from this frame
                    selfie_data = await otp_extractor.extract_face_only(
                        content=frame_bytes,
                        pre_extracted_otp=otp_code
                    )

                    if selfie_data.face_detected:
                        # Validate anti-spoofing score
                        anti_spoofing_threshold = 0.5
                        anti_spoofing_score = selfie_data.anti_spoofing_score

                        if anti_spoofing_score < anti_spoofing_threshold:
                            return {
                                "success": False,
                                "error": (
                                    f"Liveness check failed - anti-spoofing score {anti_spoofing_score:.2f} "
                                    f"below required threshold {anti_spoofing_threshold}. "
                                    f"Please submit a live video selfie."
                                ),
                                "other_checks": {
                                    "face_detected": True,
                                    "anti_spoofing_score": anti_spoofing_score,
                                }
                            }

                        self.logger.info(f"Successfully extracted face from frame at {target_time}s")

                        # SUCCESS - Return face data with embedding
                        return {
                            "success": True,
                            "face_embedding": selfie_data.face_embedding,
                            "face_image_b64": selfie_data.face_image_b64,
                            "other_checks": {
                                "face_detected": True,
                                "anti_spoofing_score": anti_spoofing_score,
                                "face_confidence": selfie_data.face_confidence,
                                "estimated_age": getattr(selfie_data, 'estimated_age', None),
                                "extracted_at_time": target_time,
                            }
                        }

            finally:
                # Clean up temp file
                import os
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

            # No face detected in any frame
            return {
                "success": False,
                "error": f"No face detected in video frames at {target_times}",
                "other_checks": {
                    "face_detected": False,
                    "anti_spoofing_score": 0.0,
                }
            }

        except Exception as e:
            self.logger.error(f"Face extraction from video failed: {str(e)}")
            return {
                "success": False,
                "error": f"Face extraction failed: {str(e)}",
            }

    def _error_response(
        self,
        job_id: str,
        error: str,
        start_time: float,
        extracted_data: Optional[Dict[str, Any]] = None,
        forgery_checks: Optional[Dict[str, Any]] = None,
        other_checks: Optional[Dict[str, Any]] = None,
        video_metadata: Optional[Any] = None,
        gesture_transitions: Optional[List[GestureTransition]] = None,
    ) -> SequentialJobResponse:
        """Build error response."""
        return SequentialJobResponse(
            result=False,
            job_id=job_id,
            verification_state=0,
            sequence_no=0,
            processing_time_seconds=round(time.time() - start_time, 2),
            error=error,
            extracted_data=extracted_data or {},
            forgery_checks=forgery_checks or {},
            other_checks=other_checks or {},
        )

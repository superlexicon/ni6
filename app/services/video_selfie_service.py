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

        # Initialize OTP broadcast service for HTTP-based inter-instance communication
        try:
            from app.services.otp_broadcast_service import otp_broadcast_service as broadcast_svc
            self.otp_broadcast_service = broadcast_svc
        except Exception as e:
            self.logger.warning(f"OTP broadcast service not available: {e}")
            self.otp_broadcast_service = None

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

        # Step 1: Look up OTP record (instead of user_keys - follow SequentialSelfieService pattern)
        otp_record = self.otp_repository.get_otp_by_public_key(client_public_key)

        if not otp_record:
            self.logger.error(f"OTP not found for public_key: {client_public_key[:16]}...")
            return self._error_response(
                job_id=job_id,
                error="OTP not found. Please request OTP first",
                start_time=start_time,
            )

        self.logger.info(f"Found OTP record for public_key: {client_public_key[:16]}...")
        # Note: user_identity_id will be created after verification (like SequentialSelfieService)
        user_identity_id = None

        # Step 2: Get expected OTP from record
        expected_otp = otp_record.get('random_number')
        if not expected_otp or len(expected_otp) != 6:
            self.logger.error(f"Invalid gesture OTP format: {expected_otp}")
            return self._error_response(
                job_id=job_id,
                error=f"Invalid gesture OTP format: expected 6 digits, got {len(expected_otp) if expected_otp else 0}",
                start_time=start_time,
            )

        # Parse OTP: odd positions = delays, even positions = expected gestures
        # Example: 131241 -> delay1=1s, gesture1=3, delay2=2s, gesture2=1, delay3=4s, gesture3=2
        # Each delay is "show gesture AFTER X seconds from previous gesture start"
        timing1 = int(expected_otp[0])  # Digit 1: delay before first gesture (seconds)
        gesture1 = int(expected_otp[1])  # Digit 2: expected finger count (1-5)
        timing2 = int(expected_otp[2])  # Digit 3: delay after first gesture (seconds)
        gesture2 = int(expected_otp[3])  # Digit 4: expected finger count (1-5)
        timing3 = int(expected_otp[4])  # Digit 5: delay after second gesture (seconds)
        gesture3 = int(expected_otp[5])  # Digit 6: expected finger count (1-5)

        self.logger.info(
            f"Gesture OTP {expected_otp}: "
            f"After {timing1}s show {gesture1} fingers, "
            f"after reaction+1s gap+{timing2}s show {gesture2} fingers, "
            f"after reaction+1s gap+{timing3}s show {gesture3} fingers"
        )

        # Step 3: Process video and verify gestures
        try:
            video_result = await self._process_video_and_verify_gestures(
                video_bytes=video_bytes,
                filename=filename,
                public_key=client_public_key,
                expected_gestures=[gesture1, gesture2, gesture3],
                timings=[timing1, timing2, timing3],
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

            # Step 4: OTP verification successful (already done in _process_video_and_verify_gestures)
            self.logger.info(f"Gesture OTP verification successful for {client_public_key[:16]}...")

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

            # Build other_checks with face data
            face_other_checks = face_result.get('other_checks', {})
            face_embedding = face_result.get('face_embedding')

            # ========================================
            # At this point, ALL verification checks have passed
            # Now we create user_identity (if needed) and user_keys
            # ========================================

            # Step 5: Check for duplicate face and determine user_identity_id
            is_resubmission = False
            is_multi_device_link = False

            if face_embedding:
                from app.repositories.face_biometrics_repository import DuplicateFaceError
                try:
                    duplicate_check = self.face_biometrics_repo.check_duplicate_embedding(
                        face_embedding=face_embedding
                    )

                    if duplicate_check:
                        matched_user_identity_id = duplicate_check.get('user_identity_id')

                        if matched_user_identity_id:
                            # Check if this public key already has a user_keys record
                            existing_key = self.user_key_repository.get_key_by_public_key(client_public_key)

                            if existing_key and existing_key.get('user_identity_id') == matched_user_identity_id:
                                # Same user, same device - this is a resubmission
                                is_resubmission = True
                                user_identity_id = matched_user_identity_id
                                self.logger.info(f"Video selfie resubmission detected for user: {user_identity_id[:16]}...")
                            else:
                                # Different device (public_key) but same face - MULTI-DEVICE LINK
                                is_multi_device_link = True
                                user_identity_id = matched_user_identity_id
                                self.logger.info(
                                    f"Multi-device: Linking new device {client_public_key[:16]}... "
                                    f"to existing identity {user_identity_id[:16]}..."
                                )
                except DuplicateFaceError as e:
                    self.logger.error(f"Duplicate face error: {str(e)}")
                    return self._error_response(
                        job_id=job_id,
                        error="Duplicate face detected - this face is already registered",
                        start_time=start_time,
                        extracted_data=video_result.extracted_data,
                    )

            # Step 6: Create user_identity if needed (ONLY after ALL checks pass)
            if not user_identity_id:
                user_identity_id = self.user_identity_repository.create_empty_identity()
                if not user_identity_id:
                    self.logger.error("Failed to create user identity")
                    return self._error_response(
                        job_id=job_id,
                        error="Failed to create user identity",
                        start_time=start_time,
                        extracted_data=video_result.extracted_data,
                    )
                self.logger.info(f"Created new user_identity: {user_identity_id[:16]}...")

            # Step 7: Handle resubmission - delete old embeddings and add new one
            if is_resubmission and user_identity_id:
                self.face_biometrics_repo.delete_user_embeddings(user_identity_id)

                biometric_id = self.face_biometrics_repo.create_face_biometric(
                    user_identity_id=user_identity_id,
                    face_embedding=face_embedding,
                    model_name=get_model_name()
                )
                if biometric_id:
                    self.logger.info(f"Added new face biometric {biometric_id} for resubmission")

                current_state = self.state_service.get_verification_state(client_public_key)
                current_seq = self.state_service.get_sequence_no(client_public_key)
                return SequentialJobResponse(
                    result=True,
                    job_id=job_id,
                    verification_state=current_state,
                    sequence_no=current_seq,
                    processing_time_seconds=round(time.time() - start_time, 2),
                    user_identity_id=user_identity_id,
                    extracted_data=video_result.extracted_data,
                    forgery_checks=video_result.forgery_checks,
                    other_checks={
                        **video_result.other_checks,
                        **face_other_checks,
                        "otp_verified": True,
                        "is_resubmission": True,
                        "is_multi_device_link": False,
                    },
                )

            # Step 8: Insert face biometric (skip for multi-device link)
            if face_embedding and not is_multi_device_link:
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

            # Step 9: Insert into user_keys (after ALL verification passes)
            from app.repositories.user_keys_pending_repository import UserKeysPendingRepository
            pending_key_repo = UserKeysPendingRepository()

            # Check if user_keys record already exists for this public_key
            existing_key = self.user_key_repository.get_key_by_public_key(client_public_key)

            if existing_key:
                # Update existing record (shouldn't happen in normal flow, but handle it)
                self.logger.info(f"Updating existing user_keys record for {client_public_key[:16]}...")
                self.user_key_repository.update_key_by_public_key(
                    public_key=client_public_key,
                    update_data={
                        'user_identity_id': user_identity_id
                    }
                )
                # Delete from pending table since we updated existing record
                pending_key_repo.delete_pending_key(client_public_key)
            else:
                # Move pending key to user_keys (this is the normal flow)
                moved = pending_key_repo.move_pending_to_user_keys(client_public_key, user_identity_id)

                if not moved:
                    self.logger.error(f"Failed to move pending key to user_keys for {client_public_key[:16]}...")
                    # Fallback: Try to create from otp_record data (for backward compatibility)
                    self.logger.info(f"Fallback: Creating user_keys from otp_record data for {client_public_key[:16]}...")
                    key_data = {
                        'mobile_number': otp_record.get('mobile_number'),
                        'country_code': otp_record.get('country_code'),
                        'user_public_key': client_public_key,
                        'encrypted_secret_share': otp_record.get('encrypted_secret_share'),
                        'user_identity_id': user_identity_id
                    }
                    self.user_key_repository.create_key(key_data)
                    pending_key_repo.delete_pending_key(client_public_key)

            self.logger.info(f"✅ Moved pending key to user_keys for public_key: {client_public_key[:16]}...")

            # Step 10: Mark OTP as verified (but DON'T delete yet - wait for user key creation)
            if otp_code and otp_record.get('mobile_number'):
                self.otp_repository.mark_otp_verified(otp_record['mobile_number'])

                # Broadcast verification to peer instances
                if self.otp_broadcast_service:
                    try:
                        asyncio.create_task(self.otp_broadcast_service.broadcast_otp_verified(otp_record['mobile_number']))
                    except Exception as sync_error:
                        self.logger.error(f"Failed to broadcast OTP verification: {sync_error}")

            # Step 10.5: Delete OTP AFTER user key is successfully created
            # This prevents the authentication gap where OTP is deleted but user_key doesn't exist yet
            if otp_record.get('mobile_number'):
                deleted = self.otp_repository.delete_otp(otp_record['mobile_number'])
                if deleted:
                    self.logger.info(f"✅ OTP deleted AFTER user key creation for mobile: {otp_record['mobile_number']}")
                else:
                    self.logger.warning(f"Failed to delete OTP for mobile: {otp_record['mobile_number']}")

            # Step 11: Increment verification state (0 -> 1) and sequence_no (0 -> 1)
            # ONLY if this is the first submission for this device
            # This applies to both new users AND multi-device links
            current_state = self.state_service.get_verification_state(client_public_key)
            current_seq = self.state_service.get_sequence_no(client_public_key)

            if current_state == 0 and not is_resubmission:
                # First submission for this device - set state in BOTH tables
                # This applies to both new users AND multi-device links
                new_state = 1
                new_seq = 1

                # Update user_keys (per-device state)
                self.user_key_repository.update_state_and_sequence(
                    user_public_key=client_public_key,
                    verification_state=new_state,
                    sequence_no=new_seq
                )

                # Update user_identity_index (overall identity state) to match
                self.user_identity_repository.set_verification_state(user_identity_id, new_state)
                self.user_identity_repository.set_sequence_no(user_identity_id, new_seq)

                self.logger.info(
                    f"First video selfie submission for device - state updated to {new_state}. "
                    f"{'(Multi-device link)' if is_multi_device_link else '(New user)'}"
                )
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
                    **face_other_checks,
                    "otp_verified": True,
                    "hand_detection_rate": video_result.other_checks.get("hand_detection_rate", 0.0),
                    "is_resubmission": is_resubmission,
                    "is_multi_device_link": is_multi_device_link,
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

    async def _process_video_and_verify_gestures(
        self,
        video_bytes: bytes,
        filename: str,
        public_key: str,
        expected_gestures: List[int],
        timings: List[int],
    ) -> VideoSelfieResult:
        """
        Process video and verify hand gestures against expected OTP.

        OTP Format: [delay1][gesture1][delay2][gesture2][delay3][gesture3]
        - Odd positions (1, 3, 5): Delays in seconds (1-5)
        - Even positions (2, 4, 6): Expected finger counts (1-5)

        Timeline (with 1s reaction time + 1s gap between gestures):
        - Gesture 1 display: delay1
        - Frame 1: delay1 + reaction_time
        - Gesture 2 display: delay1 + reaction_time + gap + delay2
        - Frame 2: Gesture 2 display + reaction_time
        - Gesture 3 display: Gesture 2 display + reaction_time + gap + delay3
        - Frame 3: Gesture 3 display + reaction_time

        Args:
            video_bytes: Video file as bytes
            filename: Video filename
            public_key: User's public key
            expected_gestures: List of 3 expected finger counts (digits 2, 4, 6 from OTP)
            timings: List of 3 delay values (digits 1, 3, 5 from OTP)

        Returns:
            VideoSelfieResult with verification result and gesture data
        """
        # Step 1: Validate format
        format_ext = validate_video_format(filename)

        # Step 2: Create temp file for video processing
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        try:
            # Get video metadata
            metadata = get_video_metadata(tmp_path)

            # Step 3: Calculate frame extraction timestamps
            # Each delay is "show gesture AFTER X seconds from previous gesture display"
            # Timeline:
            #   Gesture 1 display at: delay1
            #   Frame 1 at: delay1 + reaction_time
            #   1 second gap
            #   Gesture 2 display at: delay1 + reaction_time + gap + delay2
            #   Frame 2 at: Gesture 2 display + reaction_time
            #   1 second gap
            #   Gesture 3 display at: Gesture 2 display + reaction_time + gap + delay3
            #   Frame 3 at: Gesture 3 display + reaction_time

            reaction_time = 1.0  # 1 second reaction time for user to form gesture
            gesture_gap = 1.0  # 1 second gap between gestures (after reaction time)

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
                f"Gesture display times: {gesture1_display_time}s, {gesture2_display_time}s, {gesture3_display_time}s "
                f"(with {reaction_time}s reaction + {gesture_gap}s gap between)"
            )
            self.logger.info(
                f"Extracting frames at {frame1_time}s, {frame2_time}s, {frame3_time}s "
                f"(display time + {reaction_time}s reaction)"
            )

            # Step 4: Extract frames at calculated timestamps using FFmpeg
            extracted_frames = []

            from app.utils.video_utils import _extract_frame_with_ffmpeg

            for i, target_time in enumerate(frame_times):
                frame = _extract_frame_with_ffmpeg(tmp_path, target_time)

                if frame is None or frame.size == 0:
                    self.logger.error(f"Failed to extract frame {i+1} at {target_time}s")
                    return VideoSelfieResult(
                        result=False,
                        status="failed",
                        error=f"Failed to extract frame {i+1} at {target_time}s from video",
                        video_metadata=VideoMetadata(
                            duration_seconds=metadata.duration_seconds,
                            frame_count=metadata.frame_count,
                            fps=metadata.fps,
                            width=metadata.width,
                            height=metadata.height,
                            format=metadata.format,
                            size_bytes=metadata.size_bytes,
                        ),
                        frames_processed=i,
                        gesture_transitions=[],
                        other_checks={},
                    )

                extracted_frames.append((i, target_time, frame))
                self.logger.info(f"Successfully extracted frame {i+1} at {target_time}s")

        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        # Step 5: Detect gestures in extracted frames
        detector = get_detector()
        detected_gestures = []
        frame_results = []

        for i, target_time, frame in extracted_frames:
            gesture_result = detector.detect_gesture(frame)

            # Save frame for debugging
            debug_dir = "/tmp/video_frames_gesture"
            os.makedirs(debug_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_path = os.path.join(
                debug_dir,
                f"gesture_frame{i+1}_t{target_time}s_{timestamp}.jpg"
            )
            cv2.imwrite(debug_path, frame)
            self.logger.info(f"Saved frame {i+1} to {debug_path}")

            frame_results.append(FrameGestureResult(
                finger_count=gesture_result.finger_count,
                confidence=gesture_result.confidence,
                hand_detected=gesture_result.hand_detected,
                handedness=gesture_result.handedness,
                frame_index=i,
                timestamp_seconds=target_time,
            ))

            if not gesture_result.hand_detected:
                self.logger.error(f"No hand detected in frame {i+1} at {target_time}s")
                return VideoSelfieResult(
                    result=False,
                    status="failed",
                    error=f"No hand detected in frame {i+1} at {target_time}s. Please ensure your hand is clearly visible.",
                    video_metadata=VideoMetadata(
                        duration_seconds=metadata.duration_seconds,
                        frame_count=metadata.frame_count,
                        fps=metadata.fps,
                        width=metadata.width,
                        height=metadata.height,
                        format=metadata.format,
                        size_bytes=metadata.size_bytes,
                    ),
                    frames_processed=len(extracted_frames),
                    gesture_transitions=[],
                    other_checks={"hand_detection_rate": i / len(extracted_frames)},
                )

            detected_gestures.append(gesture_result.finger_count)

        # Don't close the singleton detector - it will be reused

        # Step 6: Compare detected gestures with expected gestures
        all_match = True
        mismatches = []

        for i, (detected, expected) in enumerate(zip(detected_gestures, expected_gestures)):
            if detected != expected:
                all_match = False
                mismatches.append(f"Frame {i+1}: expected {expected} fingers, detected {detected}")
                self.logger.warning(f"Gesture mismatch at frame {i+1}: expected {expected}, got {detected}")

        if not all_match:
            mismatch_detail = "; ".join(mismatches)
            return VideoSelfieResult(
                result=False,
                status="failed",
                error=f"Gesture verification failed: {mismatch_detail}",
                video_metadata=VideoMetadata(
                    duration_seconds=metadata.duration_seconds,
                    frame_count=metadata.frame_count,
                    fps=metadata.fps,
                    width=metadata.width,
                    height=metadata.height,
                    format=metadata.format,
                    size_bytes=metadata.size_bytes,
                ),
                frames_processed=len(extracted_frames),
                gesture_transitions=[
                    GestureTransition(
                        digit=detected,
                        frame_start=i,
                        frame_end=i,
                        duration_seconds=0.0,
                        confidence=frame_results[i].confidence,
                    )
                    for i, detected in enumerate(detected_gestures)
                ],
                other_checks={
                    "hand_detection_rate": 1.0,
                    "detected_gestures": detected_gestures,
                    "expected_gestures": expected_gestures,
                },
            )

        self.logger.info(
            f"All gestures matched: {detected_gestures} == {expected_gestures}"
        )

        # Step 7: Build transitions list for response
        transitions = [
            GestureTransition(
                digit=detected,
                frame_start=i,
                frame_end=i,
                duration_seconds=0.0,
                confidence=frame_results[i].confidence,
            )
            for i, detected in enumerate(detected_gestures)
        ]

        # Step 8: Run PhotoHolmes forgery detection (on first frame) - DISABLED
        # forgery_checks = await self._run_photoholmes_on_frame(extracted_frames[0][2])
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
            frames_processed=len(extracted_frames),
            extracted_data={"otp_number": "".join(map(str, expected_gestures))},
            gesture_transitions=transitions,
            forgery_checks=forgery_checks,
            other_checks={
                "hand_detection_rate": 1.0,
                "detected_gestures": detected_gestures,
                "expected_gestures": expected_gestures,
                "gesture_count": len(detected_gestures),
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

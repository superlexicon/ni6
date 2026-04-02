"""
Sequential Selfie Service - Handles selfie processing in sequential mode.

State is tracked via verification_state column in user_keys table (per-device).

Multi-Device Support Flow:
1. Look up from otp table (instead of user_keys)
2. Run ALL selfie verification checks (OTP extraction, PhotoHolmes, anti-spoofing)
3. Handle duplicate face check:
   - If duplicate face found: Link to existing user_identity (multi-device)
   - If no duplicate: Create new user_identity
4. Insert face biometric (skip for multi-device link)
5. Insert into user_keys (after ALL verification passes)
6. Mark OTP as verified
7. Increment verification state in user_keys (per-device) and user_identity_index (overall)

NOTE: user_identity and user_keys are created HERE, not during OTP request.
This enables multiple public_keys (devices) to share the same user_identity.

State Management:
- user_keys: Stores per-device verification_state and sequence_no
- user_identity_index: Stores overall best state across devices
- On first selfie: Set state=1, seq=1 in user_keys for this device
- On multi-device link: ALSO increment state (new device needs its own state progression)
"""

from typing import Optional
import time
import base64
import uuid
import asyncio
from app.dto.verification_session import SequentialJobResponse
from app.services.verification_state_service import VerificationStateService
from app.services.selfie_verification_flow import SelfieVerificationFlow
from app.core.logger import get_logger
from app.repositories import UserKeyRepository
from app.repositories.otp_repository import OTPRepository
from app.repositories.user_identity_repository import UserIdentityRepository
from app.repositories.face_biometrics_repository import FaceBiometricsRepository, DuplicateFaceError
from app.dto import DocumentErrorCode


class SequentialSelfieService:
    """Service for handling selfie processing in sequential mode"""

    # Maximum number of selfie resubmissions allowed
    MAX_SELFIE_RESUBMISSIONS = 3

    def __init__(self,
                 user_key_repository: UserKeyRepository,
                 otp_repository: OTPRepository):
        self.user_key_repository = user_key_repository
        self.otp_repository = otp_repository
        self.logger = get_logger()
        self.state_service = VerificationStateService()
        self.selfie_verification = SelfieVerificationFlow()
        self.user_identity_repo = UserIdentityRepository()

        # Initialize OTP broadcast service for HTTP-based inter-instance communication
        try:
            from app.services.otp_broadcast_service import otp_broadcast_service as broadcast_svc
            self.otp_broadcast_service = broadcast_svc
        except Exception as e:
            self.logger.warning(f"OTP broadcast service not available: {e}")
            self.otp_broadcast_service = None

    async def process_selfie(self, client_public_key: str, file_data: str, filename: str,
                           iv: str, secret_share: str,
                           mobile_number: str, country_code: str, callback_url: Optional[str] = None) -> SequentialJobResponse:
        """
        Process a selfie document in sequential mode.

        Multi-Device Support Flow:
        1. Look up from otp table (instead of user_keys)
        2. Run ALL selfie verification checks
        3. Handle duplicate face: Link to existing identity (multi-device) or create new
        4. Create user_identity (if needed)
        5. Insert face biometric (skip for multi-device link)
        6. Insert into user_keys (after ALL verification passes)
        7. Mark OTP as verified
        8. Increment verification state

        Returns simplified SequentialJobResponse with:
        - result: bool (all checks passed)
        - verification_state: int (new state after this step)
        - extracted_data: OTP value
        - forgery_checks: PhotoHolmes results
        - other_checks: anti-spoofing, face detection, age estimation, is_multi_device_link
        """
        job_id = f"selfie_{uuid.uuid4().hex[:12]}"
        start_time = time.time()

        try:
            # Step 1: Look up OTP record (instead of user_keys - user_keys created after verification)
            otp_record = self.otp_repository.get_otp_by_public_key(client_public_key)

            if not otp_record:
                self.logger.error(
                    f"OTP not found for public_key: {client_public_key[:16]}... "
                    f"User must request OTP first"
                )
                current_seq = self.state_service.get_sequence_no(client_public_key)
                return SequentialJobResponse(
                    result=False,
                    job_id=job_id,
                    verification_state=0,
                    sequence_no=current_seq,
                    processing_time_seconds=round(time.time() - start_time, 2),
                    error='OTP not found. Please request OTP first',
                    error_code=DocumentErrorCode.SELFIE_OTP_NOT_FOUND,
                    extracted_data={},
                    forgery_checks={},
                    other_checks={}
                )

            self.logger.info(f"Found OTP record for mobile_number: {otp_record.get('mobile_number')}")

            # Note: user_identity_id will be determined after duplicate face check
            user_identity_id = None

            # Step 2: Run selfie verification flow (OTP, PhotoHolmes, anti-spoofing, face extraction)
            self.logger.info("Starting selfie verification flow...")
            image_bytes = base64.b64decode(file_data)

            verification_result = await self.selfie_verification.verify_selfie(
                selfie_bytes=image_bytes,
                public_key=client_public_key,
                filename=filename,
                require_otp=True
            )

            # Build other_checks
            other_checks = {
                "anti_spoofing_score": verification_result.anti_spoofing_score,
                "face_detected": verification_result.face_detected,
                "face_confidence": verification_result.face_confidence,
                "estimated_age": verification_result.estimated_age,
                "otp_verified": verification_result.extracted_otp is not None
            }

            # Use forgery_checks from verification result (detailed per-method format)
            forgery_checks = verification_result.forgery_checks

            # Build extracted_data
            extracted_data = {
                "otp_number": verification_result.extracted_otp
            }

            if not verification_result.success:
                self.logger.error(f"Selfie verification failed: {verification_result.error}")
                current_state = self.state_service.get_verification_state(user_identity_id)
                current_seq = self.state_service.get_sequence_no(user_identity_id)

                # Revert state if it was incremented past initial state (in user_keys)
                if current_state > 0:
                    # Revert state in user_keys for this device
                    self.user_key_repository.update_state_and_sequence(
                        user_public_key=client_public_key,
                        verification_state=0,
                        sequence_no=0
                    )
                    self.logger.info(f"Reverted verification state after failure: {current_state} -> 0")

                # Determine error code from verification result if available, otherwise use PROCESSING_ERROR
                error_code = getattr(verification_result, 'error_code', None) or DocumentErrorCode.PROCESSING_ERROR

                return SequentialJobResponse(
                    result=False,
                    job_id=job_id,
                    verification_state=0,
                    sequence_no=0,
                    processing_time_seconds=round(time.time() - start_time, 2),
                    error=verification_result.error,
                    error_code=error_code,
                    extracted_data=extracted_data,
                    forgery_checks=forgery_checks,
                    other_checks=other_checks,
                    user_identity_id=user_identity_id
                )

            # Check for duplicate/similar face in database
            face_biometrics_repo = FaceBiometricsRepository()
            is_resubmission = False
            is_multi_device_link = False

            if verification_result.face_embedding:
                duplicate_match = face_biometrics_repo.check_duplicate_embedding(
                    face_embedding=verification_result.face_embedding
                )

                if duplicate_match:
                    # Found similar face - check if it belongs to this public key
                    matched_user_identity_id = duplicate_match.get('user_identity_id')

                    if matched_user_identity_id:
                        # Check if this public key already has a user_keys record
                        existing_key = self.user_key_repository.get_key_by_public_key(client_public_key)

                        if existing_key and existing_key.get('user_identity_id') == matched_user_identity_id:
                            # Same user, same device - this is a resubmission
                            is_resubmission = True
                            user_identity_id = matched_user_identity_id
                            self.logger.info(f"Selfie resubmission detected for user: {user_identity_id[:16]}...")
                        else:
                            # Different device (public_key) but same face - MULTI-DEVICE LINK
                            is_multi_device_link = True
                            user_identity_id = matched_user_identity_id
                            self.logger.info(
                                f"Multi-device: Linking new device {client_public_key[:16]}... "
                                f"to existing identity {user_identity_id[:16]}..."
                            )

            other_checks["is_resubmission"] = is_resubmission
            other_checks["is_multi_device_link"] = is_multi_device_link

            # Handle resubmission - delete old embeddings and add new one
            if is_resubmission and user_identity_id:
                # Delete old embeddings to prevent accumulation
                face_biometrics_repo.delete_user_embeddings(user_identity_id)

                # Add new face embedding
                biometric_id = face_biometrics_repo.create_face_biometric(
                    user_identity_id=user_identity_id,
                    face_embedding=verification_result.face_embedding
                )
                if biometric_id:
                    self.logger.info(f"Added new face biometric {biometric_id} for resubmission")

                current_state = self.state_service.get_verification_state(user_identity_id)
                current_seq = self.state_service.get_sequence_no(user_identity_id)
                return SequentialJobResponse(
                    result=True,
                    job_id=job_id,
                    verification_state=current_state,
                    sequence_no=current_seq,
                    processing_time_seconds=round(time.time() - start_time, 2),
                    extracted_data=extracted_data,
                    forgery_checks=forgery_checks,
                    other_checks=other_checks,
                    user_identity_id=user_identity_id
                )

            # ========================================
            # At this point, ALL verification checks have passed
            # Now we create user_identity (if needed) and user_keys
            # ========================================

            # Step 4: Create user_identity if needed (ONLY after ALL checks pass)
            if not user_identity_id:
                user_identity_id = self.user_identity_repo.create_empty_identity()
                if not user_identity_id:
                    self.logger.error("Failed to create user identity")
                    return SequentialJobResponse(
                        result=False,
                        job_id=job_id,
                        verification_state=0,
                        sequence_no=0,
                        processing_time_seconds=round(time.time() - start_time, 2),
                        error='Failed to create user identity',
                        error_code=DocumentErrorCode.PROCESSING_ERROR,
                        extracted_data=extracted_data,
                        forgery_checks=forgery_checks,
                        other_checks=other_checks
                    )
                self.logger.info(f"Created new user_identity: {user_identity_id[:16]}...")

            # Step 5: Insert face biometric (skip for multi-device link)
            if verification_result.face_embedding and not is_multi_device_link:
                try:
                    biometric_id = face_biometrics_repo.create_face_biometric(
                        user_identity_id=user_identity_id,
                        face_embedding=verification_result.face_embedding
                    )
                    if biometric_id:
                        self.logger.info(f"Stored face biometric {biometric_id}")
                except DuplicateFaceError as e:
                    # This shouldn't happen since we checked above, but handle just in case
                    self.logger.error(f"Duplicate face detected: {str(e)}")
                    return SequentialJobResponse(
                        result=False,
                        job_id=job_id,
                        verification_state=0,
                        sequence_no=0,
                        processing_time_seconds=round(time.time() - start_time, 2),
                        error="Duplicate face detected - this face is already registered to another user",
                        error_code=DocumentErrorCode.SELFIE_DUPLICATE_FACE,
                        extracted_data=extracted_data,
                        forgery_checks=forgery_checks,
                        other_checks=other_checks,
                        user_identity_id=user_identity_id
                    )

            # Step 6: Insert into user_keys (after ALL verification passes)
            # Move from user_keys_pending to user_keys (after ALL verification passes)
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

            # Step 7: Mark OTP as verified (but DON'T delete yet - wait for user key creation)
            if verification_result.extracted_otp and otp_record.get('mobile_number'):
                self.otp_repository.mark_otp_verified(otp_record['mobile_number'])

                # Broadcast verification to peer instances
                if self.otp_broadcast_service:
                    try:
                        asyncio.create_task(self.otp_broadcast_service.broadcast_otp_verified(otp_record['mobile_number']))
                    except Exception as sync_error:
                        self.logger.error(f"Failed to broadcast OTP verification: {sync_error}")

            # Step 7.5: Delete OTP AFTER user key is successfully created
            # This prevents the authentication gap where OTP is deleted but user_key doesn't exist yet
            if otp_record.get('mobile_number'):
                deleted = self.otp_repository.delete_otp(otp_record['mobile_number'])
                if deleted:
                    self.logger.info(f"✅ OTP deleted AFTER user key creation for mobile: {otp_record['mobile_number']}")
                else:
                    self.logger.warning(f"Failed to delete OTP for mobile: {otp_record['mobile_number']}")

            # Increment verification state (0 -> 1) and sequence_no (0 -> 1)
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
                # Use SET instead of INCREMENT for consistency with per-device state
                self.user_identity_repo.set_verification_state(user_identity_id, new_state)
                self.user_identity_repo.set_sequence_no(user_identity_id, new_seq)

                self.logger.info(
                    f"First selfie submission for device - state updated to {new_state}. "
                    f"{'(Multi-device link)' if is_multi_device_link else '(New user)'}"
                )
            else:
                # Resubmission - keep current state and sequence
                # Only user_keys matters for device-specific state
                new_state = current_state
                new_seq = current_seq
                self.logger.info(f"Selfie resubmission (state={current_state}). State unchanged.")

            self.logger.info(f"Selfie processing completed. New state: {new_state}, sequence_no: {new_seq}")

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
            self.logger.error(f"Error processing selfie: {str(e)}")
            # Try to get user_identity_id for error response and state reversion
            user_key = self.user_key_repository.get_key_by_public_key(client_public_key)
            user_identity_id = user_key.get('user_identity_id') if user_key else None

            # Revert state if it was incremented (in user_keys for this device)
            if user_identity_id:
                current_state = self.state_service.get_verification_state(client_public_key)
                if current_state > 0:  # State was incremented past expected
                    # Revert state in user_keys for this device
                    self.user_key_repository.update_state_and_sequence(
                        user_public_key=client_public_key,
                        verification_state=0,
                        sequence_no=0
                    )
                    self.logger.info(f"Reverted verification state after failure: {current_state} -> 0")

            current_seq = self.state_service.get_sequence_no(client_public_key)
            current_state = self.state_service.get_verification_state(client_public_key)
            return SequentialJobResponse(
                result=False,
                job_id=job_id,
                verification_state=current_state,
                sequence_no=current_seq,
                processing_time_seconds=round(time.time() - start_time, 2),
                error=f"Error processing selfie: {str(e)}",
                error_code=DocumentErrorCode.PROCESSING_ERROR,
                user_identity_id=user_identity_id
            )

    def _store_user_key(self, mobile_number: str, country_code: str,
                       client_public_key: str, secret_share: str, user_identity_id: str):
        """
        DEPRECATED: User keys are now created during selfie submission (after verification).
        This method is kept for backward compatibility but should not be called.
        """
        self.logger.warning("_store_user_key is deprecated - user keys are created during selfie submission")

    def _validate_share_format(self, share: str) -> bool:
        """
        DEPRECATED: Share format validation is now done during OTP request.
        This method is kept for backward compatibility but should not be called.
        """
        self.logger.warning("_validate_share_format is deprecated - validation is done during OTP request")
        return True

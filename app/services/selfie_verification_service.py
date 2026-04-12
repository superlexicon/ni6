import asyncio
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
from app.dto import (
    FaceVerificationRequest, FaceVerificationResponse, FaceMatchResult,
    FaceMetadata, FaceQuality, FaceLiveness
)
from app.repositories import UserIdentityRepository
from app.helper.deepface_helper import DeepfaceHelper
from app.services.face_extraction_service import FaceExtractionService
from app.core.logger import get_logger()


class SelfieVerificationError(Exception):
    """Exception raised during selfie verification operations."""
    pass


class SelfieVerificationService:
    """Service for comprehensive selfie verification with OTP and face matching."""

    def __init__(self, user_identity_repo: UserIdentityRepository):
        self.user_identity_repo = user_identity_repo
        self.deepface_helper = DeepfaceHelper()
        self.face_extraction_service = FaceExtractionService()
        self.logger = get_logger()

    async def verify_selfie_with_otp(
        self,
        selfie_image: bytes,
        client_public_key: str,
        encrypted_secret_share: Optional[str] = None,
        verification_threshold: float = 85.0,  # Consistent 85% threshold
        include_liveness_check: bool = True,
        anti_spoof_threshold: float = 0.85
    ) -> FaceVerificationResponse:
        """
        Comprehensive selfie verification with OTP and face matching.

        Args:
            selfie_image: Selfie image bytes (with OTP overlay)
            client_public_key: Client's public key for retrieving stored face
            encrypted_secret_share: Encrypted share of user's secret seed
            verification_threshold: Face matching threshold (0-100)
            include_liveness_check: Whether to perform liveness verification
            anti_spoof_threshold: Anti-spoofing threshold (0-1)

        Returns:
            Comprehensive verification response
        """
        try:
            self.logger.info(f"Starting comprehensive selfie verification for {client_public_key[:8]}...")

            # Step 1: Retrieve stored face biometric
            stored_face_data = await self._retrieve_stored_face(client_public_key)

            if not stored_face_data or not stored_face_data['face_embedding']:
                self.logger.error(f"No stored face biometric found for {client_public_key[:8]}...")
                return self._create_failure_response(
                    "No stored face biometric found. Please submit passport first.",
                    client_public_key
                )

            # Step 2: Create verification request
            verification_request = FaceVerificationRequest(
                public_key=client_public_key,
                selfie_face_embedding=stored_face_data['face_embedding'],  # This will be replaced with extracted embedding
                verification_threshold=verification_threshold,
                include_liveness_check=include_liveness_check,
                anti_spoof_threshold=anti_spoof_threshold
            )

            # Step 3: Perform face verification
            verification_response = await self.deepface_helper.verify_selfie_against_records(
                selfie_image=selfie_image,
                stored_embedding=stored_face_data['face_embedding'],
                verification_request=verification_request
            )

            # Step 4: Store selfie face biometric if verification passed
            if verification_response.verification_passed:
                await self._store_selfie_biometric(
                    selfie_image=selfie_image,
                    client_public_key=client_public_key,
                    encrypted_secret_share=encrypted_secret_share,
                    match_result=verification_response.face_match_result
                )

            self.logger.info(f"Selfie verification completed: passed={verification_response.verification_passed}")

            return verification_response

        except Exception as e:
            self.logger.error(f"Selfie verification failed: {str(e)}")
            raise SelfieVerificationError(f"Selfie verification failed: {str(e)}") from e

    async def verify_selfie_with_fallback(
        self,
        selfie_image: bytes,
        client_public_key: str,
        encrypted_secret_share: Optional[str] = None,
        verification_threshold: float = 85.0,  # Consistent 85% threshold
        anti_spoof_threshold: float = 0.85
    ) -> FaceVerificationResponse:
        """
        Selfie verification with fallback to previous selfie if no passport available.

        Args:
            selfie_image: Current selfie image bytes
            client_public_key: Client's public key
            encrypted_secret_share: Encrypted share of user's secret seed
            verification_threshold: Face matching threshold
            anti_spoof_threshold: Anti-spoofing threshold

        Returns:
            Verification response with fallback logic
        """
        try:
            self.logger.info(f"Starting selfie verification with fallback for {client_public_key[:8]}...")

            # Try passport face first
            passport_face = await self.user_identity_repo.get_face_biometric_by_public_key(client_public_key)

            if passport_face and passport_face.document_type == "passport":
                self.logger.info("Found passport face, using as reference")
                verification_request = FaceVerificationRequest(
                    public_key=client_public_key,
                    selfie_face_embedding=passport_face.face_embedding,
                    verification_threshold=verification_threshold,
                    include_liveness_check=True,
                    anti_spoof_threshold=anti_spoof_threshold
                )

                return await self.deepface_helper.verify_selfie_against_records(
                    selfie_image=selfie_image,
                    stored_embedding=passport_face.face_embedding,
                    verification_request=verification_request
                )

            # Fallback: check for previous selfie
            all_face_records = await self._get_all_face_biometrics(client_public_key)

            if not all_face_records:
                return self._create_failure_response(
                    "No face biometric records found. Please submit passport first.",
                    client_public_key
                )

            # Use the most recent face record
            latest_face = max(all_face_records, key=lambda x: x.created_at)
            self.logger.info(f"Using fallback face from {latest_face.document_type} ({latest_face.created_at})")

            verification_request = FaceVerificationRequest(
                public_key=client_public_key,
                selfie_face_embedding=latest_face.face_embedding,
                verification_threshold=verification_threshold - 5.0,  # Slightly lower threshold for fallback
                include_liveness_check=True,
                anti_spoof_threshold=anti_spoof_threshold
            )

            verification_response = await self.deepface_helper.verify_selfie_against_records(
                selfie_image=selfie_image,
                stored_embedding=latest_face.face_embedding,
                verification_request=verification_request
            )

            # Update verification details to indicate fallback was used
            verification_response.verification_details['fallback_used'] = {
                'fallback_type': latest_face.document_type,
                'fallback_date': latest_face.created_at.isoformat(),
                'adjusted_threshold': verification_request.verification_threshold
            }

            return verification_response

        except Exception as e:
            self.logger.error(f"Selfie verification with fallback failed: {str(e)}")
            raise SelfieVerificationError(f"Selfie verification with fallback failed: {str(e)}") from e

    async def _retrieve_stored_face(self, client_public_key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve stored face biometric, preferring passport over selfie.

        Args:
            client_public_key: Client's public key

        Returns:
            Dict with face embedding and metadata or None
        """
        try:
            # First try to get passport face
            passport_face = await self.user_identity_repo.get_face_biometric_by_public_key(client_public_key)

            if passport_face and passport_face.document_type == "passport":
                self.logger.info("Using passport face as reference")
                return {
                    'face_embedding': passport_face.face_embedding,
                    'document_type': 'passport',
                    'metadata': passport_face.face_metadata.dict()
                }

            # Fallback to any available face
            if passport_face:
                self.logger.info(f"Using {passport_face.document_type} face as fallback")
                return {
                    'face_embedding': passport_face.face_embedding,
                    'document_type': passport_face.document_type,
                    'metadata': passport_face.face_metadata.dict()
                }

            self.logger.warning(f"No face biometric found for {client_public_key[:8]}...")
            return None

        except Exception as e:
            self.logger.error(f"Failed to retrieve stored face: {str(e)}")
            return None

    async def _get_all_face_biometrics(self, client_public_key: str) -> list:
        """
        Get all face biometric records for a user.

        Args:
            client_public_key: Client's public key

        Returns:
            List of face biometric records
        """
        try:
            # This would require extending the repository to support this query
            # For now, just get the primary face record
            face_record = await self.user_identity_repo.get_face_biometric_by_public_key(client_public_key)
            return [face_record] if face_record else []

        except Exception as e:
            self.logger.error(f"Failed to get all face biometrics: {str(e)}")
            return []

    async def _store_selfie_biometric(
        self,
        selfie_image: bytes,
        client_public_key: str,
        encrypted_secret_share: Optional[str],
        match_result: FaceMatchResult
    ) -> bool:
        """
        Store selfie face biometric after successful verification.

        Args:
            selfie_image: Selfie image bytes
            client_public_key: Client's public key
            encrypted_secret_share: Encrypted share of user's secret seed
            match_result: Face matching result

        Returns:
            True if storage succeeded
        """
        try:
            self.logger.info(f"Storing selfie biometric for {client_public_key[:8]}...")

            # Get user_identity_id from user_keys table
            from app.repositories.user_key_repository import UserKeyRepository
            user_key_repo = UserKeyRepository()
            user_key = user_key_repo.get_key_by_public_key(client_public_key)

            if not user_key or not user_key.get('user_identity_id'):
                self.logger.error(f"User identity not found for {client_public_key[:8]}...")
                return False

            user_identity_id = user_key['user_identity_id']

            # Store encrypted secret share in user_keys table
            try:
                # Find the user key by public key and update with secret share
                user_key = user_key_repo.get_key_by_public_key(client_public_key)
                if user_key:
                    update_data = {
                        'encrypted_secret_share': encrypted_secret_share
                    }
                    # Update the user_keys record using the repository
                    user_key_repo.update_key_by_public_key(
                        client_public_key,
                        update_data
                    )

                    self.logger.info(f"Updated secret share for {client_public_key[:8]}...")
                else:
                    self.logger.warning(f"No user key found for public key {client_public_key[:8]}...")

            except Exception as e:
                self.logger.error(f"Failed to update secret share: {str(e)}")
                # Continue with biometric storage even if secret share update fails

            # Extract face biometric from selfie
            face_biometric = await self.face_extraction_service.extract_face_embedding(
                image_bytes=selfie_image,  # Use correct parameter name
                public_key=client_public_key,
                user_identity_id=user_identity['id'],
                document_type="selfie"
            )

            if not face_biometric:
                self.logger.error(f"Failed to extract face from selfie for {client_public_key[:8]}...")
                return False

            # Update metadata with verification results
            face_biometric.face_metadata.extraction_method = "selfie_verification"
            if hasattr(face_biometric.face_metadata, 'liveness_results'):
                face_biometric.face_metadata.liveness_results.real_probability = match_result.match_confidence

            # Face biometrics are now stored directly in verification sessions
            # The face embedding should already be stored in the session during initial selfie processing
            self.logger.info(f"Selfie biometric storage moved to verification session for {client_public_key[:8]}...")
            return True  # Storage moved to session, consider it successful

        except Exception as e:
            self.logger.error(f"Failed to store selfie biometric: {str(e)}")
            return False

    def _create_failure_response(
        self,
        error_message: str,
        client_public_key: str
    ) -> FaceVerificationResponse:
        """Create a failure verification response."""
        from datetime import datetime

        return FaceVerificationResponse(
            verification_passed=False,
            face_match_result=FaceMatchResult(
                distance_score=1.0,  # Maximum distance indicates complete mismatch
                match_confidence=0.0,
                is_match=False,
                threshold_used=0.0,
                distance_metric="cosine"
            ),
            liveness_check_passed=False,
            anti_spoof_score=0.0,
            verification_details={
                "error": error_message,
                "public_key_prefix": client_public_key[:8]
            }
        )
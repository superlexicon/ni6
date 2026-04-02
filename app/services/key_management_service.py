"""
Key Management Service - Add/remove public keys with face verification.

This service handles adding and removing public keys for a user identity.
Both operations require face verification (same security as secret share recovery).

Architecture: Composition-based - uses abstracted services to avoid duplication:
- FaceMatchingService: Shared face matching logic
- KeyValidationService: Shared key validation logic
- SelfieVerificationFlow: Complete selfie verification pipeline
"""

from typing import Dict, Optional
import uuid
from app.services.face_matching_service import FaceMatchingService
from app.services.key_validation_service import KeyValidationService
from app.services.selfie_verification_flow import SelfieVerificationFlow
from app.config.verification_config import verification_settings
from app.repositories.user_key_repository import UserKeyRepository
from app.repositories.face_biometrics_repository import FaceBiometricsRepository, DuplicateFaceError
from app.core.key_injection import key_injection_manager
from app.core.logger import get_logger


class KeyManagementError(Exception):
    """Exception raised during key management operations"""
    pass


class KeyManagementService:
    """
    Add/remove public keys with face verification.

    Flow for add_public_key:
    1. Validate encrypting_public_key exists and has identity
    2. Get user_identity_id and face_embeddings
    3. Perform selfie verification (OTP → PhotoHolmes → anti-spoofing → face extraction)
    4. Match face against stored embeddings (FaceMatchingService)
    5. Validate new_public_key is not already registered
    6. Encrypt client_secret_share at rest
    7. Create new user_keys record linked to same user_identity_id
    8. Store face embedding for audit trail

    Flow for remove_public_key:
    1. Validate encrypting_public_key exists
    2. Get user_identity_id and face_embeddings
    3. Perform selfie verification
    4. Match face against stored embeddings
    5. Validate key to remove belongs to same user_identity_id
    6. Check user has at least 2 keys (prevent lockout)
    7. Delete the specified key
    8. Store removal face embedding for audit trail
    """

    def __init__(self):
        """
        Initialize key management service.
        """
        self.logger = get_logger()

        # REUSE: Existing face verification flow
        self.selfie_verification = SelfieVerificationFlow()

        # REUSE: Abstracted face matching service (no duplication)
        confidence_percent = verification_settings.secret_share_face_match_threshold
        self.face_matching = FaceMatchingService(
            confidence_threshold=confidence_percent
        )

        # REUSE: Abstracted key validation service (no duplication)
        self.key_validation = KeyValidationService()

        # REUSE: Existing repositories
        self.user_key_repo = UserKeyRepository()
        self.face_biometrics_repo = FaceBiometricsRepository()

        # REUSE: At-rest encryption
        self.key_injection_manager = key_injection_manager

        self.logger.info("KeyManagementService initialized")

    async def add_public_key(
        self,
        encrypting_public_key: str,
        new_public_key: str,
        client_secret_share: str,
        selfie_bytes: bytes,
        otp_code: str,
        filename: Optional[str] = None
    ) -> Dict:
        """
        Add a new public key with face verification.

        Args:
            encrypting_public_key: Existing key that encrypted the request
            new_public_key: New public key to register
            client_secret_share: Client's secret share to store (plaintext or base64)
            selfie_bytes: Selfie image bytes for verification
            otp_code: OTP code for validation
            filename: Optional filename (may contain OTP)

        Returns:
            Dict with success status, new key details, and metadata

        Raises:
            KeyManagementError: If add operation fails at any step
        """
        self.logger.info(
            f"Starting add_public_key: encrypting={encrypting_public_key[:16]}..., "
            f"new_key={new_public_key[:16]}..."
        )

        # Step 1: Validate encrypting_public_key exists and has identity
        exists, user_key_data, error = self.key_validation.validate_key_exists_with_identity(
            encrypting_public_key
        )
        if not exists:
            raise KeyManagementError(
                f"Encrypting key validation failed: {error}"
            )

        user_identity_id = user_key_data.get('user_identity_id')
        self.logger.info(f"User identity: {user_identity_id[:16]}...")

        # Step 2: Validate user has completed verification
        is_valid, error = self.key_validation.validate_user_has_completed_verification_by_identity_id(
            user_identity_id
        )
        if not is_valid:
            raise KeyManagementError(f"User verification check failed: {error}")

        # Step 3: Get stored face embeddings
        recovery_data = self.user_key_repo.get_key_with_identity_embeddings(encrypting_public_key)
        stored_embeddings = recovery_data.get('face_embeddings', [])

        if not stored_embeddings:
            raise KeyManagementError(
                "No face biometrics found. Cannot verify identity for key addition."
            )

        self.logger.info(f"Found {len(stored_embeddings)} stored face embeddings")

        # Step 4: Perform selfie verification
        verification_result = await self.selfie_verification.verify_selfie(
            selfie_bytes=selfie_bytes,
            public_key=encrypting_public_key,
            otp_code=otp_code,
            filename=filename,
            require_otp=True
        )

        if not verification_result.success:
            raise KeyManagementError(
                f"Selfie verification failed: {verification_result.error}"
            )

        self.logger.info("Selfie verification passed all checks")

        # Step 5: Extract face embedding from verification result
        new_embedding = verification_result.face_embedding

        if not new_embedding:
            raise KeyManagementError(
                "Face extraction failed - no valid face detected in selfie"
            )

        self.logger.info(f"Extracted face embedding (dim: {len(new_embedding)})")

        # Step 6: Match face against stored embeddings
        match_result = await self.face_matching.match_embedding(
            new_embedding,
            stored_embeddings
        )

        if not match_result['match_found']:
            raise KeyManagementError(
                f"Face verification failed - no matching face found. "
                f"Best similarity: {match_result['best_similarity']*100:.1f}%, "
                f"threshold: {self.face_matching.face_match_confidence_threshold}%"
            )

        self.logger.info(
            f"Face match found! Similarity: {match_result['best_similarity']:.3f}, "
            f"matched against embedding from {match_result['matched_date']}"
        )

        # Step 7: Validate new_public_key is not already registered
        is_unique, error = self.key_validation.validate_public_key_unique(new_public_key)
        if not is_unique:
            raise KeyManagementError(f"New public key validation failed: {error}")

        # Step 8: Validate client_secret_share is provided
        if not client_secret_share:
            raise KeyManagementError("Client secret share is required")

        # Step 9: Encrypt client_secret_share at rest
        try:
            encrypted_secret_share = self.key_injection_manager.encrypt_data(
                client_secret_share
            )
            self.logger.info("Encrypted secret share for storage")
        except Exception as e:
            self.logger.error(f"Failed to encrypt secret share: {str(e)}")
            raise KeyManagementError(f"Encryption failed: {str(e)}")

        # Step 10: Get mobile number and country code from existing key
        mobile_number = user_key_data.get('mobile_number')
        country_code = user_key_data.get('country_code')

        # Step 11: Create new user_keys record
        key_data = {
            'mobile_number': mobile_number,
            'country_code': country_code,
            'user_public_key': new_public_key,
            'encrypted_secret_share': encrypted_secret_share,
            'user_identity_id': user_identity_id
        }

        result = self.user_key_repo.create_key(key_data)

        if not result:
            self.logger.error("Failed to store new key in database")
            raise KeyManagementError("Failed to store new public key")

        # Step 12: Store face embedding for audit trail
        await self._store_management_selfie(
            user_identity_id,
            new_embedding,
            operation='add'
        )

        self.logger.info(
            f"Successfully added public key: {new_public_key[:16]}... "
            f"for user {user_identity_id[:16]}..."
        )

        return {
            'success': True,
            'public_key': new_public_key,
            'user_identity_id': user_identity_id,
            'message': 'Public key added successfully',
            'face_match_confidence': match_result['best_similarity'],
            'faces_checked': match_result['embeddings_checked']
        }

    async def remove_public_key(
        self,
        encrypting_public_key: str,
        public_key_to_remove: str,
        selfie_bytes: bytes,
        otp_code: str,
        filename: Optional[str] = None
    ) -> Dict:
        """
        Remove a public key with face verification.

        Args:
            encrypting_public_key: Existing key that encrypted the request
            public_key_to_remove: Public key to delete
            selfie_bytes: Selfie image bytes for verification
            otp_code: OTP code for validation
            filename: Optional filename (may contain OTP)

        Returns:
            Dict with success status and metadata

        Raises:
            KeyManagementError: If remove operation fails at any step
        """
        self.logger.info(
            f"Starting remove_public_key: encrypting={encrypting_public_key[:16]}..., "
            f"key_to_remove={public_key_to_remove[:16]}..."
        )

        # Step 1: Validate encrypting_public_key exists and has identity
        exists, user_key_data, error = self.key_validation.validate_key_exists_with_identity(
            encrypting_public_key
        )
        if not exists:
            raise KeyManagementError(
                f"Encrypting key validation failed: {error}"
            )

        user_identity_id = user_key_data.get('user_identity_id')
        self.logger.info(f"User identity: {user_identity_id[:16]}...")

        # Step 2: Get stored face embeddings
        recovery_data = self.user_key_repo.get_key_with_identity_embeddings(encrypting_public_key)
        stored_embeddings = recovery_data.get('face_embeddings', [])

        if not stored_embeddings:
            raise KeyManagementError(
                "No face biometrics found. Cannot verify identity for key removal."
            )

        self.logger.info(f"Found {len(stored_embeddings)} stored face embeddings")

        # Step 3: Perform selfie verification
        verification_result = await self.selfie_verification.verify_selfie(
            selfie_bytes=selfie_bytes,
            public_key=encrypting_public_key,
            otp_code=otp_code,
            filename=filename,
            require_otp=True
        )

        if not verification_result.success:
            raise KeyManagementError(
                f"Selfie verification failed: {verification_result.error}"
            )

        self.logger.info("Selfie verification passed all checks")

        # Step 4: Extract face embedding from verification result
        new_embedding = verification_result.face_embedding

        if not new_embedding:
            raise KeyManagementError(
                "Face extraction failed - no valid face detected in selfie"
            )

        self.logger.info(f"Extracted face embedding (dim: {len(new_embedding)})")

        # Step 5: Match face against stored embeddings
        match_result = await self.face_matching.match_embedding(
            new_embedding,
            stored_embeddings
        )

        if not match_result['match_found']:
            raise KeyManagementError(
                f"Face verification failed - no matching face found. "
                f"Best similarity: {match_result['best_similarity']*100:.1f}%, "
                f"threshold: {self.face_matching.face_match_confidence_threshold}%"
            )

        self.logger.info(
            f"Face match found! Similarity: {match_result['best_similarity']:.3f}"
        )

        # Step 6: Validate key to remove belongs to the same user_identity_id
        is_valid, error = self.key_validation.validate_key_belongs_to_user_identity(
            public_key_to_remove,
            user_identity_id
        )
        if not is_valid:
            raise KeyManagementError(f"Key ownership validation failed: {error}")

        # Step 7: Check user has at least 2 keys before removal (prevent lockout)
        key_count = self._count_keys_for_user_identity(user_identity_id)
        if key_count < 2:
            raise KeyManagementError(
                f"Cannot remove last key. User has {key_count} key(s). "
                "At least one key must remain to prevent lockout."
            )

        self.logger.info(
            f"User has {key_count} key(s), allowing removal of {public_key_to_remove[:16]}..."
        )

        # Step 8: Delete the key
        success = self.user_key_repo.delete_key_by_public_key(
            public_key_to_remove
        )

        if not success:
            self.logger.error("Failed to delete key from database")
            raise KeyManagementError("Failed to remove public key")

        # Step 9: Store face embedding for audit trail
        await self._store_management_selfie(
            user_identity_id,
            new_embedding,
            operation='remove'
        )

        self.logger.info(
            f"Successfully removed public key: {public_key_to_remove[:16]}... "
            f"for user {user_identity_id[:16]}..."
        )

        return {
            'success': True,
            'public_key': public_key_to_remove,
            'message': 'Public key removed successfully',
            'face_match_confidence': match_result['best_similarity'],
            'faces_checked': match_result['embeddings_checked'],
            'remaining_keys': key_count - 1
        }

    async def _store_management_selfie(
        self,
        user_identity_id: str,
        face_embedding: list,
        operation: str
    ) -> bool:
        """
        Store the management operation's face embedding for audit trail.

        Args:
            user_identity_id: User identity ID
            face_embedding: Face embedding to store
            operation: Operation type ('add' or 'remove')

        Returns:
            True if successful, False otherwise
        """
        try:
            biometric_id = self.face_biometrics_repo.create_face_biometric(
                user_identity_id=user_identity_id,
                face_embedding=face_embedding
            )

            if biometric_id:
                self.logger.info(
                    f"Stored {operation} operation selfie embedding: {biometric_id}"
                )
                return True

            return False

        except DuplicateFaceError as e:
            # During key management, duplicate detection is a warning, not a failure
            # The face already exists, which is expected for the same user
            self.logger.warning(
                f"{operation} operation selfie matches existing embedding: {str(e)}"
            )
            return True  # Still consider operation successful

        except Exception as e:
            self.logger.warning(f"Failed to store {operation} selfie: {str(e)}")
            return False

    def _count_keys_for_user_identity(self, user_identity_id: str) -> int:
        """
        Count the number of keys associated with a user_identity_id.

        Used for safety checks (preventing removal of last key).

        Args:
            user_identity_id: User identity ID to count keys for

        Returns:
            Number of keys associated with the user
        """
        from app.core.db.database import get_db_connection_context
        try:
            with get_db_connection_context() as conn:
                cursor = conn.cursor()
                query = "SELECT COUNT(*) FROM user_keys WHERE user_identity_id = %s"
                cursor.execute(query, (user_identity_id,))
                count = cursor.fetchone()[0]
                cursor.close()
                return count if count else 0
        except Exception as e:
            self.logger.error(f"Failed to count keys for user: {str(e)}")
            return 0

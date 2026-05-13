"""
Key Recovery Service

This service handles key recovery with face verification:
1. Validate public key exists and has associated identity/face biometrics
2. Perform selfie verification (OTP, PhotoHolmes, anti-spoofing)
3. Match new selfie embedding against stored embeddings
4. Re-encrypt secret share with temporary public key using ECDH+Salsa20
"""

from typing import Dict, List, Optional
import os

from app.repositories.user_key_repository import UserKeyRepository
from app.repositories.face_biometrics_repository import FaceBiometricsRepository, DuplicateFaceError
from app.services.selfie_verification_flow import SelfieVerificationFlow
from app.core.key_injection import key_injection_manager
from app.core.key.scalsa20_crypto import Scalsa20Crypto
from app.core.key.secp256k1 import KeyPair
from app.core.logger import get_logger
from app.helper.face_recognition_factory import get_model_name
from app.dto import DocumentErrorCode


class KeyRecoveryError(Exception):
    """Exception raised during key recovery operations"""

    def __init__(self, message: str, error_code: Optional[str] = None):
        self.error_code = error_code
        super().__init__(message)


class KeyRecoveryService:
    """
    Service for key recovery with face verification.

    Flow:
    1. Validate public key exists, get identity + embeddings
    2. Perform selfie verification (OTP, PhotoHolmes, anti-spoofing)
    3. Match extracted face embedding against stored embeddings
    4. On match: decrypt secret share, re-encrypt with temp_public_key
    5. Return re-encrypted share

    Face matching uses cosine distance:
    - Cosine similarity = dot(embedding1, embedding2) / (norm1 * norm2)
    - Cosine distance = 1 - cosine_similarity
    - Lower distance = more similar faces (0 = identical, 1 = completely different)

    Threshold is configurable via SECRET_SHARE_FACE_MATCH_THRESHOLD env var:
    - Format: Confidence percentage (0-100), same as passport verification
    - Default: 70% confidence (cosine distance 0.3) - same strictness as passport
    - Conversion: cosine_distance = 1 - (confidence_percent / 100)

    Example:
    - 70% confidence → distance 0.3 ✓ (threshold)
    - 80% confidence → distance 0.2 ✓ (very confident)
    - 60% confidence → distance 0.4 ✗ (below threshold)
    """

    def __init__(self):
        self.logger = get_logger()
        self.user_key_repo = UserKeyRepository()
        self.face_biometrics_repo = FaceBiometricsRepository()
        self.selfie_verification = SelfieVerificationFlow()
        self.scalsa20_crypto = Scalsa20Crypto()
        self.key_pair = KeyPair()

        # Face matching uses find_matching_identity() from face_biometrics_repo
        # Configuration for error messages
        from app.config.verification_config import verification_settings
        self.face_match_confidence_threshold = verification_settings.secret_share_face_match_threshold

        # Supported video and image formats
        self.video_extensions = {'.mp4', '.mov', '.webm', '.avi', '.mkv'}
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
        self.video_magic_bytes = {
            b'ftyp': 'mp4',  # MP4/MOV (ISO Base Media File Format)
        }

    def _detect_file_type(self, filename: str, file_bytes: bytes) -> str:
        """
        Detect if file is image or video.

        Args:
            filename: Name of the file
            file_bytes: Raw file bytes for magic byte detection

        Returns:
            'video', 'image', or 'unknown'
        """
        # Check filename extension first
        _, ext = os.path.splitext(filename.lower())

        if ext in self.video_extensions:
            return 'video'
        if ext in self.image_extensions:
            return 'image'

        # Check magic bytes for video files
        # Look for ftyp marker (first 4 bytes after any leading bytes)
        if len(file_bytes) >= 8:
            # Check at position 4 (common for MP4/MOV)
            header_start = file_bytes[4:8]
            if header_start in self.video_magic_bytes:
                return 'video'

            # Check at position 0 (some formats)
            header_start = file_bytes[:4]
            if header_start in self.video_magic_bytes:
                return 'video'

        # Default to image if undetermined
        self.logger.warning(f"Could not detect file type for {filename}, defaulting to image")
        return 'image'

    async def recover_key(
        self,
        temp_public_key: str,
        selfie_bytes: bytes,
        otp_code: str,
        filename: Optional[str] = None,
        client_public_key: Optional[str] = None
    ) -> Dict:
        """
        Recover and re-encrypt ALL secret shares for the authenticated user's identity.

        The user is identified via face biometrics matching. All secret shares
        associated with the matched identity_id are returned (handles multiple devices).

        Args:
            temp_public_key: Temporary public key for re-encryption
            selfie_bytes: Selfie image bytes for verification
            otp_code: OTP code for validation
            filename: Optional filename (may contain OTP)
            client_public_key: Client's registered public key (for OTP validation)

        Returns:
            Dict containing:
            - identity_id: The matched identity ID
            - shares: List of re-encrypted secret shares with their metadata
            - total_shares: Number of shares recovered

        Raises:
            KeyRecoveryError: If recovery fails at any step
        """

        # Step 1: Perform selfie verification (OTP, PhotoHolmes, anti-spoofing)
        self.logger.info("Starting key recovery via face authentication")

        verification_result = await self.selfie_verification.verify_selfie(
            selfie_bytes=selfie_bytes,
            public_key=client_public_key,  # Use provided public_key for OTP validation
            otp_code=otp_code,
            filename=filename,
            require_otp=True
        )

        if not verification_result.success:
            raise KeyRecoveryError(
                f"Selfie verification failed: {verification_result.error}",
                error_code=verification_result.error_code
            )

        self.logger.info("Selfie verification passed all checks")

        # Step 2: Extract face embedding from verification result
        new_embedding = verification_result.face_embedding

        if not new_embedding:
            raise KeyRecoveryError(
                "Face extraction failed - no valid face detected in selfie"
            )

        self.logger.info(f"Extracted face embedding (dim: {len(new_embedding)})")

        # Step 3: Find matching identity and return identity_id
        # Convert confidence threshold to cosine distance threshold
        distance_threshold = 1.0 - (self.face_match_confidence_threshold / 100.0)

        match_result = self.face_biometrics_repo.find_matching_identity(
            new_embedding,
            distance_threshold=distance_threshold
        )

        if not match_result:
            raise KeyRecoveryError(
                f"No matching identity found. Face authentication failed. "
                f"Threshold: {self.face_match_confidence_threshold}% confidence"
            )

        identity_id = match_result['identity_id']
        similarity = match_result['similarity']

        self.logger.info(
            f"Face matched for identity_id: {identity_id[:16]}..., "
            f"similarity: {similarity*100:.1f}%"
        )

        # Step 4: Get ALL keys/shares for this identity_id
        # Filter by api_url if provided (only return shares from the specified API)
        all_keys = self.user_key_repo.get_keys_by_identity_id(identity_id, api_url=api_url)

        if not all_keys:
            raise KeyRecoveryError(f"No secret shares found for identity_id: {identity_id}")

        self.logger.info(f"Found {len(all_keys)} keys for identity {identity_id[:16]}...")

        # Step 5: Re-encrypt ALL shares to temp_public_key
        recovered_shares = []
        for user_key in all_keys:
            encrypted_share = user_key.get('encrypted_secret_share')
            if encrypted_share:
                try:
                    re_encrypted_share = await self._re_encrypt_share(
                        encrypted_share,
                        temp_public_key
                    )
                    recovered_shares.append({
                        'public_key': user_key['user_public_key'],
                        'encrypted_share': re_encrypted_share,  # ECIES envelope dict
                    })
                except Exception as e:
                    self.logger.warning(
                        f"Failed to re-encrypt share for {user_key['user_public_key'][:16]}...: {e}"
                    )
                    # Continue with other shares

        if not recovered_shares:
            raise KeyRecoveryError("Failed to re-encrypt any secret shares")

        self.logger.info(f"Successfully re-encrypted {len(recovered_shares)} secret shares")

        # Step 6: Store recovery selfie embedding for audit trail (optional)
        await self._store_recovery_selfie(
            identity_id,
            new_embedding
        )

        return {
            'success': True,
            'identity_id': identity_id,
            'shares': recovered_shares,
            'total_shares': len(recovered_shares),
            'message': f'Key recovery successful - recovered {len(recovered_shares)} share(s)',
            'face_match_confidence': similarity
        }

    async def _re_encrypt_share(
        self,
        encrypted_share: str,
        temp_public_key: str
    ) -> Dict[str, str]:
        """
        Decrypt secret share and ECIES-encrypt with temporary public key.

        Returns:
            Dict with ECIES envelope: {version, ephemeral_public_key, encrypted_data, iv}
        """
        try:
            # Debug: log temp_public_key format
            self.logger.info(f"temp_public_key length: {len(temp_public_key) if temp_public_key else 0}, value: {temp_public_key[:20] if temp_public_key else 'None'}...")

            # Step 1: Decrypt using server's symmetric key
            plaintext_share = key_injection_manager.decrypt_data(encrypted_share)
            self.logger.info("Decrypted secret share from storage")

            # Step 2: Generate ephemeral keypair for THIS share
            from ecdsa import SigningKey, SECP256k1
            ephemeral_private = SigningKey.generate(curve=SECP256k1)
            ephemeral_public = ephemeral_private.get_verifying_key()
            ephemeral_public_hex = ephemeral_public.to_string().hex()
            ephemeral_private_hex = ephemeral_private.to_string().hex()

            # Step 3: ECIES-encrypt with temp_public_key (not server's key)
            encrypted_result = self.scalsa20_crypto.encrypt_message(
                ephemeral_private_hex,  # Ephemeral private key as hex string
                temp_public_key,        # Client's temp public key
                plaintext_share
            )

            self.logger.info("ECIES-encrypted secret share with temp public key")

            # Return ECIES envelope (client can decrypt with temp_private_key)
            return {
                "version": "ecies_v1",
                "ephemeral_public_key": ephemeral_public_hex,
                "encrypted_data": encrypted_result.enc,
                "iv": encrypted_result.iv
            }

        except Exception as e:
            self.logger.error(f"Re-encryption failed: {str(e)}")
            raise KeyRecoveryError(f"Failed to re-encrypt secret share: {str(e)}")

    async def recover_key_with_identity(
        self,
        temp_public_key: str,
        selfie_bytes: bytes,
        otp_code: str,
        filename: Optional[str] = None,
        identity_id: Optional[str] = None,
        mobile_number: Optional[str] = None,
        api_url: Optional[str] = None
    ) -> Dict:
        """
        Recover and re-encrypt ALL secret shares for the authenticated user's identity.

        This method is used when the user doesn't have their original key.
        Supports both image and video selfie verification. The identity_id is
        discovered from face matching (same as recover_key flow).

        Args:
            temp_public_key: Temporary public key for re-encryption
            selfie_bytes: Selfie image or video bytes for verification
            otp_code: OTP code for validation
            filename: Optional filename (may contain OTP)
            identity_id: (Deprecated) Identity ID is now discovered from face matching
            mobile_number: User's mobile number (for logging/validation)

        Returns:
            Dict containing:
            - identity_id: The matched identity ID (discovered from face)
            - shares: List of re-encrypted secret shares with their metadata
            - total_shares: Number of shares recovered

        Raises:
            KeyRecoveryError: If recovery fails at any step
        """

        # Step 1: Perform selfie verification (OTP, PhotoHolmes, anti-spoofing)
        self.logger.info("Starting key recovery via face authentication (with identity_id)")

        # Detect file type (video vs image) for routing
        file_type = self._detect_file_type(filename or 'selfie.jpg', selfie_bytes)
        self.logger.info(f"Detected file type: {file_type}")

        # Route to appropriate verification method based on file type
        if file_type == 'video':
            # Ensure filename has video extension for validate_video_format()
            video_filename = filename
            if video_filename:
                _, ext = os.path.splitext(video_filename.lower())
                if ext not in self.video_extensions:
                    # Replace extension with .mp4 for validation
                    base_name = os.path.splitext(video_filename)[0]
                    video_filename = base_name + '.mp4'
                    self.logger.info(f"Adjusted filename for video processing: {filename} -> {video_filename}")
            else:
                video_filename = 'recovery_selfie.mp4'

            self.logger.info("Using video selfie verification with hand gesture OTP")
            # Verify video selfie using the registered public key (temp_public_key parameter)
            verification_result = await self.selfie_verification.verify_video_selfie(
                video_bytes=selfie_bytes,
                public_key=temp_public_key,  # Use registered public key for OTP lookup
                filename=video_filename,  # Use adjusted filename with video extension
                require_otp=True,
                skip_photoholmes=True  # Skip PhotoHolmes for video (faster)
            )
        else:
            self.logger.info("Using image selfie verification with OCR OTP")
            # For OTP validation, we pass None as public_key for recovery mode
            # The new validate_otp() flow will:
            # 1. Lookup OTP by code (not public_key)
            # 2. Get mobile_number from OTP
            # 3. Get identity_id from user_keys using mobile_number
            verification_result = await self.selfie_verification.verify_selfie(
                selfie_bytes=selfie_bytes,
                public_key=None,  # No registered key for OTP validation - use recovery flow
                otp_code=otp_code,
                filename=filename,
                require_otp=True
            )

        if not verification_result.success:
            raise KeyRecoveryError(
                f"Selfie verification failed: {verification_result.error}",
                error_code=verification_result.error_code
            )

        self.logger.info("Selfie verification passed all checks")

        # Step 2: Extract face embedding from verification result
        new_embedding = verification_result.face_embedding

        if not new_embedding:
            raise KeyRecoveryError(
                "Face extraction failed - no valid face detected in selfie"
            )

        self.logger.info(f"Extracted face embedding (dim: {len(new_embedding)})")

        # Step 3: Find matching identity from face embedding (same as image flow)
        # Convert confidence threshold to cosine distance threshold
        distance_threshold = 1.0 - (self.face_match_confidence_threshold / 100.0)

        match_result = self.face_biometrics_repo.find_matching_identity(
            new_embedding,
            distance_threshold=distance_threshold
        )

        if not match_result:
            raise KeyRecoveryError(
                f"No matching identity found. Face authentication failed. "
                f"Threshold: {self.face_match_confidence_threshold}% confidence"
            )

        identity_id = match_result['identity_id']
        similarity = match_result['similarity']

        self.logger.info(
            f"Face matched for identity_id: {identity_id[:16]}..., "
            f"similarity: {similarity*100:.1f}%"
        )

        # Note: identity_id is now discovered from face, not from OTP
        # The OTP was already validated during selfie_verification

        # Step 4: Get ALL keys/shares for this identity_id
        # Filter by api_url if provided (only return shares from the specified API)
        all_keys = self.user_key_repo.get_keys_by_identity_id(identity_id, api_url=api_url)

        if not all_keys:
            raise KeyRecoveryError(f"No secret shares found for identity_id: {identity_id}")

        self.logger.info(f"Found {len(all_keys)} keys for identity {identity_id[:16]}...")

        # Step 5: Re-encrypt ALL shares to temp_public_key
        recovered_shares = []
        for user_key in all_keys:
            encrypted_share = user_key.get('encrypted_secret_share')
            if encrypted_share:
                try:
                    re_encrypted_share = await self._re_encrypt_share(
                        encrypted_share,
                        temp_public_key
                    )
                    recovered_shares.append({
                        'public_key': user_key['user_public_key'],
                        'encrypted_share': re_encrypted_share,  # ECIES envelope dict
                    })
                except Exception as e:
                    self.logger.warning(
                        f"Failed to re-encrypt share for {user_key['user_public_key'][:16]}...: {e}"
                    )
                    # Continue with other shares

        if not recovered_shares:
            raise KeyRecoveryError("Failed to re-encrypt any secret shares")

        self.logger.info(f"Successfully re-encrypted {len(recovered_shares)} secret shares")

        # Step 6: Store recovery selfie embedding for audit trail (optional)
        await self._store_recovery_selfie(
            identity_id,
            new_embedding
        )

        return {
            'success': True,
            'identity_id': identity_id,
            'shares': recovered_shares,
            'total_shares': len(recovered_shares),
            'message': f'Key recovery successful - recovered {len(recovered_shares)} share(s)',
            'face_match_confidence': similarity
        }

    async def _store_recovery_selfie(
        self,
        user_identity_id: str,
        face_embedding: List[float]
    ) -> bool:
        """
        Store the recovery selfie's face embedding for audit trail.
        """
        try:
            biometric_id = self.face_biometrics_repo.create_face_biometric(
                user_identity_id=user_identity_id,
                face_embedding=face_embedding,
                model_name=get_model_name()
            )

            if biometric_id:
                self.logger.info(f"Stored recovery selfie embedding: {biometric_id}")
                return True

            return False

        except DuplicateFaceError as e:
            # During recovery, duplicate detection is a warning, not a failure
            # The face already exists, which is expected for the same user
            self.logger.warning(f"Recovery selfie matches existing embedding: {str(e)}")
            return True  # Still consider recovery successful

        except Exception as e:
            self.logger.warning(f"Failed to store recovery selfie: {str(e)}")
            return False

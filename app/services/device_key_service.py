from typing import Dict
from app.core.key.ecdsa_recovery import ECDSARecovery
from app.core.key_injection import key_injection_manager
from app.repositories.user_key_repository import UserKeyRepository
from app.services.verification_state_service import VerificationStateService
from app.core.logger import get_logger


class DeviceKeyService:
    """Service for managing device key operations"""

    def __init__(
        self,
        user_key_repository: UserKeyRepository,
        verification_state_service: VerificationStateService
    ):
        self.user_key_repository = user_key_repository
        self.verification_state_service = verification_state_service
        self.logger = get_logger()
        self.ecdsa_recovery = ECDSARecovery()

    async def add_device_key(
        self,
        new_public_key: str,
        secret_share: str,
        signature_r: str,
        signature_s: str
    ) -> Dict:
        """
        Add a new device key after verifying signature and user status.

        Steps:
        1. Recover public key from signature
        2. Find user with matching existing public key
        3. Verify user has completed all verification steps
        4. Encrypt secret share with server's public key
        5. Store new device key in user_keys table

        Args:
            new_public_key: New device's public key (hex)
            secret_share: Plaintext Shamir share
            signature_r: Signature r component
            signature_s: Signature s component

        Returns:
            Dict with success status, device info, and message

        Raises:
            ValueError: If signature invalid, user not found, or not verified
        """

        # Step 1: Recover public key from signature
        self.logger.info(f"Attempting public key recovery from signature")
        recovered_public_key = self.ecdsa_recovery.recover_public_key_from_signature(
            message=new_public_key,
            r=signature_r,
            s=signature_s
        )

        if not recovered_public_key:
            self.logger.error(f"Public key recovery failed - invalid signature")
            raise ValueError("Invalid signature - could not recover public key")

        self.logger.info(f"Recovered public key: {recovered_public_key[:16]}...")

        # Step 2: Find user with matching existing public key
        existing_key_record = self.user_key_repository.get_key_by_public_key(
            recovered_public_key
        )

        if not existing_key_record:
            self.logger.error(f"No user found with recovered public key")
            raise ValueError("Authentication failed - public key not found in system")

        mobile_number = existing_key_record.get('mobile_number')
        country_code = existing_key_record.get('country_code')
        self.logger.debug(f"User identified: {mobile_number}")

        # Step 3: Verify user has completed all verification steps
        # State is derived from user_identity_index data
        state = self.verification_state_service.get_verification_state(recovered_public_key)

        if state != 'completed':
            self.logger.error(
                f"User not fully verified - current state: {state}"
            )
            raise ValueError(
                f"Verification incomplete - current state: {state}. "
                "Complete all three steps (selfie, passport, bank statement) first."
            )

        self.logger.info(f"User verification status confirmed: COMPLETED")

        # Step 4: Encrypt secret share with server's public key
        try:
            encrypted_secret_share = key_injection_manager.encrypt_data(secret_share)
            self.logger.info(f"Encrypted secret share for storage")
        except Exception as e:
            self.logger.error(f"Failed to encrypt secret share: {str(e)}")
            raise ValueError(f"Encryption failed: {str(e)}")

        # Step 5: Store new device key
        key_data = {
            'mobile_number': mobile_number,
            'country_code': country_code,
            'user_public_key': new_public_key,
            'encrypted_secret_share': encrypted_secret_share,
            'user_identity_id': existing_key_record.get('user_identity_id')
        }

        result = self.user_key_repository.create_key(key_data)

        if not result:
            self.logger.error(f"Failed to store device key in database")
            raise ValueError("Failed to store device key")

        self.logger.info(
            f"Successfully added device key: {new_public_key[:16]}... "
            f"for user {country_code}{mobile_number}"
        )

        return {
            'success': True,
            'public_key': new_public_key,
            'created_at': result.get('created_at'),
            'message': "Device key added successfully"
        }

"""
Key Validation Service - Shared key validation logic.

This service extracts duplicated key validation logic from DeviceKeyService
and provides a reusable implementation for all services that need to validate
public keys, user verification status, and device identifiers.
"""

from typing import Dict, Optional, Tuple
from app.repositories.user_key_repository import UserKeyRepository
from app.repositories.user_identity_repository import UserIdentityRepository
from app.services.verification_state_service import VerificationStateService
from app.core.logger import get_logger


class KeyValidationError(Exception):
    """Exception raised during key validation operations"""
    pass


class KeyValidationService:
    """
    Reusable key validation logic.

    Eliminates code duplication across DeviceKeyService, KeyManagementService,
    and other services that need key validation.
    """

    def __init__(self):
        """
        Initialize key validation service.
        """
        self.logger = get_logger()
        self.user_key_repo = UserKeyRepository()
        self.user_identity_repo = UserIdentityRepository()
        self.verification_state_service = VerificationStateService()

    def validate_key_exists_with_identity(
        self,
        public_key: str
    ) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Validate public key exists and return user data.

        Checks if a public key exists in the user_keys table and has an
        associated user_identity_id.

        Args:
            public_key: Public key to validate (hex string)

        Returns:
            Tuple of (exists, user_key_data, error_message)
            - exists: True if key exists with identity, False otherwise
            - user_key_data: Dict with user key data if found, None otherwise
            - error_message: Error message if validation fails, None otherwise
        """
        user_key = self.user_key_repo.get_key_by_public_key(public_key)

        if not user_key:
            self.logger.error(f"Public key not found in system: {public_key[:16]}...")
            return False, None, "Public key not found in system"

        user_identity_id = user_key.get('user_identity_id')
        if not user_identity_id:
            self.logger.error(f"Public key has no associated identity: {public_key[:16]}...")
            return False, user_key, "Public key not linked to user identity"

        self.logger.info(
            f"Public key validated: {public_key[:16]}... "
            f"→ user_identity_id: {user_identity_id[:16]}..."
        )

        return True, user_key, None

    def validate_user_has_completed_verification(
        self,
        public_key: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate user has completed verification (state = 3).

        Checks if the user associated with the public key has completed
        all verification steps (selfie, passport, bank statement).

        Args:
            public_key: Public key to check (hex string)

        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if user has completed verification, False otherwise
            - error_message: Error message if validation fails, None otherwise
        """
        state = self.verification_state_service.get_verification_state(public_key)

        # State 3 = completed
        if state != 3:
            step_names = {0: 'selfie', 1: 'passport', 2: 'bank_statement'}
            expected_step = step_names.get(state, 'verification')

            self.logger.error(
                f"User not fully verified for public key {public_key[:16]}... "
                f"- current state: {state}"
            )
            return False, (
                f"Verification incomplete - please complete {expected_step} step first. "
                f"Current state: {state}"
            )

        self.logger.info(f"User verification confirmed as completed for: {public_key[:16]}...")

        return True, None

    def validate_user_has_completed_verification_by_identity_id(
        self,
        user_identity_id: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate user has completed verification by user_identity_id.

        Args:
            user_identity_id: User identity ID to check

        Returns:
            Tuple of (is_valid, error_message)
        """
        state = self.user_identity_repo.get_verification_state(user_identity_id)

        # State 3 = completed
        if state != 3:
            step_names = {0: 'selfie', 1: 'passport', 2: 'bank_statement'}
            expected_step = step_names.get(state, 'verification')

            self.logger.error(
                f"User not fully verified for identity {user_identity_id[:16]}... "
                f"- current state: {state}"
            )
            return False, (
                f"Verification incomplete - please complete {expected_step} step first. "
                f"Current state: {state}"
            )

        self.logger.info(f"User verification confirmed as completed for identity: {user_identity_id[:16]}...")

        return True, None

    def validate_public_key_unique(
        self,
        public_key: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate public key doesn't already exist.

        Args:
            public_key: Public key to check

        Returns:
            Tuple of (is_unique, error_message)
            - is_unique: True if public key is unique, False if exists
            - error_message: Error message if validation fails, None otherwise
        """
        existing_key = self.user_key_repo.get_key_by_public_key(public_key)

        if existing_key:
            self.logger.error(f"Public key already exists: {public_key[:16]}...")
            return False, f"Public key already registered in the system"

        self.logger.info(f"Public key validated as unique: {public_key[:16]}...")

        return True, None

    def validate_key_belongs_to_user_identity(
        self,
        public_key: str,
        expected_user_identity_id: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate a public key belongs to a specific user_identity_id.

        Used for security checks (e.g., ensuring key to remove belongs to
        the same user as the encrypting key).

        Args:
            public_key: Public key to validate
            expected_user_identity_id: Expected user_identity_id

        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if key belongs to expected user, False otherwise
            - error_message: Error message if validation fails, None otherwise
        """
        user_key = self.user_key_repo.get_key_by_public_key(public_key)

        if not user_key:
            self.logger.error(f"Public key not found: {public_key[:16]}...")
            return False, "Public key not found in system"

        actual_user_identity_id = user_key.get('user_identity_id')

        if actual_user_identity_id != expected_user_identity_id:
            self.logger.error(
                f"Key ownership mismatch: key {public_key[:16]}... "
                f"belongs to {actual_user_identity_id[:16] if actual_user_identity_id else 'None'}, "
                f"expected {expected_user_identity_id[:16]}..."
            )
            return False, "Public key belongs to a different user"

        self.logger.info(
            f"Key ownership validated: {public_key[:16]}... "
            f"→ {expected_user_identity_id[:16]}..."
        )

        return True, None

    def count_keys_for_user_identity(
        self,
        user_identity_id: str
    ) -> int:
        """
        Count the number of keys associated with a user_identity_id.

        Used for safety checks (e.g., preventing removal of last key).

        Args:
            user_identity_id: User identity ID to check

        Returns:
            Number of keys associated with the user
        """
        # Use the repository to count keys by user_identity_id
        # This method may need to be added to UserKeyRepository
        keys = self.user_key_repo.get_keys_by_user_identity_id(user_identity_id)
        count = len(keys) if keys else 0

        self.logger.info(
            f"User {user_identity_id[:16]}... has {count} key(s)"
        )

        return count

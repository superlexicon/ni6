"""
Verification State Service - Uses verification_state column from user_keys table.

Multi-Device Support:
- Per-device state tracking: Each device (client_public_key) has its own verification_state
- user_keys table: Stores per-device state (verification_state, sequence_no)
- user_identity_index table: Stores best unexpired state across devices (overall identity state)

State values:
- 0: Initial (ready for selfie)
- 1: After selfie (ready for passport)
- 2: After passport (ready for bank statement)
- 3: Complete (bank statement submitted)

Note: Document submissions are allowed at ANY state - repeated submissions
do not cause state regression on failure.

Smart Parameter Detection:
- Automatically detects whether the input is a client_public_key or user_identity_id
- UUIDs contain dashes (e.g., "b866286c-1b63-43..."), public keys don't
- This provides backward compatibility with existing call sites
"""

from typing import Dict, Any, Tuple, Optional
from app.core.logger import get_logger


class VerificationStateService:
    """Manages verification state from user_identity_index.verification_state column"""

    # Expected verification_state for each document type
    EXPECTED_STATE = {
        'selfie': 0,           # State 0 means ready for selfie
        'video_selfie': 0,     # State 0 means ready for video selfie (alternative to selfie)
        'passport': 1,         # State 1 means selfie done, ready for passport
        'bank_statement': 2,   # State 2 means passport done, ready for bank statement
        'tax_return': 1,       # State 1 means selfie done, ready for tax return (optional)
        'national_id': 1,      # Alternative to passport - at state 1 (selfie done)
        'driving_license': 1,  # Alternative to passport - at state 1 (selfie done)
        'resume': 1,           # Optional additional doc - at state 1 (selfie done)
    }

    # Minimum sequence_no required for each document type
    EXPECTED_SEQUENCE = {
        'selfie': 0,
        'video_selfie': 0,     # Same as selfie
        'passport': 1,
        'bank_statement': 2,   # Requires passport data extraction (sequence_no >= 2)
        'tax_return': 1,
        'national_id': 1,      # Alternative to passport
        'driving_license': 1,  # Alternative to passport
        'resume': 1,
    }

    # Independent documents that don't require state validation or increment
    INDEPENDENT_DOCUMENTS = {
        'tax_statement',      # Legacy tax statement (independent)
        'id_card',            # ID card can be submitted independently
        'add_public_key',
        'remove_public_key'
    }

    def __init__(self):
        self.logger = get_logger()
        # Lazy load repositories to avoid circular import
        self._user_key_repo = None
        self._user_identity_repo = None

    @property
    def user_key_repo(self):
        if self._user_key_repo is None:
            from app.repositories.user_key_repository import UserKeyRepository
            self._user_key_repo = UserKeyRepository()
        return self._user_key_repo

    @property
    def user_identity_repo(self):
        if self._user_identity_repo is None:
            from app.repositories.user_identity_repository import UserIdentityRepository
            self._user_identity_repo = UserIdentityRepository()
        return self._user_identity_repo

    def get_verification_state(self, client_public_key: str) -> int:
        """
        Get current verification state for a device.

        Multi-Device Support: Returns the verification_state from user_keys
        table for this specific client_public_key (per-device state).

        Smart Parameter Detection: Automatically detects whether the input
        is a client_public_key or user_identity_id (UUID format with dashes) for
        backward compatibility with existing call sites.

        Args:
            client_public_key: Client's public key (device-specific) OR
                              user_identity_id (UUID format with dashes)

        Returns:
            State integer: 0=initial, 1=selfie done, 2=passport done, 3=complete
        """
        try:
            # Detect if input is user_identity_id (UUID format) or client_public_key
            # UUIDs contain dashes (e.g., "b866286c-1b63-43..."), public keys don't
            is_user_identity_id = '-' in str(client_public_key)

            if is_user_identity_id:
                # Input is user_identity_id - direct lookup for backward compatibility
                self.logger.debug(
                    f"Detected user_identity_id format, using direct lookup: {client_public_key[:16]}..."
                )
                user_identity_id = client_public_key
                return self.user_identity_repo.get_verification_state(user_identity_id)

            # Input is client_public_key - get per-device state from user_keys
            device_state = self.user_key_repo.get_verification_state(client_public_key)

            self.logger.debug(
                f"Device verification_state for {client_public_key[:16]}...: {device_state}"
            )
            return device_state

        except Exception as e:
            self.logger.error(f"Error getting verification state: {str(e)}")
            return 0

    def validate_document_submission(
        self,
        client_public_key: str,
        document_type: str
    ) -> Tuple[bool, str, bool]:
        """
        Validate if document type can be submitted.

        NOW: Always allows submission (no state blocking).
        Only validates that document_type is known.

        Args:
            client_public_key: Client's public key
            document_type: Type of document ('selfie', 'passport', 'bank_statement', 'tax_statement', etc.)

        Returns:
            Tuple of (is_valid, error_message, is_resubmission)
        """
        # Independent documents skip all validation (unchanged)
        if document_type in self.INDEPENDENT_DOCUMENTS:
            return True, "", False

        # Special handling for selfie/video_selfie - always allow
        if document_type in ('selfie', 'video_selfie'):
            return True, "", False

        # All other document types - always allow (removed state check)
        known_types = {
            'passport', 'national_id', 'driving_license',
            'bank_statement', 'tax_return', 'resume'
        }
        if document_type in known_types:
            return True, "", False

        return False, f"Unknown document type: {document_type}", False

    def get_user_identity_id(self, client_public_key: str) -> Optional[str]:
        """
        Get user_identity_id for a given public key.

        Args:
            client_public_key: Client's public key

        Returns:
            user_identity_id or None if not found
        """
        user_key = self.user_key_repo.get_key_by_public_key(client_public_key)
        if user_key:
            return user_key.get('user_identity_id')
        return None

    def increment_state(self, user_identity_id: str) -> int:
        """
        Increment verification state after successful document processing.

        Args:
            user_identity_id: User identity ID

        Returns:
            New state after increment
        """
        return self.user_identity_repo.increment_verification_state(user_identity_id)

    def get_sequence_no(self, client_public_key: str) -> int:
        """
        Get current sequence_no for a device.

        Multi-Device Support: Returns the sequence_no from user_keys
        table for this specific client_public_key (per-device state).

        Args:
            client_public_key: Client's public key

        Returns:
            Sequence number: 0=initial, 1=selfie done, 2=passport data extracted, 3=complete
        """
        try:
            # Detect if input is user_identity_id (UUID format) or client_public_key
            is_user_identity_id = '-' in str(client_public_key)

            if is_user_identity_id:
                # Input is user_identity_id - use user_identity_index
                user_identity_id = client_public_key
                return self.user_identity_repo.get_sequence_no(user_identity_id)

            # Input is client_public_key - get per-device state from user_keys
            return self.user_key_repo.get_sequence_no(client_public_key)

        except Exception as e:
            self.logger.error(f"Error getting sequence_no: {str(e)}")
            return 0

    def increment_sequence_no(self, user_identity_id: str) -> int:
        """
        Increment sequence_no after document data extraction.

        Args:
            user_identity_id: User identity ID

        Returns:
            New sequence_no after increment
        """
        return self.user_identity_repo.increment_sequence_no(user_identity_id)

    def set_sequence_no(self, user_identity_id: str, sequence_no: int) -> bool:
        """
        Set sequence_no to a specific value.

        Args:
            user_identity_id: User identity ID
            sequence_no: New sequence number (0-3)

        Returns:
            True if updated successfully
        """
        return self.user_identity_repo.set_sequence_no(user_identity_id, sequence_no)

    def set_verification_state(self, user_identity_id: str, verification_state: int) -> bool:
        """
        Set verification_state to a specific value.

        Args:
            user_identity_id: User identity ID
            verification_state: New verification state (0-3)

        Returns:
            True if updated successfully
        """
        return self.user_identity_repo.set_verification_state(user_identity_id, verification_state)

    def get_state_info(self, client_public_key: str) -> Dict[str, Any]:
        """
        Get detailed verification state information.

        Args:
            client_public_key: Client's public key

        Returns:
            Dict with:
            - state: Current verification state (0-3)
            - user_identity_id: User identity ID if exists
        """
        state = self.get_verification_state(client_public_key)
        user_identity_id = self.get_user_identity_id(client_public_key)

        return {
            'state': state,
            'user_identity_id': user_identity_id
        }

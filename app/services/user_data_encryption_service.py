"""
User Data Encryption Service

Provides per-user encryption/decryption of PII using ECDH+Salsa20.
Each user's data is encrypted with their public key, so only they can decrypt it.

Encryption Envelope Structure:
{
    "version": "ecdh_v1",
    "server_public_key": "<hex>",
    "encrypted_data": "<base64>",
    "iv": "<base64>"
}
"""

import json
from typing import Dict, Any, Optional
from app.core.key.scalsa20_crypto import Scalsa20Crypto
from app.core.key.secp256k1 import KeyPair
from app.dto import ServerKeyPair, EncryptedMessageData, DecryptedMessageData
from app.core import logger


class UserDataEncryptionService:
    """
    Service for encrypting/decrypting user PII using ECDH+Salsa20 asymmetric encryption.

    Pattern:
    - Server has its own ECDH keypair (generated deterministically from seed)
    - Each user has their own public key (stored in user_keys table)
    - Server encrypts PII with user's public key (using server's private key)
    - User decrypts with their private key (using server's public key)
    - Server cannot decrypt without user's private key (asymmetric guarantee)
    """

    ENCRYPTION_VERSION = "ecdh_v1"

    def __init__(self):
        self.scalsa20_crypto = Scalsa20Crypto()
        self._server_keys: Optional[ServerKeyPair] = None

    @property
    def server_keys(self) -> ServerKeyPair:
        """Lazy load server keypair (generated once per instance)"""
        if self._server_keys is None:
            self._server_keys = KeyPair.generate_secp256k1_keys()
            logger.info(f"UserDataEncryptionService: Generated server keypair: {self._server_keys.public_key[:16]}...")
        return self._server_keys

    @property
    def server_public_key(self) -> str:
        """Get server's public key (needed for user to decrypt)"""
        return self.server_keys.public_key

    @property
    def server_private_key(self) -> str:
        """Get server's private key (needed for server to encrypt)"""
        return self.server_keys.private_key

    def create_encryption_envelope(
        self,
        extracted_data: Dict[str, Any],
        client_public_key: str
    ) -> str:
        """
        Create an encryption envelope for extracted PII data.

        The envelope contains the encrypted data along with metadata needed for decryption.
        This is stored as JSON in the extracted_data_encrypted column.

        Args:
            extracted_data: PII data dict to encrypt (e.g., passport data)
            client_public_key: User's public key in hex format

        Returns:
            JSON string containing the encryption envelope

        Raises:
            ValueError: If encryption fails
        """
        if not extracted_data:
            return None

        if not client_public_key:
            raise ValueError("client_public_key is required for encryption")

        try:
            # Convert data to JSON for encryption
            message_json = json.dumps(extracted_data, separators=(',', ':'))

            # Encrypt using ECDH: server private + user public
            encrypted: EncryptedMessageData = self.scalsa20_crypto.encrypt_message(
                private_hex=self.server_private_key,
                public_hex=client_public_key,
                message=message_json
            )

            # Create encryption envelope
            envelope = {
                "version": self.ENCRYPTION_VERSION,
                "server_public_key": self.server_public_key,
                "encrypted_data": encrypted.enc,
                "iv": encrypted.iv
            }

            return json.dumps(envelope, separators=(',', ':'))

        except Exception as e:
            logger.error(f"Failed to create encryption envelope: {e}")
            raise ValueError(f"Encryption failed: {e}")

    def encrypt_for_user(
        self,
        extracted_data: Dict[str, Any],
        client_public_key: str
    ) -> Dict[str, str]:
        """
        Encrypt extracted data for a specific user.

        Convenience method that returns a dict instead of JSON string.

        Args:
            extracted_data: PII data dict to encrypt
            client_public_key: User's public key in hex format

        Returns:
            Dict with encryption envelope fields

        Example:
            encrypted = svc.encrypt_for_user(
                {"full_name": "John Doe", "passport_number": "AB1234567"},
                user_public_key
            )
            # Returns: {"version": "ecdh_v1", "server_public_key": "...", ...}
        """
        envelope_json = self.create_encryption_envelope(extracted_data, client_public_key)
        return json.loads(envelope_json)

    def decrypt_for_user(
        self,
        encrypted_envelope: Dict[str, str],
        user_private_key: str
    ) -> Dict[str, Any]:
        """
        Decrypt data that was encrypted for a user.

        Called by the client/user who has their private key.

        Args:
            encrypted_envelope: Encryption envelope dict with keys:
                - version: "ecdh_v1"
                - server_public_key: Server's public key
                - encrypted_data: Base64 encrypted data
                - iv: Base64 IV
            user_private_key: User's private key in hex format

        Returns:
            Decrypted PII data dict

        Raises:
            ValueError: If decryption fails or envelope is invalid

        Example:
            decrypted = svc.decrypt_for_user(encrypted_envelope, user_private_key)
            # Returns: {"full_name": "John Doe", "passport_number": "AB1234567"}
        """
        if not encrypted_envelope:
            raise ValueError("encrypted_envelope is required")

        if not user_private_key:
            raise ValueError("user_private_key is required for decryption")

        # Validate envelope format
        if encrypted_envelope.get("version") != self.ENCRYPTION_VERSION:
            raise ValueError(f"Unsupported encryption version: {encrypted_envelope.get('version')}")

        required_fields = ["server_public_key", "encrypted_data", "iv"]
        for field in required_fields:
            if field not in encrypted_envelope:
                raise ValueError(f"Missing required field in envelope: {field}")

        try:
            # Decrypt using ECDH: user private + server public
            decrypted: DecryptedMessageData = self.scalsa20_crypto.decrypt_message(
                private_hex=user_private_key,
                public_hex=encrypted_envelope["server_public_key"],
                b64_encrypted=encrypted_envelope["encrypted_data"],
                b64_iv=encrypted_envelope["iv"]
            )

            # Parse decrypted JSON
            return json.loads(decrypted.plain_text)

        except Exception as e:
            logger.error(f"Failed to decrypt user data: {e}")
            raise ValueError(f"Decryption failed: {e}")

    def decrypt_for_server(
        self,
        encrypted_envelope: Dict[str, str],
        client_public_key: str
    ) -> Dict[str, Any]:
        """
        Decrypt data for server-side operations (e.g., migration, validation).

        Note: This demonstrates that server can also decrypt with server private key.
        In production, user data should only be decrypted by the user.

        Args:
            encrypted_envelope: Encryption envelope dict
            client_public_key: User's public key in hex format

        Returns:
            Decrypted PII data dict

        Raises:
            ValueError: If decryption fails
        """
        if not encrypted_envelope:
            raise ValueError("encrypted_envelope is required")

        if not client_public_key:
            raise ValueError("client_public_key is required")

        try:
            # Decrypt using ECDH: server private + user public
            decrypted: DecryptedMessageData = self.scalsa20_crypto.decrypt_message(
                private_hex=self.server_private_key,
                public_hex=client_public_key,
                b64_encrypted=encrypted_envelope["encrypted_data"],
                b64_iv=encrypted_envelope["iv"]
            )

            return json.loads(decrypted.plain_text)

        except Exception as e:
            logger.error(f"Failed to decrypt data for server: {e}")
            raise ValueError(f"Decryption failed: {e}")


# Singleton instance for the application
_user_encryption_service: Optional[UserDataEncryptionService] = None


def get_user_encryption_service() -> UserDataEncryptionService:
    """Get the singleton user encryption service instance"""
    global _user_encryption_service
    if _user_encryption_service is None:
        _user_encryption_service = UserDataEncryptionService()
    return _user_encryption_service


def reset_user_encryption_service():
    """Reset the singleton (useful for testing)"""
    global _user_encryption_service
    _user_encryption_service = None

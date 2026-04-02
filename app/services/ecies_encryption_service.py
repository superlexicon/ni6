"""
ECIES Encryption Service for User-Only Decryption

Provides encryption of PII using ephemeral ECIES (Elliptic Curve Integrated Encryption Scheme).
Each encryption uses a new ephemeral keypair, ensuring ONLY the user can decrypt.

Encryption Envelope Structure:
{
    "version": "ecies_v1",
    "ephemeral_public_key": "<hex>",
    "encrypted_data": "<base64>",
    "iv": "<base64>"
}

Key Difference from ECDH (UserDataEncryptionService):
- ECDH: Uses server's static keypair → Both user AND server can decrypt ❌
- ECIES: Uses ephemeral keypair → Only user can decrypt ✅

How User-Only Decryption Works:
1. Server generates ephemeral keypair (new for each encryption)
2. Derive shared secret: ECDH(ephemeral_private, user_public)
3. Encrypt data with Salsa20 using shared secret
4. Store ephemeral_public_key + encrypted_data + iv
5. Discard ephemeral_private_key (not stored anywhere)

Result: Only user (with their private key) can derive the shared secret and decrypt.
"""

import json
from typing import Dict, Any, Optional, Union
from ecdsa import SigningKey, SECP256k1
from app.core.key.scalsa20_crypto import Scalsa20Crypto
from app.core import logger
from app.dto import EncryptedMessageData, DecryptedMessageData


class ECIESEncryptionService:
    """
    Service for encrypting PII using ECIES with ephemeral keys (user-only decryption).

    Pattern:
    - Generate ephemeral SECP256k1 keypair for each encryption (new key each time)
    - Derive shared secret via ECDH: ephemeral_private + user_public
    - Encrypt data with Salsa20 using derived shared secret
    - Store ephemeral_public_key + encrypted_data + iv
    - Discard ephemeral_private_key after encryption

    Only the user (with their private key) can decrypt:
    - User derives shared secret: ephemeral_public + user_private
    - Server cannot decrypt (ephemeral_private was discarded)
    """

    ENCRYPTION_VERSION = "ecies_v1"

    def __init__(self):
        self.scalsa20_crypto = Scalsa20Crypto()

    def _generate_ephemeral_keypair(self) -> tuple[str, str]:
        """
        Generate an ephemeral SECP256k1 keypair for this encryption only.

        Returns:
            Tuple of (private_key_hex, public_key_hex)
        """
        # Generate new ephemeral keypair
        ephemeral_private = SigningKey.generate(curve=SECP256k1)
        ephemeral_public = ephemeral_private.get_verifying_key()

        # Convert to hex
        private_key_hex = ephemeral_private.to_string().hex()
        public_key_hex = ephemeral_public.to_string().hex()

        logger.debug(f"Generated ephemeral keypair: {public_key_hex[:16]}...")
        return private_key_hex, public_key_hex

    def encrypt_for_user_only(
        self,
        data: Union[str, bytes, Dict[str, Any]],
        user_public_key: str
    ) -> Dict[str, str]:
        """
        Encrypt data so ONLY the user can decrypt (ephemeral key ECIES).

        Args:
            data: PII data to encrypt (str, bytes, or dict)
            user_public_key: User's public key in hex format

        Returns:
            Dict with encryption envelope fields:
            {
                "version": "ecies_v1",
                "ephemeral_public_key": "<hex>",
                "encrypted_data": "<base64>",
                "iv": "<base64>"
            }

        Raises:
            ValueError: If encryption fails
        """
        if not data:
            raise ValueError("Data is required for encryption")

        if not user_public_key:
            raise ValueError("user_public_key is required for encryption")

        try:
            # Convert data to JSON for encryption
            if isinstance(data, dict):
                message_json = json.dumps(data, separators=(',', ':'))
            elif isinstance(data, str):
                message_json = data
            elif isinstance(data, bytes):
                message_json = data.decode('utf-8')
            else:
                message_json = str(data)

            # Generate ephemeral keypair for THIS encryption only
            ephemeral_private, ephemeral_public = self._generate_ephemeral_keypair()

            # Encrypt using ECDH: ephemeral_private + user_public
            encrypted: EncryptedMessageData = self.scalsa20_crypto.encrypt_message(
                private_hex=ephemeral_private,
                public_hex=user_public_key,
                message=message_json
            )

            # Create ECIES encryption envelope
            envelope = {
                "version": self.ENCRYPTION_VERSION,
                "ephemeral_public_key": ephemeral_public,
                "encrypted_data": encrypted.enc,
                "iv": encrypted.iv
            }

            logger.info(f"Encrypted data with ECIES (ephemeral key: {ephemeral_public[:16]}...)")

            # Ephemeral private key is DISCARDED here - only stored in memory during encryption
            # This ensures server cannot decrypt later
            return envelope

        except Exception as e:
            logger.error(f"Failed to encrypt with ECIES: {e}")
            raise ValueError(f"ECIES encryption failed: {e}")

    def create_encryption_envelope(
        self,
        extracted_data: Dict[str, Any],
        user_public_key: str
    ) -> str:
        """
        Create an ECIES encryption envelope for extracted PII data.

        The envelope contains the encrypted data along with metadata needed for decryption.
        This is stored as JSON in the extracted_data_encrypted column.

        This method name matches UserDataEncryptionService.create_encryption_envelope()
        for easier migration.

        Args:
            extracted_data: PII data dict to encrypt (e.g., passport data)
            user_public_key: User's public key in hex format

        Returns:
            JSON string containing the ECIES encryption envelope

        Raises:
            ValueError: If encryption fails
        """
        if not extracted_data:
            return None

        envelope = self.encrypt_for_user_only(extracted_data, user_public_key)
        return json.dumps(envelope, separators=(',', ':'))

    def decrypt_for_user_only(
        self,
        envelope: Dict[str, str],
        user_private_key: str
    ) -> Dict[str, Any]:
        """
        Decrypt ECIES-encrypted data (user-side only).

        Called by the client/user who has their private key.

        Args:
            envelope: ECIES encryption envelope dict with keys:
                - version: "ecies_v1"
                - ephemeral_public_key: Ephemeral public key (hex)
                - encrypted_data: Base64 encrypted data
                - iv: Base64 IV
            user_private_key: User's private key in hex format

        Returns:
            Decrypted PII data dict

        Raises:
            ValueError: If decryption fails or envelope is invalid
        """
        if not envelope:
            raise ValueError("envelope is required")

        if not user_private_key:
            raise ValueError("user_private_key is required for decryption")

        # Validate envelope format
        if envelope.get("version") != self.ENCRYPTION_VERSION:
            raise ValueError(f"Unsupported encryption version: {envelope.get('version')}")

        required_fields = ["ephemeral_public_key", "encrypted_data", "iv"]
        for field in required_fields:
            if field not in envelope:
                raise ValueError(f"Missing required field in envelope: {field}")

        try:
            # Decrypt using ECDH: user_private + ephemeral_public
            decrypted: DecryptedMessageData = self.scalsa20_crypto.decrypt_message(
                private_hex=user_private_key,
                public_hex=envelope["ephemeral_public_key"],
                b64_encrypted=envelope["encrypted_data"],
                b64_iv=envelope["iv"]
            )

            # Parse decrypted JSON
            return json.loads(decrypted.plain_text)

        except Exception as e:
            logger.error(f"Failed to decrypt ECIES data: {e}")
            raise ValueError(f"ECIES decryption failed: {e}")

    def decrypt_message_from_client(
        self,
        envelope: Dict[str, str],
        server_private_key: str
    ) -> str:
        """
        Decrypt ECIES-encrypted message from client using server's static private key.

        Args:
            envelope: ECIES envelope with {version, ephemeral_public_key, encrypted_data, iv}
            server_private_key: Server's private key (hex) from KeyPair.generate_secp256k1_keys()

        Returns:
            Decrypted plaintext message
        """
        if envelope.get("version") != self.ENCRYPTION_VERSION:
            raise ValueError(f"Unsupported version: {envelope.get('version')}")

        decrypted = self.scalsa20_crypto.decrypt_message(
            private_hex=server_private_key,
            public_hex=envelope["ephemeral_public_key"],
            b64_encrypted=envelope["encrypted_data"],
            b64_iv=envelope["iv"]
        )

        return decrypted.plain_text

    def encrypt_json(self, data: dict) -> str:
        """
        Legacy method name for DatabaseEncryptionService compatibility.
        NOT IMPLEMENTED - requires user_public_key for ECIES.

        This method is intentionally not implemented to prevent misuse.
        Use encrypt_for_user_only() instead.
        """
        raise NotImplementedError(
            "ECIES requires user_public_key. Use encrypt_for_user_only(data, user_public_key) instead."
        )


# Singleton instance for the application
_ecies_encryption_service: Optional[ECIESEncryptionService] = None


def get_ecies_encryption_service() -> ECIESEncryptionService:
    """Get the singleton ECIES encryption service instance"""
    global _ecies_encryption_service
    if _ecies_encryption_service is None:
        _ecies_encryption_service = ECIESEncryptionService()
        logger.info("ECIES encryption service initialized")
    return _ecies_encryption_service


def reset_ecies_encryption_service():
    """Reset the singleton (useful for testing)"""
    global _ecies_encryption_service
    _ecies_encryption_service = None

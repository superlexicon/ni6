"""
Hybrid Encryption/Decryption Module

Implements a hybrid encryption scheme:
1. Payload is encrypted with AES-256-GCM (symmetric)
2. Symmetric key is encrypted with ECDH + Salsa20 (asymmetric)

Envelope structure:
{
    "client_public_key": "<hex>",        # Client's ephemeral public key for ECDH
    "encrypted_key": "<base64>",         # Symmetric key encrypted via ECDH+Salsa20
    "key_iv": "<base64>",                # IV for key decryption (Salsa20)
    "encrypted_payload": "<base64>",     # JSON payload encrypted with AES-256-GCM
    "payload_iv": "<base64>"             # IV/nonce for AES-GCM (12 bytes)
}
"""

import base64
import json
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass

from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

from app.core.key.scalsa20_crypto import Scalsa20Crypto
from app.core.key.secp256k1 import KeyPair
from app.core.logger import get_logger


logger = get_logger()


@dataclass
class DecryptedEnvelope:
    """Result of envelope decryption"""
    payload: Dict[str, Any]  # Decrypted and parsed JSON payload
    client_public_key: str   # Client's public key (for reference)


class HybridCryptoError(Exception):
    """Exception raised during hybrid encryption/decryption operations"""
    pass


class HybridCrypto:
    """
    Hybrid encryption using AES-256-GCM + ECDH.

    - AES-256-GCM for payload encryption (authenticated encryption)
    - ECDH + Salsa20 for symmetric key encryption
    """

    # AES-GCM constants
    AES_KEY_SIZE = 32  # 256 bits
    AES_NONCE_SIZE = 12  # 96 bits (standard for GCM)
    AES_TAG_SIZE = 16  # 128 bits

    def __init__(self):
        self.scalsa20_crypto = Scalsa20Crypto()
        self.key_pair = KeyPair()

    def decrypt_envelope(self, envelope: Dict[str, Any]) -> DecryptedEnvelope:
        """
        Decrypt a hybrid-encrypted envelope.

        Args:
            envelope: Dictionary containing:
                - client_public_key: Hex string of client's ephemeral public key
                - encrypted_key: Base64 encoded encrypted symmetric key
                - key_iv: Base64 encoded IV for key decryption
                - encrypted_payload: Base64 encoded encrypted JSON payload
                - payload_iv: Base64 encoded IV for payload decryption

        Returns:
            DecryptedEnvelope with parsed JSON payload

        Raises:
            HybridCryptoError: If decryption fails
        """
        try:
            # Validate required fields
            required_fields = ['client_public_key', 'encrypted_key', 'key_iv',
                             'encrypted_payload', 'payload_iv']
            for field in required_fields:
                if field not in envelope:
                    raise HybridCryptoError(f"Missing required field: {field}")

            client_public_key = envelope['client_public_key']
            encrypted_key_b64 = envelope['encrypted_key']
            key_iv_b64 = envelope['key_iv']
            encrypted_payload_b64 = envelope['encrypted_payload']
            payload_iv_b64 = envelope['payload_iv']

            # Step 1: Decrypt the symmetric key using ECDH + Salsa20
            symmetric_key = self._decrypt_symmetric_key(
                client_public_key=client_public_key,
                encrypted_key_b64=encrypted_key_b64,
                key_iv_b64=key_iv_b64
            )

            logger.info("Successfully decrypted symmetric key")

            # Step 2: Decrypt the payload using AES-256-GCM
            payload_json = self._decrypt_payload(
                symmetric_key=symmetric_key,
                encrypted_payload_b64=encrypted_payload_b64,
                payload_iv_b64=payload_iv_b64
            )

            logger.info("Successfully decrypted payload")

            # Step 3: Parse JSON
            try:
                payload = json.loads(payload_json)
            except json.JSONDecodeError as e:
                raise HybridCryptoError(f"Failed to parse decrypted payload as JSON: {e}")

            return DecryptedEnvelope(
                payload=payload,
                client_public_key=client_public_key
            )

        except HybridCryptoError:
            raise
        except Exception as e:
            logger.error(f"Envelope decryption failed: {e}")
            raise HybridCryptoError(f"Envelope decryption failed: {e}")

    def _decrypt_symmetric_key(
        self,
        client_public_key: str,
        encrypted_key_b64: str,
        key_iv_b64: str
    ) -> bytes:
        """
        Decrypt the symmetric key using ECDH + Salsa20.

        Args:
            client_public_key: Client's ephemeral public key (hex)
            encrypted_key_b64: Base64 encoded encrypted symmetric key
            key_iv_b64: Base64 encoded IV for Salsa20

        Returns:
            Decrypted symmetric key (32 bytes)
        """
        try:
            # Get server's private key
            server_keys = self.key_pair.generate_secp256k1_keys()
            server_private_key = server_keys.private_key

            # Decrypt using ECDH + Salsa20
            decrypted_key = self.scalsa20_crypto.decrypt_bytes(
                private_hex=server_private_key,
                public_hex=client_public_key,
                b64_encrypted=encrypted_key_b64,
                b64_iv=key_iv_b64
            )

            if len(decrypted_key) != self.AES_KEY_SIZE:
                raise HybridCryptoError(
                    f"Invalid symmetric key size: expected {self.AES_KEY_SIZE}, got {len(decrypted_key)}"
                )

            return decrypted_key

        except Exception as e:
            raise HybridCryptoError(f"Failed to decrypt symmetric key: {e}")

    def _decrypt_payload(
        self,
        symmetric_key: bytes,
        encrypted_payload_b64: str,
        payload_iv_b64: str
    ) -> str:
        """
        Decrypt the payload using AES-256-GCM.

        Args:
            symmetric_key: 32-byte symmetric key
            encrypted_payload_b64: Base64 encoded ciphertext + auth tag
            payload_iv_b64: Base64 encoded IV/nonce (12 bytes)

        Returns:
            Decrypted JSON string
        """
        try:
            # Decode from base64
            payload_iv = base64.b64decode(payload_iv_b64)
            encrypted_payload = base64.b64decode(encrypted_payload_b64)

            if len(payload_iv) != self.AES_NONCE_SIZE:
                raise HybridCryptoError(
                    f"Invalid payload IV size: expected {self.AES_NONCE_SIZE}, got {len(payload_iv)}"
                )

            # AES-GCM ciphertext includes the auth tag at the end
            if len(encrypted_payload) < self.AES_TAG_SIZE:
                raise HybridCryptoError("Encrypted payload too short (missing auth tag)")

            # Split ciphertext and tag
            ciphertext = encrypted_payload[:-self.AES_TAG_SIZE]
            auth_tag = encrypted_payload[-self.AES_TAG_SIZE:]

            # Decrypt using AES-256-GCM
            cipher = AES.new(symmetric_key, AES.MODE_GCM, nonce=payload_iv)
            plaintext = cipher.decrypt_and_verify(ciphertext, auth_tag)

            return plaintext.decode('utf-8')

        except ValueError as e:
            # decrypt_and_verify raises ValueError if auth tag is invalid
            raise HybridCryptoError(f"Payload authentication failed (tampered data): {e}")
        except Exception as e:
            raise HybridCryptoError(f"Failed to decrypt payload: {e}")

    def is_encrypted_envelope(self, data: Dict[str, Any]) -> bool:
        """
        Check if the data appears to be an encrypted envelope.

        Args:
            data: Dictionary to check

        Returns:
            True if data has the envelope structure with non-empty values
        """
        required_fields = ['client_public_key', 'encrypted_key', 'key_iv',
                          'encrypted_payload', 'payload_iv']
        return all(data.get(field) for field in required_fields)

    # --- Encryption methods (for testing/reference) ---

    def encrypt_envelope(
        self,
        payload: Dict[str, Any],
        server_public_key: str,
        client_private_key: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Encrypt a payload into a hybrid-encrypted envelope.

        This method is primarily for testing. In production, the client
        performs encryption.

        Args:
            payload: Dictionary to encrypt
            server_public_key: Server's public key (hex)
            client_private_key: Client's private key (hex), or None to generate

        Returns:
            Encrypted envelope dictionary
        """
        try:
            # Generate client keypair if not provided
            if client_private_key is None:
                client_keys = self.key_pair.generate_secp256k1_keys()
                client_private_key = client_keys.private_key
                client_public_key = client_keys.public_key
            else:
                # Derive public key from private key
                from ecdsa import SigningKey, SECP256k1
                sk = SigningKey.from_string(bytes.fromhex(client_private_key), curve=SECP256k1)
                client_public_key = sk.get_verifying_key().to_string().hex()

            # Step 1: Generate random symmetric key
            symmetric_key = get_random_bytes(self.AES_KEY_SIZE)

            # Step 2: Encrypt payload with AES-256-GCM
            payload_json = json.dumps(payload)
            payload_iv = get_random_bytes(self.AES_NONCE_SIZE)

            cipher = AES.new(symmetric_key, AES.MODE_GCM, nonce=payload_iv)
            ciphertext, auth_tag = cipher.encrypt_and_digest(payload_json.encode('utf-8'))

            # Combine ciphertext and auth tag
            encrypted_payload = ciphertext + auth_tag

            # Step 3: Encrypt symmetric key with ECDH + Salsa20
            encrypted_key_result = self.scalsa20_crypto.encrypt_message(
                private_hex=client_private_key,
                public_hex=server_public_key,
                message=symmetric_key
            )

            # Build envelope
            return {
                'client_public_key': client_public_key,
                'encrypted_key': encrypted_key_result.enc,
                'key_iv': encrypted_key_result.iv,
                'encrypted_payload': base64.b64encode(encrypted_payload).decode('utf-8'),
                'payload_iv': base64.b64encode(payload_iv).decode('utf-8')
            }

        except Exception as e:
            raise HybridCryptoError(f"Envelope encryption failed: {e}")

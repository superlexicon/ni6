"""Utility for signing internal server-to-server requests."""

import time
from typing import Dict
from ecdsa import SigningKey, SECP256k1, util
from hashlib import sha256

from app.core.key.secp256k1 import KeyPair
from app.core.logger import get_logger

logger = get_logger()


class InternalSignature:
    """Handles signing for internal server-to-server communication."""

    @staticmethod
    def create_signature_headers(private_key_hex: str, public_key_hex: str) -> Dict[str, str]:
        """
        Create signature headers for internal API requests.

        Args:
            private_key_hex: Server's private key in hex
            public_key_hex: Server's public key in hex (for identification)

        Returns:
            Dict with headers for HTTP request
        """
        timestamp = int(time.time())
        message = f"internal:{timestamp}"

        try:
            # Create signing key from private key hex
            sk = SigningKey.from_string(
                bytes.fromhex(private_key_hex),
                curve=SECP256k1
            )

            # Sign the message
            signature = sk.sign(
                message.encode('utf-8'),
                hashfunc=sha256,
                sigencode=util.sigencode_string
            )

            # Decode to r, s components (each 32 bytes for secp256k1)
            r = signature[:32].hex()
            s = signature[32:].hex()

            return {
                'X-Internal-Public-Key': public_key_hex,
                'X-Internal-Timestamp': str(timestamp),
                'X-Internal-Signature-R': r,
                'X-Internal-Signature-S': s
            }
        except Exception as e:
            logger.error(f"Failed to create internal signature: {e}")
            raise

    @staticmethod
    def get_server_keys() -> tuple[str, str]:
        """Get server's public and private key pair."""
        keys = KeyPair.generate_secp256k1_keys()
        return keys.public_key, keys.private_key

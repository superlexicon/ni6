from cryptography.fernet import Fernet
import os
import base64
import json
import hashlib
from app.core.logger import get_logger
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

logger = get_logger()


class DatabaseEncryptionService:
    """
    Application-level field encryption for database at-rest protection.

    Uses AES-128-CBC via Fernet (symmetric encryption).
    Key is derived from SEED (same seed used for keypair and NaCl encryption).
    """

    def __init__(self):
        """Initialize encryption service with key derived from SEED."""
        # Get SEED and derive DB encryption key from it
        from app.core.key.seed_generator import GenerateSeed

        seed = GenerateSeed.get_seed()

        # Derive a different key for DB encryption using HKDF with context
        # This ensures key separation from NaCl key (even though both derive from SEED)
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'im_osint_db_encryption',  # Context for key separation
            info=b'db_field_encryption',
        )

        derived_key = hkdf.derive(seed.encode('utf-8'))
        self.master_key = base64.urlsafe_b64encode(derived_key).decode()

        self.cipher = Fernet(self.master_key.encode())
        logger.info("DatabaseEncryptionService initialized (key derived from SEED)")

    def hash_passport(self, passport_country: str, passport_number: str) -> str:
        """
        [DEPRECATED] Generate SHA-256 hash of passport_country + passport_number.

        DEPRECATED: passport_hash column removed from user_identity_index table.
        Face biometrics is now used as the primary identity uniqueness constraint
        via the trg_face_biometrics_cross_identity_check trigger.

        This method is kept for backward compatibility only and should not be used.
        """
        combined = f"{passport_country}:{passport_number}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def encrypt_json(self, data: dict) -> str:
        """
        Encrypt JSON data for storage.

        Converts dict to JSON, then encrypts.
        Returns base64-encoded ciphertext.
        """
        if not data:
            return None

        json_str = json.dumps(data)
        ciphertext = self.cipher.encrypt(json_str.encode())
        return base64.urlsafe_b64encode(ciphertext).decode()

    def decrypt_json(self, ciphertext: str) -> dict:
        """Decrypt JSON data from storage."""
        if not ciphertext:
            return None

        try:
            decoded = base64.urlsafe_b64decode(ciphertext.encode())
            plaintext = self.cipher.decrypt(decoded)
            return json.loads(plaintext.decode())
        except Exception as e:
            logger.error(f"Failed to decrypt JSON: {e}")
            return None


# Singleton instance
_encryption_service = None


def get_encryption_service() -> DatabaseEncryptionService:
    """Get singleton encryption service instance."""
    global _encryption_service
    if _encryption_service is None:
        _encryption_service = DatabaseEncryptionService()
    return _encryption_service

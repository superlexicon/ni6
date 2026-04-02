import base64
from dataclasses import dataclass
from typing import Tuple, Union, Optional

from ecdsa import SigningKey, VerifyingKey, SECP256k1
from Crypto.Cipher import Salsa20
from app.dto import DecryptedMessageData, EncryptedMessageData


@dataclass
class KeyValidationError(ValueError):
    message: str


class Scalsa20Crypto:
    # Constants
    SECRET_KEY_LENGTH = 32
    IV_LENGTH = 8
    PADDING_CHAR = '0'

    def _left_pad(self, s: str, width: int) -> str:
        if len(s) >= width:
            return s
        return s.rjust(width, self.PADDING_CHAR)

    def _derive_shared_secret(self, private_key: SigningKey, public_key: VerifyingKey) -> Tuple[bytes, bytes]:
        try:

            if not hasattr(private_key, 'privkey') or not private_key.privkey:
                raise KeyValidationError(
                    "Invalid private key: missing privkey attribute")
            if not hasattr(private_key.privkey, 'secret_multiplier'):
                raise KeyValidationError(
                    "Invalid private key: missing secret_multiplier")
            if not hasattr(public_key, 'pubkey') or not public_key.pubkey:
                raise KeyValidationError(
                    "Invalid public key: missing pubkey attribute")
            if not hasattr(public_key.pubkey, 'point') or not public_key.pubkey.point:
                raise KeyValidationError(
                    "Invalid public key: missing point attribute")

            shared_point = private_key.privkey.secret_multiplier * public_key.pubkey.point
            x_s = format(shared_point.x(), 'x')
            y_s = format(shared_point.y(), 'x')

            hex_x = self._left_pad(x_s, 64)
            hex_y = self._left_pad(y_s, 64)
            secret_bytes = bytes.fromhex(hex_x + hex_y)

            return (bytes(secret_bytes[:self.SECRET_KEY_LENGTH]),
                    bytes(secret_bytes[self.SECRET_KEY_LENGTH:self.SECRET_KEY_LENGTH + self.IV_LENGTH]))

        except Exception as e:
            raise KeyValidationError(f"Failed to derive shared secret: {e}")

    def _get_keys(self, private_hex: str, public_hex: str) -> Tuple[SigningKey, VerifyingKey]:
        try:
            private_key = SigningKey.from_string(
                bytes.fromhex(private_hex), curve=SECP256k1)
            public_hex_clean = public_hex[2:] if public_hex.startswith(
                '04') else public_hex
            public_key = VerifyingKey.from_string(
                bytes.fromhex(public_hex_clean), curve=SECP256k1)

            return private_key, public_key

        except (ValueError, TypeError) as e:
            raise KeyValidationError(f"Invalid key format: {e}")

    def encrypt_message(self, private_hex: str, public_hex: str, message: Union[str, bytes]) -> EncryptedMessageData:
        private_key, public_key = self._get_keys(private_hex, public_hex)
        secret_key, iv = self._derive_shared_secret(private_key, public_key)

        if isinstance(message, str):
            message = message.encode('utf-8')

        cipher = Salsa20.new(key=secret_key, nonce=iv)
        ciphertext = cipher.encrypt(message)

        return EncryptedMessageData(
            enc=base64.b64encode(ciphertext).decode('utf-8'),
            iv=base64.b64encode(iv).decode('utf-8')
        )

    def decrypt_message(self, private_hex: str, public_hex: str,
                        b64_encrypted: str, b64_iv: Optional[str] = None) -> DecryptedMessageData:
        private_key, public_key = self._get_keys(private_hex, public_hex)

        if b64_iv:
            iv = base64.b64decode(b64_iv)
            secret_key, _ = self._derive_shared_secret(private_key, public_key)
        else:
            secret_key, iv = self._derive_shared_secret(
                private_key, public_key)

        cipher = Salsa20.new(key=secret_key, nonce=iv)
        ciphertext = base64.b64decode(b64_encrypted)
        plaintext = cipher.decrypt(ciphertext)

        return DecryptedMessageData(plain_text=plaintext.decode('utf-8'))

    def decrypt_bytes(self, private_hex: str, public_hex: str,
                      b64_encrypted: str, b64_iv: Optional[str] = None) -> bytes:
        """
        Decrypt message and return raw bytes (for binary data like zip files).

        Args:
            private_hex: Private key in hex format
            public_hex: Public key in hex format
            b64_encrypted: Base64 encoded ciphertext
            b64_iv: Optional base64 encoded IV

        Returns:
            Decrypted data as bytes
        """
        private_key, public_key = self._get_keys(private_hex, public_hex)

        if b64_iv:
            iv = base64.b64decode(b64_iv)
            secret_key, _ = self._derive_shared_secret(private_key, public_key)
        else:
            secret_key, iv = self._derive_shared_secret(
                private_key, public_key)

        cipher = Salsa20.new(key=secret_key, nonce=iv)
        ciphertext = base64.b64decode(b64_encrypted)
        plaintext = cipher.decrypt(ciphertext)

        return plaintext

    @staticmethod
    def encrypt_message_static(private_hex: str, public_hex: str, message: Union[str, bytes]) -> EncryptedMessageData:
        return Scalsa20Crypto().encrypt_message(private_hex, public_hex, message)

    @staticmethod
    def decrypt_message_static(private_hex: str, public_hex: str,
                               b64_encrypted: str, b64_iv: Optional[str] = None) -> DecryptedMessageData:
        return Scalsa20Crypto().decrypt_message(private_hex, public_hex, b64_encrypted, b64_iv)

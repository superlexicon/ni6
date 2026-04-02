

from app.core import logger
from app.core.key.secp256k1 import KeyPair
from app.core.key.scalsa20_crypto import Scalsa20Crypto
from app.dto import (
    PublicKeyResponse,
    UserShareKeyResponse,
    UserShareKeyRequest
)
from app.repositories import UserKeyRepository, OTPRepository
from app.utils import DecodeBase64
from app.helper import KeyHelper
from typing import Optional, Dict, Any

class KeyService:
    def __init__(self,
                 key_repository: UserKeyRepository,
                 otp_repository: OTPRepository,
                 decode_base64: DecodeBase64,
                 key_helper: KeyHelper,
                 key_pair: KeyPair,
                 salsa20_crypto: Scalsa20Crypto):
        self.logger = logger
        self.decode_base64 = decode_base64
        self.key_repository = key_repository
        self.otp_repository = otp_repository
        self.key_helper = key_helper
        self.key_pair = key_pair
        self.salsa20_crypto = salsa20_crypto

    async def get_server_public_key(self) -> PublicKeyResponse:
        key = self.key_pair.generate_secp256k1_keys()
        return PublicKeyResponse(public_key=key.public_key)

    async def create_key(self, request: UserShareKeyRequest) -> UserShareKeyResponse:
        # Generate shared encryption key using user's public key
        shared_secret = self.salsa20_crypto.derive_shared_secret(
            self.key_pair.private_key,
            request.user_public_key
        )
        share_encrypt_key = self.salsa20_crypto.encrypt_message(
            "shared_secret",
            shared_secret
        )

        key_data = {
            'mobile_number': request.email,  # Legacy email-based flow
            'country_code': None,
            'user_public_key': request.user_public_key,
            'share_encrypt_key': share_encrypt_key,
            'encrypted_secret_share': None
        }

        result = self.key_repository.create_key(key_data)
        if result is None:
            raise Exception("Failed to create key in the database")

        response_data = {
            "user_public_key": result['user_public_key'],
            "share_encrypt_key": result['share_encrypt_key']
        }
        return UserShareKeyResponse(**response_data)

    async def get_user_key(self, public_key: str) -> Optional[Dict[str, Any]]:
        """Get user key by public key"""
        return self.key_repository.get_key_by_public_key(public_key)

    async def update_user_key(self, public_key: str, encrypted_secret_share: str) -> bool:
        """Update user's encrypted secret share"""
        update_data = {
            'encrypted_secret_share': encrypted_secret_share
        }
        return self.key_repository.update_key_by_public_key(public_key, update_data)

"""
OTP Service - Public Key Only

Unified service for OTP generation, verification, and signed request processing.
"""

import asyncio
import uuid
from typing import Dict, Any, Optional
from datetime import datetime, timedelta, timezone

from app.core.logger import get_logger
from app.core.key.ecdsa_recovery import ECDSARecovery
from app.core.key.hybrid_crypto import HybridCrypto
from app.repositories.user_identity_repository import UserIdentityRepository
from app.repositories.user_key_repository import UserKeyRepository
from app.repositories.otp_repository import OTPRepository
from app.utils.unique_random_generator import UniqueRandomGenerator
from app.config.aws_config import aws_settings


class OTPService:
    """Unified OTP service using public_key only."""

    def __init__(self,
                 unique_random_generator: UniqueRandomGenerator = None,
                 otp_repository: OTPRepository = None,
                 otp_broadcast_service: Optional["OTPBroadcastService"] = None):
        from app.services.otp_broadcast_service import otp_broadcast_service as broadcast_svc

        self.user_identity_repo = UserIdentityRepository()
        self.user_key_repo = UserKeyRepository()
        self.otp_repo = otp_repository if otp_repository else OTPRepository()
        self.hybrid_crypto = HybridCrypto()
        self.logger = get_logger()
        self.unique_random_generator = unique_random_generator if unique_random_generator else UniqueRandomGenerator()

        # OTP broadcast service
        try:
            self.otp_broadcast_service = otp_broadcast_service if otp_broadcast_service else broadcast_svc
            self.logger.info("OTP broadcast service initialized")
        except Exception as e:
            self.logger.warning(f"OTP broadcast service not available: {e}")
            self.otp_broadcast_service = None

    async def process_signed_request(
        self,
        client_public_key: str,
        secret_share: Optional[str] = None,
        secret_share_encrypted: Optional[Dict[str, str]] = None,
        timestamp: int = 0,
        signature_r: str = "",
        signature_s: str = "",
        target_server_public_key: str = "",
        otp_length: int = 6,
        generate_otp: bool = True,
        gesture_mode: bool = False,
        device_id: Optional[str] = None,
        api_url: Optional[str] = None,
        mobile_number: Optional[str] = None,
        country_code: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process signed OTP request (public_key only).

        Returns:
            {
                'success': bool,
                'user_identity_id': str (if successful),
                'encrypted_response': dict (if successful),
                'error': str (if failed)
            }
        """
        try:
            # Step 1: Verify signature
            message = f"otp:{timestamp}"
            if not ECDSARecovery.verify_signature(
                message=message,
                r=signature_r,
                s=signature_s,
                public_key=client_public_key
            ):
                self.logger.warning(f"Signature verification failed for {client_public_key[:16]}...")
                return {'success': False, 'error': 'Invalid signature'}

            self.logger.info(f"✅ Signature verified for {client_public_key[:16]}...")

            # Step 2: Decrypt and encrypt secret_share
            actual_secret_share = None
            if secret_share_encrypted:
                from app.core.key.secp256k1 import KeyPair
                from app.services.ecies_encryption_service import get_ecies_encryption_service
                server_keys = KeyPair.generate_secp256k1_keys()
                ecies_service = get_ecies_encryption_service()
                actual_secret_share = ecies_service.decrypt_message_from_client(
                    envelope=secret_share_encrypted,
                    server_private_key=server_keys.private_key
                )
            elif secret_share:
                actual_secret_share = secret_share

            # Encrypt with NaCl for storage
            encrypted_for_storage = None
            if actual_secret_share:
                from app.core.key_injection.key_injection_manager import key_injection_manager
                encrypted_for_storage = key_injection_manager.encrypt_data(actual_secret_share)

            # Step 3: Check existing user
            existing_user = self.user_key_repo.get_key_by_public_key(client_public_key)
            user_identity_id = existing_user.get('user_identity_id') if existing_user else None
            if existing_user:
                self.logger.info(f"User already verified: {user_identity_id[:16]}...")

            # Step 4: Store in user_keys_pending
            from app.repositories.user_keys_pending_repository import UserKeysPendingRepository
            pending_key_repo = UserKeysPendingRepository()
            pending_key_repo.create_or_update_pending_key({
                'mobile_number': mobile_number,
                'country_code': country_code,
                'user_public_key': client_public_key,
                'encrypted_secret_share': encrypted_for_storage,
                'device_id': device_id,
                'api_url': api_url
            })
            self.logger.info(f"✅ Stored in user_keys_pending for {client_public_key[:16]}...")

            # Step 5: Generate OTP (if requested)
            otp_data = {'otp_id': None, 'expires_at': None, 'sent_at': None, 'otp': None}

            if generate_otp:
                otp_result = self._generate_otp(
                    length=otp_length,
                    client_public_key=client_public_key,
                    gesture_mode=gesture_mode
                )

                # Prepare OTP data for database and broadcast
                otp_broadcast_data = {
                    'public_key': client_public_key,
                    'random_number': otp_result['otp'],
                    'otp_id': otp_result['otp_id'],
                    'expires_at': otp_result['expires_at'],
                    'delivery_method': 'encrypted_response',
                    'attempts': 0,
                    'max_attempts': 3,
                    'is_verified': False
                }
                if encrypted_for_storage is not None:
                    otp_broadcast_data['encrypted_secret_share'] = encrypted_for_storage
                if device_id is not None:
                    otp_broadcast_data['device_id'] = device_id
                if api_url is not None:
                    otp_broadcast_data['api_url'] = api_url

                # Check for existing OTP
                existing_otp = self.otp_repo.get_unverified_otp_by_public_key(client_public_key)
                if not existing_otp:
                    self.otp_repo.create_otp(otp_broadcast_data)
                    self.logger.info(f"Created OTP for {client_public_key[:16]}...")
                else:
                    self.otp_repo.update_otp_by_public_key(client_public_key, otp_broadcast_data)
                    self.logger.info(f"Updated OTP for {client_public_key[:16]}...")

                # Broadcast to peer instances
                if self.otp_broadcast_service:
                    self.logger.info(f"📢 Broadcasting OTP to peer instances")
                    asyncio.create_task(self.otp_broadcast_service.broadcast_otp_created(otp_broadcast_data))
                else:
                    self.logger.warning("⚠️ OTP broadcast service not available")

                otp_data = {
                    'otp_id': otp_result['otp_id'],
                    'expires_at': otp_result['expires_at'],
                    'sent_at': otp_result['sent_at'],
                    'otp': otp_result['otp']
                }
                self.logger.info(f"✅ OTP generated for {client_public_key[:16]}...")
            else:
                self.logger.info(f"⏭️ Skipping OTP generation (generate_otp=False)")

            # Step 6: Prepare and encrypt response
            response_payload = {
                'otp': otp_data.get('otp'),
                'otp_id': otp_data.get('otp_id'),
                'expires_at': otp_data.get('expires_at').isoformat() if otp_data.get('expires_at') else None,
                'sent_at': otp_data.get('sent_at').isoformat() if otp_data.get('sent_at') else None,
                'user_identity_id': user_identity_id
            }

            encrypted_response = self._encrypt_response(
                payload=response_payload,
                client_public_key=client_public_key
            )

            self.logger.info(f"✅ Encrypted OTP response for {client_public_key[:16]}...")

            return {
                'success': True,
                'user_identity_id': user_identity_id,
                'encrypted_response': encrypted_response
            }

        except Exception as e:
            self.logger.error(f"Error processing signed OTP request: {str(e)}")
            return {'success': False, 'error': f'Failed to process OTP request: {str(e)}'}

    async def process_recovery_request(
        self,
        client_public_key: str,
        timestamp: int = 0,
        signature_r: str = "",
        signature_s: str = "",
        target_server_public_key: str = "",
        otp_length: int = 6,
        generate_otp: bool = True,
        gesture_mode: bool = False,
        mobile_number: Optional[str] = None,
        country_code: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process signed OTP request for recovery flow (no persistence).

        Returns:
            {
                'success': bool,
                'encrypted_response': dict (if successful),
                'error': str (if failed)
            }
        """
        try:
            # Verify signature
            message = f"otp:{timestamp}"
            if not ECDSARecovery.verify_signature(
                message=message,
                r=signature_r,
                s=signature_s,
                public_key=client_public_key
            ):
                return {'success': False, 'error': 'Invalid signature'}

            self.logger.info(f"✅ Recovery signature verified for {client_public_key[:16]}...")

            # Generate OTP (if requested)
            otp_data = {'otp_id': None, 'expires_at': None, 'sent_at': None, 'otp': None}

            if generate_otp:
                otp_result = self._generate_otp(
                    length=otp_length,
                    client_public_key=client_public_key,
                    gesture_mode=gesture_mode
                )
                otp_data = {
                    'otp_id': otp_result['otp_id'],
                    'expires_at': otp_result['expires_at'],
                    'sent_at': otp_result['sent_at'],
                    'otp': otp_result['otp']
                }
                self.logger.info(f"✅ Recovery OTP generated for {client_public_key[:16]}...")

            # Prepare and encrypt response (no user_identity_id for recovery)
            response_payload = {
                'otp': otp_data.get('otp'),
                'otp_id': otp_data.get('otp_id'),
                'expires_at': otp_data.get('expires_at').isoformat() if otp_data.get('expires_at') else None,
                'sent_at': otp_data.get('sent_at').isoformat() if otp_data.get('sent_at') else None,
            }

            encrypted_response = self._encrypt_response(
                payload=response_payload,
                client_public_key=client_public_key
            )

            return {'success': True, 'encrypted_response': encrypted_response}

        except Exception as e:
            self.logger.error(f"Error processing recovery OTP request: {str(e)}")
            return {'success': False, 'error': f'Failed to process recovery OTP request: {str(e)}'}

    def verify_otp_from_selfie(
        self,
        mobile_number: Optional[str],
        otp_code: str,
        client_public_key: str
    ) -> dict:
        """
        Verify OTP code from video selfie submission.

        For gesture OTP, we look up by public_key since the OTP was generated
        for that specific public key. The mobile_number parameter is kept
        for backward compatibility but is not used for lookup.

        Args:
            mobile_number: User's mobile number (not used for gesture OTP lookup)
            otp_code: The 6-digit gesture OTP extracted from video
            client_public_key: Client's public key used for OTP lookup

        Returns:
            {
                'valid': bool,
                'message': str,
                'otp_status': str,
            }
        """
        # For gesture OTP, we verify using public_key (the OTP was generated for this key)
        return self.verify_otp(public_key=client_public_key, otp_code=otp_code)

    def verify_otp(
        self,
        public_key: str,
        otp_code: str
    ) -> dict:
        """
        Verify OTP code for public_key.
        """
        try:
            # Get OTP by public_key
            stored_otp = self.otp_repo.get_unverified_otp_by_public_key(public_key)

            if not stored_otp:
                return {
                    'valid': False,
                    'message': 'OTP not found, expired, or already used',
                    'otp_status': 'not_found'
                }

            # Check attempts limit
            attempts = stored_otp.get('attempts', 0)
            max_attempts = stored_otp.get('max_attempts', 3)
            if attempts >= max_attempts:
                return {
                    'valid': False,
                    'message': 'Maximum verification attempts exceeded',
                    'otp_status': 'max_attempts_exceeded'
                }

            # Increment attempts
            self.otp_repo.increment_otp_attempts(public_key, identifier_type='public_key')

            # Verify OTP code
            if stored_otp.get('random_number') == otp_code:
                # Mark as verified
                self.otp_repo.mark_otp_verified_by_public_key(public_key)

                # Broadcast verification to peer instances
                if self.otp_broadcast_service:
                    try:
                        asyncio.create_task(self.otp_broadcast_service.broadcast_otp_verified(public_key, "public_key"))
                    except Exception as sync_error:
                        self.logger.error(f"Failed to broadcast OTP verification: {sync_error}")

                self.logger.info(f"✅ OTP verified for {public_key[:16]}...")

                return {
                    'valid': True,
                    'message': 'OTP verified successfully',
                    'otp_status': 'verified',
                    'public_key': public_key,
                    'otp_id': stored_otp.get('otp_id'),
                    'expires_at': stored_otp.get('expires_at')
                }
            else:
                return {
                    'valid': False,
                    'message': f'Invalid OTP code',
                    'otp_status': 'invalid_code'
                }

        except Exception as e:
            self.logger.error(f"OTP verification failed for {public_key[:16]}...: {type(e).__name__}")
            return {
                'valid': False,
                'message': 'Verification failed due to server error',
                'otp_status': 'error'
            }

    def _generate_gesture_otp(self, client_public_key: str) -> Dict[str, Any]:
        """
        Generate a 6-digit gesture OTP for video selfie verification.

        Format: [gesture1][seconds1][gesture2][seconds2][gesture3][seconds3]
        All digits are 1-5.

        Args:
            client_public_key: The client's public key for OTP association

        Returns:
            {
                'otp': '131241',
                'otp_id': str,
                'expires_at': datetime,
                'sent_at': datetime,
                'public_key': str,
            }
        """
        # Gesture OTP is always 6 digits, all in range 1-5
        GESTURE_OTP_LENGTH = 6
        GESTURE_ALLOWED_DIGITS = '12345'

        otp_code = self.unique_random_generator.generate_random_number(
            GESTURE_OTP_LENGTH,
            allowed_digits=GESTURE_ALLOWED_DIGITS
        )

        # Generate unique OTP request ID
        otp_id = str(uuid.uuid4())

        # Calculate expiry time (use UTC)
        expiry_minutes = aws_settings.otp_expiry_minutes
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=expiry_minutes)

        self.logger.info(f"Generated gesture OTP: {otp_code} for {client_public_key[:16]}...")

        return {
            'otp': otp_code,
            'otp_id': otp_id,
            'expires_at': expires_at,
            'sent_at': datetime.now(timezone.utc),
            'public_key': client_public_key
        }

    def _generate_otp(
        self,
        length: int,
        client_public_key: str,
        gesture_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Generate OTP code for public_key.

        Returns dict with OTP details.
        """
        if gesture_mode:
            return self._generate_gesture_otp(client_public_key)

        # Generate new OTP code for regular OTP
        otp_code = self.unique_random_generator.generate_random_number(length)

        # Generate unique OTP request ID
        otp_id = str(uuid.uuid4())

        # Calculate expiry time (use UTC)
        expiry_minutes = aws_settings.otp_expiry_minutes
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=expiry_minutes)

        return {
            'otp': otp_code,
            'otp_id': otp_id,
            'expires_at': expires_at,
            'sent_at': datetime.now(timezone.utc),
            'public_key': client_public_key
        }

    def _encrypt_response(
        self,
        payload: Dict[str, Any],
        client_public_key: str
    ) -> Dict[str, str]:
        """Encrypt OTP response using hybrid encryption."""
        encrypted_envelope = self.hybrid_crypto.encrypt_envelope(
            payload=payload,
            server_public_key=client_public_key
        )
        return {
            'client_public_key': encrypted_envelope['client_public_key'],
            'encrypted_key': encrypted_envelope['encrypted_key'],
            'key_iv': encrypted_envelope['key_iv'],
            'encrypted_payload': encrypted_envelope['encrypted_payload'],
            'payload_iv': encrypted_envelope['payload_iv']
        }

    # Legacy methods for backward compatibility
    def generate_otp(
        self,
        length: int,
        client_public_key: str,
        gesture_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Generate OTP code for public_key (legacy method).
        """
        return self._generate_otp(length, client_public_key, gesture_mode)

    def generate_otp_without_sms(self, *args, **kwargs) -> Dict[str, Any]:
        """Legacy method - use generate_otp instead."""
        return self.generate_otp(*args, **kwargs)

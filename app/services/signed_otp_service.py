"""
Signed OTP Service - Handles signed OTP requests.

Flow (Multi-Device Support):
1. Verify signature
2. Store in otp table only (country_code, encrypted_secret_share, public_key)
3. Generate OTP
4. Send OTP via SMS
5. Encrypt response

NOTE: user_identity and user_keys are NOT created here anymore.
They are created AFTER selfie verification passes (in sequential_selfie_service.py).
This enables multi-device support where multiple public_keys share the same user_identity.
"""

from typing import Dict, Any, Optional, Tuple
from app.core.logger import get_logger
from app.core.key.ecdsa_recovery import ECDSARecovery
from app.core.key.hybrid_crypto import HybridCrypto
from app.repositories.user_identity_repository import UserIdentityRepository
from app.repositories.user_key_repository import UserKeyRepository
from app.services.otp_service import OTPService
from app.utils.country_code_converter import get_phone_prefix


class SignedOTPService:
    """
    Service for processing signed OTP requests with user identity creation.
    """

    def __init__(self):
        from app.services.otp_broadcast_service import OTPBroadcastService

        self.user_identity_repo = UserIdentityRepository()
        self.user_key_repo = UserKeyRepository()
        self.hybrid_crypto = HybridCrypto()
        self.logger = get_logger()

        # Initialize OTP broadcast service for HTTP-based inter-instance communication
        try:
            from app.services.otp_broadcast_service import otp_broadcast_service as broadcast_svc
            self.otp_broadcast_service = broadcast_svc
            self.logger.info("OTP broadcast service initialized in SignedOTPService")
        except Exception as e:
            self.logger.warning(f"OTP broadcast service not available: {e}")
            self.otp_broadcast_service = None

        # Initialize OTP service WITH broadcast service
        from app.utils.unique_random_generator import UniqueRandomGenerator
        from app.repositories.otp_repository import OTPRepository

        self.otp_service = OTPService(
            unique_random_generator=None,  # Will be initialized by OTPService
            otp_repository=None,  # Will be initialized by OTPService
            otp_sync_service=None,  # RethinkDB OTP sync removed
            otp_broadcast_service=self.otp_broadcast_service
        )

        # Initialize OTP service dependencies
        self.otp_service.unique_random_generator = UniqueRandomGenerator()
        self.otp_service.otp_repository = OTPRepository()

    async def process_signed_request(
        self,
        client_public_key: str,
        mobile_number: str,
        country_code: str,
        secret_share: Optional[str],
        secret_share_encrypted: Optional[Dict[str, str]],
        timestamp: int,
        signature_r: str,
        signature_s: str,
        target_server_public_key: str,
        otp_length: int,
        generate_otp: bool = True,
        gesture_mode: bool = False,
        device_id: Optional[str] = None,
        api_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process signed OTP request and return encrypted response.

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

            is_valid = ECDSARecovery.verify_signature(
                message=message,
                r=signature_r,
                s=signature_s,
                public_key=client_public_key
            )

            if not is_valid:
                self.logger.warning(
                    f"Signature verification failed for public key: {client_public_key[:16]}..."
                )
                return {
                    'success': False,
                    'error': 'Invalid signature'
                }

            self.logger.info(f"✅ Signature verified for {client_public_key[:16]}...")

            # Step 1.5: Extract secret_share (with ECIES decryption support)
            actual_secret_share = None

            if secret_share_encrypted:
                # NEW: Decrypt ECIES-encrypted share
                from app.core.key.secp256k1 import KeyPair
                from app.services.ecies_encryption_service import get_ecies_encryption_service

                server_keys = KeyPair.generate_secp256k1_keys()
                ecies_service = get_ecies_encryption_service()
                actual_secret_share = ecies_service.decrypt_message_from_client(
                    envelope=secret_share_encrypted,
                    server_private_key=server_keys.private_key
                )
                self.logger.info("Decrypted ECIES-encrypted secret share")
            elif secret_share:
                # LEGACY: Plaintext share
                actual_secret_share = secret_share

            # Encrypt with NaCl for storage
            if actual_secret_share:
                from app.core.key_injection.key_injection_manager import key_injection_manager
                encrypted_for_storage = key_injection_manager.encrypt_data(actual_secret_share)
            else:
                encrypted_for_storage = None

            # Step 2: Check if user already has a verified user_keys record
            existing_user = self.user_key_repo.get_key_by_public_key(client_public_key)

            user_identity_id = None
            if existing_user:
                # User already verified - return existing identity
                user_identity_id = existing_user.get('user_identity_id')
                self.logger.info(
                    f"User already verified: {user_identity_id[:16]}..., "
                    f"will update OTP record"
                )

            # Step 3: Convert country code to phone prefix for consistent mobile number format
            # All OTP operations use the full mobile number (with prefix) to ensure consistency
            phone_prefix = get_phone_prefix(country_code) if country_code else None
            if not phone_prefix:
                return {
                    'success': False,
                    'error': f'Invalid country code: {country_code}'
                }

            full_mobile_number = f"{phone_prefix}{mobile_number}"

            # Step 4: Store in user_keys_pending IMMEDIATELY (before OTP operations)
            # This ensures encrypted_secret_share is stored regardless of OTP generation timing
            from app.repositories.user_keys_pending_repository import UserKeysPendingRepository
            pending_key_repo = UserKeysPendingRepository()

            pending_key_data = {
                'mobile_number': full_mobile_number,
                'country_code': country_code,
                'user_public_key': client_public_key,
                'encrypted_secret_share': encrypted_for_storage,
                'device_id': device_id,
                'api_url': api_url
            }

            pending_key_repo.create_or_update_pending_key(pending_key_data)
            self.logger.info(f"✅ Stored in user_keys_pending for {client_public_key[:16]}...")

            # Step 5: Handle OTP table operations
            # Only insert into otp table if generate_otp=True or if OTP record doesn't exist yet
            # Note: encrypted_secret_share is already stored in user_keys_pending, so this is just for OTP operations

            # Check if OTP already exists for this mobile_number
            from app.repositories.otp_repository import OTPRepository
            otp_repo = OTPRepository()
            existing_otp = otp_repo.get_otp_by_mobile_number(full_mobile_number)

            # Only include encrypted_secret_share if we have a value (additional safeguard)
            otp_update_data = {
                'public_key': client_public_key,
                'country_code': country_code,
                'is_verified': False
            }
            if encrypted_for_storage is not None:
                otp_update_data['encrypted_secret_share'] = encrypted_for_storage
            if device_id is not None:
                otp_update_data['device_id'] = device_id
            if api_url is not None:
                otp_update_data['api_url'] = api_url

            if existing_otp:
                # Update existing OTP record with metadata
                self.logger.info(f"Updating existing OTP record for mobile_number: {full_mobile_number}")
                otp_repo.update_otp(full_mobile_number, otp_update_data)
            elif not generate_otp:
                # No existing OTP and not generating OTP - create record with encrypted_secret_share now
                # This ensures each node stores its own secret share before broadcast arrives
                otp_create_data = {
                    'mobile_number': full_mobile_number,
                    'public_key': client_public_key,
                    'country_code': country_code,
                    'encrypted_secret_share': encrypted_for_storage,
                    'device_id': device_id,
                    'api_url': api_url,
                    'is_verified': False,
                    'delivery_method': 'sms',  # Will be sent by another node
                    'attempts': 0,
                    'max_attempts': 3
                }
                otp_repo.create_otp(otp_create_data)
                self.logger.info(f"Created OTP record with encrypted_secret_share for mobile_number: {full_mobile_number} (generate_otp=false)")

            # Step 6: Generate OTP and send via SMS (only if generate_otp is True)
            # Note: generate_and_send_otp_via_sms will create the OTP record if it doesn't exist
            otp_data = {}

            if generate_otp:
                otp_result = await self.otp_service.generate_and_send_otp_via_sms(
                    length=otp_length,
                    mobile_number=full_mobile_number,
                    client_public_key=client_public_key,
                    country_code=country_code,
                    gesture_mode=gesture_mode
                )

                # Update the newly created OTP record with metadata (if no existing OTP was found before)
                if not existing_otp:
                    self.logger.info(f"Updating newly created OTP record with metadata for mobile_number: {full_mobile_number}")
                    otp_repo.update_otp(full_mobile_number, otp_update_data)

                # The OTP response doesn't have a 'success' field, check if OTP was sent
                if otp_result and otp_result.otp_id:
                    otp_data = {
                        'otp_id': otp_result.otp_id,
                        'expires_at': otp_result.expires_at,
                        'sent_at': otp_result.sent_at,
                        'otp': None  # Initialize with None
                    }
                    # For testing/debugging, include the OTP if present
                    if hasattr(otp_result, 'random_number') and otp_result.random_number:
                        otp_data['otp'] = otp_result.random_number
                    else:
                        from app.repositories.otp_repository import OTPRepository
                        otp_repo = OTPRepository()
                        otp_record = otp_repo.get_otp_by_mobile_number(full_mobile_number)
                        if otp_record and otp_record.get('random_number'):
                            otp_data['otp'] = otp_record['random_number']
                else:
                    return {
                        'success': False,
                        'error': 'Failed to generate OTP'
                    }

                self.logger.info(f"✅ OTP generated and sent to {mobile_number}")
            else:
                self.logger.info(f"⏭️ Skipping OTP generation for {mobile_number} (generate_otp=False)")
                # Return empty otp_data - client will wait for peer sync
                otp_data = {
                    'otp_id': None,
                    'expires_at': None,
                    'sent_at': None,
                    'otp': None
                }

            self.logger.info(f"✅ Stored in otp table (user_keys will be created after selfie verification)")

            # Step 7: Prepare response payload
            response_payload = {
                'otp': otp_data.get('otp'),
                'otp_id': otp_data.get('otp_id'),
                'expires_at': otp_data.get('expires_at').isoformat() if otp_data.get('expires_at') else None,
                'sent_at': otp_data.get('sent_at').isoformat() if otp_data.get('sent_at') else None,
                'user_identity_id': user_identity_id
            }

            # Step 8: Encrypt response with hybrid encryption
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
            return {
                'success': False,
                'error': f'Failed to process OTP request: {str(e)}'
            }

    async def process_recovery_request(
        self,
        client_public_key: str,
        mobile_number: str,
        country_code: str,
        timestamp: int,
        signature_r: str,
        signature_s: str,
        target_server_public_key: str,
        otp_length: int = 6,
        generate_otp: bool = True,
        gesture_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Process signed OTP request for recovery flow WITHOUT inserting into user_keys.

        This is for temporary key OTP verification where we don't want to persist
        the temporary key to the database.

        Flow:
        1. Verify signature
        2. Generate OTP
        3. Send OTP via SMS
        4. Encrypt response

        NOTE: Does NOT create user_identity or user_keys records.

        Returns:
            {
                'success': bool,
                'encrypted_response': dict (if successful),
                'error': str (if failed)
            }
        """
        try:
            # Step 1: Verify signature
            message = f"otp:{timestamp}"
            is_valid = ECDSARecovery.verify_signature(
                message=message,
                r=signature_r,
                s=signature_s,
                public_key=client_public_key
            )

            if not is_valid:
                self.logger.warning(
                    f"Signature verification failed for recovery request: {client_public_key[:16]}..."
                )
                return {
                    'success': False,
                    'error': 'Invalid signature'
                }

            self.logger.info(f"✅ Recovery signature verified for {client_public_key[:16]}...")

            # Step 2: Generate OTP and send via SMS (only if generate_otp is True)
            otp_data = {}

            if generate_otp:
                # Convert ISO country code to phone prefix for SMS sending
                phone_prefix = get_phone_prefix(country_code) if country_code else None
                if not phone_prefix:
                    return {
                        'success': False,
                        'error': f'Invalid country code: {country_code}'
                    }

                full_mobile_number = f"{phone_prefix}{mobile_number}"
                otp_result = await self.otp_service.generate_and_send_otp_via_sms(
                    length=otp_length,
                    mobile_number=full_mobile_number,
                    client_public_key=client_public_key,
                    country_code=country_code,
                    gesture_mode=gesture_mode
                )

                if otp_result and otp_result.otp_id:
                    otp_data = {
                        'otp_id': otp_result.otp_id,
                        'expires_at': otp_result.expires_at,
                        'sent_at': otp_result.sent_at,
                    }
                    # For testing/debugging, include the OTP if present
                    if hasattr(otp_result, 'random_number') and otp_result.random_number:
                        otp_data['otp'] = otp_result.random_number
                    else:
                        from app.repositories.otp_repository import OTPRepository
                        otp_repo = OTPRepository()
                        otp_record = otp_repo.get_otp_by_mobile_number(full_mobile_number)
                        if otp_record and otp_record.get('random_number'):
                            otp_data['otp'] = otp_record['random_number']
                else:
                    return {
                        'success': False,
                        'error': 'Failed to generate OTP'
                    }

                self.logger.info(f"✅ Recovery OTP generated and sent to {mobile_number}")
            else:
                self.logger.info(f"⏭️ Skipping OTP generation for {mobile_number} (generate_otp=False)")
                otp_data = {
                    'otp_id': None,
                    'expires_at': None,
                    'sent_at': None,
                    'otp': None
                }

            # Step 3: Prepare response payload (no user_identity_id for recovery)
            response_payload = {
                'otp': otp_data.get('otp'),
                'otp_id': otp_data.get('otp_id'),
                'expires_at': otp_data.get('expires_at').isoformat() if otp_data.get('expires_at') else None,
                'sent_at': otp_data.get('sent_at').isoformat() if otp_data.get('sent_at') else None,
            }

            # Step 4: Encrypt response with hybrid encryption
            encrypted_response = self._encrypt_response(
                payload=response_payload,
                client_public_key=client_public_key
            )

            self.logger.info(f"✅ Encrypted recovery OTP response for {client_public_key[:16]}...")

            return {
                'success': True,
                'encrypted_response': encrypted_response
            }

        except Exception as e:
            self.logger.error(f"Error processing recovery OTP request: {str(e)}")
            return {
                'success': False,
                'error': f'Failed to process recovery OTP request: {str(e)}'
            }

    def _encrypt_response(
        self,
        payload: Dict[str, Any],
        client_public_key: str
    ) -> Dict[str, str]:
        """
        Encrypt OTP response using hybrid encryption.

        Uses client's public key for ECDH key exchange.
        """
        import json

        # Serialize payload to JSON
        payload_json = json.dumps(payload)

        # Encrypt with hybrid crypto (uses client's public_key)
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

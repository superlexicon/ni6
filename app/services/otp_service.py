
from typing import Optional, TYPE_CHECKING, Dict, Any
from datetime import datetime, timedelta, timezone
import uuid
import asyncio

from app.core import logger
from app.dto import OTPResponse
from app.utils.unique_random_generator import UniqueRandomGenerator
from app.repositories import OTPRepository
from app.services.aws_sms_service import AWSSMSService
from app.config.aws_config import aws_settings

if TYPE_CHECKING:
    from app.services.otp_sync_service import OTPSyncService
    from app.services.otp_broadcast_service import OTPBroadcastService


class OTPService:
    def __init__(self,
                 unique_random_generator: UniqueRandomGenerator,
                 otp_repository: OTPRepository,
                 aws_sms_service: Optional[AWSSMSService] = None,
                 otp_sync_service: Optional["OTPSyncService"] = None,
                 otp_broadcast_service: Optional["OTPBroadcastService"] = None):
        self.logger = logger
        self.unique_random_generator = unique_random_generator
        self.otp_repository = otp_repository
        self.aws_sms_service = aws_sms_service
        self.otp_sync_service = otp_sync_service
        self.otp_broadcast_service = otp_broadcast_service

        # Initialize AWS SMS service if not provided
        if not self.aws_sms_service:
            try:
                self.aws_sms_service = AWSSMSService(
                    aws_access_key_id=aws_settings.aws_access_key_id,
                    aws_secret_access_key=aws_settings.aws_secret_access_key,
                    aws_region=aws_settings.aws_region,
                    sns_topic_arn=aws_settings.sns_topic_arn
                )
                self.logger.info("AWS SMS service initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize AWS SMS service: {str(e)}")
                self.aws_sms_service = None

    async def generate_and_send_otp_via_sms(
        self,
        length: int,
        mobile_number: str,
        client_public_key: Optional[str] = None,
        country_code: Optional[str] = None,
        expiry_minutes: Optional[int] = None,
        gesture_mode: bool = False
    ) -> OTPResponse:
        """
        Generate OTP code and send it via AWS SMS.

        Args:
            length: Length of OTP code to generate
            mobile_number: Mobile number to send OTP to (with country code)
            client_public_key: Client's public key for security binding
            country_code: Country code for the mobile number (e.g., '+1', '+44')
            expiry_minutes: Optional custom expiry time
            gesture_mode: If True, restrict OTP to digits 1-5 only (for hand gesture verification, 0 mis-detected as 1)

        Returns:
            OTPResponse with delivery confirmation (no actual OTP returned for security)
        """
        try:
            # Check if there's already a valid (unexpired and unverified) OTP for this mobile number
            existing_valid_otp = self.otp_repository.get_valid_otp_by_mobile_number(mobile_number)

            # Determine if we should reuse existing OTP or generate a new one
            should_generate_new = True
            existing_valid_otp = self.otp_repository.get_valid_otp_by_mobile_number(mobile_number)

            if existing_valid_otp:
                otp_code = existing_valid_otp['random_number']

                # Check if existing OTP is compatible with gesture_mode
                if gesture_mode:
                    # For gesture mode, OTP must only contain digits 1-5
                    if any(d not in '12345' for d in otp_code):
                        self.logger.info(f"Existing OTP has digits 6-9 but gesture_mode=True, regenerating...")
                        # Delete the old OTP and generate a new one
                        self.otp_repository.delete_otp(mobile_number)
                        should_generate_new = True
                    else:
                        # Reuse existing valid OTP - calculate ACTUAL remaining time
                        otp_id = existing_valid_otp['otp_id']
                        expires_at = existing_valid_otp['expires_at']
                        # Normalize timezone: treat naive datetimes as UTC
                        if expires_at.tzinfo is None:
                            expires_at = expires_at.replace(tzinfo=timezone.utc)
                        remaining_delta = expires_at - datetime.now(timezone.utc)
                        remaining_seconds = int(remaining_delta.total_seconds())
                        remaining_minutes = remaining_seconds // 60
                        remaining_secs = remaining_seconds % 60

                        # Format as "X minutes" or "X minutes Y seconds"
                        if remaining_minutes > 0 and remaining_secs > 0:
                            expiry_text = f"{remaining_minutes} minutes {remaining_secs} seconds"
                        elif remaining_minutes > 0:
                            expiry_text = f"{remaining_minutes} minutes"
                        else:
                            expiry_text = f"{remaining_secs} seconds"

                        should_generate_new = False
                        self.logger.debug(f"Reusing existing valid OTP for mobile number: {mobile_number}, expires at: {expires_at}")
                else:
                    # Not gesture mode, reuse any existing OTP - calculate ACTUAL remaining time
                    otp_id = existing_valid_otp['otp_id']
                    expires_at = existing_valid_otp['expires_at']
                    # Normalize timezone: treat naive datetimes as UTC
                    if expires_at.tzinfo is None:
                        expires_at = expires_at.replace(tzinfo=timezone.utc)
                    remaining_delta = expires_at - datetime.now(timezone.utc)
                    remaining_seconds = int(remaining_delta.total_seconds())
                    remaining_minutes = remaining_seconds // 60
                    remaining_secs = remaining_seconds % 60

                    # Format as "X minutes" or "X minutes Y seconds"
                    if remaining_minutes > 0 and remaining_secs > 0:
                        expiry_text = f"{remaining_minutes} minutes {remaining_secs} seconds"
                    elif remaining_minutes > 0:
                        expiry_text = f"{remaining_minutes} minutes"
                    else:
                        expiry_text = f"{remaining_secs} seconds"

                    should_generate_new = False
                    self.logger.debug(f"Reusing existing valid OTP for mobile number: {mobile_number}, expires at: {expires_at}")

            if should_generate_new:
                # No valid OTP found or existing OTP incompatible - clean up any expired/verified OTP
                existing_stale = self.otp_repository.get_otp_by_mobile_number(mobile_number)
                if existing_stale:
                    self.otp_repository.delete_otp(mobile_number)
                    self.logger.debug(f"Deleted stale OTP for mobile_number: {mobile_number}")

                # Generate new OTP code using secure random generator
                # For gesture mode, restrict to digits 1-5 (single-hand finger representation, 0 mis-detected as 1)
                allowed_digits = '12345' if gesture_mode else None
                otp_code = self.unique_random_generator.generate_random_number(length, allowed_digits=allowed_digits)

                # Generate unique OTP request ID
                otp_id = str(uuid.uuid4())

                # Calculate expiry time (use UTC)
                expiry_minutes = expiry_minutes or aws_settings.otp_expiry_minutes
                expires_at = datetime.now(timezone.utc) + timedelta(minutes=expiry_minutes)

                # Format expiry time for new OTP (e.g., "30 minutes")
                expiry_text = f"{expiry_minutes} minutes"

                # Store OTP in database with new schema fields
                otp_data = {
                    'mobile_number': mobile_number,
                    'country_code': country_code,
                    'random_number': otp_code,
                    'otp_id': otp_id,
                    'expires_at': expires_at,
                    'delivery_method': 'sms',
                    'attempts': 0,
                    'max_attempts': 3,
                    'is_verified': False
                }

                # Add client public key if provided for linking to selfie verification
                if client_public_key:
                    otp_data['public_key'] = client_public_key

                # Check if mobile number already has an OTP (expired or verified)
                existing_otp = self.otp_repository.get_otp_by_mobile_number(mobile_number)
                if existing_otp:
                    self.otp_repository.update_otp(mobile_number, otp_data)
                    self.logger.debug(f"Updated existing OTP for mobile number: {mobile_number}")
                else:
                    self.otp_repository.create_otp(otp_data)
                    self.logger.debug(f"Created new OTP for mobile number: {mobile_number}")

                # Sync OTP to peer instances via HTTP broadcast
                if self.otp_broadcast_service:
                    try:
                        # Create background task for broadcast (non-blocking)
                        asyncio.create_task(self.otp_broadcast_service.broadcast_otp_created(otp_data))
                    except Exception as sync_error:
                        self.logger.error(f"Failed to broadcast OTP creation: {sync_error}")
                        # Continue - local DB has the OTP, sync is best-effort
                elif self.otp_sync_service:
                    # Fallback to RethinkDB sync for compatibility during migration
                    try:
                        self.otp_sync_service.publish_otp_event(otp_data)
                    except Exception as sync_error:
                        self.logger.error(f"Failed to sync OTP to RethinkDB: {sync_error}")

            # Send OTP via SMS
            if not self.aws_sms_service:
                raise RuntimeError("AWS SMS service not available - cannot send OTP")

            # Format message with expiry time
            message_template = aws_settings.otp_message_template.format(
                otp_code=otp_code,
                expiry_minutes=expiry_text
            )

            sms_result = await self.aws_sms_service.send_otp_sms(
                mobile_number=mobile_number,
                otp_code=otp_code,
                message_template=message_template,
                sender_id=aws_settings.sms_sender_id,
                sms_type=aws_settings.sms_type
            )

            if not sms_result.get('success'):
                # SMS delivery failed - we should consider cleaning up the stored OTP
                error_msg = f"Failed to send SMS to {mobile_number}: {sms_result.get('error_message', 'Unknown error')}"
                self.logger.error(error_msg)

                # Delete the OTP since it couldn't be delivered
                try:
                    self.otp_repository.delete_otp(mobile_number)
                except Exception as cleanup_error:
                    self.logger.error(f"Failed to cleanup OTP after SMS failure: {str(cleanup_error)}")

                raise RuntimeError(error_msg)

            # Log successful delivery
            self.logger.debug(f"OTP successfully sent via SMS to {mobile_number}, Message ID: {sms_result.get('message_id')}")

            # Determine appropriate response message
            if not should_generate_new:
                response_message = f"Existing OTP resent to {mobile_number} (valid until {expires_at.strftime('%Y-%m-%d %H:%M:%S')} UTC)"
            else:
                response_message = f"New OTP successfully sent to {mobile_number}"

            # Return success response (without the actual OTP code)
            return OTPResponse(
                message=response_message,
                mobile_number=mobile_number,
                otp_length=length,
                delivery_method="sms",
                sent_at=datetime.now(timezone.utc),
                otp_id=otp_id,
                expires_at=expires_at,
                random_number=None  # Explicitly set to None for security
            )

        except Exception as e:
            masked_number = f"+******{mobile_number[-4:]}" if mobile_number and len(mobile_number) > 4 else mobile_number
            self.logger.error(f"Failed to generate and send OTP to {masked_number}: {type(e).__name__}")
            raise

    async def get_random_number(self, length: int, email: str) -> OTPResponse:
        """
        Legacy method for backward compatibility.

        Args:
            length: Length of OTP code to generate
            email: Email address (legacy parameter)

        Returns:
            OTPResponse (legacy format)
        """
        # For backward compatibility, generate OTP but don't send via SMS
        random_number = self.unique_random_generator.generate_random_number(length)

        exists = self.otp_repository.otp_exists(email)
        if exists:
            self.otp_repository.update_otp(email, {
                'random_number': random_number
            })
        else:
            self.otp_repository.create_otp({
                'email': email,
                'random_number': random_number
            })

        # Return legacy response format
        return OTPResponse(
            message=f"Generated OTP code: {random_number} (legacy mode - not sent via SMS)",
            mobile_number="N/A",
            otp_length=length,
            delivery_method="legacy",
            random_number=random_number  # Include in legacy mode for compatibility
        )

    def verify_otp_from_selfie(
        self,
        mobile_number: str,
        otp_code: str,
        client_public_key: Optional[str] = None
    ) -> dict:
        """
        Verify OTP code extracted from selfie against stored value.
        This method is called during selfie document processing.

        Args:
            mobile_number: Mobile number (optional, not required if otp_code provided)
            otp_code: OTP code extracted from selfie via gesture detection
            client_public_key: Optional client public key for additional verification

        Returns:
            Dict with verification result
        """
        try:
            # Primary lookup: by OTP code (most reliable - what we extracted from video)
            if otp_code:
                stored_otp = self.otp_repository.get_otp_by_otp_code(otp_code)
                if stored_otp:
                    self.logger.debug(
                        f"Found OTP by code: {otp_code[:4]}*** "
                        f"for mobile: {stored_otp.get('mobile_number')}"
                    )
                else:
                    self.logger.warning(f"No valid OTP found for code: {otp_code[:4]}***")
                    return {
                        'valid': False,
                        'message': 'OTP not found, expired, or already used',
                        'otp_status': 'not_found'
                    }
            else:
                # Fallback: lookup by mobile_number (for backward compatibility)
                self.logger.debug(f"No otp_code provided, trying mobile_number lookup: {mobile_number}")
                stored_otp = self.otp_repository.get_valid_otp_by_mobile_number(mobile_number)
                if not stored_otp:
                    all_otp = self.otp_repository.get_otp_by_mobile_number(mobile_number)
                    if not all_otp:
                        return {
                            'valid': False,
                            'message': 'No OTP found for this mobile number',
                            'otp_status': 'not_found'
                        }
                    elif all_otp.get('is_verified'):
                        return {
                            'valid': False,
                            'message': 'OTP has already been used',
                            'otp_status': 'already_used'
                        }
                    else:
                        return {
                            'valid': False,
                            'message': f'OTP has expired',
                            'otp_status': 'expired'
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

            # Determine identifier for OTP operations
            # Prioritize stored_otp's mobile_number (from DB lookup) over potentially empty parameter
            identifier_for_operations = (
                stored_otp.get('mobile_number') or
                stored_otp.get('email') or
                mobile_number or
                client_public_key
            )

            # Increment attempts counter
            self.otp_repository.increment_otp_attempts(identifier_for_operations)

            # Verify OTP code
            if stored_otp.get('random_number') == otp_code:
                # Delete OTP after successful verification (one-time use)
                self.otp_repository.delete_otp(identifier_for_operations)

                # Broadcast deletion to peer instances
                if self.otp_broadcast_service:
                    try:
                        mobile = stored_otp.get('mobile_number') or mobile_number
                        asyncio.create_task(self.otp_broadcast_service.broadcast_otp_deleted(mobile))
                    except Exception as sync_error:
                        self.logger.error(f"Failed to broadcast OTP deletion: {sync_error}")

                self.logger.debug(f"OTP successfully verified and deleted via selfie processing")

                return {
                    'valid': True,
                    'message': 'OTP verified successfully via selfie',
                    'otp_status': 'verified',
                    'otp_code': stored_otp.get('random_number'),
                    'otp_id': stored_otp.get('otp_id'),
                    'expires_at': stored_otp.get('expires_at')
                }
            else:
                return {
                    'valid': False,
                    'message': f'Invalid OTP code found in selfie. Expected: {stored_otp.get("random_number")}, Got: {otp_code}',
                    'otp_status': 'invalid_code',
                    'expected_otp': stored_otp.get('random_number')
                }

        except Exception as e:
            masked_number = f"+******{mobile_number[-4:]}" if mobile_number and len(mobile_number) > 4 else mobile_number
            self.logger.error(f"OTP verification failed for {masked_number}: {type(e).__name__}")
            return {
                'valid': False,
                'message': 'Verification failed due to server error',
                'otp_status': 'error'
            }

    def verify_mobile_and_otp_from_selfie(
        self,
        mobile_number: str,
        otp_code: Optional[str] = None,
        client_public_key: Optional[str] = None
    ) -> dict:
        """
        Enhanced verification for selfie processing that checks mobile number and OTP.

        This method validates:
        1. Mobile number format and existence in OTP table
        2. If OTP code is provided, validates it against stored valid OTP
        3. Returns comprehensive result including OTP status

        Args:
            mobile_number: Mobile number from selfie request
            otp_code: OTP code extracted from selfie via OCR (optional)
            client_public_key: Optional client public key for additional verification

        Returns:
            Dict with comprehensive verification result including mobile and OTP validation
        """
        try:
            result = {
                'mobile_number_valid': False,
                'otp_check_passed': False,
                'otp_status': None,
                'overall_valid': False,
                'details': {}
            }

            # Validate mobile number format
            if not mobile_number or not mobile_number.startswith('+'):
                result['details']['mobile_number_error'] = 'Invalid mobile number format - must start with +'
                result['otp_status'] = 'invalid_mobile_format'
                return result

            # Check if there's any OTP record for this mobile number (valid or not)
            all_otp = self.otp_repository.get_otp_by_mobile_number(mobile_number)
            if not all_otp:
                result['details']['mobile_number_error'] = f'No OTP records found for mobile number: {mobile_number}'
                result['otp_status'] = 'no_otp_records'
                return result

            result['mobile_number_valid'] = True
            result['details']['mobile_number'] = mobile_number

            # If no OTP code provided, just check mobile number validity
            if not otp_code:
                # Check if there's a valid unexpired OTP
                valid_otp = self.otp_repository.get_valid_otp_by_mobile_number(mobile_number)
                if valid_otp:
                    result['otp_check_passed'] = True
                    result['otp_status'] = 'valid_otp_exists'
                    result['details']['otp_exists'] = True
                    result['details']['otp_expires_at'] = valid_otp.get('expires_at')
                    result['details']['otp_id'] = valid_otp.get('otp_id')
                else:
                    result['otp_status'] = 'no_valid_otp'
                    result['details']['otp_exists'] = False
                    if all_otp.get('is_verified'):
                        result['details']['reason'] = 'OTP already used'
                    elif all_otp.get('expires_at'):
                        expires_at = all_otp.get('expires_at')
                        now_utc = datetime.now(timezone.utc)
                        # Handle naive datetime
                        if expires_at.tzinfo is None:
                            expires_at = expires_at.replace(tzinfo=timezone.utc)
                        if now_utc > expires_at:
                            result['details']['reason'] = 'OTP expired'
                        else:
                            result['details']['reason'] = 'OTP not valid'
                    else:
                        result['details']['reason'] = 'OTP not valid'

                result['overall_valid'] = result['mobile_number_valid'] and result['otp_check_passed']
                return result

            # If OTP code is provided, perform full validation
            verification_result = self.verify_otp_from_selfie(mobile_number, otp_code, client_public_key)

            result['otp_check_passed'] = verification_result['valid']
            result['otp_status'] = verification_result.get('otp_status', verification_result.get('valid') and 'verified' or 'invalid')
            result['details']['verification_result'] = verification_result

            if verification_result['valid']:
                result['details']['verification_message'] = verification_result['message']
                result['details']['verified_otp'] = verification_result.get('otp_code')
                result['details']['verified_otp_id'] = verification_result.get('otp_id')
                result['details']['expires_at'] = verification_result.get('expires_at')
            else:
                result['details']['verification_error'] = verification_result['message']
                if 'expected_otp' in verification_result:
                    result['details']['expected_otp'] = verification_result['expected_otp']

            result['overall_valid'] = result['mobile_number_valid'] and result['otp_check_passed']
            return result

        except Exception as e:
            masked_number = f"+******{mobile_number[-4:]}" if mobile_number and len(mobile_number) > 4 else mobile_number
            self.logger.error(f"Mobile and OTP verification failed for {masked_number}: {type(e).__name__}")
            return {
                'mobile_number_valid': False,
                'otp_check_passed': False,
                'otp_status': 'error',
                'overall_valid': False,
                'details': {'error': f'Verification failed: {str(e)}'}
            }




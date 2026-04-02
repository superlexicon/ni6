import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import uuid

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    ClientError = Exception
    NoCredentialsError = Exception

from app.core.logger import get_logger


class AWSSMSService:
    """Service for sending SMS messages via AWS SNS"""

    def __init__(
        self,
        aws_access_key_id: str = None,
        aws_secret_access_key: str = None,
        aws_region: str = "us-east-1",
        sns_topic_arn: str = None
    ):
        """
        Initialize AWS SNS SMS service.

        Args:
            aws_access_key_id: AWS access key ID
            aws_secret_access_key: AWS secret access key
            aws_region: AWS region for SNS
            sns_topic_arn: SNS topic ARN (optional, for topic-based messaging)
        """
        self.logger = get_logger()
        self.aws_region = aws_region
        self.sns_topic_arn = sns_topic_arn

        # Check if boto3 is available
        if not BOTO3_AVAILABLE:
            self.logger.warning("boto3 not available - AWS SMS functionality will be disabled")
            self.sns_client = None
            return

        try:
            # Initialize SNS client
            if aws_access_key_id and aws_secret_access_key:
                self.sns_client = boto3.client(
                    'sns',
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    region_name=aws_region
                )
            else:
                # Use default AWS credential chain (IAM roles, environment variables, etc.)
                self.sns_client = boto3.client('sns', region_name=aws_region)

            self.logger.info(f"AWS SNS SMS service initialized for region: {aws_region}")

        except NoCredentialsError:
            self.logger.error("AWS credentials not found. Please configure AWS credentials.")
            self.sns_client = None
        except Exception as e:
            self.logger.error(f"Failed to initialize AWS SNS SMS service: {str(e)}")
            raise

    async def send_otp_sms(
        self,
        mobile_number: str,
        otp_code: str,
        message_template: str = None,
        sender_id: str = None,
        sms_type: str = "Transactional"
    ) -> Dict[str, Any]:
        """
        Send OTP code via AWS SNS SMS.

        Args:
            mobile_number: Mobile number with country code (e.g., +1234567890)
            otp_code: OTP code to send
            message_template: Custom message template (should contain {otp_code} placeholder)
            sender_id: Custom sender ID (depends on region support)
            sms_type: SMS type - 'Transactional', 'Promotional', or 'Transactional'

        Returns:
            Dict with delivery status and metadata
        """
        # Check if AWS service is available
        if not self.sns_client:
            return {
                'success': False,
                'error_message': 'AWS SMS service not available - boto3 not installed or credentials not configured',
                'message_id': None,
                'mobile_number': mobile_number
            }

        try:
            # Validate and normalize mobile number
            is_valid, clean_mobile_number = self._validate_mobile_number(mobile_number)
            if not is_valid:
                raise ValueError(f"Invalid mobile number format: {mobile_number}")

            # Create message
            if message_template:
                # message_template is already formatted by otp_service.py with otp_code and expiry_text
                # Use it directly as the message
                message = message_template
            else:
                message = f"Your verification code is: {otp_code}. This code will expire in 10 minutes."

            # Prepare SMS attributes
            sms_attributes = {
                'AWS.SNS.SMS.SMSType': {
                    'DataType': 'String',
                    'StringValue': sms_type
                }
            }

            # Add sender ID if provided
            if sender_id:
                sms_attributes['AWS.SNS.SMS.SenderID'] = {
                    'DataType': 'String',
                    'StringValue': sender_id
                }

            # Send SMS
            response = await asyncio.to_thread(
                self._send_sms_direct,
                clean_mobile_number,
                message,
                sms_attributes
            )

            # Log successful delivery
            self.logger.debug(f"OTP SMS sent successfully to {clean_mobile_number}")

            return {
                'success': True,
                'message_id': response.get('MessageId'),
                'mobile_number': clean_mobile_number,
                'sent_at': datetime.now(timezone.utc),
                'sms_type': sms_type,
                'response': response
            }

        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))

            self.logger.error(f"AWS SNS SMS error: {error_code} - {error_message}")

            return {
                'success': False,
                'error_code': error_code,
                'error_message': error_message,
                'mobile_number': mobile_number
            }

        except Exception as e:
            masked_number = f"+******{mobile_number[-4:]}" if mobile_number and len(mobile_number) > 4 else mobile_number
            self.logger.error(f"Failed to send OTP SMS to {masked_number}: {type(e).__name__}")
            return {
                'success': False,
                'error_message': str(e),
                'mobile_number': mobile_number
            }

    def _send_sms_direct(
        self,
        mobile_number: str,
        message: str,
        sms_attributes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Send SMS using boto3 SNS client (synchronous for asyncio.to_thread).

        Args:
            mobile_number: Mobile number with country code
            message: SMS message content
            sms_attributes: SMS attributes dict

        Returns:
            AWS SNS response
        """
        return self.sns_client.publish(
            PhoneNumber=mobile_number,
            Message=message,
            MessageAttributes=sms_attributes
        )

    def _validate_mobile_number(self, mobile_number: str) -> tuple[bool, str]:
        """
        Validate mobile number format and return normalized version.

        Args:
            mobile_number: Mobile number to validate

        Returns:
            Tuple of (is_valid, normalized_mobile_number)
        """
        if not mobile_number:
            return False, ""

        # Strip whitespace for validation
        mobile_number = mobile_number.strip().replace(' ', '')

        # Basic validation: should start with + and contain only digits after
        if not mobile_number.startswith('+'):
            return False, mobile_number

        # Remove + and check if remaining are digits
        digits = mobile_number[1:]
        if not digits.isdigit():
            return False, mobile_number

        # Check reasonable length (10-15 digits typical)
        if len(digits) < 10 or len(digits) > 15:
            return False, mobile_number

        return True, mobile_number

    async def check_delivery_status(self, message_id: str) -> Dict[str, Any]:
        """
        Check SMS delivery status (limited by SNS capabilities).

        Args:
            message_id: SNS message ID to check

        Returns:
            Delivery status information
        """
        try:
            # Note: SNS doesn't provide detailed delivery status for SMS
            # This is a placeholder for potential future enhancements

            return {
                'message_id': message_id,
                'status': 'sent',  # SNS considers it sent when accepted
                'note': 'Detailed delivery tracking not available for SNS SMS'
            }

        except Exception as e:
            self.logger.error(f"Failed to check delivery status for {message_id}: {str(e)}")
            return {
                'message_id': message_id,
                'status': 'unknown',
                'error': str(e)
            }

    async def send_bulk_otp_sms(
        self,
        recipients: list[Dict[str, str]],
        otp_generator_func = None
    ) -> Dict[str, Any]:
        """
        Send OTP SMS to multiple recipients (each gets unique OTP).

        Args:
            recipients: List of dicts with 'mobile_number' and optional 'name'
            otp_generator_func: Function to generate OTP codes

        Returns:
            Dict with bulk delivery results
        """
        results = []

        for recipient in recipients:
            mobile_number = recipient.get('mobile_number')

            if not mobile_number:
                results.append({
                    'mobile_number': None,
                    'success': False,
                    'error': 'Mobile number not provided'
                })
                continue

            # Generate unique OTP for each recipient
            if otp_generator_func:
                otp_code = otp_generator_func()
            else:
                # Default simple OTP generation
                import random
                otp_code = f"{random.randint(100000, 999999)}"

            # Personalize message if name is provided
            name = recipient.get('name', '')
            message_template = f"Hi {name}, your verification code is: {{otp_code}}." if name else None

            # Send SMS
            result = await self.send_otp_sms(
                mobile_number=mobile_number,
                otp_code=otp_code,
                message_template=message_template
            )

            result['otp_code'] = otp_code  # Include for verification purposes
            results.append(result)

        return {
            'total_recipients': len(recipients),
            'successful_sends': sum(1 for r in results if r.get('success')),
            'failed_sends': sum(1 for r in results if not r.get('success')),
            'results': results
        }
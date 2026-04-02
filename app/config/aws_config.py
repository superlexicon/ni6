try:
    from pydantic import BaseSettings, Field
except ImportError:
    from pydantic_settings import BaseSettings
    from pydantic import Field
from typing import Optional


class AWSSettings(BaseSettings):
    """AWS configuration settings"""

    # AWS SNS Settings
    aws_access_key_id: Optional[str] = Field(None, description="AWS Access Key ID")
    aws_secret_access_key: Optional[str] = Field(None, description="AWS Secret Access Key")
    aws_region: str = Field("us-east-1", description="AWS region for SNS")
    sns_topic_arn: Optional[str] = Field(None, description="SNS topic ARN for topic-based messaging")

    # SMS Settings
    sms_sender_id: Optional[str] = Field(None, description="Custom sender ID for SMS (region-dependent)")
    sms_type: str = Field("Transactional", description="Default SMS type (Transactional, Promotional)")
    otp_expiry_minutes: int = Field(30, description="OTP expiry time in minutes")

    # Message Templates
    otp_message_template: str = Field(
        "Your verification code is: {otp_code}. This code will expire in {expiry_minutes} minutes.",
        description="Default OTP message template"
    )

    model_config = {
        "env_prefix": "AWS_",
        "env_file": ".env",
        "extra": "ignore"  # Ignore extra environment variables
    }


# Global settings instance
aws_settings = AWSSettings()
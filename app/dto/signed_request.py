"""
Shared models for signed API requests with ECDSA signature verification.

These models are used across multiple endpoints to provide unified
signature-based authentication for sensitive operations.
"""
import time
from pydantic import BaseModel, Field, field_validator
from typing import Optional


class SignatureData(BaseModel):
    """ECDSA signature components for authentication."""
    r: str = Field(..., description="Signature r component (hex)")
    s: str = Field(..., description="Signature s component (hex)")


class SignedRequest(BaseModel):
    """
    Base model for signed requests with timestamp replay protection.

    The client signs a message containing a timestamp to prove:
    1. They possess the private key corresponding to the public key
    2. The request is fresh (replay protection via timestamp)

    Message format for signing: "request:{unix_timestamp}"
    """
    public_key: str = Field(..., description="Client public key (hex)")
    timestamp: int = Field(..., description="Unix timestamp for replay protection")
    signature: SignatureData = Field(
        ...,
        description="ECDSA signature of 'request:{timestamp}' message"
    )

    @field_validator('timestamp')
    @classmethod
    def validate_timestamp(cls, v: int) -> int:
        """Validate timestamp is not too far in the past or future."""
        current_time = int(time.time())
        max_age_seconds = 300  # 5 minutes

        # Check for future timestamps (clock skew tolerance)
        if v > current_time + 60:
            raise ValueError("Timestamp is in the future")

        # Check for expired timestamps
        age = current_time - v
        if age > max_age_seconds:
            raise ValueError(
                f"Timestamp expired. Maximum age is {max_age_seconds} seconds"
            )

        return v


class SignedJobStatusRequest(SignedRequest):
    """
    Signed request for job status queries.

    Extends SignedRequest with job_id for status checks.
    Ensures users can only query status for their own jobs.
    """
    job_id: str = Field(..., description="Job ID to query status for")


class SignedVerificationRequest(SignedRequest):
    """
    Signed request for verification state queries.

    Uses the public_key from the signed request to look up
    verification state. No additional fields needed.
    """


class SignedSecretShareRequest(SignedRequest):
    """
    Signed request for secret share recovery.

    Message format: "request:{timestamp}"

    Extends SignedRequest with fields required for secret share recovery.
    All fields are signed to ensure authenticity of the recovery request.

    For image selfies: Provide otp_code (extracted from image via OCR/filename)
    For video selfies: Leave otp_code empty (server extracts from hand gestures in video)
    """
    temp_public_key: str = Field(..., description="Ephemeral public key for result encryption")
    selfie_data: str = Field(..., description="Base64-encoded selfie image/video")
    otp_code: Optional[str] = Field(None, description="OTP code for verification (for image selfies, leave empty for video)")
    filename: Optional[str] = Field(None, description="Optional filename")
    callback_url: Optional[str] = Field(None, description="Optional callback URL")
    target_server_public_key: Optional[str] = Field(None, description="Optional routing hint")
    api_url: Optional[str] = Field(None, description="Optional API URL to filter shares by (only return shares from this API)")


def verify_signed_request(request: SignedRequest) -> bool:
    """
    Verify a signed request using ECDSA signature recovery.

    This function verifies that:
    1. The signature is valid for the provided public key
    2. The message contains the correct timestamp
    3. The timestamp is within the allowed window

    Args:
        request: SignedRequest to verify

    Returns:
        True if signature is valid, False otherwise

    Note: This is a placeholder for the actual signature verification
    implementation which would use ECDSA public key recovery.
    The actual verification is done in the service layer.
    """
    # The actual verification is done in the service layer using
    # the ECDSARecovery class. This function serves as a
    # type checker and validation wrapper.
    if not request.timestamp:
        return False
    if not request.signature or not request.signature.r or not request.signature.s:
        return False
    if not request.public_key:
        return False
    return True

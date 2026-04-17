from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict
from datetime import datetime
import uuid
import time


class OTPResponse(BaseModel):
    """Response model for OTP generation and SMS delivery"""

    message: str = Field(..., description="Success message confirming OTP delivery")
    mobile_number: str = Field(..., description="Mobile number OTP was sent to")
    otp_length: int = Field(..., description="Length of generated OTP")
    delivery_method: str = Field(default="sms", description="Method used to deliver OTP")
    sent_at: Optional[datetime] = Field(None, description="Timestamp when OTP was sent")
    otp_id: Optional[str] = Field(None, description="Unique identifier for this OTP request")
    expires_at: Optional[datetime] = Field(None, description="Timestamp when OTP expires")

    # For backward compatibility during transition
    random_number: Optional[str] = Field(None, description="Deprecated: OTP codes are no longer returned for security")


class SignatureData(BaseModel):
    """ECDSA signature components for authentication."""
    r: str = Field(..., description="Signature r component (hex)")
    s: str = Field(..., description="Signature s component (hex)")


class SignedOTPRequest(BaseModel):
    """
    Signed OTP request with user identity creation.

    **Security Features:**
    1. Signature verification (proves key ownership)
    2. Timestamp validation (replay protection)
    3. Creates user_identity_index and user_keys before sending OTP

    **Request Flow:**
    1. Client signs "otp:{timestamp}" with their private key
    2. Server verifies signature against client_public_key
    3. Server creates user_identity_index and user_keys records
    4. Server generates and sends OTP via SMS
    5. Server encrypts OTP response with hybrid encryption

    **Fields:**
    - client_public_key: User's public key for signature verification
    - mobile_number: Mobile number WITHOUT country code (e.g., '5551234567'). Country prefix will be added using country_code.
    - country_code: ISO country code (e.g., 'US', 'SG', 'GB'). Will be converted to phone prefix.
    - secret_share: Shamir secret share for multi-device support
    - timestamp: Unix timestamp for replay protection
    - signature: ECDSA signature of "otp:{timestamp}"
    - target_server_public_key: Server public key for response encryption
    """
    # Client identification
    client_public_key: str = Field(..., description="Client's public key (hex)")
    mobile_number: Optional[str] = Field(None, description="Mobile number WITHOUT country code/prefix (e.g., '5551234567'). If provided, OTP will be sent via SMS. Otherwise, OTP is only returned in encrypted response.")
    country_code: Optional[str] = Field(None, description="ISO country code (e.g., 'US', 'SG', 'GB'). Required if mobile_number is provided.")

    # Secret share (MOVED from selfie submission)
    secret_share: Optional[str] = Field(None, description="DEPRECATED - Shamir secret share (plaintext)")

    # NEW: ECIES-encrypted secret share
    secret_share_encrypted: Optional[Dict[str, str]] = Field(
        None,
        description="ECIES-encrypted secret share envelope: {version, ephemeral_public_key, encrypted_data, iv}"
    )

    # OTP configuration
    otp_length: int = Field(default=6, description="OTP code length (default 6)")

    # Signature verification
    timestamp: int = Field(..., description="Unix timestamp for replay protection")
    signature: SignatureData = Field(..., description="ECDSA signature of 'otp:{timestamp}'")

    # Response encryption
    target_server_public_key: str = Field(..., description="Server public key for response encryption")

    # OTP generation control
    generate_otp: bool = Field(
        default=True,
        description="Whether to generate and send OTP. Set to false for instances that should only process secret shares."
    )
    gesture_mode: bool = Field(
        default=False,
        description="If True, restrict OTP to digits 1-5 only (for hand gesture verification, 0 mis-detected as 1)"
    )

    # Device and API tracking
    device_id: Optional[str] = Field(None, description="Device identifier (optional)")
    api_url: Optional[str] = Field(None, description="API URL that should receive this share (optional)")

    @field_validator('timestamp')
    @classmethod
    def validate_timestamp(cls, v: int) -> int:
        """Validate timestamp is within 5 minutes."""
        current_time = int(time.time())
        max_age_seconds = 300  # 5 minutes

        if v > current_time + 60:
            raise ValueError("Timestamp is in the future")
        if current_time - v > max_age_seconds:
            raise ValueError(f"Timestamp expired. Maximum age is {max_age_seconds} seconds")
        return v


class EncryptedOTPResponse(BaseModel):
    """
    Encrypted OTP response using hybrid encryption.

    The OTP is encrypted with AES-256-GCM, and the symmetric key
    is encrypted using ECDH + Salsa20 with client's public key.
    """
    # Client's ephemeral public key for this response
    client_public_key: str = Field(..., description="Ephemeral public key for ECDH (hex)")

    # Encrypted symmetric key (encrypted with client's public_key + ECDH + Salsa20)
    encrypted_key: str = Field(..., description="Base64 encrypted AES key")

    # IV for key decryption (Salsa20)
    key_iv: str = Field(..., description="Base64 IV for key decryption")

    # Encrypted payload (encrypted with AES-256-GCM)
    encrypted_payload: str = Field(..., description="Base64 encrypted JSON payload")

    # IV for payload decryption (AES-GCM)
    payload_iv: str = Field(..., description="Base64 IV for payload decryption")

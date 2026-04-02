from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
import base64
import time


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CALLBACK_FAILED = "callback_failed"


class FileObject(BaseModel):
    """Individual file object with type specification"""
    filename: str
    file_data: str  # Base64 encoded file content
    file_type: str  # "document" or "selfie"
    document_type: Optional[str] = "auto"  # "auto" = detect, specific value = hint
    media_type: Optional[str] = None  # "image" or "video" - auto-detected from filename if not provided
    country: Optional[str] = "auto"  # "auto" = detect, specific value = hint (ISO2)
    entity: Optional[str] = "auto"  # "auto" = detect, specific value = hint
    expected_values: Optional[Dict[str, Any]] = None  # Expected values for field verification (document_type="field_verification")


class JobRequest(BaseModel):
    """
    Request model for creating a new async document analysis job - stores raw request data.

    NOTE: secret_share, mobile_number, and country_code are now
    provided during the OTP request. User data is looked up via client_public_key
    from the user_keys table. The device_identifier field has been removed.

    For secret_share_recovery:
    - identity_id: Direct identity_id for face biometrics lookup (from user_keys via mobile_number)
    - mobile_number: Mobile number for user_keys lookup (from OTP table)
    """
    # Store raw request data - validation and processing done by worker
    encrypted_archive: Optional[str] = None  # Encrypted archive (will be processed by worker)
    client_public_key: str  # Hex ephemeral public key for ECDH decryption (used for user lookup)
    iv: str = ""  # IV for decryption (will be processed by worker) - optional for encrypted envelope

    # New format - explicit file objects with type information
    files: Optional[List[FileObject]] = None  # Array of file objects (will be processed by worker)

    # NOTE: secret_share, country_code are now in the OTP request.
    # User is looked up by client_public_key.

    # Optional callback URL (overrides default from env)
    callback_url: Optional[str] = None

    # Target server's public key for multi-instance routing
    # The server whose public key was used to encrypt the envelope
    target_server_public_key: Optional[str] = None

    # Encrypted envelope fields (for hybrid encryption)
    encrypted_key: Optional[str] = None  # Base64 encoded encrypted AES key
    key_iv: Optional[str] = None  # Base64 IV for key decryption
    encrypted_payload: Optional[str] = None  # Base64 encoded encrypted JSON payload
    payload_iv: Optional[str] = None  # Base64 IV for payload decryption

    # Fields for secret_share_recovery (used when client doesn't have registered key)
    identity_id: Optional[str] = None  # For direct face biometrics lookup
    mobile_number: Optional[str] = None  # For user_keys lookup
    otp_code: Optional[str] = None  # OTP code for verification
    api_url: Optional[str] = None  # API URL to filter shares by (only return shares from this API)

    # Legacy fields (deprecated - kept for backward compatibility)
    # These are no longer used and will be ignored
    secret_share: Optional[str] = Field(None, deprecated=True, description="Deprecated: Use OTP request instead")
    country_code: Optional[str] = Field(None, deprecated=True, description="Deprecated: Use OTP request instead")
    temp_public_key: Optional[str] = Field(None, deprecated=True, description="Deprecated: No longer needed")


class JobSubmissionResponse(BaseModel):
    """Response model for job submission"""
    success: bool
    job_id: str
    status: JobStatus
    message: str = "Job queued successfully"
    expected_completion_time_seconds: Optional[float] = None  # Estimated wait time in seconds


class JobInfo(BaseModel):
    """Job information model"""
    job_id: str
    status: JobStatus
    created_at: datetime
    updated_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    callback_attempted_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    error_message: Optional[str] = None
    callback_url: Optional[str] = None
    request_data: Optional[Dict[str, Any]] = None


class JobStatusResponse(BaseModel):
    """Response model for job status query"""
    job_info: JobInfo
    results: Optional[Dict[str, Any]] = None
    callback_sent: Optional[bool] = None


class JobDatabaseRecord(BaseModel):
    """Database record model for jobs"""
    id: str
    status: JobStatus
    request_data: Dict[str, Any]
    response_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    callback_url: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime
    updated_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    callback_attempted_at: Optional[datetime] = None


class SignatureData(BaseModel):
    """ECDSA signature components for authentication."""
    r: str = Field(..., description="Signature r component (hex)")
    s: str = Field(..., description="Signature s component (hex)")


class SignedJobRequest(BaseModel):
    """
    Signed job request with state validation.

    Uses client_public_key for both:
    1. Signature verification (proves ownership of registered key)
    2. User identity lookup

    The ephemeral ECDH key is handled internally during envelope decryption.

    **Security Features:**
    1. Signature verification (proves key ownership)
    2. Timestamp validation (replay protection)
    3. State validation BEFORE queuing (prevents garbage submissions)

    **Request Flow:**
    1. Client signs "request:{timestamp}" with their registered private key
    2. Server verifies signature against client_public_key (registered key)
    3. Server validates timestamp is within 5 minutes
    4. Server validates verification state before queuing
    5. Job is only queued if all validations pass

    **State Validation:**
    - Selfie: No previous state required (state 0)
    - Passport: Requires state == 1 (selfie completed)
    - Bank Statement: Requires sequence_no >= 2 (passport data extracted)
    - National ID: Requires state == 1 (selfie completed)
    - Driving License: Requires state == 1 (selfie completed)
    - Resume: Requires state == 3 (full verification complete)

    **Note:** The encrypted envelope contains an ephemeral ECDH key for decryption.
    After the worker decrypts the envelope, it extracts the real client_public_key
    (registered key) from the payload for user lookups.
    """
    # Client's registered public key (for signature verification + state validation)
    client_public_key: str = Field(..., description="Client's registered public key (hex)")
    timestamp: int = Field(..., description="Unix timestamp for replay protection")
    signature: SignatureData = Field(..., description="ECDSA signature of timestamp")

    # Encrypted envelope fields (ephemeral key for decryption, discarded after)
    encrypted_key: str = Field(..., description="Base64 encrypted AES key")
    key_iv: str = Field(..., description="Base64 IV for key decryption")
    encrypted_payload: str = Field(..., description="Base64 encrypted JSON payload")
    payload_iv: str = Field(..., description="Base64 IV for payload decryption")

    # Required fields
    target_server_public_key: str = Field(..., description="Server public key for routing")
    callback_url: Optional[str] = Field(None, description="Optional callback URL")

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
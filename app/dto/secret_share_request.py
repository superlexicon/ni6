from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class SecretShareRequestPayload(BaseModel):
    """Request model for secret share retrieval"""
    # public_key REMOVED - will be looked up via identity_id from face match
    temp_public_key: str = Field(..., description="Temporary public key for re-encryption")
    selfie_data: str = Field(..., description="Base64 encoded selfie image")
    filename: Optional[str] = Field(None, description="Original filename (may contain OTP)")
    otp_code: Optional[str] = Field(None, description="OTP code for validation during recovery")
    callback_url: Optional[str] = Field(None, description="URL for async callback when recovery completes")

class RecoveredShare(BaseModel):
    """Single recovered secret share"""
    public_key: str = Field(..., description="User's public key for this share")
    encrypted_share: str = Field(..., description="Secret share re-encrypted with temp_public_key")
    iv: str = Field(..., description="IV for Salsa20 decryption")


class SecretShareResponse(BaseModel):
    """Response model for secret share retrieval"""
    success: bool = Field(..., description="Whether retrieval succeeded")
    identity_id: Optional[str] = Field(None, description="Matched identity ID")
    shares: Optional[list[RecoveredShare]] = Field(None, description="List of recovered shares (all devices for this identity)")
    total_shares: Optional[int] = Field(None, description="Total number of shares recovered")
    # Legacy fields for backward compatibility (deprecated)
    encrypted_secret_share: Optional[str] = Field(None, description="DEPRECATED: Use shares[] instead")
    iv: Optional[str] = Field(None, description="DEPRECATED: Use shares[].iv instead")
    message: str = Field(..., description="Success or error message")
    face_match_confidence: Optional[float] = Field(None, description="Cosine similarity score (0-1)")
    faces_checked: Optional[int] = Field(None, description="Number of face embeddings checked")
    extracted_data_encrypted: Optional[Dict[str, Any]] = Field(None, description="ECIES envelope for client-side decryption")

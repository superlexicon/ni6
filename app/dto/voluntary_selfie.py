from pydantic import BaseModel, Field
from typing import Optional

class VoluntarySelfieRequest(BaseModel):
    """Request model for voluntary selfie submission"""
    public_key: str = Field(..., description="User's public key")
    selfie_data: str = Field(..., description="Base64 encoded selfie image")
    filename: Optional[str] = Field(None, description="Original filename")

class VoluntarySelfieResponse(BaseModel):
    """Response model for voluntary selfie submission"""
    success: bool = Field(..., description="Whether submission succeeded")
    message: str = Field(..., description="Success or error message")
    face_quality_score: Optional[float] = Field(None, description="Face quality score (0-1)")
    anti_spoofing_score: Optional[float] = Field(None, description="Liveness score (0-1)")
    biometric_id: Optional[str] = Field(None, description="Stored biometric ID")

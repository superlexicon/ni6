from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class FaceQuality(BaseModel):
    """Face quality metrics."""

    sharpness: Optional[float] = Field(None, description="Face sharpness score (0-1)")
    brightness: Optional[float] = Field(None, description="Face brightness score (0-1)")
    contrast: Optional[float] = Field(None, description="Face contrast score (0-1)")
    pose_quality: Optional[float] = Field(None, description="Face pose quality score (0-1)")
    resolution_quality: Optional[float] = Field(None, description="Image resolution quality score (0-1)")


class FaceLiveness(BaseModel):
    """Face liveness detection results."""

    anti_spoof_score: float = Field(..., description="Anti-spoofing confidence score (0-1)")
    real_probability: float = Field(..., description="Probability face is real (0-1)")
    spoof_probability: float = Field(..., description="Probability face is spoof (0-1)")
    liveness_method: str = Field(..., description="Method used for liveness detection")
    eye_detection_score: Optional[float] = Field(None, description="Eye detection confidence (0-1)")
    mouth_detection_score: Optional[float] = Field(None, description="Mouth detection confidence (0-1)")
    texture_analysis_score: Optional[float] = Field(None, description="Texture analysis score (0-1)")


class FaceMetadata(BaseModel):
    """Metadata about extracted face."""

    face_bbox: Optional[Dict[str, int]] = Field(None, description="Face bounding box coordinates")
    face_size: Optional[Dict[str, int]] = Field(None, description="Face width and height")
    face_confidence: Optional[float] = Field(None, description="Face detection confidence (0-1)")
    image_size: Optional[Dict[str, int]] = Field(None, description="Original image dimensions")
    extraction_method: str = Field(default="passport", description="Method used for face extraction")
    quality_metrics: Optional[FaceQuality] = Field(None, description="Face quality assessment")
    liveness_results: Optional[FaceLiveness] = Field(None, description="Liveness detection results")
    detection_timestamp: Optional[datetime] = Field(None, description="When face was detected")


class FaceMatchResult(BaseModel):
    """Result of face matching between two biometrics."""

    distance_score: float = Field(..., ge=0.0, le=1.0, description="Face distance score (0=identical, 1=completely different)")
    match_confidence: float = Field(..., ge=0.0, le=1.0, description="Match confidence")
    is_match: bool = Field(..., description="Whether faces match below distance threshold")
    threshold_used: float = Field(..., description="Distance threshold applied (lower = more similar)")
    distance_metric: str = Field(default="cosine", description="Distance metric used")
    match_timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class FaceVerificationRequest(BaseModel):
    """Request model for face verification."""

    public_key: str = Field(..., description="User's public key to retrieve stored face")
    encrypted_secret_share: Optional[str] = Field(None, description="Encrypted share of user's secret seed")
    selfie_face_embedding: List[float] = Field(..., description="Face embedding from selfie")
    verification_threshold: float = Field(default=0.31, ge=0.0, le=1.0, description="Distance threshold (lower=more strict)")
    include_liveness_check: bool = Field(default=True, description="Include liveness verification")
    anti_spoof_threshold: float = Field(default=0.85, ge=0.0, le=1.0, description="Anti-spoof threshold")


class FaceVerificationResponse(BaseModel):
    """Response model for face verification results."""

    verification_passed: bool = Field(..., description="Overall verification result")
    face_match_result: FaceMatchResult = Field(..., description="Face matching details")
    liveness_check_passed: Optional[bool] = Field(None, description="Liveness verification result")
    anti_spoof_score: Optional[float] = Field(None, description="Selfie anti-spoof score")
    verification_timestamp: datetime = Field(default_factory=datetime.now)
    verification_details: Dict[str, Any] = Field(default_factory=dict, description="Additional details")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
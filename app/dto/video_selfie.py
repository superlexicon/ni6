"""
Video selfie verification models.

Pydantic models for video processing results, gesture transitions,
and video metadata for video-based selfie verification with hand gestures.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime


class VideoMetadata(BaseModel):
    """Metadata extracted from video file."""

    duration_seconds: float = Field(..., description="Video duration in seconds")
    frame_count: int = Field(..., description="Total number of frames")
    fps: float = Field(..., description="Frames per second")
    width: int = Field(..., description="Video width in pixels")
    height: int = Field(..., description="Video height in pixels")
    format: str = Field(..., description="Video file format (mp4, mov, webm)")
    size_bytes: int = Field(..., description="File size in bytes")

    @field_validator('duration_seconds', 'frame_count', 'fps', 'width', 'height', 'size_bytes')
    @classmethod
    def validate_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError('Value must be non-negative')
        return v


class FrameGestureResult(BaseModel):
    """Result of gesture detection for a single frame."""

    finger_count: int = Field(..., ge=0, le=5, description="Number of fingers detected (0-5)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    hand_detected: bool = Field(..., description="Whether a hand was detected")
    handedness: Optional[str] = Field(None, description="Left or Right hand")
    frame_index: int = Field(..., ge=0, description="Frame index in video")
    timestamp_seconds: float = Field(..., ge=0.0, description="Timestamp in seconds")


class GestureTransition(BaseModel):
    """A detected stable gesture transition."""

    digit: int = Field(..., ge=0, le=5, description="The finger count representing OTP digit")
    frame_start: int = Field(..., ge=0, description="First frame index of this gesture")
    frame_end: int = Field(..., ge=0, description="Last frame index of this gesture")
    duration_seconds: float = Field(..., ge=0.0, description="Duration of this gesture in seconds")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Average detection confidence")

    @field_validator('frame_end')
    @classmethod
    def validate_frame_range(cls, v: int, info) -> int:
        if 'frame_start' in info.data:
            if v < info.data['frame_start']:
                raise ValueError('frame_end must be >= frame_start')
        return v


class GestureOTPExtractionResult(BaseModel):
    """Result of OTP extraction from gesture sequence."""

    otp: str = Field(..., description="Extracted OTP string")
    transitions: List[GestureTransition] = Field(default_factory=list, description="All detected transitions")
    success: bool = Field(..., description="Whether extraction was successful")
    error: Optional[str] = Field(None, description="Error message if failed")
    frames_processed: int = Field(default=0, ge=0, description="Total frames processed")
    hand_detection_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Rate of frames with hand detected")


class VideoSelfieResult(BaseModel):
    """
    Complete result of video selfie verification.

    Includes OTP extraction, video metadata, gesture transitions,
    forgery checks, and other validation results.
    """

    # Result
    result: bool = Field(..., description="Overall verification result")
    status: str = Field(..., description="Status: 'completed' or 'failed'")

    # Video info
    video_metadata: Optional[VideoMetadata] = Field(None, description="Video file metadata")
    frames_processed: int = Field(default=0, ge=0, description="Number of frames processed")

    # OTP extraction
    extracted_data: Dict[str, Any] = Field(default_factory=dict, description="Extracted OTP and related data")
    gesture_transitions: List[GestureTransition] = Field(default_factory=list, description="Gesture transitions detected")

    # Validation results
    forgery_checks: Dict[str, Any] = Field(default_factory=dict, description="PhotoHolmes forgery detection results")
    other_checks: Dict[str, Any] = Field(default_factory=dict, description="Anti-spoofing, face detection, etc.")

    # Processing info
    job_id: Optional[str] = Field(None, description="Job identifier")
    processing_time_seconds: float = Field(default=0.0, ge=0.0, description="Processing time in seconds")
    error: Optional[str] = Field(None, description="Error message if failed")
    user_identity_id: Optional[str] = Field(None, description="User identity ID")

    # Verification state
    verification_state: int = Field(default=0, ge=0, le=3, description="Verification state after processing")
    sequence_no: int = Field(default=0, ge=0, le=3, description="Sequence number after processing")


class VideoSelfieRequest(BaseModel):
    """Request model for video selfie processing."""

    client_public_key: str = Field(..., description="Client's public key")
    video_data: str = Field(..., description="Base64 encoded video data")
    filename: str = Field(..., description="Video filename")
    file_type: str = Field(default="video_selfie", description="Document type")
    iv: str = Field(..., description="IV for decryption")
    mobile_number: Optional[str] = Field(None, description="User's mobile number")
    country_code: Optional[str] = Field(None, description="Country code for mobile number")
    callback_url: Optional[str] = Field(None, description="Callback URL")

    @field_validator('file_type')
    @classmethod
    def validate_file_type(cls, v: str) -> str:
        if v != "video_selfie":
            raise ValueError("file_type must be 'video_selfie'")
        return v


class VideoSelfieResponse(BaseModel):
    """Response model for video selfie processing."""

    result: bool = Field(..., description="Overall success status")
    status: str = Field(..., description="Processing status")
    job_id: Optional[str] = Field(None, description="Job ID")
    verification_state: int = Field(..., description="Verification state (0-3)")
    sequence_no: int = Field(..., description="Sequence number (0-3)")
    processing_time_seconds: float = Field(..., description="Processing time")

    # Extracted data
    extracted_data: Dict[str, Any] = Field(default_factory=dict)
    forgery_checks: Dict[str, Any] = Field(default_factory=dict)
    other_checks: Dict[str, Any] = Field(default_factory=dict)

    # Video-specific
    video_metadata: Optional[VideoMetadata] = Field(None)
    gesture_transitions: List[GestureTransition] = Field(default_factory=list)

    # Optional error
    error: Optional[str] = Field(None)
    user_identity_id: Optional[str] = Field(None)

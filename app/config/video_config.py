"""
Video processing configuration for video selfie verification.

Contains all constraints and parameters for video-based selfie processing
including gesture detection, frame sampling, and OTP extraction.
"""

from pydantic import BaseModel


class VideoConstraints(BaseModel):
    """Video file constraints for selfie uploads."""

    max_duration_seconds: int = 30
    max_size_mb: int = 100
    supported_formats: list[str] = ['mp4', 'mov', 'webm', 'avi', 'mkv']


class FrameSamplingConfig(BaseModel):
    """Configuration for frame extraction from video."""

    # Mobile app UI sequence for gesture mode (4-digit OTP):
    # 0-1s: "Ready" overlay
    # 1-2s: "Set" overlay
    # 2-3s: "Go" overlay
    # For each digit (repeated 4 times):
    #   - Digit displayed (2.0s) <- sample frame here
    #   - GAP period with NO digit shown (2.0s)
    # Total: 3s + 4 * (2s + 2s) = 19s for 4-digit gesture OTP

    digit_display_start_second: float = 3.0  # Digits start after "Go" (3s total)
    digit_display_duration: float = 2.0  # Each digit shows for 2s with no countdown
    gap_duration: float = 2.0  # Gap between digits with nothing shown

    # Each digit cycle = 2.0s (digit display) + 2.0s (gap) = 4.0s total
    # Frames sampled at MIDDLE of each digit display window (for more stable hand positions)
    # Digit 1: 3.5s (middle of 3-5s window)
    # Digit 2: 7.5s (middle of 7-9s window)
    # Digit 3: 11.5s (middle of 11-13s window)
    # Digit 4: 15.5s (middle of 15-17s window)
    # Aligned with face extraction timing for consistency
    digit_sample_times: list[float] = [3.5, 7.5, 11.5, 15.5]

    # Validation
    min_video_duration_seconds: int = 16  # Minimum for 4-digit gesture OTP (3s prep + 4*4.0s cycles)
    max_video_duration_seconds: int = 25  # Accommodate 4-digit gesture OTP with buffer

    # Legacy (for backward compatibility)
    frame_sampling_fps: int = 3  # Not used for guided recording
    min_frames_required: int = 4  # Minimum frames for validation


class GestureDetectionConfig(BaseModel):
    """Configuration for hand gesture detection."""

    min_gesture_stability_frames: int = 3  # Consecutive frames for stable gesture
    max_gesture_transition_seconds: float = 5.0  # Max time between gesture changes
    min_hand_detection_confidence: float = 0.5  # MediaPipe detection threshold


class OTPGestureConfig(BaseModel):
    """Configuration for gesture-based OTP."""

    allowed_digits: str = '12345'  # Only 1-5 for single-hand representation (0 mis-detected as 1, 6 not possible with one hand)
    min_length: int = 4  # Minimum OTP length
    max_length: int = 6  # Maximum OTP length


class VideoConfig:
    """
    Central configuration for video selfie processing.

    Example:
        config = VideoConfig()
        max_duration = config.constraints.max_duration_seconds
    """

    def __init__(self):
        self.constraints = VideoConstraints()
        self.frame_sampling = FrameSamplingConfig()
        self.gesture_detection = GestureDetectionConfig()
        self.otp_gesture = OTPGestureConfig()

    # Convenience properties for direct access
    @property
    def max_video_duration_seconds(self) -> int:
        # Use frame_sampling max duration for guided recording
        return self.frame_sampling.max_video_duration_seconds

    @property
    def max_video_size_mb(self) -> int:
        return self.constraints.max_size_mb

    @property
    def supported_formats(self) -> list[str]:
        return self.constraints.supported_formats

    @property
    def frame_sampling_fps(self) -> int:
        # Kept for backward compatibility but not used for guided recording
        return self.frame_sampling.frame_sampling_fps

    @property
    def min_frames_required(self) -> int:
        return self.frame_sampling.min_frames_required

    @property
    def min_gesture_stability_frames(self) -> int:
        return self.gesture_detection.min_gesture_stability_frames

    @property
    def max_gesture_transition_seconds(self) -> float:
        return self.gesture_detection.max_gesture_transition_seconds

    @property
    def otp_allowed_digits(self) -> str:
        return self.otp_gesture.allowed_digits

    @property
    def otp_min_length(self) -> int:
        return self.otp_gesture.min_length

    @property
    def otp_max_length(self) -> int:
        return self.otp_gesture.max_length


# Global instance for easy import
video_config = VideoConfig()

"""
Gesture OTP Extractor - Extract OTP sequence from hand gesture transitions.

Detects stable gesture changes in video frames and builds OTP sequence
from finger count transitions (0-5).
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np

from app.core.logger import get_logger
from app.config.video_config import video_config


@dataclass
class GestureTransition:
    """A detected stable gesture transition."""

    digit: int  # The finger count (0-5)
    frame_start: int  # First frame where this gesture appears
    frame_end: int  # Last frame where this gesture appears
    duration_seconds: float  # Time duration of this gesture
    confidence: float  # Average detection confidence


@dataclass
class OTPExtractionResult:
    """Result of OTP extraction from gesture sequence."""

    otp: str  # Extracted OTP string
    transitions: List[GestureTransition]  # All detected transitions
    success: bool
    error: Optional[str] = None
    frames_processed: int = 0
    hand_detection_rate: float = 0.0


class ExtractionError(Enum):
    """Error types for OTP extraction."""
    NO_HANDS_DETECTED = "No hands detected in video"
    INSUFFICIENT_FRAMES = "Insufficient valid frames for analysis"
    NO_TRANSITIONS = "No gesture transitions detected"
    INVALID_OTP_LENGTH = "OTP length outside valid range"
    UNSTABLE_GESTURES = "Gestures too unstable for reliable extraction"


class GestureOTPExtractor:
    """
    Extract OTP from hand gesture sequence.

    For guided recording: Each frame corresponds to a specific digit position.
    Frame 0 = digit 1 (at 5s), Frame 1 = digit 2 (at 7s), etc.

    For unguided recording: Uses transition detection to find stable gestures.
    """

    def __init__(
        self,
        min_stability_frames: Optional[int] = None,
        max_transition_seconds: Optional[float] = None,
        fps: float = 2.0
    ):
        """
        Initialize gesture OTP extractor.

        Args:
            min_stability_frames: Minimum consecutive frames for stable gesture
            max_transition_seconds: Maximum time allowed between transitions
            fps: Video FPS for time calculations
        """
        self.logger = get_logger()
        self.min_stability_frames = min_stability_frames or video_config.min_gesture_stability_frames
        self.max_transition_seconds = max_transition_seconds or video_config.max_gesture_transition_seconds
        self.fps = fps

    def extract_otp_guided(
        self,
        finger_counts: List[int],
        confidences: List[float]
    ) -> OTPExtractionResult:
        """
        Extract OTP from guided recording where each frame = one digit position.

        Frame 0 = digit 1, Frame 1 = digit 2, etc.
        Much simpler than transition detection.

        Args:
            finger_counts: List of finger counts per frame (0-5, -1 for no hand)
            confidences: Detection confidence per frame

        Returns:
            OTPExtractionResult with extracted OTP or error
        """
        frames_total = len(finger_counts)
        hand_detected_count = sum(1 for c in finger_counts if c >= 0)
        hand_detection_rate = hand_detected_count / frames_total if frames_total > 0 else 0.0

        # Check for sufficient hand detection
        min_required = video_config.otp_min_length
        if hand_detected_count < min_required:
            self.logger.warning(
                f"Insufficient hand detection: {hand_detected_count}/{frames_total} frames (need {min_required})"
            )
            return OTPExtractionResult(
                otp="",
                transitions=[],
                success=False,
                error=ExtractionError.NO_HANDS_DETECTED.value,
                frames_processed=frames_total,
                hand_detection_rate=hand_detection_rate
            )

        # Build OTP directly from finger counts (each frame = one digit)
        otp_digits = []
        transitions = []

        for idx, count in enumerate(finger_counts):
            if count >= 0:  # Valid detection
                otp_digits.append(str(count))
                # Create a simple transition object for compatibility
                transitions.append(GestureTransition(
                    digit=count,
                    frame_start=idx,
                    frame_end=idx,
                    duration_seconds=0.0,
                    confidence=confidences[idx] if idx < len(confidences) else 0.0
                ))

        otp = ''.join(otp_digits)

        # Validate OTP length
        min_len = video_config.otp_min_length
        max_len = video_config.otp_max_length

        if len(otp) < min_len or len(otp) > max_len:
            self.logger.warning(
                f"Invalid OTP length: {len(otp)} (expected {min_len}-{max_len})"
            )
            return OTPExtractionResult(
                otp=otp,
                transitions=transitions,
                success=False,
                error=ExtractionError.INVALID_OTP_LENGTH.value,
                frames_processed=frames_total,
                hand_detection_rate=hand_detection_rate
            )

        # Validate OTP contains only allowed digits (1-5, configured in video_config)
        allowed = set(video_config.otp_allowed_digits)
        if not all(d in allowed for d in otp):
            self.logger.warning(f"OTP contains invalid digits: {otp}")
            return OTPExtractionResult(
                otp=otp,
                transitions=transitions,
                success=False,
                error="OTP contains invalid digits (only 1-5 allowed for gesture mode)",
                frames_processed=frames_total,
                hand_detection_rate=hand_detection_rate
            )

        self.logger.info(
            f"Successfully extracted OTP from guided recording: {otp} from {len(finger_counts)} frames"
        )

        return OTPExtractionResult(
            otp=otp,
            transitions=transitions,
            success=True,
            frames_processed=frames_total,
            hand_detection_rate=hand_detection_rate
        )

    def apply_stability_filter(
        self,
        finger_counts: List[int]
    ) -> List[int]:
        """
        Apply stability filter to reduce noise in gesture detection.

        Requires N consecutive frames with the same count to be considered stable.
        Frames that don't meet stability threshold are set to -1 (invalid).

        Args:
            finger_counts: List of finger counts per frame (-1 if no hand)

        Returns:
            Filtered list with -1 for unstable/invalid frames
        """
        if not finger_counts:
            return []

        filtered = [-1] * len(finger_counts)
        n = self.min_stability_frames

        # Sliding window to find stable sequences
        for i in range(len(finger_counts) - n + 1):
            window = finger_counts[i:i + n]

            # Check if all values in window are valid (not -1) and same
            if all(w == window[0] and w >= 0 for w in window):
                # Mark all frames in window as stable
                for j in range(i, i + n):
                    filtered[j] = window[0]

        return filtered

    def find_transitions(
        self,
        stable_counts: List[int],
        confidences: List[float]
    ) -> List[GestureTransition]:
        """
        Find gesture transitions from stable filtered counts.

        Args:
            stable_counts: Stability-filtered finger counts
            confidences: Detection confidence per frame

        Returns:
            List of GestureTransition objects
        """
        transitions: List[GestureTransition] = []
        if not stable_counts:
            return transitions

        # Find unique gesture segments
        current_digit = None
        start_idx = None

        for idx, count in enumerate(stable_counts):
            if count < 0:  # Invalid frame
                if current_digit is not None:
                    # End current segment
                    self._add_transition(
                        transitions,
                        current_digit,
                        start_idx,
                        idx - 1,
                        confidences
                    )
                    current_digit = None
                    start_idx = None
                continue

            if current_digit is None:
                # Start new segment
                current_digit = count
                start_idx = idx
            elif count != current_digit:
                # Transition detected
                self._add_transition(
                    transitions,
                    current_digit,
                    start_idx,
                    idx - 1,
                    confidences
                )
                current_digit = count
                start_idx = idx

        # Add final segment if exists
        if current_digit is not None and start_idx is not None:
            self._add_transition(
                transitions,
                current_digit,
                start_idx,
                len(stable_counts) - 1,
                confidences
            )

        return transitions

    def _add_transition(
        self,
        transitions: List[GestureTransition],
        digit: int,
        frame_start: int,
        frame_end: int,
        confidences: List[float]
    ) -> None:
        """Add a transition to the list."""
        # Calculate duration
        duration = (frame_end - frame_start + 1) / self.fps

        # Calculate average confidence
        segment_confs = confidences[frame_start:frame_end + 1]
        avg_confidence = np.mean(segment_confs) if segment_confs else 0.0

        transitions.append(GestureTransition(
            digit=digit,
            frame_start=frame_start,
            frame_end=frame_end,
            duration_seconds=duration,
            confidence=avg_confidence
        ))

    def extract_otp(
        self,
        finger_counts: List[int],
        confidences: List[float]
    ) -> OTPExtractionResult:
        """
        Extract OTP from finger count sequence.

        Args:
            finger_counts: List of finger counts per frame (0-5, -1 for no hand)
            confidences: Detection confidence per frame

        Returns:
            OTPExtractionResult with extracted OTP or error
        """
        frames_total = len(finger_counts)
        hand_detected_count = sum(1 for c in finger_counts if c >= 0)
        hand_detection_rate = hand_detected_count / frames_total if frames_total > 0 else 0.0

        # Check for sufficient hand detection
        if hand_detected_count < self.min_stability_frames:
            self.logger.warning(
                f"Insufficient hand detection: {hand_detected_count}/{frames_total} frames"
            )
            return OTPExtractionResult(
                otp="",
                transitions=[],
                success=False,
                error=ExtractionError.NO_HANDS_DETECTED.value,
                frames_processed=frames_total,
                hand_detection_rate=hand_detection_rate
            )

        # Apply stability filter
        stable_counts = self.apply_stability_filter(finger_counts)
        stable_count = sum(1 for c in stable_counts if c >= 0)

        if stable_count < self.min_stability_frames:
            self.logger.warning(
                f"Insufficient stable frames: {stable_count}/{frames_total}"
            )
            return OTPExtractionResult(
                otp="",
                transitions=[],
                success=False,
                error=ExtractionError.INSUFFICIENT_FRAMES.value,
                frames_processed=frames_total,
                hand_detection_rate=hand_detection_rate
            )

        # Find transitions
        transitions = self.find_transitions(stable_counts, confidences)

        if not transitions:
            self.logger.warning("No gesture transitions detected")
            return OTPExtractionResult(
                otp="",
                transitions=[],
                success=False,
                error=ExtractionError.NO_TRANSITIONS.value,
                frames_processed=frames_total,
                hand_detection_rate=hand_detection_rate
            )

        # Build OTP from transition digits
        otp = ''.join(str(t.digit) for t in transitions)

        # Validate OTP length
        min_len = video_config.otp_min_length
        max_len = video_config.otp_max_length

        if len(otp) < min_len or len(otp) > max_len:
            self.logger.warning(
                f"Invalid OTP length: {len(otp)} (expected {min_len}-{max_len})"
            )
            return OTPExtractionResult(
                otp=otp,
                transitions=transitions,
                success=False,
                error=ExtractionError.INVALID_OTP_LENGTH.value,
                frames_processed=frames_total,
                hand_detection_rate=hand_detection_rate
            )

        # Validate OTP contains only allowed digits (1-5, configured in video_config)
        allowed = set(video_config.otp_allowed_digits)
        if not all(d in allowed for d in otp):
            self.logger.warning(f"OTP contains invalid digits: {otp}")
            return OTPExtractionResult(
                otp=otp,
                transitions=transitions,
                success=False,
                error="OTP contains invalid digits (only 1-5 allowed for gesture mode)",
                frames_processed=frames_total,
                hand_detection_rate=hand_detection_rate
            )

        self.logger.info(
            f"Successfully extracted OTP: {otp} from {len(transitions)} transitions"
        )

        return OTPExtractionResult(
            otp=otp,
            transitions=transitions,
            success=True,
            frames_processed=frames_total,
            hand_detection_rate=hand_detection_rate
        )

    def validate_gesture_quality(
        self,
        transitions: List[GestureTransition]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate gesture quality metrics.

        Args:
            transitions: List of detected transitions

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not transitions:
            return False, ExtractionError.NO_TRANSITIONS.value

        # Check transition timing
        for i, trans in enumerate(transitions):
            if trans.duration_seconds < 0.1:  # Too brief
                return False, f"Transition {i + 1} too brief ({trans.duration_seconds:.2f}s)"

            if trans.duration_seconds > self.max_transition_seconds:
                return False, f"Transition {i + 1} too long ({trans.duration_seconds:.2f}s)"

        # Check average confidence
        avg_confidence = np.mean([t.confidence for t in transitions])
        if avg_confidence < 0.5:
            return False, f"Low detection confidence: {avg_confidence:.2f}"

        return True, None


def extract_otp_from_gestures(
    finger_counts: List[int],
    confidences: List[float],
    fps: float = 2.0
) -> OTPExtractionResult:
    """
    Convenience function to extract OTP from gesture sequence.

    Args:
        finger_counts: List of finger counts per frame
        confidences: Detection confidence per frame
        fps: Video frame rate

    Returns:
        OTPExtractionResult with extracted OTP
    """
    extractor = GestureOTPExtractor(fps=fps)
    return extractor.extract_otp(finger_counts, confidences)

"""
Hand Gesture Detector - MediaPipe Hands integration for finger counting.

Detects hand gestures in video frames and counts extended fingers (0-5).
Uses geometric analysis of hand landmarks for robust finger detection.

MediaPipe 0.10.x uses the new task-based API with downloadable models.
"""

import os
import threading
import fcntl  # For file locking on Unix-like systems
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import tempfile
import urllib.request

import cv2

# Optional MediaPipe imports (may not be installed)
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
    # Import logger inside try block to ensure it's available
    from app.core.logger import get_logger
    logger = get_logger()
    logger.info("✅ MediaPipe imported successfully")
except ImportError as e:
    MEDIAPIPE_AVAILABLE = False
    mp = None
    python = None
    vision = None
    # Log after logger import
    from app.core.logger import get_logger
    logger = get_logger()
    logger.error(f"❌ MediaPipe import failed: {e}")
    logger.error("Gesture-based OTP will not be available")

from app.core.logger import get_logger
from app.config.video_config import video_config


class FingerIndex(Enum):
    """Index of each finger in hand landmarks."""
    THUMB = 0
    INDEX = 1
    MIDDLE = 2
    RING = 3
    PINKY = 4


# MediaPipe hand landmark indices (same for old and new API)
LANDMARKS = {
    'thumb_tip': 4,
    'thumb_ip': 3,
    'thumb_mcp': 2,
    'index_tip': 8,
    'index_pip': 6,
    'middle_tip': 12,
    'middle_pip': 10,
    'ring_tip': 16,
    'ring_pip': 14,
    'pinky_tip': 20,
    'pinky_pip': 18,
    'wrist': 0,
}


def _download_model_file() -> str:
    """
    Download the MediaPipe hand landmarker model file with file locking.

    Returns:
        Path to the downloaded model file
    """
    model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

    # Create cache directory
    cache_dir = os.path.join(tempfile.gettempdir(), "mediapipe_models")
    os.makedirs(cache_dir, exist_ok=True)

    model_path = os.path.join(cache_dir, "hand_landmarker.task")

    # Download if not exists (with file locking to prevent race conditions)
    if not os.path.exists(model_path):
        logger = get_logger()
        logger.info(f"Downloading MediaPipe hand landmarker model to {model_path}")
        # Create a lock file
        lock_path = f"{model_path}.lock"
        try:
            with open(lock_path, 'w') as lock_file:
                # Try to acquire exclusive lock (non-blocking)
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

                # Double-check after acquiring lock (another process may have downloaded)
                if not os.path.exists(model_path):
                    try:
                        urllib.request.urlretrieve(model_url, model_path)
                        logger.info("✅ Model download complete")
                    except Exception as e:
                        logger.error(f"❌ Failed to download model: {e}")
                        raise
                else:
                    logger.info(f"Model already downloaded by another process: {model_path}")
        finally:
            # Clean up lock file
            if os.path.exists(lock_path):
                os.remove(lock_path)

    return model_path


@dataclass
class GestureResult:
    """Result of gesture detection for a single frame."""

    finger_count: int  # 0-5
    confidence: float  # Detection confidence
    hand_detected: bool
    handedness: Optional[str] = None  # "Left" or "Right"
    landmarks: Optional[List[Tuple[float, float]]] = None  # Normalized (x, y)


class HandGestureDetector:
    """
    MediaPipe Hands-based gesture detector for counting fingers (0-5).

    Uses MediaPipe 0.10.x task-based API with HandLandmarker.

    Algorithm:
    1. Detect hand landmarks using MediaPipe HandLandmarker
    2. For each finger, check if extended based on geometric relationships
    3. Count total extended fingers (0-5)

    Landmark Analysis:
    - Thumb: Tip.x > IP.x (right hand) or Tip.x < IP.x (left hand)
    - Other fingers: Tip.y < PIP.y (tip above pip joint)
    """

    def __init__(self, min_detection_confidence: Optional[float] = None):
        """
        Initialize hand gesture detector.

        Args:
            min_detection_confidence: Minimum confidence for hand detection

        Raises:
            ImportError: If MediaPipe is not installed
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError(
                "MediaPipe is not installed. Install it with: pip install mediapipe"
            )

        self.logger = get_logger()
        self.min_confidence = min_detection_confidence or video_config.gesture_detection.min_hand_detection_confidence

        # Download model file if needed
        try:
            model_path = _download_model_file()
        except Exception as e:
            raise ImportError(f"Failed to download MediaPipe model: {e}")

        # Initialize MediaPipe HandLandmarker with task-based API
        base_options = python.BaseOptions(model_asset_path=model_path)

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,  # Only need one hand for 0-5
            min_hand_detection_confidence=float(self.min_confidence),
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.detector = vision.HandLandmarker.create_from_options(options)
        self.mp_image = mp.Image

        self.logger.info("HandGestureDetector initialized with MediaPipe HandLandmarker (0.10.x task API)")

    def _is_thumb_extended(
        self,
        landmarks: List,
        handedness: str
    ) -> bool:
        """
        Check if thumb is extended using geometric analysis.

        Thumb extension logic:
        - Right hand: thumb tip.x > thumb IP.x
        - Left hand: thumb tip.x < thumb IP.x

        Args:
            landmarks: MediaPipe hand landmarks
            handedness: "Left" or "Right"

        Returns:
            True if thumb is extended
        """
        thumb_tip = landmarks[LANDMARKS['thumb_tip']]
        thumb_ip = landmarks[LANDMARKS['thumb_ip']]

        if handedness == "Right":
            return thumb_tip.x > thumb_ip.x
        else:  # Left hand
            return thumb_tip.x < thumb_ip.x

    def _is_finger_extended(self, landmarks: List, finger: FingerIndex) -> bool:
        """
        Check if finger (index, middle, ring, pinky) is extended.

        Extension logic: Finger tip is above PIP joint (tip.y < pip.y)
        assuming hand is upright in image.

        Args:
            landmarks: MediaPipe hand landmarks
            finger: Finger to check

        Returns:
            True if finger is extended
        """
        if finger == FingerIndex.THUMB:
            # Use thumb-specific check
            return False

        finger_map = {
            FingerIndex.INDEX: ('index_tip', 'index_pip'),
            FingerIndex.MIDDLE: ('middle_tip', 'middle_pip'),
            FingerIndex.RING: ('ring_tip', 'ring_pip'),
            FingerIndex.PINKY: ('pinky_tip', 'pinky_pip'),
        }

        tip_name, pip_name = finger_map[finger]
        tip_idx = LANDMARKS[tip_name]
        pip_idx = LANDMARKS[pip_name]

        tip = landmarks[tip_idx]
        pip = landmarks[pip_idx]

        # Finger is extended if tip is above PIP (lower y value)
        return tip.y < pip.y

    def count_fingers(
        self,
        landmarks: List,
        handedness: str
    ) -> int:
        """
        Count extended fingers (0-5) from hand landmarks.

        Args:
            landmarks: MediaPipe hand landmarks (21 points)
            handedness: "Left" or "Right"

        Returns:
            Finger count 0-5
        """
        count = 0

        # Check thumb
        if self._is_thumb_extended(landmarks, handedness):
            count += 1

        # Check other fingers
        for finger in [FingerIndex.INDEX, FingerIndex.MIDDLE,
                       FingerIndex.RING, FingerIndex.PINKY]:
            if self._is_finger_extended(landmarks, finger):
                count += 1

        return count

    def detect_gesture(
        self,
        frame: np.ndarray
    ) -> GestureResult:
        """
        Detect hand gesture and count fingers in a frame.

        Args:
            frame: RGB image as numpy array (H, W, 3)

        Returns:
            GestureResult with finger count and confidence
        """
        try:
            # Convert to RGB with comprehensive format handling
            if len(frame.shape) == 2:
                # Grayscale - convert to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                self.logger.debug(f"Converted grayscale frame {frame.shape} to RGB")
            elif frame.shape[2] == 3:
                # BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.logger.debug(f"Converted BGR frame {frame.shape} to RGB")
            elif frame.shape[2] == 4:
                # BGRA to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                self.logger.debug(f"Converted BGRA frame {frame.shape} to RGB")
            else:
                raise ValueError(f"Unexpected frame shape: {frame.shape}")

            # Add diagnostic logging for frame quality
            self.logger.debug(
                f"Processing frame: shape={frame_rgb.shape}, "
                f"dtype={frame_rgb.dtype}, "
                f"range=[{frame_rgb.min()}, {frame_rgb.max()}]"
            )

            # Create MediaPipe Image
            mp_image = self.mp_image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Process with MediaPipe HandLandmarker
            detection_result = self.detector.detect(mp_image)

            # Check if hand detected with diagnostic logging
            if not detection_result.hand_landmarks:
                self.logger.warning(
                    f"No hand detected in frame. "
                    f"Frame shape: {frame_rgb.shape}, "
                    f"Detection result empty: {not detection_result.hand_landmarks}"
                )
                return GestureResult(
                    finger_count=0,
                    confidence=0.0,
                    hand_detected=False,
                    handedness=None,
                    landmarks=None
                )

            # Get first hand (only one expected for 0-5 counting)
            hand_landmarks = detection_result.hand_landmarks[0]
            handedness_list = detection_result.handedness

            # Get handedness from the first result
            handedness = handedness_list[0][0].category_name if handedness_list else "Unknown"
            # Extract actual detection confidence from handedness result
            handedness_entry = handedness_list[0][0] if handedness_list else None
            confidence = handedness_entry.score if handedness_entry else 0.9
            self.logger.debug(f"Hand detected: {handedness}, confidence: {confidence:.3f}")

            # Count fingers
            finger_count = self.count_fingers(
                hand_landmarks,
                handedness
            )

            # Extract landmarks as list of (x, y) tuples
            landmarks_list = [
                (lm.x, lm.y) for lm in hand_landmarks
            ]

            return GestureResult(
                finger_count=finger_count,
                confidence=confidence,
                hand_detected=True,
                handedness=handedness,
                landmarks=landmarks_list
            )

        except Exception as e:
            self.logger.error(f"Gesture detection error: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return GestureResult(
                finger_count=0,
                confidence=0.0,
                hand_detected=False,
                handedness=None,
                landmarks=None
            )

    def detect_gestures_batch(
        self,
        frames: List[np.ndarray]
    ) -> List[GestureResult]:
        """
        Detect gestures in multiple frames.

        Args:
            frames: List of RGB images as numpy arrays

        Returns:
            List of GestureResult (one per frame)
        """
        results = []
        for frame in frames:
            result = self.detect_gesture(frame)
            results.append(result)
        return results

    def close(self):
        """Close MediaPipe HandLandmarker resources."""
        if hasattr(self, 'detector') and self.detector:
            # Task-based API doesn't have explicit close, but __del__ handles cleanup
            self.detector = None
        self.logger.debug("HandGestureDetector resources closed")

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except Exception:
            pass


# Module-level singleton detector instance
_detector_singleton: Optional[HandGestureDetector] = None
_detector_lock = threading.Lock()


def get_detector(min_detection_confidence: Optional[float] = None) -> HandGestureDetector:
    """
    Get or create a singleton HandGestureDetector instance.

    This is more efficient than creating a new detector for each request.

    Args:
        min_detection_confidence: Minimum confidence for hand detection

    Returns:
        HandGestureDetector instance

    Raises:
        ImportError: If MediaPipe is not installed
    """
    global _detector_singleton

    if _detector_singleton is None:
        with _detector_lock:
            # Double-check pattern
            if _detector_singleton is None:
                _detector_singleton = HandGestureDetector(
                    min_detection_confidence=min_detection_confidence
                )
                logger = get_logger()
                logger.info("✅ HandGestureDetector singleton created")

    return _detector_singleton


def count_fingers_in_frames(
    frames: List[np.ndarray],
    detector: Optional[HandGestureDetector] = None
) -> List[int]:
    """
    Convenience function to count fingers in multiple frames.

    Args:
        frames: List of RGB images
        detector: Optional HandGestureDetector instance

    Returns:
        List of finger counts (0-5) per frame
    """
    own_detector = False
    if detector is None:
        detector = HandGestureDetector()
        own_detector = True

    try:
        results = detector.detect_gestures_batch(frames)
        finger_counts = [r.finger_count if r.hand_detected else -1 for r in results]
        return finger_counts
    finally:
        if own_detector:
            detector.close()

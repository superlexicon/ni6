"""
Video processing utilities for video selfie verification.

Provides functions for video validation, frame extraction, and metadata parsing.
Uses OpenCV for video processing operations.
"""

import os
import io
import tempfile
import hashlib
from typing import Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np
import ffmpeg

from app.core.logger import get_logger
from app.config.video_config import video_config


@dataclass
class VideoMetadata:
    """Metadata extracted from video file."""

    duration_seconds: float
    frame_count: int
    fps: float
    width: int
    height: int
    format: str
    size_bytes: int


@dataclass
class FrameResult:
    """Result of frame extraction."""

    frame: np.ndarray
    timestamp_seconds: float
    frame_index: int


class VideoValidationError(Exception):
    """Raised when video validation fails."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


def _extract_frame_with_ffmpeg(video_path: str, target_time: float) -> Optional[np.ndarray]:
    """
    Extract frame at exact timestamp using FFmpeg.

    FFmpeg provides frame-accurate seeking, unlike OpenCV's CAP_PROP_POS_MSEC
    which seeks to keyframes (I-frames). In videos with GOP intervals of 1-2 seconds,
    OpenCV seeking can miss the actual hand gesture moment.

    Args:
        video_path: Path to video file
        target_time: Target timestamp in seconds

    Returns:
        Frame as numpy array (BGR format), or None if extraction fails
    """
    logger = get_logger()
    try:
        out, err = (
            ffmpeg
            .input(video_path)
            .output('-', ss=target_time, vframes=1, format='image2pipe', pix_fmt='bgr24')
            .run(capture_stdout=True, capture_stderr=True, quiet=True)
        )
        frame = cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            logger.error(f"FFmpeg decoded empty frame at {target_time}s")
            return None
        return frame
    except ffmpeg.Error as e:
        stderr_msg = e.stderr.decode() if e.stderr else 'unknown error'
        logger.error(f"FFmpeg extraction failed at {target_time}s: {stderr_msg}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during FFmpeg extraction at {target_time}s: {str(e)}")
        return None


def validate_video_format(filename: str) -> str:
    """
    Validate video file format and return extension.

    Args:
        filename: Name of the video file

    Returns:
        Lowercase file extension without dot (e.g., 'mp4')

    Raises:
        VideoValidationError: If format is not supported
    """
    _, ext = os.path.splitext(filename.lower())
    format_ext = ext.lstrip('.')

    if format_ext not in video_config.supported_formats:
        supported = ', '.join(video_config.supported_formats).upper()
        raise VideoValidationError(
            f"Unsupported video format: {format_ext.upper()}. "
            f"Use {supported}"
        )

    return format_ext


def get_video_metadata(video_path: str) -> VideoMetadata:
    """
    Extract metadata from video file.

    Args:
        video_path: Path to video file

    Returns:
        VideoMetadata with duration, dimensions, etc.

    Raises:
        VideoValidationError: If video cannot be read
    """
    logger = get_logger()

    try:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise VideoValidationError("Cannot open video file")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate duration
        if fps > 0:
            duration = frame_count / fps
        else:
            duration = 0.0

        # Get file size
        size_bytes = os.path.getsize(video_path)

        cap.release()

        metadata = VideoMetadata(
            duration_seconds=duration,
            frame_count=frame_count,
            fps=fps,
            width=width,
            height=height,
            format=os.path.splitext(video_path)[1].lstrip('.'),
            size_bytes=size_bytes
        )

        logger.debug(
            f"Video metadata: {duration:.2f}s, {frame_count} frames, "
            f"{width}x{height}, {fps:.2f} fps"
        )

        return metadata

    except Exception as e:
        if isinstance(e, VideoValidationError):
            raise
        raise VideoValidationError(f"Failed to extract video metadata: {str(e)}")


def validate_video_constraints(metadata: VideoMetadata) -> None:
    """
    Validate video meets processing constraints.

    Args:
        metadata: Video metadata to validate

    Raises:
        VideoValidationError: If constraints are not met
    """
    # Check duration
    if metadata.duration_seconds > video_config.max_video_duration_seconds:
        raise VideoValidationError(
            f"Video too long: {metadata.duration_seconds:.1f}s. "
            f"Maximum {video_config.max_video_duration_seconds}s allowed"
        )

    # Check file size
    size_mb = metadata.size_bytes / (1024 * 1024)
    if size_mb > video_config.max_video_size_mb:
        raise VideoValidationError(
            f"Video too large: {size_mb:.1f}MB. "
            f"Maximum {video_config.max_video_size_mb}MB allowed"
        )

    # Check minimum frames
    if metadata.frame_count < video_config.min_frames_required:
        raise VideoValidationError(
            f"Video too short: {metadata.frame_count} frames. "
            f"Minimum {video_config.min_frames_required} frames required"
        )


def extract_frames(
    video_bytes: bytes,
    target_fps: Optional[int] = None
) -> Tuple[List[FrameResult], VideoMetadata]:
    """
    Extract frames from video at guided digit timestamps.

    The Flutter app guides the user through recording:
    - 0-1s: Warmup
    - 1-2s: "Read" overlay
    - 2-3s: "Steady" overlay
    - 3-4s: "Go" overlay
    - 4-6s: First digit (sample at 5s)
    - 6-8s: Second digit (sample at 7s)
    - etc.

    We extract frames at the middle of each digit display window.

    Args:
        video_bytes: Video file as bytes
        target_fps: Ignored (kept for compatibility, uses guided timestamps instead)

    Returns:
        Tuple of (list of FrameResult, VideoMetadata)

    Raises:
        VideoValidationError: If extraction fails
    """
    logger = get_logger()

    # Create temporary file for OpenCV in /tmp
    os.makedirs('/tmp', exist_ok=True)
    tmp_path = os.path.join('/tmp', f'temp_video_{os.getpid()}_{datetime.now().timestamp()}.mp4')
    with open(tmp_path, 'wb') as tmp:
        tmp.write(video_bytes)

    try:
        # Get metadata first
        metadata = get_video_metadata(tmp_path)

        # Validate duration for guided recording
        min_duration = video_config.frame_sampling.min_video_duration_seconds
        max_duration = video_config.frame_sampling.max_video_duration_seconds
        if metadata.duration_seconds < min_duration:
            raise VideoValidationError(
                f"Video too short: {metadata.duration_seconds:.2f}s. "
                f"Minimum {min_duration}s required for guided digit recording"
            )
        if metadata.duration_seconds > max_duration:
            raise VideoValidationError(
                f"Video too long: {metadata.duration_seconds:.2f}s. "
                f"Maximum {max_duration}s allowed"
            )

        # Validate file size
        size_mb = metadata.size_bytes / (1024 * 1024)
        if size_mb > video_config.max_video_size_mb:
            raise VideoValidationError(
                f"Video too large: {size_mb:.1f}MB. "
                f"Maximum {video_config.max_video_size_mb}MB allowed"
            )

        # Extract frames at guided timestamps (middle of each digit display window)
        # Use FFmpeg for frame-accurate extraction (OpenCV seeking is not frame-accurate)
        sample_times = video_config.frame_sampling.digit_sample_times
        frames: List[FrameResult] = []

        for digit_idx, target_time in enumerate(sample_times):
            # Skip if target time is beyond video duration
            if target_time >= metadata.duration_seconds:
                logger.warning(f"Target time {target_time}s is beyond video duration {metadata.duration_seconds}s")
                break

            # Use FFmpeg for frame-accurate extraction
            frame = _extract_frame_with_ffmpeg(tmp_path, target_time)

            if frame is None:
                logger.error(f"Failed to extract frame at {target_time}s using FFmpeg")
                continue

            # Verify frame is valid (not empty, correct dimensions)
            if frame.size == 0:
                logger.error(f"Extracted empty frame at {target_time}s")
                continue

            frames.append(FrameResult(
                frame=frame,
                timestamp_seconds=target_time,
                frame_index=digit_idx
            ))

            # Log frame hash for debugging (helps identify duplicate frames)
            frame_hash = hashlib.md5(frame.tobytes()).hexdigest()[:8]
            logger.debug(f"Extracted frame for digit {digit_idx + 1} at {target_time}s (hash: {frame_hash})")

            # Save frame for diagnostic purposes (to debug hand detection issues)
            debug_dir = "/tmp/video_frames_debug"
            os.makedirs(debug_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_path = os.path.join(debug_dir, f"frame_digit{digit_idx + 1}_t{target_time}s_{timestamp}.jpg")
            cv2.imwrite(debug_path, frame)
            logger.debug(f"Saved debug frame to {debug_path}")

        min_required = video_config.otp_min_length
        if len(frames) < min_required:
            raise VideoValidationError(
                f"Insufficient frames extracted: {len(frames)}. "
                f"Need at least {min_required} frames for {min_required}-digit OTP"
            )

        logger.info(
            f"Extracted {len(frames)} frames from {metadata.duration_seconds:.2f}s video "
            f"at guided timestamps: {sample_times[:len(frames)]}"
        )

        return frames, metadata

    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def decode_frame_from_bytes(frame_data: bytes) -> np.ndarray:
    """
    Decode frame from bytes.

    Args:
        frame_data: Frame encoded as bytes (JPEG/PNG)

    Returns:
        numpy array of shape (height, width, channels)
    """
    nparr = np.frombuffer(frame_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return frame


def encode_frame_to_bytes(frame: np.ndarray, format: str = '.jpg') -> bytes:
    """
    Encode frame to bytes.

    Args:
        frame: numpy array of shape (height, width, channels)
        format: Output format (e.g., '.jpg', '.png')

    Returns:
        Encoded frame as bytes
    """
    success, encoded = cv2.imencode(format, frame)
    if not success:
        raise ValueError("Failed to encode frame")
    return encoded.tobytes()


def frame_to_rgb(frame: np.ndarray) -> np.ndarray:
    """
    Convert BGR frame (OpenCV default) to RGB.

    Args:
        frame: BGR image as numpy array

    Returns:
        RGB image as numpy array
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

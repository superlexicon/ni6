"""
Research-backed thresholds for various face recognition models.
Based on academic benchmarks and LFW (Labeled Faces in the Wild) dataset results.
Compatible with DeepFace v0.0.93 and similar versions.

Enhanced with confidence-weighted adaptive thresholds for improved accuracy.
"""

from typing import Dict, Optional
import numpy as np

# Research-backed thresholds based on academic benchmarks and LFW dataset
MODEL_THRESHOLDS: Dict[str, float] = {
    "VGG-Face": 0.31,     # Research-backed threshold for VGG-Face
    "Facenet": 0.40,      # Research-backed threshold for Facenet
    "Facenet512": 0.30,   # Research-backed threshold for Facenet512
    "OpenFace": 0.40,     # Research-backed threshold for OpenFace
    "DeepFace": 0.23,     # Research-backed threshold for DeepFace
    "ArcFace": 0.68,      # Research-backed threshold for ArcFace
    "Dlib": 0.07          # Research-backed threshold for Dlib
}

# Default fallback threshold for unknown models
DEFAULT_THRESHOLD = 0.4


def get_model_threshold(model_name: str) -> float:
    """
    Get the research-backed threshold for a specific face recognition model.

    Args:
        model_name: Name of the face recognition model

    Returns:
        float: Threshold value for the model, or default if model not found
    """
    return MODEL_THRESHOLDS.get(model_name, DEFAULT_THRESHOLD)


def get_available_models() -> Dict[str, float]:
    """
    Get all available models and their thresholds.

    Returns:
        Dict[str, float]: Dictionary mapping model names to their thresholds
    """
    return MODEL_THRESHOLDS.copy()


def calculate_adaptive_threshold(
    base_threshold: float,
    face_quality: Optional[float] = None,
    detection_confidence: Optional[float] = None,
    image_quality: Optional[Dict[str, float]] = None
) -> float:
    """
    Calculate adaptive threshold based on face and image quality metrics.

    This implements confidence-weighted thresholds that adjust dynamically
    based on detection confidence and image quality, improving accuracy in
    challenging conditions while maintaining processing speed.

    Args:
        base_threshold: Base threshold for the model (from research-backed values)
        face_quality: Face detection confidence score (0.0-1.0)
        detection_confidence: Face detection confidence from detector (0.0-1.0)
        image_quality: Dictionary with quality metrics {
            'sharpness': 0.0-1.0,
            'brightness': 0.0-1.0,
            'contrast': 0.0-1.0,
            'resolution': 0.0-1.0
        }

    Returns:
        float: Adaptive threshold adjusted for quality factors
    """
    # Initialize multipliers
    quality_multiplier = 1.0
    confidence_multiplier = 1.0

    # Calculate quality multiplier based on image quality metrics
    if image_quality:
        # Sharpness is the most important factor for face recognition
        sharpness = image_quality.get('sharpness', 0.5)
        quality_multiplier *= (0.8 + sharpness * 0.4)  # 0.8 to 1.2 range

        # Brightness - penalize very dark or very bright images
        brightness = image_quality.get('brightness', 0.5)
        brightness_score = 1.0 - abs(brightness - 0.5) * 2  # Peak at 0.5
        quality_multiplier *= (0.9 + brightness_score * 0.2)  # 0.9 to 1.1 range

        # Contrast - higher contrast is generally better for face recognition
        contrast = image_quality.get('contrast', 0.5)
        quality_multiplier *= (0.85 + contrast * 0.3)  # 0.85 to 1.15 range

        # Resolution - ensure adequate resolution
        resolution = image_quality.get('resolution', 0.5)
        quality_multiplier *= (0.9 + resolution * 0.2)  # 0.9 to 1.1 range

    # Calculate confidence multiplier based on face detection
    if face_quality is not None:
        # Higher face quality confidence allows for slightly stricter threshold
        confidence_multiplier *= (0.95 + face_quality * 0.1)  # 0.95 to 1.05 range

    if detection_confidence is not None:
        # Higher detection confidence allows for slightly stricter threshold
        confidence_multiplier *= (0.95 + detection_confidence * 0.1)  # 0.95 to 1.05 range

    # Apply adaptive threshold
    adaptive_threshold = base_threshold * quality_multiplier * confidence_multiplier

    # Clamp to reasonable bounds to prevent extreme values
    adaptive_threshold = max(0.1, min(0.9, adaptive_threshold))  # Keep between 0.1 and 0.9

    return adaptive_threshold


def get_adaptive_model_threshold(
    model_name: str,
    face_quality: Optional[float] = None,
    detection_confidence: Optional[float] = None,
    image_quality: Optional[Dict[str, float]] = None
) -> float:
    """
    Get adaptive threshold for a specific model with quality-based adjustment.

    Args:
        model_name: Name of the face recognition model
        face_quality: Face detection confidence score (0.0-1.0)
        detection_confidence: Face detection confidence from detector (0.0-1.0)
        image_quality: Dictionary with quality metrics

    Returns:
        float: Adaptive threshold for the model with quality adjustments
    """
    base_threshold = get_model_threshold(model_name)
    return calculate_adaptive_threshold(
        base_threshold,
        face_quality,
        detection_confidence,
        image_quality
    )


def get_model_threshold_with_confidence(
    model_name: str,
    confidence_score: float,
    image_quality: Optional[Dict[str, float]] = None
) -> float:
    """
    Convenience function for getting threshold with confidence score.
    Combines face quality and detection confidence into a single metric.

    Args:
        model_name: Name of the face recognition model
        confidence_score: Combined confidence score (0.0-1.0)
        image_quality: Dictionary with quality metrics

    Returns:
        float: Adaptive threshold adjusted for confidence
    """
    base_threshold = get_model_threshold(model_name)
    return calculate_adaptive_threshold(
        base_threshold,
        face_quality=confidence_score,
        detection_confidence=confidence_score,
        image_quality=image_quality
    )
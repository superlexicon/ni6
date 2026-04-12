"""Factory for face recognition backend selection based on environment variable."""

import os
from typing import Optional
from app.core.logger import get_logger
from app.helper.face_recognition_base import FaceRecognitionBase

_face_recognition_backend: Optional[FaceRecognitionBase] = None
_backend_name: Optional[str] = None


def get_face_recognition_backend() -> FaceRecognitionBase:
    """
    Get face recognition backend based on environment variable.

    Environment variable: FACE_RECOGNITION_BACKEND
    Options: 'deepface' (default) or 'insightface'

    Returns:
        FaceRecognitionBase instance configured with the selected backend
    """
    global _face_recognition_backend, _backend_name

    if _face_recognition_backend is None:
        backend = os.getenv('FACE_RECOGNITION_BACKEND', 'deepface').lower()
        logger = get_logger()

        if backend == 'insightface':
            from app.helper.insightface_helper import InsightFaceHelper
            _face_recognition_backend = InsightFaceHelper()
            _backend_name = 'insightface'
            logger.info("Using InsightFace backend for face recognition")
        else:
            from app.helper.deepface_helper import DeepfaceHelper
            _face_recognition_backend = DeepfaceHelper()
            _backend_name = 'deepface'
            logger.info("Using DeepFace backend for face recognition")

    return _face_recognition_backend


def get_backend_name() -> str:
    """Get the name of the currently configured backend."""
    global _backend_name

    if _backend_name is None:
        # Trigger backend initialization
        get_face_recognition_backend()

    return _backend_name


def get_model_name() -> str:
    """
    Get the model name for database storage.

    Returns the full model identifier that should be stored in the
    face_biometrics table to track which model was used for extraction.

    Returns:
        Model name (e.g., 'deepface_vgg-face', 'insightface_buffalo_l')
    """
    backend = get_face_recognition_backend()
    return backend.get_model_name()


def reset_backend():
    """Reset the backend instance (useful for testing or reconfiguration)."""
    global _face_recognition_backend, _backend_name
    _face_recognition_backend = None
    _backend_name = None

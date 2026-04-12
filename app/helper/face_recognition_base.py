"""Abstract base class for face recognition backends."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np


class FaceRecognitionBase(ABC):
    """Abstract base class for face recognition backends (DeepFace, InsightFace, etc.)."""

    @abstractmethod
    def detect_faces(self, img: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in image.

        Args:
            img: Input image as numpy array (BGR or RGB format)

        Returns:
            List of face detections with bbox, landmarks, confidence, and optional embedding
        """
        pass

    @abstractmethod
    def get_embedding(self, img: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding from image.

        Args:
            img: Input image containing exactly one face

        Returns:
            Embedding vector as numpy array, or None if no face detected
        """
        pass

    @abstractmethod
    def verify_faces(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        threshold: float = 0.15
    ) -> Dict[str, Any]:
        """
        Verify if two face embeddings match.

        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            threshold: Distance threshold for verification

        Returns:
            Dictionary with 'verified', 'distance', 'similarity', 'threshold' keys
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Return the model name for storage/tracking purposes.

        Returns:
            Model identifier string (e.g., 'deepface_facenet512', 'insightface_buffalo_l')
        """
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        Return the embedding dimension.

        Returns:
            Dimension of embedding vectors (e.g., 512)
        """
        pass

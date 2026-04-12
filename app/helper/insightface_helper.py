"""InsightFace wrapper for GPU-accelerated face recognition operations."""

import numpy as np
from typing import List, Dict, Any, Optional
from scipy.spatial.distance import cosine

from app.core.logger import get_logger
from app.core.device_manager import DeviceType, get_recommended_device
from app.helper.face_recognition_base import FaceRecognitionBase


class InsightFaceHelper(FaceRecognitionBase):
    """
    InsightFace wrapper using PyTorch/ONNX Runtime.

    Supports GPU acceleration via CUDA/ROCm and works with AMD GPUs.
    Uses the Buffalo_L model for state-of-the-art face recognition.
    """

    def __init__(self, model_name: str = 'buffalo_l'):
        """Initialize InsightFace with GPU support."""
        self.logger = get_logger()
        self.model_name = model_name
        self.device_info = get_recommended_device()

        # Determine execution providers based on device type
        if self.device_info.device_type == DeviceType.ROCM:
            # ROCm uses CUDA interface in PyTorch/ONNX
            self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        elif self.device_info.device_type == DeviceType.CUDA:
            self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            self.providers = ['CPUExecutionProvider']

        self._initialize_model()

    def _initialize_model(self):
        """Initialize InsightFace model with appropriate execution providers."""
        try:
            from insightface.app import FaceAnalysis

            self.face_app = FaceAnalysis(
                name=self.model_name,
                providers=self.providers,
                allowed_modules=['detection', 'recognition']
            )

            # Determine context ID (0 for GPU, -1 for CPU)
            if self.device_info.device_type in [DeviceType.CUDA, DeviceType.ROCM]:
                ctx_id = 0
            else:
                ctx_id = -1

            self.face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))

            self.logger.info(
                f"InsightFace {self.model_name} initialized with "
                f"providers={self.providers}, ctx_id={ctx_id}"
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize InsightFace: {e}")
            raise

    def detect_faces(self, img: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in image.

        Args:
            img: Input image as numpy array (RGB or BGR format)

        Returns:
            List of face detections with bbox, landmarks, confidence, and embedding
        """
        try:
            faces = self.face_app.get(img)
            results = []

            for face in faces:
                results.append({
                    'bbox': self._convert_bbox(face.bbox),
                    'landmarks': face.landmark.tolist() if hasattr(face, 'landmark') else [],
                    'confidence': float(face.det_score),
                    'embedding': face.embedding.tolist() if hasattr(face, 'embedding') else None
                })

            return results

        except Exception as e:
            self.logger.error(f"InsightFace face detection failed: {e}")
            return []

    def get_embedding(self, img: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding from image.

        Args:
            img: Input image as numpy array

        Returns:
            512-dimensional embedding vector, or None if no face detected
        """
        try:
            faces = self.face_app.get(img)
            if faces:
                return faces[0].embedding
            return None

        except Exception as e:
            self.logger.error(f"InsightFace embedding extraction failed: {e}")
            return None

    def verify_faces(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        threshold: float = 0.25
    ) -> Dict[str, Any]:
        """
        Verify if two face embeddings match using cosine similarity.

        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            threshold: Cosine distance threshold (default 0.25 for Buffalo_L)

        Returns:
            Dictionary with verification results
        """
        try:
            # Compute cosine distance (0 = identical, 1 = completely different)
            distance = cosine(embedding1, embedding2)
            similarity = 1 - distance

            return {
                'verified': distance <= threshold,
                'distance': float(distance),
                'similarity': float(similarity),
                'threshold': threshold
            }

        except Exception as e:
            self.logger.error(f"InsightFace verification failed: {e}")
            return {
                'verified': False,
                'distance': 1.0,
                'similarity': 0.0,
                'threshold': threshold,
                'error': str(e)
            }

    def get_model_name(self) -> str:
        """Return model name for database storage."""
        return f"insightface_{self.model_name}"

    def get_embedding_dimension(self) -> int:
        """Return embedding dimension (512 for Buffalo_L)."""
        return 512

    def _convert_bbox(self, bbox: np.ndarray) -> Dict[str, int]:
        """Convert InsightFace bbox format to standard format."""
        return {
            'x': int(bbox[0]),
            'y': int(bbox[1]),
            'w': int(bbox[2] - bbox[0]),
            'h': int(bbox[3] - bbox[1])
        }

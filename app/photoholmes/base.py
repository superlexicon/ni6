"""
Base classes for PhotoHolmes methods adapted for IM-OSINT integration.

This module provides the foundational classes that all PhotoHolmes forgery detection
methods inherit from, adapted to work with the IM-OSINT GPU resource manager.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import numpy as np
from PIL import Image
from torch import nn

from app.core.gpu_manager import ModelType, get_gpu_manager

logger = logging.getLogger(__name__)


class BasePhotoHolmesMethod(ABC):
    """
    Base class for all PhotoHolmes forgery detection methods.

    This class provides common functionality for forgery detection methods,
    including GPU resource management, device handling, and common preprocessing.
    """

    def __init__(self, device: Optional[str] = None, model_type: Optional[ModelType] = None):
        """
        Initialize the base PhotoHolmes method.

        Args:
            device: Device to run the model on ('cuda', 'cpu', etc.)
            model_type: Model type for GPU resource manager
        """
        self.device = device or self._get_device()
        self.model_type = model_type
        self.gpu_manager = get_gpu_manager() if model_type else None
        self.model = None
        self.is_initialized = False

    def _get_device(self) -> str:
        """Get the best available device for computation."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def load_model(self) -> None:
        """Load the model weights and architecture. Override in subclasses that need model loading."""
        self.is_initialized = True

    def preprocess(self, image: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Preprocess the input image. Override in subclasses that need custom preprocessing.

        Args:
            image: Input image (path, numpy array, or PIL Image)

        Returns:
            Preprocessed tensor
        """
        return self.to_tensor(self.load_image(image))

    def predict(self, image: Union[str, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """
        Perform forgery detection on the input image.

        Args:
            image: Input image (path, numpy array, or PIL Image)

        Returns:
            Dictionary containing detection results
        """
        raise NotImplementedError("Subclasses must implement predict()")

    async def load_model_async(self) -> None:
        """Load the model asynchronously with GPU resource management."""
        if self.model_type and self.gpu_manager:
            # For methods that need GPU resource management
            self.model = await self.gpu_manager.get_model_with_gpu(
                model_type=self.model_type,
                model_class=self._create_model,
                device=self.device
            )
        else:
            # For methods that don't need GPU management
            self.load_model()

        self.is_initialized = True
        logger.info(f"Model loaded successfully on {self.device}")

    def _create_model(self):
        """Create the model - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _create_model")

    def load_image(self, image: Union[str, np.ndarray, Image.Image]) -> Image.Image:
        """
        Load an image from various input formats.

        Args:
            image: Input image (path, numpy array, or PIL Image)

        Returns:
            PIL Image
        """
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            return image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def to_tensor(self, image: Image.Image) -> torch.Tensor:
        """
        Convert PIL Image to tensor.

        Args:
            image: PIL Image

        Returns:
            Normalized tensor
        """
        # Convert to tensor and normalize to [0, 1]
        tensor = torch.from_numpy(np.array(image)).float() / 255.0

        # Transpose to [C, H, W] format if needed
        if tensor.dim() == 3:
            tensor = tensor.permute(2, 0, 1)

        # Add batch dimension
        tensor = tensor.unsqueeze(0)

        return tensor.to(self.device)

    async def predict_async(self, image: Union[str, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """
        Perform forgery detection asynchronously.

        Args:
            image: Input image (path, numpy array, or PIL Image)

        Returns:
            Dictionary containing detection results
        """
        if not self.is_initialized:
            await self.load_model_async()

        return self.predict(image)

    def release_resources(self) -> None:
        """Release model resources."""
        if self.model_type and self.gpu_manager:
            # Release through GPU manager
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                loop.run_until_complete(self.gpu_manager.release_model(self.model_type))
            except RuntimeError:
                asyncio.run(self.gpu_manager.release_model(self.model_type))

        self.model = None
        self.is_initialized = False
        logger.info("Resources released")


class BenchmarkOutput:
    """Output structure for benchmark results."""

    def __init__(
        self,
        prediction: float,
        segmentation: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.prediction = prediction  # Forgery probability
        self.segmentation = segmentation  # Segmentation mask
        self.metadata = metadata or {}  # Additional metadata


class BaseTorchMethod(BasePhotoHolmesMethod, nn.Module):
    """
    Base class for PyTorch-based PhotoHolmes methods.

    This extends the base class with PyTorch-specific functionality.
    """

    def __init__(self, **kwargs):
        BasePhotoHolmesMethod.__init__(self, **kwargs)
        nn.Module.__init__(self)

        # Check if this class itself is a PyTorch model (self-referencing)
        is_self_referencing = hasattr(self, 'forward') and hasattr(self, 'parameters')

        if self.model is not None:
            self.model.eval()  # Set model to evaluation mode
        elif is_self_referencing:
            # This class itself is a PyTorch model
            logger.info(f"{self.__class__.__name__} detected as self-referencing PyTorch model")
            # Set eval mode on self directly
            super(BaseTorchMethod, self).eval()
        else:
            logger.warning(f"Model is None for {self.__class__.__name__} - skipping eval() call")

    def load_weights(self, weights: Union[str, Path, dict]) -> None:
        """Load weights from a dictionary or a file when given its path."""
        if isinstance(weights, (str, Path)):
            weights_ = torch.load(weights, map_location=self.device, weights_only=False)
        else:
            weights_ = weights

        if "state_dict" in weights_.keys():
            weights_ = weights_["state_dict"]

        self.load_state_dict(weights_, assign=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the model.

        Args:
            x: Input tensor

        Returns:
            Tuple of (prediction, segmentation)
        """
        with torch.no_grad():
            if hasattr(self.model, 'forward'):
                return self.model(x)
            else:
                # Handle models that don't have standard forward method
                prediction = self.model(x)
                return prediction, None
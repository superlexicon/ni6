import os
import logging
from typing import Optional

import torch
import torch.backends.mps

from app.core.gpu_manager import ModelType, get_gpu_manager
from app.photoholmes.methods.psccnet import PSCCNet
from app.photoholmes.methods.psccnet.config import PSCCNetWeights

logger = logging.getLogger(__name__)

class PSCCNetModel:
    """
    PSCCNet model wrapper integrated with GPU resource manager.

    This class manages the PhotoHolmes PSCCNet model with centralized
    GPU resource allocation and memory management.
    """

    _instance: Optional['PSCCNetModel'] = None
    _initialized: bool = False

    ARCH_CONFIG = "pretrained"
    # Use integrated photoholmes weights paths
    FENET_WEIGHTS = "app/photoholmes/weights/psccnet/FENet.pth"
    SEGNET_WEIGHTS = "app/photoholmes/weights/psccnet/SegNet.pth"
    CLSNET_WEIGHTS = "app/photoholmes/weights/psccnet/ClsNet.pth"

    logger.info(f"PSCCNet weight paths:")
    logger.info(f"  FENet: {FENET_WEIGHTS}")
    logger.info(f"  SegNet: {SEGNET_WEIGHTS}")
    logger.info(f"  ClsNet: {CLSNET_WEIGHTS}")

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PSCCNetModel, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.gpu_manager = get_gpu_manager()
            self.weights = self._get_weights()
            self._initialized = True
            logger.info("PSCCNetModel initialized with GPU resource manager")

    def _get_weights(self) -> PSCCNetWeights:
        """Get PhotoHolmes PSCCNet weights configuration."""
        return PSCCNetWeights(
            FENet=self.FENET_WEIGHTS,
            SegNet=self.SEGNET_WEIGHTS,
            ClsNet=self.CLSNET_WEIGHTS,
        )

    async def get_model(self) -> PSCCNet:
        """
        Get PSCCNet model with GPU resource management.

        Returns:
            PSCCNet model instance with GPU resources allocated
        """
        try:
            model = await self.gpu_manager.get_model_with_gpu(
                model_type=ModelType.PHOTOHOLMES_PSccNET,
                model_class=self._create_model,
                arch_config=self.ARCH_CONFIG,
                weights=self.weights
            )
            logger.debug("PSCCNet model retrieved from GPU manager")
            return model
        except Exception as e:
            logger.error(f"Failed to get PSCCNet model: {e}")
            raise RuntimeError(f"PSCCNet model initialization failed: {str(e)}")

    def _create_model(self, arch_config: str, weights: PSCCNetWeights) -> PSCCNet:
        """
        Create PSCCNet model instance.

        Args:
            arch_config: Architecture configuration
            weights: Pre-trained weights

        Returns:
            PSCCNet model instance
        """
        model = PSCCNet(
            arch_config=arch_config,
            weights=weights,
            device="cpu"  # Device will be set by GPU manager
        )
        model.eval()
        return model

    async def release_model(self) -> bool:
        """
        Release PSCCNet model resources.

        Returns:
            True if model was released successfully
        """
        try:
            released = await self.gpu_manager.release_model(ModelType.PHOTOHOLMES_PSccNET)
            if released:
                logger.info("PSCCNet model resources released")
            return released
        except Exception as e:
            logger.error(f"Failed to release PSCCNet model: {e}")
            return False

    def get_device(self) -> str:
        """Get current device information."""
        # Use the same device logic as the GPU manager to ensure consistency
        device_info = self.gpu_manager.device_info
        if device_info.device_type.value == "cuda" and torch.cuda.is_available():
            if device_info.device_id is not None:
                return f"cuda:{device_info.device_id}"
            return "cuda"
        elif device_info.device_type.value == "mps" and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"


# Legacy function wrappers for backward compatibility
async def psccnet_init_model() -> None:
    """Initialize PSCCNet model (async wrapper for backward compatibility)."""
    # Initialization is now handled lazily by get_model()
    pass

async def psccnet_get_model() -> PSCCNet:
    """Get PSCCNet model with GPU resource management."""
    return await PSCCNetModel().get_model()

async def psccnet_get_device() -> str:
    """Get PSCCNet device information."""
    return PSCCNetModel().get_device()

# Synchronous wrappers for backward compatibility
def psccnet_get_model_sync() -> PSCCNet:
    """Synchronous wrapper for get_model - creates PSCCNet model directly without async."""
    try:
        # Create model directly without async GPU manager to avoid event loop issues
        from app.photoholmes.methods.psccnet.config import PSCCNetWeights

        # Initialize weights with actual paths
        weights_dir = os.path.join(os.path.dirname(__file__), '..', 'photoholmes', 'weights', 'psccnet')
        weights: PSCCNetWeights = {
            "FENet": os.path.join(weights_dir, "FENet.pth"),
            "SegNet": os.path.join(weights_dir, "SegNet.pth"),
            "ClsNet": os.path.join(weights_dir, "ClsNet.pth"),
        }

        # Create model directly on the best available device
        if torch.cuda.is_available():
            device = "cuda:0"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        # Create and initialize model
        model = PSCCNet(
            arch_config="pretrained",  # Use pretrained architecture config
            weights=weights,
            device=device
        )

        # Move model to device and set to evaluation mode
        model.to(device)
        model.eval()

        logger.info(f"PSCCNet model created successfully on {device}")
        return model

    except Exception as e:
        logger.error(f"Failed to create PSCCNet model: {e}")
        raise RuntimeError(f"PSCCNet model initialization failed: {str(e)}")

def psccnet_get_device_sync() -> str:
    """Synchronous wrapper for get_device."""
    return PSCCNetModel().get_device()

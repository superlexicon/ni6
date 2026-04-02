import io
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from app.photoholmes.methods.psccnet import psccnet_preprocessing

from app.dto import PsccnetMethodData
from app.core.logger import get_logger
from app.core import psccnet_get_model_sync, psccnet_get_device_sync
from app.utils.autocast_utils import DeviceAwareAutocast


class PsccnetMethod:
    MAX_IMAGE_SIZE: int = 512
    SUPPORTED_CONTENT_TYPE: str = 'image/'
    RGB_MODE: str = 'RGB'

    def __init__(self):
        self.logger = get_logger()

        # Initialize model first to ensure device is determined
        try:
            self.logger.info("Loading PSCCNet model...")
            self.method = psccnet_get_model_sync()
            self.logger.info(f"PSCCNet model loaded successfully. Type: {type(self.method)}")

            # Get device from the loaded model's parameters to ensure consistency
            try:
                # Get device from actual model parameters, not device attribute
                param_device = next(self.method.parameters()).device
                self.device = str(param_device)
                self.logger.info(f"PSCCNet model parameter device: {self.device}")
            except Exception as e:
                # Fallback to device detection if can't get parameter device
                self.device = psccnet_get_device_sync()
                self.logger.info(f"Using fallback device due to error {e}: {self.device}")

            self.logger.info(f"Initializing PSCCNET Method on device: {self.device}")
            self.logger.info(f"Device type: {type(self.device)}")

        except Exception as e:
            self.logger.error(f"Failed to load PSCCNet model: {type(e).__name__}")
            raise

        self.logger.info("PSCCNET Method initialized")

    def _preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                if image.mode != self.RGB_MODE:
                    image = image.convert(self.RGB_MODE)

                image = image.resize(
                    (self.MAX_IMAGE_SIZE, self.MAX_IMAGE_SIZE),
                    Image.Resampling.LANCZOS
                )
                image_np = np.array(image)

                if image_np.size == 0 or np.any(np.isnan(image_np)) or np.any(np.isinf(image_np)):
                    raise ValueError(
                        "Invalid image data: contains invalid values")

                return image_np

        except Exception as e:
            self.logger.error(f"Image preprocessing failed: {str(e)}")
            raise ValueError(f"Failed to preprocess image: {str(e)}")

    def _prepare_model_input(self, image_np: np.ndarray) -> Dict[str, Tensor]:
        self.logger.info(f"PSCCNet _prepare_model_input called, target device: {self.device}")
        image_data = {"image": image_np}
        input_data = psccnet_preprocessing(**image_data)

        # Debug: Check what we're getting from preprocessing
        self.logger.info(f"Preprocessing returned: {type(input_data)}, keys: {list(input_data.keys()) if isinstance(input_data, dict) else 'Not a dict'}")

        result = {}
        for k, v in input_data.items():
            # Debug: Check each value before processing
            self.logger.info(f"Processing key {k}: type={type(v)}, shape={getattr(v, 'shape', 'No shape')}")

            # Check if v is a coroutine (async result)
            import asyncio
            if asyncio.iscoroutine(v):
                self.logger.error(f"Found coroutine for key {k}, this shouldn't happen!")
                raise ValueError(f"Expected tensor for key {k}, but got coroutine")

            tensor = torch.from_numpy(v)
            self.logger.info(f"After torch.from_numpy: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")

            self.logger.info(f"Calling .to({self.device}) on tensor...")
            tensor = tensor.permute(2, 0, 1).unsqueeze(0).float().to(self.device)
            self.logger.info(f"After .to(self.device): shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
            result[k] = tensor

        self.logger.info(f"Final input tensors devices: {[(k, v.device) for k, v in result.items()]}")
        return result

    def _predict(self, input_data: Dict[str, Tensor]) -> float:
        self.logger.info(f"PSCCNet _predict called with device: {self.device}")

        # Check input tensor devices
        for k, v in input_data.items():
            self.logger.info(f"  Input tensor {k}: shape={v.shape}, device={v.device}")

        # Check if method is a coroutine
        import asyncio
        if asyncio.iscoroutine(self.method):
            self.logger.error("ERROR: self.method is a coroutine! This shouldn't happen!")
            raise ValueError("self.method is a coroutine, not a model instance")

        # Check if method is None or not callable
        if self.method is None:
            self.logger.error("ERROR: self.method is None!")
            raise ValueError("self.method is None - not properly initialized")

        if not hasattr(self.method, 'predict'):
            self.logger.error(f"ERROR: self.method has no predict method! Type: {type(self.method)}")
            raise ValueError(f"self.method has no predict method - type: {type(self.method)}")

        # Check model device
        if hasattr(self.method, 'device'):
            self.logger.info(f"  Model device: {self.method.device}")
        if hasattr(self.method, 'FENet') and hasattr(self.method.FENet, 'device'):
            self.logger.info(f"  FENet device: {self.method.FENet.device}")

        self.logger.info(f"Calling PSCCNet predict method...")
        with torch.no_grad():
            with DeviceAwareAutocast(self.device, torch.float16, self.logger):
                try:
                    output = self.method.predict(**input_data)
                    self.logger.info(f"PSCCNet predict output: type={type(output)}, len={len(output) if hasattr(output, '__len__') else 'No len'}")

                    if isinstance(output, (tuple, list)) and len(output) >= 2:
                        return output[1].item()
                    else:
                        self.logger.error(f"Unexpected output format: {type(output)}, output: {output}")
                        raise ValueError(f"Expected tuple/list with at least 2 elements, got {type(output)}")
                except Exception as e:
                    self.logger.error(f"PSCCNet predict failed: {e}")
                    self.logger.error(f"Model device: {getattr(self.method, 'device', 'No device attr')}")
                    self.logger.error(f"Input devices: {[(k, v.device) for k, v in input_data.items()]}")
                    raise


    async def psccnet_method(
        self,
        image_bytes: bytes,
        pre_decoded_image: Optional[np.ndarray] = None,
    ) -> PsccnetMethodData:
        try:
            # OPTIMIZATION: Use pre-decoded image if available to avoid redundant decoding
            if pre_decoded_image is not None:
                image_np = pre_decoded_image
                self.logger.debug("Using pre-decoded shared image")
            else:
                image_np = self._preprocess_image(image_bytes)
            input_data = self._prepare_model_input(image_np)
            confidence_score = self._predict(input_data)

            self.logger.info(
                f"PSCCNET analysis completed. Confidence score: {confidence_score:.4f}")
            self.logger.info("PSCCNET Method analysis completed successfully")
            return PsccnetMethodData(psccnet_confidence_score=confidence_score)

        except Exception as e:
            self.logger.error(
                f"Error processing image",
                context={"error": str(e), "exception_type": type(e).__name__}
            )
            raise

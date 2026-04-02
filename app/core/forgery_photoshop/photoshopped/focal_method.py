import io
import os
from pathlib import Path
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import numpy as np
import torch
from PIL import Image
from app.photoholmes.methods.focal import Focal, focal_preprocessing
from app.core.forgery_photoshop.photoshopped.focal_optimized_vit import OptimizedViTFocal

from app.dto import FocalMethodData
from app.core.logger import get_logger
from app.utils.autocast_utils import DeviceAwareAutocast


class FocalMethod:
    """FOCAL method for detecting focal length inconsistencies in images.

    Note: This method requires 1024x1024 processing due to pre-trained ViT weights.
    Not recommended for CPU-only systems due to computational requirements.
    """
    MAX_IMAGE_SIZE: int = 512  # Input resize before processing
    SUPPORTED_CONTENT_TYPE: str = 'image/'
    RGB_MODE: str = 'RGB'

    def __init__(self, enabled: bool = True):
        self.logger = get_logger()
        self.enabled = enabled
        self.device = self._get_device()
        self.focal_instance = None
        self._weights_loaded = False

        if not self.enabled:
            self.logger.info("FOCAL: Method disabled")
            return

        # Check if we should enable FOCAL based on device capabilities
        if self.device == "cpu":
            self.logger.warning("FOCAL: Method requires GPU, disabling on CPU")
            self.enabled = False
            return

        try:
            weights_config = self._get_weights_config()
            if not weights_config:
                self.logger.warning("FOCAL: Weights not found, disabling method")
                self.enabled = False
                return

            self.logger.info("FOCAL: Initializing with ViT and HRNet weights")
            focal_raw = Focal(weights=weights_config, device=self.device)
            self.focal_instance = OptimizedViTFocal(focal_raw, self.logger)
            self._weights_loaded = True
            self.logger.info("FOCAL: Method initialized successfully (1024x1024 processing)")
        except Exception as e:
            self.logger.error(f"FOCAL: Failed to initialize: {e}")
            self.enabled = False

    def _get_weights_config(self) -> Dict[str, str]:
        """Get the configuration for Focal method weights."""
        try:
            weights_dir = Path(__file__).parent.parent.parent.parent.parent / "photoholmes" / "weights" / "focal"
            vit_weights = weights_dir / "VIT_weights.pth"
            hrnet_weights = weights_dir / "HRNet_weights.pth"

            if vit_weights.exists() and hrnet_weights.exists():
                self.logger.info(f"FOCAL: Weights found - VIT: {vit_weights.stat().st_size:,} bytes, HRNet: {hrnet_weights.stat().st_size:,} bytes")
                return {
                    "ViT": str(vit_weights),
                    "HRNet": str(hrnet_weights)
                }
            else:
                self.logger.warning("FOCAL: Weights not found")
                return {}
        except Exception as e:
            self.logger.error(f"FOCAL: Failed to check weights: {e}")
            return {}

    @staticmethod
    def _get_device() -> str:
        if torch.cuda.is_available():
            return "cuda:0"
        # MPS disabled due to compatibility issues
        return "cpu"

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
                    raise ValueError("Invalid image data: contains invalid values")

                return image_np
        except Exception as e:
            self.logger.error(f"FOCAL: Image preprocessing failed: {str(e)}")
            raise ValueError(f"Failed to preprocess image for FOCAL: {str(e)}")

    def _prepare_model_input(self, image_np: np.ndarray) -> Dict[str, Any]:
        # Convert numpy array to tensor (HWC -> CHW)
        image_tensor = torch.from_numpy(image_np).float().permute(2, 0, 1)
        input_data = {"image": image_tensor}

        # Apply focal preprocessing
        preprocessed_data = focal_preprocessing(**input_data)

        # Ensure tensor format and device
        if "image" in preprocessed_data:
            if isinstance(preprocessed_data["image"], np.ndarray):
                preprocessed_data["image"] = torch.from_numpy(preprocessed_data["image"]).float()
                if preprocessed_data["image"].dim() == 3:  # H, W, C -> C, H, W
                    preprocessed_data["image"] = preprocessed_data["image"].permute(2, 0, 1)

            preprocessed_data["image"] = preprocessed_data["image"].to(self.device)

        return preprocessed_data

    def _calculate_confidence_score(self, result: Any) -> float:
        """Calculate confidence score from FOCAL result."""
        try:
            if isinstance(result, (tuple, list)):
                if len(result) > 0:
                    first_element = result[0]
                    if isinstance(first_element, (torch.Tensor, np.ndarray)):
                        score = float(np.mean(first_element.flatten()))
                    else:
                        score = float(first_element) if first_element is not None else 0.0
                else:
                    score = 0.0
            elif isinstance(result, torch.Tensor):
                score = float(np.mean(result.detach().cpu().numpy().flatten()))
            elif isinstance(result, np.ndarray):
                score = float(np.mean(result.flatten()))
            elif isinstance(result, (int, float)):
                score = float(result)
            elif isinstance(result, dict):
                score = 0.0
                for key in ['confidence', 'score', 'probability', 'detection']:
                    if key in result:
                        value = result[key]
                        if isinstance(value, (torch.Tensor, np.ndarray)):
                            if value.size > 0:
                                score = float(np.mean(value.flatten() if hasattr(value, 'flatten') else value))
                                break
                        else:
                            score = float(value) if value is not None else 0.0
                            break
            else:
                score = float(result) if result is not None else 0.0

            return min(max(score, 0.0), 1.0)
        except Exception as e:
            self.logger.error(f"FOCAL: Error calculating confidence score: {str(e)}")
            return 0.0

    def _predict_with_timeout(self, input_data: Dict[str, Any], timeout: int = 180) -> Any:
        """Run prediction with timeout to prevent hanging"""
        def prediction_task():
            return self._predict_internal(input_data)

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(prediction_task)
                result = future.result(timeout=timeout)
                return result
        except FutureTimeoutError:
            self.logger.error(f"FOCAL: Prediction timed out after {timeout} seconds")
            raise TimeoutError(f"FOCAL prediction timed out after {timeout} seconds")

    def _predict_internal(self, input_data: Dict[str, Any]) -> Any:
        """Internal prediction method without timeout"""
        if not self.enabled or not self.focal_instance:
            raise RuntimeError("FOCAL method not initialized or disabled")

        try:
            with torch.no_grad():
                with DeviceAwareAutocast(self.device, torch.float16, self.logger):
                    result = self.focal_instance.predict(**input_data)
                    return result
        except Exception as e:
            self.logger.error(f"FOCAL: Prediction failed: {str(e)}")
            raise

    async def focal_method(self, image_bytes: bytes, pre_decoded_image: Optional[np.ndarray] = None) -> FocalMethodData:
        if not self.enabled:
            self.logger.info("FOCAL: Method disabled, returning 0.0 confidence")
            return FocalMethodData(focal_confidence_score=0.0)

        try:
            # OPTIMIZATION: Use pre-decoded image if available to avoid redundant decoding
            if pre_decoded_image is not None:
                image_np = pre_decoded_image
                self.logger.debug("Using pre-decoded shared image")
            else:
                image_np = self._preprocess_image(image_bytes)
            input_data = self._prepare_model_input(image_np)

            try:
                result = self._predict_with_timeout(input_data, timeout=180)
            except TimeoutError:
                self.logger.warning("FOCAL: Prediction timed out, returning 0.0 confidence")
                return FocalMethodData(focal_confidence_score=0.0)

            confidence_score = self._calculate_confidence_score(result)
            self.logger.info(f"FOCAL: Analysis completed. Confidence: {confidence_score:.4f}")
            return FocalMethodData(focal_confidence_score=confidence_score)

        except Exception as e:
            self.logger.error(f"FOCAL: Error processing image: {str(e)}")
            return FocalMethodData(focal_confidence_score=0.0)
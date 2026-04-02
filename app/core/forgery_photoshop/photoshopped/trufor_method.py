import io
import os
from typing import Dict, Any, Optional

import numpy as np
import torch
from PIL import Image
from app.photoholmes.methods.trufor import TruFor, trufor_preprocessing

from app.dto import TruforMethodData
from app.core.logger import get_logger
from app.utils.autocast_utils import DeviceAwareAutocast


class TruforMethod:
    MAX_IMAGE_SIZE: int = 512
    SUPPORTED_CONTENT_TYPE: str = 'image/'
    RGB_MODE: str = 'RGB'
    # TruFor weights path - can be updated to actual weights location
    TRUFOR_WEIGHTS_PATH = "app/photoholmes/weights/trufor/TruFor_weights.pth"

    def __init__(self):
        self.logger = get_logger()
        self.device = self._get_device()

        # Try to load TruFor with weights
        if os.path.exists(self.TRUFOR_WEIGHTS_PATH):
            self.logger.info(f"Loading TruFor with weights from {self.TRUFOR_WEIGHTS_PATH}")
            self.trufor_instance = TruFor(weights=self.TRUFOR_WEIGHTS_PATH)
        else:
            self.logger.warning(f"TruFor weights not found at {self.TRUFOR_WEIGHTS_PATH}, using random weights")
            self.logger.info("Note: To download TruFor weights, visit https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip")
            self.trufor_instance = TruFor()  # Fallback to random weights

        self.logger.info("TruFor Method initialized")

    @staticmethod
    def _get_device() -> str:
        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
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
                    raise ValueError(
                        "Invalid image data: contains invalid values")

                return image_np

        except Exception as e:
            self.logger.error(f"TruFor image preprocessing failed: {str(e)}")
            raise ValueError(f"Failed to preprocess image for TruFor: {str(e)}")

    def _prepare_model_input(self, image_np: np.ndarray) -> Dict[str, Any]:
        image_data = {"image": image_np}
        input_data = trufor_preprocessing(**image_data)

        # Convert arrays to tensors and ensure proper device placement
        if "image" in input_data:
            image = input_data["image"]
            if isinstance(image, np.ndarray):
                # Convert numpy array to tensor properly
                image_tensor = torch.from_numpy(image).float()
                # Ensure proper dimension ordering and normalization
                if image_tensor.dim() == 3:  # H, W, C -> C, H, W
                    image_tensor = image_tensor.permute(2, 0, 1)
                image = image_tensor
            elif isinstance(image, torch.Tensor):
                # Ensure it's a float tensor
                image = image.float()

            # Move tensor to the correct device
            if isinstance(image, torch.Tensor):
                image = image.to(self.device)

            input_data["image"] = image

        self.logger.debug(f"TruFor input prepared. Image type: {type(input_data.get('image'))}, shape: {getattr(input_data.get('image'), 'shape', 'N/A')}, device: {getattr(input_data.get('image'), 'device', 'N/A')}")
        return input_data

    def _calculate_confidence_score(self, result: Any) -> tuple[float, Optional[float]]:
        """
        Calculate confidence scores from TruFor result.
        TruFor typically returns detection score and heatmap.
        """
        detection_score = 0.0
        heatmap_score = None

        try:
            if isinstance(result, dict):
                # Handle dictionary result from TruFor predict method
                detection_score = result.get('detection_score', 0.0)
                heatmap_tensor = result.get('heatmap')

                if isinstance(detection_score, torch.Tensor):
                    # Move to CPU before converting, especially for MPS
                    if detection_score.device.type == 'mps':
                        detection_score = detection_score.cpu()
                    detection_score = float(detection_score.item())
                elif isinstance(detection_score, (int, float, np.number)):
                    detection_score = float(detection_score)

                if heatmap_tensor is not None and isinstance(heatmap_tensor, torch.Tensor):
                    # Move to CPU before converting, especially for MPS
                    if heatmap_tensor.device.type == 'mps':
                        heatmap_tensor = heatmap_tensor.cpu()
                    heatmap_score = float(torch.mean(heatmap_tensor).item())
                elif heatmap_tensor is not None and isinstance(heatmap_tensor, np.ndarray):
                    heatmap_score = float(np.mean(heatmap_tensor))

            elif isinstance(result, (tuple, list)):
                if len(result) >= 1:
                    first_element = result[0]
                    if isinstance(first_element, torch.Tensor):
                        # Move to CPU before converting, especially for MPS
                        if first_element.device.type == 'mps':
                            first_element = first_element.cpu()
                        if first_element.numel() == 1:
                            detection_score = float(first_element.item())
                        else:
                            detection_score = float(torch.mean(first_element).item())
                    elif isinstance(first_element, np.ndarray):
                        detection_score = float(np.mean(first_element))
                    elif isinstance(first_element, (int, float)):
                        detection_score = float(first_element)

                if len(result) >= 2:
                    second_element = result[1]
                    if isinstance(second_element, torch.Tensor):
                        # Move to CPU before converting, especially for MPS
                        if second_element.device.type == 'mps':
                            second_element = second_element.cpu()
                        heatmap_score = float(torch.mean(second_element).item())
                    elif isinstance(second_element, np.ndarray):
                        heatmap_score = float(np.mean(second_element))
                    elif hasattr(second_element, 'mean'):
                        heatmap_score = float(second_element.mean())

            elif isinstance(result, torch.Tensor):
                # Handle PyTorch tensor safely - move to CPU first, especially for MPS
                if result.device.type == 'mps':
                    result = result.cpu()
                if result.numel() == 1:
                    detection_score = float(result.item())
                else:
                    detection_score = float(torch.mean(result).item())
            elif isinstance(result, np.ndarray):
                detection_score = float(np.mean(result))
            elif isinstance(result, (int, float, np.number)):
                detection_score = float(result)

            # Ensure scores are between 0 and 1
            detection_score = min(max(detection_score, 0.0), 1.0)
            if heatmap_score is not None:
                heatmap_score = min(max(float(heatmap_score), 0.0), 1.0)

            return detection_score, heatmap_score

        except (ValueError, TypeError, AttributeError) as e:
            self.logger.warning(f"Error calculating TruFor confidence score: {e}, using default scores")
            return 0.0, None

    def _predict(self, input_data: Dict[str, Any]) -> Any:
        with torch.no_grad():
            with DeviceAwareAutocast(self.device, torch.float16, self.logger):
                return self.trufor_instance.predict(**input_data)

    async def trufor_method(self, image_bytes: bytes, pre_decoded_image: Optional[np.ndarray] = None) -> TruforMethodData:
        # Skip TruFor on CPU and MPS - too slow for practical use (requires CUDA)
        if self.device in ("cpu", "mps"):
            self.logger.warning(f"TruFor skipped on {self.device} - requires CUDA GPU for practical performance")
            return TruforMethodData(
                trufor_confidence_score=None,
                trufor_heatmap_score=None
            )

        try:
            # OPTIMIZATION: Use pre-decoded image if available to avoid redundant decoding
            if pre_decoded_image is not None:
                image_np = pre_decoded_image
                self.logger.debug("Using pre-decoded shared image")
            else:
                image_np = self._preprocess_image(image_bytes)
            input_data = self._prepare_model_input(image_np)
            result = self._predict(input_data)
            confidence_score, heatmap_score = self._calculate_confidence_score(result)

            self.logger.info(
                f"TruFor analysis completed. Confidence score: {confidence_score:.4f}, heatmap: {heatmap_score}")
            self.logger.info("TruFor Method analysis completed successfully")
            return TruforMethodData(
                trufor_confidence_score=confidence_score,
                trufor_heatmap_score=heatmap_score
            )

        except Exception as e:
            self.logger.error(
                f"Error processing image with TruFor",
                context={"error": str(e), "exception_type": type(e).__name__}
            )
            # Return 0.0 confidence on error rather than raising
            self.logger.warning("Returning 0.0 confidence due to TruFor processing error")
            return TruforMethodData(
                trufor_confidence_score=0.0,
                trufor_heatmap_score=None
            )
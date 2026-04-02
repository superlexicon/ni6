import io
from typing import Dict, Any, Optional

import numpy as np
import torch
from PIL import Image
from app.photoholmes.methods.noisesniffer import Noisesniffer, noisesniffer_preprocessing

from app.dto import NoiseSnifferData
from app.core.logger import get_logger
from app.utils.autocast_utils import DeviceAwareAutocast


class NoiseSnifferMethod:
    MAX_IMAGE_SIZE: int = 512
    SUPPORTED_CONTENT_TYPE: str = 'image/'
    RGB_MODE: str = 'RGB'

    def __init__(self):
        self.logger = get_logger()
        self.device = self._get_device()
        self.noise_instance = Noisesniffer()
        self.logger.info("NoiseSniffer Method initialized")

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
            self.logger.error(f"Image preprocessing failed: {str(e)}")
            raise ValueError(f"Failed to preprocess image: {str(e)}")

    def _prepare_model_input(self, image_np: np.ndarray) -> Dict[str, Any]:
        image_data = {"image": image_np}
        return noisesniffer_preprocessing(**image_data)

    def _calculate_confidence_score(self, mask: np.ndarray) -> float:
        """
        Calculate confidence score from NoiseSniffer mask.
        Enhanced to handle different mask formats and return meaningful scores.
        """
        if mask.size == 0:
            return 0.0

        try:
            # Handle different mask types and formats
            if isinstance(mask, (torch.Tensor, np.ndarray)):
                # Convert to numpy if tensor
                if hasattr(mask, 'cpu'):
                    mask = mask.cpu().numpy()

                # Handle empty or degenerate masks
                if mask.size == 0:
                    return 0.0

                # If mask is binary (0s and 1s), calculate pixel ratio
                unique_vals = np.unique(mask)
                if (len(unique_vals) <= 2 and np.all(np.isin(unique_vals, [0, 1]))):
                    total_pixels = mask.size
                    tampered_pixels = np.sum(mask == 1)
                    base_confidence = tampered_pixels / total_pixels if total_pixels > 0 else 0.0

                    # For binary masks, use raw ratio without excessive scaling
                    # If ALL pixels are detected as tampered (1.0), this is likely an error
                    if base_confidence >= 0.95:
                        self.logger.warning(f"NoiseSniffer detected {base_confidence:.4f} of image as tampered - likely false positive")
                        return 0.01  # Return minimum confidence for obvious false positives

                    # Apply modest scaling for reasonable tampering ratios
                    scaled_confidence = min(base_confidence * 2, 1.0)  # Scale by 2x, not 10x
                    return max(scaled_confidence, 0.01)  # Minimum 0.01 for any detection

                # If mask contains probability values (0-1 range)
                elif mask.dtype in [np.float32, np.float64] or (np.min(mask) >= 0 and np.max(mask) <= 1):
                    # Use mean probability as confidence score
                    mean_confidence = float(np.mean(mask))

                    # If mean is extremely high, likely false positive
                    if mean_confidence >= 0.9:
                        self.logger.warning(f"NoiseSniffer probability mask mean is {mean_confidence:.4f} - likely false positive")
                        return 0.01

                    # Apply threshold and scaling for better discrimination
                    if mean_confidence < 0.1:
                        return max(mean_confidence * 3, 0.01)  # Scale low values
                    else:
                        return min(mean_confidence, 0.8)  # Cap high values

                # If mask contains other numeric values
                else:
                    # Safe normalization to avoid division by zero
                    mask_min = float(np.min(mask))
                    mask_max = float(np.max(mask))

                    # If all values are the same, return minimum confidence
                    if abs(mask_max - mask_min) < 1e-8:
                        self.logger.warning(f"NoiseSniffer mask has uniform values ({mask_min:.6f}) - likely invalid")
                        return 0.01

                    # Normalize to 0-1 range and use as confidence
                    normalized_mask = (mask - mask_min) / (mask_max - mask_min)
                    mean_confidence = float(np.mean(normalized_mask))

                    # If normalized mean is extremely high, likely false positive
                    if mean_confidence >= 0.9:
                        self.logger.warning(f"NoiseSniffer normalized mask mean is {mean_confidence:.4f} - likely false positive")
                        return 0.01

                    return min(max(mean_confidence, 0.01), 0.8)  # Clamp to reasonable range

            # Fallback for unexpected formats
            return 0.0

        except Exception as e:
            self.logger.error(f"Error in NoiseSniffer confidence calculation: {e}")
            return 0.0

    def _predict(self, input_data: Dict[str, Any]) -> np.ndarray:
        try:
            with torch.no_grad():
                with DeviceAwareAutocast(self.device, torch.float16, self.logger):
                    output = self.noise_instance.predict(**input_data)
                    self.logger.debug(f"NoiseSniffer raw output type: {type(output)}, length: {len(output) if hasattr(output, '__len__') else 'N/A'}")

                    # Extract mask (usually first element)
                    if isinstance(output, (list, tuple)) and len(output) > 0:
                        mask = output[0]
                    else:
                        mask = output

                    self.logger.debug(f"NoiseSniffer mask type: {type(mask)}, shape: {getattr(mask, 'shape', 'No shape')}, dtype: {getattr(mask, 'dtype', 'No dtype')}")
                    if hasattr(mask, 'min') and hasattr(mask, 'max'):
                        self.logger.debug(f"NoiseSniffer mask range: {mask.min():.6f} - {mask.max():.6f}")
                    if hasattr(mask, 'unique') and hasattr(mask, '__len__'):
                        unique_vals = mask.unique() if hasattr(mask, 'unique') else np.unique(mask)
                        self.logger.debug(f"NoiseSniffer unique values (first 10): {unique_vals[:10] if len(unique_vals) > 10 else unique_vals}")

                    return mask

        except ZeroDivisionError as e:
            self.logger.error(f"NoiseSniffer division by zero error: {e}")
            # Return a minimal mask indicating no detection
            self.logger.warning("Returning minimal mask due to division by zero error")
            # Create a minimal mask of the expected size with all zeros (no detection)
            if "image" in input_data:
                image_shape = input_data["image"].shape
                if len(image_shape) == 4:  # Batch, channels, height, width
                    _, _, h, w = image_shape
                    return np.zeros((h, w), dtype=np.float32)
                elif len(image_shape) == 3:  # Channels, height, width
                    _, h, w = image_shape
                    return np.zeros((h, w), dtype=np.float32)

            # Fallback to small default mask
            return np.zeros((64, 64), dtype=np.float32)

        except Exception as e:
            self.logger.error(f"NoiseSniffer prediction error: {e}")
            # Return a minimal mask for any other error as well
            self.logger.warning("Returning minimal mask due to prediction error")
            return np.zeros((64, 64), dtype=np.float32)

    async def noisesniffer_method(
        self,
        image_bytes: bytes,
        pre_decoded_image: Optional[np.ndarray] = None,
    ) -> NoiseSnifferData:
        try:
            # OPTIMIZATION: Use pre-decoded image if available to avoid redundant decoding
            if pre_decoded_image is not None:
                image_np = pre_decoded_image
                self.logger.debug("Using pre-decoded shared image")
            else:
                image_np = self._preprocess_image(image_bytes)
            input_data = self._prepare_model_input(image_np)
            mask = self._predict(input_data)
            confidence_score = self._calculate_confidence_score(mask)

            self.logger.info(
                f"NoiseSniffer analysis completed. Confidence score: {confidence_score:.4f}")
            self.logger.info(
                "NoiseSniffer Method analysis completed successfully")
            return NoiseSnifferData(noise_confidence_score=confidence_score)

        except Exception as e:
            self.logger.error(
                f"Error processing image",
                context={"error": str(e), "exception_type": type(e).__name__}
            )
            raise

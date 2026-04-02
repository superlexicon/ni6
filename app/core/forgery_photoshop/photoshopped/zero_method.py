import io
from typing import Dict, Any, Optional

import numpy as np
import torch
from PIL import Image
from app.photoholmes.methods.zero import Zero, zero_preprocessing

from app.dto import ZeroMethodData
from app.core.logger import get_logger
from app.utils.autocast_utils import DeviceAwareAutocast


class ZeroMethod:
    MAX_IMAGE_SIZE: int = 512
    SUPPORTED_CONTENT_TYPE: str = 'image/'
    RGB_MODE: str = 'RGB'

    def __init__(self):
        self.logger = get_logger()
        self.device = self._get_device()
        self.zero_instance = Zero()
        self.logger.info("ZERO Method initialized")

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
            self.logger.error(f"ZERO image preprocessing failed: {str(e)}")
            raise ValueError(f"Failed to preprocess image for ZERO: {str(e)}")

    def _prepare_model_input(self, image_np: np.ndarray) -> Dict[str, Any]:
        image_data = {"image": image_np}
        return zero_preprocessing(**image_data)

    def _calculate_confidence_score(self, result: Any) -> float:
        """
        Calculate confidence score from ZERO result.
        ZERO method returns (joint_mask, forgery_mask, votes, mask_missing_grids).
        """
        try:
            # Debug: Log what we received
            self.logger.info(f"ZERO: Calculating score from result type: {type(result)}")

            if isinstance(result, (tuple, list)):
                self.logger.info(f"ZERO: Result is {type(result).__name__} with {len(result)} elements")

                if len(result) >= 2:  # Should have at least joint_mask and forgery_mask
                    # Extract the masks
                    joint_mask = result[0]  # Main forgery detection mask
                    forgery_mask = result[1]  # Forgery-only mask
                    votes = result[2] if len(result) > 2 else None

                    # Debug: Log mask information
                    if isinstance(joint_mask, (torch.Tensor, np.ndarray)):
                        mask_sum = float(np.sum(joint_mask.flatten() if hasattr(joint_mask, 'flatten') else joint_mask))
                        mask_size = float(np.prod(joint_mask.shape) if hasattr(joint_mask, 'shape') else len(joint_mask))
                        forgery_ratio = mask_sum / mask_size if mask_size > 0 else 0.0
                        self.logger.info(f"ZERO: Joint mask sum={mask_sum}, size={mask_size}, forgery_ratio={forgery_ratio:.6f}")

                        # Use forgery ratio as confidence score (percentage of forged pixels)
                        score = forgery_ratio
                    else:
                        self.logger.warning(f"ZERO: Unexpected mask type: {type(joint_mask)}")
                        score = 0.0
                else:
                    self.logger.warning("ZERO: Result tuple has insufficient elements")
                    score = 0.0

            elif isinstance(result, torch.Tensor):
                self.logger.info(f"ZERO: Single tensor result with shape: {result.shape}")
                score = float(result.mean().item())
            elif isinstance(result, np.ndarray):
                self.logger.info(f"ZERO: Single array result with shape: {result.shape}")
                score = float(np.mean(result))
            elif isinstance(result, (int, float)):
                score = float(result)
            else:
                self.logger.warning(f"ZERO: Unexpected result type: {type(result)}, using default score")
                score = 0.0

            # Ensure it's between 0 and 1
            final_score = min(max(score, 0.0), 1.0)
            self.logger.info(f"ZERO: Final confidence score: {final_score:.6f}")
            return final_score

        except Exception as e:
            self.logger.error(f"ZERO: Error calculating confidence score: {type(e).__name__}")
            return 0.0

    def _predict(self, input_data: Dict[str, Any]) -> Any:
        self.logger.info(f"ZERO: Starting prediction with input keys: {list(input_data.keys())}")

        # Debug input data
        for key, value in input_data.items():
            if hasattr(value, 'shape'):
                self.logger.info(f"ZERO: Input '{key}' shape: {value.shape}, dtype: {getattr(value, 'dtype', 'N/A')}")
            else:
                self.logger.info(f"ZERO: Input '{key}' type: {type(value)}, value: {value}")

        with torch.no_grad():
            with DeviceAwareAutocast(self.device, torch.float16, self.logger):
                result = self.zero_instance.predict(**input_data)
                self.logger.info(f"ZERO: Prediction completed successfully")
                return result

    async def zero_method(self, image_bytes: bytes, pre_decoded_image: Optional[np.ndarray] = None) -> ZeroMethodData:
        try:
            # OPTIMIZATION: Use pre-decoded image if available to avoid redundant decoding
            if pre_decoded_image is not None:
                image_np = pre_decoded_image
                self.logger.debug("Using pre-decoded shared image")
            else:
                image_np = self._preprocess_image(image_bytes)
            input_data = self._prepare_model_input(image_np)
            result = self._predict(input_data)
            confidence_score = self._calculate_confidence_score(result)

            self.logger.info(
                f"ZERO analysis completed. Confidence score: {confidence_score:.4f}")
            self.logger.info("ZERO Method analysis completed successfully")
            return ZeroMethodData(zero_confidence_score=confidence_score)

        except Exception as e:
            self.logger.error(
                f"Error processing image with ZERO",
                context={"error": str(e), "exception_type": type(e).__name__}
            )
            # Return 0.0 confidence on error rather than raising
            self.logger.warning("Returning 0.0 confidence due to ZERO processing error")
            return ZeroMethodData(zero_confidence_score=0.0)
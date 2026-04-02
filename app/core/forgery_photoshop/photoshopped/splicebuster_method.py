import asyncio
import io
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional

import numpy as np
from PIL import Image
from app.photoholmes.methods.splicebuster import Splicebuster, splicebuster_preprocessing

from app.dto import SplicebusterMethodData
from app.core.logger import get_logger


class SplicebusterMethod:
    MAX_IMAGE_SIZE: int = 512
    SUPPORTED_CONTENT_TYPE: str = 'image/'
    RGB_MODE: str = 'RGB'

    # Thread pool for CPU-bound prediction to allow asyncio timeouts to work
    _executor = ThreadPoolExecutor(max_workers=2)

    def __init__(self):
        self.logger = get_logger()
        # Splicebuster is a pure numpy method - no GPU needed
        self.splicebuster_instance = Splicebuster()
        self.logger.info("Splicebuster Method initialized")

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
            self.logger.error(f"Splicebuster image preprocessing failed: {str(e)}")
            raise ValueError(f"Failed to preprocess image for Splicebuster: {str(e)}")

    def _prepare_model_input(self, image_np: np.ndarray) -> Dict[str, Any]:
        image_data = {"image": image_np}
        return splicebuster_preprocessing(**image_data)

    def _calculate_confidence_score(self, result: Any) -> float:
        """
        Calculate confidence score from Splicebuster result.
        Splicebuster detects copy-paste/splicing forgery.
        """
        if isinstance(result, (tuple, list)):
            # If it returns a tuple, take the first element as score
            score = float(result[0]) if len(result) > 0 else 0.0
        elif isinstance(result, np.ndarray):
            score = float(np.mean(result))
        elif isinstance(result, (int, float)):
            score = float(result)
        else:
            # Default case
            score = 0.0

        return min(max(score, 0.0), 1.0)  # Ensure it's between 0 and 1

    def _predict(self, input_data: Dict[str, Any]) -> Any:
        # Splicebuster is pure numpy - no PyTorch context needed
        return self.splicebuster_instance.predict(**input_data)

    async def splicebuster_method(self, image_bytes: bytes, pre_decoded_image: Optional[np.ndarray] = None) -> SplicebusterMethodData:
        try:
            # OPTIMIZATION: Use pre-decoded image if available to avoid redundant decoding
            if pre_decoded_image is not None:
                image_np = pre_decoded_image
                self.logger.debug("Using pre-decoded shared image")
            else:
                image_np = self._preprocess_image(image_bytes)
            input_data = self._prepare_model_input(image_np)

            # Run CPU-bound prediction in thread pool so asyncio timeout works
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                self._executor,
                self._predict,
                input_data
            )

            confidence_score = self._calculate_confidence_score(result)

            self.logger.info(
                f"Splicebuster analysis completed. Confidence score: {confidence_score:.4f}")
            self.logger.info("Splicebuster Method analysis completed successfully")
            return SplicebusterMethodData(splicebuster_confidence_score=confidence_score)

        except Exception as e:
            self.logger.error(
                f"Error processing image with Splicebuster",
                context={"error": str(e), "exception_type": type(e).__name__}
            )
            # Return 0.0 confidence on error rather than raising
            self.logger.warning("Returning 0.0 confidence due to Splicebuster processing error")
            return SplicebusterMethodData(splicebuster_confidence_score=0.0)

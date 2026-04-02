import os
import tempfile
from typing import Dict, Any
import numpy as np
import torch
from app.photoholmes.utils.image import read_image, read_jpeg_data
from app.photoholmes.methods.dq import DQ, dq_preprocessing
from app.dto import DQMethodData
from .base_forgery_method import BaseForgeryMethod
from app.core.logger import get_logger
from app.utils.autocast_utils import DeviceAwareAutocast


class DQMethod(BaseForgeryMethod[DQMethodData]):

    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg'}

    @staticmethod
    def _get_device() -> str:
        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def __init__(self):
        super().__init__()
        self.device = self._get_device()
        self.method = DQ()
        self.logger = get_logger()

    async def _save_temp_image(self, image_bytes: bytes) -> str:
        # Determine the correct extension based on the image content
        if self._is_jpeg_image(image_bytes):
            suffix = '.jpg'
        else:
            # For non-JPEG images, save with actual format to avoid parsing issues
            suffix = '.png'  # Default to PNG for non-JPEG images

        temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        try:
            temp_file.write(image_bytes)
            return temp_file.name
        except Exception as e:
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
            self.logger.error(f"Failed to save temporary image: {e}")
            raise
        finally:
            temp_file.close()

    async def _process_image(self, image_bytes: bytes) -> Dict[str, Any]:
        try:
            self.logger.debug("Starting DQ image processing...")

            # Early return for non-JPEG images - DQ only works with JPEG
            if not self._is_jpeg_image(image_bytes):
                self.logger.warning("DQMethod: Non-JPEG image detected, cannot process")
                raise ValueError("DQMethod only supports JPEG images")

            temp_path = await self._save_temp_image(image_bytes)
            try:
                self.logger.debug(f"DQ: Reading JPEG image from {temp_path}")
                image = read_image(temp_path)
                self.logger.debug(f"DQ: Image shape: {getattr(image, 'shape', 'Unknown shape')}")

                self.logger.debug("DQ: Reading JPEG data and DCT coefficients...")
                dct, q_tables = read_jpeg_data(temp_path)
                self.logger.debug(f"DQ: DCT shape: {getattr(dct, 'shape', 'Unknown shape')}")
                # Check if q_tables is a tensor or list/tuple
                q_tables_len = len(q_tables) if q_tables is not None and hasattr(q_tables, '__len__') else 0
                self.logger.debug(f"DQ: Quantization tables: {q_tables_len}")

                return {"image": image, "dct_coefficients": dct, "quantization_tables": q_tables}
            finally:
                if os.path.exists(temp_path):
                    self.logger.debug(f"DQ: Cleaning up temp file {temp_path}")
                    os.unlink(temp_path)
        except Exception as e:
            self.logger.error(f"Error during DQ image processing: {type(e).__name__}")
            raise ValueError(f"Image processing failed: {type(e).__name__}")

    def _analyze(self, processed_data: Dict[str, Any]) -> DQMethodData:
        try:
            self.logger.info(f"DQ analysis starting - input data keys: {list(processed_data.keys())}")

            # Check if DCT coefficients are available
            if 'dct_coefficients' not in processed_data:
                raise ValueError("DCT coefficients not found in processed data")

            dct_coefficients = processed_data['dct_coefficients']
            self.logger.debug(f"DQ: DCT coefficients shape: {getattr(dct_coefficients, 'shape', 'Unknown')}")

            # Validate DCT coefficients
            if dct_coefficients is None or (hasattr(dct_coefficients, 'size') and dct_coefficients.size == 0):
                raise ValueError("DCT coefficients are empty or None")

            self.logger.debug("DQ: Starting preprocessing...")
            input_data = dq_preprocessing(**processed_data)
            self.logger.debug(f"DQ: Preprocessing completed - keys: {list(input_data.keys())}")

            # Validate preprocessing output
            if 'dct_coefficients' not in input_data:
                raise ValueError("DCT coefficients lost during preprocessing")

            self.logger.info("DQ: Starting prediction...")
            with torch.no_grad():
                with DeviceAwareAutocast(self.device, torch.float16, self.logger):
                    self.logger.debug(f"DQ: Using device: {self.device}")
                    # Catch numpy warnings for divide by zero in PhotoHolmes DQ method
                    # This handles edge-case images with low-variance DCT coefficients
                    with np.errstate(divide='raise'):
                        try:
                            output = self.method.predict(**input_data)
                        except FloatingPointError:
                            self.logger.warning("DQ: Encountered zero values in spectrogram (edge-case image), returning default probability=0.0")
                            return DQMethodData(max_probability=0.0)
                    self.logger.debug(f"DQ: Prediction completed - output type: {type(output)}, shape: {getattr(output, 'shape', 'No shape')}")

            max_probability = self._extract_max_probability(output)
            self.logger.info(
                f"DQ analysis completed successfully. Max probability: {max_probability:.6f}")
            return DQMethodData(max_probability=max_probability)
        except Exception as e:
            self.logger.error(f"Error during DQ analysis: {type(e).__name__}")
            # Return a default result instead of raising to avoid breaking the pipeline
            self.logger.warning("Returning default DQ result due to analysis failure")
            return DQMethodData(max_probability=0.0)

    def _extract_max_probability(self, output) -> float:
        if isinstance(output, np.ndarray):
            return float(np.max(output))
        return float(output)

    async def dq_method(self, image_bytes: bytes) -> DQMethodData:
        # Check if this is actually a JPEG file first
        if not self._is_jpeg_image(image_bytes):
            self.logger.info("DQMethod received non-JPEG image, returning default result (DQ only works with JPEG)")
            return DQMethodData(max_probability=0.0)

        try:
            return await self.analyze_image(image_bytes)
        except ValueError as e:
            if "DQMethod only supports JPEG images" in str(e):
                self.logger.warning("DQMethod JPEG validation failed, returning default result")
                return DQMethodData(max_probability=0.0)
            else:
                # Re-raise other ValueError exceptions
                raise
        except Exception as e:
            self.logger.error(f"DQMethod analysis failed: {e}")
            return DQMethodData(max_probability=0.0)

    def _is_jpeg_image(self, image_bytes: bytes) -> bool:
        """Check if the image bytes represent a JPEG file"""
        # JPEG files start with FF D8 and end with FF D9
        if len(image_bytes) < 4:
            return False

        # Check JPEG signature
        is_jpeg = image_bytes.startswith(b'\xFF\xD8\xFF')

        # Additional check for JPEG markers
        if is_jpeg:
            self.logger.debug("DQ: Detected JPEG format from image bytes")

        return is_jpeg

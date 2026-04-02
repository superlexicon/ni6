import io
import signal
from typing import Dict, Any, Optional
import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import numpy as np
import torch
from PIL import Image
from app.photoholmes.methods.exif_as_language import EXIFAsLanguage, exif_as_language_preprocessing
from app.core.shared_exif_extractor import get_shared_exif_extractor

from app.dto import ExifAsLanguageData
from app.core.logger import get_logger
from app.utils.autocast_utils import DeviceAwareAutocast


class ExifAsLanguageMethod:
    MAX_IMAGE_SIZE: int = 512
    SUPPORTED_CONTENT_TYPE: str = 'image/'
    RGB_MODE: str = 'RGB'

    def __init__(self):
        self.logger = get_logger()
        self.device = self._get_device()
        # Use integrated photoholmes weights path
        weights_path = "app/photoholmes/weights/exif_as_language/weights.pth"
        try:
            # Use optimized configuration for 512x512 images to reduce computational load
            from app.photoholmes.methods.exif_as_language.config import EXIFAsLanguageArchConfig, ClipModelConfig

            # Create optimized config for our image size
            optimized_arch = EXIFAsLanguageArchConfig(
                clip_model=ClipModelConfig(vision="resnet50", text="distilbert", pooling="mean"),
                patch_size=128,
                num_per_dim=4,  # Reduced from 30 to match 512/128 = 4 patches per dimension
                feat_batch_size=8,  # Reduced from 32 to reduce memory usage
                pred_batch_size=16,  # Reduced from 1024 to prevent hanging
                ms_window=5,  # Reduced from 10
                ms_iter=3,  # Reduced from 5
            )

            self.exif_instance = EXIFAsLanguage(weights=weights_path, arch_config=optimized_arch, device=self.device)
            # Ensure the model is moved to the correct device after weight loading
            self.exif_instance.to_device(self.device)
            self.logger.info(f"EXIF As Language Method initialized with optimized config on device: {self.device}")
        except Exception as e:
            self.logger.warning(f"Failed to load EXIF As Language weights: {e}. Using model without weights.")
            # Fallback to basic config without weights
            try:
                from app.photoholmes.methods.exif_as_language.config import ClipModelConfig, EXIFAsLanguageArchConfig
                basic_arch = EXIFAsLanguageArchConfig(
                    clip_model=ClipModelConfig(vision="resnet50", text="distilbert", pooling="mean"),
                    patch_size=128,
                    num_per_dim=4,
                    feat_batch_size=4,
                    pred_batch_size=8,
                    ms_window=3,
                    ms_iter=2,
                )
                self.exif_instance = EXIFAsLanguage(arch_config=basic_arch, device=self.device)
            except:
                self.exif_instance = EXIFAsLanguage(device=self.device)
            self.exif_instance.to_device(self.device)
            self.logger.info(f"EXIF As Language Method initialized with basic config on device: {self.device}")

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
            self.logger.error(f"EXIF image preprocessing failed: {str(e)}")
            raise ValueError(f"Failed to preprocess image for EXIF: {str(e)}")

    async def _prepare_model_input(self, image_bytes: bytes, shared_exif_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Use shared EXIF data if available, otherwise extract it
        if shared_exif_data and 'photoholmes_exif' in shared_exif_data:
            self.logger.info("EXIF As Language: Using shared EXIF data")
            exif_dict = shared_exif_data['photoholmes_exif']
        else:
            self.logger.info("EXIF As Language: Extracting EXIF data (fallback)")
            # Fallback: Extract EXIF data from the image bytes
            exif_dict = {}
            try:
                with Image.open(io.BytesIO(image_bytes)) as img:
                    exif_data = img._getexif()
                    if exif_data is not None:
                        # Convert EXIF data to a serializable format
                        for tag_id, value in exif_data.items():
                            tag_name = str(tag_id)  # Simplified - in production would map to readable names
                            try:
                                # Try to serialize the value
                                json.dumps(value)
                                exif_dict[tag_name] = str(value)
                            except (TypeError, ValueError):
                                exif_dict[tag_name] = f"non_serializable_{type(value).__name__}"
            except Exception as e:
                self.logger.warning(f"Failed to extract EXIF data: {e}")

        # First preprocess the image to get numpy array
        image_np = self._preprocess_image(image_bytes)

        # Debug: Log the actual image shape after preprocessing
        self.logger.info(f"EXIF As Language: Preprocessed image shape: {image_np.shape}")

        image_data = {"image": image_np, "exif": exif_dict}

        # Debug: Check EXIF instance patch_size
        self.logger.info(f"EXIF As Language: Model patch_size: {self.exif_instance.patch_size}")

        # Only pass the image to preprocessing, as EXIF data is not used by the pipeline
        preprocessed_data = exif_as_language_preprocessing(image=image_data["image"])

        # Ensure the preprocessed tensor is moved to the correct device
        if "image" in preprocessed_data:
            preprocessed_data["image"] = preprocessed_data["image"].to(self.device)

        return preprocessed_data

    def _calculate_confidence_score(self, result: Any) -> float:
        """
        Calculate confidence score from EXIF As Language result.
        This method analyzes EXIF data for inconsistencies.
        """
        try:
            if isinstance(result, (tuple, list)):
                # If it returns a tuple, take the first element as score
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
                # If it returns a dict, look for common score keys
                score = 0.0
                for key in ['confidence', 'score', 'probability', 'detection']:
                    if key in result:
                        value = result[key]
                        if isinstance(value, (torch.Tensor, np.ndarray)):
                            score = float(np.mean(value.flatten() if hasattr(value, 'flatten') else value))
                        else:
                            score = float(value) if value is not None else 0.0
                        break
            else:
                # Default case - try to convert directly
                score = float(result) if result is not None else 0.0

            return min(max(score, 0.0), 1.0)  # Ensure it's between 0 and 1

        except (ValueError, TypeError, AttributeError) as e:
            self.logger.warning(f"Could not calculate confidence score from result: {e}, using default 0.0")
            return 0.0

    def _extract_exif_analysis(self, result: Any) -> Dict[str, Any]:
        """Extract detailed EXIF analysis if available"""
        try:
            if isinstance(result, dict):
                return {k: v for k, v in result.items() if k not in ['confidence', 'score', 'probability', 'detection']}
            else:
                return {"raw_result": str(result)}
        except Exception:
            return {}

    def _predict_with_timeout(self, input_data: Dict[str, Any], timeout: int = 60) -> Any:
        """Run prediction with timeout to prevent hanging"""
        def prediction_task():
            return self._predict_internal(input_data)

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(prediction_task)
                result = future.result(timeout=timeout)
                return result
        except FutureTimeoutError:
            self.logger.error(f"EXIF As Language: Prediction timed out after {timeout} seconds")
            raise TimeoutError(f"EXIF As Language prediction timed out after {timeout} seconds")

    def _predict_internal(self, input_data: Dict[str, Any]) -> Any:
        """Internal prediction method without timeout"""
        self.logger.info(f"EXIF As Language: Starting prediction on device: {self.device}")

        try:
            with torch.no_grad():
                self.logger.info("EXIF As Language: Entering torch.no_grad() context")
                with DeviceAwareAutocast(self.device, torch.float16, self.logger):
                    self.logger.info("EXIF As Language: Entering autocast context, calling model.predict()")

                    # Check input tensor details
                    if "image" in input_data:
                        image_tensor = input_data["image"]
                        self.logger.info(f"EXIF As Language: Input tensor shape: {image_tensor.shape}, dtype: {image_tensor.dtype}, device: {image_tensor.device}")

                    result = self.exif_instance.predict(**input_data)
                    self.logger.info("EXIF As Language: Model prediction completed successfully")
                    return result

        except Exception as e:
            self.logger.error(f"EXIF As Language: Prediction failed with error: {str(e)}")
            raise

    async def exif_as_language_method(self, image_bytes: bytes, shared_exif_data: Optional[Dict[str, Any]] = None) -> ExifAsLanguageData:
        try:
            input_data = await self._prepare_model_input(image_bytes, shared_exif_data)

            # Use prediction with timeout to prevent hanging
            try:
                result = self._predict_with_timeout(input_data, timeout=30)  # 30 second timeout
            except TimeoutError:
                self.logger.warning("EXIF As Language prediction timed out, returning default values")
                return ExifAsLanguageData(
                    exif_confidence_score=0.0,
                    exif_analysis={"error": "prediction_timeout", "message": "Prediction timed out after 30 seconds"}
                )

            # Debug: Log what the model returned
            self.logger.info(f"EXIF As Language: Model result type: {type(result)}")
            if isinstance(result, (tuple, list)):
                self.logger.info(f"EXIF As Language: Result is tuple/list with {len(result)} elements")
                if len(result) > 0:
                    self.logger.info(f"EXIF As Language: First element type: {type(result[0])}, shape: {getattr(result[0], 'shape', 'N/A')}")
            elif isinstance(result, dict):
                self.logger.info(f"EXIF As Language: Result is dict with keys: {list(result.keys())}")
            else:
                self.logger.info(f"EXIF As Language: Result shape: {getattr(result, 'shape', 'N/A')}, dtype: {getattr(result, 'dtype', 'N/A')}")

            confidence_score = self._calculate_confidence_score(result)
            exif_analysis = self._extract_exif_analysis(result)

            self.logger.info(
                f"EXIF As Language analysis completed. Confidence score: {confidence_score:.4f}")
            self.logger.info("EXIF As Language Method analysis completed successfully")
            return ExifAsLanguageData(
                exif_confidence_score=confidence_score,
                exif_analysis=exif_analysis
            )

        except Exception as e:
            self.logger.error(
                f"Error processing image with EXIF As Language",
                context={"error": str(e), "exception_type": type(e).__name__}
            )
            # Return 0.0 confidence on error rather than raising
            self.logger.warning("Returning 0.0 confidence due to EXIF processing error")
            return ExifAsLanguageData(
                exif_confidence_score=0.0,
                exif_analysis={"error": str(e)}
            )
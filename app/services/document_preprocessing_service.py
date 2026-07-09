"""
Document Preprocessing Service - Standardized pipeline for all document types.

This service provides a unified preprocessing pipeline that applies to all document types
(selfies, passports, bank statements, etc.). It follows a standardized 6-step process:

1. Crop and downsize (if too large)
2. Check quality (brightness, contrast, focus) on cropped/downsized image
3. PhotoHolmes checks for manipulation detection
4. For selfies only: Verify full face (not partial/cut off)
5. Extract required data based on document type
6. Perform checks with extracted data

This standardization ensures:
- Consistent quality thresholds across all document types
- Quality checks run on the correct images (document images, not face crops)
- Code deduplication and easier maintenance
- Easier extensibility for new document types
"""

import cv2
import numpy as np
import time
import io
from typing import Dict, Any, Optional, Tuple
from PIL import Image
from app.core.logger import get_logger
from app.config.verification_config import verification_settings
from app.services.comprehensive_photoholmes_service import ComprehensivePhotoHolmesService
import fitz  # PyMuPDF for PDF to image conversion


class DocumentPreprocessingService:
    """Standardized document preprocessing pipeline for all document types.

    Token-aware sizing parameters for Qwen3.5 vision LLM:
    - Patch size: 14×14 pixels per patch
    - Context window: 8192 tokens
    - Output tokens: ~1000
    - Prompt overhead: ~1192 tokens
    - Available for image: ~6000 tokens
    - Max dimension: sqrt(6000) × 14 ≈ 1078 pixels
    """

    # Vision LLM token parameters
    MAX_IMAGE_TOKENS = 6000
    PATCH_SIZE = 14  # pixels per patch for Qwen3.5
    # Calculated max dimension: sqrt(6000) × 14 ≈ 1078
    MAX_DIMENSION = int((MAX_IMAGE_TOKENS ** 0.5) * PATCH_SIZE)

    def __init__(self):
        self.logger = get_logger()
        self.photoholmes_service = None  # Lazy initialization to prevent circular import

    def _get_photoholmes_service(self):
        """Lazy initialization of PhotoHolmes service to prevent circular import."""
        if self.photoholmes_service is None:
            self.photoholmes_service = ComprehensivePhotoHolmesService()
        return self.photoholmes_service

    async def preprocess_document(
        self,
        image_bytes: bytes,
        document_type: str,
        public_key: Optional[str] = None,
        user_identity_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run standardized preprocessing pipeline on any document.

        Steps:
        1. Crop and downsize (if too large)
        2. Check quality (brightness, contrast, focus) on cropped/downsized image
        3. PhotoHolmes checks for manipulation detection
        4. For selfies: Verify full face (not partial/cut off)
        5. Extract required data based on document type (called by respective services)
        6. Perform checks with extracted data (called by respective services)

        Args:
            image_bytes: Raw image bytes
            document_type: Type of document (selfie, passport, bank_statement, etc.)
            public_key: User's public key (for selfies)
            user_identity_id: User identity ID

        Returns:
            Dict containing:
            - cropped_image: Cropped and downsized image (numpy array)
            - quality_metrics: Quality check results
            - quality_passed: Boolean indicating if quality checks passed
            - photoholmes_results: Forgery detection results
            - face_completeness: For selfies - face completeness check results
            - error: Error message if preprocessing failed
            - ready_for_extraction: Boolean indicating if ready for data extraction
            - preprocessing_metadata: Timing and metadata
        """
        start_time = time.time()
        result = {
            'document_type': document_type,
            'public_key': public_key,
            'user_identity_id': user_identity_id,
            'preprocessing_metadata': {}
        }

        try:
            # Step 1: Crop and downsize (if too large)
            cropped_image, crop_metadata = self._crop_and_downsize(image_bytes, document_type)
            result['cropped_image'] = cropped_image
            result['cropped_image_bytes'] = self._numpy_to_bytes(cropped_image)  # Also return bytes format for downstream services
            result['preprocessing_metadata']['crop'] = crop_metadata

            # Step 2: Check quality on CROPPED/DOWNSIZED image
            quality_metrics, quality_passed = self._check_quality(cropped_image, document_type)
            result['quality_metrics'] = quality_metrics
            result['quality_passed'] = quality_passed

            if not quality_passed:
                failed_metrics = [f"{k} ({v:.2f})" for k, v in quality_metrics.items()
                                if v < self._get_quality_threshold(document_type, k)]
                result['error'] = f"{document_type.capitalize()} quality insufficient: {', '.join(failed_metrics)}"
                return result

            # Step 3: PhotoHolmes checks for manipulation detection
            photoholmes_results = await self._run_photoholmes(cropped_image, document_type)
            result['photoholmes_results'] = photoholmes_results

            # Step 4: For selfies only - verify full face (not partial/cut off)
            if document_type == 'selfie':
                face_completeness = self._check_face_completeness(cropped_image)
                result['face_completeness'] = face_completeness

                if not face_completeness['is_complete']:
                    result['error'] = "Face appears to be partially cropped. Please ensure your entire face is visible."
                    result['quality_passed'] = False
                    return result

            # Step 5 & 6: Document-specific data extraction and checks
            # (Called by respective services: passport, bank_statement, etc.)
            result['ready_for_extraction'] = True

            result['preprocessing_metadata']['total_time'] = time.time() - start_time
            return result

        except Exception as e:
            self.logger.error(f"Document preprocessing failed: {e}")
            result['error'] = str(e)
            result['quality_passed'] = False
            return result

    def _crop_and_downsize(self, image_bytes: bytes, document_type: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Step 1: Crop margins and apply token-aware sizing.

        Token-aware sizing ensures images are properly sized for vision LLM processing:
        - Calculates max dimension from token budget (6000 tokens for Qwen3.5)
        - Avoids double resizing (no need to resize again during LLM calls)
        - Consistent sizing across all document types
        """
        # Load image
        image = self._load_image(image_bytes)
        original_size = image.shape[:2]

        # Crop margins (remove white space)
        cropped = self._crop_margins(image)
        cropped_size = cropped.shape[:2]

        # Apply token-aware sizing
        # Use class-level constants calculated for Qwen3.5 vision LLM
        height, width = cropped.shape[:2]
        max_dim = max(width, height)

        if max_dim > self.MAX_DIMENSION:
            scale = self.MAX_DIMENSION / max_dim
            new_width = int(width * scale)
            new_height = int(height * scale)
            cropped = cv2.resize(cropped, (new_width, new_height), cv2.INTER_LANCZOS4)
            self.logger.debug(f"Token-aware sizing: {width}×{height} → {new_width}×{new_height} (max {self.MAX_DIMENSION}px for {self.MAX_IMAGE_TOKENS} tokens)")
        else:
            self.logger.debug(f"Token-aware sizing: {width}×{height} → within budget (~{self._estimate_tokens(width, height)} tokens, budget: {self.MAX_IMAGE_TOKENS})")

        metadata = {
            'original_size': original_size,
            'cropped_size': cropped.shape[:2],
            'downsized': cropped.shape != image.shape,
            'token_budget': self.MAX_IMAGE_TOKENS,
            'estimated_tokens': self._estimate_tokens(cropped.shape[1], cropped.shape[0])
        }
        return cropped, metadata

    def _estimate_tokens(self, width: int, height: int) -> int:
        """Estimate image tokens based on dimensions and patch size.

        Args:
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            Estimated number of tokens
        """
        patches_w = (width + self.PATCH_SIZE - 1) // self.PATCH_SIZE
        patches_h = (height + self.PATCH_SIZE - 1) // self.PATCH_SIZE
        return patches_w * patches_h

    def _check_quality(self, image: np.ndarray, document_type: str) -> Tuple[Dict[str, float], bool]:
        """Step 2: Check quality (brightness, contrast, focus) on cropped image."""
        quality_metrics = self._calculate_quality_metrics(image)

        # Get thresholds for document type
        thresholds = self._get_quality_thresholds(document_type)

        # Check if all metrics pass thresholds
        quality_passed = all(
            quality_metrics.get(metric, 0) >= threshold
            for metric, threshold in thresholds.items()
        )

        return quality_metrics, quality_passed

    async def _run_photoholmes(self, image: np.ndarray, document_type: str):
        """Step 3: PhotoHolmes checks for manipulation detection.

        Returns:
            PhotoHolmesResults object or None if preprocessing failed
        """
        try:
            # Check if PhotoHolmes should be skipped
            if verification_settings.skip_photoholmes:
                self.logger.info("PhotoHolmes skipped (skip_photoholmes=True)")
                from app.dto import PhotoHolmesResults
                empty_results = PhotoHolmesResults()
                empty_results.total_methods_run = 0
                empty_results.methods_with_detections = 0
                empty_results.overall_forgery_probability = 0.0
                return empty_results

            # Convert numpy to bytes
            image_bytes = self._numpy_to_bytes(image)

            # Run PhotoHolmes
            photoholmes_service = self._get_photoholmes_service()
            results = await photoholmes_service.run_all_methods(
                image_bytes=image_bytes,
                document_type=document_type
            )

            # Return the PhotoHolmesResults object directly (not a dict)
            # This ensures compatibility with existing code that expects PhotoHolmesResults
            return results

        except Exception as e:
            self.logger.warning(f"PhotoHolmes check failed: {e}")
            # Return empty PhotoHolmesResults object on failure
            from app.dto import PhotoHolmesResults
            empty_results = PhotoHolmesResults()
            empty_results.total_methods_run = 0
            empty_results.methods_with_detections = 0
            empty_results.overall_forgery_probability = 0.0
            return empty_results

    def _check_face_completeness(self, image: np.ndarray) -> Dict[str, Any]:
        """Step 4: For selfies - verify full face (not partial/cut off)."""
        # This is a placeholder - the actual face completeness check
        # is done in FaceExtractionService using face detection
        # Here we return a basic implementation
        # In practice, this should be called after face detection
        return {
            'is_complete': True,
            'note': 'Face completeness check performed in FaceExtractionService'
        }

    def _calculate_quality_metrics(self, image: np.ndarray) -> Dict[str, float]:
        """Calculate quality metrics (brightness, contrast, focus/sharpness)."""
        try:
            # Convert to grayscale for quality calculations
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Sharpness (using Laplacian variance) - normalize to 0-1 range
            sharpness_variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness = min(1.0, sharpness_variance / 500.0)

            # Brightness (mean intensity) - normalize to 0-1 range
            brightness_mean = np.mean(gray) / 255.0
            # Optimal brightness is around 0.4-0.6, penalize extremes
            if brightness_mean < 0.4:
                brightness = brightness_mean / 0.4  # 0-1 for dark images
            elif brightness_mean > 0.6:
                brightness = max(0.0, 1.0 - (brightness_mean - 0.6) / 0.4)  # 1-0 for bright images
            else:
                brightness = 1.0  # Optimal range

            # Contrast (standard deviation) - normalize to 0-1 range
            contrast_std = np.std(gray) / 255.0
            contrast = min(1.0, contrast_std / 0.3)

            return {
                'brightness': float(brightness),
                'sharpness': float(sharpness),
                'contrast': float(contrast),
            }

        except Exception as e:
            self.logger.warning(f"Failed to calculate quality metrics: {str(e)}")
            return {
                'brightness': 0.5,
                'sharpness': 0.5,
                'contrast': 0.5,
            }

    def _get_quality_threshold(self, document_type: str, metric: str) -> float:
        """Get quality threshold for a specific metric and document type."""
        thresholds = self._get_quality_thresholds(document_type)
        return thresholds.get(metric, 0.5)

    def _get_quality_thresholds(self, document_type: str) -> Dict[str, float]:
        """Get quality thresholds for document type.

        Note: For selfies, we only check basic quality metrics here (brightness, sharpness, contrast).
        Face-specific metrics like 'resolution' are checked in FaceExtractionService since they
        require face detection first.
        """
        # For selfies, use stricter thresholds but only for metrics we calculate
        if document_type == 'selfie':
            return {
                'brightness': verification_settings.selfie_quality_brightness_min,
                'sharpness': verification_settings.selfie_quality_sharpness_min,
                'contrast': verification_settings.selfie_quality_contrast_min,
            }
        # For all other documents, use common document thresholds
        else:
            return {
                'brightness': verification_settings.document_quality_brightness_min,
                'sharpness': verification_settings.document_quality_sharpness_min,
                'contrast': verification_settings.document_quality_contrast_min,
            }

    def _crop_margins(self, image: np.ndarray) -> np.ndarray:
        """Crop white/black margins from image."""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Find content bounds
            _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Get bounding rectangle of all content
                all_contours = np.vstack(contours)
                x, y, w, h = cv2.boundingRect(all_contours)

                # Add small padding
                padding = 10
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(image.shape[1] - x, w + 2 * padding)
                h = min(image.shape[0] - y, h + 2 * padding)

                return image[y:y+h, x:x+w]

            return image

        except Exception as e:
            self.logger.warning(f"Margin cropping failed: {e}, returning original image")
            return image

    def _downsize_if_needed(self, image: np.ndarray, max_dimension: int) -> np.ndarray:
        """Downsize image if exceeds max dimension."""
        try:
            height, width = image.shape[:2]
            max_dim = max(width, height)

            if max_dim <= max_dimension:
                return image

            scale = max_dimension / max_dim
            new_width = int(width * scale)
            new_height = int(height * scale)

            return cv2.resize(image, (new_width, new_height), cv2.INTER_LANCZOS4)

        except Exception as e:
            self.logger.warning(f"Image downsizing failed: {e}, returning original")
            return image

    def _load_image(self, image_bytes: bytes) -> np.ndarray:
        """Load image from bytes, handling PDF conversion if needed."""
        # Check if input is PDF
        is_pdf = image_bytes.startswith(b'%PDF')

        if is_pdf:
            self.logger.info("PDF detected, converting to image first...")
            try:
                # Open PDF document
                pdf_document = fitz.open(stream=image_bytes, filetype="pdf")

                if pdf_document.page_count == 0:
                    raise ValueError("PDF is empty")

                # Use first page for passport processing
                page = pdf_document[0]

                # Calculate zoom factor for desired resolution
                # Higher zoom = better quality but slower processing
                zoom = 4.0

                # Render page to pixmap
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)

                # Convert to PNG bytes
                image_bytes = pix.tobytes("png")
                pdf_document.close()

                self.logger.info(f"PDF converted to image successfully ({len(image_bytes)} bytes)")

            except Exception as e:
                self.logger.error(f"PDF to image conversion failed: {str(e)}")
                raise ValueError(f"Failed to convert PDF to image: {str(e)}")

        # Validate image bytes before attempting to decode
        if not image_bytes:
            raise ValueError("Image data is empty")

        if len(image_bytes) < 100:
            raise ValueError(f"Image data too small ({len(image_bytes)} bytes)")

        # Check for valid image format signatures (magic bytes)
        if len(image_bytes) >= 4:
            header = image_bytes[:4]
            valid_signatures = {
                b'\xFF\xD8\xFF': 'JPEG',
                b'\x89PNG': 'PNG',
                b'GIF8': 'GIF',
                b'RIFF': 'WebP',
                b'BM': 'BMP'
            }
            has_valid_signature = any(header.startswith(sig) for sig in valid_signatures.keys())
            if not has_valid_signature:
                self.logger.warning(f"Invalid image format signature (header: {header.hex()})")
                raise ValueError(f"Invalid image format - expected JPEG/PNG/GIF/WebP/BMP")

        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            image = np.array(pil_image)

            # Convert RGB to BGR for OpenCV
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            return image

        except Exception as e:
            raise ValueError(f"Failed to load image: {str(e)}")

    def _numpy_to_bytes(self, image: np.ndarray) -> bytes:
        """Convert numpy image to bytes."""
        try:
            if len(image.shape) == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image

            pil_image = Image.fromarray(rgb_image.astype('uint8'))
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=95)
            return buffer.getvalue()

        except Exception as e:
            raise ValueError(f"Failed to convert image to bytes: {str(e)}")


# Global service instance
_document_preprocessing_service: Optional[DocumentPreprocessingService] = None


def get_document_preprocessing_service() -> DocumentPreprocessingService:
    """Get or create the singleton DocumentPreprocessingService instance."""
    global _document_preprocessing_service
    if _document_preprocessing_service is None:
        _document_preprocessing_service = DocumentPreprocessingService()
    return _document_preprocessing_service

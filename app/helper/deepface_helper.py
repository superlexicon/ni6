import base64
import asyncio
import io
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import cv2
from PIL import Image
# Use forked DeepFace with anti-spoofing return values
from app.deepface.deepface import DeepFace
import tensorflow as tf

from app.dto import DeepfaceResponse, FaceMatchResult, FaceVerificationRequest, FaceVerificationResponse
from app.core.logger import get_logger
from app.core.device_manager import device_manager, DeviceType
from app.core.gpu_manager import ModelType, get_gpu_manager
from app.core.framework_coordinator import get_framework_coordinator, FrameworkType
from app.utils.model_thresholds import get_adaptive_model_threshold
from app.utils.deepface_compatibility import ensure_deepface_compatibility
from app.utils.retinaface_compatibility import ensure_retinaface_compatibility

logger = get_logger()

# Framework coordinator disabled - DeepFace manages its own TensorFlow configuration
def _initialize_deepface_environment():
    """DeepFace manages its own TensorFlow configuration for maximum compatibility"""
    try:
        import tensorflow as tf

        # Force eager execution mode to prevent KerasTensor errors
        tf.config.run_functions_eagerly(True)
        logger.info(f"TensorFlow eager execution FORCED ON for DeepFace (version: {tf.__version__})")
        logger.info(f"TensorFlow executing eagerly: {tf.executing_eagerly()}")

        # Framework coordinator DISABLED - causes KerasTensor conflicts with DeepFace
        logger.info("DeepFace initialization: framework coordinator disabled to prevent KerasTensor conflicts")
        logger.info("DeepFace will manage its own TensorFlow configuration automatically")

        # NO framework coordinator - DeepFace configures TensorFlow optimally internally
        # This prevents external TensorFlow interference that causes KerasTensor errors

    except Exception as e:
        logger.error(f"DeepFace environment setup failed: {e}")

# Initialize DeepFace environment
_initialize_deepface_environment()


def convert_numpy_types(obj):
    """
    Convert numpy types and KerasTensors to native Python types for JSON serialization.
    This prevents KerasTensor leaks from DeepFace into PyTorch operations.

    Args:
        obj: Object that may contain numpy types or KerasTensors

    Returns:
        Object with numpy types and KerasTensors converted to native Python types
    """
    # Check for TensorFlow tensors first - this prevents TF tensors from leaking into PyTorch
    if hasattr(obj, 'numpy') and hasattr(obj, 'dtype'):
        tensor_type = str(type(obj)).lower()
        if 'keras' in tensor_type or 'tensorflow' in tensor_type or 'tf.' in tensor_type:
            # Convert TensorFlow/Keras tensor to numpy array, then to appropriate Python type
            try:
                obj = obj.numpy()
            except Exception:
                # If conversion fails, convert to float/int if possible
                try:
                    obj = float(obj)
                except (ValueError, TypeError):
                    obj = str(obj)

    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, np.ndarray):  # numpy arrays - check BEFORE hasattr
        if obj.size == 1:  # Single-element array
            return obj.item()
        else:  # Multi-element array
            return obj.tolist()
    elif hasattr(obj, 'item'):  # numpy scalars and other types with .item()
        return obj.item()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        return obj

# Constants
# ⚡ OPTIMIZATION: Use only VGG-Face for better performance - it's faster than Facenet512
SUPPORTED_MODELS = ["VGG-Face"]  # Reduced from 3 models to 1 for speed
DEFAULT_MODEL = "VGG-Face"  # Fast and reliable model for production
DEFAULT_SIMILARITY_THRESHOLD = 85.0  # User-preferred threshold
SIMILARITY_RANGE = 30.0
HIGH_CONFIDENCE_THRESHOLD = 90.0  # For passive liveness verification
PASSPORT_SELFIE_THRESHOLD = 85.0  # User-preferred threshold for passport-selfie matching
SELFIE_SELFIE_THRESHOLD = 85.0    # Consistent threshold for all selfie matching
FALLBACK_SIMILARITY_THRESHOLD = 85.0  # Same threshold for fallback method


class FaceVerificationError(Exception):
    pass


class DeepfaceHelper:
    """DeepFace helper class integrated with GPU resource manager."""

    _models_preloaded = False
    _preloaded_models = {}

    @classmethod
    async def get_model_with_gpu_management(cls, model_name: str = DEFAULT_MODEL) -> bool:
        """
        Get DeepFace model with GPU resource management.

        Args:
            model_name: DeepFace model name

        Returns:
            True if model is ready for use
        """
        try:
            gpu_manager = get_gpu_manager()

            # For DeepFace, we don't directly create model instances but ensure
            # GPU resources are properly allocated before DeepFace operations
            await gpu_manager.get_memory_usage()
            logger.debug(f"DeepFace GPU resources verified for model: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to prepare GPU resources for DeepFace model {model_name}: {e}")
            return False

    @classmethod
    async def release_deepface_resources(cls) -> bool:
        """
        Release DeepFace-specific GPU resources.

        Returns:
            True if resources were released successfully
        """
        try:
            # Clear TensorFlow session to free GPU memory
            # Use try-except to handle cases where TensorFlow might be in an invalid state
            try:
                tf.keras.backend.clear_session()
            except Exception as e:
                logger.debug(f"TensorFlow session already cleared or invalid: {e}")

            # Force garbage collection
            import gc
            gc.collect()

            logger.info("DeepFace GPU resources released")
            return True
        except Exception as e:
            logger.error(f"Failed to release DeepFace GPU resources: {e}")
            return False

    @classmethod
    async def preload_all_models(cls) -> bool:
        """
        Preload all DeepFace models during application startup to eliminate runtime download delays.

        Returns:
            True if models were preloaded successfully, False otherwise
        """
        if cls._models_preloaded:
            logger.info("DeepFace models already preloaded")
            return True

        try:
            logger.info("🔄 Preloading DeepFace models...")

            # Create dummy image for model warmup (224x224 RGB)
            dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)

            # Preload face detection models - use RetinaFace (faster than MTCNN)
            await asyncio.to_thread(
                DeepFace.extract_faces,
                dummy_image,
                detector_backend='retinaface',  # ⚡ OPTIMIZATION: Faster than MTCNN
                enforce_detection=False
            )
            logger.info("✅ Face detection models preloaded")

            # ⚡ OPTIMIZATION: Preload only the essential Facenet512 model
            model_names = ['Facenet512']
            for model_name in model_names:
                try:
                    await asyncio.to_thread(
                        DeepFace.verify,
                        dummy_image,
                        dummy_image,
                        model_name=model_name,
                        enforce_detection=False,
                        detector_backend='retinaface'  # ⚡ Faster detector
                    )
                    cls._preloaded_models[model_name] = True
                    logger.info(f"✅ DeepFace model preloaded: {model_name}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to preload {model_name}: {e}")
                    cls._preloaded_models[model_name] = False

            # Preload anti-spoofing model (FasNet) - used by DeepFace.represent with anti_spoofing=True
            try:
                await asyncio.to_thread(
                    DeepFace.represent,
                    dummy_image,
                    model_name='Facenet512',
                    detector_backend='retinaface',
                    anti_spoofing=True,
                    enforce_detection=False
                )
                logger.info("✅ Anti-spoofing model (FasNet) preloaded")
            except Exception as e:
                logger.warning(f"⚠️ Failed to preload anti-spoofing model: {e}")

            # Age analysis disabled for performance - skip preloading

            cls._models_preloaded = True
            logger.info("✅ All DeepFace models preloaded successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to preload DeepFace models: {e}")
            return False

    @staticmethod
    def calculate_similarity(verification_result: Dict[str, Any]) -> float:
        try:
            distance = verification_result["distance"]
            threshold = verification_result["threshold"]

            if distance >= threshold:
                return 0.0
            return round(
                DEFAULT_SIMILARITY_THRESHOLD +
                ((threshold - distance) / threshold) * SIMILARITY_RANGE,
                2
            )
        except KeyError as e:
            logger.error(f"Missing required key in verification result: {e}")
            return 0.0

    @classmethod
    def load_and_convert_image(cls, image_data: bytes) -> np.ndarray:
        try:
            with Image.open(io.BytesIO(image_data)) as img:
                return np.array(img.convert("RGB"))
        except Exception as e:
            error_msg = f"Failed to load image: {str(e)}"
            logger.error(error_msg)
            raise FaceVerificationError(error_msg) from e

    @classmethod
    async def verify_faces(
        cls,
        image1: bytes,
        image2: bytes,
        model_names: Optional[List[str]] = None,
        enforce_detection: bool = True,  # Fix: Enforce face detection by default
        adaptive_thresholds: Optional[Dict[str, float]] = None
    ) -> List[DeepfaceResponse]:
        model_names = model_names or SUPPORTED_MODELS
        results = []

        try:
            img1_array = cls.load_and_convert_image(image1)
            img2_array = cls.load_and_convert_image(image2)

            # Calculate image quality for adaptive thresholds if not provided
            if adaptive_thresholds is None:
                adaptive_thresholds = {}
                for model_name in model_names:
                    adaptive_thresholds[model_name] = get_adaptive_model_threshold(model_name)

            for model_name in model_names:
                try:
                    verification_result = await asyncio.to_thread(
                        DeepFace.verify,
                        img1_path=img1_array,
                        img2_path=img2_array,
                        model_name=model_name,
                        enforce_detection=enforce_detection,
                        detector_backend="retinaface"  # ⚡ OPTIMIZATION: Faster than MTCNN
                    )

                    # Convert numpy types to native Python types
                    verification_result = convert_numpy_types(verification_result)

                    # Apply adaptive threshold for confidence-weighted verification
                    adaptive_threshold = adaptive_thresholds.get(model_name, 0.31)  # VGG-Face default
                    distance = verification_result.get("distance", 0.0)

                    # Determine verification using adaptive threshold
                    verified_adaptive = distance <= adaptive_threshold
                    original_verified = bool(verification_result["verified"])

                    response = DeepfaceResponse(
                        verify=verified_adaptive,  # Use adaptive threshold
                        percentage=cls.calculate_similarity(verification_result),
                        model=model_name
                    )

                    # Add adaptive threshold info to the response for debugging
                    response.adaptive_threshold = adaptive_threshold
                    response.distance = distance
                    response.original_verified = original_verified

                    results.append(response)

                except Exception as e:
                    logger.error(f"Error processing model {model_name}: {str(e)}")
                    continue

            return results

        except Exception as e:
            logger.error(f"Face verification failed: {str(e)}")
            raise FaceVerificationError(f"Face verification failed: {str(e)}") from e

    @classmethod
    async def verify(
        cls,
        recovery_image: bytes,
        passport_image: bytes,
        model_names: Optional[List[str]] = None,
        min_confidence: float = 70.0
    ) -> bool:
        try:
            results = await cls.verify_faces(
                recovery_image,
                passport_image,
                model_names=model_names
            )

            if not results:
                logger.warning("No verification results were returned")
                return False

            all_passed = all(
                result.verify and result.percentage >= min_confidence
                for result in results
            )
            return all_passed

        except Exception as e:
            logger.error(f"Verification failed: {str(e)}")
            return False

    @classmethod
    async def extract_face_embedding(
        cls,
        image_bytes: bytes,
        model_name: str = DEFAULT_MODEL,
        enforce_detection: bool = True
    ) -> Optional[List[float]]:
        """
        Extract face embedding from image.

        Args:
            image_bytes: Image bytes data
            model_name: DeepFace model to use
            enforce_detection: Whether to enforce face detection

        Returns:
            Face embedding vector or None if extraction failed
        """
        try:
            # Compatibility is handled at import time only - eliminates multiple applications
            img_array = cls.load_and_convert_image(image_bytes)

            representation = await asyncio.to_thread(
                DeepFace.represent,
                img_path=img_array,
                model_name=model_name,
                enforce_detection=enforce_detection,
                detector_backend="retinaface",  # ⚡ OPTIMIZATION: Faster than MTCNN
                align=True
            )

            # Convert numpy types to native Python types
            representation = convert_numpy_types(representation)

            if not representation or len(representation) == 0:
                logger.warning("No face detected in image")
                return None

            # Fix silent failure - check if embedding exists instead of returning empty list
            if 'embedding' not in representation[0] or not representation[0]['embedding']:
                logger.error("❌ DeepFace returned representation but no embedding data - invalid face detection")
                return None

            embedding = representation[0]['embedding']
            face_confidence = representation[0].get('face_confidence', 0.0)

            # Validate embedding quality immediately after extraction
            if embedding and len(embedding) > 0:
                if all(abs(x) < 1e-10 for x in embedding):
                    logger.error("❌ Extracted embedding contains only zero values - face detection failed")
                    return None
            else:
                logger.error("❌ Empty embedding returned from DeepFace")
                return None

            logger.info(f"✅ Extracted valid face embedding: len={len(embedding)}, confidence={face_confidence:.3f}")
            return embedding

        except Exception as e:
            logger.error(f"Face embedding extraction failed: {str(e)}")
            return None

    @classmethod
    async def compare_face_embeddings(
        cls,
        embedding1: List[float],
        embedding2: List[float],
        model_name: str = DEFAULT_MODEL,
        threshold: float = None,
        use_ensemble: bool = False  # Disable ensemble - eliminates KerasTensor conflicts
    ) -> FaceMatchResult:
        """
        Compare two face embeddings directly using single model verification.

        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            model_name: Primary model name for threshold determination (VGG-Face only)
            threshold: Optional custom threshold (None = use DeepFace native threshold)
            use_ensemble: Whether to use multiple models for verification (disabled - eliminates KerasTensor conflicts)

        Returns:
            FaceMatchResult with detailed comparison results including per-model distances
        """
        try:
            if len(embedding1) != len(embedding2):
                raise ValueError("Embeddings must have same dimension")

            if use_ensemble:
                # Use ensemble verification with multiple models
                return await cls._ensemble_face_matching(embedding1, embedding2, model_name, threshold)
            else:
                # Single model verification (original logic)
                return await cls._single_model_face_matching(embedding1, embedding2, model_name, threshold)

        except Exception as e:
            logger.error(f"Face embedding comparison failed: {str(e)}")
            # Use appropriate default threshold if not provided (research-backed thresholds)
            if threshold is None:
                from app.utils.model_thresholds import get_model_threshold
                threshold = get_model_threshold(model_name)

            return FaceMatchResult(
                distance_score=1.0,  # Maximum distance indicates complete mismatch
                match_confidence=0.0,
                is_match=False,
                threshold_used=threshold,
                distance_metric="cosine"
            )

    @classmethod
    async def _single_model_face_matching(
        cls,
        embedding1: List[float],
        embedding2: List[float],
        model_name: str,
        threshold: float
    ) -> FaceMatchResult:
        """Single model face matching logic."""
        # Calculate distance using cosine similarity
        embedding1_array = np.array(embedding1)
        embedding2_array = np.array(embedding2)

        # Cosine distance calculation
        dot_product = np.dot(embedding1_array, embedding2_array)
        norm1 = np.linalg.norm(embedding1_array)
        norm2 = np.linalg.norm(embedding2_array)

        if norm1 == 0 or norm2 == 0:
            if norm1 == 0 and norm2 == 0:
                raise ValueError(f"Both embeddings have zero magnitude - invalid embeddings detected")
            elif norm1 == 0:
                raise ValueError(f"First embedding has zero magnitude - invalid embedding")
            else:
                raise ValueError(f"Second embedding has zero magnitude - invalid embedding")

        cosine_similarity = dot_product / (norm1 * norm2)
        distance = 1 - cosine_similarity

        # Use model-specific threshold if not provided
        if threshold is None:
            from app.utils.model_thresholds import get_model_threshold
            threshold = get_model_threshold(model_name)
            logger.info(f"🎯 Using research-backed threshold for {model_name}: {threshold}")
        else:
            logger.info(f"📋 Using custom threshold for {model_name}: {threshold}")

        # Determine if match based on distance threshold
        is_match = distance < threshold
        # Use cosine similarity as confidence (0 = identical, 1 = completely different)
        # This matches the approach in verify_faces_with_models() at line 941
        match_confidence = max(0.0, 1.0 - distance)

        return FaceMatchResult(
            distance_score=round(distance, 4),
            match_confidence=round(match_confidence, 3),
            is_match=is_match,
            threshold_used=threshold,
            distance_metric="cosine"
        )

    @classmethod
    async def _ensemble_face_matching(
        cls,
        embedding1: List[float],
        embedding2: List[float],
        primary_model: str,
        custom_threshold: float
    ) -> FaceMatchResult:
        """Ensemble face matching using multiple models for enhanced accuracy."""

        # Single model for reliability (eliminates ensemble conflicts)
        ensemble_models = ["VGG-Face"]

        model_results = []

        for model_name in ensemble_models:
            try:
                # Get model-specific threshold
                if custom_threshold is None:
                    from app.utils.model_thresholds import get_model_threshold
                    threshold = get_model_threshold(model_name)
                else:
                    threshold = custom_threshold

                # Calculate distance for this model
                distance = await cls._calculate_model_distance(embedding1, embedding2, model_name)
                is_match = distance < threshold
                # Use cosine similarity as confidence
                confidence = max(0.0, 1.0 - distance)

                model_results.append({
                    "model": model_name,
                    "distance": distance,
                    "threshold": threshold,
                    "is_match": is_match,
                    "confidence": confidence
                })

                logger.info(f"🤖 {model_name} verification: distance={distance:.4f}, match={is_match}, threshold={threshold}")

            except Exception as e:
                logger.warning(f"⚠️  {model_name} verification failed: {str(e)}")
                continue

        if not model_results:
            logger.error("❌ All models failed for ensemble verification")
            from app.utils.model_thresholds import get_model_threshold
            fallback_threshold = get_model_threshold(primary_model)
            return FaceMatchResult(
                distance_score=1.0,
                match_confidence=0.0,
                is_match=False,
                threshold_used=fallback_threshold,
                distance_metric="cosine"
            )

        # Ensemble decision logic
        matches = sum(1 for r in model_results if r["is_match"])
        total_models = len(model_results)

        # Conservative approach: require majority agreement for match
        is_ensemble_match = matches >= (total_models / 2)

        # Weighted confidence based on individual model confidences
        avg_confidence = sum(r["confidence"] for r in model_results) / total_models

        # Use the most conservative (highest) distance from matching models
        if is_ensemble_match:
            matching_distances = [r["distance"] for r in model_results if r["is_match"]]
            consensus_distance = max(matching_distances) if matching_distances else max(r["distance"] for r in model_results)
        else:
            # For non-matches, use the average distance
            consensus_distance = sum(r["distance"] for r in model_results) / total_models

        # Use the threshold of the primary model
        from app.utils.model_thresholds import get_model_threshold
        primary_threshold = custom_threshold or get_model_threshold(primary_model)

        logger.info(f"🎯 Ensemble decision: {matches}/{total_models} models matched, "
                   f"final_match={is_ensemble_match}, consensus_distance={consensus_distance:.4f}")

        return FaceMatchResult(
            distance_score=round(consensus_distance, 4),
            match_confidence=round(avg_confidence, 3),
            is_match=is_ensemble_match,
            threshold_used=primary_threshold,
            distance_metric="cosine_ensemble"
        )

    @classmethod
    async def _calculate_model_distance(
        cls,
        embedding1: List[float],
        embedding2: List[float],
        model_name: str
    ) -> float:
        """Calculate cosine distance for a specific model."""

        # Convert to numpy arrays
        embedding1_array = np.array(embedding1)
        embedding2_array = np.array(embedding2)

        # Calculate cosine similarity
        dot_product = np.dot(embedding1_array, embedding2_array)
        norm1 = np.linalg.norm(embedding1_array)
        norm2 = np.linalg.norm(embedding2_array)

        if norm1 == 0 or norm2 == 0:
            raise ValueError(f"Zero magnitude embedding detected for {model_name}")

        cosine_similarity = dot_product / (norm1 * norm2)

        # Convert to distance (0 = identical, 1 = completely different)
        return 1 - cosine_similarity

    @classmethod
    async def verify_face_with_embedding(
        cls,
        stored_embedding: List[float],
        verification_image: bytes,
        threshold: float = PASSPORT_SELFIE_THRESHOLD,
        model_name: str = DEFAULT_MODEL,
        use_ensemble: bool = False  # Disable ensemble - eliminates KerasTensor conflicts
    ) -> FaceMatchResult:
        """
        Verify face by comparing stored embedding with new image.

        Args:
            stored_embedding: Previously stored face embedding
            verification_image: New image bytes for verification
            threshold: Matching threshold (0-100)
            model_name: DeepFace model to use

        Returns:
            FaceMatchResult with verification details
        """
        try:
            # Extract embedding from verification image
            verification_embedding = await cls.extract_face_embedding(
                verification_image,
                model_name=model_name,
                enforce_detection=True
            )

            if not verification_embedding:
                logger.error("Failed to extract face from verification image")
                return FaceMatchResult(
                    distance_score=1.0,  # Maximum distance indicates complete mismatch
                    match_confidence=0.0,
                    is_match=False,
                    threshold_used=threshold,
                    distance_metric="cosine"
                )

            # Compare embeddings
            match_result = await cls.compare_face_embeddings(
                stored_embedding,
                verification_embedding,
                model_name=model_name,
                threshold=threshold,
                use_ensemble=use_ensemble
            )

            # Apply additional threshold check (now distance-based: lower distance = more similar)
            final_is_match = match_result.is_match and match_result.distance_score <= threshold

            logger.info(f"Face verification: distance={match_result.distance_score:.4f}, "
                       f"match={final_is_match}, threshold={threshold}")

            return FaceMatchResult(
                distance_score=match_result.distance_score,
                match_confidence=match_result.match_confidence,
                is_match=final_is_match,
                threshold_used=threshold,
                distance_metric=match_result.distance_metric
            )

        except Exception as e:
            logger.error(f"Face verification with embedding failed: {str(e)}")
            raise FaceVerificationError(f"Face verification failed: {str(e)}") from e

    @classmethod
    async def verify_selfie_against_records(
        cls,
        selfie_image: bytes,
        stored_embedding: List[float],
        verification_request: FaceVerificationRequest
    ) -> FaceVerificationResponse:
        """
        Comprehensive selfie verification against stored biometric records.

        Args:
            selfie_image: Selfie image bytes with OTP overlay
            stored_embedding: Previously stored face embedding (passport or previous selfie)
            verification_request: Verification request parameters

        Returns:
            Comprehensive verification response
        """
        try:
            logger.info(f"Starting selfie verification for public_key: {verification_request.public_key[:8]}...")

            # Step 1: Face matching
            face_match_result = await cls.verify_face_with_embedding(
                stored_embedding=stored_embedding,
                verification_image=selfie_image,
                threshold=verification_request.verification_threshold,
                model_name=DEFAULT_MODEL
            )

            # Step 2: Basic face quality check (for selfies)
            quality_passed = await cls._check_selfie_quality(selfie_image)

            # Step 3: Liveness check (basic anti-spoofing for selfies)
            liveness_passed = not verification_request.include_liveness_check
            anti_spoof_score = None

            if verification_request.include_liveness_check:
                liveness_result = await cls._perform_selfie_anti_spoofing(selfie_image)
                anti_spoof_score = liveness_result.get('anti_spoof_score', 0.0)
                liveness_passed = anti_spoof_score >= verification_request.anti_spoof_threshold

            # Overall verification decision
            verification_passed = (
                face_match_result.is_match and
                quality_passed and
                liveness_passed
            )

            verification_details = {
                "face_matching": {
                    "distance_score": face_match_result.distance_score,
                    "threshold_used": face_match_result.threshold_used,
                    "passed": face_match_result.is_match
                },
                "quality_check": {
                    "passed": quality_passed,
                    "details": "Basic face quality assessment"
                },
                "liveness_check": {
                    "passed": liveness_passed,
                    "anti_spoof_score": anti_spoof_score,
                    "threshold": verification_request.anti_spoof_threshold if verification_request.include_liveness_check else None
                }
            }

            logger.info(f"Selfie verification completed: passed={verification_passed}, "
                       f"distance={face_match_result.distance_score:.4f}")

            return FaceVerificationResponse(
                verification_passed=verification_passed,
                face_match_result=face_match_result,
                liveness_check_passed=liveness_passed if verification_request.include_liveness_check else None,
                anti_spoof_score=anti_spoof_score,
                verification_details=verification_details
            )

        except Exception as e:
            logger.error(f"Selfie verification failed: {str(e)}")
            raise FaceVerificationError(f"Selfie verification failed: {str(e)}") from e

    @classmethod
    async def _check_selfie_quality(cls, image_bytes: bytes) -> bool:
        """
        Basic quality check for selfie images.

        Args:
            image_bytes: Image bytes to check

        Returns:
            True if quality passes basic checks
        """
        try:
            img_array = cls.load_and_convert_image(image_bytes)

            # Check image dimensions
            height, width = img_array.shape[:2]
            if width < 200 or height < 200:
                logger.warning("Selfie image too small for reliable verification")
                return False

            # Check for face detection
            embedding = await cls.extract_face_embedding(image_bytes, enforce_detection=True)
            if not embedding:
                logger.warning("No face detected in selfie")
                return False

            return True

        except Exception as e:
            logger.error(f"Selfie quality check failed: {str(e)}")
            return False

    @classmethod
    async def _perform_selfie_anti_spoofing(cls, image_bytes: bytes) -> Dict[str, float]:
        """
        Basic anti-spoofing check for selfie images.

        Args:
            image_bytes: Selfie image bytes

        Returns:
            Dict with anti-spoofing scores
        """
        try:
            # For now, implement basic quality-based anti-spoofing
            # In production, this would use dedicated anti-spoofing models
            img_array = cls.load_and_convert_image(image_bytes)

            # Basic image quality metrics
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array

            # Calculate blurriness (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_score = min(1.0, laplacian_var / 500.0)

            # Calculate brightness uniformity (check for overly uniform illumination)
            brightness_std = np.std(gray) / 255.0
            uniformity_score = 1.0 - min(1.0, brightness_std * 2)  # Lower std = more uniform

            # Overall anti-spoof score (basic heuristic)
            anti_spoof_score = (blur_score * 0.6) + (uniformity_score * 0.4)

            return {
                "anti_spoof_score": round(anti_spoof_score, 3),
                "blur_score": round(blur_score, 3),
                "uniformity_score": round(uniformity_score, 3)
            }

        except Exception as e:
            logger.error(f"Selfie anti-spoofing check failed: {str(e)}")
            return {
                "anti_spoof_score": 0.5,
                "blur_score": 0.5,
                "uniformity_score": 0.5
            }

    @classmethod
    async def verify_faces_with_models(
        cls,
        image1_b64: str,
        image2_b64: str,
        models: Optional[List[str]] = None,
        detector_backend: str = "retinaface"  # ⚡ OPTIMIZATION: Faster than MTCNN
    ) -> Dict[str, Any]:
        """
        Use DeepFace.verify() with single VGG-Face model for reliable face verification.

        Args:
            image1_b64: Base64-encoded face image 1 (selfie)
            image2_b64: Base64-encoded face image 2 (passport)
            models: List of models to use for verification (default: VGG-Face only - eliminates conflicts)
            detector_backend: Face detector backend (default: mtcnn)

        Returns:
            Dict with ensemble result:
            {
                "ensemble_result": {
                    "verified": bool,
                    "confidence": float,
                    "model": str,
                    "distance": float,
                    "threshold": float,
                    "votes": str  # e.g. "1/1"
                },
                "per_model_results": {
                    "VGG-Face": { ... }
                }
            }
        """
        try:
            # Convert base64 strings back to bytes for DeepFace
            image1_bytes = cls._base64_to_bytes(image1_b64)
            image2_bytes = cls._base64_to_bytes(image2_b64)

            # Load and convert images to numpy arrays
            img1_array = cls.load_and_convert_image(image1_bytes)
            img2_array = cls.load_and_convert_image(image2_bytes)

            # Default to single VGG-Face model (eliminates ensemble conflicts)
            if models is None:
                models = ["VGG-Face"]

            per_model_results = {}
            model_votes = []

            logger.info(f"🔍 Starting DeepFace native verification with models: {models}")

            for model_name in models:
                try:
                    # Use DeepFace.verify() with native model-specific processing
                    verification_result = await asyncio.to_thread(
                        DeepFace.verify,
                        img1_path=img1_array,
                        img2_path=img2_array,
                        model_name=model_name,
                        enforce_detection=True,
                        detector_backend=detector_backend,
                        align=True  # Enable face alignment for better accuracy
                    )

                    # Convert numpy types to native Python types
                    verification_result = convert_numpy_types(verification_result)

                    # Extract verification details
                    distance = verification_result.get("distance", 1.0)
                    threshold = verification_result.get("threshold", 0.4)
                    verified = verification_result.get("verified", False)

                    # Calculate confidence from distance (DeepFace.verify() doesn't return confidence)
                    # For cosine distance: 0 = identical, 1 = completely different
                    # Convert to similarity percentage: similarity = 1 - distance
                    similarity = 1 - distance
                    # Scale to percentage (0-100%)
                    confidence = similarity * 100

                    model_result = {
                        "verified": verified,
                        "distance": round(distance, 4),
                        "threshold": round(threshold, 4),
                        "confidence": round(confidence, 3),
                        "model": model_name,
                        "detector_backend": detector_backend
                    }

                    per_model_results[model_name] = model_result
                    model_votes.append(verified)

                    logger.info(f"🤖 DeepFace {model_name}: verified={verified}, "
                               f"distance={distance:.4f}, threshold={threshold:.4f}")

                except Exception as e:
                    logger.warning(f"⚠️  DeepFace {model_name} verification failed: {str(e)}")
                    # Add failed result to maintain consistency
                    per_model_results[model_name] = {
                        "verified": False,
                        "distance": 1.0,
                        "threshold": 0.4,
                        "confidence": 0.0,
                        "model": model_name,
                        "error": str(e)
                    }
                    model_votes.append(False)

            # Ensemble decision logic
            total_models = len(models)
            affirmative_votes = sum(model_votes)

            # Conservative approach: require majority agreement for verification
            ensemble_verified = affirmative_votes >= (total_models / 2 + 0.1)  # Strict majority

            # Calculate ensemble confidence and select representative model
            if ensemble_verified:
                # Use the best (highest confidence) matching model as representative
                matching_models = [result for result in per_model_results.values()
                                 if result.get("verified", False)]
                if matching_models:
                    best_model = max(matching_models, key=lambda x: x["confidence"])
                    ensemble_confidence = best_model["confidence"]
                    ensemble_distance = best_model["distance"]
                    ensemble_threshold = best_model["threshold"]
                    representative_model = best_model["model"]
                else:
                    # Fallback (shouldn't happen if ensemble_verified is True)
                    ensemble_confidence = 0.5
                    ensemble_distance = 0.4
                    ensemble_threshold = 0.4
                    representative_model = models[0]
            else:
                # For non-matches, use average confidence and distance
                confidences = [result.get("confidence", 0.0) for result in per_model_results.values()]
                distances = [result.get("distance", 1.0) for result in per_model_results.values()]
                thresholds = [result.get("threshold", 0.4) for result in per_model_results.values()]

                ensemble_confidence = sum(confidences) / total_models
                ensemble_distance = sum(distances) / total_models
                ensemble_threshold = sum(thresholds) / total_models
                representative_model = models[0]

            # Log ensemble decision
            logger.info(f"🎯 Ensemble decision: {affirmative_votes}/{total_models} models verified, "
                       f"final_result={ensemble_verified}, confidence={ensemble_confidence:.3f}")

            return {
                "ensemble_result": {
                    "verified": ensemble_verified,
                    "confidence": round(ensemble_confidence, 3),
                    "model": representative_model,
                    "distance": round(ensemble_distance, 4),
                    "threshold": round(ensemble_threshold, 4),
                    "votes": f"{affirmative_votes}/{total_models}"
                },
                "per_model_results": per_model_results
            }

        except Exception as e:
            logger.error(f"DeepFace native verification failed: {str(e)}")
            # Return failure result
            return {
                "ensemble_result": {
                    "verified": False,
                    "confidence": 0.0,
                    "model": "error",
                    "distance": 1.0,
                    "threshold": 0.4,
                    "error": str(e)
                },
                "per_model_results": {}
            }

    @classmethod
    def _base64_to_bytes(cls, base64_string: str) -> bytes:
        """
        Convert base64 string back to image bytes.

        Args:
            base64_string: Base64-encoded image string

        Returns:
            Image bytes
        """
        try:
            # Decode base64 string to bytes
            image_bytes = base64.b64decode(base64_string)
            return image_bytes
        except Exception as e:
            raise FaceVerificationError(f"Failed to decode base64 image: {str(e)}") from e

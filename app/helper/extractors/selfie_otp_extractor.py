import re
import asyncio
import base64
from typing import Optional, Dict, Any, Tuple
import cv2
import numpy as np
from PIL import Image
import io
import tensorflow as tf
# Use forked DeepFace with anti-spoofing return values
from app.deepface.deepface import DeepFace

from app.schemas.selfie_otp_schema import SelfieOTPData
from app.dto import FaceMetadata, FaceQuality, FaceLiveness
from app.helper.doctr.document_text_extractor import DocumentTextExtractor
from app.helper.deepface_helper import convert_numpy_types
from app.utils.deepface_compatibility import ensure_deepface_compatibility
from app.core import logger


class SelfieOTPExtractor:
    """Extract OTP and face information from selfie documents with comprehensive anti-spoofing"""

    def __init__(self):
        self.logger = logger
        self.text_extractor = DocumentTextExtractor()

    async def extract(self, content: bytes, is_pdf: bool = False, filename: Optional[str] = None) -> SelfieOTPData:
        """
        Extract OTP number and face detection data from selfie with enhanced anti-spoofing.

        Args:
            content: Image bytes
            is_pdf: Whether content is PDF (selfies are typically images)
            filename: Original filename (used to extract OTP if not found in image)

        Returns:
            SelfieOTPData with OTP and comprehensive face information
        """
        try:
            self.logger.info("Starting enhanced selfie OTP extraction with anti-spoofing")
            selfie_data = SelfieOTPData()

            # Step 1: Extract OTP number using enhanced OCR with confidence
            text_blocks = await self.text_extractor.extract_text_with_geometry(content, is_pdf=is_pdf)
            text = " ".join([block.get('text', '') for block in text_blocks])
            self.logger.info(f"OCR extracted text: '{text}' ({len(text)} chars)")
            otp_result = self._extract_otp_with_confidence(text, text_blocks)
            self.logger.debug(f"OCR OTP result: {otp_result}")

            if otp_result['otp']:
                selfie_data.otp_number = otp_result['otp']
                selfie_data.otp_detected = True
                selfie_data.otp_length = len(otp_result['otp'])
                selfie_data.confidence_scores['otp_number'] = otp_result['confidence']
                self.logger.debug(f"OTP found in image: {otp_result['otp']} (confidence: {otp_result['confidence']}%)")
            else:
                # Fallback: Try extracting OTP from filename
                self.logger.info(f"No OTP found in image. Checking filename: {filename}")
                if filename:
                    otp_from_filename = self._extract_otp_from_filename(filename)
                    if otp_from_filename:
                        selfie_data.otp_number = otp_from_filename
                        selfie_data.otp_detected = True
                        selfie_data.otp_length = len(otp_from_filename)
                        selfie_data.confidence_scores['otp_number'] = 100.0  # Filename is 100% accurate
                        self.logger.debug(f"✅ OTP extracted from filename: {otp_from_filename}")
                    else:
                        self.logger.warning(f"No OTP found in filename: {filename}")
                else:
                    self.logger.warning("No filename provided for OTP fallback extraction")

            # Step 2: Extract face analysis (age, gender, anti-spoofing) - no embeddings needed
            face_analysis = await self._analyze_face_with_deepface_anti_spoofing(content)

            if face_analysis:
                selfie_data.face_detected = face_analysis['face_detected']
                selfie_data.face_count = face_analysis['face_count']
                selfie_data.face_confidence = face_analysis['face_confidence']
                selfie_data.live_person_detected = face_analysis['live_person_detected']
                selfie_data.liveness_confidence = face_analysis['liveness_confidence']
                selfie_data.anti_spoofing_score = face_analysis['anti_spoofing_score']
                selfie_data.spoof_detection_passed = face_analysis['spoof_detection_passed']

                # Add age analysis results
                selfie_data.estimated_age = face_analysis.get('estimated_age')
                selfie_data.age_confidence = face_analysis.get('age_confidence')

                # Add cropped face image for verification
                if 'face_image_b64' in face_analysis:
                    selfie_data.face_image_b64 = face_analysis['face_image_b64']
                    self.logger.info(f"✅ Added cropped face image to selfie data: {len(face_analysis['face_image_b64'])} characters")

                # Note: face_embedding is not stored since we use base64 images for verification

            # Calculate overall confidence
            overall_confidence = selfie_data.calculate_overall_confidence()
            selfie_data.overall_confidence = overall_confidence

            self.logger.debug(f"Selfie OTP extraction completed: OTP detected={selfie_data.otp_detected}, "
                           f"OTP number={selfie_data.otp_number}, "
                           f"Face detected={selfie_data.face_detected}, "
                           f"Anti-spoof score={selfie_data.anti_spoofing_score:.2f}")

            return selfie_data

        except Exception as e:
            self.logger.error(f"Enhanced selfie OTP extraction failed: {str(e)}")
            return SelfieOTPData(confidence_scores={})

    def _extract_otp_with_confidence(self, text: str, text_blocks: list) -> Dict[str, Any]:
        """
        Enhanced OTP extraction using OCR confidence scores from DocTR.

        Args:
            text: OCR extracted text
            text_blocks: OCR text blocks with confidence scores from DocTR

        Returns:
            Dict with OTP and confidence score based on actual OCR confidence
        """
        try:
            # Look for 6-digit OTP numbers
            otp_patterns = [
                r'\b([0-9]{6})\b',  # Standard 6-digit
                r'OTP[:\s]*([0-9]{6})',  # OTP: 123456
                r'Verification[:\s]*([0-9]{6})',  # Verification: 123456
                r'Code[:\s]*([0-9]{6})',  # Code: 123456
            ]

            matches = []
            for pattern in otp_patterns:
                pattern_matches = re.findall(pattern, text, re.IGNORECASE)
                for match in pattern_matches:
                    # Calculate OCR-based confidence for this OTP match
                    ocr_confidence = self._calculate_otp_confidence(match, text_blocks)

                    # Apply slight boost based on pattern quality
                    context_boost = 0.0
                    if 'OTP' in pattern.upper() or 'Verification' in pattern:
                        context_boost = 5.0  # Small boost for clear OTP context

                    final_confidence = min(95.0, ocr_confidence + context_boost)

                    matches.append({
                        'otp': match,
                        'confidence': final_confidence
                    })

            if matches:
                # Return the highest confidence match
                best_match = max(matches, key=lambda x: x['confidence'])
                return {
                    'otp': best_match['otp'],
                    'confidence': best_match['confidence']
                }

            # Fallback: look for any 6-digit number
            fallback_pattern = r'\b([0-9]{6})\b'
            fallback_matches = re.findall(fallback_pattern, text)
            if fallback_matches:
                # Use OCR confidence even for fallback matches
                fallback_confidence = self._calculate_otp_confidence(fallback_matches[0], text_blocks)
                return {
                    'otp': fallback_matches[0],
                    'confidence': fallback_confidence
                }

            # No OTP found in image - try metadata fallback
            return {'otp': None, 'confidence': 0.0}

        except Exception as e:
            self.logger.error(f"OTP extraction with OCR confidence failed: {str(e)}")
            return {'otp': None, 'confidence': 0.0}

    def _extract_otp_from_filename(self, filename: str) -> Optional[str]:
        """
        Extract OTP from filename by finding 6 characters after "OTP" substring.

        Args:
            filename: Original filename

        Returns:
            OTP string if found in filename, None otherwise
        """
        try:
            # Search for "OTP" (case-insensitive) followed by 6 digits
            # Pattern: OTP followed by exactly 6 digits
            match = re.search(r'OTP(\d{6})', filename, re.IGNORECASE)
            if match:
                otp = match.group(1)
                self.logger.debug(f"OTP found in filename '{filename}': {otp}")
                return otp

            return None
        except Exception as e:
            self.logger.warning(f"Failed to extract OTP from filename: {e}")
            return None

    def _calculate_otp_confidence(self, otp_value: str, text_blocks: list) -> float:
        """
        Calculate OCR confidence score for OTP value using ONLY DocTR confidence data.

        Args:
            otp_value: The OTP value (6-digit string)
            text_blocks: OCR text blocks with confidence scores from DocTR

        Returns:
            float: Confidence score (0-100) from OCR data only
        """
        if not otp_value or not text_blocks:
            self.logger.warning("No OTP value or text blocks available for confidence calculation")
            return 0.0  # NO HARDCODED DEFAULT - return 0 when no OCR data

        # Create a map of text to OCR confidence
        text_confidence_map = {}
        for block in text_blocks:
            text = block.get('text', '').strip()
            confidence = block.get('confidence', 0.0)  # Default to 0, not hardcoded 0.9
            text_confidence_map[text] = confidence * 100  # Convert to percentage

        # Direct text match - highest confidence
        if otp_value in text_confidence_map:
            return text_confidence_map[otp_value]

        # Find OTP within larger text blocks
        for text, confidence in text_confidence_map.items():
            if otp_value in text:
                return confidence

        # Fuzzy matching with OCR-aware penalty
        from rapidfuzz import fuzz, process
        result = process.extractOne(otp_value, list(text_confidence_map.keys()), scorer=fuzz.ratio)
        if result and result[1] >= 90:  # High threshold for OTP matching
            matched_text = result[0]
            confidence = text_confidence_map[matched_text]
            self.logger.debug(f"OCR fuzzy match: '{otp_value}' → '{matched_text}' ({result[1]:.1f}% similar)")
            return confidence * 0.95  # Minimal penalty for OCR variation

        # No OCR match found - return 0 to indicate no confidence
        self.logger.warning(f"OTP '{otp_value}' not found in OCR text blocks")
        return 0.0  # NO HARDCODED 70.0 FALLBACK

    async def _analyze_face_with_deepface_anti_spoofing(self, image_content: bytes) -> Optional[Dict[str, Any]]:
        """
        Face analysis using single DeepFace.represent call with anti_spoofing=True.
        Uses forked DeepFace that returns is_real and antispoof_score instead of throwing exception.

        Args:
            image_content: Image bytes

        Returns:
            Dict with detailed face analysis results including anti-spoofing and embedding
        """
        try:
            self.logger.info("Starting face analysis with anti-spoofing (single represent call)")

            # Single image load
            img_array = self._load_image_array(image_content)

            try:
                # Force eager execution right before DeepFace call
                tf.config.run_functions_eagerly(True)

                # Single DeepFace.represent call with anti_spoofing=True
                # Forked version returns is_real and antispoof_score instead of throwing exception
                self.logger.info("DeepFace.represent with anti_spoofing=True, detector_backend=retinaface")

                represent_results = await asyncio.to_thread(
                    DeepFace.represent,
                    img_path=img_array,
                    model_name='Facenet512',
                    detector_backend='retinaface',
                    anti_spoofing=True,
                    enforce_detection=True
                )

                # Clear TensorFlow session after DeepFace call
                tf.keras.backend.clear_session()

                if not represent_results or len(represent_results) == 0:
                    self.logger.warning("No results from DeepFace.represent")
                    return {
                        'face_detected': False,
                        'face_count': 0,
                        'face_confidence': 0.0,
                        'live_person_detected': False,
                        'liveness_confidence': 0.0,
                        'anti_spoofing_score': 0.0,
                        'spoof_detection_passed': False,
                        'estimated_age': None,
                        'age_confidence': 0.0
                    }

                represent_result = represent_results[0]
                represent_result = convert_numpy_types(represent_result)

                self.logger.info(f"DeepFace.represent result keys: {list(represent_result.keys())}")

                # Extract embedding
                face_embedding = represent_result.get('embedding', [])
                if face_embedding and len(face_embedding) > 0:
                    if all(v == 0 for v in face_embedding[:10]):
                        self.logger.warning("⚠️ Face embedding appears to be all zeros - invalid")
                        face_embedding = None
                    else:
                        self.logger.info(f"✅ Extracted face embedding: {len(face_embedding)} dimensions")
                else:
                    self.logger.warning("⚠️ No face embedding returned from DeepFace.represent")
                    face_embedding = None

                # Extract anti-spoofing results from forked DeepFace (no hardcoded defaults)
                is_real = represent_result.get('is_real', True)
                antispoof_score = represent_result.get('antispoof_score', 0.0)

                self.logger.info(f"Anti-spoofing: is_real={is_real}, antispoof_score={antispoof_score}")

                # Get facial region for cropping
                region = represent_result.get('facial_area', {})
                face_confidence = represent_result.get('face_confidence', 0.95)

                x = region.get('x', 0)
                y = region.get('y', 0)
                w = region.get('w', 0)
                h = region.get('h', 0)

                if w <= 0 or h <= 0:
                    self.logger.warning(f"Invalid region dimensions: w={w}, h={h}, using full image")
                    face_image_b64 = self._convert_cropped_face_to_base64(img_array)
                else:
                    # Crop the original image to the face region
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    cropped_face = img_bgr[y:y+h, x:x+w]
                    cropped_face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
                    face_image_b64 = self._convert_cropped_face_to_base64(cropped_face_rgb)
                    self.logger.info(f"✅ Cropped face to region: x={x}, y={y}, w={w}, h={h}")

                # Normalize antispoof_score to 0-1 range if needed
                if antispoof_score > 1.0:
                    antispoof_score = antispoof_score / 100.0

                # Calculate quality metrics
                quality_metrics = self._analyze_face_quality_enhanced(img_array)

                # Face confidence
                face_confidence_percent = face_confidence * 100 if face_confidence <= 1.0 else face_confidence

                self.logger.info(f"✅ Face analysis completed: "
                               f"anti_spoof_score={antispoof_score:.3f}, "
                               f"is_real={is_real}")

                face_analysis = {
                    'face_detected': True,
                    'face_count': len(represent_results),
                    'face_confidence': face_confidence_percent,
                    'live_person_detected': is_real,
                    'liveness_confidence': antispoof_score,
                    'quality_metrics': quality_metrics,
                    'anti_spoofing_score': antispoof_score,
                    'spoof_detection_passed': is_real,
                    'estimated_age': None,
                    'age_confidence': 0.0,
                    'face_image_b64': face_image_b64,
                    'face_embedding': face_embedding
                }

                # Convert any numpy types to native Python types
                return convert_numpy_types(face_analysis)

            except Exception as e:
                error_msg = str(e)
                self.logger.error(f"DeepFace.represent failed: {error_msg}")

                # Handle specific cases gracefully
                if "Spoof detected" in error_msg:
                    self.logger.warning("Anti-spoofing detected potential spoof")
                    return {
                        'face_detected': True,
                        'face_count': 1,
                        'face_confidence': 0.7,
                        'live_person_detected': False,
                        'liveness_confidence': 0.3,
                        'anti_spoofing_score': 0.3,
                        'spoof_detection_passed': False,
                        'estimated_age': None,
                        'age_confidence': 0.0,
                        'quality_metrics': {'overall_quality': 0.6}
                    }

                # Handle face not found
                if "Face could not be detected" in error_msg:
                    self.logger.warning("No face detected in image")
                    return {
                        'face_detected': False,
                        'face_count': 0,
                        'face_confidence': 0.0,
                        'live_person_detected': False,
                        'liveness_confidence': 0.0,
                        'anti_spoofing_score': 0.0,
                        'spoof_detection_passed': False,
                        'estimated_age': None,
                        'age_confidence': 0.0
                    }

                return None

        except Exception as e:
            self.logger.error(f"Face analysis failed: {str(e)}")
            return None

    def _load_image_array(self, image_content: bytes) -> np.ndarray:
        """Load image content as numpy array."""
        try:
            img = Image.open(io.BytesIO(image_content))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return np.array(img)
        except Exception as e:
            self.logger.error(f"Failed to load image array: {str(e)}")
            raise

    def _convert_cropped_face_to_base64(self, face_image: np.ndarray) -> str:
        """
        Convert cropped face image to base64 string for storage.

        Args:
            face_image: Face image as numpy array (RGB format)

        Returns:
            Base64-encoded face image as JPEG string
        """
        try:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(face_image.astype('uint8'))

            # Convert to JPEG bytes
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=90, optimize=True)
            image_bytes = buffer.getvalue()

            # Encode to base64 string
            base64_string = base64.b64encode(image_bytes).decode('utf-8')

            self.logger.debug(f"Converted cropped face to base64: {len(base64_string)} characters")
            return base64_string

        except Exception as e:
            self.logger.error(f"Failed to convert cropped face to base64: {str(e)}")
            return ""

    
    def _analyze_face_quality_enhanced(self, img_array: np.ndarray) -> Dict[str, float]:
        """
        Enhanced face quality analysis for selfie images.

        Args:
            img_array: Image as numpy array

        Returns:
            Dict with quality metrics
        """
        try:
            # Convert to grayscale for some calculations
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array

            # Sharpness (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(1.0, laplacian_var / 500.0)

            # Brightness analysis
            brightness_mean = np.mean(gray) / 255.0
            brightness_std = np.std(gray) / 255.0

            # Optimal brightness is 0.3-0.7
            if 0.3 <= brightness_mean <= 0.7:
                brightness_score = 1.0
            else:
                brightness_score = max(0.0, 1.0 - abs(brightness_mean - 0.5) * 2)

            # Contrast (standard deviation)
            contrast_score = min(1.0, brightness_std * 2)

            # Image size assessment
            height, width = img_array.shape[:2]
            min_dimension = min(height, width)
            size_score = min(1.0, min_dimension / 500.0)  # Expect at least 500px minimum

            # Overall quality score
            overall_quality = (
                sharpness_score * 0.3 +
                brightness_score * 0.25 +
                contrast_score * 0.25 +
                size_score * 0.2
            )

            return {
                'sharpness_score': round(sharpness_score, 3),
                'brightness_score': round(brightness_score, 3),
                'contrast_score': round(contrast_score, 3),
                'size_score': round(size_score, 3),
                'overall_quality': round(overall_quality, 3)
            }

        except Exception as e:
            self.logger.error(f"Enhanced face quality analysis failed: {str(e)}")
            return {'overall_quality': 0.0}

    # Manual anti-spoofing methods removed - now using DeepFace's built-in anti-spoofing
    # which provides more reliable ML-based liveness detection

    async def extract_otp_quick(self, content: bytes, filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Quick OTP extraction using OCR only - no face analysis.
        Used for early validation before slow PhotoHolmes processing.

        Args:
            content: Image bytes
            filename: Original filename (used to extract OTP if not found in image)

        Returns:
            Dict with 'otp' and 'confidence' keys
        """
        try:
            self.logger.info("Starting quick OTP extraction (OCR only)")

            # Extract OTP using OCR
            text_blocks = await self.text_extractor.extract_text_with_geometry(content, is_pdf=False)
            text = " ".join([block.get('text', '') for block in text_blocks])
            self.logger.info(f"Quick OCR extracted text: '{text}' ({len(text)} chars)")

            otp_result = self._extract_otp_with_confidence(text, text_blocks)

            if otp_result['otp']:
                self.logger.debug(f"✅ Quick OTP extraction found: {otp_result['otp']} (confidence: {otp_result['confidence']}%)")
                return {'otp': otp_result['otp'], 'confidence': otp_result['confidence']}

            # Fallback: check filename
            if filename:
                otp_from_filename = self._extract_otp_from_filename(filename)
                if otp_from_filename:
                    self.logger.debug(f"✅ Quick OTP extraction from filename: {otp_from_filename}")
                    return {'otp': otp_from_filename, 'confidence': 100.0}

            self.logger.warning("Quick OTP extraction: No OTP found in image or filename")
            return {'otp': None, 'confidence': 0.0}

        except Exception as e:
            self.logger.error(f"Quick OTP extraction failed: {str(e)}")
            return {'otp': None, 'confidence': 0.0}

    async def extract_face_only(self, content: bytes, pre_extracted_otp: str = None) -> SelfieOTPData:
        """
        Extract face analysis data only (no OCR). Used when OTP is already extracted.

        This method runs DeepFace analysis for anti-spoofing, age estimation, and face cropping
        without running OCR, since OTP was already extracted during early validation.

        Args:
            content: Image bytes
            pre_extracted_otp: OTP that was already extracted (will be set on result)

        Returns:
            SelfieOTPData with face analysis data and pre-extracted OTP
        """
        try:
            self.logger.info("Starting face-only extraction (OCR skipped - OTP pre-extracted)")
            selfie_data = SelfieOTPData()

            # Set the pre-extracted OTP
            if pre_extracted_otp:
                selfie_data.otp_number = pre_extracted_otp
                selfie_data.otp_detected = True
                selfie_data.otp_length = len(pre_extracted_otp)
                selfie_data.confidence_scores['otp_number'] = 100.0  # Pre-validated
                self.logger.debug(f"Using pre-extracted OTP: {pre_extracted_otp}")

            # Run face analysis (age, anti-spoofing, face cropping)
            face_analysis = await self._analyze_face_with_deepface_anti_spoofing(content)

            if face_analysis:
                selfie_data.face_detected = face_analysis['face_detected']
                selfie_data.face_count = face_analysis['face_count']
                selfie_data.face_confidence = face_analysis['face_confidence']
                selfie_data.live_person_detected = face_analysis['live_person_detected']
                selfie_data.liveness_confidence = face_analysis['liveness_confidence']
                selfie_data.anti_spoofing_score = face_analysis['anti_spoofing_score']
                selfie_data.spoof_detection_passed = face_analysis['spoof_detection_passed']

                # Add age analysis results
                selfie_data.estimated_age = face_analysis.get('estimated_age')
                selfie_data.age_confidence = face_analysis.get('age_confidence')

                # Add cropped face image for verification
                if 'face_image_b64' in face_analysis:
                    selfie_data.face_image_b64 = face_analysis['face_image_b64']
                    self.logger.info(f"Added cropped face image: {len(face_analysis['face_image_b64'])} chars")

                # Add face embedding for verification
                if 'face_embedding' in face_analysis and face_analysis['face_embedding']:
                    selfie_data.face_embedding = face_analysis['face_embedding']
                    self.logger.info(f"Added face embedding: {len(face_analysis['face_embedding'])} dimensions")

            # Calculate overall confidence
            overall_confidence = selfie_data.calculate_overall_confidence()
            selfie_data.overall_confidence = overall_confidence

            self.logger.info(f"Face-only extraction completed: Face detected={selfie_data.face_detected}, "
                           f"Anti-spoof score={selfie_data.anti_spoofing_score:.2f}, "
                           f"OTP={selfie_data.otp_number}")

            return selfie_data

        except Exception as e:
            self.logger.error(f"Face-only extraction failed: {str(e)}")
            return SelfieOTPData(confidence_scores={})

from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any, Union


class SelfieOTPData(BaseModel):
    """Enhanced fields extracted from selfie with OTP documents"""

    # OTP Information
    otp_number: Optional[str] = None
    otp_detected: bool = False
    otp_length: Optional[int] = None

    # Face Detection
    face_detected: bool = False
    face_count: int = 0
    face_confidence: Optional[float] = None

    # Enhanced Face Analysis (Internal use only - excluded from API responses)
    face_embedding: Optional[List[float]] = Field(None, exclude=True)
    face_image_b64: Optional[str] = Field(None, exclude=True)  # Base64 face image for DeepFace.verify()
    face_quality_score: Optional[float] = None

    # Liveness Detection
    live_person_detected: bool = False
    liveness_confidence: Optional[float] = None

    # Anti-Spoofing
    anti_spoofing_score: float = 0.0
    spoof_detection_passed: bool = False
    liveness_indicators: List[str] = []

    # Age Analysis (from DeepFace)
    estimated_age: Optional[int] = None
    age_confidence: Optional[float] = None

    # Image Metadata
    photo_timestamp: Optional[str] = None
    overall_confidence: Optional[float] = None

    # Face Verification Results
    face_verification_passed: Optional[bool] = None
    face_match_result: Optional[Dict] = None
    liveness_check_passed: Optional[bool] = None
    verification_details: Optional[Dict] = None

    # Mobile Number and OTP Validation Results
    mobile_number_validation: Optional[Dict[str, Any]] = None  # Results from verify_mobile_and_otp_from_selfie
    otp_validation_passed: Optional[bool] = None  # Overall OTP validation result
    mobile_number_valid: Optional[bool] = None  # Mobile number format and existence validation
    otp_check_passed: Optional[bool] = None  # OTP code validation result
    otp_status: Optional[str] = None  # OTP status (verified, expired, not_found, etc.)

    # Confidence Scores - supports both:
    # - Dict[str, float] (legacy format, 0-100)
    # - Dict[str, dict] (new format with 'overall_confidence' and 'sources')
    confidence_scores: Dict[str, Union[float, Dict[str, Any]]] = {}

    def _get_confidence_float(self, value: Union[float, Dict[str, Any]]) -> float:
        """Extract a float confidence value from various formats."""
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, dict):
            return value.get('overall_confidence', 0.0)
        return 0.0

    def calculate_overall_confidence(self) -> float:
        """Calculate overall confidence based on all extracted features."""
        scores = []

        # OTP confidence (extracted from selfie)
        if self.otp_detected and self.otp_number:
            otp_conf = self._get_confidence_float(self.confidence_scores.get('otp_number', 0.0))
            scores.append(otp_conf)

        # OTP validation confidence (from database verification)
        if self.otp_check_passed is not None:
            if self.otp_check_passed:
                scores.append(95.0)  # High confidence for successful validation
            else:
                scores.append(10.0)  # Low confidence for failed validation

        # Mobile number validation confidence
        if self.mobile_number_valid is not None:
            if self.mobile_number_valid:
                scores.append(90.0)  # High confidence for valid mobile number
            else:
                scores.append(5.0)   # Very low confidence for invalid mobile number

        # Face detection confidence
        if self.face_detected:
            face_conf = self.face_confidence or 0.0
            scores.append(face_conf)

        # Face verification confidence (highest priority)
        if self.face_verification_passed is not None and self.face_match_result:
            verification_conf = self.face_match_result.get('match_confidence', 0.0) * 100
            if self.face_verification_passed:
                scores.append(verification_conf)
            else:
                # If verification failed, still include but with reduced weight
                scores.append(verification_conf * 0.5)

        # Liveness confidence
        if self.live_person_detected:
            liveness_conf = self.liveness_confidence or 0.0
            scores.append(liveness_conf)

        # Anti-spoofing confidence
        if self.anti_spoofing_score > 0:
            anti_spoof_conf = self.anti_spoofing_score * 100
            scores.append(anti_spoof_conf)

        # Face quality confidence
        if self.face_quality_score and self.face_quality_score > 0:
            quality_conf = self.face_quality_score * 100
            scores.append(quality_conf)

        # Age analysis confidence
        if self.estimated_age is not None and self.age_confidence is not None:
            age_conf = self.age_confidence * 100
            scores.append(age_conf)

        # Calculate weighted average
        if not scores:
            return 0.0

        # Adjusted weights: OTP Extract, OTP Validation, Mobile Number, Face, Verification, Liveness, Anti-spoof, Quality, Age
        weights = [0.12, 0.15, 0.13, 0.12, 0.20, 0.10, 0.08, 0.05, 0.05]
        weighted_scores = [score * weight for score, weight in zip(scores, weights)]

        return sum(weighted_scores) / sum(weights[:len(scores)])

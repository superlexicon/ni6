from typing import Dict, Tuple, Optional
from datetime import datetime, timezone
from app.repositories.otp_repository import OTPRepository
from app.core.logger import get_logger
from app.config.verification_config import verification_settings
from app.dto import DocumentErrorCode

class SelfieValidationResult:
    """Result of selfie validation"""
    def __init__(self, success: bool, message: str, extracted_otp: Optional[str] = None, error_code: Optional[str] = None):
        self.success = success
        self.message = message
        self.extracted_otp = extracted_otp
        self.error_code = error_code  # NEW: error code for client response

class SelfieValidationService:
    """
    Reusable service for validating selfies with OTP, PhotoHolmes, and anti-spoofing checks.

    This service extracts common validation logic used across:
    - Initial verification (sequential_selfie_service)
    - Secret share requests
    - Voluntary selfie submissions
    """

    def __init__(self, otp_repository: OTPRepository = None):
        self.otp_repository = otp_repository or OTPRepository()
        self.logger = get_logger()

    def validate_otp_extraction(
        self,
        analysis_response,
        require_otp: bool = True
    ) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
        """
        Validate OTP extraction from selfie.

        Args:
            analysis_response: Document analysis response
            require_otp: Whether OTP is mandatory (default True)

        Returns:
            Tuple of (success, extracted_otp, error_message, error_code)
        """
        try:
            extracted_otp = None

            if analysis_response.documents:
                selfie_doc = next(
                    (doc for doc in analysis_response.documents if doc.document_type == "selfie"),
                    None
                )

                if selfie_doc and hasattr(selfie_doc, 'extracted_data'):
                    extracted_data = selfie_doc.extracted_data
                    if isinstance(extracted_data, dict):
                        otp_field = extracted_data.get('otp_number')
                        if otp_field and hasattr(otp_field, 'value'):
                            extracted_otp = otp_field.value
                            if extracted_otp:
                                self.logger.debug(f"Extracted OTP: {extracted_otp}")

            # Check if OTP is required but missing
            if require_otp and not extracted_otp:
                self.logger.error("OTP extraction failed - no OTP found in image or filename")
                return False, None, "OTP extraction failed - OTP must be present in image or filename", DocumentErrorCode.SELFIE_OTP_NOT_FOUND

            return True, extracted_otp, None, None

        except Exception as e:
            self.logger.error(f"Error during OTP extraction: {str(e)}")
            return False, None, f"OTP extraction error: {str(e)}", DocumentErrorCode.PROCESSING_ERROR

    def validate_otp_against_database(
        self,
        extracted_otp: str,
        public_key: str
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Validate extracted OTP against database.

        Performs 4 mandatory checks:
        1. OTP exists in database for public key
        2. OTP matches extracted value
        3. OTP not expired
        4. OTP not already verified

        Args:
            extracted_otp: OTP extracted from selfie
            public_key: User's public key

        Returns:
            Tuple of (success, error_message, error_code)
        """
        try:
            # Check 1: OTP exists in database for public key
            otp_record = self.otp_repository.get_otp_by_public_key(public_key)

            if not otp_record:
                self.logger.error(
                    f"OTP validation failed - no OTP found for public key: {public_key[:16]}..."
                )
                return False, "OTP validation failed - no OTP found for this public key", DocumentErrorCode.SELFIE_OTP_NOT_FOUND

            expected_otp = otp_record['random_number']

            # Check 2: OTP matches
            if extracted_otp != expected_otp:
                self.logger.error(
                    f"OTP validation failed - mismatch: expected {expected_otp}, got {extracted_otp}"
                )
                return False, "OTP validation failed - incorrect OTP", DocumentErrorCode.SELFIE_OTP_INCORRECT

            # Check 3: Not expired (use UTC for comparison)
            if otp_record.get('expires_at'):
                expires_at = otp_record['expires_at']
                now_utc = datetime.now(timezone.utc)
                # If expires_at is naive, assume it's UTC
                if expires_at.tzinfo is None:
                    expires_at = expires_at.replace(tzinfo=timezone.utc)
                if now_utc > expires_at:
                    self.logger.error(f"OTP validation failed - expired at {expires_at} (now: {now_utc})")
                    return False, "OTP validation failed - OTP expired", DocumentErrorCode.SELFIE_OTP_EXPIRED

            # Check 4: Not already used
            if otp_record.get('is_verified'):
                self.logger.error("OTP validation failed - OTP already used")
                return False, "OTP validation failed - OTP already verified", DocumentErrorCode.SELFIE_OTP_ALREADY_VERIFIED

            # All checks passed
            self.logger.debug(f"✅ OTP validation passed: {extracted_otp} (public_key: {public_key[:16]}...)")
            return True, None, None

        except Exception as e:
            self.logger.error(f"Error during OTP database validation: {str(e)}")
            return False, f"OTP validation error: {str(e)}", DocumentErrorCode.PROCESSING_ERROR

    def validate_otp(
        self,
        extracted_otp: str
    ) -> Tuple[bool, str, Optional[str], Optional[str]]:
        """
        Validate OTP for key recovery flow.

        This method validates the OTP code, expiry, and verification status.
        The identity_id will come from face matching, not from mobile_number lookup.

        This removes the mobile_number bottleneck - we only validate OTP validity,
        not require a successful mobile_number match in user_keys.

        Flow: OTP code -> validate code, expiry, verification status -> success
        (identity_id will be obtained later from face matching)

        Args:
            extracted_otp: OTP code extracted from selfie

        Returns:
            Tuple of (success, error_message, mobile_number, error_code)
        """
        try:
            # Step 1: Lookup OTP by code (not public_key!)
            otp_record = self.otp_repository.get_otp_by_code(extracted_otp)

            if not otp_record:
                self.logger.error(f"OTP validation failed - no OTP found for code: {extracted_otp[:4]}***")
                return False, "OTP validation failed - incorrect OTP", None, DocumentErrorCode.SELFIE_OTP_INCORRECT

            # Mobile number is optional - only used for logging/audit purposes
            mobile_number = otp_record.get('mobile_number')
            country_code = otp_record.get('country_code')
            full_mobile_number = f"{country_code}{mobile_number}" if country_code and mobile_number else mobile_number

            # Step 2: Verify OTP matches (redundant since we looked up by code, but for safety)
            expected_otp = otp_record.get('random_number')
            if extracted_otp != expected_otp:
                self.logger.error(f"OTP validation failed - mismatch: expected {expected_otp}, got {extracted_otp}")
                return False, "OTP validation failed - incorrect OTP", None, DocumentErrorCode.SELFIE_OTP_INCORRECT

            # Step 3: Check not expired
            if otp_record.get('expires_at'):
                expires_at = otp_record['expires_at']
                now_utc = datetime.now(timezone.utc)
                # If expires_at is naive, assume it's UTC
                if expires_at.tzinfo is None:
                    expires_at = expires_at.replace(tzinfo=timezone.utc)
                if now_utc > expires_at:
                    return False, "OTP validation failed - OTP expired", None, DocumentErrorCode.SELFIE_OTP_EXPIRED

            # Step 4: Check not already verified
            if otp_record.get('is_verified'):
                self.logger.error("OTP validation failed - OTP already used")
                return False, "OTP validation failed - OTP already verified", None, DocumentErrorCode.SELFIE_OTP_ALREADY_VERIFIED

            if mobile_number:
                self.logger.info(f"OTP validated successfully for mobile: {full_mobile_number}")
            else:
                self.logger.info("OTP validated successfully (no mobile number in record)")
            return True, None, full_mobile_number, None

        except Exception as e:
            self.logger.error(f"OTP validation error: {str(e)}")
            return False, f"OTP validation error: {str(e)}", None, DocumentErrorCode.PROCESSING_ERROR

    def validate_photoholmes_results(
        self,
        imdl_results,
        threshold: int = None
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Validate PhotoHolmes forgery detection results.

        Args:
            imdl_results: PhotoHolmes IMDL results
            threshold: Number of methods detecting forgery to reject (default from config)

        Returns:
            Tuple of (success, error_message, error_code)
        """
        if threshold is None:
            threshold = verification_settings.forgery_detection_threshold
        try:
            # Check if results are available
            if not imdl_results:
                self.logger.error("Forgery detection validation failed - PhotoHolmes results unavailable")
                return False, (
                    "Document validation failed - forgery detection results unavailable. "
                    "PhotoHolmes analysis is required for all documents."
                ), DocumentErrorCode.PROCESSING_ERROR

            detections = imdl_results.checks_with_detections

            # Check if too many methods detected forgery
            if detections >= threshold:
                # Get names of methods that detected forgery
                detected_methods = [
                    check.name for check in imdl_results.checks
                    if check.detected_forgery
                ]

                self.logger.error(
                    f"Forgery detection validation failed - {detections} methods detected forgery: "
                    f"{', '.join(detected_methods)}"
                )

                error_msg = (
                    f"Document validation failed - {detections} forgery detection methods "
                    f"flagged potential manipulation: {', '.join(detected_methods)}. "
                    f"This document cannot be accepted for verification."
                )
                return False, error_msg, DocumentErrorCode.LOGICAL_FORGERY_DETECTED

            # Validation passed
            self.logger.info(f"✅ PhotoHolmes validation passed - {detections} detections (threshold: {threshold})")
            return True, None, None

        except Exception as e:
            self.logger.error(f"Error during PhotoHolmes validation: {str(e)}")
            return False, f"PhotoHolmes validation error: {str(e)}", DocumentErrorCode.PROCESSING_ERROR

    def validate_anti_spoofing(
        self,
        analysis_response,
        threshold: float = None
    ) -> Tuple[bool, Optional[str], Optional[float], Optional[str]]:
        """
        Validate anti-spoofing results from face analysis.

        Args:
            analysis_response: Document analysis response
            threshold: Minimum anti-spoofing score (default from config)

        Returns:
            Tuple of (success, error_message, anti_spoofing_score, error_code)
        """
        if threshold is None:
            threshold = verification_settings.anti_spoofing_threshold
        try:
            if not analysis_response.documents:
                return False, "No documents found in analysis response", None, DocumentErrorCode.PROCESSING_ERROR

            selfie_doc = next(
                (doc for doc in analysis_response.documents if doc.document_type == "selfie"),
                None
            )

            if not selfie_doc or not hasattr(selfie_doc, 'extracted_data'):
                return False, "Selfie document not found or missing extracted data", None, DocumentErrorCode.PROCESSING_ERROR

            extracted_data = selfie_doc.extracted_data

            # Handle both SelfieOTPData object and simplified dict format
            from app.schemas.selfie_otp_schema import SelfieOTPData
            if isinstance(extracted_data, SelfieOTPData):
                anti_spoofing_score = extracted_data.anti_spoofing_score
            elif isinstance(extracted_data, dict):
                # Handle simplified format where values are ExtractedFieldData or nested dicts
                anti_spoof_field = extracted_data.get('anti_spoofing_score')
                if anti_spoof_field is None:
                    return False, "Anti-spoofing score not found in extracted data", None, DocumentErrorCode.PROCESSING_ERROR
                # Extract value from ExtractedFieldData or dict
                if hasattr(anti_spoof_field, 'value'):
                    anti_spoofing_score = float(anti_spoof_field.value)
                elif isinstance(anti_spoof_field, dict) and 'value' in anti_spoof_field:
                    anti_spoofing_score = float(anti_spoof_field['value'])
                else:
                    anti_spoofing_score = float(anti_spoof_field)
            else:
                return False, "Invalid selfie data format", None, DocumentErrorCode.PROCESSING_ERROR

            # Get anti-spoofing score (already extracted above)

            if anti_spoofing_score is None:
                self.logger.warning("Anti-spoofing score not available")
                return False, "Anti-spoofing check failed - score not available", None, DocumentErrorCode.PROCESSING_ERROR

            # Check against threshold
            if anti_spoofing_score < threshold:
                self.logger.error(
                    f"Anti-spoofing validation failed - score {anti_spoofing_score:.2f} below threshold {threshold}"
                )
                return False, (
                    f"Liveness check failed - anti-spoofing score {anti_spoofing_score:.2f} "
                    f"below required threshold {threshold}. Please submit a live selfie."
                ), anti_spoofing_score, DocumentErrorCode.SELFIE_LIVENESS_FAILED

            # Validation passed
            self.logger.info(f"✅ Anti-spoofing validation passed - score: {anti_spoofing_score:.2f}")
            return True, None, anti_spoofing_score, None

        except Exception as e:
            self.logger.error(f"Error during anti-spoofing validation: {str(e)}")
            return False, f"Anti-spoofing validation error: {str(e)}", None, DocumentErrorCode.PROCESSING_ERROR

    def validate_selfie_complete(
        self,
        analysis_response,
        public_key: str,
        require_otp: bool = True,
        photoholmes_threshold: int = None,
        anti_spoofing_threshold: float = None
    ) -> SelfieValidationResult:
        """
        Perform complete selfie validation (OTP + PhotoHolmes + anti-spoofing).

        This is a convenience method that runs all validations in sequence.

        Args:
            analysis_response: Document analysis response
            public_key: User's public key (for OTP lookup)
            require_otp: Whether OTP is mandatory (default True)
            photoholmes_threshold: Forgery detection threshold (default from config)
            anti_spoofing_threshold: Liveness threshold (default from config)

        Returns:
            SelfieValidationResult with success status and message
        """
        if photoholmes_threshold is None:
            photoholmes_threshold = verification_settings.forgery_detection_threshold
        if anti_spoofing_threshold is None:
            anti_spoofing_threshold = verification_settings.anti_spoofing_threshold
        try:
            # Step 1: Extract OTP
            otp_extracted, extracted_otp, otp_error, otp_error_code = self.validate_otp_extraction(
                analysis_response,
                require_otp=require_otp
            )

            if not otp_extracted:
                return SelfieValidationResult(success=False, message=otp_error, error_code=otp_error_code)

            # Step 2: Validate OTP against database (only if OTP was required and extracted)
            otp_error_code = None
            if require_otp and extracted_otp:
                otp_valid, otp_db_error, otp_db_error_code = self.validate_otp_against_database(
                    extracted_otp,
                    public_key
                )

                if not otp_valid:
                    return SelfieValidationResult(success=False, message=otp_db_error, error_code=otp_db_error_code)

            # Step 3: Validate PhotoHolmes
            if analysis_response.documents:
                selfie_doc = next(
                    (doc for doc in analysis_response.documents if doc.document_type == "selfie"),
                    None
                )

                if selfie_doc and hasattr(selfie_doc, 'imdl_results'):
                    photoholmes_valid, photoholmes_error, photoholmes_error_code = self.validate_photoholmes_results(
                        selfie_doc.imdl_results,
                        threshold=photoholmes_threshold
                    )

                    if not photoholmes_valid:
                        return SelfieValidationResult(success=False, message=photoholmes_error, error_code=photoholmes_error_code)

            # Step 4: Validate anti-spoofing
            anti_spoof_valid, anti_spoof_error, anti_spoof_score, anti_spoof_error_code = self.validate_anti_spoofing(
                analysis_response,
                threshold=anti_spoofing_threshold
            )

            if not anti_spoof_valid:
                return SelfieValidationResult(success=False, message=anti_spoof_error, error_code=anti_spoof_error_code)

            # All validations passed
            self.logger.info("✅ Complete selfie validation passed")
            return SelfieValidationResult(
                success=True,
                message="Selfie validation passed all checks",
                extracted_otp=extracted_otp,
                error_code=None
            )

        except Exception as e:
            self.logger.error(f"Error during complete selfie validation: {str(e)}")
            return SelfieValidationResult(
                success=False,
                message=f"Selfie validation error: {str(e)}",
                error_code=DocumentErrorCode.PROCESSING_ERROR
            )

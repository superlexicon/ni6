
import re
from typing import TYPE_CHECKING
from app.core import logger
from ..deepface_helper import DeepfaceHelper
from ..doctr.document_text_extractor import DocumentTextExtractor
from ..doctr.text_matcher import TextMatcher
from ..doctr.document_validator import DocumentValidator
from app.dto import DocumentVerifyRequest, DeepFaceVerifyRequest, OTPVerificationEvent
from app.repositories import OTPRepository
from .verification_thresholds import VerificationThresholds

if TYPE_CHECKING:
    from app.services.common_document_verification_service import CommonDocumentVerificationService
from .verification_models import (
    DocumentVerificationResult,
    ExtractedTexts,
    MatchResults
)


class VerificationHelper:

    # Use centralized thresholds
    MIN_BANK_STATEMENT_THRESHOLD = VerificationThresholds.MIN_BANK_STATEMENT_THRESHOLD
    MIN_USERNAME_THRESHOLD = VerificationThresholds.MIN_USERNAME_THRESHOLD
    MIN_PHONE_THRESHOLD = VerificationThresholds.MIN_PHONE_THRESHOLD
    MIN_NATIONALITY_THRESHOLD = VerificationThresholds.MIN_NATIONALITY_THRESHOLD
    MIN_DOB_THRESHOLD = VerificationThresholds.MIN_DOB_THRESHOLD
    MIN_AGE_YEARS = VerificationThresholds.MIN_AGE_YEARS

    def __init__(
        self,
        otp_repository: OTPRepository,
        document_text_extractor: DocumentTextExtractor,
        text_matcher: TextMatcher,
        document_validator: DocumentValidator,
        deepface_helper: DeepfaceHelper,
        common_verification_service: 'CommonDocumentVerificationService' = None
    ) -> None:
        self.logger = logger
        self.otp_repository = otp_repository
        self.document_text_extractor = document_text_extractor
        self.text_matcher = text_matcher
        self.document_validator = document_validator
        self.deepface_helper = deepface_helper
        self.common_verification_service = common_verification_service

        # Precompiled regex patterns for better performance
        self._regex_nationality = re.compile(r'Nationality\s+([A-Z]+)')
        self._regex_dob = re.compile(
            r'Date\s*of\s*(?:Birth|Dirth)\s+([0-9]{2}\s*[A-Z]{3}\s*[0-9]{4})',
            flags=re.IGNORECASE
        )

    async def otp_verify(
        self,
        request: OTPVerificationEvent,
        fake: bool = False
    ) -> bool:
        try:
            if fake:
                return False
            otp_text = await self.document_text_extractor.extract_text(
                request.selfie
            )

            otp_number = self.otp_repository.get_otp_by_email(request.email)
            if otp_number:
                if otp_number['random_number'] == otp_text:
                    return True
                else:
                    return False
            else:
                return False
        except Exception as e:
            self.logger.error(f"Error during OTP verification: {str(e)}")
            return False

    async def document_verify(
        self,
        request: DocumentVerifyRequest,
        fake: bool = False
    ) -> DocumentVerificationResult:
        try:
            if fake:
                return self._create_fake_verification_result()

            extracted_texts = await self._extract_document_texts(request)

            match_results = await self._perform_text_matching(
                extracted_texts, request
            )

            is_verified = self._evaluate_verification_result(match_results)

            return DocumentVerificationResult(
                bank_statement_match_pct=match_results.bank_statement_pct,
                username_match_pct=match_results.username_pct,
                nationality_match_pct=match_results.nationality_confidence,
                dob_match_pct=match_results.dob_confidence,
                phone_number_match_pct=match_results.phone_number_pct,
                is_valid_age=match_results.is_valid_age,
                is_verified=is_verified
            )

        except Exception as e:
            raise ValueError(f"Error during document verification: {str(e)}")

    async def deepface_verify(
        self,
        request: DeepFaceVerifyRequest,
        fake: bool = False
    ) -> bool:
        try:
            if fake:
                return False
            self.logger.info("Deepface verification Start")
            result = await self.deepface_helper.verify(
                request.selfie,
                request.passport
            )
            self.logger.info(f"Deepface verification result: {result}")
            return bool(result)

        except Exception as e:
            self.logger.error(f"Error during deepface verification: {str(e)}")
            return False

    async def _extract_document_texts(self, request: DocumentVerifyRequest) -> ExtractedTexts:
        passport_text = await self.document_text_extractor.extract_text(
            request.passport
        )
        bank_statement_text = await self.document_text_extractor.extract_text(
            request.bank_stmt
        )
        resume_text = await self.document_text_extractor.extract_text(
            request.resume, is_pdf=True
        )

        return ExtractedTexts(
            passport_text=passport_text,
            bank_statement_text=bank_statement_text,
            resume_text=resume_text
        )

    async def _perform_text_matching(
        self,
        texts: ExtractedTexts,
        request: DocumentVerifyRequest
    ) -> MatchResults:
        # Bank statement account number matching
        bank_statement_pct = self.text_matcher.calculate_match_percentage(
            texts.bank_statement_text, request.account_number
        )
        self.logger.info(
            f"Bank statement match percentage: {bank_statement_pct}%")

        # Username matching across all documents
        username_pct = self._calculate_username_match_percentage(
            texts, request.username
        )
        self.logger.debug(f"Username match percentage: {username_pct}%")

        # Phone number matching
        phone_number_match = self.text_matcher.check_match(
            texts.resume_text, request.phone_number
        )
        phone_number_pct = phone_number_match.confidence
        self.logger.info(f"Phone number match percentage: {phone_number_pct}%")

        # Extract and validate nationality and date of birth
        nationality_result = self.text_matcher.extract_field_with_regex(
            texts.passport_text, self._regex_nationality
        )
        dob_result = self.text_matcher.extract_field_with_regex(
            texts.passport_text, self._regex_dob
        )

        self.logger.info(
            f"Nationality confidence: {nationality_result.confidence}%")
        self.logger.debug(f"Date of birth confidence: {dob_result.confidence}%")

        # Age validation
        is_valid_age = False
        if dob_result.matched_text:
            is_valid_age = self.document_validator.is_valid_age(
                dob_result.matched_text
            )
            self.logger.info(f"Age validation result: {is_valid_age}")

        return MatchResults(
            bank_statement_pct=bank_statement_pct,
            username_pct=username_pct,
            phone_number_pct=phone_number_pct,
            nationality_confidence=nationality_result.confidence,
            dob_confidence=dob_result.confidence,
            is_valid_age=is_valid_age
        )

    def _calculate_username_match_percentage(
        self,
        texts: ExtractedTexts,
        username: str
    ) -> float:
        username_passport_match = self.text_matcher.check_match(
            texts.passport_text, username
        )
        username_bank_stmt_match = self.text_matcher.check_match(
            texts.bank_statement_text, username
        )
        username_resume_match = self.text_matcher.check_match(
            texts.resume_text, username
        )

        return max(
            username_passport_match.confidence,
            username_bank_stmt_match.confidence,
            username_resume_match.confidence
        )

    def _evaluate_verification_result(self, match_results: MatchResults) -> bool:
        criteria_met = [
            match_results.bank_statement_pct >= self.MIN_BANK_STATEMENT_THRESHOLD,
            match_results.username_pct >= self.MIN_USERNAME_THRESHOLD,
            match_results.phone_number_pct >= self.MIN_PHONE_THRESHOLD,
            match_results.nationality_confidence >= self.MIN_NATIONALITY_THRESHOLD,
            match_results.dob_confidence >= self.MIN_DOB_THRESHOLD,
            match_results.is_valid_age
        ]

        verification_passed = all(criteria_met)
        self.logger.info(
            f"Verification result: {verification_passed} "
            f"(criteria met: {sum(criteria_met)}/{len(criteria_met)})"
        )

        return verification_passed

    def _create_fake_verification_result(self) -> DocumentVerificationResult:
        self.logger.info("Creating fake verification result for testing")
        return DocumentVerificationResult(
            bank_statement_match_pct=0.0,
            username_match_pct=0.0,
            nationality_match_pct=0.0,
            dob_match_pct=0.0,
            phone_number_match_pct=0.0,
            is_valid_age=False,
            is_verified=False
        )

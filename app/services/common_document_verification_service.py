from typing import Tuple, Optional
from app.dto.document_analysis import (
    BasicSecurityChecks,
    BankStatementValidation,
    AuthenticityData
)
from app.services.ela_service import ELAService
from app.helper.exif_validator import ExifValidator
from app.helper.document_type_detector import DocumentTypeDetector
from app.core import logger


class DocumentVerificationResult:
    """Complete verification result for a document"""

    def __init__(
        self,
        # Basic checks
        basic_security_checks: BasicSecurityChecks,
        authenticity: Optional[AuthenticityData] = None,

        # Document type
        document_type: str = "generic",
        document_type_confidence: float = 0.0,

        # Document-specific validations
        bank_statement_validation: Optional[BankStatementValidation] = None
    ):
        self.basic_security_checks = basic_security_checks
        self.authenticity = authenticity
        self.document_type = document_type
        self.document_type_confidence = document_type_confidence
        self.bank_statement_validation = bank_statement_validation


class CommonDocumentVerificationService:
    """
    Common service for document verification shared across endpoints.
    Performs basic security checks, document type detection, and type-specific validation.
    """

    def __init__(
        self,
        ela_service: ELAService,
        exif_validator: ExifValidator,
        document_type_detector: DocumentTypeDetector
    ):
        self.ela_service = ela_service
        self.exif_validator = exif_validator
        self.document_type_detector = document_type_detector
        self.logger = logger

    async def verify_document(
        self,
        content: bytes,
        is_pdf: bool = False,
        include_authenticity_detection: bool = True,
        authenticity: Optional[AuthenticityData] = None
    ) -> DocumentVerificationResult:
        """
        Perform comprehensive document verification

        Args:
            content: Document bytes
            is_pdf: Whether document is PDF
            include_authenticity_detection: Whether to include authenticity results
            authenticity: Pre-computed authenticity result (optional)

        Returns:
            DocumentVerificationResult with all verification results
        """
        self.logger.info("Starting common document verification...")

        # PHASE 1: BASIC SECURITY CHECKS
        basic_security_checks = await self._run_basic_checks(content, is_pdf)

        # PHASE 2: DOCUMENT TYPE DETECTION
        document_type, doc_type_confidence = await self._detect_document_type(content, is_pdf)

        # PHASE 3: DOCUMENT-SPECIFIC CHECKS
        bank_validation = await self._run_specific_checks(
            content, is_pdf, document_type
        )

        return DocumentVerificationResult(
            basic_security_checks=basic_security_checks,
            authenticity=authenticity if include_authenticity_detection else None,
            document_type=document_type,
            document_type_confidence=doc_type_confidence,
            bank_statement_validation=bank_validation
        )

    async def _run_basic_checks(self, content: bytes, is_pdf: bool) -> BasicSecurityChecks:
        """
        Run basic security checks on document

        Args:
            content: Document bytes
            is_pdf: Whether document is PDF

        Returns:
            BasicSecurityChecks with EXIF and ELA results
        """
        self.logger.info("Running basic security checks...")

        # EXIF analysis
        exif_result = await self.exif_validator.analyze(content)

        # Error Level Analysis (only for images)
        ela_score = None
        if not is_pdf:
            ela_score = await self.ela_service.analyze(content)

        return BasicSecurityChecks(
            exif_authenticity_score=exif_result.score,
            editing_software_detected=exif_result.has_editing_software,
            suspicious_indicators=exif_result.findings,
            manipulation_score=ela_score
        )

    async def _detect_document_type(
        self,
        content: bytes,
        is_pdf: bool
    ) -> Tuple[str, float]:
        """
        Detect document type

        Args:
            content: Document bytes
            is_pdf: Whether document is PDF

        Returns:
            Tuple of (document_type, confidence)
        """
        self.logger.info("Detecting document type...")

        # Extract text for detection
        extracted_text = await self._extract_text_for_detection(content, is_pdf)

        # Detect type
        document_type, confidence = await self.document_type_detector.detect(
            content, extracted_text
        )

        self.logger.info(f"Detected document type: {document_type} (confidence: {confidence:.1f}%)")
        return document_type, confidence

    async def _run_specific_checks(
        self,
        content: bytes,
        is_pdf: bool,
        document_type: str
    ) -> Optional[BankStatementValidation]:
        """
        Run document-type specific checks

        Args:
            content: Document bytes
            is_pdf: Whether document is PDF
            document_type: Detected document type

        Returns:
            BankStatementValidation if applicable, None otherwise
        """
        # No document-specific checks implemented
        return None

    async def _extract_text_for_detection(self, content: bytes, is_pdf: bool) -> str:
        """
        Extract text from document for type detection

        Args:
            content: Document bytes
            is_pdf: Whether content is PDF

        Returns:
            Extracted text string
        """
        try:
            from app.helper.doctr.document_text_extractor import DocumentTextExtractor
            extractor = DocumentTextExtractor()
            text = await extractor.extract_text(content, is_pdf=is_pdf, max_pages=2)
            return text
        except Exception as e:
            self.logger.warning(f"Could not extract text for detection: {str(e)}")
            return ""

from app.utils import decode_base64
from app.utils import unique_random_generator

from app.core import key_pair, scalsa20_crypto
from app.core.logger import get_logger

from .forgery_service import ForgeryService
from .image_validation_service import ImageValidationService
from .comprehensive_photoholmes_service import ComprehensivePhotoHolmesService
from .key_service import KeyService
from .otp_service import OTPService
from .pdf_analysis_service import PDFAnalysisService
from .ela_service import ELAService
# Legacy: CommonDocumentVerificationService requires document_type_detector which was removed
# from .common_document_verification_service import CommonDocumentVerificationService
from .verification_state_service import VerificationStateService
from .database_encryption_service import get_encryption_service, DatabaseEncryptionService
from .ecies_encryption_service import get_ecies_encryption_service, ECIESEncryptionService

# Import BERT NER Resume Extractor
from app.helper.extractors.bert_ner_resume_extractor import BertNerResumeExtractor

_services_logger = get_logger()

from app.core import (
    AIGeneratedAnalyzer,
    PhotoShoppedAnalyzer
)
from app.repositories import (
    user_key_repository,
    otp_repository,
    user_identity_repository,
    document_submission_repository
)
from app.helper import (key_helper,
                        verification_helper,
                        prepare_verification_data,
                        exif_validator,
                        )

# Initialize services
image_validation_service = ImageValidationService()
aigenerated_analyzer = AIGeneratedAnalyzer()
photoshopped_analyzer = PhotoShoppedAnalyzer()
comprehensive_photoholmes_service = ComprehensivePhotoHolmesService()


key_service = KeyService(user_key_repository,
                         otp_repository,
                         decode_base64,
                         key_helper,
                         key_pair,
                         scalsa20_crypto
                         )

# Initialize OTP broadcast service for HTTP-based inter-instance communication
otp_broadcast_service = None
try:
    from .otp_broadcast_service import otp_broadcast_service as broadcast_svc
    otp_broadcast_service = broadcast_svc
    _services_logger.info("OTP broadcast service initialized")
except Exception as e:
    _services_logger.warning(f"OTP broadcast service not available: {e}")
    otp_broadcast_service = None

otp_service = OTPService(
    unique_random_generator,
    otp_repository,
    otp_broadcast_service=otp_broadcast_service
)


forgery_service = ForgeryService(
    decode_base64=decode_base64,
    aigenerated_analyzer=aigenerated_analyzer,
    photoshopped_analyzer=photoshopped_analyzer,
    image_validation_service=image_validation_service,
    comprehensive_photoholmes_service=comprehensive_photoholmes_service,
)

pdf_analysis_service = PDFAnalysisService()
ela_service = ELAService()

# Note: CommonDocumentVerificationService requires document_type_detector which was removed
# This service is legacy and should not be used
# common_verification_service = CommonDocumentVerificationService(
#     ela_service=ela_service,
#     exif_validator=exif_validator,
#     document_type_detector=document_type_detector
# )

# Initialize BERT NER Resume Extractor
bert_ner_resume_extractor = BertNerResumeExtractor()

__all__ = [
    "forgery_service",
    "AIGeneratedAnalyzer",
    "key_service",
    "otp_service",
    "otp_broadcast_service",
    "common_verification_service",
    "VerificationStateService",
    "bert_ner_resume_extractor",
    "get_encryption_service",
    "DatabaseEncryptionService",
    "get_ecies_encryption_service",
    "ECIESEncryptionService",
]

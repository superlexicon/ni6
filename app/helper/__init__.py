from app.repositories import otp_repository
from .face_recognition_factory import get_face_recognition_backend
from .doctr.document_text_extractor import DocumentTextExtractor
from .doctr.document_validator import DocumentValidator
from .doctr.text_matcher import TextMatcher
from .key_helper import KeyHelper
from .verification.verification_helper import VerificationHelper
from .verification.verification_thresholds import VerificationThresholds
from .verification.prepare_verification_data import PrepareVerificationData
from .exif_validator import ExifValidator
# Document type detector removed (was GLiNER-based)
# from .document_type_detector import DocumentTypeDetector
from .tax_statement_validator import TaxStatementValidator

document_text_extractor = DocumentTextExtractor()
exif_validator = ExifValidator()
# document_type_detector = DocumentTypeDetector()
# Use factory to get the appropriate backend (DeepFace or InsightFace)
deepface_helper = get_face_recognition_backend()

text_matcher = TextMatcher()
document_validator = DocumentValidator()
prepare_verification_data = PrepareVerificationData()

verification_helper = VerificationHelper(
    otp_repository,
    document_text_extractor,
    text_matcher,
    document_validator,
    deepface_helper
)

key_helper = KeyHelper(
    document_text_extractor,
    otp_repository,
    deepface_helper)

__all__ = [
    "deepface_helper",
    "document_text_extractor",
    "key_helper",
    "DocumentValidator",
    "TextMatcher",
    "VerificationThresholds",
    "verification_helper",
    "prepare_verification_data",
    "exif_validator",
    # "document_type_detector",  # Removed (GLiNER-based)
    "TaxStatementValidator"
]

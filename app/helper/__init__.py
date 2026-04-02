from app.repositories import otp_repository
from .deepface_helper import DeepfaceHelper
from .doctr.document_text_extractor import DocumentTextExtractor
from .doctr.document_validator import DocumentValidator
from .doctr.text_matcher import TextMatcher
from .key_helper import KeyHelper
from .verification.verification_helper import VerificationHelper
from .verification.verification_thresholds import VerificationThresholds
from .verification.prepare_verification_data import PrepareVerificationData
from .exif_validator import ExifValidator
from .document_type_detector import DocumentTypeDetector
from .bank_statement_validator import BankStatementValidator
from .tax_statement_validator import TaxStatementValidator

document_text_extractor = DocumentTextExtractor()
exif_validator = ExifValidator()
document_type_detector = DocumentTypeDetector()
bank_statement_validator = BankStatementValidator()
deepface_helper = DeepfaceHelper()

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
    "document_type_detector",
    "bank_statement_validator",
    "TaxStatementValidator"
]

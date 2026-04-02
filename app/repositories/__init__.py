from .user_key_repository import UserKeyRepository
from .otp_repository import OTPRepository
from .user_identity_repository import UserIdentityRepository
from .document_submission_repository import DocumentSubmissionRepository
from .pep_repository import PEPRepository
from .crime_repository import CrimeRepository
from .face_biometrics_repository import FaceBiometricsRepository

# Global repository instances - each creates its own connection from the pool
# This avoids connection sharing issues across threads
user_key_repository = UserKeyRepository()
otp_repository = OTPRepository()
user_identity_repository = UserIdentityRepository()
document_submission_repository = DocumentSubmissionRepository()
face_biometrics_repository = FaceBiometricsRepository()

# PEP and Crime repositories use OSSPEP database (separate from main DB)
try:
    pep_repository = PEPRepository()
    crime_repository = CrimeRepository()
except Exception as e:
    from app.core.logger import get_logger
    logger = get_logger()
    logger.warning(f"OSSPEP database not available: {e}")
    pep_repository = None
    crime_repository = None

__all__ = [
    "user_key_repository",
    "otp_repository",
    "user_identity_repository",
    "document_submission_repository",
    "face_biometrics_repository",
    "FaceBiometricsRepository",
    "PEPRepository",
    "CrimeRepository",
    "pep_repository",
    "crime_repository"
]

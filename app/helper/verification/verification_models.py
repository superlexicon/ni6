
from dataclasses import dataclass
from .verification_thresholds import VerificationThresholds


@dataclass
class DocumentVerificationResult:

    bank_statement_match_pct: float
    username_match_pct: float
    nationality_match_pct: float
    dob_match_pct: float
    phone_number_match_pct: float
    is_valid_age: bool
    is_verified: bool

    def is_bank_statement_verified(self) -> bool:
        return self.bank_statement_match_pct >= VerificationThresholds.MIN_BANK_STATEMENT_THRESHOLD

    def is_username_verified(self) -> bool:
        return self.username_match_pct >= VerificationThresholds.MIN_USERNAME_THRESHOLD

    def is_phone_number_verified(self) -> bool:
        return self.phone_number_match_pct >= VerificationThresholds.MIN_PHONE_THRESHOLD

    def is_nationality_verified(self) -> bool:
        return self.nationality_match_pct >= VerificationThresholds.MIN_NATIONALITY_THRESHOLD

    def is_dob_verified(self) -> bool:
        return self.dob_match_pct >= VerificationThresholds.MIN_DOB_THRESHOLD


@dataclass
class ExtractedTexts:
    passport_text: str
    bank_statement_text: str
    resume_text: str


@dataclass
class MatchResults:
    bank_statement_pct: float
    username_pct: float
    phone_number_pct: float
    nationality_confidence: float
    dob_confidence: float
    is_valid_age: bool

import os
try:
    from pydantic import BaseSettings, Field
except ImportError:
    from pydantic_settings import BaseSettings
    from pydantic import Field


class VerificationSettings(BaseSettings):
    """Verification threshold settings - configurable via environment variables"""

    # Face matching threshold (0-100%)
    face_match_threshold: float = Field(
        default=70.0,
        description="Minimum face match confidence percentage for passport verification (0-100)"
    )

    # Secret share recovery face matching threshold (0-100%)
    secret_share_face_match_threshold: float = Field(
        70.0,
        description="Minimum face match confidence percentage for secret share recovery (0-100)"
    )

    # Name matching threshold (0-100%)
    name_match_threshold: float = Field(
        70.0,
        description="Minimum name similarity percentage (0-100)"
    )

    # Forgery detection threshold (number of methods)
    forgery_detection_threshold: int = Field(
        3,
        description="Number of PhotoHolmes methods detecting forgery to reject document"
    )

    # Anti-spoofing threshold (0-1)
    anti_spoofing_threshold: float = Field(
        0.7,
        description="Minimum anti-spoofing score (0.0-1.0)"
    )

    # Field extraction confidence thresholds (0-1)
    field_confidence_threshold: float = Field(
        0.70,
        description="Minimum confidence for required field extraction (0.0-1.0)"
    )

    field_confidence_strict: float = Field(
        0.85,
        description="Strict mode confidence for sensitive fields like document numbers (0.0-1.0)"
    )

    # Bank statement specific settings
    bank_statement_max_age_days: int = Field(
        90,
        description="Maximum age of bank statement in days"
    )

    # Passport validity settings
    passport_min_validity_days: int = Field(
        180,
        description="Minimum passport validity in days (6 months)"
    )

    # Document resubmission settings - single rate limit across all document types
    # Rate limits configurable via environment variables for development/testing
    max_document_submissions_per_hour: int = Field(
        default=int(os.getenv("VERIFICATION_MAX_DOCUMENT_SUBMISSIONS_PER_HOUR", "10")),
        description="Maximum total document submissions per hour (all types combined). Configure via VERIFICATION_MAX_DOCUMENT_SUBMISSIONS_PER_HOUR env var."
    )

    max_document_submissions_per_day: int = Field(
        default=int(os.getenv("VERIFICATION_MAX_DOCUMENT_SUBMISSIONS_PER_DAY", "30")),
        description="Maximum total document submissions per day (all types combined). Configure via VERIFICATION_MAX_DOCUMENT_SUBMISSIONS_PER_DAY env var."
    )

    # Resubmission strategy for old documents
    resubmission_strategy: str = Field(
        "replace",
        description="Strategy for handling resubmissions: 'replace' (deactivate old) or 'keep' (keep history)"
    )

    # Document orientation validation settings
    min_text_blocks_for_valid_doc: int = Field(
        5,
        description="Minimum number of text blocks required to consider document orientation valid"
    )

    min_text_lines_for_valid_doc: int = Field(
        10,
        description="Minimum number of text lines required to consider document orientation valid"
    )

    enable_orientation_validation: bool = Field(
        True,
        description="Enable orientation validation (disabled for selfies)"
    )

    # Confidence validation override
    skip_confidence_validation: bool = Field(
        False,
        description="Skip confidence validation for extracted fields (useful for debugging/testing)"
    )

    # PhotoHolmes skip option (for testing or when PyTorch has compatibility issues)
    skip_photoholmes: bool = Field(
        False,
        description="Skip PhotoHolmes forgery detection checks (useful for testing or if PyTorch has issues)"
    )

    # Face matching skip option (for testing without selfie)
    skip_face_matching: bool = Field(
        False,
        description="Skip face matching checks (useful for testing without selfie image)"
    )

    model_config = {
        "env_prefix": "VERIFICATION_",
        "env_file": ".env",
        "extra": "ignore"
    }


# Global settings instance
verification_settings = VerificationSettings()

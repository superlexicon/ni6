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

    # Selfie quality thresholds (lowered to reduce false rejections)
    selfie_quality_brightness_min: float = Field(
        0.25,
        description="Minimum brightness threshold for selfie quality check (0.0-1.0). Reject very dark/bright images."
    )

    selfie_quality_sharpness_min: float = Field(
        0.2,
        description="Minimum sharpness threshold for selfie quality check (0.0-1.0). Reject blurry images."
    )

    selfie_quality_contrast_min: float = Field(
        0.25,
        description="Minimum contrast threshold for selfie quality check (0.0-1.0). Reject low contrast images."
    )

    selfie_quality_resolution_min: float = Field(
        0.3,
        description="Minimum resolution threshold for selfie quality check (0.0-1.0). Reject small faces."
    )

    selfie_face_margin_pixels: int = Field(
        50,
        description="Face margin in pixels from image edge for completeness check. Face extending to 2+ edges within this margin indicates cropping."
    )

    selfie_require_both_eyes: bool = Field(
        True,
        description="Require both eyes to be detected in facial landmark check"
    )

    selfie_require_nose: bool = Field(
        True,
        description="Require nose to be detected in facial landmark check"
    )

    selfie_min_landmarks: int = Field(
        3,
        description="Minimum critical landmarks required (eyes + nose) for reliable face detection"
    )

    # Selfie downsizing settings (similar to passport vision LLM sizing)
    selfie_max_dimension_pixels: int = Field(
        1024,
        description="Maximum dimension (width or height) for stored selfie face images. Images exceeding this will be proportionally scaled down."
    )

    selfie_downsize_quality: int = Field(
        90,
        description="JPEG quality for downsized selfie images (1-100)"
    )

    # Document processing settings (common to all document types)
    # Note: Token-aware sizing is now primarily handled by DocumentPreprocessingService
    # which calculates max dimension from token budget (max ~1078px for Qwen3.5)
    # This setting is kept as a fallback/upper limit for other use cases
    document_max_dimension_pixels: int = Field(
        2048,
        description="Maximum dimension (width or height) for processed document images. Images exceeding this will be proportionally scaled down."
    )

    document_quality_brightness_min: float = Field(
        0.25,
        description="Minimum brightness threshold for document quality check (0.0-1.0)"
    )

    document_quality_sharpness_min: float = Field(
        0.15,
        description="Minimum sharpness threshold for document quality check (0.0-1.0)"
    )

    document_quality_contrast_min: float = Field(
        0.2,
        description="Minimum contrast threshold for document quality check (0.0-1.0)"
    )

    model_config = {
        "env_prefix": "VERIFICATION_",
        "env_file": ".env",
        "extra": "ignore"
    }


# Global settings instance
verification_settings = VerificationSettings()

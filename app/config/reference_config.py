"""
Reference Template Configuration

Configuration for passport reference template comparison.
"""

try:
    from pydantic import BaseSettings, Field
except ImportError:
    from pydantic_settings import BaseSettings
    from pydantic import Field


class ReferenceSettings(BaseSettings):
    """Reference template comparison settings."""

    # Overall similarity threshold for pass/fail (0-1)
    similarity_threshold: float = Field(
        0.75,
        description="Minimum overall similarity score required to pass reference check (0.0-1.0)"
    )

    # Enable/disable reference checking
    enable_reference_check: bool = Field(
        True,
        description="Enable passport reference template comparison"
    )

    # Skip if no template available (pass with warning)
    skip_if_no_template: bool = Field(
        True,
        description="Pass reference check if no template available for country (with warning)"
    )

    # Enforce passport specimen check (if False, always passes regardless of score)
    enforce_passport_specimen_check: bool = Field(
        True,
        description="Enforce passport specimen similarity check. If False, reference check always passes (for testing/disabled mode)"
    )

    # Minimum region score threshold (individual regions)
    min_region_score: float = Field(
        0.5,
        description="Minimum score for individual region checks (0.0-1.0)"
    )

    # Default weights for scoring regions
    guilloche_weight: float = Field(
        0.35,
        description="Weight for guilloche pattern comparison"
    )

    ghost_photo_weight: float = Field(
        0.25,
        description="Weight for ghost photo detection"
    )

    security_thread_weight: float = Field(
        0.20,
        description="Weight for security thread detection"
    )

    color_profile_weight: float = Field(
        0.20,
        description="Weight for color profile comparison"
    )

    # Face detection settings for ghost photo
    ghost_face_confidence_threshold: float = Field(
        0.5,
        description="Minimum confidence for ghost face detection"
    )

    model_config = {
        "env_prefix": "REFERENCE_",
        "env_file": ".env",
        "extra": "ignore"
    }

    @property
    def default_weights(self) -> dict:
        """Get default weight dictionary."""
        return {
            "guilloche": self.guilloche_weight,
            "ghost_photo": self.ghost_photo_weight,
            "security_thread": self.security_thread_weight,
            "color_profile": self.color_profile_weight
        }


# Global settings instance
reference_settings = ReferenceSettings()

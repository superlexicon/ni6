try:
    from pydantic import BaseSettings, Field
except ImportError:
    from pydantic_settings import BaseSettings
    from pydantic import Field
from typing import Optional


class WorldCheckSettings(BaseSettings):
    """LSEG World Check One API configuration settings (HMAC authentication)"""

    # API Connection Settings
    api_url: str = Field(
        "https://api.risk.lseg.com",
        description="World Check One API base URL"
    )
    gateway_url: str = Field(
        "/screening/v3",
        description="Gateway URL path for World Check One"
    )

    # HMAC Credentials
    api_key: str = Field(
        ...,
        description="API key for HMAC authentication"
    )
    api_secret: str = Field(
        ...,
        description="API secret for HMAC signature"
    )

    # Screening Configuration
    group_id: str = Field(
        ...,
        description="Group ID for screening requests"
    )
    timeout: int = Field(
        30,
        description="API timeout in seconds"
    )
    match_threshold: float = Field(
        0.8,
        description="Minimum match strength to flag as a match (0.0-1.0)"
    )
    name_similarity_threshold: float = Field(
        0.85,
        description="Minimum name similarity (0.0-1.0) to consider a watchlist match as true positive"
    )

    model_config = {
        "env_prefix": "WORLDCHECK_",
        "env_file": ".env",
        "extra": "ignore"
    }


# Global settings instance
worldcheck_settings = WorldCheckSettings()

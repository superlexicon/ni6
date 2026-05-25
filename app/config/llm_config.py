try:
    from pydantic import BaseSettings, Field
except ImportError:
    from pydantic_settings import BaseSettings
    from pydantic import Field
from typing import Optional


class LLMSettings(BaseSettings):
    """LLM API configuration settings for prompt generation"""

    # API Connection Settings
    api_url: str = Field(
        "https://api.openai.com/v1",
        description="LLM API base URL"
    )

    # API Credentials
    api_key: str = Field(
        default="",
        description="API key for LLM service"
    )

    # Model Configuration
    model: str = Field(
        "gpt-4o-mini",
        description="LLM model to use for prompt generation"
    )

    # Generation Configuration
    temperature: float = Field(
        0.3,
        description="Sampling temperature (0.0-1.0), lower for more deterministic output"
    )

    max_tokens: int = Field(
        4000,
        description="Maximum tokens in completion response"
    )

    timeout: int = Field(
        60,
        description="API timeout in seconds"
    )

    # Prompt Generation Settings
    prompt_examples_count: int = Field(
        4,
        description="Number of examples to request in generated prompts"
    )

    default_threshold: float = Field(
        0.3,
        description="Default confidence threshold for GLiNER2 extraction"
    )

    # Retry Configuration
    max_retries: int = Field(
        3,
        description="Maximum number of retry attempts for failed API calls"
    )

    retry_delay: float = Field(
        1.0,
        description="Delay between retries in seconds"
    )

    model_config = {
        "env_prefix": "LLM_",
        "env_file": ".env",
        "extra": "ignore"
    }


# Global settings instance
llm_settings = LLMSettings()

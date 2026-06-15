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

    # Text-Only Model Configuration (for coordinate prediction from OCR blocks)
    text_model: str = Field(
        "llama3.2:3b",
        description="Text-only LLM model for coordinate prediction from OCR blocks (e.g., llama3.2:3b, mistral:7b)"
    )

    # Text Layout Analysis Settings
    enable_text_layout_analysis: bool = Field(
        True,
        description="Enable text-only layout analysis with OCR blocks"
    )

    # Vision Model Configuration (for direct image analysis)
    # RECOMMENDED: Use qwen2.5-vl:7b for better JSON output compliance
    # qwen3-vl:2b has issues with JSON-only output mode
    vision_model: str = Field(
        "qwen2.5-vl:7b",
        description="Vision model for direct image-based layout analysis (e.g., qwen2.5-vl:7b, qwen3-vl:7b). Note: qwen3-vl:2b has JSON output issues."
    )

    enable_vision_layout_analysis: bool = Field(
        True,
        description="Enable vision-based layout analysis with image input"
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

    # Prompt Refinement Settings
    max_prompt_refinement_retries: int = Field(
        3,
        description="Maximum number of retry attempts for prompt refinement"
    )

    prompt_refinement_temperature: float = Field(
        0.3,
        description="Sampling temperature for prompt refinement (lower for more deterministic output)"
    )

    prompt_refinement_max_tokens: int = Field(
        3000,
        description="Maximum tokens in refinement response"
    )

    save_refined_prompts_to_db: bool = Field(
        True,
        description="Save successfully refined prompts to database"
    )

    # Spatial Layout Analysis Settings
    enable_spatial_layout: bool = Field(
        False,
        description="Enable spatial layout analysis with vision models (may not work with all Ollama setups)"
    )

    # Qwen3-VL Direct Extraction Settings
    enable_qwen_direct_extraction: bool = Field(
        True,
        description="Enable Qwen3-VL direct extraction for bank statements"
    )

    qwen_extraction_temperature: float = Field(
        0.3,
        description="Sampling temperature for Qwen3-VL direct extraction (0.0-1.0)"
    )

    qwen_extraction_max_tokens: int = Field(
        8000,
        description="Maximum tokens in Qwen3-VL extraction response"
    )

    qwen_confidence_threshold: float = Field(
        0.70,
        description="Minimum confidence threshold for accepting Qwen3-VL extracted fields (0.0-1.0)"
    )

    model_config = {
        "env_prefix": "LLM_",
        "env_file": ".env",
        "extra": "ignore"
    }


# Global settings instance
llm_settings = LLMSettings()

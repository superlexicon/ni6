try:
    from pydantic import BaseSettings, Field
except ImportError:
    from pydantic_settings import BaseSettings
    from pydantic import Field
from typing import Optional


class PhotoHolmesSettings(BaseSettings):
    """PhotoHolmes forgery detection settings - configurable via environment variables"""

    # Per-method timeouts (seconds)
    # Optimized for M1 MacBook Air running 3 instances
    # Lightweight methods
    dq_method_timeout: int = Field(8, description="DQ method timeout in seconds (lightweight)")
    noisesniffer_method_timeout: int = Field(12, description="NoiseSniffer method timeout in seconds (lightweight)")

    # Moderate complexity
    adaptive_method_timeout: int = Field(25, description="Adaptive method timeout in seconds (moderate)")

    # ONNX-based methods
    trufor_method_timeout: int = Field(30, description="TruFor method timeout in seconds (ONNX)")
    splicebuster_method_timeout: int = Field(30, description="SpliceBuster method timeout in seconds (ONNX)")

    # Memory-intensive PyTorch methods
    psccnet_method_timeout: int = Field(45, description="PSCCNet method timeout in seconds (memory-intensive)")
    focal_method_timeout: int = Field(60, description="FOCAL method timeout in seconds (memory-intensive)")

    # Optional resource-intensive method (disabled by default)
    zero_method_timeout: int = Field(0, description="ZERO method timeout in seconds (0 = disabled)")

    model_config = {
        "env_file": ".env",
        "extra": "ignore"
    }


# Global instance
photoholmes_settings = PhotoHolmesSettings()

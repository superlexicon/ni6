"""
Centralized autocast utility for handling device compatibility issues.

This module provides a DeviceAwareAutocast context manager that gracefully handles
MPS (Metal Performance Shaders) autocast compatibility issues on Apple Silicon.
"""

import torch
from torch.amp.autocast_mode import autocast
from contextlib import nullcontext
from typing import Optional, Any


class DeviceAwareAutocast:
    """
    Device-aware autocast context manager with MPS compatibility handling.

    This class provides intelligent autocast functionality that:
    - Uses autocast normally for CUDA and CPU devices
    - Tests MPS autocast compatibility and falls back gracefully when unsupported
    - Provides minimal logging for critical errors and fallback events
    - Maintains performance optimizations where possible
    """

    def __init__(self, device: str, dtype: torch.dtype = torch.float16, logger: Optional[Any] = None):
        """
        Initialize the DeviceAwareAutocast context manager.

        Args:
            device: The device string (e.g., "cuda:0", "mps", "cpu")
            dtype: The data type for autocast (default: torch.float16)
            logger: Optional logger instance for error reporting
        """
        self.device = device
        self.dtype = dtype
        self.logger = logger
        self.device_type = device.split(":")[0]
        self._use_autocast = True
        self._context = None
        self._fallback_reason = None

        # Determine autocast compatibility during initialization
        self._setup_autocast_context()

    def _setup_autocast_context(self):
        """Set up the appropriate autocast context based on device compatibility."""
        if self.device_type == "cuda":
            self._context = autocast(device_type="cuda", dtype=self.dtype)

        elif self.device_type == "mps":
            # Test MPS autocast compatibility
            try:
                # Quick test to see if MPS autocast is supported
                test_device = torch.device(self.device)
                with autocast(device_type="mps", dtype=self.dtype):
                    test_tensor = torch.randn(1, device=test_device)
                    _ = test_tensor * 2

                self._context = autocast(device_type="mps", dtype=self.dtype)

            except Exception as e:
                self._use_autocast = False
                self._fallback_reason = str(e)
                self._context = nullcontext()

                # Log the fallback with minimal detail (this is expected behavior on Apple Silicon)
                if self.logger:
                    self.logger.debug(f"MPS autocast not supported, falling back to CPU execution: {self._fallback_reason}")

        else:  # CPU
            self._context = autocast(device_type="cpu", dtype=self.dtype)

    def __enter__(self):
        """Enter the autocast context."""
        return self._context.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the autocast context."""
        return self._context.__exit__(exc_type, exc_val, exc_tb)


def create_device_aware_autocast(device: str, dtype: torch.dtype = torch.float16, logger: Optional[Any] = None) -> DeviceAwareAutocast:
    """
    Factory function to create a DeviceAwareAutocast context manager.

    Args:
        device: The device string (e.g., "cuda:0", "mps", "cpu")
        dtype: The data type for autocast (default: torch.float16)
        logger: Optional logger instance for error reporting

    Returns:
        DeviceAwareAutocast context manager instance
    """
    return DeviceAwareAutocast(device, dtype, logger)
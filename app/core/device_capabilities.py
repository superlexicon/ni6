"""
Device capability detection for intelligent model selection based on GPU memory and performance.
"""
import torch
import psutil
from typing import Dict, List, Optional, Tuple
from app.core.logger import get_logger


class DeviceCapabilities:
    """Detect device capabilities and recommend appropriate models/features."""

    def __init__(self):
        self.logger = get_logger()
        self.capabilities = self._detect_capabilities()

    def _detect_capabilities(self) -> Dict[str, any]:
        """Detect system capabilities for model selection."""
        capabilities = {
            'device': self._get_device(),
            'gpu_memory_gb': self._get_gpu_memory(),
            'system_memory_gb': self._get_system_memory(),
            'cpu_cores': self._get_cpu_cores(),
            'has_cuda': torch.cuda.is_available(),
            'has_mps': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
            'has_rocm': self._has_rocm(),
        }

        self.logger.info(f"Device capabilities detected: {capabilities}")
        return capabilities

    def _get_device(self) -> str:
        """Get the optimal device for computation."""
        # Check for ROCm first since it uses CUDA interface
        if self._has_rocm():
            return "cuda:0"  # ROCm uses CUDA interface in PyTorch
        elif torch.cuda.is_available():
            return "cuda:0"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _has_rocm(self) -> bool:
        """Check if ROCm is available."""
        # Check PyTorch for ROCm support
        if hasattr(torch.version, 'hip') and torch.version.hip:
            return True

        # Check for ROCm environment variables
        import os
        if os.environ.get('ROCM_PATH') or os.environ.get('HSA_OVERRIDE_GFX_VERSION'):
            return True

        # Check for ROCm installation
        rocm_path = os.environ.get('ROCM_PATH', '/opt/rocm')
        if os.path.exists(f"{rocm_path}/bin/rocm-smi"):
            return True

        return False

    def _get_gpu_memory(self) -> float:
        """Get GPU memory in GB."""
        try:
            if torch.cuda.is_available():
                # This works for both CUDA and ROCm since ROCm uses CUDA interface
                total_memory = torch.cuda.get_device_properties(0).total_memory
                return total_memory / (1024**3)  # Convert to GB
            return 0.0
        except Exception as e:
            self.logger.warning(f"Failed to detect GPU memory: {e}")
            return 0.0

    def _get_system_memory(self) -> float:
        """Get system memory in GB."""
        try:
            return psutil.virtual_memory().total / (1024**3)
        except Exception as e:
            self.logger.warning(f"Failed to detect system memory: {e}")
            return 0.0

    def _get_cpu_cores(self) -> int:
        """Get number of CPU cores."""
        try:
            return psutil.cpu_count(logical=False) or psutil.cpu_count() or 1
        except Exception as e:
            self.logger.warning(f"Failed to detect CPU cores: {e}")
            return 1

    def get_recommended_models(self) -> Dict[str, bool]:
        """Get recommended model configuration based on device capabilities."""
        recommendations = {
            'focal': False,      # FOCAL - requires GPU, memory intensive
            'cnet': False,       # PSCCNet - requires GPU, memory intensive
            'adaptive': False,   # Adaptive CFA - moderate requirements
            'exif': True,        # EXIF As Language - lightweight
            'doctr': True,       # Doctr - moderate requirements
            'dq': True,          # DQ Method - lightweight
            'noise_sniffer': True, # NoiseSniffer - lightweight
        }

        # GPU-based recommendations (for both CUDA and ROCm)
        if (self.capabilities['has_cuda'] or self.capabilities['has_rocm']) and self.capabilities['gpu_memory_gb'] > 0:
            gpu_memory = self.capabilities['gpu_memory_gb']
            gpu_type = "ROCm" if self.capabilities['has_rocm'] else "CUDA"

            # Enable memory-intensive models based on GPU memory
            if gpu_memory >= 8.0:  # 8GB+ GPU
                recommendations['focal'] = True
                recommendations['cnet'] = True
                recommendations['adaptive'] = True
                self.logger.info(f"{gpu_type} GPU {gpu_memory:.1f}GB detected: Enabling all models including FOCAL and CNET")
            elif gpu_memory >= 4.0:  # 4GB+ GPU
                recommendations['cnet'] = True
                recommendations['adaptive'] = True
                self.logger.info(f"{gpu_type} GPU {gpu_memory:.1f}GB detected: Enabling CNET and Adaptive, disabling FOCAL")
            else:
                self.logger.info(f"{gpu_type} GPU {gpu_memory:.1f}GB detected: Disabling memory-intensive models")

        # CPU-only recommendations
        else:
            system_memory = self.capabilities['system_memory_gb']

            if system_memory >= 16.0:  # 16GB+ system memory
                recommendations['adaptive'] = True
                self.logger.info(f"CPU with {system_memory:.1f}GB RAM: Enabling Adaptive method")
            else:
                self.logger.info(f"CPU with {system_memory:.1f}GB RAM: Using lightweight models only")

        return recommendations

    def get_processing_recommendations(self) -> Dict[str, any]:
        """Get processing recommendations based on capabilities."""
        return {
            'max_image_size': self._get_max_image_size(),
            'batch_size': self._get_batch_size(),
            'timeout_multiplier': self._get_timeout_multiplier(),
            'use_mixed_precision': self.capabilities['has_cuda'],
        }

    def _get_max_image_size(self) -> int:
        """Get recommended maximum image size."""
        if self.capabilities['gpu_memory_gb'] >= 8.0:
            return 1024
        elif self.capabilities['gpu_memory_gb'] >= 4.0:
            return 768
        elif self.capabilities['system_memory_gb'] >= 16.0:
            return 512
        return 256

    def _get_batch_size(self) -> int:
        """Get recommended batch size."""
        if self.capabilities['gpu_memory_gb'] >= 16.0:
            return 8
        elif self.capabilities['gpu_memory_gb'] >= 8.0:
            return 4
        elif self.capabilities['gpu_memory_gb'] >= 4.0:
            return 2
        return 1

    def _get_timeout_multiplier(self) -> float:
        """Get timeout multiplier based on device speed."""
        if self.capabilities['has_cuda']:
            if self.capabilities['gpu_memory_gb'] >= 8.0:
                return 1.0  # Fast GPU
            else:
                return 1.5  # Slower GPU
        else:
            if self.capabilities['system_memory_gb'] >= 16.0:
                return 2.0  # Decent CPU
            else:
                return 3.0  # Slower CPU

    def should_enable_model(self, model_name: str) -> bool:
        """Check if a specific model should be enabled."""
        recommendations = self.get_recommended_models()
        return recommendations.get(model_name, False)

    def get_capability_summary(self) -> str:
        """Get human-readable summary of device capabilities."""
        if self.capabilities['has_rocm']:
            return f"AMD GPU ({self.capabilities['gpu_memory_gb']:.1f}GB VRAM, {self.capabilities['system_memory_gb']:.1f}GB RAM)"
        elif self.capabilities['has_cuda']:
            return f"NVIDIA GPU ({self.capabilities['gpu_memory_gb']:.1f}GB VRAM, {self.capabilities['system_memory_gb']:.1f}GB RAM)"
        elif self.capabilities['has_mps']:
            return f"MPS ({self.capabilities['system_memory_gb']:.1f}GB RAM)"
        else:
            return f"CPU ({self.capabilities['cpu_cores']} cores, {self.capabilities['system_memory_gb']:.1f}GB RAM)"


# Global instance for easy access
_device_capabilities = None

def get_device_capabilities() -> DeviceCapabilities:
    """Get global device capabilities instance."""
    global _device_capabilities
    if _device_capabilities is None:
        _device_capabilities = DeviceCapabilities()
    return _device_capabilities
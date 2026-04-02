"""
Unified Framework Coordinator for GPU Device Configuration

This module provides centralized coordination between TensorFlow, PyTorch, and ONNX Runtime
to eliminate device configuration conflicts and optimize resource utilization.
"""

import os
import threading
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass
from app.core.logger import get_logger
from app.core.device_manager import DeviceManager, DeviceType, DeviceInfo
from app.utils.gpu_optimizations import get_gpu_optimizations


class FrameworkType(Enum):
    """Supported ML frameworks"""
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    ONNX_RUNTIME = "onnxruntime"


@dataclass
class FrameworkConfig:
    """Framework-specific configuration"""
    framework_type: FrameworkType
    device_type: DeviceType
    device_id: int
    memory_limit_mb: Optional[int] = None
    memory_growth_enabled: bool = True
    mixed_precision_enabled: bool = False
    execution_providers: List[str] = None

    def __post_init__(self):
        if self.execution_providers is None:
            self.execution_providers = []


class FrameworkCoordinator:
    """
    Unified coordinator for ML framework device configuration.

    This class manages device allocation and configuration across TensorFlow,
    PyTorch, and ONNX Runtime to prevent conflicts and optimize performance.
    """

    _instance: Optional['FrameworkCoordinator'] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self.logger = get_logger()
            self.device_manager = DeviceManager()
            self.gpu_optimizer = get_gpu_optimizations()
            self._framework_configs: Dict[FrameworkType, FrameworkConfig] = {}
            self._allocated_devices: Dict[DeviceType, List[FrameworkType]] = {}
            self._memory_tracker: Dict[FrameworkType, int] = {}

            # Enable framework-specific GPU optimizations
            self._enable_framework_optimizations()

            # Initialize default configurations
            self._initialize_default_configs()

    def _initialize_default_configs(self):
        """Initialize default framework configurations based on available devices"""
        recommended_device = self.device_manager.get_recommended_device(prefer_gpu=True)

        # Set memory limits based on GPU memory (MB)
        memory_limit = None
        if recommended_device.memory_total:
            # Reserve 2GB for system and other frameworks
            memory_limit = max(2048, recommended_device.memory_total - 2048)

        # Default memory allocation priorities (in MB)
        self._framework_memory_requirements = {
            FrameworkType.PYTORCH: 1200,    # PhotoHolmes models
            FrameworkType.TENSORFLOW: 800,   # DeepFace models
            FrameworkType.ONNX_RUNTIME: 400  # DocTR models
        }

        self.logger.info(f"FrameworkCoordinator initialized with device: {recommended_device.device_type.value}")

    def _enable_framework_optimizations(self):
        """Enable framework-specific GPU optimizations."""
        try:
            self.logger.info("Enabling framework-specific GPU optimizations...")

            # Enable PyTorch optimizations
            pytorch_opts = self.gpu_optimizer.enable_pytorch_optimizations()
            self.logger.info(f"PyTorch optimizations: {pytorch_opts.get('status', 'unknown')}")

            # Enable TensorFlow optimizations
            tf_opts = self.gpu_optimizer.enable_tensorflow_optimizations()
            self.logger.info(f"TensorFlow optimizations: {tf_opts.get('status', 'unknown')}")

            # Enable ONNX Runtime optimizations
            onnx_opts = self.gpu_optimizer.enable_onnx_optimizations()
            self.logger.info(f"ONNX Runtime optimizations: {onnx_opts.get('status', 'unknown')}")

            # Enable environment optimizations
            env_opts = self.gpu_optimizer.enable_environment_optimizations()
            self.logger.info(f"Environment optimizations: {env_opts.get('status', 'unknown')}")

            self.logger.info("Framework-specific GPU optimizations enabled")

        except Exception as e:
            self.logger.error(f"Failed to enable framework optimizations: {e}")

    def configure_framework(
        self,
        framework_type: FrameworkType,
        device_type: Optional[DeviceType] = None,
        device_id: int = 0,
        memory_limit_mb: Optional[int] = None,
        **kwargs
    ) -> FrameworkConfig:
        """
        Configure a specific framework with device settings.

        Args:
            framework_type: The framework to configure
            device_type: Device type to use (auto-detected if None)
            device_id: Device ID
            memory_limit_mb: Memory limit in MB
            **kwargs: Additional framework-specific settings

        Returns:
            FrameworkConfig with the applied settings
        """
        with self._lock:
            # Auto-detect device if not specified
            if device_type is None:
                recommended_device = self.device_manager.get_recommended_device(prefer_gpu=True)
                device_type = recommended_device.device_type
                device_id = recommended_device.device_id

            # Calculate memory limit if not specified
            if memory_limit_mb is None:
                memory_limit_mb = self._calculate_memory_limit(framework_type, device_type)

            # Create framework configuration
            config = FrameworkConfig(
                framework_type=framework_type,
                device_type=device_type,
                device_id=device_id,
                memory_limit_mb=memory_limit_mb,
                memory_growth_enabled=kwargs.get('memory_growth_enabled', True),
                mixed_precision_enabled=kwargs.get('mixed_precision_enabled', False),  # Disable mixed precision for DeepFace compatibility
                execution_providers=self.gpu_optimizer.get_optimized_providers() if framework_type == FrameworkType.ONNX_RUNTIME else self._get_default_execution_providers(framework_type, device_type)
            )

            self._framework_configs[framework_type] = config

            # Track device allocation
            if device_type not in self._allocated_devices:
                self._allocated_devices[device_type] = []
            if framework_type not in self._allocated_devices[device_type]:
                self._allocated_devices[device_type].append(framework_type)

            self.logger.info(f"Configured {framework_type.value} for {device_type.value}:{device_id} "
                           f"with {memory_limit_mb}MB memory limit")

            return config

    def apply_tensorflow_config(self) -> bool:
        """
        Apply TensorFlow GPU configuration.

        Returns:
            True if configuration was successful
        """
        try:
            import tensorflow as tf

            config = self._framework_configs.get(FrameworkType.TENSORFLOW)
            if not config:
                config = self.configure_framework(FrameworkType.TENSORFLOW)

            if config.device_type == DeviceType.CUDA:
                # Configure TensorFlow for CUDA
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    try:
                        # Enable memory growth for optimal GPU utilization
                        tf.config.experimental.set_memory_growth(gpus[0], config.memory_growth_enabled)

                        # ⚡ OPTIMIZATION: Remove virtual device configuration for better performance
                        # Virtual devices cause memory fragmentation and slow down DeepFace
                        # Let TensorFlow use the full GPU with memory growth instead

                        # Enable mixed precision if requested
                        if config.mixed_precision_enabled:
                            tf.keras.mixed_precision.set_global_policy('mixed_float16')

                        self.logger.info("TensorFlow GPU configuration applied successfully")
                        return True

                    except RuntimeError as e:
                        self.logger.warning(f"TensorFlow GPU configuration failed: {e}")
                        return False

            elif config.device_type == DeviceType.MPS:
                # Configure TensorFlow for Apple Silicon with DeepFace compatibility
                try:
                    # SKIP TensorFlow session reset - causes KerasTensor conflicts with DeepFace
                    # DeepFace needs the existing TensorFlow graph to remain intact
                    # tf.keras.backend.clear_session()  # DISABLED - prevents KerasTensor errors

                    # Set device placement settings
                    tf.config.set_soft_device_placement(True)

                    # Disable all aggressive optimizations that cause KerasTensor issues
                    if hasattr(tf.config.optimizer, 'set_jit'):
                        tf.config.optimizer.set_jit(False)  # Disable XLA compilation

                    if hasattr(tf.config.experimental, 'disable_mixed_precision_graph_rewrite'):
                        try:
                            tf.config.experimental.disable_mixed_precision_graph_rewrite()
                        except:
                            pass  # May not be available in all versions

                    self.logger.info("TensorFlow MPS configuration applied with DeepFace compatibility")
                    return True

                except Exception as e:
                    self.logger.warning(f"TensorFlow MPS configuration failed: {e}")
                    # Fallback to CPU configuration
                    try:
                        tf.config.set_visible_devices([], 'GPU')
                        tf.config.set_visible_devices([], 'MPS')
                        self.logger.info("TensorFlow configured for CPU fallback")
                        return True
                    except Exception as fallback_error:
                        self.logger.error(f"TensorFlow CPU fallback also failed: {fallback_error}")
                        return False

            # Fallback to CPU
            self.logger.info("TensorFlow configured for CPU execution")
            return True

        except ImportError:
            self.logger.error("TensorFlow not available")
            return False
        except Exception as e:
            self.logger.error(f"Failed to apply TensorFlow configuration: {e}")
            return False

    def apply_pytorch_config(self) -> bool:
        """
        Apply PyTorch GPU configuration.

        Returns:
            True if configuration was successful
        """
        try:
            import torch

            config = self._framework_configs.get(FrameworkType.PYTORCH)
            if not config:
                config = self.configure_framework(FrameworkType.PYTORCH)

            if config.device_type == DeviceType.CUDA:
                # Configure PyTorch for CUDA
                if torch.cuda.is_available():
                    device = torch.device(f"cuda:{config.device_id}")

                    # Enable memory-efficient settings
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cudnn.deterministic = False

                    # Enable mixed precision if requested
                    if config.mixed_precision_enabled:
                        torch.set_float32_matmul_precision('medium')

                    self.logger.info(f"PyTorch configured for CUDA:{config.device_id}")
                    return device
                else:
                    self.logger.warning("CUDA not available for PyTorch")
                    return torch.device("cpu")

            elif config.device_type == DeviceType.ROCM:
                # Configure PyTorch for ROCm
                if hasattr(torch.version, 'hip') and torch.version.hip and torch.cuda.is_available():
                    device = torch.device(f"cuda:{config.device_id}")  # ROCm uses CUDA interface

                    # ROCm-specific optimizations
                    # Note: cuDNN settings don't apply to ROCm

                    # Configure MIOpen for optimal performance
                    # IMPORTANT: First run will be slow (~60s) as MIOpen benchmarks kernels
                    # Subsequent runs will be fast (~2-3s) using cached optimal kernels
                    # Enable find mode for better kernel selection
                    os.environ['MIOPEN_FIND_MODE'] = 'NORMAL'
                    # Allow MIOpen to use cached optimizations
                    os.environ['MIOPEN_DEBUG_DISABLE_FIND_DB'] = '0'
                    # Set user database path for persistent kernel cache
                    if not os.environ.get('MIOPEN_USER_DB_PATH'):
                        miopen_cache_dir = '/tmp/miopen-cache'
                        os.environ['MIOPEN_USER_DB_PATH'] = miopen_cache_dir
                        # Create cache directory if it doesn't exist
                        os.makedirs(miopen_cache_dir, exist_ok=True)
                        self.logger.info(f"MIOpen cache directory: {miopen_cache_dir}")

                    # Enable PyTorch memory allocator optimizations
                    # This helps MIOpen get workspace memory
                    torch.backends.cudnn.benchmark = True  # Works for ROCm via CUDA interface

                    # Pre-allocate memory pool to reduce allocation overhead
                    try:
                        torch.cuda.empty_cache()
                        # Set max split size to help with large workspace allocations
                        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
                    except Exception as e:
                        self.logger.warning(f"Could not configure CUDA memory: {e}")

                    # Enable mixed precision if requested
                    if config.mixed_precision_enabled:
                        torch.set_float32_matmul_precision('medium')

                    self.logger.info(f"PyTorch configured for ROCm device:{config.device_id} with MIOpen optimizations")
                    return device
                else:
                    self.logger.warning("ROCm not available for PyTorch - using CPU")
                    return torch.device("cpu")

            elif config.device_type == DeviceType.MPS:
                # Configure PyTorch for Apple Silicon
                if torch.backends.mps.is_available():
                    device = torch.device("mps")
                    self.logger.info("PyTorch configured for MPS")
                    return device
                else:
                    self.logger.warning("MPS not available for PyTorch")
                    return torch.device("cpu")

            # Fallback to CPU
            self.logger.info("PyTorch configured for CPU")
            return torch.device("cpu")

        except ImportError:
            self.logger.error("PyTorch not available")
            return False
        except Exception as e:
            self.logger.error(f"Failed to apply PyTorch configuration: {e}")
            return False

    def apply_onnx_config(self) -> List[str]:
        """
        Apply ONNX Runtime configuration.

        Returns:
            List of execution providers to use
        """
        try:
            import onnxruntime as ort

            config = self._framework_configs.get(FrameworkType.ONNX_RUNTIME)
            if not config:
                config = self.configure_framework(FrameworkType.ONNX_RUNTIME)

            providers = config.execution_providers

            # Validate and adjust providers based on availability
            available_providers = ort.get_available_providers()

            # Filter available providers
            valid_providers = [p for p in providers if p in available_providers]

            if not valid_providers:
                # Fallback to CPU
                valid_providers = ['CPUExecutionProvider']
                self.logger.warning("ONNX Runtime falling back to CPU provider")

            self.logger.info(f"ONNX Runtime configured with providers: {valid_providers}")
            return valid_providers

        except ImportError:
            self.logger.error("ONNX Runtime not available")
            return ['CPUExecutionProvider']
        except Exception as e:
            self.logger.error(f"Failed to apply ONNX Runtime configuration: {e}")
            return ['CPUExecutionProvider']

    def _calculate_memory_limit(self, framework_type: FrameworkType, device_type: DeviceType) -> int:
        """Calculate memory limit for a framework based on available GPU memory"""
        if device_type == DeviceType.CPU:
            return None

        # Get total GPU memory
        devices = self.device_manager.detect_available_devices()
        if device_type not in devices or not devices[device_type]:
            return self._framework_memory_requirements[framework_type]

        device_info = devices[device_type][0]
        if not device_info.memory_total:
            return self._framework_memory_requirements[framework_type]

        # Reserve memory for system and other frameworks
        total_framework_memory = sum(self._framework_memory_requirements.values())
        reserve_memory = 2048  # 2GB reserve

        if device_info.memory_total > total_framework_memory + reserve_memory:
            # Enough memory for all frameworks with full allocation
            return self._framework_memory_requirements[framework_type]
        else:
            # Limited memory - allocate proportionally
            available_memory = device_info.memory_total - reserve_memory
            proportion = self._framework_memory_requirements[framework_type] / total_framework_memory
            return max(512, int(available_memory * proportion))  # Minimum 512MB

    def _get_default_execution_providers(self, framework_type: FrameworkType, device_type: DeviceType) -> List[str]:
        """Get default execution providers for ONNX Runtime"""
        if framework_type != FrameworkType.ONNX_RUNTIME:
            return []

        providers = []

        if device_type == DeviceType.CUDA:
            providers.extend([
                'CUDAExecutionProvider',
                'TensorrtExecutionProvider'  # Will be ignored if TensorRT not available
            ])
        elif device_type == DeviceType.ROCM:
            # ROCm execution provider requires custom ONNX Runtime build
            # For now, fall back to CPU
            self.logger.info("ROCm execution provider not available in standard ONNX Runtime")
        elif device_type == DeviceType.MPS:
            # Apple Silicon: Try CoreML first, then fallback to CPU
            try:
                import onnxruntime as ort
                available = ort.get_available_providers()
                if 'CoreMLExecutionProvider' in available:
                    providers.append('CoreMLExecutionProvider')
                    self.logger.info("Using CoreML Execution Provider for Apple Silicon")
                else:
                    self.logger.info("CoreML Execution Provider not available, using CPU")
            except Exception as e:
                self.logger.warning(f"CoreML provider not available: {e}")

        providers.append('CPUExecutionProvider')  # Always include as fallback
        return providers

    def get_framework_config(self, framework_type: FrameworkType) -> Optional[FrameworkConfig]:
        """Get the current configuration for a framework"""
        return self._framework_configs.get(framework_type)

    def get_device_string(self, framework_type: FrameworkType) -> str:
        """Get device string for a specific framework"""
        config = self.get_framework_config(framework_type)
        if not config:
            # Configure with defaults
            config = self.configure_framework(framework_type)

        if config.device_type == DeviceType.CUDA:
            return f"cuda:{config.device_id}"
        elif config.device_type == DeviceType.ROCM:
            # ROCm uses CUDA interface in PyTorch
            return f"cuda:{config.device_id}"
        elif config.device_type == DeviceType.MPS:
            return "mps"
        else:
            return "cpu"

    def coordinate_all_frameworks(self) -> Dict[str, bool]:
        """
        Apply configuration to all frameworks.

        Returns:
            Dictionary mapping framework names to success status
        """
        results = {}

        # Configure TensorFlow
        results['tensorflow'] = self.apply_tensorflow_config()

        # Configure PyTorch
        pytorch_result = self.apply_pytorch_config()
        results['pytorch'] = pytorch_result is not False

        # Configure ONNX Runtime
        onnx_providers = self.apply_onnx_config()
        results['onnxruntime'] = len(onnx_providers) > 0

        success_count = sum(1 for success in results.values() if success)
        self.logger.info(f"Framework coordination completed: {success_count}/{len(results)} frameworks configured successfully")

        return results

    def get_memory_usage_report(self) -> Dict[str, Any]:
        """Get a report of current memory usage by framework"""
        report = {
            'framework_configs': {},
            'device_allocations': self._allocated_devices,
            'total_allocated_memory_mb': sum(self._memory_tracker.values())
        }

        for framework_type, config in self._framework_configs.items():
            report['framework_configs'][framework_type.value] = {
                'device_type': config.device_type.value,
                'device_id': config.device_id,
                'memory_limit_mb': config.memory_limit_mb,
                'memory_growth_enabled': config.memory_growth_enabled,
                'mixed_precision_enabled': config.mixed_precision_enabled,
                'execution_providers': config.execution_providers
            }

        return report


# Global instance
_framework_coordinator: Optional[FrameworkCoordinator] = None


def get_framework_coordinator() -> FrameworkCoordinator:
    """Get the global framework coordinator instance"""
    global _framework_coordinator
    if _framework_coordinator is None:
        _framework_coordinator = FrameworkCoordinator()
    return _framework_coordinator


def coordinate_frameworks() -> Dict[str, bool]:
    """Convenience function to coordinate all frameworks"""
    coordinator = get_framework_coordinator()
    return coordinator.coordinate_all_frameworks()


def get_framework_device(framework_type: FrameworkType) -> str:
    """Get device string for a specific framework"""
    coordinator = get_framework_coordinator()
    return coordinator.get_device_string(framework_type)
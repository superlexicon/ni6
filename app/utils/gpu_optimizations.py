"""
Framework-Specific GPU Optimizations

This module provides GPU optimization utilities for TensorFlow and PyTorch
to maximize performance on RTX 4000 and similar GPUs.

Expected Improvement: 20-50% per framework
"""

import os
from typing import Dict, Any, Optional, List
from app.core.logger import get_logger


class GPUOptimizations:
    """Framework-specific GPU optimization utilities."""

    def __init__(self):
        self.logger = get_logger()

    def enable_pytorch_optimizations(self) -> Dict[str, Any]:
        """
        Enable PyTorch-specific GPU optimizations.

        Returns:
            Dict with optimization status and details
        """
        optimizations = {}

        try:
            import torch

            if torch.cuda.is_available():
                # Enable cuDNN auto-tuner for best performance
                torch.backends.cudnn.benchmark = True
                optimizations['cudnn_benchmark'] = True

                # Enable cuDNN deterministic for reproducible results when needed
                torch.backends.cudnn.deterministic = False
                optimizations['cudnn_deterministic'] = False

                # Enable TF32 for better performance on RTX GPUs (Tensor Cores)
                if hasattr(torch.backends.cudnn, 'allow_tf32'):
                    torch.backends.cudnn.allow_tf32 = True
                    optimizations['cudnn_tf32'] = True

                if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
                    torch.backends.cuda.matmul.allow_tf32 = True
                    optimizations['matmul_tf32'] = True

                # Set memory allocation strategy for better GPU utilization
                if hasattr(torch.cuda, 'memory'):
                    torch.cuda.memory.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory
                    optimizations['memory_fraction'] = 0.9

                # Enable mixed precision training/inference if available
                if hasattr(torch, 'cuda') and hasattr(torch.cuda, 'amp'):
                    optimizations['mixed_precision_available'] = True
                    # Note: Actual mixed precision usage should be handled in individual models

                # Enable torch.compile for PyTorch 2.0+ if available
                if hasattr(torch, 'compile'):
                    optimizations['torch_compile_available'] = True
                    self.logger.info("PyTorch torch.compile available for additional performance")

                # Optimize memory allocation
                if hasattr(torch.cuda, 'empty_cache'):
                    # Clear any existing memory
                    torch.cuda.empty_cache()
                    optimizations['memory_cleanup'] = True

                self.logger.info("PyTorch GPU optimizations enabled successfully")
                optimizations['status'] = 'success'

            else:
                self.logger.warning("CUDA not available for PyTorch optimizations")
                optimizations['status'] = 'cuda_not_available'

        except ImportError:
            self.logger.error("PyTorch not available for optimizations")
            optimizations['status'] = 'pytorch_not_available'
        except Exception as e:
            self.logger.error(f"Failed to enable PyTorch optimizations: {e}")
            optimizations['status'] = 'failed'
            optimizations['error'] = str(e)

        return optimizations

    def enable_tensorflow_optimizations(self) -> Dict[str, Any]:
        """
        Enable TensorFlow-specific GPU optimizations.

        Returns:
            Dict with optimization status and details
        """
        optimizations = {}

        try:
            import tensorflow as tf

            # Clear any existing TensorFlow session to avoid KerasTensor conflicts
            tf.keras.backend.clear_session()

            # Get available GPUs
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                # Enable memory growth to prevent TensorFlow from allocating all GPU memory
                for gpu in gpus:
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                        optimizations['memory_growth'] = True
                    except RuntimeError as e:
                        self.logger.warning(f"Failed to set memory growth: {e}")

                # XLA compilation disabled for DeepFace compatibility
                # Can cause KerasTensor issues with DeepFace models
                optimizations['xla_compilation'] = 'disabled_for_deepface_compatibility'
                self.logger.info("XLA compilation disabled for DeepFace compatibility")

                # Note: Mixed precision disabled for DeepFace compatibility
                # DeepFace models have issues with mixed precision enabled
                optimizations['mixed_precision'] = 'disabled_for_deepface_compatibility'
                self.logger.info("Mixed precision disabled for DeepFace compatibility")

                # Set GPU thread pool size for better performance (disabled for DeepFace compatibility)
                # Thread pool optimization can interfere with DeepFace model loading
                optimizations['thread_pool_size'] = 'disabled_for_deepface_compatibility'
                self.logger.info("Thread pool optimization disabled for DeepFace compatibility")

                self.logger.info("TensorFlow GPU optimizations enabled successfully")
                optimizations['status'] = 'success'
                optimizations['gpu_count'] = len(gpus)

            else:
                self.logger.warning("No GPUs found for TensorFlow optimizations")
                optimizations['status'] = 'no_gpus'

        except ImportError:
            self.logger.error("TensorFlow not available for optimizations")
            optimizations['status'] = 'tensorflow_not_available'
        except Exception as e:
            self.logger.error(f"Failed to enable TensorFlow optimizations: {e}")
            optimizations['status'] = 'failed'
            optimizations['error'] = str(e)

        return optimizations

    def enable_environment_optimizations(self) -> Dict[str, Any]:
        """
        Enable environment-level GPU optimizations.

        Returns:
            Dict with applied optimizations
        """
        optimizations = {}

        try:
            # NVIDIA GPU optimizations
            if os.environ.get('CUDA_VISIBLE_DEVICES') is None:
                # Use first GPU by default
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                optimizations['cuda_visible_devices'] = '0'

            # CUDA optimizations
            os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async CUDA operations
            optimizations['cuda_launch_blocking'] = '0'

            # PyTorch optimizations
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'  # Better memory allocation
            optimizations['pytorch_cuda_alloc_conf'] = 'max_split_size_mb:512'

            # Set number of threads for parallel operations
            os.environ['OMP_NUM_THREADS'] = '4'
            os.environ['MKL_NUM_THREADS'] = '4'
            optimizations['thread_count'] = '4'

            self.logger.info("Environment-level GPU optimizations enabled")
            optimizations['status'] = 'success'

        except Exception as e:
            self.logger.error(f"Failed to set environment optimizations: {e}")
            optimizations['status'] = 'failed'
            optimizations['error'] = str(e)

        return optimizations

    def compile_pytorch_model(self, model, example_input: Optional[Any] = None, mode: str = 'default') -> Any:
        """
        Compile a PyTorch model using torch.compile for better performance.

        Args:
            model: PyTorch model to compile
            example_input: Example input for tracing (optional)
            mode: Compilation mode ('default', 'reduce-overhead', 'max-autotune')

        Returns:
            Compiled model or original model if compilation fails
        """
        try:
            import torch

            if not hasattr(torch, 'compile'):
                self.logger.warning("torch.compile not available (requires PyTorch 2.0+)")
                return model

            # Choose compilation mode
            compilation_modes = {
                'default': 'default',
                'reduce-overhead': 'reduce-overhead',
                'max-autotune': 'max-autotune'
            }

            compile_mode = compilation_modes.get(mode, 'default')

            # Compile the model
            self.logger.info(f"Compiling PyTorch model with mode: {compile_mode}")

            if example_input is not None:
                # Use example input for better optimization
                compiled_model = torch.compile(
                    model,
                    mode=compile_mode,
                    fullgraph=True
                )
            else:
                # Compile without example input
                compiled_model = torch.compile(
                    model,
                    mode=compile_mode
                )

            self.logger.info("PyTorch model compilation successful")
            return compiled_model

        except Exception as e:
            self.logger.warning(f"PyTorch model compilation failed: {e}")
            return model

    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive GPU memory summary.

        Returns:
            Dict with memory usage details
        """
        memory_summary = {}

        try:
            import torch

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    device_memory = {
                        'allocated': torch.cuda.memory_allocated(i),
                        'cached': torch.cuda.memory_reserved(i),
                        'max_allocated': torch.cuda.max_memory_allocated(i),
                        'max_cached': torch.cuda.max_memory_reserved(i),
                        'total': torch.cuda.get_device_properties(i).total_memory
                    }

                    # Convert to GB for readability
                    for key, value in device_memory.items():
                        device_memory[f"{key}_gb"] = value / (1024**3)

                    memory_summary[f"cuda_{i}"] = device_memory

            return memory_summary

        except ImportError:
            memory_summary['error'] = 'PyTorch not available'
        except Exception as e:
            memory_summary['error'] = str(e)

        return memory_summary


# Global GPU optimizations instance
_gpu_optimizations: Optional[GPUOptimizations] = None


def get_gpu_optimizations() -> GPUOptimizations:
    """Get the global GPU optimizations instance."""
    global _gpu_optimizations
    if _gpu_optimizations is None:
        _gpu_optimizations = GPUOptimizations()
    return _gpu_optimizations


def enable_all_framework_optimizations() -> Dict[str, Any]:
    """
    Enable GPU optimizations for all supported frameworks.

    Returns:
        Dict with optimization results for each framework
    """
    optimizer = get_gpu_optimizations()
    results = {}

    # Enable PyTorch optimizations
    results['pytorch'] = optimizer.enable_pytorch_optimizations()

    # Enable TensorFlow optimizations
    results['tensorflow'] = optimizer.enable_tensorflow_optimizations()

    # Enable environment optimizations
    results['environment'] = optimizer.enable_environment_optimizations()

    return results
"""
Centralized GPU Resource Manager for IM-OSINT

This module provides a unified interface for managing GPU resources across
PhotoHolmes (PyTorch), DeepFace (TensorFlow), and DocTR (ONNX Runtime)
to prevent memory conflicts and optimize resource allocation.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Type, TypeVar, Union
import threading
import gc

import torch
import tensorflow as tf
import onnxruntime as ort

from app.core.device_manager import DeviceInfo, DeviceType, get_recommended_device

logger = logging.getLogger(__name__)

T = TypeVar('T')

class ModelType(Enum):
    """Enum for different CV model types"""
    PHOTOHOLMES = "photoholmes"
    PHOTOHOLMES_PSccNET = "photoholmes_psccnet"
    PHOTOHOLMES_ADAPTIVE_CFA = "photoholmes_adaptive_cfa"
    DEEPFACE_FACE_DETECTION = "deepface_face_detection"
    DEEPFACE_FACE_VERIFICATION = "deepface_face_verification"
    DOCTR_OCR = "doctr_ocr"
    BERT_NER = "bert_ner"
    GLINER_NER = "gliner_ner"


@dataclass
class ModelAllocation:
    """Model allocation information"""
    model_type: ModelType
    model_instance: Any
    device_type: DeviceType
    device_id: Optional[int]
    memory_usage_mb: Optional[float]
    last_access_time: float
    allocation_time: float
    is_active: bool = True


@dataclass
class GPUMemoryInfo:
    """GPU memory information"""
    total_memory_mb: float
    allocated_memory_mb: float
    free_memory_mb: float
    utilization_percent: float


class GPUResourceManager:
    """
    Centralized GPU resource manager for coordinating model allocation
    across PhotoHolmes, DeepFace, and DocTR frameworks.
    """

    def __init__(self, memory_threshold: float = 0.8, model_timeout: int = 300):
        """
        Initialize GPU resource manager.

        Args:
            memory_threshold: GPU memory usage threshold (0.0-1.0)
            model_timeout: Model inactivity timeout in seconds
        """
        self.device_info = get_recommended_device()
        self.memory_threshold = memory_threshold
        self.model_timeout = model_timeout

        self.active_models: Dict[ModelType, ModelAllocation] = {}
        self.memory_lock = asyncio.Lock()
        self.cleanup_task: Optional[asyncio.Task] = None
        self._background_cleanup_running = False

        # Framework-specific configurations
        self._configure_tensorflow()
        self._configure_pytorch()
        self._configure_onnxruntime()

        logger.info(f"GPU Resource Manager initialized on device: {self.device_info.device_type.value}")

    def _configure_tensorflow(self):
        """Configure TensorFlow GPU settings"""
        try:
            if self.device_info.device_type == DeviceType.CUDA:
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    # Enable memory growth for all GPUs
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"TensorFlow configured with memory growth for {len(gpus)} GPU(s)")
            elif self.device_info.device_type == DeviceType.ROCM:
                # ROCm support in TensorFlow requires ROCm-enabled build
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    # Enable memory growth for all GPUs
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"TensorFlow configured with memory growth for ROCm GPU(s)")
                else:
                    logger.info("No ROCm GPUs detected by TensorFlow, falling back to CPU")
            elif self.device_info.device_type == DeviceType.MPS:
                # TensorFlow MPS support is limited, fallback to CPU
                logger.info("TensorFlow MPS support limited, using CPU fallback")
        except Exception as e:
            logger.warning(f"Failed to configure TensorFlow GPU: {e}")

    def _configure_pytorch(self):
        """Configure PyTorch GPU settings"""
        try:
            if self.device_info.device_type == DeviceType.CUDA and torch.cuda.is_available():
                # Set default CUDA device
                if self.device_info.device_id is not None:
                    torch.cuda.set_device(self.device_info.device_id)
                logger.info(f"PyTorch configured on CUDA device {self.device_info.device_id}")
            elif self.device_info.device_type == DeviceType.ROCM:
                # ROCm uses CUDA interface in PyTorch
                if hasattr(torch.version, 'hip') and torch.version.hip:
                    if torch.cuda.is_available():  # This checks HIP for ROCm
                        if self.device_info.device_id is not None:
                            torch.cuda.set_device(self.device_info.device_id)
                        logger.info(f"PyTorch configured on ROCm device {self.device_info.device_id}")
                    else:
                        logger.warning("PyTorch ROCm/HIP not available, will use CPU")
                else:
                    logger.warning("PyTorch not built with ROCm support, will use CPU")
            elif self.device_info.device_type == DeviceType.MPS:
                if torch.backends.mps.is_available():
                    logger.info("PyTorch configured on MPS (Apple Silicon)")
                else:
                    logger.info("PyTorch MPS not available, will use CPU")
        except Exception as e:
            logger.warning(f"Failed to configure PyTorch: {e}")

    def _configure_onnxruntime(self):
        """Configure ONNX Runtime execution providers"""
        try:
            providers = []

            if self.device_info.device_type == DeviceType.CUDA:
                providers.append('CUDAExecutionProvider')
            elif self.device_info.device_type == DeviceType.ROCM:
                # ONNX Runtime ROCm support requires ROCm-enabled build
                # For now, use CPU provider as fallback
                logger.info("ONNX Runtime ROCm support not available, using CPU")
            elif self.device_info.device_type == DeviceType.MPS:
                providers.append('CoreMLExecutionProvider')

            providers.append('CPUExecutionProvider')
            ort.set_default_logger_severity(3)  # Warning level
            logger.info(f"ONNX Runtime configured with providers: {providers}")
        except Exception as e:
            logger.warning(f"Failed to configure ONNX Runtime: {e}")

    async def get_model_with_gpu(self,
                               model_type: ModelType,
                               model_class: Type[T],
                               **kwargs) -> T:
        """
        Get model instance with GPU allocation and automatic cleanup.

        Args:
            model_type: Type identifier for the model
            model_class: Model class to instantiate
            **kwargs: Arguments to pass to model constructor

        Returns:
            Model instance with GPU resources allocated
        """
        async with self.memory_lock:
            # Check if model already exists and is active
            if model_type in self.active_models:
                allocation = self.active_models[model_type]
                if allocation.is_active:
                    allocation.last_access_time = time.time()
                    logger.debug(f"Reusing existing model: {model_type.value}")
                    return allocation.model_instance
                else:
                    # Clean up inactive model
                    await self._cleanup_model_allocation(allocation)

            # Check available memory
            memory_info = await self.get_memory_info()
            if memory_info.utilization_percent > (self.memory_threshold * 100):
                logger.warning(f"GPU memory usage high ({memory_info.utilization_percent:.1f}%), "
                             f"cleaning up inactive models")
                await self.cleanup_inactive_models()

                # Recheck memory after cleanup
                memory_info = await self.get_memory_info()
                if memory_info.utilization_percent > (self.memory_threshold * 100):
                    logger.error(f"GPU memory still too high after cleanup. "
                               f"Consider running model on CPU.")

            # Allocate new model
            model_instance = await self._allocate_model(model_type, model_class, **kwargs)
            return model_instance

    async def _allocate_model(self,
                            model_type: ModelType,
                            model_class: Type[T],
                            **kwargs) -> T:
        """Allocate and initialize a model on GPU"""
        try:
            # Create model instance
            model_instance = model_class(**kwargs)

            # Move model to appropriate device if it's a PyTorch model
            if hasattr(model_instance, 'to') and callable(getattr(model_instance, 'to')):
                device = self._get_pytorch_device()
                model_instance = model_instance.to(device)
                logger.info(f"Moved {model_type.value} to device: {device}")

                # Special case for PSCCNet: update the internal device attribute
                if hasattr(model_instance, 'device') and hasattr(model_instance, 'FENet'):
                    try:
                        old_device = model_instance.device
                        model_instance.device = torch.device(device)
                        logger.info(f"Updated PSCCNet device from {old_device} to {device}")
                    except Exception as e:
                        logger.warning(f"Failed to update PSCCNet device attribute: {e}")

            # Get memory usage
            memory_usage = await self._estimate_model_memory(model_instance)

            # Create allocation record
            allocation = ModelAllocation(
                model_type=model_type,
                model_instance=model_instance,
                device_type=self.device_info.device_type,
                device_id=self.device_info.device_id,
                memory_usage_mb=memory_usage,
                last_access_time=time.time(),
                allocation_time=time.time()
            )

            self.active_models[model_type] = allocation

            # Start background cleanup task if not running
            if not self._background_cleanup_running:
                self._start_background_cleanup()

            logger.info(f"Successfully allocated model: {model_type.value} "
                       f"(~{memory_usage:.1f}MB memory)")

            return model_instance

        except Exception as e:
            logger.error(f"Failed to allocate model {model_type.value}: {e}")
            raise

    def _get_pytorch_device(self) -> Union[torch.device, str]:
        """Get appropriate PyTorch device"""
        if self.device_info.device_type == DeviceType.CUDA and torch.cuda.is_available():
            if self.device_info.device_id is not None:
                return torch.device(f"cuda:{self.device_info.device_id}")
            return torch.device("cuda")
        elif self.device_info.device_type == DeviceType.ROCM:
            # ROCm uses CUDA interface in PyTorch
            if hasattr(torch.version, 'hip') and torch.version.hip and torch.cuda.is_available():
                if self.device_info.device_id is not None:
                    return torch.device(f"cuda:{self.device_info.device_id}")
                return torch.device("cuda")
            return torch.device("cpu")
        elif self.device_info.device_type == DeviceType.MPS and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    async def _estimate_model_memory(self, model_instance: Any) -> float:
        """Estimate model memory usage in MB"""
        try:
            if hasattr(model_instance, 'parameters') and callable(getattr(model_instance, 'parameters')):
                # PyTorch model
                param_size = sum(p.numel() * p.element_size() for p in model_instance.parameters())
                buffer_size = sum(b.numel() * b.element_size() for b in model_instance.buffers())
                total_size = (param_size + buffer_size) / (1024 * 1024)  # Convert to MB
                return total_size
            else:
                # For other frameworks, return estimated size
                return 100.0  # Default estimate
        except Exception:
            return 100.0  # Default estimate if measurement fails

    async def release_model(self, model_type: ModelType) -> bool:
        """
        Release GPU memory for a specific model.

        Args:
            model_type: Type of model to release

        Returns:
            True if model was released, False if not found
        """
        async with self.memory_lock:
            if model_type in self.active_models:
                allocation = self.active_models[model_type]
                await self._cleanup_model_allocation(allocation)
                del self.active_models[model_type]
                logger.info(f"Released model: {model_type.value}")
                return True
            return False

    async def _cleanup_model_allocation(self, allocation: ModelAllocation):
        """Clean up model allocation and free memory"""
        try:
            model_instance = allocation.model_instance

            # PyTorch cleanup
            if hasattr(model_instance, 'cpu') and callable(getattr(model_instance, 'cpu')):
                model_instance.cpu()

            # Clear references
            allocation.is_active = False
            allocation.model_instance = None

            # Force garbage collection
            gc.collect()

            # PyTorch CUDA cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # PyTorch MPS cleanup (Apple Silicon)
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

            logger.debug(f"Cleaned up allocation: {allocation.model_type.value}")

        except Exception as e:
            logger.warning(f"Error during model cleanup: {e}")

    async def cleanup_inactive_models(self) -> int:
        """
        Clean up inactive models based on timeout.

        Returns:
            Number of models cleaned up
        """
        current_time = time.time()
        models_to_remove = []

        for model_type, allocation in self.active_models.items():
            if current_time - allocation.last_access_time > self.model_timeout:
                models_to_remove.append(model_type)

        async with self.memory_lock:
            for model_type in models_to_remove:
                allocation = self.active_models[model_type]
                await self._cleanup_model_allocation(allocation)
                del self.active_models[model_type]
                logger.info(f"Cleaned up inactive model: {model_type.value}")

        return len(models_to_remove)

    def _start_background_cleanup(self):
        """Start background cleanup task"""
        if self.cleanup_task is None or self.cleanup_task.done():
            self.cleanup_task = asyncio.create_task(self._background_cleanup_loop())
            self._background_cleanup_running = True

    async def _background_cleanup_loop(self):
        """Background loop for periodic cleanup"""
        while self.active_models:
            try:
                await asyncio.sleep(60)  # Check every minute
                cleaned_count = await self.cleanup_inactive_models()
                if cleaned_count > 0:
                    logger.info(f"Background cleanup removed {cleaned_count} inactive models")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in background cleanup loop: {e}")

        self._background_cleanup_running = False

    async def get_memory_info(self) -> GPUMemoryInfo:
        """Get current GPU memory information"""
        try:
            if (self.device_info.device_type == DeviceType.CUDA or
                self.device_info.device_type == DeviceType.ROCM) and torch.cuda.is_available():
                device_id = self.device_info.device_id or 0
                total_memory = torch.cuda.get_device_properties(device_id).total_memory
                allocated_memory = torch.cuda.memory_allocated(device_id)
                free_memory = total_memory - allocated_memory
                utilization_percent = (allocated_memory / total_memory) * 100

                return GPUMemoryInfo(
                    total_memory_mb=total_memory / (1024 * 1024),
                    allocated_memory_mb=allocated_memory / (1024 * 1024),
                    free_memory_mb=free_memory / (1024 * 1024),
                    utilization_percent=utilization_percent
                )
            else:
                # Non-GPU device or CPU fallback
                return GPUMemoryInfo(
                    total_memory_mb=0.0,
                    allocated_memory_mb=0.0,
                    free_memory_mb=0.0,
                    utilization_percent=0.0
                )
        except Exception as e:
            logger.warning(f"Failed to get GPU memory info: {e}")
            return GPUMemoryInfo(0.0, 0.0, 0.0, 0.0)

    async def get_memory_usage(self) -> Dict[str, Any]:
        """Get detailed memory usage statistics"""
        memory_info = await self.get_memory_info()

        active_models = []
        for model_type, allocation in self.active_models.items():
            if allocation.is_active:
                active_models.append({
                    "model_type": model_type.value,
                    "device_type": allocation.device_type.value,
                    "memory_usage_mb": allocation.memory_usage_mb,
                    "last_access_time": allocation.last_access_time,
                    "allocation_time": allocation.allocation_time
                })

        return {
            "gpu_available": self.device_info.is_available,
            "device_type": self.device_info.device_type.value,
            "device_name": self.device_info.device_name,
            "memory_info": {
                "total_memory_mb": memory_info.total_memory_mb,
                "allocated_memory_mb": memory_info.allocated_memory_mb,
                "free_memory_mb": memory_info.free_memory_mb,
                "utilization_percent": memory_info.utilization_percent
            },
            "active_models": active_models,
            "total_active_models": len(active_models),
            "memory_threshold": self.memory_threshold,
            "model_timeout": self.model_timeout
        }

    async def shutdown(self):
        """Shutdown the GPU resource manager and cleanup all models"""
        logger.info("Shutting down GPU Resource Manager...")

        # Cancel background cleanup task
        if self.cleanup_task and not self.cleanup_task.done():
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

        # Clean up all active models
        async with self.memory_lock:
            for allocation in list(self.active_models.values()):
                await self._cleanup_model_allocation(allocation)
            self.active_models.clear()

        logger.info("GPU Resource Manager shutdown complete")

    @asynccontextmanager
    async def temporary_model(self, model_type: ModelType, model_class: Type[T], **kwargs):
        """
        Context manager for temporary model usage with automatic cleanup.

        Args:
            model_type: Type identifier for the model
            model_class: Model class to instantiate
            **kwargs: Arguments to pass to model constructor
        """
        model = None
        try:
            model = await self.get_model_with_gpu(model_type, model_class, **kwargs)
            yield model
        finally:
            if model is not None:
                await self.release_model(model_type)


# Global instance for application-wide use
_gpu_manager: Optional[GPUResourceManager] = None
_manager_lock = threading.Lock()


def get_gpu_manager() -> GPUResourceManager:
    """Get the global GPU resource manager instance"""
    global _gpu_manager
    if _gpu_manager is None:
        with _manager_lock:
            if _gpu_manager is None:
                _gpu_manager = GPUResourceManager()
    return _gpu_manager


async def shutdown_gpu_manager():
    """Shutdown the global GPU resource manager"""
    global _gpu_manager
    if _gpu_manager is not None:
        await _gpu_manager.shutdown()
        _gpu_manager = None
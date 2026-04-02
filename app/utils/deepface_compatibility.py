"""
DeepFace TensorFlow Compatibility Utilities

This module provides utilities to ensure DeepFace compatibility with TensorFlow
by resetting problematic configurations that cause KerasTensor errors.
"""

import os
from typing import Dict, Any
from app.core.logger import get_logger


def reset_tensorflow_for_deepface() -> bool:
    """
    Reset TensorFlow configuration to ensure DeepFace compatibility.

    Returns:
        True if reset was successful
    """
    logger = get_logger()

    try:
        import tensorflow as tf

        logger.info("Resetting TensorFlow configuration for DeepFace compatibility...")

        # Clear all TensorFlow sessions and graphs
        tf.keras.backend.clear_session()

        # Reset all GPU/MPS device configurations
        try:
            # Reset GPU configuration
            tf.config.experimental.reset_all_memory_stats()
        except:
            pass  # May not be available or needed

        # Disable all aggressive optimizations that cause KerasTensor issues
        optimization_configs = [
            ('set_jit', False),
            ('set_experimental_options', {}),
        ]

        for config_name, config_value in optimization_configs:
            try:
                if hasattr(tf.config.optimizer, config_name):
                    getattr(tf.config.optimizer, config_name)(config_value)
                    logger.debug(f"Reset {config_name} to {config_value}")
            except Exception as e:
                logger.debug(f"Could not reset {config_name}: {e}")

        # Reset device placement
        tf.config.set_soft_device_placement(True)
        try:
            tf.config.set_log_device_placement(False)
        except AttributeError:
            # This function may not be available in all TensorFlow versions
            pass

        # Disable mixed precision completely
        try:
            if hasattr(tf.keras.mixed_precision, 'set_global_policy'):
                tf.keras.mixed_precision.set_global_policy('float32')
                logger.info("Disabled mixed precision policy")
        except Exception as e:
            logger.debug(f"Could not disable mixed precision: {e}")

        # Set environment variables for DeepFace compatibility
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
        os.environ['TF_DETERMINISTIC_OPS'] = '1'  # Enable deterministic operations
        os.environ['PYTHONHASHSEED'] = '0'  # Ensure reproducibility

        logger.info("TensorFlow configuration reset for DeepFace compatibility")
        return True

    except ImportError:
        logger.warning("TensorFlow not available for reset")
        return False
    except Exception as e:
        logger.error(f"Failed to reset TensorFlow configuration: {e}")
        return False


def configure_deepface_environment() -> Dict[str, Any]:
    """
    Configure environment variables specifically for DeepFace stability.

    Returns:
        Dict with configuration status
    """
    logger = get_logger()
    config_status = {}

    try:
        # Disable TensorFlow optimizations that interfere with DeepFace
        tf_configs = {
            'TF_ENABLE_ONEDNN_OPTS': '0',      # Disable oneDNN
            'TF_CPP_MIN_LOG_LEVEL': '1',       # Reduce TensorFlow logging
            'TF_DETERMINISTIC_OPS': '1',       # Deterministic operations
            'TF_GPU_ALLOCATOR': 'cuda_malloc',  # Simple allocator
            'HOROVOD_GPU_ALLREDUCE': 'NCCL',   # NCCL for communication
            'CUDA_LAUNCH_BLOCKING': '0',       # Async CUDA operations
            'TF_FORCE_GPU_ALLOW_GROWTH': 'true', # Memory growth
        }

        # Apply configurations
        for key, value in tf_configs.items():
            os.environ[key] = value
            config_status[key] = value

        logger.info("DeepFace environment configuration applied")
        config_status['status'] = 'success'

    except Exception as e:
        logger.error(f"Failed to configure DeepFace environment: {e}")
        config_status['status'] = 'failed'
        config_status['error'] = str(e)

    return config_status


def ensure_deepface_compatibility() -> bool:
    """
    Ensure DeepFace compatibility with minimal interference (prevents KerasTensor errors).

    Returns:
        True if compatibility was ensured successfully
    """
    logger = get_logger()

    try:
        logger.info("Setting minimal DeepFace compatibility...")

        # MINIMAL COMPATIBILITY: Only set environment variables, NO TensorFlow resets
        # TensorFlow resets are causing KerasTensor conflicts with DeepFace
        env_configs = {
            'TF_CPP_MIN_LOG_LEVEL': '1',       # Reduce TensorFlow logging
            'TF_DETERMINISTIC_OPS': '0',       # Allow non-deterministic (better performance)
            'TF_ENABLE_ONEDNN_OPTS': '1',      # Enable oneDNN optimizations
        }

        for key, value in env_configs.items():
            os.environ[key] = value

        logger.info("Minimal DeepFace compatibility applied (no TensorFlow resets)")
        return True

    except Exception as e:
        logger.error(f"Failed to set minimal DeepFace compatibility: {e}")
        return False
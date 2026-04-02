"""
RetinaFace TensorFlow Compatibility

This module provides specific TensorFlow configurations to ensure RetinaFace compatibility.
RetinaFace is sensitive to certain TensorFlow optimizations and requires specific settings.
"""

import os
import tensorflow as tf
from app.core.logger import get_logger


def configure_tensorflow_for_retinaface() -> bool:
    """
    Configure TensorFlow specifically for RetinaFace compatibility.

    RetinaFace requires specific TensorFlow settings to avoid KerasTensor conflicts.

    Returns:
        True if configuration was successful
    """
    logger = get_logger()

    try:
        logger.info("Configuring TensorFlow for RetinaFace compatibility...")

        # CRITICAL: RetinaFace-specific TensorFlow configurations

        # 1. Disable XLA compilation (causes KerasTensor conflicts with RetinaFace)
        tf.config.optimizer.set_jit(False)
        logger.debug("Disabled XLA compilation for RetinaFace")

        # 2. Disable mixed precision (interferes with RetinaFace operations)
        try:
            if hasattr(tf.keras.mixed_precision, 'set_global_policy'):
                tf.keras.mixed_precision.set_global_policy('float32')
                logger.debug("Disabled mixed precision for RetinaFace")
        except Exception as e:
            logger.debug(f"Could not disable mixed precision: {e}")

        # 3. Set environment variables specifically for RetinaFace
        retinaface_env_vars = {
            'TF_ENABLE_ONEDNN_OPTS': '0',      # Disable oneDNN (conflicts with RetinaFace)
            'TF_XLA_FLAGS': '--tf_xla_enable_xla_devices=false',  # Disable XLA completely
            'TF_DETERMINISTIC_OPS': '0',       # Allow non-deterministic for RetinaFace performance
            'TF_CPP_MIN_LOG_LEVEL': '2',       # Reduce logging noise
        }

        for key, value in retinaface_env_vars.items():
            os.environ[key] = value
            logger.debug(f"Set {key}={value} for RetinaFace")

        # 4. Configure device settings for RetinaFace
        tf.config.set_soft_device_placement(True)

        # 5. Disable aggressive graph optimizations that break RetinaFace
        try:
            if hasattr(tf.config.experimental, 'disable_mixed_precision_graph_rewrite'):
                tf.config.experimental.disable_mixed_precision_graph_rewrite()
                logger.debug("Disabled mixed precision graph rewrite for RetinaFace")
        except Exception as e:
            logger.debug(f"Could not disable graph rewrite: {e}")

        # 6. Ensure eager execution is enabled (RetinaFace requires this)
        if not tf.executing_eagerly():
            logger.warning("TensorFlow not in eager mode - RetinaFace may have issues")

        logger.info("TensorFlow configured successfully for RetinaFace compatibility")
        return True

    except Exception as e:
        logger.error(f"Failed to configure TensorFlow for RetinaFace: {e}")
        return False


def reset_tensorflow_for_retinaface() -> bool:
    """
    Reset TensorFlow state and configure specifically for RetinaFace.

    This should be called before any RetinaFace operations to ensure compatibility.

    Returns:
        True if reset and configuration were successful
    """
    logger = get_logger()

    try:
        logger.info("Resetting TensorFlow for RetinaFace operations...")

        # Clear TensorFlow state
        tf.keras.backend.clear_session()

        # Configure for RetinaFace
        success = configure_tensorflow_for_retinaface()

        if success:
            logger.info("TensorFlow reset and configured for RetinaFace successfully")
        else:
            logger.warning("TensorFlow reset for RetinaFace had issues")

        return success

    except Exception as e:
        logger.error(f"Failed to reset TensorFlow for RetinaFace: {e}")
        return False


def ensure_retinaface_compatibility() -> bool:
    """
    Ensure RetinaFace compatibility with complete TensorFlow reset.

    RetinaFace is extremely sensitive to TensorFlow state and requires a clean environment.

    Returns:
        True if RetinaFace compatibility is ensured
    """
    logger = get_logger()

    try:
        logger.info("Ensuring RetinaFace compatibility with complete reset...")

        # CRITICAL: Complete TensorFlow reset required for RetinaFace
        success = reset_tensorflow_for_retinaface()

        if success:
            logger.info("RetinaFace compatibility ensured with complete reset")
        else:
            logger.warning("RetinaFace complete reset had issues")

        return success

    except Exception as e:
        logger.error(f"Failed to ensure RetinaFace compatibility: {e}")
        return False
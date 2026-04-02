import os
import logging
import sys

# ============================================================================
# PATCH: Fix segfault with PyTorch 2.6 + DeBERTa V2 JIT compilation
# This MUST be done before any torch import
# ============================================================================

import builtins

_original_import = builtins.__import__

# Use a dict for thread-safe mutable flag (avoids closure scoping issues)
_patch_state = {'torch_patched': False}

def _patched_import(name, *args, **kwargs):
    """Intercept torch import and patch JIT before it's used."""
    module = _original_import(name, *args, **kwargs)

    # Patch torch.jit immediately when torch is first imported
    # torch.jit is available immediately after torch module is loaded
    if name == 'torch' and not _patch_state['torch_patched']:
        _patch_state['torch_patched'] = True

        # Replace torch.jit.script with a no-op
        def _noop_script(fn, *script_args, **script_kwargs):
            return fn

        if hasattr(module, 'jit'):
            module.jit.script = _noop_script
            module.jit.trace = _noop_script
            print("torch.jit.script/trace patched to prevent segfault with PyTorch 2.6")

    return module

builtins.__import__ = _patched_import

# Detect MPS and set fallback automatically
try:
    import torch
    if torch.backends.mps.is_available():
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        print("MPS detected - enabling automatic CPU fallback for unsupported operations")
    else:
        print("MPS not available - fallback not needed")
except ImportError:
    print("PyTorch not available - skipping MPS detection")
except Exception as e:
    print(f"Error detecting MPS: {e}")
    # Set fallback anyway to be safe
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from app import app
from app.api import api_router
from app.core.device_manager import device_manager
from app.core.framework_coordinator import get_framework_coordinator


# Configure Python standard logging to ensure extraction logs are visible
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Set specific loggers to INFO level to match custom logger behavior
logging.getLogger('app.core.key_injection').setLevel(logging.INFO)
logging.getLogger('app.core.key_injection.key_injection_manager').setLevel(logging.INFO)
logging.getLogger('app.core.key_injection.business_rules_analyzer').setLevel(logging.INFO)
logging.getLogger('app.core.key_injection.key_field_detector').setLevel(logging.INFO)
logging.getLogger('app.core.key_injection.spatial_analyzer').setLevel(logging.INFO)
logging.getLogger('app.core.key_injection.simple_bank_analyzer').setLevel(logging.DEBUG)  # Enable DEBUG for clustering
logging.getLogger('app.helper.doctr').setLevel(logging.INFO)
logging.getLogger('app.services').setLevel(logging.INFO)

# Configure root logger for all other modules
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

# Ensure all handlers are set to INFO level to show extraction logs
for handler in root_logger.handlers:
    handler.setLevel(logging.INFO)


# Include API router
app.include_router(api_router)

# Add secret share endpoints
from app.api.secret_share_endpoint import router as secret_share_router
app.include_router(secret_share_router, prefix="/api")


def validate_photoholmes_weights(logger):
    """Validate PhotoHolmes weights are present during startup"""
    try:
        from app.utils.weight_startup_check import verify_photoholmes_weights_on_startup

        logger.info("🔍 Verifying PhotoHolmes weights...")
        weights_ready = verify_photoholmes_weights_on_startup(auto_download=True)

        if weights_ready:
            logger.info("✅ PhotoHolmes weights verification passed")
        else:
            logger.warning("⚠️ PhotoHolmes weights verification failed - some features may be limited")

    except Exception as e:
        logger.error(f"❌ PhotoHolmes weights verification error: {str(e)}")
        logger.warning("⚠️ Proceeding without weight verification - PhotoHolmes may not function properly")


def validate_startup_environment():
    """Validate GPU/CPU environment during application startup"""
    logger = logging.getLogger(__name__)

    logger.info("🚀 Validating application startup environment...")

    # First, validate PhotoHolmes weights
    validate_photoholmes_weights(logger)

    try:
        # Get device summary
        device_summary = device_manager.get_device_summary()

        logger.info(f"📊 Platform: {device_summary['platform']}")
        logger.info(f"🐍 Python: {device_summary['python_version']}")

        # Log available devices
        for device_type, devices in device_summary['devices'].items():
            if devices:
                logger.info(f"🔧 {device_type.upper()} devices available: {len(devices)}")
                for device in devices:
                    status = "✅" if device['is_available'] else "❌"
                    memory_info = f" ({device['memory_total']}MB)" if device['memory_total'] else ""
                    logger.info(f"  {status} {device['name']}{memory_info}")

        # Log recommended device
        recommended = device_summary['recommended_device']
        logger.info(f"🎯 Recommended device: {recommended['type'].upper()} - {recommended['name']}")

        # Validate the recommended device
        device_type = recommended['type']
        device_info = device_manager.get_recommended_device()

        from app.core.device_manager import DeviceType
        validation_result = device_manager.validate_gpu_device(
            device_type= DeviceType(device_type),
            device_id=recommended['id']
        )

        if validation_result.is_valid:
            logger.info("✅ Startup environment validation PASSED")
            if validation_result.warnings:
                for warning in validation_result.warnings:
                    logger.warning(f"⚠️  {warning}")
        else:
            logger.error("❌ Startup environment validation FAILED")
            for issue in validation_result.issues:
                logger.error(f"  🚫 {issue}")
            logger.info("🔄 Application will continue with fallback configuration")

        # Log recommendations
        if validation_result.recommendations:
            logger.info("💡 Recommendations:")
            for rec in validation_result.recommendations:
                logger.info(f"  • {rec}")

        # Initialize and coordinate all frameworks if device validation passed
        if validation_result.is_valid:
            logger.info("🤖 Initializing unified framework coordinator...")
            try:
                framework_coordinator = get_framework_coordinator()
                coordination_results = framework_coordinator.coordinate_all_frameworks()

                for framework, success in coordination_results.items():
                    status = "✅" if success else "❌"
                    logger.info(f"{status} {framework.upper()}: {'Configured' if success else 'Failed'}")

                # Log framework configuration summary
                memory_report = framework_coordinator.get_memory_usage_report()
                logger.info(f"📊 Framework coordination summary:")
                logger.info(f"  - Total allocated memory: {memory_report['total_allocated_memory_mb']}MB")
                for framework_name, config_info in memory_report['framework_configs'].items():
                    logger.info(f"  - {framework_name.upper()}: {config_info['device_type']} "
                               f"({config_info['memory_limit_mb']}MB limit)")

            except Exception as e:
                logger.error(f"❌ Framework coordination failed: {e}")
                logger.warning("⚠️ Application will proceed with individual framework configurations")

        return validation_result.is_valid

    except Exception as e:
        logger.error(f"❌ Environment validation failed: {e}")
        return False


def main():
    import uvicorn

    # Validate startup environment first
    validation_passed = validate_startup_environment()

    if validation_passed:
        logging.getLogger(__name__).info("🎉 Starting application with validated environment")
    else:
        logging.getLogger(__name__).warning("⚠️  Starting application with validation warnings")

    port = int(os.getenv("PORT", 12410))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)

if __name__ == "__main__":
    main()

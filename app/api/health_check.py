from fastapi import APIRouter, status, HTTPException
from datetime import datetime
from typing import Dict, Any
from .endpoint import Endpoint

from app.core.gpu_manager import get_gpu_manager
from app.core.logger import get_logger

logger = get_logger()

router = APIRouter(tags=["Health"])


@router.get(
    Endpoint.HEALTH,
    status_code=status.HTTP_200_OK,
    summary="Health check endpoint",
    response_description="Service health status",
)
async def health_check() -> dict:
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "IM-OSINT-API",
        "version": "0.1.0",
        "environment": "development",
        "api_prefix": Endpoint.API_PREFIX,
        "api_list": [
            {
                "endpoint": Endpoint.HEALTH,
                "method": "GET",
                "description": "Health check endpoint",
            },
              {
                "endpoint": Endpoint.ANALYZE_RESUME,
                "method": "POST",
                "description": "Analyze resume",
            },
            {
                "endpoint": Endpoint.DETECT_FORGERY+"?text_require=true",
                "method": "POST",
                "description": "Detect forgery with extract data from image",
            },
            {
                "endpoint": Endpoint.GET_KEY,
                "method": "GET",
                "description": "Get server public key",
            },
            {
                "endpoint": Endpoint.CREATE_KEY,
                "method": "POST",
                "description": "Create Share Key"
            },
            {
                "endpoint": Endpoint.KEY_RECOVERY,
                "method": "POST",
                "description": "Key recovery"
            },
            {
                "endpoint": Endpoint.DELETE_KEY_RECOVERY,
                "method": "DELETE",
                "description": "Delete key recovery"
            },
            {
                "endpoint": Endpoint.OTP_NUMBER+"?email=example.gmail.com",
                "method": "GET",
                "description": "Generate and send OTP with specified length "
            },
            {
                "endpoint": Endpoint.GET_ALL_VERIFICATION+"?email=example.gmail.com",
                "method": "GET",
                "description": "Get all verification"
            },
            {
                "endpoint": Endpoint.UPDATE_MANUAL_CHECK,
                "method": "PATCH",
                "description": "Update manual check"
            },
            {
                "endpoint": Endpoint.VERIFY + "?fake=true",
                "method": "POST",
                "description": "Verification KYC Or Onboarding"
            },
            {
                "endpoint": Endpoint.EXTRACT_DATA,
                "method": "POST",
                "description": "Extract data from PDF"
            },
            {
                "endpoint": "/optimized/analyze",
                "method": "POST",
                "description": "Analyze document using optimized 5-step workflow",
            },
            {
                "endpoint": "/optimized/analyze-batch",
                "method": "POST",
                "description": "Analyze multiple documents in batch",
            },
            {
                "endpoint": "/optimized/status",
                "method": "GET",
                "description": "Get optimized analysis service status",
            },
            {
                "endpoint": "/health/optimized",
                "method": "GET",
                "description": "Optimized workflow health check and device capabilities",
            },
            {
                "endpoint": "/health/gpu",
                "method": "GET",
                "description": "GPU health and memory usage check",
            }
        ],
    }


@router.get(
    "/health/gpu",
    status_code=status.HTTP_200_OK,
    summary="GPU health check endpoint",
    response_description="GPU resource status and memory usage",
    responses={
        200: {"description": "GPU status retrieved successfully"},
        503: {"description": "GPU resources unavailable or experiencing issues"}
    }
)
async def gpu_health_check() -> Dict[str, Any]:
    """
    Comprehensive GPU health and memory usage check for all CV frameworks.

    Returns detailed information about:
    - GPU availability and device information
    - Current memory usage and allocation
    - Active models and their resource usage
    - Framework-specific GPU status

    Useful for monitoring and debugging GPU resource allocation.
    """
    try:
        gpu_manager = get_gpu_manager()
        memory_usage = await gpu_manager.get_memory_usage()

        # Determine overall health status
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "gpu_available": memory_usage["gpu_available"],
            "device_info": {
                "type": memory_usage["device_type"],
                "name": memory_usage["device_name"],
            },
            "memory_info": memory_usage["memory_info"],
            "active_models": memory_usage["active_models"],
            "total_active_models": memory_usage["total_active_models"],
            "memory_threshold": memory_usage["memory_threshold"],
            "model_timeout": memory_usage["model_timeout"],
            "warnings": [],
            "recommendations": []
        }

        # Add health warnings based on memory usage
        utilization = memory_usage["memory_info"]["utilization_percent"]
        if utilization > 90:
            health_status["warnings"].append(f"GPU memory usage critical: {utilization:.1f}%")
            health_status["recommendations"].append("Consider releasing unused models or increasing GPU memory")
            health_status["status"] = "critical"
        elif utilization > 80:
            health_status["warnings"].append(f"GPU memory usage high: {utilization:.1f}%")
            health_status["recommendations"].append("Monitor memory usage closely")
            health_status["status"] = "warning"
        elif utilization > 60:
            health_status["warnings"].append(f"GPU memory usage moderate: {utilization:.1f}%")

        # Add warnings for inactive models that could be cleaned up
        current_time = datetime.now().timestamp()
        for model in memory_usage["active_models"]:
            time_since_access = current_time - model["last_access_time"]
            if time_since_access > 600:  # 10 minutes
                health_status["warnings"].append(
                    f"Model {model['model_type']} inactive for {time_since_access/60:.1f} minutes"
                )
                health_status["recommendations"].append(
                    f"Consider releasing {model['model_type']} to free memory"
                )

        # Log health check results
        logger.info(f"GPU health check completed: {health_status['status']}, "
                   f"utilization: {utilization:.1f}%, models: {memory_usage['total_active_models']}")

        # Return appropriate HTTP status based on health
        if health_status["status"] == "critical":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=health_status
            )
        elif health_status["status"] == "warning":
            # Still return 200 but include warnings
            return health_status

        return health_status

    except Exception as e:
        logger.error(f"GPU health check failed: {str(e)}")

        # Return error status but still provide basic information
        error_status = {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "gpu_available": False,
            "error": str(e),
            "warnings": ["GPU health check failed - see logs for details"],
            "recommendations": ["Check GPU driver installation and CUDA availability"]
        }

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=error_status
        )


@router.get(
    "/health/verification-state-worker",
    status_code=status.HTTP_200_OK,
    summary="Verification state update worker health check",
    response_description="Verification state update worker status",
    responses={
        200: {"description": "Worker status retrieved successfully"},
        503: {"description": "Worker service unavailable"}
    }
)
async def verification_state_worker_health(request) -> Dict[str, Any]:
    """
    Health check endpoint for the verification state update worker.

    Returns the current status of the background worker that updates
    verification states based on document expiry dates.
    """
    from fastapi import Request

    if not isinstance(request, Request):
        # FastAPI dependency injection wrapper
        request = request

    worker = request.app.state.verification_state_update_worker if hasattr(request.app.state, 'verification_state_update_worker') else None

    if not worker:
        return {
            "status": "disabled",
            "timestamp": datetime.now().isoformat(),
            "message": "Verification state update worker is not enabled or not available"
        }

    return {
        "status": "healthy" if worker.is_healthy() else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        **worker.get_stats()
    }

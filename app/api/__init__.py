from fastapi import APIRouter
from .health_check import router as health_router
from .key_endpoint import router as key_router
from .otp_endpoint import router as otp_router
from .job_endpoints import router as job_router
from .deletion_endpoint import router as deletion_router
from .endpoint import Endpoint
from .internal_sync_endpoints import router as internal_sync_router


# Create the main API router
api_router = APIRouter()

# Include all routers with their respective prefixes
api_router.include_router(health_router, prefix="", tags=["Health"])
api_router.include_router(job_router, prefix="", tags=["Jobs"])
api_router.include_router(key_router, prefix="", tags=["KEY"])
api_router.include_router(otp_router, prefix="", tags=["OTP"])
api_router.include_router(deletion_router, prefix="/api", tags=["User Management"])
api_router.include_router(internal_sync_router, tags=["Internal Sync"])

__all__ = ["api_router", "Endpoint"]

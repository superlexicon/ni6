from app.core import get_db_connection
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from contextlib import asynccontextmanager
import asyncio
import logging
import json
import base64
import os

logger = logging.getLogger(__name__)


def get_mobile_number_key(request: Request) -> str:
    """
    Extract mobile number from request body for rate limiting.
    Falls back to IP address if mobile number is not available.
    """
    # Try to get mobile_number from request state (set by endpoint)
    mobile_number = getattr(request.state, 'mobile_number', None)
    if mobile_number:
        return f"mobile:{mobile_number}"
    # Fallback to IP-based rate limiting
    return f"ip:{get_remote_address(request)}"


# Initialize rate limiter with in-memory storage
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri="memory://",
    default_limits=["100/hour"],
    headers_enabled=True  # Include rate limit headers in response
)


class SafeJSONResponse(JSONResponse):
    """Custom JSONResponse that handles binary data safely"""

    def render(self, content) -> bytes:
        # Custom JSON encoder that handles bytes by converting them to base64
        def safe_default(obj):
            if isinstance(obj, bytes):
                return base64.b64encode(obj).decode('ascii')
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
            default=safe_default
        ).encode("utf-8")


def safe_jsonable_encoder(obj):
    """Custom jsonable encoder that handles binary data safely"""
    def safe_serialize(value):
        if isinstance(value, dict):
            return {k: safe_serialize(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [safe_serialize(item) for item in value]
        elif isinstance(value, bytes):
            return base64.b64encode(value).decode('ascii')
        else:
            return value

    try:
        return safe_serialize(obj)
    except Exception:
        return str(obj)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - handles startup and shutdown"""
    # Startup
    logger.info("Starting IM-OSINT API application...")

    # Preload ML models to avoid first-request delays
    logger.info("Preloading ML models for faster first request...")

    # 1. DeepFace warm-up (already has method)
    try:
        from app.helper.deepface_helper import DeepfaceHelper
        await DeepfaceHelper.preload_all_models()
        logger.info("DeepFace models preloaded")
    except Exception as e:
        logger.warning(f"DeepFace preload failed: {e}")

    # 2. DocTR warm-up (initialize model)
    try:
        from app.core.doctr_model import DoctrModel
        import numpy as np
        doctr = DoctrModel()
        model = doctr.get_model()
        # Warm-up with dummy image to trigger JIT compilation
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        await asyncio.to_thread(model, [dummy_img])
        logger.info("DocTR models preloaded")
    except Exception as e:
        logger.warning(f"DocTR preload failed: {e}")

    # 3. GLiNER warm-up (initialize model for bank statement NER)
    try:
        from app.core.gliner_ner_model import get_gliner_ner_model
        gliner = get_gliner_ner_model()
        _ = gliner.get_model()
        logger.info("GLiNER NER model preloaded")
    except Exception as e:
        logger.warning(f"GLiNER preload failed: {e}")

    # 4. PhotoHolmes warm-up (trigger singleton initialization via __new__)
    # Only run on GPU servers - too slow on CPU and may cause startup issues
    try:
        import torch
        if torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            from app.services.comprehensive_photoholmes_service import ComprehensivePhotoHolmesService
            _ = ComprehensivePhotoHolmesService()
            logger.info("PhotoHolmes models preloaded")
        else:
            logger.info("PhotoHolmes warm-up skipped on CPU-only server")
    except Exception as e:
        logger.warning(f"PhotoHolmes preload failed: {e}")

    logger.info("ML model preloading complete")

    # Initialize NLP service (after ML models, before job system)
    app.state.nlp_service = None
    try:
        from app.services.nlp.nlp_service import nlp_service
        from app.config.osint_config import osint_settings

        if osint_settings.enable_nlp_enhanced_analysis:
            await nlp_service.initialize()
            app.state.nlp_service = nlp_service
            logger.info("NLP service initialized (VADER + spaCy)")
            logger.info("FinBERT will be loaded on first use")
        else:
            logger.info("NLP-enhanced analysis disabled via configuration")
    except Exception as e:
        logger.warning(f"NLP service not available: {e}")
        app.state.nlp_service = None

    # Initialize Sanctions and PEP checkers
    app.state.sanctions_checker = None
    app.state.pep_checker = None
    try:
        from app.services.osint.sanctions import sanctions_list_checker as sanctions_module
        from app.services.osint.pep import pep_checker as pep_module
        from app.repositories import crime_repository, pep_repository

        # Initialize sanctions checker
        if sanctions_module.sanctions_checker is None and crime_repository is not None:
            from app.services.osint.sanctions.sanctions_list_checker import SanctionsListChecker
            sanctions_module.sanctions_checker = SanctionsListChecker(
                sanctions_repository=crime_repository
            )
            app.state.sanctions_checker = sanctions_module.sanctions_checker
            logger.info("Sanctions checker initialized")

        # Initialize PEP checker
        if pep_module.pep_checker is None and pep_repository is not None:
            from app.services.osint.pep.pep_checker import PEPChecker
            pep_module.pep_checker = PEPChecker(
                pep_repository=pep_repository
            )
            app.state.pep_checker = pep_module.pep_checker
            logger.info("PEP checker initialized")

    except Exception as e:
        logger.warning(f"Sanctions/PEP checkers not available: {e}")
        app.state.sanctions_checker = None
        app.state.pep_checker = None

    # Initialize job system
    try:
        from app.api.job_endpoints import initialize_job_system, shutdown_job_system
        job_manager, worker, local_job_queue = initialize_job_system()
        app.state.job_manager = job_manager
        app.state.worker = worker
        app.state.local_job_queue = local_job_queue
        logger.info("Job system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize job system: {e}")
        raise

    # Initialize OTP broadcast service for HTTP-based inter-instance communication
    app.state.otp_broadcast_service = None
    try:
        from app.services.otp_broadcast_service import otp_broadcast_service
        app.state.otp_broadcast_service = otp_broadcast_service
        logger.info("OTP broadcast service initialized")
    except Exception as e:
        logger.warning(f"OTP broadcast service not available: {e}")
        app.state.otp_broadcast_service = None

    # Initialize startup sync service for HTTP-based inter-instance communication
    app.state.startup_sync_service = None
    try:
        from app.services.startup_sync_service import StartupSyncService
        from app.config.instance_config import instance_config

        if app.state.local_job_queue and instance_config.startup_sync_enabled:
            sync_service = StartupSyncService(
                local_job_queue=app.state.local_job_queue,
                instance_public_key=instance_config.instance_public_key
            )
            sync_service.start_sync()
            app.state.startup_sync_service = sync_service
            logger.info("Startup sync service started")
    except Exception as e:
        logger.warning(f"Startup sync service not available: {e}")
        app.state.startup_sync_service = None

    # Initialize cleanup worker for abandoned identity flows
    app.state.cleanup_worker = None
    try:
        from app.services.cleanup_worker import CleanupWorker

        cleanup_enabled = os.getenv('CLEANUP_ENABLED', 'true').lower() == 'true'
        cleanup_interval = int(os.getenv('CLEANUP_INTERVAL_HOURS', '1'))

        if cleanup_enabled:
            cleanup_worker = CleanupWorker(cleanup_interval_hours=cleanup_interval)
            cleanup_worker.start()
            app.state.cleanup_worker = cleanup_worker
            logger.info("Cleanup worker started")
        else:
            logger.info("Cleanup worker disabled via configuration")
    except Exception as e:
        logger.warning(f"Cleanup worker not available: {e}")
        app.state.cleanup_worker = None

    # Initialize verification state update worker for document expiry checks
    app.state.verification_state_update_worker = None
    try:
        from app.services.verification_state_update_worker import VerificationStateUpdateWorker

        update_enabled = os.getenv('VERIFICATION_STATE_UPDATE_ENABLED', 'true').lower() == 'true'
        update_interval = int(os.getenv('VERIFICATION_STATE_UPDATE_INTERVAL_HOURS', '24'))

        if update_enabled:
            verification_state_worker = VerificationStateUpdateWorker(interval_hours=update_interval)
            verification_state_worker.start()
            app.state.verification_state_update_worker = verification_state_worker
            logger.info("Verification state update worker started")
        else:
            logger.info("Verification state update worker disabled via configuration")
    except Exception as e:
        logger.warning(f"Verification state update worker not available: {e}")
        app.state.verification_state_update_worker = None

    yield

    # Shutdown
    logger.info("Shutting down IM-OSINT API application...")

    # Shutdown cleanup worker
    try:
        if hasattr(app.state, 'cleanup_worker') and app.state.cleanup_worker:
            app.state.cleanup_worker.stop()
            logger.info("Cleanup worker stopped")
    except Exception as e:
        logger.error(f"Error stopping cleanup worker: {e}")

    # Shutdown cleanup worker
    try:
        if hasattr(app.state, 'cleanup_worker') and app.state.cleanup_worker:
            app.state.cleanup_worker.stop()
            logger.info("Cleanup worker stopped")
    except Exception as e:
        logger.error(f"Error stopping cleanup worker: {e}")

    # Shutdown verification state update worker
    try:
        if hasattr(app.state, 'verification_state_update_worker') and app.state.verification_state_update_worker:
            app.state.verification_state_update_worker.stop()
            logger.info("Verification state update worker stopped")
    except Exception as e:
        logger.error(f"Error stopping verification state update worker: {e}")

    # NOTE: Sanctions and PEP sync are now handled by OSSPEP service
    # No scheduler shutdown needed in this app

    try:
        from app.api.job_endpoints import shutdown_job_system
        shutdown_job_system()
        logger.info("Job system shutdown successfully")
    except Exception as e:
        logger.error(f"Error during job system shutdown: {e}")


app = FastAPI(
    title="IM-OSINT API",
    description="API for IM-OSINT - KYC analyze and verify service",
    version="0.1.0",
    lifespan=lifespan,
    default_response_class=SafeJSONResponse
)

# Add rate limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# Global exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    """Handle validation errors with safe JSON encoding and detailed logging"""
    # Log detailed validation error information
    logger.error(f"Validation error for {request.method} {request.url}: {exc.errors()}")

    # Try to log request body if possible (sanitized for PII)
    try:
        import json
        body = await request.json()
        # Sanitize PII from request body before logging
        PII_FIELDS = {'full_name', 'email', 'phone_number', 'address', 'passport_number',
                     'date_of_birth', 'passport_country', 'mobile_number', 'password'}
        safe_body = {k: v for k, v in body.items() if k not in PII_FIELDS}
        logger.debug(f"Request body (PII sanitized): {json.dumps(safe_body, indent=2, default=str)}")
    except Exception:
        logger.debug("Could not log request body")

    return SafeJSONResponse(
        status_code=422,
        content={"detail": safe_jsonable_encoder(exc.errors())}
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc: StarletteHTTPException):
    """Handle HTTP exceptions with safe JSON encoding"""
    return SafeJSONResponse(
        status_code=exc.status_code,
        content={"detail": safe_jsonable_encoder(exc.detail)}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle general exceptions with safe JSON encoding"""
    logger.error(f"Unhandled exception: {exc}")
    return SafeJSONResponse(
        status_code=500,
        content={"detail": safe_jsonable_encoder("Internal server error")}
    )

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection pool is available via get_db_connection()
# Do NOT create a global connection as it causes issues when shared across threads
# Each service should get its own connection from the pool when needed

__all__ = ["app"]

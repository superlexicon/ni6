from fastapi import APIRouter, HTTPException, Depends, Query, Path, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from typing import Optional
import logging
import json
import base64
import time
from app.dto.job_models import (
    JobRequest, JobSubmissionResponse, JobStatusResponse,
    JobStatus, JobInfo, SignedJobRequest, SignatureData,
    JobDatabaseRecord
)
from app.dto import SignedVerificationRequest
# VerificationStep enum no longer used - state derived from user_identity_index data
from app.services.job_manager import JobManager
from app.services.document_analysis_worker import DocumentAnalysisWorker
from app.core import get_db_connection
from app.core.key.secp256k1 import KeyPair
from app.config.instance_config import instance_config
from app.utils.exception_handler import HTTPExceptionHelper
from app.core.logger import get_logger

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


router = APIRouter(prefix="/api/jobs", tags=["jobs"])
logger = get_logger()

# Global instances (will be initialized in main.py)
job_manager: Optional[JobManager] = None
worker: Optional[DocumentAnalysisWorker] = None

# Note: Exception handlers are defined at the application level in app/__init__.py
# to avoid conflicts with APIRouter which doesn't support exception_handler method


def safe_jsonable_encoder(obj):
    """Custom jsonable encoder that handles binary data safely"""
    def convert_binary_to_base64(o):
        if isinstance(o, bytes):
            return base64.b64encode(o).decode('ascii')
        return o

    # Custom recursion function to handle nested objects
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
        # First try to convert using safe_serialize
        return safe_serialize(obj)
    except Exception:
        # Fallback: convert to string representation
        return str(obj)


# Note: Exception handlers are defined at the application level in app/__init__.py
# to avoid conflicts with APIRouter which doesn't support exception_handler method


def get_job_manager(request: Request) -> JobManager:
    """Dependency to get job manager instance"""
    if not hasattr(request.app.state, 'job_manager') or request.app.state.job_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Job service not available"
        )
    return request.app.state.job_manager


@router.post("/analyze-async-signed", response_model=JobSubmissionResponse, status_code=202)
async def submit_signed_analysis_job(
    request: SignedJobRequest,
    job_mgr: JobManager = Depends(get_job_manager)
):
    """
    Submit a document analysis job with signature verification and state validation.

    **SECURE ENDPOINT - RECOMMENDED**

    This endpoint requires:
    1. ECDSA signature verification (proves key ownership)
    2. State validation BEFORE queuing (prevents garbage submissions)
    3. Encrypted envelope (payload security)

    **Request Body:**
    - client_public_key: Client's registered public key (for signature verification)
    - timestamp: Unix timestamp for replay protection
    - signature: ECDSA signature of timestamp
    - encrypted_key: Base64 encrypted AES key
    - key_iv: Base64 IV for key decryption
    - encrypted_payload: Base64 encrypted JSON payload
    - payload_iv: Base64 IV for payload decryption
    - target_server_public_key: Server public key for routing
    - callback_url: Optional callback URL

    **State Validation:**
    - Selfie: No previous state required, but filename must contain "otpXXXXXX" pattern
    - Passport: Requires state == 1 (selfie completed)
    - Bank Statement: Requires state == 1 (selfie completed)
    - National ID: Requires state == 1 (selfie completed)
    - Driving License: Requires state == 1 (selfie completed)
    - Resume: Requires state == 1 (selfie completed)

    **Selfie Filename Requirement:**
    Selfie filenames must contain "otp" followed by 6 digits (e.g., "selfie_otp123456.jpg").
    This prevents garbage data submissions to the selfie endpoint.

    Invalid submissions are rejected BEFORE database/queue insertion.

    **Response:**
    - job_id: Unique identifier for tracking the job
    - status: Initial job status (should be "pending")
    - message: Confirmation message
    """
    try:
        # 1. Verify signature using ECDSARecovery
        from app.repositories.user_key_repository import UserKeyRepository
        from app.repositories.otp_repository import OTPRepository
        from app.core.key.ecdsa_recovery import ECDSARecovery

        # Check both tables to determine authentication source
        # - user_keys exists: Verified user (passport, bank_statement, etc.)
        # - otp exists with is_verified=False: Initial selfie submission (before user_keys record created)
        # - neither exists: Invalid public key
        user_key_repo = UserKeyRepository()
        user_key = user_key_repo.get_key_by_public_key(request.client_public_key)

        otp_repo = OTPRepository()
        otp_record = otp_repo.get_otp_by_public_key(request.client_public_key)

        # Determine auth source based on verification state
        if user_key:
            # Verified user - authenticate via user_keys table
            logger.info(f"Authenticating verified user via user_keys table: {request.client_public_key[:16]}...")
        elif otp_record and not otp_record.get('is_verified'):
            # Unverified OTP record - this is initial selfie submission
            logger.info(f"Authenticating initial selfie via otp table: {request.client_public_key[:16]}...")
        else:
            # Not found in user_keys, and no pending OTP record
            raise HTTPException(
                status_code=401,
                detail="Invalid public key - complete OTP verification first"
            )

        # 2. Verify signature using the client_public_key (registered key or OTP key)
        # The client signs "request:{timestamp}" with their registered private key
        message = f"request:{request.timestamp}"

        # Verify signature
        is_valid = ECDSARecovery.verify_signature(
            message=message,
            r=request.signature.r,
            s=request.signature.s,
            public_key=request.client_public_key
        )

        if not is_valid:
            logger.warning(f"Signature verification failed for public key: {request.client_public_key[:16]}...")
            raise HTTPException(
                status_code=401,
                detail="Invalid signature"
            )

        # 3. Build JobRequest from signed request
        # Note: files array is empty for encrypted envelope mode
        # The worker will decrypt the payload and extract the files
        job_request = JobRequest(
            client_public_key=request.client_public_key,
            iv="",  # Encrypted envelope mode - no separate IV
            files=[],  # Extracted from encrypted_payload by worker
            target_server_public_key=request.target_server_public_key,
            encrypted_key=request.encrypted_key,
            key_iv=request.key_iv,
            encrypted_payload=request.encrypted_payload,
            payload_iv=request.payload_iv,
            callback_url=request.callback_url
        )

        # 4. Submit job with state validation (skip_state_validation=False means validate)
        # State validation happens inside create_job before queuing
        response = await job_mgr.create_job(job_request, skip_state_validation=False)

        if response.success:
            logger.info(f"Signed job {response.job_id} successfully queued with state validation")
            return response
        else:
            raise HTTPExceptionHelper.internal_server_error(
                message=response.message,
                details="Failed to queue analysis job"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in submit_signed_analysis_job: {str(e)}")
        raise HTTPExceptionHelper.internal_server_error(
            message="Failed to submit analysis job",
            details=str(e)
        )


# Helper functions for score calculation
def _calculate_docs_auth_score(selfie_result, passport_result, idcard_result, bank_statement_result) -> float:
    """
    Calculate document authentication score from PhotoHolmes forgery checks.

    Returns pass rate % (0-100) based on PhotoHolmes detection results.
    Missing documents are treated as failed checks (penalty).
    """
    total_checks = 0
    passed_checks = 0

    # Selfie PhotoHolmes checks
    if selfie_result and selfie_result.get('forgery_checks'):
        for method, data in selfie_result['forgery_checks'].items():
            total_checks += 1
            if data.get('score', 0) < data.get('threshold', 0.5):
                passed_checks += 1
    else:
        total_checks += 1  # Missing = penalty

    # Passport PhotoHolmes checks
    if passport_result and passport_result.get('forgery_checks'):
        for method, data in passport_result['forgery_checks'].items():
            total_checks += 1
            if data.get('score', 0) < data.get('threshold', 0.5):
                passed_checks += 1
    else:
        total_checks += 1

    # ID Card PhotoHolmes checks
    if idcard_result and idcard_result.get('forgery_checks'):
        for method, data in idcard_result['forgery_checks'].items():
            total_checks += 1
            if data.get('score', 0) < data.get('threshold', 0.5):
                passed_checks += 1
    else:
        total_checks += 1

    # Bank Statement PhotoHolmes checks
    if bank_statement_result and bank_statement_result.get('forgery_checks'):
        for method, data in bank_statement_result['forgery_checks'].items():
            total_checks += 1
            if data.get('score', 0) < data.get('threshold', 0.5):
                passed_checks += 1
    else:
        total_checks += 1

    return round((passed_checks / total_checks * 100), 2) if total_checks > 0 else 0.0


def _calculate_id_veri_score(selfie_result, passport_result, idcard_result, bank_statement_result) -> float:
    """
    Calculate identity verification score from non-PhotoHolmes checks.

    Returns pass rate % (0-100) based on validation checks.
    Missing documents are treated as failed checks (penalty).
    """
    total_checks = 0
    passed_checks = 0

    # Selfie: anti_spoofing, face_detected, otp_verified (3 checks)
    if selfie_result and selfie_result.get('other_checks'):
        oc = selfie_result['other_checks']
        total_checks += 3
        # Handle None values for failed/incomplete jobs
        anti_spoofing_score = oc.get('anti_spoofing_score')
        if anti_spoofing_score is not None and anti_spoofing_score >= 0.5:
            passed_checks += 1
        if oc.get('face_detected', False):
            passed_checks += 1
        if oc.get('otp_verified', False):
            passed_checks += 1
    else:
        total_checks += 3

    # Passport: face_match_confidence, document_expiry_valid, osint, worldcheck (4 checks)
    if passport_result and passport_result.get('other_checks'):
        oc = passport_result['other_checks']
        total_checks += 4  # Was 2, now 4 (added osint, worldcheck)
        # Handle None values for failed/incomplete jobs
        face_match_confidence = oc.get('face_match_confidence')
        if face_match_confidence is not None and face_match_confidence >= 80:
            passed_checks += 1
        if oc.get('document_expiry_valid', True):
            passed_checks += 1
        if oc.get('osint_result', 'FAIL') == 'PASS':
            passed_checks += 1
        # World-Check: pass if available and no match, OR if not available (auto-pass)
        if oc.get('worldcheck_available', False):
            if not oc.get('worldcheck_match', False):
                passed_checks += 1
        else:
            # World-Check not configured = auto-pass
            passed_checks += 1
    else:
        total_checks += 4

    # ID Card: face_match_confidence (1 check) - simplified validation
    if idcard_result and idcard_result.get('other_checks'):
        oc = idcard_result['other_checks']
        total_checks += 1
        # Handle None values for failed/incomplete jobs
        face_match_confidence = oc.get('face_match_confidence')
        if face_match_confidence is not None and face_match_confidence >= 80:
            passed_checks += 1
    else:
        total_checks += 1

    # Bank Statement: statement_age (1 check) - osint/worldcheck moved to passport
    if bank_statement_result and bank_statement_result.get('other_checks'):
        oc = bank_statement_result['other_checks']
        total_checks += 1  # Was 4, now 1 (name_match disabled, osint/worldcheck moved to passport)
        if oc.get('statement_age_valid', False):
            passed_checks += 1
    else:
        total_checks += 1

    return round((passed_checks / total_checks * 100), 2) if total_checks > 0 else 0.0


# Verification state endpoints (state derived from user_identity_index)
@router.post("/verification")
async def get_verification_state_signed(
    request: SignedVerificationRequest
):
    """
    Get verification state and results for all submitted documents (with signature verification).

    **SECURE ENDPOINT - RECOMMENDED**

    This endpoint requires ECDSA signature verification to prevent user enumeration
    and protect sensitive verification data.

    **Authentication:**
    - Client signs a message "request:{timestamp}" with their private key
    - Server verifies the signature using the provided public_key
    - Timestamp must be within 5 minutes (replay protection)

    **Request Body:**
    ```json
    {
        "public_key": "<client_public_key_hex>",
        "timestamp": 1234567890,
        "signature": {
            "r": "<signature_r_hex>",
            "s": "<signature_s_hex>"
        }
    }
    ```

    **Response:**
    - state: Current verification state string (selfie_pending, passport_pending, bank_pending, completed)
    - verification_state: Current verification state int (0-3)
    - user_identity_id: User identity ID
    - sequence_no: Current sequence number (0-3)
    - docs_auth_score: Document authentication score % (0-100) based on PhotoHolmes checks
    - id_veri_score: Identity verification score % (0-100) based on validation checks
    - selfie_result: Analysis result if submitted (including failed)
    - passport_result: Analysis result if submitted (including failed)
    - idcard_result: Analysis result if submitted (including failed) - NEW
    - bank_statement_result: Analysis result if submitted (including failed)
    - jobid_inprogress: Job ID of currently processing job (if any)

    **IMPORTANT:** PII is encrypted with ECIES (user-only decryption).
    The API returns encrypted envelopes in extracted_data_encrypted field.
    Client must decrypt with their private key to access PII.
    """
    try:
        global job_manager  # Fix: Declare global to access module-level job_manager
        from app.services.verification_state_service import VerificationStateService
        from app.repositories.document_submission_repository import DocumentSubmissionRepository
        from app.repositories.user_key_repository import UserKeyRepository
        from app.repositories.otp_repository import OTPRepository
        from app.core.key.ecdsa_recovery import ECDSARecovery

        # Verify signature first using ECDSARecovery
        message = f"request:{request.timestamp}"
        is_valid = ECDSARecovery.verify_signature(
            message=message,
            r=request.signature.r,
            s=request.signature.s,
            public_key=request.public_key
        )

        if not is_valid:
            logger.warning(f"Signature verification failed for public key: {request.public_key[:16]}...")
            raise HTTPException(
                status_code=401,
                detail="Invalid signature"
            )

        # Check both tables to determine authentication source
        user_key_repo = UserKeyRepository()
        user_key = user_key_repo.get_key_by_public_key(request.public_key)

        otp_repo = OTPRepository()
        otp_record = otp_repo.get_otp_by_public_key(request.public_key)

        if user_key:
            # Verified user - authenticate via user_keys table
            logger.info(f"Authenticating verified user via user_keys table: {request.public_key[:16]}...")
        elif otp_record and not otp_record.get('is_verified'):
            # Unverified OTP record - user hasn't completed selfie yet
            logger.info(f"Authenticating unverified user via otp table: {request.public_key[:16]}...")
        else:
            # Not found in user_keys, and no pending OTP record
            raise HTTPException(
                status_code=401,
                detail="Invalid public key - complete OTP verification first"
            )

        # Timestamp validation is done by Pydantic in SignedVerificationRequest
        client_public_key = request.public_key

        # Create services without passing connections - they will acquire their own
        state_service = VerificationStateService()
        doc_submission_repo = DocumentSubmissionRepository()

        # Get verification state info
        state_info = state_service.get_state_info(client_public_key)
        user_identity_id = state_info['user_identity_id']

        # Get document results if user exists
        selfie_result = None
        passport_result = None
        idcard_result = None
        bank_statement_result = None
        other_results = []  # Will hold all unmapped document types (tax_return, national_id, etc.)

        if user_identity_id:
            # Get all submissions for this user
            # With ECIES encryption, PII remains encrypted (client-side decryption)
            # The repository now constructs response_data from individual columns
            submissions = doc_submission_repo.get_user_document_submissions(
                user_identity_id=user_identity_id,
                limit=10,
                client_public_key=client_public_key,
                decrypt_extracted_data=False  # ECIES: No server-side decryption, return encrypted envelope
            )

            # Map submissions by document_type
            # Return results for ANY submitted document (including failed)
            # Document types that have dedicated response fields
            MAPPED_DOC_TYPES = {'selfie', 'video_selfie', 'passport', 'id_card', 'bank_statement'}

            for submission in submissions:
                doc_type = submission.get('document_type')
                # response_data is now constructed by the repository from individual columns
                response_data = submission.get('response_data', {})

                if doc_type == 'selfie':
                    selfie_result = response_data
                elif doc_type == 'video_selfie':
                    selfie_result = response_data
                elif doc_type == 'passport':
                    passport_result = response_data
                elif doc_type == 'id_card':
                    idcard_result = response_data
                elif doc_type == 'bank_statement':
                    bank_statement_result = response_data
                else:
                    # All other document types go to other_results
                    # (tax_return, tax_statement, national_id, driving_license, resume, etc.)
                    other_results.append(response_data)

        # Get sequence_no for the response
        sequence_no = state_service.get_sequence_no(client_public_key)

        # Get job_id of currently processing job (if any)
        jobid_inprogress = None
        if user_identity_id:
            try:
                if job_manager:
                    recent_jobs = job_manager.get_in_progress_jobs_by_user_identity_id(
                        user_identity_id, limit=10
                    )
                    # Return the first processing/pending job ID
                    for job_status in recent_jobs:
                        if job_status.job_info.status in (JobStatus.PENDING, JobStatus.PROCESSING):
                            jobid_inprogress = job_status.job_info.job_id
                            break
            except Exception as e:
                logger.warning(f"Error checking job status for {client_public_key[:16]}...: {e}")

        # Calculate summary scores
        docs_auth_score = _calculate_docs_auth_score(selfie_result, passport_result, idcard_result, bank_statement_result)
        id_veri_score = _calculate_id_veri_score(selfie_result, passport_result, idcard_result, bank_statement_result)

        # Get both state (string) and verification_state (int)
        verification_state_int = state_service.get_verification_state(client_public_key)

        response = {
            "client_public_key": client_public_key,
            "state": state_info['state'],           # String: selfie_pending, passport_pending, bank_pending, completed
            "verification_state": verification_state_int,  # Int: 0-3
            "user_identity_id": state_info['user_identity_id'],
            "sequence_no": sequence_no,
            "docs_auth_score": docs_auth_score,
            "id_veri_score": id_veri_score,
            "selfie_result": selfie_result,
            "passport_result": passport_result,
            "idcard_result": idcard_result,
            "bank_statement_result": bank_statement_result,
            "other_results": other_results,  # All other document types (tax_return, national_id, etc.)
            "jobid_inprogress": jobid_inprogress
        }

        logger.info(f"Retrieved verification state and results for: {client_public_key[:16]}...")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_verification_state_signed: {str(e)}")
        raise HTTPExceptionHelper.internal_server_error(
            message="Failed to get verification state",
            details=str(e)
        )


# Initialize function to be called from main.py
def initialize_job_system():
    """
    Initialize job system components with HTTP-based inter-instance communication.

    Each instance processes only jobs tagged with its server public key.
    Jobs are distributed via HTTP API calls to peer instances.
    """
    global job_manager, worker

    if job_manager is None:
        # Get server public key as instance identifier
        key_pair = KeyPair.generate_secp256k1_keys()
        instance_public_key = key_pair.public_key

        # Set to instance_config for use by other services (OTP broadcast, etc.)
        instance_config.set_instance_public_key(instance_public_key)

        logger.info(f"Initializing job system for instance: {instance_public_key[:16]}...")
        logger.info("Job system will use HTTP-based inter-instance communication")
        logger.info("Jobs will be distributed via HTTP API calls to peer instances")

        # Create local job queue for HTTP-based communication
        from app.core.job_queue import JobQueue

        local_queue = JobQueue()

        # Create job manager with local queue
        job_manager = JobManager(None, local_queue, instance_public_key)

        # Create worker (for direct processing, e.g., secret share endpoint)
        worker = DocumentAnalysisWorker(job_manager, None)
        job_manager.set_worker(worker)  # Set worker reference for signaling
        worker.start()  # Start the worker thread to process jobs from the queue

        # Load any pending jobs from previous runs
        loaded_count = job_manager.load_pending_jobs_on_startup()
        if loaded_count > 0:
            logger.info(f"Loaded {loaded_count} pending jobs on startup")

        # Warn when peers rely on INSTANCE_URL for replication routing but it
        # is left at the default (shadow rows would point peers at the wrong URL)
        if instance_config.has_peers() and instance_config.instance_url == "http://localhost:12410":
            logger.warning(
                "INSTANCE_URL is left at the default http://localhost:12410 while peers are "
                "configured - job replication and recovery calls depend on INSTANCE_URL being "
                "the peer-reachable URL of this instance"
            )

        # Resolve pending shadow rows (replicated jobs) against their processing
        # servers: finalize completed ones, mark dropped/failed ones
        try:
            from app.services.shadow_job_recovery_service import ShadowJobRecoveryService
            ShadowJobRecoveryService().start()
            logger.info("Shadow job recovery service started")
        except Exception as e:
            logger.warning(f"Shadow job recovery service failed to start: {e}")

        # Initialize startup sync service to fetch jobs from peers on startup
        if instance_config.startup_sync_enabled:
            from app.services.startup_sync_service import StartupSyncService
            sync_service = StartupSyncService(local_queue, instance_public_key)
            sync_service.start_sync()
            logger.info("Startup sync service started - will fetch jobs from peer instances")

        # Set job_manager for secret share endpoint
        from app.api.secret_share_endpoint import set_job_manager as set_secret_share_job_manager
        set_secret_share_job_manager(job_manager)

        logger.info("Job system initialized with HTTP-based inter-instance communication")
        return job_manager, worker, local_queue

    return job_manager, worker, None


# Cleanup function to be called on shutdown
def shutdown_job_system():
    """Shutdown job system gracefully"""
    global worker

    if worker:
        worker.stop()
        logger.info("Document analysis worker stopped")

    logger.info("Job system shutdown complete")
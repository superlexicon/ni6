from fastapi import APIRouter, status, HTTPException, Depends, Body
from app.dto.secret_share_request import (
    SecretShareRequestPayload,
    SecretShareResponse
)
from app.dto.voluntary_selfie import (
    VoluntarySelfieRequest,
    VoluntarySelfieResponse
)
from app.dto import DataResponse, SignedJobStatusRequest, SignedSecretShareRequest
from app.dto.job_models import JobRequest, FileObject, JobStatus, JobSubmissionResponse
from app.services.voluntary_selfie_service import VoluntarySelfieService
from app.services.job_manager import JobManager
from app.core.logger import get_logger
from app.core.key.hybrid_crypto import HybridCrypto
from typing import Optional, Dict, Any

logger = get_logger()
router = APIRouter(tags=["SECRET_SHARE"])

# Global job manager (will be set from main.py)
_job_manager: Optional[JobManager] = None


def get_job_manager() -> JobManager:
    """Get the global job manager instance."""
    if _job_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Job manager not initialized"
        )
    return _job_manager


def set_job_manager(manager: JobManager):
    """Set the global job manager instance."""
    global _job_manager
    _job_manager = manager


@router.post(
    "/share/request",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobSubmissionResponse,
    summary="Request secret share with signature verification",
    description="""
    Queue a secret share recovery job with ECDSA signature verification.

    **Requirements:**
    - ECDSA signature to prove request authenticity
    - User must have completed initial verification (all 3 steps)
    - Verification selfie must contain valid OTP
    - Face must match stored face embeddings (cosine similarity)
    - PhotoHolmes forgery detection must pass

    **Authentication:**
    - Client signs message "request:{timestamp}" with their private key
    - Server verifies signature before processing
    - Timestamp must be within 5 minutes (replay protection)

    **Supported Formats:**
    - Image selfie: jpg, jpeg, png, webp (OTP extracted via OCR/filename)
    - Video selfie: mp4, mov, webm, avi, mkv (OTP extracted via hand gestures)

    **Process:**
    1. Job is queued and returns immediately with job_id
    2. Worker processes: OTP validation → PhotoHolmes → face matching → re-encryption
    3. Poll POST /share/status with signature for results OR provide callback_url
    """
)
async def request_secret_share(
    request: SignedSecretShareRequest,
    job_mgr: JobManager = Depends(get_job_manager)
) -> JobSubmissionResponse:
    """
    Queue secret share recovery job for asynchronous processing.

    Requires ECDSA signature verification to prevent unauthorized recovery requests.
    """
    try:
        from app.repositories.user_key_repository import UserKeyRepository
        from app.core.key.ecdsa_recovery import ECDSARecovery

        # Step 1: Get user_key and identity_id using recovery lookup strategy
        user_key_repo = UserKeyRepository()
        user_key = None
        identity_id = None
        mobile_number = None
        country_code = None

        # Recovery lookup strategy:
        # 1. If mobile_number provided -> lookup by mobile_number (with country code)
        # 2. If mobile_number NOT provided -> use facial matching (NOT public_key lookup!)
        # IMPORTANT: For recovery, temporary public_keys won't exist in user_keys table
        if hasattr(request, 'mobile_number') and request.mobile_number:
            # Concatenate country_code with mobile_number to match database format
            full_mobile_number = f"{request.country_code}{request.mobile_number}" if request.country_code else request.mobile_number
            user_key = user_key_repo.get_key_by_mobile_number(
                mobile_number=full_mobile_number,
                country_code=None  # Not needed - we're passing the full number
            )
            if user_key:
                identity_id = user_key.get('user_identity_id')
                mobile_number = user_key.get('mobile_number')
                country_code = user_key.get('country_code')
                logger.info(f"Found user_key via mobile_number lookup: {full_mobile_number}")
            else:
                logger.warning(f"No user_key found for mobile_number: {full_mobile_number}")
                # Continue - will use facial matching to identify user
        else:
            # No mobile_number provided - will use facial matching in job processor
            # Do NOT attempt public_key lookup for recovery (temporary keys don't exist)
            logger.info(f"No mobile_number provided - will use facial matching for identification")

        # Step 2: Verify signature with temp_public_key (user's recovery key)
        message = f"request:{request.timestamp}"
        is_valid = ECDSARecovery.verify_signature(
            message=message,
            r=request.signature.r,
            s=request.signature.s,
            public_key=request.temp_public_key  # This is temp_public_key
        )

        if not is_valid:
            logger.warning(
                f"Signature verification failed for temp_public_key: {request.temp_public_key[:16]}..."
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid signature"
            )

        logger.info(f"Signature verified for temp_public_key {request.temp_public_key[:16]}...")

        # Step 3: Create job with identity_id, mobile_number, and country_code
        # The temp_public_key (request.temp_public_key) will be used for re-encryption
        job_request = JobRequest(
            client_public_key=request.temp_public_key,  # temp_public_key for re-encryption
            iv="",
            files=[FileObject(
                filename=request.filename or "recovery_selfie.jpg",
                file_data=request.selfie_data,
                file_type="selfie",  # For classification (selfie vs document)
                document_type="secret_share_recovery"  # For routing to correct processor
            )],
            identity_id=identity_id,  # None when recovery_mode=True (will be found via facial matching)
            mobile_number=mobile_number,  # None when no mobile_number provided
            country_code=country_code,
            otp_code=request.otp_code,
            callback_url=request.callback_url,
            target_server_public_key=request.target_server_public_key,
            api_url=request.api_url  # Optional: filter shares by API URL
        )

        response = await job_mgr.create_job(job_request)

        if response.success:
            logger.info(f"Secret share recovery job queued: {response.job_id}")
            return response
        else:
            logger.error(f"Failed to queue recovery job: {response.message}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=response.message
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in request_secret_share: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to queue secret share recovery: {str(e)}"
        )


@router.post(
    "/share/status",
    response_model=SecretShareResponse,
    summary="Get secret share recovery status (with signature verification)",
    description="""
    Get the status and results of a secret share recovery job with signature verification.

    **SECURE ENDPOINT - RECOMMENDED**

    This endpoint requires ECDSA signature verification to ensure users
    can only check status for their own recovery jobs.

    **Authentication:**
    - Client signs a message "request:{timestamp}" with their private key
    - Server verifies the signature and that the job_id belongs to the public_key
    - Timestamp must be within 5 minutes (replay protection)

    **Request Body:**
    ```json
    {
        "job_id": "<job_id>",
        "public_key": "<client_public_key_hex>",
        "timestamp": 1234567890,
        "signature": {
            "r": "<signature_r_hex>",
            "s": "<signature_s_hex>"
        }
    }
    ```

    **Response:**
    - If pending/processing: returns status only
    - If completed: returns encrypted_secret_share and iv (or shares list for multiple devices)
    - If failed: returns error message
    """
)
async def get_recovery_status_signed(
    request: SignedJobStatusRequest,
    job_mgr: JobManager = Depends(get_job_manager)
) -> SecretShareResponse:
    """
    Get secret share recovery status by job ID with signature verification.

    Checks for active jobs and returns results when complete.
    """
    try:
        from app.core.key.ecdsa_recovery import ECDSARecovery

        # Verify signature (signature verification is sufficient to prove key ownership)
        # NOTE: We do NOT check user_keys table here because share recovery uses temp public keys
        # that are only stored in document_submissions, not in user_keys
        message = f"request:{request.timestamp}"
        is_valid = ECDSARecovery.verify_signature(
            message=message,
            r=request.signature.r,
            s=request.signature.s,
            public_key=request.public_key
        )

        if not is_valid:
            logger.warning(
                f"Signature verification failed for public key: {request.public_key[:16]}..."
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid signature"
            )

        logger.info(f"Signature verified for {request.public_key[:16]}...")

        # Get job status by job_id
        job_status = job_mgr.get_job_status(request.job_id)

        # If not found in document_analysis_jobs, check document_submissions (completed jobs)
        if not job_status:
            from app.repositories.document_submission_repository import DocumentSubmissionRepository
            submission_repo = DocumentSubmissionRepository()

            # Query by client_public_key (temp_public_key) which was stored during submission
            submission = submission_repo.get_submission_by_public_key(request.public_key)

            if submission and submission.get('response_data', {}).get('extracted_data_encrypted'):
                # Found completed job with ECIES-encrypted results
                logger.info(f"Found completed recovery for temp_public_key {request.public_key[:16]}... in document_submissions")

                # Return ECIES envelope for client-side decryption (parsed dict from response_data)
                return SecretShareResponse(
                    success=True,
                    message="Recovery completed",
                    extracted_data_encrypted=submission['response_data']['extracted_data_encrypted']
                )
            else:
                # Job not found anywhere
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No recovery job found for this public key"
                )

        # Verify the job belongs to this user (by checking client_public_key in request_data)
        job_request_data = job_status.job_info.request_data
        job_public_key = job_request_data.get('client_public_key')

        if job_public_key != request.public_key:
            logger.warning(f"Access denied: {request.public_key[:16]}... tried to access job for {job_public_key[:16]}...")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: Job does not belong to this public key"
            )

        job_info = job_status.job_info
        logger.info(f"Found job {request.job_id}, status: {job_info.status.value}")

        # If job is still pending or processing, return status
        if job_info.status in (JobStatus.PENDING, JobStatus.PROCESSING):
            return SecretShareResponse(
                success=False,
                message=f"Recovery in progress (status: {job_info.status.value})"
            )

        # If job completed, return results from job
        if job_info.status == JobStatus.COMPLETED:
            results = job_status.results or {}

            # Handle new response format with shares list
            if results.get('shares'):
                from app.dto.secret_share_request import RecoveredShare
                shares_data = [
                    RecoveredShare(**share) for share in results['shares']
                ]
                logger.info(f"Returning completed job results for {request.job_id}: {results.get('total_shares')} shares")
                return SecretShareResponse(
                    success=results.get('success', True),
                    identity_id=results.get('identity_id'),
                    shares=shares_data,
                    total_shares=results.get('total_shares'),
                    message=results.get('message', 'Recovery completed'),
                    face_match_confidence=results.get('face_match_confidence'),
                    faces_checked=results.get('faces_checked')
                )
            # Legacy format for backward compatibility
            elif results.get('encrypted_secret_share'):
                logger.info(f"Returning completed job results for {request.job_id}")
                return SecretShareResponse(
                    success=results.get('success', True),
                    encrypted_secret_share=results.get('encrypted_secret_share'),
                    iv=results.get('iv'),
                    message=results.get('message', 'Recovery completed'),
                    face_match_confidence=results.get('face_match_confidence'),
                    faces_checked=results.get('faces_checked')
                )

        # If job failed, return error
        if job_info.status == JobStatus.FAILED:
            return SecretShareResponse(
                success=False,
                message=job_info.error_message or "Recovery failed"
            )

        # Unknown status
        return SecretShareResponse(
            success=False,
            message=f"Unknown job status: {job_info.status.value}"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_recovery_status_signed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get recovery status: {str(e)}"
        )


@router.post(
    "/face/submit",
    status_code=status.HTTP_201_CREATED,
    response_model=DataResponse[VoluntarySelfieResponse],
    summary="Submit voluntary selfie to improve face matching",
    description="""
    Submit a voluntary selfie to improve face matching accuracy.

    **Requirements:**
    - User must be registered in system
    - Face quality score >= 70%
    - Liveness check must pass (70%+)

    **Purpose:**
    Build up face biometric history to improve accuracy of:
    - Secret share requests
    - Future face verification operations

    **Payload Encryption:**
    - Entire request payload must be encrypted with server's public key
    """
)
async def submit_voluntary_selfie(
    request: VoluntarySelfieRequest
) -> DataResponse[VoluntarySelfieResponse]:
    """
    Submit voluntary selfie to improve face matching accuracy.

    This endpoint allows users to build up their face biometric history
    for improved verification accuracy.
    """
    try:
        # Decode base64 selfie image
        from app.utils import DecodeBase64
        decode_base64 = DecodeBase64()

        try:
            selfie_image_bytes = decode_base64.decode_base64(request.selfie_data)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid selfie data: {str(e)}"
            )

        # Initialize service
        voluntary_selfie_service = VoluntarySelfieService()

        # Submit selfie
        result = await voluntary_selfie_service.submit_voluntary_selfie(
            public_key=request.public_key,
            selfie_image_bytes=selfie_image_bytes
        )

        # Build response
        response = VoluntarySelfieResponse(
            success=result['success'],
            message=result['message'],
            face_quality_score=result.get('face_quality_score'),
            anti_spoofing_score=result.get('anti_spoofing_score'),
            biometric_id=result.get('biometric_id')
        )

        return DataResponse[VoluntarySelfieResponse](data=response)

    except ValueError as e:
        # Business logic errors
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        # Unexpected errors
        logger.error(f"Unexpected error in submit_voluntary_selfie: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit voluntary selfie: {str(e)}"
        )

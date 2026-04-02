"""
Key Management API Endpoints - Add/remove public keys with face verification.

These endpoints allow users to:
1. Add new public keys to their identity (with face verification)
2. Remove existing public keys from their identity (with face verification)

Both operations use the same security as secret share recovery:
- Face verification (OTP → PhotoHolmes → anti-spoofing → face matching)
- Request encrypted with existing registered key (proves ownership)
- Audit trail with face embedding storage
"""

from fastapi import APIRouter, status, HTTPException, Depends, Body
from app.dto import DataResponse, SignedJobStatusRequest
from app.dto.job_models import JobRequest, FileObject, JobStatus, JobSubmissionResponse
from app.services.job_manager import JobManager
from app.core.logger import get_logger
from app.core.key.hybrid_crypto import HybridCrypto
from typing import Dict, Any, Optional

logger = get_logger()
router = APIRouter(tags=["KEY_MANAGEMENT"])

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
    "/keys/add",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobSubmissionResponse,
    summary="Add a new public key to your identity (async)",
    description="""
    Queue a job to add a new public key to your user identity.

    **Requirements:**
    - Request must be encrypted with an existing registered key (proves ownership)
    - Verification selfie must contain valid OTP
    - Face must match stored face embeddings (70% confidence)
    - User must have completed full verification (selfie, passport, bank statement)
    - New public key must not already exist

    **Request Format (encrypted envelope):**
    ```json
    {
        "client_public_key": "<existing_key_hex>",
        "encrypted_key": "<base64>",
        "key_iv": "<base64>",
        "encrypted_payload": "<base64>",
        "payload_iv": "<base64>",
        "otp_code": "123456"
    }
    ```

    **Encrypted Payload Contents:**
    ```json
    {
        "new_public_key": "<key_to_add>",
        "secret_share": "<client_secret>",
        "selfie_data": "<base64>",
        "filename": "selfie_123456.jpg",
        "otp_code": "123456"
    }
    ```

    **Process:**
    1. Job is queued and returns immediately with job ID
    2. Worker processes: decrypt → validate → selfie verification → face matching → add key
    3. Poll GET /keys/status/{job_id} for results OR provide callback_url

    **Security:**
    - Encrypting key must already be registered
    - Face verification prevents key hijacking
    - Cannot add key without matching face
    """
)
async def add_public_key(
    request_data: Dict[str, Any] = Body(...),
    job_mgr: JobManager = Depends(get_job_manager)
) -> JobSubmissionResponse:
    """
    Queue a job to add a new public key to user identity.

    Accepts encrypted envelope requests only for security.
    The encrypting key must already be registered in the system.
    """
    try:
        hybrid_crypto = HybridCrypto()

        # Verify request is an encrypted envelope
        if not hybrid_crypto.is_encrypted_envelope(request_data):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Request must be encrypted envelope"
            )

        logger.info("Received encrypted envelope for add_public_key")

        # Decrypt envelope to get payload
        try:
            from app.core.key.hybrid_crypto import HybridCryptoError
            decrypted = hybrid_crypto.decrypt_envelope(request_data)
            decrypted_payload = decrypted.payload

            # Extract required fields
            encrypting_public_key = request_data.get('client_public_key')
            new_public_key = decrypted_payload.get('new_public_key')
            client_secret_share = decrypted_payload.get('secret_share')
            selfie_data = decrypted_payload.get('selfie_data')
            filename = decrypted_payload.get('filename', 'add_key_selfie.jpg')
            otp_code = request_data.get('otp_code') or decrypted_payload.get('otp_code')
            callback_url = decrypted_payload.get('callback_url')

            # Validate required fields
            if not encrypting_public_key:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="client_public_key is required"
                )
            if not new_public_key:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="new_public_key is required in encrypted payload"
                )
            if not client_secret_share:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="secret_share is required in encrypted payload"
                )
            if not selfie_data:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="selfie_data is required in encrypted payload"
                )

            logger.info(
                f"Decrypted add_public_key request: encrypting={encrypting_public_key[:16]}..., "
                f"new_key={new_public_key[:16]}..."
            )

            # Create job request
            # Use encrypting_public_key as job_id for status lookup
            job_id = f"add_key_{encrypting_public_key[:32]}"

            job_request = JobRequest(
                client_public_key=encrypting_public_key,
                iv="",  # Not used for key management
                files=[FileObject(
                    filename=filename,
                    file_data=selfie_data,
                    file_type="add_public_key"
                )],
                # Pass additional data via metadata
                metadata={
                    'new_public_key': new_public_key,
                    'client_secret_share': client_secret_share,
                    'document_type': 'add_public_key'
                },
                otp_code=otp_code,
                callback_url=callback_url
            )

            # Override job_id with our custom one
            job_request.job_id = job_id

        except HybridCryptoError as e:
            logger.error(f"Failed to decrypt add_public_key envelope: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to decrypt request: {str(e)}"
            )

        # Create job
        response = await job_mgr.create_job(job_request)

        if response.success:
            logger.info(f"Add key job queued: {job_id}")
            return response
        else:
            logger.error(f"Failed to queue add key job: {response.message}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=response.message
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in add_public_key: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to queue add key job: {str(e)}"
        )


@router.delete(
    "/keys/remove",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobSubmissionResponse,
    summary="Remove a public key from your identity (async)",
    description="""
    Queue a job to remove an existing public key from your user identity.

    **Requirements:**
    - Request must be encrypted with an existing registered key (can be different from key being removed)
    - Verification selfie must contain valid OTP
    - Face must match stored face embeddings (70% confidence)
    - User must have at least 2 keys (cannot remove last key - prevents lockout)
    - Key to remove must belong to the same user identity

    **Request Format (encrypted envelope):**
    ```json
    {
        "client_public_key": "<encrypting_key_hex>",
        "encrypted_key": "<base64>",
        "key_iv": "<base64>",
        "encrypted_payload": "<base64>",
        "payload_iv": "<base64>",
        "otp_code": "654321"
    }
    ```

    **Encrypted Payload Contents:**
    ```json
    {
        "public_key_to_remove": "<key_to_delete>",
        "selfie_data": "<base64>",
        "filename": "selfie_789012.jpg",
        "otp_code": "654321"
    }
    ```

    **Process:**
    1. Job is queued and returns immediately with job ID
    2. Worker processes: decrypt → validate → selfie verification → face matching → remove key
    3. Poll GET /keys/status/{job_id} for results OR provide callback_url

    **Security:**
    - Encrypting key must already be registered
    - Face verification prevents unauthorized key removal
    - Cannot remove last key (safety check)
    - Audit trail with face embedding storage
    """
)
async def remove_public_key(
    request_data: Dict[str, Any] = Body(...),
    job_mgr: JobManager = Depends(get_job_manager)
) -> JobSubmissionResponse:
    """
    Queue a job to remove a public key from user identity.

    Accepts encrypted envelope requests only for security.
    The encrypting key must already be registered in the system.
    """
    try:
        hybrid_crypto = HybridCrypto()

        # Verify request is an encrypted envelope
        if not hybrid_crypto.is_encrypted_envelope(request_data):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Request must be encrypted envelope"
            )

        logger.info("Received encrypted envelope for remove_public_key")

        # Decrypt envelope to get payload
        try:
            from app.core.key.hybrid_crypto import HybridCryptoError
            decrypted = hybrid_crypto.decrypt_envelope(request_data)
            decrypted_payload = decrypted.payload

            # Extract required fields
            encrypting_public_key = request_data.get('client_public_key')
            public_key_to_remove = decrypted_payload.get('public_key_to_remove')
            selfie_data = decrypted_payload.get('selfie_data')
            filename = decrypted_payload.get('filename', 'remove_key_selfie.jpg')
            otp_code = request_data.get('otp_code') or decrypted_payload.get('otp_code')
            callback_url = decrypted_payload.get('callback_url')

            # Validate required fields
            if not encrypting_public_key:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="client_public_key is required"
                )
            if not public_key_to_remove:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="public_key_to_remove is required in encrypted payload"
                )
            if not selfie_data:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="selfie_data is required in encrypted payload"
                )

            logger.info(
                f"Decrypted remove_public_key request: encrypting={encrypting_public_key[:16]}..., "
                f"key_to_remove={public_key_to_remove[:16]}..."
            )

            # Create job request
            # Use encrypting_public_key as job_id for status lookup
            job_id = f"remove_key_{encrypting_public_key[:32]}"

            job_request = JobRequest(
                client_public_key=encrypting_public_key,
                iv="",  # Not used for key management
                files=[FileObject(
                    filename=filename,
                    file_data=selfie_data,
                    file_type="remove_public_key"
                )],
                # Pass additional data via metadata
                metadata={
                    'public_key_to_remove': public_key_to_remove,
                    'document_type': 'remove_public_key'
                },
                otp_code=otp_code,
                callback_url=callback_url
            )

            # Override job_id with our custom one
            job_request.job_id = job_id

        except HybridCryptoError as e:
            logger.error(f"Failed to decrypt remove_public_key envelope: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to decrypt request: {str(e)}"
            )

        # Create job
        response = await job_mgr.create_job(job_request)

        if response.success:
            logger.info(f"Remove key job queued: {job_id}")
            return response
        else:
            logger.error(f"Failed to queue remove key job: {response.message}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=response.message
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in remove_public_key: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to queue remove key job: {str(e)}"
        )


@router.post(
    "/keys/status",
    response_model=DataResponse[Dict[str, Any]],
    summary="Get key management job status (with signature verification)",
    description="""
    Get the status and results of a key management job with signature verification.

    **SECURE ENDPOINT - RECOMMENDED**

    This endpoint requires ECDSA signature verification to ensure users
    can only check status for their own jobs.

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

    **Job ID Format:**
    - Add key: `add_key_<public_key_prefix>`
    - Remove key: `remove_key_<public_key_prefix>`

    **Response:**
    - If pending/processing: returns status only
    - If completed: returns operation results
    - If failed: returns error message
    """
)
async def get_key_management_status_signed(
    request: SignedJobStatusRequest,
    job_mgr: JobManager = Depends(get_job_manager)
) -> DataResponse[Dict[str, Any]]:
    """
    Get key management job status by job ID with signature verification.
    """
    try:
        from app.repositories.user_key_repository import UserKeyRepository

        # Verify public_key exists in user_keys table
        user_key_repo = UserKeyRepository()
        user_key = user_key_repo.get_key_by_public_key(request.public_key)

        if not user_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid public key"
            )

        job_id = request.job_id
        job_status = job_mgr.get_job_status(job_id)

        if not job_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job not found: {job_id}"
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
        logger.info(f"Found job {job_id}, status: {job_info.status.value}")

        response_data = {
            'job_id': job_id,
            'status': job_info.status.value,
            'created_at': job_info.created_at.isoformat() if job_info.created_at else None
        }

        # If job completed, return results
        if job_info.status == JobStatus.COMPLETED:
            results = job_status.results or {}
            response_data.update({
                'success': results.get('success', True),
                'message': results.get('message', 'Operation completed'),
                **{k: v for k, v in results.items() if k not in ['success', 'message']}
            })
            logger.info(f"Returning completed results for {job_id}")
        elif job_info.status == JobStatus.FAILED:
            response_data.update({
                'success': False,
                'message': job_info.error_message or "Operation failed"
            })

        return DataResponse[Dict[str, Any]](data=response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_key_management_status_signed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get job status: {str(e)}"
        )


@router.get(
    "/keys/status/{job_id}",
    response_model=DataResponse[Dict[str, Any]],
    summary="Get key management job status (deprecated)",
    description="""
    [DEPRECATED] Use POST /keys/status with signature verification instead.

    Get the status and results of a key management job.

    **Job ID Format:**
    - Add key: `add_key_<public_key_prefix>`
    - Remove key: `remove_key_<public_key_prefix>`

    **Response:**
    - If pending/processing: returns status only
    - If completed: returns operation results
    - If failed: returns error message
    """,
    deprecated=True
)
async def get_key_management_status(
    job_id: str,
    job_mgr: JobManager = Depends(get_job_manager)
) -> DataResponse[Dict[str, Any]]:
    """
    Get key management job status by job ID.
    """
    try:
        job_status = job_mgr.get_job_by_id(job_id)

        if not job_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job not found: {job_id}"
            )

        job_info = job_status.job_info
        logger.info(f"Found job {job_id}, status: {job_info.status.value}")

        response_data = {
            'job_id': job_id,
            'status': job_info.status.value,
            'created_at': job_info.created_at.isoformat() if job_info.created_at else None
        }

        # If job completed, return results
        if job_info.status == JobStatus.COMPLETED:
            results = job_status.results or {}
            response_data.update({
                'success': results.get('success', True),
                'message': results.get('message', 'Operation completed'),
                **{k: v for k, v in results.items() if k not in ['success', 'message']}
            })
            logger.info(f"Returning completed results for {job_id}")
        elif job_info.status == JobStatus.FAILED:
            response_data.update({
                'success': False,
                'message': job_info.error_message or "Operation failed"
            })

        return DataResponse[Dict[str, Any]](data=response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_key_management_status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get job status: {str(e)}"
        )

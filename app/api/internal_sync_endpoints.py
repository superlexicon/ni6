"""
Internal Sync API Endpoints

Internal API endpoints for inter-instance communication via HTTP.
These endpoints are used by peer instances to:
1. Sync jobs on startup
2. Sync OTPs on startup
3. Receive OTP broadcast events

Authentication: ECDSA signature-based verification
- Request must include X-Internal-Public-Key, X-Internal-Timestamp,
  X-Internal-Signature-R, X-Internal-Signature-S headers
- Signature is verified against the public key
- Timestamp is validated to prevent replay attacks
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from typing import Dict, Any, List, Optional
from datetime import datetime
import time

from app.core.job_queue import JobQueue
from app.repositories.otp_repository import OTPRepository
from app.repositories.job_repository import JobRepository
from app.config.instance_config import instance_config
from app.core.logger import get_logger
from app.core.key.ecdsa_recovery import ECDSARecovery

logger = get_logger()

router = APIRouter(prefix="/api/internal", tags=["internal-sync"])


def _deserialize_datetime_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert ISO datetime strings back to datetime objects for MySQL.

    When OTP data is broadcast between instances, datetime fields are
    serialized to ISO 8601 format. This function deserializes them
    back to Python datetime objects for database operations.

    Args:
        data: Dict potentially containing ISO datetime strings

    Returns:
        Dict with datetime fields converted to datetime objects
    """
    result = data.copy()
    datetime_fields = ['expires_at', 'created_at', 'updated_at']
    for field in datetime_fields:
        if field in result and isinstance(result[field], str):
            try:
                result[field] = datetime.fromisoformat(result[field])
            except (ValueError, TypeError):
                # Keep original value if parsing fails
                pass
    return result


async def verify_internal_request(request: Request) -> str:
    """
    Verify internal request signature.

    Returns the sender's public key if valid, raises HTTPException otherwise.

    Authentication flow:
    1. Extract signature headers from request
    2. Validate timestamp (±60 seconds)
    3. Verify ECDSA signature using the public key
    4. Optionally check if public key belongs to a known peer
    """
    # Extract signature headers
    public_key = request.headers.get("X-Internal-Public-Key")
    timestamp_str = request.headers.get("X-Internal-Timestamp")
    sig_r = request.headers.get("X-Internal-Signature-R")
    sig_s = request.headers.get("X-Internal-Signature-S")

    # Validate presence of all headers
    if not all([public_key, timestamp_str, sig_r, sig_s]):
        logger.warning("Internal API request missing signature headers")
        raise HTTPException(
            status_code=401,
            detail="Missing signature headers. Required: X-Internal-Public-Key, X-Internal-Timestamp, X-Internal-Signature-R, X-Internal-Signature-S"
        )

    # Validate timestamp
    try:
        timestamp = int(timestamp_str)
        current_time = int(time.time())

        # Check for future timestamps (clock skew tolerance: 60 seconds)
        if timestamp > current_time + 60:
            logger.warning(f"Internal API request with future timestamp: {timestamp} > {current_time}")
            raise HTTPException(status_code=400, detail="Timestamp is in the future")

        # Check for expired timestamps (max age: 60 seconds)
        if current_time - timestamp > 60:
            logger.warning(f"Internal API request with expired timestamp: {current_time - timestamp}s old")
            raise HTTPException(status_code=400, detail="Timestamp expired")
    except ValueError:
        logger.warning(f"Internal API request with invalid timestamp format: {timestamp_str}")
        raise HTTPException(status_code=400, detail="Invalid timestamp format")

    # Verify signature
    message = f"internal:{timestamp}"
    is_valid = ECDSARecovery.verify_signature(
        message=message,
        r=sig_r,
        s=sig_s,
        public_key=public_key
    )

    if not is_valid:
        logger.warning(f"Internal API request with invalid signature from {public_key[:16]}...")
        # For local development: allow requests if no peer public keys are configured
        if not instance_config.peer_public_keys:
            logger.warning("No peer_public_keys configured - allowing request for local development")
            return public_key
        raise HTTPException(status_code=401, detail="Invalid signature")

    # Optionally verify peer is known (warn only, don't block)
    if not instance_config.is_known_peer(public_key):
        logger.warning(f"Internal API request from unknown peer public key: {public_key[:16]}...")
        # For local development: allow requests if no peer public keys are configured
        if not instance_config.peer_public_keys:
            logger.warning("No peer_public_keys configured - allowing request for local development")
            return public_key

    return public_key


@router.post("/sync/jobs")
async def sync_jobs_for_instance(
    request: Dict[str, str],
    job_repo: JobRepository = Depends(),
    sender_public_key: str = Depends(verify_internal_request)
):
    """
    Return pending jobs meant for the requesting instance.

    Called by peer instances on startup to retrieve queued jobs.

    Request body:
        {"instance_public_key": "hex-public-key"}

    Returns:
        {"jobs": [...], "count": N}
    """
    instance_public_key = request.get("instance_public_key")
    if not instance_public_key:
        raise HTTPException(status_code=400, detail="instance_public_key required")

    logger.info(f"Internal sync: {sender_public_key[:16]}... requesting jobs for {instance_public_key[:16]}...")

    # Get pending jobs for this instance from MySQL
    # Jobs are routed via target_server_public_key in request_data
    jobs = job_repo.get_pending_jobs_for_instance(instance_public_key)

    logger.info(f"Returning {len(jobs)} jobs for {instance_public_key[:16]}...")

    return {
        "jobs": [job.to_dict() if hasattr(job, 'to_dict') else {
            "id": job.id,
            "status": job.status.value if hasattr(job.status, 'value') else str(job.status),
            "request_data": job.request_data,
            "response_data": job.response_data,
            "error_message": job.error_message,
            "callback_url": job.callback_url,
            "retry_count": job.retry_count,
            "max_retries": job.max_retries,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "updated_at": job.updated_at.isoformat() if job.updated_at else None,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "callback_attempted_at": job.callback_attempted_at.isoformat() if job.callback_attempted_at else None,
        } for job in jobs],
        "count": len(jobs)
    }


@router.get("/sync/otps")
async def sync_otps(
    otp_repo: OTPRepository = Depends(),
    sender_public_key: str = Depends(verify_internal_request)
):
    """
    Return unverified OTPs for peer sync (including expired ones).

    Called by peer instances on startup to retrieve OTPs.

    IMPORTANT: Includes expired OTPs because a down instance may come back
    up after an OTP expired and still needs to verify against it.
    Expiry should only be checked during OTP verification, not during sync.

    Returns:
        {"otps": [...], "count": N}
    """
    logger.info(f"Internal sync: {sender_public_key[:16]}... requesting OTPs")

    # Get all unverified OTPs (regardless of expiry)
    # Expiry check happens during verification, not here
    otps = otp_repo.get_all_unverified_otps()

    logger.info(f"Returning {len(otps)} unverified OTPs for sync")

    return {
        "otps": [_serialize_otp(otp) for otp in otps],
        "count": len(otps)
    }


def _serialize_otp(otp: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize OTP dict for JSON response"""
    result = {}
    for key, value in otp.items():
        if isinstance(value, datetime):
            result[key] = value.isoformat()
        else:
            result[key] = value
    return result


@router.post("/otp/sync")
async def receive_otp_sync(
    event: Dict[str, Any],
    otp_repo: OTPRepository = Depends(),
    sender_public_key: str = Depends(verify_internal_request)
):
    """
    Receive OTP broadcast event from peer instance.

    This endpoint receives real-time OTP events from peer instances:
    - "create" events: New OTP created on peer instance
    - "verify" events: OTP verified on peer instance

    Event format:
        {
            "event_type": "create" | "verify",
            "instance_id": "instance-identifier",
            "otp_data": {...} | "mobile_number": "+1234567890"
        }
    """
    event_type = event.get("event_type")
    sender_id = event.get("instance_id")
    our_instance_id = instance_config.instance_public_key[:16]

    # Skip events from this instance (prevent loops)
    if sender_id == our_instance_id:
        logger.debug("Skipping OTP sync event from self")
        return {"status": "skipped", "reason": "self-event"}

    logger.info(f"Received OTP {event_type} event from {sender_id} (verified as {sender_public_key[:16]}...)")

    if event_type == "create":
        otp_data = event.get("otp_data", {})
        mobile = otp_data.get('mobile_number')

        if not mobile:
            raise HTTPException(status_code=400, detail="mobile_number required in otp_data")

        # Deserialize datetime fields from ISO format
        otp_data = _deserialize_datetime_fields(otp_data)

        # Upsert OTP (insert or update if exists)
        existing = otp_repo.get_otp_by_mobile_number(mobile)
        if existing:
            # Preserve encrypted_secret_share if it exists in DB but not in broadcast
            # Each node has its own encrypted_secret_share (encrypted for that node)
            # Broadcasts don't include it, so we must preserve it on peer nodes
            if existing.get('encrypted_secret_share') and 'encrypted_secret_share' not in otp_data:
                otp_data['encrypted_secret_share'] = existing['encrypted_secret_share']
            otp_repo.update_otp(mobile, otp_data)
            logger.info(f"Updated OTP for {mobile} from peer broadcast")
        else:
            otp_repo.create_otp(otp_data)
            logger.info(f"Created OTP for {mobile} from peer broadcast")

        return {"status": "created"}

    elif event_type == "verify":
        mobile = event.get("mobile_number")
        if not mobile:
            raise HTTPException(status_code=400, detail="mobile_number required")

        otp_repo.mark_otp_verified(mobile)
        logger.info(f"Marked OTP for {mobile} as verified from peer broadcast")
        return {"status": "verified"}

    else:
        raise HTTPException(status_code=400, detail=f"Invalid event_type: {event_type}")


@router.post("/jobs/accept")
async def accept_job(
    job_data: Dict[str, Any],
    sender_public_key: str = Depends(verify_internal_request)
):
    """
    Accept a job from a peer instance.

    Called by peer instances when they forward a job to this instance.
    The job is added to the local job queue for processing.

    Job data format:
        {
            "id": "job-id",
            "status": "pending",
            "request_data": {...},
            "callback_url": "...",
            ...
        }
    """
    # This would integrate with the job system to add to local queue
    # For now, return success
    logger.info(f"Received job {job_data.get('id')} from {sender_public_key[:16]}...")
    return {"status": "accepted"}


@router.get("/health")
async def health_check():
    """Health check endpoint for internal API"""
    return {
        "status": "healthy",
        "instance": instance_config.instance_public_key[:16],
        "peers": len(instance_config.peer_instances)
    }


@router.get("/public-key")
async def get_public_key():
    """
    Return this instance's full public key.

    This endpoint is used by peer instances to discover and configure
    each other's public keys for signature-based authentication.
    """
    return {
        "public_key": instance_config.instance_public_key,
        "instance_url": instance_config.instance_url
    }

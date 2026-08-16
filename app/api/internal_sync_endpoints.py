"""
Internal Sync API Endpoints - Simplified (Public Key Only)

Internal API endpoints for inter-instance communication via HTTP.
Authentication: ECDSA signature-based verification
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import time

from app.repositories.otp_repository import OTPRepository
from app.repositories.job_repository import JobRepository
from app.config.instance_config import instance_config
from app.core.logger import get_logger
from app.core.key.ecdsa_recovery import ECDSARecovery

logger = get_logger()

router = APIRouter(prefix="/api/internal", tags=["internal-sync"])


def _deserialize_datetime_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert ISO datetime strings back to datetime objects."""
    result = data.copy()
    datetime_fields = ['expires_at', 'created_at', 'updated_at']
    for field in datetime_fields:
        if field in result and isinstance(result[field], str):
            try:
                result[field] = datetime.fromisoformat(result[field])
            except (ValueError, TypeError):
                pass
    return result


async def verify_internal_request(request: Request) -> str:
    """Verify internal request signature. Returns sender's public key if valid."""
    public_key = request.headers.get("X-Internal-Public-Key")
    timestamp_str = request.headers.get("X-Internal-Timestamp")
    sig_r = request.headers.get("X-Internal-Signature-R")
    sig_s = request.headers.get("X-Internal-Signature-S")

    if not all([public_key, timestamp_str, sig_r, sig_s]):
        logger.warning("Internal API request missing signature headers")
        raise HTTPException(status_code=401, detail="Missing signature headers")

    # Validate timestamp
    try:
        timestamp = int(timestamp_str)
        current_time = int(time.time())
        if timestamp > current_time + 60:
            raise HTTPException(status_code=400, detail="Timestamp is in the future")
        if current_time - timestamp > 60:
            raise HTTPException(status_code=400, detail="Timestamp expired")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid timestamp format")

    # Verify signature
    message = f"internal:{timestamp}"
    is_valid = ECDSARecovery.verify_signature(
        message=message, r=sig_r, s=sig_s, public_key=public_key
    )

    if not is_valid:
        logger.warning(f"Internal API request with invalid signature from {public_key[:16]}...")
        if not instance_config.peer_public_keys:
            logger.warning("No peer_public_keys configured - allowing request for local development")
            return public_key
        raise HTTPException(status_code=401, detail="Invalid signature")

    if not instance_config.is_known_peer(public_key):
        logger.warning(f"Internal API request from unknown peer: {public_key[:16]}...")
        if not instance_config.peer_public_keys:
            return public_key

    return public_key


@router.post("/sync/jobs")
async def sync_jobs_for_instance(
    request: Dict[str, str],
    job_repo: JobRepository = Depends(),
    sender_public_key: str = Depends(verify_internal_request)
):
    """Return pending jobs for the requesting instance."""
    instance_public_key = request.get("instance_public_key")
    if not instance_public_key:
        raise HTTPException(status_code=400, detail="instance_public_key required")

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
        } for job in jobs],
        "count": len(jobs)
    }


@router.get("/sync/otps")
async def sync_otps(
    otp_repo: OTPRepository = Depends(),
    sender_public_key: str = Depends(verify_internal_request)
):
    """Return unverified OTPs for peer sync (public_key only)."""
    logger.info(f"Internal sync: {sender_public_key[:16]}... requesting OTPs")

    all_otps = otp_repo.get_all_unverified_otps()

    # Only sync OTPs with public_key AND random_number
    otps = [otp for otp in all_otps if otp.get('public_key') and otp.get('random_number')]

    logger.info(f"Returning {len(otps)} unverified OTPs for sync")

    return {
        "otps": [_serialize_otp(otp) for otp in otps],
        "count": len(otps)
    }


def _serialize_otp(otp: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize OTP dict for JSON response."""
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
    """Receive OTP broadcast event from peer instance."""
    event_type = event.get("event_type")
    sender_id = event.get("instance_id")
    our_instance_id = instance_config.instance_public_key[:16]

    if sender_id == our_instance_id:
        logger.debug("Skipping OTP sync event from self")
        return {"status": "skipped", "reason": "self-event"}

    logger.info(f"Received OTP {event_type} event from {sender_id}")

    if event_type == "create":
        otp_data = event.get("otp_data", {})
        public_key = otp_data.get('public_key')

        if not public_key:
            return {"status": "skipped", "reason": "no-public-key"}

        random_number = otp_data.get('random_number')
        if not random_number:
            logger.warning(f"⚠️ Skipping OTP without random_number: public_key={public_key[:16]}...")
            return {"status": "skipped", "reason": "no-random-number"}

        # Deserialize datetime fields
        otp_data = _deserialize_datetime_fields(otp_data)

        # Upsert OTP
        existing = otp_repo.get_otp_by_public_key(public_key)
        if existing:
            # Preserve encrypted_secret_share if exists
            if existing.get('encrypted_secret_share') and 'encrypted_secret_share' not in otp_data:
                otp_data['encrypted_secret_share'] = existing['encrypted_secret_share']
            otp_repo.update_otp_by_public_key(public_key, otp_data)
            logger.info(f"Updated OTP for {public_key[:16]}... from peer broadcast")
        else:
            otp_repo.create_otp(otp_data)
            logger.info(f"Created OTP for {public_key[:16]}... from peer broadcast")

        return {"status": "created"}

    elif event_type == "verify":
        public_key = event.get("public_key")
        if not public_key:
            return {"status": "skipped", "reason": "no-public-key"}

        otp_repo.mark_otp_verified_by_public_key(public_key)
        logger.info(f"Marked OTP for {public_key[:16]}... as verified from peer broadcast")
        return {"status": "verified"}

    elif event_type == "delete":
        public_key = event.get("public_key")
        if not public_key:
            return {"status": "skipped", "reason": "no-public-key"}

        # Delete OTP by public_key
        otp_repo.delete_otp_by_public_key(public_key)
        logger.info(f"Deleted OTP for {public_key[:16]}... from peer broadcast")
        return {"status": "deleted"}

    else:
        raise HTTPException(status_code=400, detail=f"Invalid event_type: {event_type}")


@router.post("/jobs/sync")
async def sync_job_event(
    event: Dict[str, Any],
    sender_public_key: str = Depends(verify_internal_request)
):
    """
    Receive a job lifecycle event from a peer instance.

    Event types:
    - job_created: store a shadow copy of the job (never processed locally)
    - job_result : store the submission from the origin's result, delete shadow
    - job_failed : mark the shadow row failed
    """
    from app.services.replication_handlers import (
        handle_job_created_event, handle_job_result_event, handle_job_failed_event
    )

    event_type = event.get("event_type")
    sender_id = event.get("instance_id")
    our_instance_id = instance_config.instance_public_key[:16]

    if sender_id == our_instance_id:
        logger.debug("Skipping job sync event from self")
        return {"status": "skipped", "reason": "self-event"}

    logger.info(f"Received job {event_type} event from {sender_id}")

    if event_type == "job_created":
        return handle_job_created_event(event)
    elif event_type == "job_result":
        return handle_job_result_event(event)
    elif event_type == "job_failed":
        return handle_job_failed_event(event)
    else:
        raise HTTPException(status_code=400, detail=f"Invalid event_type: {event_type}")


@router.post("/jobs/status")
async def get_jobs_status(
    request: Dict[str, Any],
    job_repo: JobRepository = Depends(),
    sender_public_key: str = Depends(verify_internal_request)
):
    """
    Report the status of jobs to a peer instance running the recovery pull.

    Served by every instance regardless of LLM role - it only reads local
    tables. Per job: 'pending'/'processing'/'failed' from the jobs table,
    'completed' (with the reconstructed result payload) from
    document_submissions, or 'unknown' when the job was dropped.
    """
    from app.repositories.document_submission_repository import DocumentSubmissionRepository

    job_ids = request.get("job_ids")
    if not isinstance(job_ids, list) or not job_ids:
        raise HTTPException(status_code=400, detail="job_ids list required")

    logger.info(f"Internal job status request from {sender_public_key[:16]}... for {len(job_ids)} job(s)")

    submission_repo = DocumentSubmissionRepository()
    results = []

    for job_id in job_ids:
        job = job_repo.get_job_by_id(job_id)
        if job:
            results.append({"job_id": job_id, "state": job.status.value})
            continue

        submission = submission_repo.get_submission_by_job_id(job_id)
        if submission:
            # Reconstruct the result payload from the submission row. The
            # extracted_data_encrypted envelope is passed through verbatim -
            # ECIES envelopes decrypt identically for the client regardless
            # of which instance produced them.
            response_data = submission_repo._construct_response_data(submission)
            response_data["user_identity_id"] = submission.get("user_identity_id")
            if submission.get("verification_state") is not None and "verification_state" not in response_data:
                response_data["verification_state"] = submission.get("verification_state")
            if submission.get("sequence_no") is not None and "sequence_no" not in response_data:
                response_data["sequence_no"] = submission.get("sequence_no")

            request_data = submission.get("request_data")
            if isinstance(request_data, str):
                try:
                    request_data = json.loads(request_data)
                except (json.JSONDecodeError, TypeError):
                    request_data = {}

            results.append({
                "job_id": job_id,
                "state": "completed",
                "result": {
                    "response_data": response_data,
                    "request_data": request_data or {}
                }
            })
        else:
            results.append({"job_id": job_id, "state": "unknown"})

    return {"jobs": results, "count": len(results)}


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "instance": instance_config.instance_public_key[:16],
        "peers": len(instance_config.peer_instances)
    }


@router.get("/public-key")
async def get_public_key():
    """Return this instance's full public key."""
    return {
        "public_key": instance_config.instance_public_key,
        "instance_url": instance_config.instance_url
    }

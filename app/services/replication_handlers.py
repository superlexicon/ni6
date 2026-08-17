"""
Replicated Job Handlers

Receiving-side logic for peer job replication events, shared by the
/api/internal/jobs/sync endpoint and the startup shadow-job recovery pull.

Event flow (origin = instance that received the client document):
- job_created: store a shadow row in document_analysis_jobs (status 'pending',
  processing_server = origin URL). Never enqueued for local processing.
- job_result : create the document_submissions row from the origin's values,
  upsert the user's user_keys mapping, delete the shadow row. Idempotent.
- job_failed : mark the shadow row failed (kept for inspection).
"""

from typing import Any, Dict

from app.core.logger import get_logger

logger = get_logger()


def handle_job_created_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Store a shadow copy of a job created on a peer instance.

    The row is stored pending with processing_server set, marking it as owned
    by the peer - it is never enqueued for local processing.
    """
    job = event.get("job") or {}
    job_id = job.get("id")
    processing_server = event.get("processing_server")

    if not job_id or not processing_server:
        return {"status": "skipped", "reason": "missing job id or processing_server"}

    request_data = job.get("request_data") or {}
    if not isinstance(request_data, dict):
        return {"status": "skipped", "reason": "invalid request_data"}

    from app.repositories.job_repository import JobRepository
    job_repo = JobRepository()

    result = job_repo.create_replicated_job(
        job_id=job_id,
        request_data=request_data,
        processing_server=processing_server,
        client_public_key=job.get("client_public_key"),
        user_identity_id=job.get("user_identity_id"),
        callback_url=job.get("callback_url")
    )

    if result is False:
        return {"status": "error", "reason": "failed to store replicated job"}
    if isinstance(result, dict) and result.get("exists"):
        return {"status": "exists", "job_id": job_id}
    return {"status": "created", "job_id": job_id}


def handle_job_result_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Finalize a replicated job from the origin's result payload.

    Idempotent: if a submission already exists for the job, the shadow row is
    simply deleted. Tolerates a missing shadow row (result arriving without
    the job_created event, e.g. peer was down during job creation).
    """
    job_id = event.get("job_id")
    if not job_id:
        return {"status": "skipped", "reason": "missing job_id"}

    response_data = event.get("response_data") or {}
    request_data = event.get("request_data") or {}
    if not isinstance(response_data, dict) or not isinstance(request_data, dict):
        return {"status": "skipped", "reason": "invalid payload"}

    from app.repositories.job_repository import JobRepository
    from app.repositories.document_submission_repository import DocumentSubmissionRepository

    submission_repo = DocumentSubmissionRepository()
    job_repo = JobRepository()

    existing = submission_repo.get_submission_by_job_id(job_id)
    if existing:
        job_repo.delete_job(job_id)
        return {"status": "exists", "job_id": job_id}

    success, error = submission_repo.create_submission_from_peer(
        response_data=response_data,
        request_data=request_data,
        job_id=job_id
    )
    if not success:
        logger.error(f"Failed to store replicated submission for job {job_id}: {error}")
        return {"status": "error", "reason": error, "job_id": job_id}

    _upsert_user_key_info(event.get("user_key_info") or {}, response_data, request_data)

    job_repo.delete_job(job_id)
    logger.info(f"Finalized replicated job {job_id}: submission stored, shadow row deleted")
    return {"status": "created", "job_id": job_id}


def handle_job_failed_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """Mark a shadow row failed; tolerate unknown jobs (no shadow was stored)."""
    job_id = event.get("job_id")
    if not job_id:
        return {"status": "skipped", "reason": "missing job_id"}

    error_message = event.get("error_message") or "Job failed on processing server"

    from app.repositories.job_repository import JobRepository
    job_repo = JobRepository()
    job_repo.mark_job_failed_replica(job_id, error_message)
    return {"status": "failed", "job_id": job_id}


def _is_secret_share_recovery(request_data: Dict[str, Any]) -> bool:
    """True if the request is a key-recovery submission (ephemeral temp key)."""
    files = (request_data or {}).get("files") or []
    for f in files:
        if not isinstance(f, dict):
            continue
        doc_type = (f.get("document_type") or f.get("file_type") or "").lower().strip()
        if doc_type == "secret_share_recovery":
            return True
    return False


def _upsert_user_key_info(
    user_key_info: Dict[str, Any],
    response_data: Dict[str, Any],
    request_data: Dict[str, Any]
) -> None:
    """
    Sync the user's key -> identity/state mapping so this instance can serve
    /api/jobs/verification for the replicated document submission.

    Values come from the origin (it owns the state machine); nothing is
    derived from local tables.

    Key-recovery invariants:
    - Secret-share recovery submits under an EPHEMERAL temp public key - its
      results must never create user_keys rows.
    - user_keys rows are never created share-less: each instance's secret
      share lives in its own user_keys_pending row (registered directly by
      the client), so a missing user_keys row is completed by migrating the
      LOCAL pending row (which carries this instance's share), never by
      inserting an empty one. Share-less rows satisfy recovery lookups but
      cannot produce shares, breaking the 2-of-3 quorum.
    """
    client_public_key = (
        user_key_info.get("client_public_key")
        or request_data.get("client_public_key")
    )
    if not client_public_key:
        return

    if _is_secret_share_recovery(request_data):
        logger.debug("Skipping user_keys sync for secret_share_recovery result (temp key)")
        return

    user_identity_id = (
        user_key_info.get("user_identity_id")
        or response_data.get("user_identity_id")
    )
    verification_state = user_key_info.get("verification_state", response_data.get("verification_state"))
    sequence_no = user_key_info.get("sequence_no", response_data.get("sequence_no"))

    from app.repositories.user_key_repository import UserKeyRepository
    user_key_repo = UserKeyRepository()

    existing = user_key_repo.get_key_by_public_key(client_public_key)
    row_present = bool(existing)
    if existing:
        if user_identity_id and user_identity_id != existing.get("user_identity_id"):
            user_key_repo.update_key_by_public_key(
                client_public_key, {"user_identity_id": user_identity_id}
            )
    elif user_identity_id:
        # Complete the LOCAL registration migration instead of creating an
        # empty row: this instance's pending key holds its secret share.
        from app.repositories.user_keys_pending_repository import UserKeysPendingRepository
        moved = UserKeysPendingRepository().move_pending_to_user_keys(
            client_public_key, user_identity_id
        )
        if moved:
            row_present = True
            logger.info(
                f"Migrated local pending key (with secret share) for "
                f"{client_public_key[:16]}... from replicated result"
            )
        else:
            logger.debug(
                f"No local pending key for {client_public_key[:16]}... - "
                f"no user_keys row created (row creation belongs to this "
                f"instance's own registration flow)"
            )

    # Sync per-device state (valid range 0-3 enforced by the repository)
    if row_present and isinstance(verification_state, int) and isinstance(sequence_no, int) and \
            0 <= verification_state <= 3 and 0 <= sequence_no <= 3:
        user_key_repo.update_state_and_sequence(client_public_key, verification_state, sequence_no)

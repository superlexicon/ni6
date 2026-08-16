import os
import uuid
import hashlib
import json
import asyncio
from typing import Optional, List, Union
import threading
import time
from app.dto.job_models import (
    JobRequest, JobSubmissionResponse, JobStatus, JobDatabaseRecord,
    JobStatusResponse, JobInfo
)
from app.repositories.job_repository import JobRepository
from app.core.job_queue import JobQueue
from app.services.job_timing_service import get_job_timing_service
from app.services.job_broadcast_service import job_broadcast_service
from app.config.instance_config import instance_config
from app.config.llm_config import is_llm_server_configured
from app.core.logger import get_logger


class JobManager:
    """
    Manages job creation, persistence, and queue operations.

    Uses JobQueue for in-memory job storage.
    MySQL is maintained for audit/history and job status queries.
    """

    def __init__(self, db_connection, job_queue: Union[JobQueue, object], instance_public_key: str):
        """
        Initialize JobManager with job queue.

        Args:
            db_connection: MySQL database connection (optional, will create own if None)
            job_queue: Job queue for job storage (JobQueue or any queue-like object)
            instance_public_key: Server public key for this instance
        """
        if db_connection is None:
            from app.core.db.database import get_db_connection
            db_connection = get_db_connection()
        self.db = db_connection
        self.job_queue = job_queue
        self.instance_public_key = instance_public_key
        # JobRepository creates its own connection for operations
        self.job_repo = JobRepository()
        self.logger = get_logger()
        self.default_callback_url = os.getenv("CALLBACK_URL")
        self.max_job_retries = int(os.getenv("MAX_JOB_RETRIES", "3"))
        self._worker = None  # Reference to worker (if needed for compatibility)
        self.timing_service = get_job_timing_service()  # Initialize job timing service

    def set_worker(self, worker) -> None:
        """Set reference to worker (for compatibility with existing code)."""
        self._worker = worker

    def _extract_document_type(self, request_data: dict) -> Optional[str]:
        """
        Extract document_type from request data.

        Args:
            request_data: Request data containing files

        Returns:
            Document type string or None if not found
        """
        files = request_data.get("files", [])
        if files and len(files) > 0:
            file_obj = files[0]
            return file_obj.get("document_type") or file_obj.get("file_type")
        return None

    def _calculate_expected_completion_time(
        self,
        routing_key: str,
        request_data: dict
    ) -> Optional[float]:
        """
        Calculate expected completion time for a job.

        Takes into account:
        1. Jobs already in the queue ahead of this one
        2. Average processing time per job type

        Args:
            routing_key: The instance public key for routing
            request_data: Request data containing files

        Returns:
            Expected time in seconds, or None if no historical data available
        """
        try:
            # Get document type for current job
            document_type = self._extract_document_type(request_data)
            if not document_type:
                return None

            # Normalize document type
            document_type = document_type.lower()

            # Get pending jobs ahead in queue (before this job was added)
            # Note: The job was just added to the in-memory queue, so we need to get all tracked
            # jobs and calculate the time for jobs ahead of this one
            all_jobs = self.job_queue.get_all_tracked_jobs()

            # Filter jobs for this routing key and exclude current job
            jobs_ahead = []
            for job_id, job_record in all_jobs.items():
                if job_id != routing_key:  # Exclude current job
                    jobs_ahead.append(job_record)

            # Get time for jobs ahead in queue
            queue_time = self.timing_service.get_queue_position_time(jobs_ahead)

            # Get average time for current job type
            job_type_avg = self.timing_service.get_average_time(document_type)

            # Calculate total expected time
            if queue_time is not None and job_type_avg is not None:
                total_time = queue_time + job_type_avg
                self.logger.debug(
                    f"Expected completion time for {document_type}: {total_time:.2f}s "
                    f"(queue: {queue_time:.2f}s + job: {job_type_avg:.2f}s, "
                    f"{len(jobs_ahead)} jobs ahead)"
                )
                return total_time
            elif job_type_avg is not None:
                # Only have average for current job type (empty queue)
                self.logger.debug(
                    f"Expected completion time for {document_type}: {job_type_avg:.2f}s "
                    f"(empty queue)"
                )
                return job_type_avg
            else:
                self.logger.debug(
                    f"No timing data available for {document_type}"
                )
                return None

        except Exception as e:
            self.logger.error(f"Error calculating expected completion time: {e}")
            return None

    def _get_user_identity_id_from_public_key(self, client_public_key: str) -> Optional[str]:
        """
        Look up user_identity_id from user_keys table using client_public_key.

        Args:
            client_public_key: Client's public key

        Returns:
            user_identity_id if found, None otherwise
        """
        try:
            from app.repositories.user_key_repository import UserKeyRepository
            user_key_repo = UserKeyRepository()
            user_key = user_key_repo.get_key_by_public_key(client_public_key)
            if user_key:
                return user_key.get('user_identity_id')
            return None
        except Exception as e:
            self.logger.error(f"Error looking up user_identity_id for {client_public_key[:16]}...: {e}")
            return None

    async def _hash_content_async(self, content: dict) -> str:
        """
        Hash content asynchronously to avoid blocking event loop.

        This is important for large payloads (e.g., 466KB encrypted data)
        where synchronous JSON serialization + SHA256 hashing can block
        the event loop and cause issues when multiple requests arrive.
        """
        loop = asyncio.get_running_loop()
        # Serialize JSON synchronously (fast)
        content_json = json.dumps(content, sort_keys=True)
        # Hash in executor to avoid blocking
        return await loop.run_in_executor(
            None,
            lambda: hashlib.sha256(content_json.encode()).hexdigest()
        )

    # Document types whose extraction runs on the vision LLM server.
    # Everything else (selfie/video liveness, key operations, DocTR-only
    # types) is processed with local models. Mirrors the worker's routing.
    NON_LLM_DOCUMENT_TYPES = {
        'selfie', 'video_selfie', 'secret_share_recovery', 'tax_statement',
        'tax_return', 'resume', 'driving_license', 'national_id',
        'add_public_key', 'remove_public_key',
    }

    def _request_requires_vision_llm(self, job_request: JobRequest) -> bool:
        """True if any file in the request needs the vision LLM for extraction."""
        if not job_request.files:
            return False
        for f in job_request.files:
            file_type = (f.file_type or '').lower().strip()
            doc_type = (f.document_type or file_type or 'auto').lower().strip().replace(' ', '_')
            if file_type == 'selfie' or doc_type in self.NON_LLM_DOCUMENT_TYPES:
                continue
            return True
        return False

    async def create_job(self, job_request: JobRequest, skip_state_validation: bool = False) -> JobSubmissionResponse:
        """
        Create a new job and add it to the queue.

        Args:
            job_request: Job request with encrypted envelope and file information
            skip_state_validation: If True, skip state validation (for signed endpoint only)
        """
        # Shadow-only instances (no LLM server configured) never process
        # LLM-dependent documents. Such submissions are accepted and silently
        # dropped instead of erroring: an error response leaves clients
        # waiting on a job that will never exist, and no row is persisted.
        if not is_llm_server_configured():
            if self._request_requires_vision_llm(job_request):
                drop_job_id = str(uuid.uuid4())
                self.logger.warning(
                    f"Silently dropping LLM-dependent job {drop_job_id} "
                    f"(no LLM server configured on this instance)"
                )
                return JobSubmissionResponse(
                    success=True,
                    job_id=drop_job_id,
                    status=JobStatus.PENDING,
                    message="Job queued successfully"
                )

            self.logger.warning(
                "Rejecting non-LLM job: no LLM server configured on this instance"
            )
            return JobSubmissionResponse(
                success=False,
                job_id="",
                status=JobStatus.FAILED,
                message=("This instance has no LLM server configured and cannot "
                         "process documents. Submit to the LLM-enabled instance.")
            )

        # Check if this is a secret share recovery job (allows empty iv)
        is_secret_share_recovery = False
        if job_request.files:
            is_secret_share_recovery = any(
                f.file_type == "secret_share_recovery" or f.document_type == "secret_share_recovery"
                for f in job_request.files
            )

        # Check if this is an encrypted envelope request
        is_encrypted_envelope = bool(job_request.encrypted_payload)

        # Basic validation only - worker handles detailed validation
        # Secret share recovery jobs and encrypted envelope requests don't need iv
        if not job_request.client_public_key:
            return JobSubmissionResponse(
                success=False,
                job_id="",
                status=JobStatus.FAILED,
                message="Required field: client_public_key"
            )

        if not is_secret_share_recovery and not is_encrypted_envelope and not job_request.iv:
            return JobSubmissionResponse(
                success=False,
                job_id="",
                status=JobStatus.FAILED,
                message="Required fields: client_public_key, iv"
            )

        # Validate encrypted envelope has all required fields
        if is_encrypted_envelope:
            if not all([job_request.encrypted_key, job_request.key_iv, job_request.payload_iv]):
                return JobSubmissionResponse(
                    success=False,
                    job_id="",
                    status=JobStatus.FAILED,
                    message="Encrypted envelope requires: encrypted_key, key_iv, encrypted_payload, payload_iv"
                )

        # Convert FileObject list to dict format for JSON serialization
        # Must do this BEFORE validation loop since we need dicts, not Pydantic models
        files_dict = []
        if job_request.files:
            files_dict = [file.model_dump() for file in job_request.files]

        # NEW: Validate state BEFORE queuing (for non-selfie documents)
        # This prevents clients from submitting garbage data for public keys
        # that haven't completed previous verification steps
        if not skip_state_validation and job_request.files:
            import re
            from app.services.verification_state_service import VerificationStateService

            state_service = VerificationStateService()

            for file_obj in files_dict:
                file_type = file_obj.get('file_type')
                doc_type = file_obj.get('document_type')

                # Secret share recovery: skip state validation (handled by worker)
                # This operation matches the submitted selfie against the stored one
                if file_type == 'secret_share_recovery' or doc_type == 'secret_share_recovery':
                    continue

                # Selfie validation: check filename contains "otpXXXXXX" pattern
                # This adds a minimal validation to prevent garbage selfie submissions
                if file_type == 'selfie':
                    filename = file_obj.get('filename', '')
                    # Check if filename contains "otp" followed by 6 digits (case-insensitive)
                    if not re.search(r'otp\d{6}', filename, re.IGNORECASE):
                        self.logger.warning(
                            f"Selfie filename validation failed: {filename} does not contain otpXXXXXX pattern"
                        )
                        return JobSubmissionResponse(
                            success=False,
                            job_id="",
                            status=JobStatus.FAILED,
                            message="Selfie filename must contain otpXXXXXX pattern (e.g., selfie_otp123456.jpg)"
                        )
                    continue

                # Validate state for all other document types
                is_valid, error_msg, is_resubmission = state_service.validate_document_submission(
                    job_request.client_public_key,
                    doc_type
                )

                if not is_valid:
                    self.logger.warning(
                        f"State validation failed for {doc_type}: {error_msg}"
                    )
                    return JobSubmissionResponse(
                        success=False,
                        job_id="",
                        status=JobStatus.FAILED,
                        message=f"State validation failed: {error_msg}"
                    )

        # Prepare request data for storage - ensure no None values
        request_data = {
            "encrypted_archive": job_request.encrypted_archive or "",
            "client_public_key": job_request.client_public_key,
            "iv": job_request.iv or "",
            "files": files_dict,
            "secret_share": job_request.secret_share,
            "mobile_number": job_request.mobile_number,
            "country_code": job_request.country_code,
            "temp_public_key": job_request.temp_public_key,
            "otp_code": job_request.otp_code,
            # Encrypted envelope fields
            "encrypted_key": job_request.encrypted_key,
            "key_iv": job_request.key_iv,
            "encrypted_payload": job_request.encrypted_payload,
            "payload_iv": job_request.payload_iv,
            # Target server for routing (if provided)
            "target_server_public_key": job_request.target_server_public_key,
            # API URL filter for secret share recovery
            "api_url": job_request.api_url
        }

        # Debug logging for encrypted envelope
        if is_encrypted_envelope:
            self.logger.info(f"📝 [JOB_MANAGER_DEBUG] Encrypted envelope request detected")
            self.logger.info(f"📝 [JOB_MANAGER_DEBUG] Payload size: {len(job_request.encrypted_payload)} chars")

        # Debug logging to verify request-level fields for selfie submissions
        has_selfie = any(file.get('file_type') == 'selfie' for file in request_data.get("files", []))
        if has_selfie:
            self.logger.info(f"📝 [JOB_MANAGER_DEBUG] Selfie request detected")
            self.logger.info(f"📝 [JOB_MANAGER_DEBUG] Request-level secret_share present: {'Yes' if request_data.get('secret_share') else 'No'}")

        # Use provided callback URL or default from environment
        callback_url = job_request.callback_url or self.default_callback_url

        # Look up user_identity_id FIRST (before creating job)
        # This ensures atomic job creation with the correct user_identity_id,
        # avoiding race conditions where verification endpoint reads job before update
        user_identity_id = self._get_user_identity_id_from_public_key(
            job_request.client_public_key
        )
        if user_identity_id:
            self.logger.debug(f"Found user_identity_id: {user_identity_id[:16]}...")

        # Generate unique job ID using standard UUID format
        # This ensures unique job IDs per request, preventing collisions when multiple
        # instances receive identical encrypted content
        job_id = str(uuid.uuid4())

        # Create job in database first (for audit/history)
        # Include user_identity_id atomically in INSERT to avoid race condition
        try:
            result = self.job_repo.create_job(
                job_id=job_id,
                request_data=request_data,
                callback_url=callback_url,
                max_retries=self.max_job_retries,
                user_identity_id=user_identity_id
            )

            # Handle idempotency - job already exists
            if isinstance(result, dict) and result.get("exists"):
                return JobSubmissionResponse(
                    success=True,
                    job_id=job_id,
                    status=JobStatus[result["status"]],
                    message=f"Job already {result['status']}"
                )

            # Handle failure
            if not result:
                return JobSubmissionResponse(
                    success=False,
                    job_id="",
                    status=JobStatus.FAILED,
                    message="Failed to create job in database"
                )
        except Exception as e:
            self.logger.error(f"Database error creating job {job_id}: {str(e)}")
            return JobSubmissionResponse(
                success=False,
                job_id="",
                status=JobStatus.FAILED,
                message=f"Database error: {str(e)}"
            )

        # Note: user_identity_id is now included atomically in the INSERT above
        # No separate update needed - race condition fixed

        # Create JobDatabaseRecord for in-memory queue processing
        # Jobs are routed via HTTP-based P2P communication between instances

        # target_server_public_key is REQUIRED for all requests (multi-instance routing)
        if not job_request.target_server_public_key:
            return JobSubmissionResponse(
                success=False,
                job_id="",
                status=JobStatus.FAILED,
                message="target_server_public_key is required for routing"
            )

        routing_key = job_request.target_server_public_key

        # Debug logging to verify routing
        self.logger.info(f"🎯 [ROUTING_DEBUG] Using target_server_public_key: {routing_key[:16]}...")
        self.logger.info(f"🎯 [ROUTING_DEBUG] Receiving instance key: {self.instance_public_key[:16]}...")

        # Create JobDatabaseRecord for the job
        from datetime import datetime
        from app.dto.job_models import JobDatabaseRecord

        job_record = JobDatabaseRecord(
            id=job_id,
            status=JobStatus.PENDING,
            request_data=request_data,
            callback_url=callback_url,
            max_retries=self.max_job_retries,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )

        # Add to in-memory queue
        self.job_queue.put(job_record)
        self.logger.info(f"Successfully created job {job_id} in in-memory queue")

        # Signal worker that a job is available
        if self._worker:
            self._worker.signal_job_available()

        # Replicate a shadow copy of the job record to peers (fire-and-forget).
        # Peers store it unprocessed and finalize it when the result is pushed.
        try:
            job_broadcast_service.broadcast_job_created(
                job_id=job_id,
                request_data_stripped=self.job_repo._strip_large_fields(request_data),
                client_public_key=request_data.get('client_public_key'),
                user_identity_id=user_identity_id,
                callback_url=callback_url
            )
        except Exception as e:
            self.logger.warning(f"Failed to broadcast job_created for {job_id}: {e}")

        # Calculate expected completion time based on queue position
        expected_time = self._calculate_expected_completion_time(
            routing_key, request_data
        )

        return JobSubmissionResponse(
            success=True,
            job_id=job_id,
            status=JobStatus.PENDING,
            message="Job queued successfully",
            expected_completion_time_seconds=expected_time
        )

    def get_job_status(self, job_id: str) -> Optional[JobStatusResponse]:
        """Get current job status and results if available"""
        job_record = self.job_repo.get_job_by_id(job_id)
        if not job_record:
            return None

        # Build job info
        job_info = JobInfo(
            job_id=job_record.id,
            status=job_record.status,
            created_at=job_record.created_at,
            updated_at=job_record.updated_at,
            started_at=job_record.started_at,
            completed_at=job_record.completed_at,
            callback_attempted_at=job_record.callback_attempted_at,
            retry_count=job_record.retry_count,
            max_retries=job_record.max_retries,
            error_message=job_record.error_message,
            callback_url=job_record.callback_url,
            request_data=job_record.request_data
        )

        # Determine if callback was sent
        callback_sent = job_record.callback_attempted_at is not None

        return JobStatusResponse(
            job_info=job_info,
            results=job_record.response_data,
            callback_sent=callback_sent
        )

    def get_job_by_public_key(self, public_key: str) -> Optional[JobStatusResponse]:
        """Get job status by client public key"""
        job_record = self.job_repo.get_job_by_public_key(public_key)
        if not job_record:
            return None

        # Build job info
        job_info = JobInfo(
            job_id=job_record.id,
            status=job_record.status,
            created_at=job_record.created_at,
            updated_at=job_record.updated_at,
            started_at=job_record.started_at,
            completed_at=job_record.completed_at,
            callback_attempted_at=job_record.callback_attempted_at,
            retry_count=job_record.retry_count,
            max_retries=job_record.max_retries,
            error_message=job_record.error_message,
            callback_url=job_record.callback_url,
            request_data=job_record.request_data
        )

        # Determine if callback was sent
        callback_sent = job_record.callback_attempted_at is not None

        return JobStatusResponse(
            job_info=job_info,
            results=job_record.response_data,
            callback_sent=callback_sent
        )

    def get_in_progress_jobs_by_user_identity_id(
        self, user_identity_id: str, limit: int = 10
    ) -> List[JobStatusResponse]:
        """Get in-progress jobs by user identity ID."""
        job_records = self.job_repo.get_in_progress_jobs_by_user_identity_id(
            user_identity_id, limit
        )
        return [
            JobStatusResponse(
                job_info=JobInfo(
                    job_id=job.id,
                    status=job.status,
                    created_at=job.created_at,
                    updated_at=job.updated_at,
                    started_at=job.started_at,
                    completed_at=job.completed_at,
                    callback_attempted_at=job.callback_attempted_at,
                    retry_count=job.retry_count,
                    max_retries=job.max_retries,
                    error_message=job.error_message,
                    callback_url=job.callback_url,
                    request_data=job.request_data
                ),
                results=job.response_data,
                callback_sent=job.callback_attempted_at is not None
            )
            for job in job_records
        ]

    def update_job_status(self, job_id: str, status: JobStatus,
                         error_message: Optional[str] = None,
                         response_data: Optional[dict] = None) -> bool:
        """Update job status in database"""
        success = self.job_repo.update_job_status(
            job_id=job_id,
            status=status,
            error_message=error_message,
            response_data=response_data
        )

        if not success:
            self.logger.error(f"Failed to update job {job_id} status to {status.value}")

        return success

    def increment_job_retry(self, job_id: str) -> bool:
        """Increment retry count for a failed job"""
        return self.job_repo.increment_job_retry(job_id)

    def mark_callback_attempted(self, job_id: str) -> bool:
        """Mark that callback was attempted for a job"""
        return self.job_repo.mark_callback_attempted(job_id)

    def get_next_job(self, timeout: float = 1.0) -> Optional[JobDatabaseRecord]:
        """
        Get the next job from the in-memory queue.

        Args:
            timeout: Maximum time to wait for a job (in seconds)

        Returns:
            JobDatabaseRecord if a job is available, None otherwise
        """
        try:
            job_id = self.job_queue.get(timeout=timeout)
            if job_id:
                return self.job_queue.get_job_data(job_id)
            return None
        except Exception as e:
            self.logger.error(f"Error getting next job from queue: {e}")
            return None

    def mark_job_completed(self, job_id: str, response_data: dict) -> bool:
        """
        Mark job as completed in MySQL.
        """
        success = self.update_job_status(
            job_id=job_id,
            status=JobStatus.COMPLETED,
            response_data=response_data
        )

        if success:
            self.logger.info(f"Job {job_id} marked as completed in MySQL")
        else:
            self.logger.error(f"Failed to mark job {job_id} as completed")

        return success

    def mark_job_failed(self, job_id: str, error_message: str) -> bool:
        """
        Mark job as failed in MySQL and manage retry logic.

        For retries, the job is re-added to the in-memory queue.
        On max retries, job stays in queue for manual inspection.
        """
        job_record = self.job_repo.get_job_by_id(job_id)
        if not job_record:
            self.logger.error(f"Job {job_id} not found for failure marking")
            return False

        # Check if we should retry
        if job_record.retry_count < job_record.max_retries:
            self.logger.info(f"Retrying job {job_id} (attempt {job_record.retry_count + 1}/{job_record.max_retries})")
            self.increment_job_retry(job_id)
            self.update_job_status(job_id, JobStatus.PENDING, f"Retry {job_record.retry_count + 1}: {error_message}")

            # Re-add to in-memory queue for retry
            # Use target_server_public_key from request_data
            routing_key = job_record.request_data.get("target_server_public_key")
            if not routing_key:
                self.logger.error(f"Job {job_id} missing target_server_public_key, cannot retry")
                return self.update_job_status(job_id, JobStatus.FAILED, "Missing target_server_public_key for retry")

            # Update job record status to pending and re-add to queue
            job_record.status = JobStatus.PENDING
            job_record.updated_at = datetime.utcnow()
            self.job_queue.put(job_record)

            # Signal worker that a retry job is available
            if self._worker:
                self._worker.signal_job_available()

            return True
        else:
            # Max retries reached, mark as failed
            self.logger.error(f"Job {job_id} failed after {job_record.retry_count} retries: {error_message}")
            result = self.update_job_status(job_id, JobStatus.FAILED, error_message)

            # Notify peers so they mark their shadow rows failed
            try:
                job_broadcast_service.broadcast_job_failed(
                    job_id, error_message or "Job failed after max retries"
                )
            except Exception as e:
                self.logger.warning(f"Failed to broadcast job_failed for {job_id}: {e}")

            return result

    def load_pending_jobs_on_startup(self) -> int:
        """
        Load pending jobs from MySQL into in-memory queue on startup.

        Role-aware with peer job replication:
        - Instances with an LLM server enqueue their own pending jobs
          (processing_server NULL). Replicated shadow rows (processing_server
          set) are never enqueued - they are owned by the peer that received
          the client request.
        - Shadow-only instances (no LLM server) enqueue nothing. Leftover own
          pending rows are marked failed: this instance can no longer process
          them, and their stored payloads are stripped so no other instance
          could either. Clients must resubmit.
        - On all instances, shadow rows older than the TTL are marked failed
          (origin crashed before delivering the result and is unreachable).
        """
        self.logger.info("Loading pending jobs from database into in-memory queue...")

        # Safety net: fail stale replicated rows past the TTL
        self.job_repo.fail_stale_replicated_jobs(instance_config.shadow_job_ttl_hours)

        # First, reset any stale jobs that were processing when the application stopped
        stale_jobs_count = self.job_repo.reset_stale_jobs()
        if stale_jobs_count > 0:
            self.logger.info(f"Reset {stale_jobs_count} stale jobs to pending")

        # Load pending jobs and re-insert into in-memory queue
        pending_jobs = self.job_repo.get_pending_jobs(limit=1000)
        loaded_count = 0

        if not is_llm_server_configured():
            # Shadow-only instance: mark leftover own jobs failed (resubmit needed)
            for job in pending_jobs:
                if job.processing_server:
                    continue  # shadow row - resolved by the recovery pull
                self.logger.warning(
                    f"Failing own pending job {job.id}: LLM server no longer configured"
                )
                self.update_job_status(
                    job.id,
                    JobStatus.FAILED,
                    "LLM server no longer configured on this instance - resubmit to the LLM-enabled instance"
                )
                try:
                    job_broadcast_service.broadcast_job_failed(
                        job.id, "LLM server no longer configured on the processing instance - resubmit"
                    )
                except Exception:
                    pass
            self.logger.info("Shadow-only instance (no LLM server): no jobs enqueued for local processing")
            return 0

        for job in pending_jobs:
            # Check if job is still pending and hasn't exceeded retry limit
            if job.status == JobStatus.PENDING and job.retry_count < job.max_retries:
                # Never enqueue replicated shadow rows - they are owned by a peer
                if job.processing_server:
                    continue

                # Check if job is already being tracked in the queue
                if not self.job_queue.is_job_tracked(job.id):
                    # Convert to JobDatabaseRecord and add to queue
                    # Skip jobs without target_server_public_key (cannot route)
                    routing_key = job.request_data.get("target_server_public_key")
                    if not routing_key:
                        self.logger.warning(f"Skipping job {job.id} - missing target_server_public_key")
                        continue

                    from datetime import datetime
                    from app.dto.job_models import JobDatabaseRecord

                    job_record = JobDatabaseRecord(
                        id=job.id,
                        status=job.status,
                        request_data=job.request_data,
                        callback_url=job.callback_url,
                        max_retries=job.max_retries,
                        created_at=job.created_at,
                        updated_at=datetime.utcnow()
                    )
                    self.job_queue.put(job_record)
                    loaded_count += 1

        self.logger.info(f"Loaded {loaded_count} pending jobs into in-memory queue")

        # Signal worker that jobs are available
        if loaded_count > 0 and self._worker:
            self._worker.signal_job_available()

        return loaded_count

    def delete_job(self, job_id: str) -> bool:
        """Delete job from both MySQL and in-memory queue."""
        # Delete from MySQL
        mysql_success = self.job_repo.delete_job(job_id)
        # Remove from in-memory queue tracking
        self.job_queue.remove_job(job_id)
        return mysql_success

    def delete_completed_job(self, job_id: str) -> bool:
        """Delete a completed job from database (MySQL only, in-memory queue should not have it)."""
        return self.job_repo.delete_completed_job(job_id)

    def get_queue_stats(self) -> dict:
        """Get job statistics from in-memory queue."""
        # Return stats from in-memory queue
        queue_stats = self.job_queue.get_stats()
        pending_count = len(self.job_repo.get_pending_jobs(limit=1000))
        return {
            "pending_jobs_in_queue": queue_stats.get("queue_size", 0),
            "tracked_jobs": queue_stats.get("tracked_jobs", 0),
            "pending_jobs_in_mysql": pending_count,
            "processed_count": queue_stats.get("processed_count", 0),
            "failed_count": queue_stats.get("failed_count", 0)
        }

    def cleanup_old_completed_jobs(self, days_old: int = 7) -> int:
        """Clean up completed jobs older than specified days."""
        # This would be implemented to delete old completed jobs
        # For now, just log that cleanup was requested
        self.logger.info(f"Cleanup of jobs older than {days_old} days requested")
        return 0
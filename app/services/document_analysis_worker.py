import asyncio
import threading
import time
import json
import aiohttp
import requests
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from app.dto.job_models import JobDatabaseRecord, JobStatus
from app.services.job_manager import JobManager
from app.core.logger import get_logger
from app.services.pdf_analysis_service import PDFAnalysisService
from app.services.image_validation_service import ImageValidationService
from app.services.ela_service import ELAService
from app.helper.exif_validator import ExifValidator
from app.repositories.otp_repository import OTPRepository
from app.repositories.user_identity_repository import UserIdentityRepository
from app.repositories.document_submission_repository import DocumentSubmissionRepository
from app.services.comprehensive_photoholmes_service import ComprehensivePhotoHolmesService
from app.services.job_timing_service import get_job_timing_service
from app.services.detailed_analysis_service import DetailedAnalysisService
import os
import base64

from app.core.key.hybrid_crypto import HybridCrypto, HybridCryptoError


class ValidationException(Exception):
    """Exception for validation errors that should not trigger retries"""
    pass


def normalize_document_type(document_type: str) -> str:
    """
    Normalize document type to handle naming variations.

    Args:
        document_type: Raw document type string

    Returns:
        Normalized document type string
    """
    document_type_mapping = {
        "bankstatement": "bank_statement",
        # Add other mappings as needed for future consistency
    }

    normalized = document_type_mapping.get(document_type.lower(), document_type.lower())
    return normalized


class DocumentAnalysisWorker:
    """Background worker for processing document analysis jobs"""

    def __init__(self, job_manager: JobManager, db_connection):
        self.job_manager = job_manager
        # Create own connection if None provided
        if db_connection is None:
            from app.core.db.database import get_db_connection
            db_connection = get_db_connection()
        self.db_connection = db_connection
        self.logger = get_logger()
        self.running = False
        self.worker_thread = None
        self.job_timeout = int(os.getenv("JOB_TIMEOUT_SECONDS", "300"))  # 5 minutes default
        self.callback_timeout = int(os.getenv("CALLBACK_TIMEOUT_SECONDS", "30"))  # 30 seconds default

        # Event loop for async operations (persistent per worker thread)
        self._loop = None

        # Reactive signaling for job availability
        self._job_available = threading.Event()

        # Initialize dependencies for sequential services
        self._initialize_dependencies()

    def _initialize_dependencies(self):
        """Initialize all required dependencies for sequential services"""
        try:
            # Initialize repositories (each gets its own connection from pool)
            self.otp_repository = OTPRepository()
            self.user_identity_repository = UserIdentityRepository()
            self.document_submission_repository = DocumentSubmissionRepository()
            from app.repositories.user_key_repository import UserKeyRepository
            self.user_key_repository = UserKeyRepository()

            # Initialize validators
            self.exif_validator = ExifValidator()

            # Initialize services
            self.pdf_analysis_service = PDFAnalysisService()
            self.image_validation_service = ImageValidationService()
            self.ela_service = ELAService()

            # Use the shared comprehensive PhotoHolmes service (singleton)
            from app.services import comprehensive_photoholmes_service
            self.comprehensive_photoholmes_service = comprehensive_photoholmes_service

            # Initialize detailed analysis service
            self.detailed_analysis_service = DetailedAnalysisService()

            # Initialize hybrid crypto for payload decryption
            self.hybrid_crypto = HybridCrypto()

            # Initialize job timing service for queue time estimation
            self.timing_service = get_job_timing_service()

            # Initialize sequential processing services (state derived from user_identity_index)
            from app.services.sequential_selfie_service import SequentialSelfieService
            from app.services.sequential_passport_service import SequentialPassportService
            from app.services.sequential_id_card_service import SequentialIDCardService
            from app.services.sequential_bank_statement_service import SequentialBankStatementService
            from app.services.sequential_tax_statement_service import SequentialTaxStatementService
            from app.services.sequential_resume_service import SequentialResumeService
            from app.services.key_management_service import KeyManagementService

            self.sequential_selfie_service = SequentialSelfieService(
                user_key_repository=self.user_key_repository,
                otp_repository=self.otp_repository
            )
            self.sequential_passport_service = SequentialPassportService()
            self.sequential_id_card_service = SequentialIDCardService()
            self.sequential_bank_statement_service = SequentialBankStatementService()
            self.sequential_tax_statement_service = SequentialTaxStatementService()
            self.sequential_resume_service = SequentialResumeService()
            self.key_management_service = KeyManagementService()

            # Initialize video_selfie_service (optional - requires mediapipe)
            self.video_selfie_service = None
            try:
                from app.services.video_selfie_service import VideoSelfieService
                self.video_selfie_service = VideoSelfieService(
                    otp_repository=self.otp_repository,
                    user_key_repository=self.user_key_repository,
                )
                self.logger.info("VideoSelfieService initialized successfully")
            except ImportError as e:
                self.logger.warning(f"VideoSelfieService not available (mediapipe not installed): {e}")
            except Exception as e:
                self.logger.warning(f"VideoSelfieService initialization failed: {e}")

            self.logger.info("DocumentAnalysisWorker dependencies initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize DocumentAnalysisWorker dependencies: {e}")
            raise

    def _get_event_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create the worker's dedicated event loop."""
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        return self._loop

    def signal_job_available(self) -> None:
        """Signal that a new job is available (called by job_manager)."""
        self._job_available.set()

    def start(self) -> None:
        """Start the background worker"""
        if self.running:
            self.logger.warning("Worker is already running")
            return

        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        self.logger.info("Document analysis worker started")

    def stop(self, timeout: int = 30) -> None:
        """Stop the background worker"""
        if not self.running:
            return

        self.logger.info("Stopping document analysis worker...")
        self.running = False

        # Signal the worker to wake up and check running flag
        self._job_available.set()

        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=timeout)
            if self.worker_thread.is_alive():
                self.logger.warning("Worker thread did not stop gracefully within timeout")
            else:
                self.logger.info("Document analysis worker stopped gracefully")

        # Clean up the event loop
        if self._loop and not self._loop.is_closed():
            try:
                self._loop.run_until_complete(self._loop.shutdown_asyncgens())
                self._loop.close()
            except Exception as e:
                self.logger.warning(f"Error closing event loop: {e}")
        self._loop = None

    def _worker_loop(self) -> None:
        """Main worker loop with reactive signaling"""
        loop_count = 0

        while self.running:
            loop_count += 1
            try:
                # Wait for job signal or timeout (reactive instead of busy-waiting)
                self._job_available.wait(timeout=1.0)
                self._job_available.clear()

                # Check if we should exit
                if not self.running:
                    break

                # Process all available jobs
                job_record = self.job_manager.get_next_job()
                while job_record and self.running:
                    self._process_job(job_record)
                    job_record = self.job_manager.get_next_job()

            except Exception as e:
                self.logger.error(f"Error in worker loop #{loop_count}: {type(e).__name__}")
                time.sleep(1)  # Wait before retrying

        self.logger.info(f"Worker loop ended after {loop_count} loops")

    def _process_job(self, job_record: JobDatabaseRecord) -> None:
        """Process a single job (batch or sequential mode)"""
        job_id = job_record.id

        # Update job status to processing
        if not self.job_manager.update_job_status(job_id, JobStatus.PROCESSING):
            self.logger.error(f"Failed to update job {job_id} to processing status")
            return

        try:
            # Extract request data
            request_data = job_record.request_data

            # Decrypt envelope if needed (do this here so decrypted data is available for storage)
            if self.hybrid_crypto.is_encrypted_envelope(request_data):
                self.logger.info(f"Job {job_id}: Decrypting encrypted envelope...")

                # === DIAGNOSTIC LOGGING ===
                self.logger.debug(f"=== ENVELOPE DIAGNOSTICS ===")
                self.logger.debug(f"request_data type: {type(request_data)}")
                self.logger.debug(f"request_data keys: {request_data.keys() if isinstance(request_data, dict) else 'N/A'}")
                if isinstance(request_data, dict):
                    if 'encryptedEnvelope' in request_data:
                        env = request_data['encryptedEnvelope']
                        self.logger.debug(f"encryptedEnvelope keys: {env.keys() if isinstance(env, dict) else 'N/A'}")
                        self.logger.debug(f"encryptedPayload length: {len(env.get('encryptedPayload', ''))}")
                        self.logger.debug(f"payloadIv length: {len(env.get('payloadIv', ''))}")
                    # Check if request_data itself is the envelope
                    if 'encrypted_payload' in request_data:
                        self.logger.debug(f"request_data is encrypted envelope (has encrypted_payload)")
                        self.logger.debug(f"encrypted_payload length: {len(request_data.get('encrypted_payload', ''))}")
                        self.logger.debug(f"payload_iv length: {len(request_data.get('payload_iv', ''))}")
                        self.logger.debug(f"encrypted_key length: {len(request_data.get('encrypted_key', ''))}")
                        self.logger.debug(f"key_iv length: {len(request_data.get('key_iv', ''))}")
                self.logger.debug(f"==============================")

                try:
                    decrypted = self.hybrid_crypto.decrypt_envelope(request_data)
                    request_data = decrypted.payload
                    # Preserve envelope's client_public_key for OTP lookup (removed from payload)
                    request_data["client_public_key"] = decrypted.client_public_key
                    self.logger.info(f"Job {job_id}: Envelope decrypted successfully")
                except HybridCryptoError as e:
                    # Technical error - map to specific error codes
                    error_code = "TECHNICAL_DECRYPTION_FAILED"
                    error_msg_lower = str(e).lower()

                    if "missing" in error_msg_lower or "invalid" in error_msg_lower or "not found" in error_msg_lower:
                        error_code = "TECHNICAL_INVALID_ENVELOPE"
                    elif "authentication" in error_msg_lower or "auth" in error_msg_lower or "signature" in error_msg_lower:
                        error_code = "TECHNICAL_DECRYPTION_FAILED"

                    self.logger.error(
                        f"Job {job_id}: Decryption failed - {str(e)} - code: {error_code}, deleting without retry"
                    )

                    # CRITICAL FIX: Send error callback BEFORE deleting job so client knows what went wrong
                    error_response = {
                        "success": False,
                        "error": str(e),
                        "error_code": error_code,
                        "job_id": job_id,
                        "status": "failed"
                    }
                    if job_record.callback_url:
                        self._send_callback(job_record.callback_url, job_id, error_response)
                        self.logger.info(f"Sent error callback for decryption failure: {job_record.callback_url}")
                    else:
                        self.logger.warning(f"Job {job_id}: No callback URL - client won't be notified of decryption failure")

                    # Delete job immediately - technical errors won't fix on retry
                    if self.job_manager.delete_job(job_id):
                        self.logger.info(f"Job {job_id} deleted due to technical error")

                    # Raise ValidationException to bypass retry logic
                    raise ValidationException(f"Decryption failed (code: {error_code}): {str(e)}")

            # Process job (always sequential mode - batch mode has been removed)
            response_data = self._process_sequential_job(job_id, request_data)

            # Check if this is a step validation error (session has progressed)
            if (response_data and
                isinstance(response_data, dict) and
                response_data.get("sequential_response") and  # Check sequential_response is not None
                isinstance(response_data.get("sequential_response"), dict) and  # Check it's a dict
                not response_data.get("sequential_response", {}).get("success", True) and
                "Invalid step for" in response_data.get("sequential_response", {}).get("message", "")):

                # Don't retry this job - the session has moved on
                error_msg = response_data.get("sequential_response", {}).get("message", "Unknown error")
                self.logger.info(f"Job {job_id} cannot be retried - session has progressed: {error_msg}")
                # Delete the job since it's a stale retry
                if self.job_manager.delete_job(job_id):
                    self.logger.info(f"Deleted stale retry job {job_id}")
                return

            # Store document submission
            storage_success, storage_error = self._store_document_submission(request_data, response_data, job_id)

            if storage_success:
                # Record completion time for queue estimation
                self._record_job_completion(request_data, response_data)

                # Delete job from database instead of marking as completed
                if self.job_manager.delete_job(job_id):
                    self.logger.info(f"Job {job_id} deleted after successful document submission storage")

                    # Push the result to peer instances so they finalize their
                    # shadow rows (fire-and-forget, never blocks the worker)
                    self._broadcast_job_result(job_id, request_data, response_data)

                    # Attempt callback if URL is provided
                    callback_url = job_record.callback_url
                    if callback_url:
                        self._send_callback(callback_url, job_id, response_data)
                    else:
                        self.logger.info(f"No callback URL provided for job {job_id}")
                else:
                    self.logger.error(f"Failed to delete job {job_id} from database")
                    # Fallback: mark as completed
                    self.job_manager.mark_job_completed(job_id, response_data)
            else:
                self.logger.error(f"Failed to store document submission for job {job_id}: {storage_error}")
                # Mark as failed if storage failed
                self.job_manager.mark_job_failed(job_id, storage_error)

        except asyncio.TimeoutError:
            error_msg = f"Job {job_id} timed out after {self.job_timeout} seconds"
            self.logger.error(error_msg)
            self.job_manager.mark_job_failed(job_id, error_msg)

        except ValidationException as e:
            # Validation errors: job already deleted, don't retry
            error_msg = f"Job {job_id} validation failed: {str(e)}"
            self.logger.error(error_msg)
            # Don't call mark_job_failed - job is already deleted

        except Exception as e:
            # Other errors: retry normally
            error_msg = f"Job {job_id} failed: {str(e)}"
            self.logger.error(error_msg)
            self.job_manager.mark_job_failed(job_id, error_msg)

    def _process_sequential_job(self, job_id: str, request_data: dict) -> dict:
        """Process a sequential job with single document processing and timeout enforcement

        Note: request_data should already be decrypted by _process_job if it was an encrypted envelope.
        """
        start_time = time.time()

        try:
            # Handle recovery payload format (from encrypted share request)
            # Recovery payloads have 'public_key' and 'selfie_data' instead of 'client_public_key' and 'files'
            if 'public_key' in request_data and 'selfie_data' in request_data and 'files' not in request_data:
                self.logger.info(f"Job {job_id}: Detected recovery payload format, transforming...")
                # Transform recovery payload to standard format
                request_data['client_public_key'] = request_data['public_key']
                request_data['files'] = [{
                    'filename': request_data.get('filename', 'recovery_selfie.jpg'),
                    'file_data': request_data['selfie_data'],
                    'file_type': 'selfie',  # For classification
                    'document_type': 'secret_share_recovery'  # For routing
                }]
                self.logger.info(f"Job {job_id}: Transformed recovery payload successfully")

            # Extract sequential job parameters
            client_public_key = request_data["client_public_key"]
            files_data = request_data.get("files", [])

            if not files_data or len(files_data) != 1:
                error_msg = f"Sequential jobs must contain exactly one file, got {len(files_data)}"
                self.logger.error(f"Job {job_id}: {error_msg} - validation error, deleting without retry")

                # Send error callback before deleting job
                callback_url = request_data.get("callback_url") or job_record.callback_url
                error_response = {
                    "success": False,
                    "error": error_msg,
                    "error_code": "TECHNICAL_INVALID_ENVELOPE",
                    "job_id": job_id,
                    "status": "failed"
                }
                if callback_url:
                    self._send_callback(callback_url, job_id, error_response)
                    self.logger.info(f"Sent error callback for validation error (empty files): {callback_url}")
                else:
                    self.logger.warning(f"Job {job_id}: No callback URL - client won't be notified of validation error")

                # Delete job immediately - don't retry validation errors
                if self.job_manager.delete_job(job_id):
                    self.logger.info(f"Job {job_id} deleted due to validation error (empty files)")
                else:
                    self.logger.error(f"Failed to delete job {job_id} after validation error")
                # Raise ValidationException to bypass retry logic
                raise ValidationException(error_msg)

            file_data = files_data[0]
            document_type = file_data.get("document_type") or file_data.get("file_type")

            # Normalize document type to handle naming variations
            original_document_type = document_type
            document_type = normalize_document_type(document_type)

            if original_document_type != document_type:
                self.logger.info(f"Normalized document type: '{original_document_type}' -> '{document_type}'")

            # Check rate limit BEFORE processing any document (single limit across all types)
            # Skip rate limit check for key management operations
            if document_type not in ("secret_share_recovery", "add_public_key", "remove_public_key"):
                user_identity_id = self._get_user_identity_id_from_public_key(client_public_key)
                if user_identity_id:
                    is_allowed, rate_error = self.document_submission_repository.check_rate_limit(user_identity_id)
                    if not is_allowed:
                        self.logger.warning(f"Job {job_id}: Rate limit exceeded - {rate_error}")
                        raise ValueError(rate_error)
                    self.logger.info(f"Job {job_id}: Rate limit check passed for user {user_identity_id[:16]}...")

            callback_url = request_data.get("callback_url")

            # Extract request-level selfie parameters
            secret_share = request_data.get("secret_share")

            # Get the persistent event loop
            loop = self._get_event_loop()

            # Route to appropriate sequential service with timeout enforcement
            if document_type == "video_selfie":
                # Video selfie with hand gesture OTP
                if self.video_selfie_service is None:
                    raise ValueError("VideoSelfieService not available - mediapipe is not installed")

                import base64
                mobile_number = request_data.get("mobile_number", "")
                country_code = request_data.get("country_code", "")

                video_bytes = base64.b64decode(file_data["file_data"])

                coro = self.video_selfie_service.process_video_selfie(
                    client_public_key=client_public_key,
                    video_bytes=video_bytes,
                    filename=file_data["filename"],
                    mobile_number=mobile_number,
                    country_code=country_code,
                    callback_url=callback_url
                )
                response = loop.run_until_complete(
                    asyncio.wait_for(coro, timeout=self.job_timeout)
                )
            elif document_type == "selfie":
                # Extract mobile number and country code from request
                mobile_number = request_data.get("mobile_number", "")
                country_code = request_data.get("country_code", "")

                coro = self.sequential_selfie_service.process_selfie(
                    client_public_key=client_public_key,
                    file_data=file_data["file_data"],
                    filename=file_data["filename"],
                    secret_share=secret_share,
                    mobile_number=mobile_number,
                    country_code=country_code,
                    callback_url=callback_url
                )
                response = loop.run_until_complete(
                    asyncio.wait_for(coro, timeout=self.job_timeout)
                )
            elif document_type == "passport":
                # Use the strict pipeline with dynamic region exclusion (v2.0)
                coro = self.sequential_passport_service.process_passport_strict(
                    client_public_key=client_public_key,
                    file_data=file_data["file_data"],
                    filename=file_data["filename"],
                    callback_url=callback_url
                )
                response = loop.run_until_complete(
                    asyncio.wait_for(coro, timeout=self.job_timeout)
                )
            elif document_type == "bank_statement":
                coro = self.sequential_bank_statement_service.process_bank_statement(
                    client_public_key=client_public_key,
                    file_data=file_data["file_data"],
                    filename=file_data["filename"],
                    callback_url=callback_url
                )
                response = loop.run_until_complete(
                    asyncio.wait_for(coro, timeout=self.job_timeout)
                )
            elif document_type == "tax_statement":
                # Tax statements are independent of sequential flow
                coro = self.sequential_tax_statement_service.process_tax_statement(
                    client_public_key=client_public_key,
                    file_data=file_data["file_data"],
                    filename=file_data["filename"],
                    callback_url=callback_url
                )
                response = loop.run_until_complete(
                    asyncio.wait_for(coro, timeout=self.job_timeout)
                )
            elif document_type == "resume":
                # Resumes are optional documents with PII extraction
                coro = self.sequential_resume_service.process_resume(
                    client_public_key=client_public_key,
                    file_data=file_data["file_data"],
                    filename=file_data["filename"],
                    callback_url=callback_url
                )
                response = loop.run_until_complete(
                    asyncio.wait_for(coro, timeout=self.job_timeout)
                )
            elif document_type == "secret_share_recovery":
                # Secret share recovery - uses KeyRecoveryService
                from app.services.key_recovery_service import KeyRecoveryService, KeyRecoveryError
                import base64
                import os

                key_recovery_service = KeyRecoveryService()

                # Decode selfie from base64
                selfie_bytes = base64.b64decode(file_data["file_data"])

                # Get filename for type detection (video vs image)
                filename = file_data.get("filename", "recovery_selfie.jpg")

                # Auto-detect media type from magic bytes if not explicitly set
                media_type = file_data.get("media_type")
                if not media_type:
                    # Use KeyRecoveryService's _detect_file_type which checks magic bytes
                    # This correctly identifies video files even if filename has .jpg extension
                    media_type = key_recovery_service._detect_file_type(filename, selfie_bytes)

                # Log media type for debugging
                self.logger.info(f"Secret share recovery - media_type: {media_type}, filename: {filename}")

                # For secret share recovery, use identity_id from job request (if available)
                # This allows recovery when the user doesn't have their original key
                identity_id = request_data.get("identity_id")
                mobile_number = request_data.get("mobile_number")
                otp_code = request_data.get("otp_code")
                api_url = request_data.get("api_url")  # Optional: filter shares by API URL

                # Always use recover_key_with_identity() - it handles both image and video selfies
                # For image selfies with OTP: identity_id and mobile_number are pre-populated
                # For video selfies: identity_id will be extracted from verification result
                coro = key_recovery_service.recover_key_with_identity(
                    temp_public_key=client_public_key,  # temp_public_key for re-encryption
                    selfie_bytes=selfie_bytes,
                    otp_code=otp_code,
                    filename=file_data.get("filename"),
                    identity_id=identity_id,  # None for video selfies (extracted from verification)
                    mobile_number=mobile_number,  # None for video selfies (used for logging only)
                    api_url=api_url  # Optional: filter shares by API URL
                )
                response = loop.run_until_complete(
                    asyncio.wait_for(coro, timeout=self.job_timeout)
                )
            elif document_type == "add_public_key":
                # Add new public key - uses KeyManagementService
                import base64

                # Extract data from metadata (passed via encrypted payload)
                metadata = request_data.get("metadata", {})
                new_public_key = metadata.get("new_public_key")
                client_secret_share = metadata.get("client_secret_share")

                # Decode selfie from base64
                selfie_bytes = base64.b64decode(file_data["file_data"])

                coro = self.key_management_service.add_public_key(
                    encrypting_public_key=client_public_key,
                    new_public_key=new_public_key,
                    client_secret_share=client_secret_share,
                    selfie_bytes=selfie_bytes,
                    otp_code=request_data.get("otp_code"),
                    filename=file_data.get("filename")
                )
                response = loop.run_until_complete(
                    asyncio.wait_for(coro, timeout=self.job_timeout)
                )
            elif document_type == "remove_public_key":
                # Remove public key - uses KeyManagementService
                import base64

                # Extract data from metadata (passed via encrypted payload)
                metadata = request_data.get("metadata", {})
                public_key_to_remove = metadata.get("public_key_to_remove")

                # Decode selfie from base64
                selfie_bytes = base64.b64decode(file_data["file_data"])

                coro = self.key_management_service.remove_public_key(
                    encrypting_public_key=client_public_key,
                    public_key_to_remove=public_key_to_remove,
                    selfie_bytes=selfie_bytes,
                    otp_code=request_data.get("otp_code"),
                    filename=file_data.get("filename")
                )
                response = loop.run_until_complete(
                    asyncio.wait_for(coro, timeout=self.job_timeout)
                )
            elif document_type == "field_verification":
                # Field verification - verify expected values against extracted data
                from app.services.field_verification_service import FieldVerificationService

                self.logger.info("Field verification requested, using field verification service")

                field_verification_service = FieldVerificationService(
                    detailed_analysis_service=self.detailed_analysis_service,
                    state_service=None,  # Not used for field verification
                    user_identity_repo=self.user_identity_repository
                )

                coro = field_verification_service.verify_fields(
                    request_data=request_data,
                    job_id=job_id,
                    client_public_key=client_public_key
                )
                response = loop.run_until_complete(
                    asyncio.wait_for(coro, timeout=self.job_timeout)
                )
            elif document_type == "auto":
                # Auto-detection for generic documents
                # Uses GLiNER2-powered generic document type detector
                from app.services.generic_document_service import GenericDocumentService

                self.logger.info("Auto-detection requested, using generic document service")

                generic_service = GenericDocumentService(
                    user_identity_repo=self.user_identity_repository
                )

                # Process with auto-detection (runs async)
                # Note: Auto-detection mode requires document_type to be provided
                # For backward compatibility, we default to 'id_card' when None
                coro = generic_service.process_auto_document(
                    file_data=file_data,
                    client_public_key=client_public_key,
                    user_identity_id=user_identity_id,
                    document_type='id_card',  # Default for auto-detection
                    country_code=None,
                    entity=None
                )
                response = loop.run_until_complete(
                    asyncio.wait_for(coro, timeout=self.job_timeout)
                )
            else:
                # Handle optional ID documents (does NOT affect verification state)
                # These are for supplementary information only
                OPTIONAL_ID_DOCUMENT_TYPES = {
                    "id_card", "nric", "driving_license", "pan_card"
                }

                if document_type in OPTIONAL_ID_DOCUMENT_TYPES:
                    # Import base64 for decoding
                    import base64

                    self.logger.info(f"Processing optional ID document: {document_type}")

                    # Decode image from base64
                    image_bytes = base64.b64decode(file_data["file_data"])
                    filename = file_data.get("filename", "id_document.jpg")

                    # Process as optional ID document (does not affect verification state)
                    coro = self._process_optional_id_document(
                        document_type=document_type,
                        image_bytes=image_bytes,
                        filename=filename,
                        user_identity_id=user_identity_id
                    )
                    response = loop.run_until_complete(
                        asyncio.wait_for(coro, timeout=self.job_timeout)
                    )
                else:
                    # Generic document types - route to GenericDocumentService
                    # Includes: tax_return, tax_residency_certificate, utility_bill,
                    #           payslip, insurance_policy, employment_letter, residence_proof, etc.
                    GENERIC_DOCUMENT_TYPES = {
                        "tax_return", "tax_residency_certificate",
                        "utility_bill", "payslip", "insurance_policy", "employment_letter",
                        "residence_proof"
                    }

                    if document_type in GENERIC_DOCUMENT_TYPES:
                        # Generic document type - uses GenericDocumentService
                        from app.services.generic_document_service import GenericDocumentService

                        self.logger.info(f"Generic document type '{document_type}' requested, using generic document service")

                        generic_service = GenericDocumentService(
                            user_identity_repo=self.user_identity_repository
                        )

                        # Extract document type, country, and entity for direct passing
                        doc_type = document_type  # Already have this
                        country = file_data.get("country")
                        entity = file_data.get("entity")

                        coro = generic_service.process_auto_document(
                            file_data=file_data,
                            client_public_key=client_public_key,
                            user_identity_id=user_identity_id,
                            document_type=doc_type,
                            country_code=country,
                            entity=entity
                        )
                        response = loop.run_until_complete(
                            asyncio.wait_for(coro, timeout=self.job_timeout)
                        )
                    else:
                        raise ValueError(f"Unsupported document type for sequential processing: {document_type}")

            processing_time = time.time() - start_time
            self.logger.info(f"Sequential job {job_id} completed in {processing_time:.2f} seconds")

            # Prepare response data (simplified structure)
            # Handle dict responses (from secret_share_recovery) vs object responses
            if isinstance(response, dict):
                # For secret share recovery, wrap results in extracted_data for ECIES encryption
                extracted_data = None
                if response.get('shares'):
                    extracted_data = {
                        "identity_id": response.get('identity_id'),
                        "shares": response.get('shares'),  # ECIES envelopes
                        "total_shares": response.get('total_shares'),
                        "face_match_confidence": response.get('face_match_confidence'),
                    }

                response_data = {
                    "result": response.get('success', False),
                    "job_id": job_id,
                    "processing_time_seconds": round(processing_time, 2),
                    # Include extracted_data for ECIES encryption and storage
                    "extracted_data": extracted_data,
                    # Top-level fields for backward compatibility
                    "success": response.get('success'),
                    "face_match_confidence": response.get('face_match_confidence'),
                    "faces_checked": response.get('faces_checked')
                }
            else:
                # Use model_dump() to serialize SequentialJobResponse (includes computed 'status' field)
                response_data = response.model_dump()

            return response_data

        except Exception as e:
            self.logger.error(f"Sequential job processing failed: {type(e).__name__}: {str(e)}")
            # Extract error_code if KeyRecoveryError
            error_code = None
            if hasattr(e, 'error_code'):
                error_code = e.error_code
            # Return error response in consistent format
            error_response = {
                "result": False,
                "status": "failed",
                "job_id": job_id,
                "verification_state": 0,
                "processing_time_seconds": round(time.time() - start_time, 2),
                "error": str(e)
            }
            # Include error_code if available
            if error_code:
                error_response["error_code"] = error_code
            return error_response

    def _broadcast_job_result(self, job_id: str, request_data: dict, response_data: dict) -> None:
        """
        Push a completed job's result to peer instances (fire-and-forget).

        Sends the full result payload plus the decrypted (stripped) request
        data and the user's key/identity/state mapping so peers can store an
        identical submission row without reprocessing the document.
        """
        try:
            from app.repositories.document_submission_repository import strip_large_fields
            from app.services.job_broadcast_service import job_broadcast_service

            client_public_key = request_data.get('client_public_key')
            user_key_info = {
                'client_public_key': client_public_key,
                'user_identity_id': response_data.get('user_identity_id'),
                'verification_state': response_data.get('verification_state'),
                'sequence_no': response_data.get('sequence_no'),
            }

            # Fill state from user_keys when the response shape lacks it
            if client_public_key and (
                user_key_info['verification_state'] is None
                or user_key_info['sequence_no'] is None
            ):
                try:
                    from app.repositories.user_key_repository import UserKeyRepository
                    user_key_repo = UserKeyRepository()
                    if user_key_info['verification_state'] is None:
                        user_key_info['verification_state'] = user_key_repo.get_verification_state(client_public_key)
                    if user_key_info['sequence_no'] is None:
                        user_key_info['sequence_no'] = user_key_repo.get_sequence_no(client_public_key)
                except Exception:
                    pass

            job_broadcast_service.broadcast_job_result(
                job_id=job_id,
                response_data=response_data,
                request_data_decrypted_stripped=strip_large_fields(request_data),
                user_key_info=user_key_info
            )
        except Exception as e:
            self.logger.warning(f"Failed to broadcast job_result for {job_id}: {e}")

    def _send_callback(self, callback_url: str, job_id: str, response_data: dict) -> bool:
        """Send callback with job results using async HTTP (non-blocking)"""
        try:
            self.logger.info(f"Sending callback for job {job_id} to {callback_url}")

            # Use the persistent event loop for async callback
            loop = self._get_event_loop()
            return loop.run_until_complete(
                self._send_callback_async(callback_url, job_id, response_data)
            )

        except Exception as e:
            error_msg = f"Unexpected error during callback for job {job_id}: {str(e)}"
            self.logger.error(error_msg)
            return False

    async def _send_callback_async(self, callback_url: str, job_id: str, response_data: dict) -> bool:
        """Async implementation of callback sending"""
        try:
            # Prepare callback payload
            callback_payload = {
                "job_id": job_id,
                "status": JobStatus.COMPLETED.value,
                "created_at": datetime.now().isoformat(),
                "results": response_data
            }

            # Mark callback attempt in database
            self.job_manager.mark_callback_attempted(job_id)

            # Send async HTTP POST request with timeout
            timeout = aiohttp.ClientTimeout(total=self.callback_timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    callback_url,
                    json=callback_payload,
                    headers={
                        "Content-Type": "application/json",
                        "User-Agent": "DocumentAnalysisWorker/1.0"
                    }
                ) as response:
                    if response.status == 200:
                        self.logger.info(f"Callback sent successfully for job {job_id}")
                        return True
                    else:
                        response_text = await response.text()
                        error_msg = f"Callback failed with status {response.status}: {response_text}"
                        self.logger.error(error_msg)
                        return False

        except asyncio.TimeoutError:
            error_msg = f"Callback timeout for job {job_id}"
            self.logger.error(error_msg)
            return False

        except aiohttp.ClientError as e:
            error_msg = f"Callback request failed for job {job_id}: {str(e)}"
            self.logger.error(error_msg)
            return False

        except Exception as e:
            error_msg = f"Unexpected error during async callback for job {job_id}: {str(e)}"
            self.logger.error(error_msg)
            return False

    async def _process_optional_id_document(
        self,
        document_type: str,
        image_bytes: bytes,
        filename: str,
        user_identity_id: str
    ) -> Dict[str, Any]:
        """
        Process optional ID document without affecting verification state.

        This is for supplementary information only and does NOT count towards
        verification requirements or affect the verification sequence.
        """
        from app.services.extractors.qwen_universal_id_extractor import QwenUniversalIDExtractor

        try:
            # Extract using universal ID extractor
            extractor = QwenUniversalIDExtractor()
            extraction_result = await extractor.extract_from_image(
                image_bytes=image_bytes
            )

            extracted_data = extraction_result.extracted_data

            # Validate required fields
            required_fields = ['issuing_country', 'id_type', 'id_number', 'full_name']
            missing_fields = [f for f in required_fields if f not in extracted_data]

            if missing_fields:
                return {
                    'success': False,
                    'error': f'Missing required fields: {", ".join(missing_fields)}',
                    'document_type': document_type
                }

            # Return extracted data without affecting verification state
            # Format: Compatible with existing response handling
            return {
                'success': True,
                'status': 'completed',
                'document_type': document_type,
                'extracted_data': {
                    'issuing_country': extracted_data.get('issuing_country', {}).get('value'),
                    'id_type': extracted_data.get('id_type', {}).get('value'),
                    'id_number': extracted_data.get('id_number', {}).get('value'),
                    'full_name': extracted_data.get('full_name', {}).get('value'),
                    'expiry_date': extracted_data.get('expiry_date', {}).get('value'),
                    'is_supplementary': True  # Flag indicating this doesn't affect verification
                },
                # Verification state is NOT affected (remains unchanged)
                'verification_state': None,  # Indicates no change to verification state
                'is_supplementary': True
            }

        except Exception as e:
            self.logger.error(f"Optional ID processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'document_type': document_type
            }

    def get_worker_stats(self) -> dict:
        """Get worker statistics"""
        return {
            "running": self.running,
            "thread_alive": self.worker_thread.is_alive() if self.worker_thread else False,
            "job_timeout": self.job_timeout,
            "callback_timeout": self.callback_timeout,
            "queue_stats": self.job_manager.get_queue_stats()
        }

    def _store_document_submission(
        self,
        request_data: dict,
        response_data: dict,
        job_id: str
    ) -> tuple[bool, str]:
        """
        Store document submission with analysis results.

        Note: document_hash removed - uniqueness enforced by face biometrics trigger.
        Same document can be submitted with different encryption (multi-device).

        Args:
            request_data: Original request data containing decrypted files
            response_data: Response data with analysis results
            job_id: Job ID to link submission to original job

        Returns:
            Tuple of (success, error_message):
            - (True, "") if storage successful
            - (False, error_message) if failed
        """
        try:
            # Store using create_submission
            result = self.document_submission_repository.create_submission(
                response_data=response_data,
                request_data=request_data,
                job_id=job_id
            )

            if result:
                self.logger.info(f"Stored document submission for job {job_id}")
                return True, ""
            else:
                self.logger.error(f"Failed to store document submission for job {job_id}")
                return False, "Failed to store document submission"

        except Exception as e:
            self.logger.error(f"Error storing document submission: {str(e)}")
            return False, f"Storage error: {str(e)}"

    def _get_user_identity_id_from_public_key(self, client_public_key: str) -> Optional[str]:
        """
        Get user_identity_id from client public key.

        Args:
            client_public_key: Client's public key

        Returns:
            user_identity_id or None if not found
        """
        try:
            user_key = self.user_key_repository.get_key_by_public_key(client_public_key)
            if user_key and user_key.get('user_identity_id'):
                return user_key['user_identity_id']
            return None
        except Exception as e:
            self.logger.error(f"Error getting user_identity_id from public key: {e}")
            return None

    def _record_job_completion(self, request_data: dict, response_data: dict) -> None:
        """
        Record job completion time for queue time estimation.

        Extracts document type and processing time from the completed job
        and updates the running averages in JobTimingService.

        Args:
            request_data: Original request data containing files
            response_data: Response data containing processing_time_seconds
        """
        try:
            # Extract document type from request
            files_data = request_data.get("files", [])
            if not files_data:
                return

            file_data = files_data[0]
            document_type = file_data.get("document_type") or file_data.get("file_type")
            if not document_type:
                return

            # Get processing time from response
            processing_time = response_data.get("processing_time_seconds")
            if processing_time is None or processing_time <= 0:
                return

            # Normalize document type
            document_type = normalize_document_type(document_type)

            # Record completion time
            self.timing_service.record_completion(document_type, processing_time)
            self.logger.debug(
                f"Recorded completion time for {document_type}: {processing_time:.2f}s"
            )

        except Exception as e:
            self.logger.error(f"Error recording job completion time: {e}")

    def is_healthy(self) -> bool:
        """Check if worker is healthy"""
        return (
            self.running and
            self.worker_thread and
            self.worker_thread.is_alive()
        )
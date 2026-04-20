"""
Startup Sync Service

Non-blocking startup synchronization service that runs in a separate thread
to request jobs and OTPs from peer instances on startup.

This replaces RethinkDB changefeeds with a simpler HTTP-based approach:
1. On startup, request jobs meant for this instance from peer instances
2. On startup, request OTPs from peer instances
3. Insert received jobs into local JobQueue for processing
4. Insert received OTPs into MariaDB
"""

import asyncio
import threading
from typing import Dict, Any, Tuple, List
import httpx
from datetime import datetime

from app.core.job_queue import JobQueue
from app.repositories.otp_repository import OTPRepository
from app.config.instance_config import instance_config
from app.core.logger import get_logger
from app.core.key.internal_signature import InternalSignature

logger = get_logger()


class StartupSyncService:
    """
    Non-blocking startup synchronization service.

    Runs in a separate thread to request jobs and OTPs from peer instances.
    This allows the main application to start immediately while sync happens in background.
    """

    def __init__(self, local_job_queue: JobQueue, instance_public_key: str):
        """
        Initialize startup sync service.

        Args:
            local_job_queue: Local in-memory job queue
            instance_public_key: This instance's public key
        """
        self.local_job_queue = local_job_queue
        self.instance_public_key = instance_public_key
        self.otp_repo = OTPRepository()
        self._sync_thread = None
        self._sync_complete = False
        self.logger = logger
        # Get server keys for signing
        self.server_public_key, self.server_private_key = InternalSignature.get_server_keys()

    def start_sync(self) -> None:
        """Start sync in background thread (non-blocking)"""
        if not instance_config.peer_instances or not instance_config.startup_sync_enabled:
            self.logger.info("No peer instances configured or startup sync disabled")
            self._sync_complete = True
            return

        self._sync_thread = threading.Thread(
            target=self._run_sync,
            daemon=True,
            name="StartupSyncThread"
        )
        self._sync_thread.start()
        self.logger.info(
            f"Startup sync thread started for {len(instance_config.peer_instances)} peers"
        )

    def _run_sync(self) -> None:
        """Run synchronization in background thread"""
        try:
            self.logger.info("Startup sync thread: beginning synchronization")
            # Run async sync in new event loop
            asyncio.run(self._sync_from_peers())
            self._sync_complete = True
            self.logger.info("Startup sync thread: synchronization complete")
        except Exception as e:
            self.logger.error(f"Startup sync failed: {e}", exc_info=True)
            self._sync_complete = True

    async def _sync_from_peers(self) -> None:
        """Request jobs and OTPs from all peer instances"""
        peer_urls = instance_config.get_peer_urls()

        if not peer_urls:
            self.logger.info("No peer URLs configured for sync")
            return

        async with httpx.AsyncClient(timeout=instance_config.request_timeout) as client:
            tasks = []
            for peer_url in peer_urls:
                tasks.append(self._sync_from_peer(client, peer_url))

            # Run all requests in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Log results
            total_jobs = 0
            total_otps = 0
            for peer_url, result in zip(peer_urls, results):
                if isinstance(result, Exception):
                    self.logger.warning(f"Failed to sync from {peer_url}: {result}")
                else:
                    jobs, otps = result
                    total_jobs += jobs
                    total_otps += otps
                    self.logger.info(f"Synced from {peer_url}: {jobs} jobs, {otps} OTPs")

            if total_jobs > 0 or total_otps > 0:
                self.logger.info(f"Startup sync complete: {total_jobs} jobs, {total_otps} OTPs")
            else:
                self.logger.info("Startup sync complete: no new data from peers")

    async def _sync_from_peer(self, client: httpx.AsyncClient, peer_url: str) -> Tuple[int, int]:
        """
        Request jobs and OTPs from a single peer instance.

        Returns:
            Tuple of (jobs_count, otps_count)
        """
        jobs_count = 0
        otps_count = 0

        # Create signature headers
        headers = InternalSignature.create_signature_headers(
            self.server_private_key,
            self.server_public_key
        )

        # Request jobs meant for this instance
        try:
            response = await client.post(
                f"{peer_url}/api/internal/sync/jobs",
                json={"instance_public_key": self.instance_public_key},
                headers=headers,
                timeout=instance_config.request_timeout
            )
            if response.status_code == 200:
                data = response.json()
                jobs_data = data.get("jobs", [])
                for job_data in jobs_data:
                    self._insert_job_into_local_queue(job_data)
                    jobs_count += 1
            else:
                self.logger.warning(
                    f"Failed to request jobs from {peer_url}: status {response.status_code}"
                )
        except Exception as e:
            self.logger.warning(f"Failed to request jobs from {peer_url}: {e}")

        # Request OTPs
        try:
            response = await client.get(
                f"{peer_url}/api/internal/sync/otps",
                headers=headers,
                timeout=instance_config.request_timeout
            )
            if response.status_code == 200:
                data = response.json()
                otps_data = data.get("otps", [])
                for otp_data in otps_data:
                    self._insert_otp_into_mariadb(otp_data)
                    otps_count += 1
            else:
                self.logger.warning(
                    f"Failed to request OTPs from {peer_url}: status {response.status_code}"
                )
        except Exception as e:
            self.logger.warning(f"Failed to request OTPs from {peer_url}: {e}")

        return jobs_count, otps_count

    def _insert_job_into_local_queue(self, job_data: Dict[str, Any]) -> None:
        """
        Insert job into local in-memory queue.

        The job data should be in the format returned by the internal API:
        {
            "id": "job-id",
            "status": "pending",
            "request_data": {...},
            "created_at": "2024-01-01T00:00:00",
            ...
        }
        """
        try:
            job_id = job_data.get('id')
            if not job_id:
                self.logger.warning("Job data missing id, skipping")
                return

            # Convert to JobDatabaseRecord format if needed
            from app.dto.job_models import JobDatabaseRecord, JobStatus

            # Parse datetime strings
            created_at = job_data.get('created_at')
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))

            job_record = JobDatabaseRecord(
                id=job_id,
                status=JobStatus(job_data.get('status', 'pending')),
                request_data=job_data.get('request_data', {}),
                response_data=job_data.get('response_data'),
                error_message=job_data.get('error_message'),
                callback_url=job_data.get('callback_url'),
                retry_count=job_data.get('retry_count', 0),
                max_retries=job_data.get('max_retries', 3),
                created_at=created_at,
                updated_at=job_data.get('updated_at'),
                started_at=job_data.get('started_at'),
                completed_at=job_data.get('completed_at'),
                callback_attempted_at=job_data.get('callback_attempted_at')
            )

            self.local_job_queue.put(job_record)
            self.logger.info(f"Inserted job {job_id} into local queue from sync")
        except Exception as e:
            self.logger.error(f"Failed to insert job into local queue: {e}", exc_info=True)

    def _insert_otp_into_mariadb(self, otp_data: Dict[str, Any]) -> None:
        """
        Insert OTP into MariaDB (upsert to handle duplicates).

        The OTP data should be in the format returned by the internal API:
        {
            "id": "otp-id",
            "mobile_number": "+1234567890",
            "random_number": "123456",
            ...
        }
        """
        try:
            # Only sync OTPs that have public_key
            public_key = otp_data.get('public_key')
            if not public_key:
                return

            existing = self.otp_repo.get_otp_by_public_key(public_key)

            # Prepare OTP data for database
            # Remove fields that shouldn't be inserted/updated
            db_otp_data = {
                'public_key': public_key,
                'random_number': otp_data.get('random_number'),
                'otp_id': otp_data.get('otp_id'),
                'delivery_method': otp_data.get('delivery_method', 'sms'),
                'expires_at': otp_data.get('expires_at'),
                'attempts': otp_data.get('attempts', 0),
                'max_attempts': otp_data.get('max_attempts', 3),
                'is_verified': otp_data.get('is_verified', False)
            }

            # Add optional fields if present
            if otp_data.get('mobile_number'):
                db_otp_data['mobile_number'] = otp_data['mobile_number']

            if otp_data.get('country_code'):
                db_otp_data['country_code'] = otp_data['country_code']

            if existing:
                # Update existing OTP
                self.otp_repo.update_otp_by_public_key(public_key, db_otp_data)
                self.logger.debug(f"Updated existing OTP for public_key {public_key[:16]}... from sync")
            else:
                # Insert new OTP
                self.otp_repo.create_otp(db_otp_data)
                self.logger.debug(f"Inserted new OTP for public_key {public_key[:16]}... from sync")

        except Exception as e:
            self.logger.error(f"Failed to insert OTP from sync: {e}", exc_info=True)

    def wait_for_sync(self, timeout: float = 30.0) -> bool:
        """
        Wait for sync to complete (optional, for testing).

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if sync completed, False if timeout
        """
        start_time = datetime.now()
        while not self._sync_complete:
            if (datetime.now() - start_time).total_seconds() > timeout:
                return False
            import time
            time.sleep(0.1)
        return True

    def is_sync_complete(self) -> bool:
        """Check if sync is complete"""
        return self._sync_complete

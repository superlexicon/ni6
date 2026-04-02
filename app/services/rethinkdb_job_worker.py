"""
RethinkDB Job Worker

Subscribes to RethinkDB change stream and processes jobs for this instance.
Jobs are filtered by instance_public_key so each instance only processes its own jobs.
"""

import time
import threading
from typing import Callable, Optional, Dict, Any

from rethinkdb import RethinkDB
from app.core import logger
from app.config.rethinkdb_config import rethinkdb_settings

r = RethinkDB()


class RethinkDBJobWorker:
    """
    Worker that processes jobs from RethinkDB change stream.

    Subscribes to filtered changefeed (only this instance's jobs)
    and processes them in the order received.
    """

    def __init__(
        self,
        rethinkdb_connection,
        instance_public_key: str,
        job_processor: Callable,
        job_queue=None,
        job_manager=None
    ):
        """
        Initialize RethinkDB job worker.

        Args:
            rethinkdb_connection: RethinkDB connection manager
            instance_public_key: Server public key for this instance
            job_processor: Callable that processes jobs - should have signature:
                          process_job(job_id: str, job_data: Dict) -> Dict
                          Must update MySQL status before returning
            job_queue: Optional RethinkDBJobQueue for manual job deletion
            job_manager: Optional JobManager for MySQL operations
        """
        self.rethinkdb_connection = rethinkdb_connection
        self.instance_public_key = instance_public_key
        self.job_processor = job_processor
        self.job_queue = job_queue
        self.job_manager = job_manager
        self.db_name = rethinkdb_settings.db
        self.table_name = rethinkdb_settings.jobs_table
        self.logger = logger

        self.running = False
        self._worker_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the worker thread."""
        if self.running:
            self.logger.warning("Worker already running")
            return

        self.running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="RethinkDBJobWorker"
        )
        self._worker_thread.start()
        self.logger.info(f"RethinkDB job worker started for instance {self.instance_public_key[:16]}...")

    def stop(self) -> None:
        """Stop the worker thread."""
        self.running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)
        self.logger.info("RethinkDB job worker stopped")

    def _worker_loop(self) -> None:
        """Main worker loop - subscribes to change stream and processes jobs."""
        backoff_time = 1.0

        while self.running:
            conn = None
            try:
                conn = self.rethinkdb_connection.new_connection()

                # Subscribe to changes for this instance only
                # Filter table FIRST, then get changes
                # include_initial=True to process jobs queued while instance was down
                # Python variable is interpolated into ReQL before sending to server
                cursor = r.db(self.db_name).table(self.table_name).filter(
                    r.row["instance_public_key"] == self.instance_public_key
                ).changes(include_initial=True, include_types=False).run(conn)

                self.logger.info(f"Subscribed to change stream for instance {self.instance_public_key[:16]}...")
                backoff_time = 1.0  # Reset backoff on successful connection

                for change in cursor:
                    if not self.running:
                        break

                    self._process_change(change, conn)

            except Exception as e:
                self.logger.error(f"Change stream error: {e}")
                if self.running:
                    # Exponential backoff before reconnect
                    self.logger.info(f"Reconnecting in {backoff_time}s...")
                    time.sleep(backoff_time)
                    backoff_time = min(backoff_time * 2, 30.0)  # Max 30s

            finally:
                if conn:
                    try:
                        conn.close()
                    except Exception:
                        pass

    def _process_change(self, change: Dict[str, Any], conn) -> None:
        """
        Process a job from the change stream.

        Flow:
        1. Extract job data from change
        2. Call job_processor to process the job
        3. Delete from RethinkDB ONLY after MySQL is updated

        Args:
            change: RethinkDB change document
            conn: Active RethinkDB connection
        """
        new_val = change.get("new_val")
        if not new_val:
            # Skip deletions (we don't care about them)
            return

        job_id = new_val.get("id")
        if not job_id:
            self.logger.warning("Received change without job_id, skipping")
            return

        job_data = new_val.get("request_data")
        status = new_val.get("status", "pending")

        if status != "pending":
            # Skip non-pending jobs (may be retried jobs)
            self.logger.debug(f"Skipping job {job_id} with status {status}")
            return

        # Defensive check: Skip jobs not meant for this instance
        job_instance_key = new_val.get("instance_public_key")
        if job_instance_key != self.instance_public_key:
            self.logger.debug(
                f"Skipping job {job_id} - belongs to instance {job_instance_key[:16] if job_instance_key else None}..."
            )
            return

        self.logger.info(f"Processing job {job_id} from change stream")

        # Ensure job exists in local MySQL database before processing
        # Jobs are created in one instance's MySQL but may be processed by another instance
        job_record = self.job_manager.job_repo.get_job_by_id(job_id)
        if not job_record:
            self.logger.info(f"Job {job_id} not found in local MySQL, inserting from RethinkDB data")
            # Insert job into local MySQL from RethinkDB data (idempotent - handles duplicates)
            result = self.job_manager.job_repo.create_job(
                job_id=job_id,
                request_data=job_data,
                callback_url=new_val.get("callback_url"),
                max_retries=new_val.get("max_retries", 3)
            )

            # create_job returns {"exists": True, ...} if job already exists (idempotency)
            if isinstance(result, dict) and result.get("exists"):
                self.logger.info(f"Job {job_id} already exists in local MySQL")
            elif not result:
                self.logger.error(f"Failed to create job {job_id} in local MySQL")
                return

        try:
            # 1. Process the job
            # The job_processor should:
            # - Update MySQL status to 'processing'
            # - Do the actual work
            # - Update MySQL with final status and results
            result = self.job_processor(job_id, job_data)

            # 2. Delete from RethinkDB ONLY after MySQL is updated
            # This ensures job data is persisted even if processing fails
            delete_result = r.db(self.db_name).table(self.table_name).get(job_id).delete().run(conn)

            if delete_result.get("deleted"):
                self.logger.info(f"Job {job_id} completed and deleted from RethinkDB")
            else:
                self.logger.warning(f"Job {job_id} was already deleted or not found")

        except Exception as e:
            # On failure, update job status in MySQL to 'failed'
            # Do NOT delete from RethinkDB - allows manual retry
            self.logger.error(f"Job {job_id} failed: {e}")
            # Note: job_processor should handle MySQL status update on failure

    def is_running(self) -> bool:
        """Check if worker is running."""
        return self.running and self._worker_thread is not None and self._worker_thread.is_alive()

"""
Job Timing Statistics Service

Tracks average completion times per job type for queue time estimation.
Maintains running averages that update as jobs complete.
Bootstraps from historical data on startup.
"""

import threading
from typing import Optional, Dict, List
from datetime import datetime, timedelta
from app.core.logger import get_logger
from app.core.db.database import get_db_connection_context


class JobTimingService:
    """
    Tracks average completion times per job type.

    Maintains running averages that update as jobs complete.
    Bootstraps from historical data on startup.

    Attributes:
        job_type_averages: Dict mapping document_type to average processing time in seconds
        job_type_counts: Dict mapping document_type to total completed job count
    """

    def __init__(self, enable_bootstrap: bool = True):
        """
        Initialize JobTimingService.

        Args:
            enable_bootstrap: If True, bootstrap averages from historical data on startup
        """
        self.job_type_averages: Dict[str, float] = {}
        self.job_type_counts: Dict[str, int] = {}
        self.lock = threading.Lock()
        self.logger = get_logger()
        self._bootstrap_completed = False

        if enable_bootstrap:
            self._bootstrap_from_history()

    def _bootstrap_from_history(self) -> None:
        """
        Calculate averages from document_submissions.processing_time_seconds.

        Loads completed submissions with valid processing times
        and calculates average time per document type.
        """
        try:
            self.logger.info("Bootstrapping job timing averages from historical data...")

            query = """
                SELECT document_type, processing_time_seconds
                FROM document_submissions
                WHERE processing_time_seconds IS NOT NULL
                  AND processing_time_seconds > 0
                  AND result_status = TRUE
                  AND submitted_at > DATE_SUB(NOW(), INTERVAL 30 DAY)
            """

            totals: Dict[str, float] = {}
            counts: Dict[str, int] = {}

            with get_db_connection_context() as conn:
                with conn.cursor(dictionary=True) as cursor:
                    cursor.execute(query)
                    for row in cursor:
                        doc_type = row.get('document_type')
                        processing_time = row.get('processing_time_seconds')

                        if doc_type and processing_time:
                            doc_type = doc_type.lower()
                            totals[doc_type] = totals.get(doc_type, 0) + processing_time
                            counts[doc_type] = counts.get(doc_type, 0) + 1

            with self.lock:
                for doc_type in totals:
                    if counts[doc_type] > 0:
                        self.job_type_averages[doc_type] = totals[doc_type] / counts[doc_type]
                        self.job_type_counts[doc_type] = counts[doc_type]

            if self.job_type_averages:
                self.logger.info(
                    f"Bootstrapped timing averages for {len(self.job_type_averages)} document types: "
                    f"{self.job_type_averages}"
                )
            else:
                self.logger.info("No historical timing data found - starting with empty averages")

            self._bootstrap_completed = True

        except Exception as e:
            self.logger.error(f"Error bootstrapping job timing data: {e}")
            self._bootstrap_completed = False

    def get_average_time(self, document_type: str) -> Optional[float]:
        """
        Get average completion time for a document type.

        Args:
            document_type: Type of document (e.g., "passport", "bank_statement", "selfie")

        Returns:
            Average processing time in seconds, or None if no data available
        """
        with self.lock:
            # Normalize to lowercase for consistency
            normalized_type = document_type.lower()
            return self.job_type_averages.get(normalized_type)

    def record_completion(self, document_type: str, duration_seconds: float) -> None:
        """
        Update running average when a job completes.

        Uses incremental average update:
        new_avg = old_avg + (new_value - old_avg) / (count + 1)

        Args:
            document_type: Type of document that was processed
            duration_seconds: Actual processing time in seconds
        """
        if duration_seconds <= 0:
            self.logger.warning(f"Invalid processing time {duration_seconds} for {document_type}")
            return

        normalized_type = document_type.lower()

        with self.lock:
            old_avg = self.job_type_averages.get(normalized_type)
            old_count = self.job_type_counts.get(normalized_type, 0)

            if old_avg is None:
                # First data point for this type
                self.job_type_averages[normalized_type] = duration_seconds
                self.job_type_counts[normalized_type] = 1
                self.logger.info(
                    f"Initial timing data for {normalized_type}: {duration_seconds:.2f}s"
                )
            else:
                # Update running average incrementally
                new_avg = old_avg + (duration_seconds - old_avg) / (old_count + 1)
                self.job_type_averages[normalized_type] = new_avg
                self.job_type_counts[normalized_type] = old_count + 1

                self.logger.debug(
                    f"Updated timing average for {normalized_type}: "
                    f"{old_avg:.2f}s -> {new_avg:.2f}s (n={old_count + 1})"
                )

    def get_queue_position_time(
        self,
        pending_jobs: List
    ) -> Optional[float]:
        """
        Calculate expected wait time based on jobs ahead in queue.

        Sums the average processing time for each pending job ahead
        of the newly submitted job.

        Args:
            pending_jobs: List of pending JobDatabaseRecord objects or dicts

        Returns:
            Total expected wait time in seconds, or None if no timing data available
                    for any job type in the queue
        """
        if not pending_jobs:
            return 0.0

        total_time = 0.0
        has_valid_data = False

        with self.lock:
            for job in pending_jobs:
                # Handle both JobDatabaseRecord (Pydantic) and dict types
                if hasattr(job, 'request_data'):
                    # JobDatabaseRecord - extract from request_data
                    request_data = job.request_data
                else:
                    # Dict type
                    request_data = job

                # Extract document_type from request_data.files[0]
                doc_type = None
                if isinstance(request_data, dict):
                    files = request_data.get('files', [])
                    if files and len(files) > 0:
                        file_obj = files[0]
                        doc_type = file_obj.get('document_type') or file_obj.get('file_type')

                if doc_type:
                    doc_type = doc_type.lower()
                    avg_time = self.job_type_averages.get(doc_type)

                    if avg_time:
                        total_time += avg_time
                        has_valid_data = True
                    else:
                        self.logger.debug(f"No timing data for document type: {doc_type}")
                else:
                    self.logger.debug(f"No document_type found in job")

        return total_time if has_valid_data else None

    def get_all_averages(self) -> Dict[str, float]:
        """
        Get all current average times (for debugging/monitoring).

        Returns:
            Copy of the job_type_averages dictionary
        """
        with self.lock:
            return self.job_type_averages.copy()

    def reset(self) -> None:
        """Reset all timing statistics (for testing)."""
        with self.lock:
            self.job_type_averages.clear()
            self.job_type_counts.clear()
            self.logger.info("Job timing statistics reset")


# Global singleton instance
_job_timing_service: Optional[JobTimingService] = None
_service_lock = threading.Lock()


def get_job_timing_service() -> JobTimingService:
    """
    Get the global JobTimingService singleton instance.

    Returns:
        The shared JobTimingService instance
    """
    global _job_timing_service

    if _job_timing_service is None:
        with _service_lock:
            if _job_timing_service is None:
                _job_timing_service = JobTimingService()

    return _job_timing_service


def reset_job_timing_service() -> None:
    """Reset the global JobTimingService instance (for testing)."""
    global _job_timing_service

    with _service_lock:
        if _job_timing_service is not None:
            _job_timing_service.reset()
        _job_timing_service = None

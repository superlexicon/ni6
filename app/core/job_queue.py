import queue
import threading
import uuid
from typing import Optional, Dict, Any
from app.dto.job_models import JobDatabaseRecord
from app.core.logger import get_logger


class JobQueue:
    """Thread-safe job queue for in-memory processing"""

    def __init__(self):
        self._queue = queue.Queue()
        self._jobs = {}  # job_id -> JobDatabaseRecord for tracking
        self._lock = threading.RLock()
        self._logger = get_logger()
        self._processed_count = 0
        self._failed_count = 0

    def put(self, job: JobDatabaseRecord) -> None:
        """Add a job to the queue"""
        with self._lock:
            self._queue.put(job.id)
            self._jobs[job.id] = job
            self._logger.info(f"Added job {job.id} to queue. Queue size: {self._queue.qsize()}")

    def get(self, timeout: Optional[float] = None) -> Optional[str]:
        """Get a job ID from the queue (blocks until job available or timeout)"""
        try:
            job_id = self._queue.get(timeout=timeout)
            self._logger.info(f"Retrieved job {job_id} from queue. Queue size: {self._queue.qsize()}")
            return job_id
        except queue.Empty:
            return None

    def get_job_data(self, job_id: str) -> Optional[JobDatabaseRecord]:
        """Get job data by ID"""
        with self._lock:
            return self._jobs.get(job_id)

    def remove_job(self, job_id: str) -> None:
        """Remove job from tracking (called after successful completion)"""
        with self._lock:
            if job_id in self._jobs:
                del self._jobs[job_id]
                self._processed_count += 1
                self._logger.info(f"Removed completed job {job_id}. Processed: {self._processed_count}")

    def mark_job_failed(self, job_id: str) -> None:
        """Mark job as failed in tracking"""
        with self._lock:
            if job_id in self._jobs:
                self._failed_count += 1
                self._logger.info(f"Marked job {job_id} as failed. Failed: {self._failed_count}")

    def size(self) -> int:
        """Get current queue size"""
        return self._queue.qsize()

    def get_stats(self) -> Dict[str, int]:
        """Get queue statistics"""
        with self._lock:
            return {
                "queue_size": self._queue.qsize(),
                "tracked_jobs": len(self._jobs),
                "processed_count": self._processed_count,
                "failed_count": self._failed_count
            }

    def clear(self) -> None:
        """Clear the queue and all tracked jobs"""
        with self._lock:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

            self._jobs.clear()
            self._logger.info("Cleared job queue and tracking")

    def is_job_tracked(self, job_id: str) -> bool:
        """Check if a job is currently being tracked"""
        with self._lock:
            return job_id in self._jobs

    def get_all_tracked_jobs(self) -> Dict[str, JobDatabaseRecord]:
        """Get all currently tracked jobs"""
        with self._lock:
            return self._jobs.copy()

    def task_done(self) -> None:
        """Mark a task as done (for queue.Join functionality)"""
        self._queue.task_done()

    def join(self) -> None:
        """Wait until all items in the queue have been processed"""
        self._queue.join()

    def __len__(self) -> int:
        """Get queue length (alias for size)"""
        return self.size()
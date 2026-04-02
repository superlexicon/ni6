import threading
import time
import os
from typing import Optional
from app.core.logger import get_logger
from app.repositories.user_identity_repository import UserIdentityRepository

logger = get_logger()


class CleanupWorker:
    """
    Background worker that periodically cleans up abandoned records.

    With multi-device support, user_identity and user_keys are only created
    after selfie verification passes. The following can be abandoned:
    - OTP records (users who request OTP but never complete selfie submission)
    - Pending keys (users who store key data but never complete verification)

    Runs on a periodic schedule (default: every hour) to remove:
    - Expired OTP records
    - Old unverified OTP records (older than threshold)
    - Old pending keys (older than 24 hours)
    """

    def __init__(self, cleanup_interval_hours: int = 1):
        """
        Initialize the cleanup worker.

        Args:
            cleanup_interval_hours: How often to run cleanup (default: 1 hour)
        """
        self.user_identity_repo = UserIdentityRepository()
        self.cleanup_interval_hours = cleanup_interval_hours
        self.cleanup_interval_seconds = cleanup_interval_hours * 3600
        self.hours_threshold = int(os.getenv('CLEANUP_ABANDONED_HOURS', '168'))  # 7 days default

        self.running = False
        self.worker_thread = None
        self.logger = logger

    def start(self) -> None:
        """Start the background cleanup worker as daemon thread"""
        if self.running:
            self.logger.warning("Cleanup worker is already running")
            return

        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        self.logger.info(
            f"Cleanup worker started (interval: {self.cleanup_interval_hours}h, "
            f"threshold: {self.hours_threshold}h)"
        )

    def stop(self, timeout: int = 30) -> None:
        """Stop the background cleanup worker gracefully"""
        if not self.running:
            return

        self.logger.info("Stopping cleanup worker...")
        self.running = False

        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=timeout)
            if self.worker_thread.is_alive():
                self.logger.warning("Cleanup worker thread did not stop gracefully within timeout")
            else:
                self.logger.info("Cleanup worker stopped gracefully")

    def _worker_loop(self) -> None:
        """Main worker loop - runs cleanup periodically"""
        while self.running:
            try:
                # Run cleanup
                self._run_cleanup()

                # Wait for next interval (check running flag periodically)
                sleep_chunks = 60  # Check every minute
                sleep_time = self.cleanup_interval_seconds / sleep_chunks
                for _ in range(sleep_chunks):
                    if not self.running:
                        break
                    time.sleep(sleep_time)

            except Exception as e:
                self.logger.error(f"Error in cleanup worker loop: {e}")
                # Wait before retrying (avoid tight error loop)
                time.sleep(60)

        self.logger.info("Cleanup worker loop ended")

    def _run_cleanup(self) -> None:
        """Run the actual cleanup operation"""
        try:
            self.logger.info(f"Running cleanup (threshold: {self.hours_threshold}h)")
            result = self.user_identity_repo.cleanup_abandoned_flows(
                hours_old=self.hours_threshold
            )

            deleted_parts = []
            if result.get('otps', 0) > 0:
                deleted_parts.append(f"{result['otps']} OTPs")
            if result.get('pending_keys', 0) > 0:
                deleted_parts.append(f"{result['pending_keys']} pending keys")

            if deleted_parts:
                self.logger.info(f"Cleanup complete: {', '.join(deleted_parts)} deleted")
            else:
                self.logger.debug("Cleanup complete: no abandoned records found")

        except Exception as e:
            self.logger.error(f"Error running cleanup: {e}")

    def run_once(self) -> dict:
        """
        Run cleanup once (manual trigger).

        Returns:
            Dict with cleanup results: {'otps': int, 'pending_keys': int, 'identities': 0, 'keys': 0}
        """
        self.logger.info("Manual cleanup trigger")
        return self.user_identity_repo.cleanup_abandoned_flows(
            hours_old=self.hours_threshold
        )

    def is_healthy(self) -> bool:
        """Check if the worker is healthy and running"""
        return (
            self.running and
            self.worker_thread is not None and
            self.worker_thread.is_alive()
        )

    def get_stats(self) -> dict:
        """Get worker statistics"""
        return {
            "running": self.running,
            "thread_alive": self.worker_thread.is_alive() if self.worker_thread else False,
            "cleanup_interval_hours": self.cleanup_interval_hours,
            "cleanup_threshold_hours": self.hours_threshold
        }

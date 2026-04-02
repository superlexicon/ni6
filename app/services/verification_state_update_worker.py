import threading
import time
import os
from typing import Optional
from app.core.logger import get_logger
from app.repositories.user_identity_repository import UserIdentityRepository

logger = get_logger()


class VerificationStateUpdateWorker:
    """
    Background worker that periodically updates verification_state based on document expiry.

    Runs daily (24-hour interval) to:
    - Set state=2 for users with old bank statements (6+ months)
    - Set state=1 for users with expiring passports (within 6 months)
    """

    def __init__(self, interval_hours: int = 24):
        """
        Initialize the verification state update worker.

        Args:
            interval_hours: How often to run updates (default: 24 hours)
        """
        self.user_identity_repo = UserIdentityRepository()
        self.interval_hours = interval_hours
        self.interval_seconds = interval_hours * 3600

        self.running = False
        self.worker_thread = None
        self.logger = logger

        # Statistics tracking
        self._last_run_time = None
        self._last_run_result = None
        self._total_runs = 0

    def start(self) -> None:
        """Start the background update worker as daemon thread"""
        if self.running:
            self.logger.warning("Verification state update worker is already running")
            return

        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        self.logger.info(
            f"Verification state update worker started (interval: {self.interval_hours}h)"
        )

    def stop(self, timeout: int = 30) -> None:
        """Stop the background update worker gracefully"""
        if not self.running:
            return

        self.logger.info("Stopping verification state update worker...")
        self.running = False

        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=timeout)
            if self.worker_thread.is_alive():
                self.logger.warning("Verification state update worker thread did not stop gracefully within timeout")
            else:
                self.logger.info("Verification state update worker stopped gracefully")

    def _worker_loop(self) -> None:
        """Main worker loop - runs updates periodically"""
        while self.running:
            try:
                # Run update
                self._run_update()

                # Wait for next interval (check running flag periodically)
                sleep_chunks = 60  # Check every minute
                sleep_time = self.interval_seconds / sleep_chunks
                for _ in range(sleep_chunks):
                    if not self.running:
                        break
                    time.sleep(sleep_time)

            except Exception as e:
                self.logger.error(f"Error in verification state update worker loop: {e}")
                # Wait before retrying (avoid tight error loop)
                time.sleep(60)

        self.logger.info("Verification state update worker loop ended")

    def _run_update(self) -> None:
        """Run the actual verification state update operation"""
        try:
            self.logger.info("Running verification state update")
            result = self.user_identity_repo.update_verification_state_by_document_expiry()

            # Update statistics
            self._last_run_time = time.time()
            self._last_run_result = result
            self._total_runs += 1

            if result['state_to_2'] > 0 or result['state_to_1'] > 0:
                self.logger.info(
                    f"Verification state update complete: {result['state_to_2']} -> state 2, "
                    f"{result['state_to_1']} -> state 1"
                )
            else:
                self.logger.debug("Verification state update complete: no records needed updating")

        except Exception as e:
            self.logger.error(f"Error running verification state update: {e}")

    def run_once(self) -> dict:
        """
        Run update once (manual trigger).

        Returns:
            Dict with update results: {'state_to_2': int, 'state_to_1': int}
        """
        self.logger.info("Manual verification state update trigger")
        result = self.user_identity_repo.update_verification_state_by_document_expiry()

        # Update statistics
        self._last_run_time = time.time()
        self._last_run_result = result
        self._total_runs += 1

        return result

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
            "interval_hours": self.interval_hours,
            "total_runs": self._total_runs,
            "last_run_time": self._last_run_time,
            "last_run_result": self._last_run_result
        }

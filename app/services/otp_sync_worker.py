import threading
import time
from typing import Optional
from datetime import datetime
from rethinkdb import RethinkDB
from app.core.logger import get_logger
from app.core.db.rethinkdb_connection import RethinkDBConnection
from app.repositories.otp_repository import OTPRepository
from app.config.rethinkdb_config import rethinkdb_settings

logger = get_logger()
r = RethinkDB()


class OTPSyncWorker:
    """
    Background worker that listens to RethinkDB change stream and syncs
    OTP events to the local MariaDB database.
    """

    def __init__(
        self,
        otp_repository: OTPRepository,
        rethinkdb_connection: RethinkDBConnection,
        instance_id: Optional[str] = None
    ):
        self.otp_repository = otp_repository
        self.rethinkdb_connection = rethinkdb_connection
        self.instance_id = instance_id or rethinkdb_settings.instance_id
        self.db_name = rethinkdb_settings.db
        self.table_name = rethinkdb_settings.table
        self.logger = logger
        self.running = False
        self.worker_thread = None
        self._stream_connection = None
        self.reconnect_delay = 5  # seconds

    def start(self) -> None:
        """Start the background worker as daemon thread"""
        if self.running:
            self.logger.warning("OTP sync worker is already running")
            return

        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        self.logger.info(f"OTP sync worker started (instance: {self.instance_id[:8]}...)")

    def stop(self, timeout: int = 30) -> None:
        """Stop the background worker gracefully"""
        if not self.running:
            return

        self.logger.info("Stopping OTP sync worker...")
        self.running = False

        # Close the stream connection to unblock the cursor
        if self._stream_connection:
            try:
                self._stream_connection.close()
            except Exception:
                pass

        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=timeout)
            if self.worker_thread.is_alive():
                self.logger.warning("OTP sync worker thread did not stop gracefully within timeout")
            else:
                self.logger.info("OTP sync worker stopped gracefully")

    def _worker_loop(self) -> None:
        """Main worker loop - listens to RethinkDB change stream"""
        while self.running:
            try:
                # Create a new connection for the change stream
                self._stream_connection = self.rethinkdb_connection.new_connection()
                self.logger.info("Connected to RethinkDB change stream")

                # Subscribe to changes on the otp_events table
                cursor = r.db(self.db_name).table(self.table_name).changes(
                    include_initial=False,  # Don't replay existing data
                    include_types=True  # Include change type (add, change, remove)
                ).run(self._stream_connection)

                # Process changes as they arrive
                for change in cursor:
                    if not self.running:
                        break
                    self._process_change(change)

            except Exception as e:
                if self.running:
                    self.logger.error(f"RethinkDB change stream error: {e}")
                    self.logger.info(f"Reconnecting in {self.reconnect_delay} seconds...")
                    time.sleep(self.reconnect_delay)
            finally:
                # Clean up stream connection
                if self._stream_connection:
                    try:
                        self._stream_connection.close()
                    except Exception:
                        pass
                    self._stream_connection = None

        self.logger.info("OTP sync worker loop ended")

    def _process_change(self, change: dict) -> None:
        """Process a single change from the change stream"""
        try:
            change_type = change.get("type", "change")
            new_val = change.get("new_val")
            old_val = change.get("old_val")

            # Handle deletions
            if change_type == "remove" or (old_val and not new_val):
                if old_val:
                    mobile_number = old_val.get("mobile_number")
                    source_instance = old_val.get("instance_id", "unknown")

                    # Skip if this was our own deletion
                    if source_instance == self.instance_id:
                        return

                    self.logger.debug(
                        f"Received OTP deletion from instance {source_instance[:8]}... "
                        f"for mobile: {mobile_number}"
                    )
                    # Optionally handle deletion sync here if needed
                return

            # Handle insertions and updates
            if not new_val:
                return

            source_instance = new_val.get("instance_id", "")

            # Skip if this is our own event
            if source_instance == self.instance_id:
                self.logger.debug(f"Skipping self-generated event for mobile: {new_val.get('mobile_number')}")
                return

            # Sync to local MariaDB
            self._sync_to_local_db(new_val)

        except Exception as e:
            self.logger.error(f"Error processing change: {e}")

    def _sync_to_local_db(self, event_data: dict) -> None:
        """
        Sync OTP event to local MariaDB.

        Uses the flexible otp_data JSON field to dynamically sync all fields.
        This allows schema changes without requiring code updates.
        """
        try:
            mobile_number = event_data.get("mobile_number")
            if not mobile_number:
                self.logger.warning("Received OTP event without mobile_number, skipping")
                return

            # Get the otp_data from the flexible JSON field
            # This contains ALL fields from the source MariaDB (except mobile_number which is at top level)
            otp_data = event_data.get("otp_data", {}).copy()

            # If otp_data is empty, fallback to legacy format (backwards compatibility)
            if not otp_data:
                self.logger.debug("Using legacy event format (no otp_data field)")
                otp_data = {
                    "random_number": event_data.get("random_number", ""),
                    "otp_id": event_data.get("otp_id", ""),
                    "expires_at": event_data.get("expires_at"),
                    "delivery_method": event_data.get("delivery_method", "sms"),
                    "attempts": event_data.get("attempts", 0),
                    "max_attempts": event_data.get("max_attempts", 3),
                    "is_verified": event_data.get("is_verified", False),
                }
                if event_data.get("public_key"):
                    otp_data["public_key"] = event_data["public_key"]

            # Add mobile_number from top level (required for MariaDB operations)
            otp_data["mobile_number"] = mobile_number

            # Parse datetime fields if they're strings
            for datetime_field in ["expires_at", "created_at", "updated_at"]:
                if datetime_field in otp_data and isinstance(otp_data[datetime_field], str):
                    try:
                        otp_data[datetime_field] = datetime.fromisoformat(
                            otp_data[datetime_field].replace("Z", "+00:00")
                        )
                    except ValueError:
                        otp_data[datetime_field] = None

            # Check if OTP already exists for this mobile number
            existing = self.otp_repository.get_otp_by_mobile_number(mobile_number)
            source_instance = event_data.get('instance_id', 'unknown')[:8]

            if existing:
                # Update existing record with all fields from otp_data
                self.otp_repository.update_otp(mobile_number, otp_data)
                self.logger.info(
                    f"Synced OTP update from instance {source_instance}... "
                    f"for mobile: {mobile_number}"
                )
            else:
                # Create new record with all fields from otp_data
                self.otp_repository.create_otp(otp_data)
                self.logger.info(
                    f"Synced new OTP from instance {source_instance}... "
                    f"for mobile: {mobile_number}"
                )

        except Exception as e:
            self.logger.error(f"Failed to sync OTP to local database: {e}")

    def is_healthy(self) -> bool:
        """Check if the worker is healthy and running"""
        return self.running and self.worker_thread is not None and self.worker_thread.is_alive()

    def get_stats(self) -> dict:
        """Get worker statistics"""
        return {
            "running": self.running,
            "thread_alive": self.worker_thread.is_alive() if self.worker_thread else False,
            "instance_id": self.instance_id,
            "reconnect_delay": self.reconnect_delay,
            "stream_connected": self._stream_connection is not None and self._stream_connection.is_open() if self._stream_connection else False
        }

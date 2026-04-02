from typing import Optional
from rethinkdb import RethinkDB
from app.core.logger import get_logger
from app.core.db.rethinkdb_connection import RethinkDBConnection
from app.dto.otp_sync_event import OTPSyncEvent
from app.config.rethinkdb_config import rethinkdb_settings

logger = get_logger()
r = RethinkDB()


class OTPSyncService:
    """
    Service for publishing OTP events to RethinkDB for cross-instance synchronization.
    """

    def __init__(
        self,
        rethinkdb_connection: RethinkDBConnection,
        instance_id: Optional[str] = None
    ):
        self.rethinkdb_connection = rethinkdb_connection
        self.instance_id = instance_id or rethinkdb_settings.instance_id
        self.db_name = rethinkdb_settings.db
        self.table_name = rethinkdb_settings.table
        self.logger = logger

    def publish_otp_event(self, otp_data: dict) -> bool:
        """
        Publish an OTP creation/update event to RethinkDB.
        Called after successful MariaDB insertion/update.

        Args:
            otp_data: Dictionary containing OTP data (mobile_number, random_number, etc.)

        Returns:
            True if published successfully, False otherwise
        """
        try:
            # Create sync event from OTP data
            sync_event = OTPSyncEvent.from_otp_data(otp_data, self.instance_id)
            event_dict = sync_event.to_rethinkdb_dict()

            # Get connection and insert
            conn = self.rethinkdb_connection.get_connection()
            result = r.db(self.db_name).table(self.table_name).insert(
                event_dict,
                conflict="replace"  # Update if same mobile_number exists
            ).run(conn)

            inserted = result.get("inserted", 0)
            replaced = result.get("replaced", 0)

            if inserted > 0 or replaced > 0:
                self.logger.info(
                    f"OTP event published to RethinkDB: mobile={otp_data.get('mobile_number')}, "
                    f"instance={self.instance_id[:8]}..."
                )
                return True
            else:
                self.logger.warning(f"OTP event publish returned unexpected result: {result}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to publish OTP event to RethinkDB: {e}")
            return False

    def publish_otp_verification(self, mobile_number: str, is_verified: bool = True) -> bool:
        """
        Publish an OTP verification status update to RethinkDB.

        Args:
            mobile_number: The mobile number whose OTP was verified
            is_verified: Verification status

        Returns:
            True if published successfully, False otherwise
        """
        try:
            conn = self.rethinkdb_connection.get_connection()

            # Update the existing event for this mobile number
            # Update both top-level event_type and otp_data.is_verified for consistency
            result = r.db(self.db_name).table(self.table_name).filter(
                {"mobile_number": mobile_number}
            ).update({
                "event_type": "verify",
                "instance_id": self.instance_id,  # Track which instance verified
                "otp_data": r.row["otp_data"].merge({"is_verified": is_verified})
            }).run(conn)

            if result.get("replaced", 0) > 0:
                self.logger.info(
                    f"OTP verification status published: mobile={mobile_number}, verified={is_verified}"
                )
                return True
            return False

        except Exception as e:
            self.logger.error(f"Failed to publish OTP verification to RethinkDB: {e}")
            return False

    def publish_otp_deletion(self, mobile_number: str) -> bool:
        """
        Publish an OTP deletion event to RethinkDB.
        This removes the OTP record from RethinkDB.

        Args:
            mobile_number: The mobile number whose OTP should be deleted

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            conn = self.rethinkdb_connection.get_connection()

            result = r.db(self.db_name).table(self.table_name).filter(
                {"mobile_number": mobile_number}
            ).delete().run(conn)

            deleted = result.get("deleted", 0)
            if deleted > 0:
                self.logger.debug(f"OTP deletion published to RethinkDB: mobile={mobile_number}")
                return True
            return False

        except Exception as e:
            self.logger.error(f"Failed to publish OTP deletion to RethinkDB: {e}")
            return False

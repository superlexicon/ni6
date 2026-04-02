from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime, timezone


class OTPSyncEvent(BaseModel):
    """
    Model for OTP synchronization events sent via RethinkDB.

    Uses a flexible JSON `otp_data` field to store all OTP fields dynamically.
    This allows schema changes in MariaDB without requiring RethinkDB schema updates.
    """

    # Primary key for lookups
    mobile_number: str = Field(..., description="Mobile number the OTP was generated for (primary key)")

    # Instance tracking for self-event filtering
    instance_id: str = Field(..., description="ID of the instance that generated this OTP")

    # Flexible JSON field containing ALL OTP data from MariaDB
    otp_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Complete OTP data dict matching MariaDB schema (dynamic)"
    )

    # Event metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When the event was created")
    event_type: str = Field(default="create", description="Event type: create, update, verify, delete")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }

    def to_rethinkdb_dict(self) -> dict:
        """Convert to dictionary suitable for RethinkDB insertion."""
        # Serialize datetime fields in otp_data
        otp_data_serialized = {}
        for key, value in self.otp_data.items():
            if isinstance(value, datetime):
                otp_data_serialized[key] = value.isoformat()
            else:
                otp_data_serialized[key] = value

        return {
            "mobile_number": self.mobile_number,
            "instance_id": self.instance_id,
            "otp_data": otp_data_serialized,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "event_type": self.event_type
        }

    @classmethod
    def from_otp_data(cls, otp_data: dict, instance_id: str, event_type: str = "create") -> "OTPSyncEvent":
        """
        Create OTPSyncEvent from OTP repository data format.

        The otp_data dict is stored without mobile_number (which is at top level),
        allowing dynamic schema support without code changes.
        """
        mobile_number = otp_data.get("mobile_number", "")

        # Remove mobile_number from otp_data since it's at the top level
        otp_data_without_mobile = {k: v for k, v in otp_data.items() if k != "mobile_number"}

        return cls(
            mobile_number=mobile_number,
            instance_id=instance_id,
            otp_data=otp_data_without_mobile,  # Store without mobile_number
            event_type=event_type
        )

    @classmethod
    def from_rethinkdb_dict(cls, data: dict) -> "OTPSyncEvent":
        """Create OTPSyncEvent from RethinkDB document."""
        # Parse created_at if it's a string
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except ValueError:
                created_at = datetime.now(timezone.utc)

        # Parse datetime fields in otp_data
        otp_data = data.get("otp_data", {})
        if "expires_at" in otp_data and isinstance(otp_data["expires_at"], str):
            try:
                otp_data["expires_at"] = datetime.fromisoformat(
                    otp_data["expires_at"].replace("Z", "+00:00")
                )
            except ValueError:
                pass

        return cls(
            mobile_number=data.get("mobile_number", ""),
            instance_id=data.get("instance_id", ""),
            otp_data=otp_data,
            created_at=created_at or datetime.now(timezone.utc),
            event_type=data.get("event_type", "create")
        )

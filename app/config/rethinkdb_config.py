try:
    from pydantic import BaseSettings, Field
except ImportError:
    from pydantic_settings import BaseSettings
    from pydantic import Field
from typing import Optional
import uuid


class RethinkDBSettings(BaseSettings):
    """RethinkDB configuration settings for OTP synchronization and job queue"""

    # Connection Settings
    # Field names match env vars: RETHINKDB_HOST, RETHINKDB_PORT, etc.
    host: str = Field("localhost", description="RethinkDB server host")
    port: int = Field(28015, description="RethinkDB server port")
    db: str = Field("im_osint_sync", description="RethinkDB database name")
    table: str = Field("otp_events", description="RethinkDB table for OTP events")
    jobs_table: str = Field("document_analysis_jobs", description="RethinkDB table for async job queue")
    user: str = Field("", description="RethinkDB user (leave empty for no auth)")
    password: str = Field("", description="RethinkDB password (leave empty for no auth)")
    timeout: int = Field(5, description="Connection timeout in seconds")

    # Instance identification for filtering self-events (OTP sync)
    instance_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique instance identifier for self-event filtering"
    )

    # Instance identification for job queue (server public key)
    instance_public_key: Optional[str] = Field(
        None,
        description="Server public key for instance identification in job queue"
    )

    model_config = {
        "env_prefix": "RETHINKDB_",
        "env_file": ".env",
        "extra": "ignore"
    }


# Global settings instance
rethinkdb_settings = RethinkDBSettings()

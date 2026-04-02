"""
Instance configuration for HTTP-based inter-instance communication.

This configuration replaces RethinkDB for:
1. Job queue distribution across instances
2. OTP synchronization between instances

Each instance has:
- A unique identifier (SERVER_PUBLIC_KEY)
- A publicly accessible URL (INSTANCE_URL)
- A list of peer instance URLs for HTTP communication
"""

from typing import List, Optional, Dict, Any
import os
import json

try:
    from pydantic import BaseSettings, Field
except ImportError:
    from pydantic_settings import BaseSettings
    from pydantic import Field


class InstanceConfig(BaseSettings):
    """Configuration for HTTP-based inter-instance communication"""

    # This instance's identifier
    instance_public_key: str = Field(
        default='',
        description="This instance's public key for identification"
    )

    # This instance's publicly accessible URL
    instance_url: str = Field(
        default='http://localhost:12410',
        description="This instance's public URL for peer communication"
    )

    # Peer instances for HTTP communication (comma-separated string)
    # Example: http://instance2:12410,http://instance3:12410
    # Stored as string to avoid Pydantic Settings JSON parsing
    peer_instances_raw: str = Field(
        default='',
        description="Comma-separated list of peer instance URLs",
        alias='PEER_INSTANCES'
    )

    # Sync settings
    startup_sync_enabled_raw: str = Field(
        default='true',
        description="Enable startup synchronization from peer instances",
        alias='STARTUP_SYNC_ENABLED'
    )

    startup_sync_timeout: int = Field(
        default=30,
        description="Timeout for startup sync requests (seconds)"
    )

    request_timeout: int = Field(
        default=10,
        description="Timeout for peer HTTP requests (seconds)"
    )

    # Peer public keys mapping for signature-based authentication (JSON string)
    peer_public_keys_raw: str = Field(
        default='{}',
        description="JSON map of peer URLs to their public keys",
        alias='PEER_PUBLIC_KEYS'
    )

    model_config = {
        "env_file": ".env",
        "extra": "ignore",
        "populate_by_name": True  # Allow using both field name and alias
    }

    @property
    def peer_instances(self) -> List[str]:
        """Parse peer_instances from comma-separated string."""
        if not self.peer_instances_raw:
            return []
        # Try JSON first, then comma-separated
        if self.peer_instances_raw.startswith('['):
            try:
                return json.loads(self.peer_instances_raw)
            except json.JSONDecodeError:
                pass
        return [url.strip() for url in self.peer_instances_raw.split(',') if url.strip()]

    @property
    def peer_public_keys(self) -> Dict[str, str]:
        """Parse peer_public_keys from JSON string."""
        if not self.peer_public_keys_raw or self.peer_public_keys_raw == '{}':
            return {}
        try:
            return json.loads(self.peer_public_keys_raw)
        except json.JSONDecodeError:
            return {}

    @property
    def startup_sync_enabled(self) -> bool:
        """Parse startup_sync_enabled as boolean."""
        if isinstance(self.startup_sync_enabled_raw, bool):
            return self.startup_sync_enabled_raw
        return self.startup_sync_enabled_raw.lower() in ('true', '1', 'yes', 'on')

    def get_peer_urls(self) -> List[str]:
        """Get list of peer instance URLs, filtering out self"""
        return [url for url in self.peer_instances if url != self.instance_url]

    def has_peers(self) -> bool:
        """Check if this instance has any configured peers"""
        return len(self.get_peer_urls()) > 0

    def get_peer_public_key(self, peer_url: str) -> Optional[str]:
        """Get public key for a peer instance."""
        return self.peer_public_keys.get(peer_url)

    def is_known_peer(self, public_key: str) -> bool:
        """Check if a public key belongs to a known peer."""
        return public_key in self.peer_public_keys.values()

    def set_instance_public_key(self, public_key: str) -> None:
        """Set the instance public key (called after keypair generation)."""
        self.instance_public_key = public_key


# Global settings instance
instance_config = InstanceConfig()

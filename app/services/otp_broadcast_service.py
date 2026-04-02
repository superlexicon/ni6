"""
OTP Broadcast Service

Broadcasts OTP events to peer instances via HTTP.
Replaces RethinkDB OTP synchronization with a simpler HTTP-based approach.

This service:
1. Broadcasts OTP creation events to all peer instances
2. Broadcasts OTP verification events to all peer instances
3. Handles failures gracefully (fire-and-forget)
"""

import asyncio
from typing import Dict, Any, Optional
import httpx

from app.config.instance_config import instance_config
from app.core.logger import get_logger
from app.core.key.internal_signature import InternalSignature
from app.utils.json_serializer import serialize_datetime_dict

logger = get_logger()


class OTPBroadcastService:
    """
    Broadcast OTP events to peer instances via HTTP.

    This replaces RethinkDB changefeeds with direct HTTP calls to peers.
    Events are broadcast in parallel, with failures logged but not blocking.
    """

    def __init__(self):
        # instance_id is now a property that reads from instance_config dynamically
        self.logger = logger
        # Get server keys for signing
        self.server_public_key, self.server_private_key = InternalSignature.get_server_keys()

    @property
    def instance_id(self) -> str:
        """Get instance ID from instance_config dynamically."""
        return instance_config.instance_public_key[:16]

    async def broadcast_otp_created(self, otp_data: Dict[str, Any]) -> None:
        """
        Broadcast new OTP to all peer instances.

        Args:
            otp_data: OTP data dict containing mobile_number, random_number, etc.
        """
        await self._broadcast_event({
            "event_type": "create",
            "instance_id": self.instance_id,
            "otp_data": otp_data
        })

    async def broadcast_otp_verified(self, mobile_number: str) -> None:
        """
        Broadcast OTP verification to all peer instances.

        Args:
            mobile_number: Mobile number whose OTP was verified
        """
        await self._broadcast_event({
            "event_type": "verify",
            "instance_id": self.instance_id,
            "mobile_number": mobile_number
        })

    async def broadcast_otp_deleted(self, mobile_number: str) -> None:
        """
        Broadcast OTP deletion to all peer instances.

        Args:
            mobile_number: Mobile number whose OTP was deleted
        """
        await self._broadcast_event({
            "event_type": "delete",
            "instance_id": self.instance_id,
            "mobile_number": mobile_number
        })

    async def _broadcast_event(self, event: Dict[str, Any]) -> None:
        """
        Send event to all peer instances (fire-and-forget).

        Args:
            event: Event dict to broadcast
        """
        peer_urls = instance_config.get_peer_urls()

        if not peer_urls:
            self.logger.debug("No peer instances configured for broadcast")
            return

        async with httpx.AsyncClient(timeout=5.0) as client:
            tasks = []
            for peer_url in peer_urls:
                tasks.append(self._send_to_peer(client, peer_url, event))

            # Run all requests in parallel, don't wait for failures
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Log any failures
            for peer_url, result in zip(peer_urls, results):
                if isinstance(result, Exception):
                    self.logger.warning(f"Failed to broadcast to {peer_url}: {result}")

    async def _send_to_peer(
        self,
        client: httpx.AsyncClient,
        peer_url: str,
        event: Dict[str, Any]
    ) -> None:
        """
        Send event to single peer (with error handling).

        Args:
            client: HTTP client
            peer_url: Peer instance URL
            event: Event to send
        """
        try:
            # Serialize datetime objects to JSON-compatible format
            json_event = serialize_datetime_dict(event)

            # Create signature headers
            headers = InternalSignature.create_signature_headers(
                self.server_private_key,
                self.server_public_key
            )

            response = await client.post(
                f"{peer_url}/api/internal/otp/sync",
                json=json_event,  # Use serialized event
                timeout=5.0,
                headers=headers
            )
            if response.status_code == 200:
                self.logger.debug(f"OTP event broadcasted to {peer_url}")
            else:
                self.logger.warning(
                    f"OTP event broadcast to {peer_url} failed: status {response.status_code}"
                )
        except Exception as e:
            self.logger.warning(f"Failed to broadcast OTP event to {peer_url}: {e}")
            raise  # Re-raise for logging in gather


# Singleton instance
otp_broadcast_service = OTPBroadcastService()

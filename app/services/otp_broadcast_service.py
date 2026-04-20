"""
OTP Broadcast Service - Simplified (Public Key Only)

Broadcasts OTP events to peer instances via HTTP.
"""

import asyncio
from typing import Dict, Any
import httpx

from app.config.instance_config import instance_config
from app.core.logger import get_logger
from app.core.key.internal_signature import InternalSignature
from app.utils.json_serializer import serialize_datetime_dict

logger = get_logger()


class OTPBroadcastService:
    """Broadcast OTP events to peer instances via HTTP."""

    def __init__(self):
        self.logger = logger
        self.server_public_key, self.server_private_key = InternalSignature.get_server_keys()

    @property
    def instance_id(self) -> str:
        return instance_config.instance_public_key[:16]

    async def broadcast_otp_created(self, otp_data: Dict[str, Any]) -> None:
        """Broadcast new OTP to all peer instances."""
        self.logger.info(f"📤 Broadcasting OTP created: public_key={otp_data.get('public_key', '')[:16]}..., random_number={otp_data.get('random_number')}")
        await self._broadcast_event({
            "event_type": "create",
            "instance_id": self.instance_id,
            "otp_data": otp_data
        })

    async def broadcast_otp_verified(self, public_key: str) -> None:
        """Broadcast OTP verification to all peer instances."""
        self.logger.info(f"📤 Broadcasting OTP verified: public_key={public_key[:16]}...")
        await self._broadcast_event({
            "event_type": "verify",
            "instance_id": self.instance_id,
            "public_key": public_key
        })

    async def broadcast_otp_deleted(self, public_key: str) -> None:
        """Broadcast OTP deletion to all peer instances."""
        self.logger.info(f"📤 Broadcasting OTP deleted: public_key={public_key[:16]}...")
        await self._broadcast_event({
            "event_type": "delete",
            "instance_id": self.instance_id,
            "public_key": public_key
        })

    async def _broadcast_event(self, event: Dict[str, Any]) -> None:
        """Send event to all peer instances (fire-and-forget)."""
        peer_urls = instance_config.get_peer_urls()

        self.logger.info(f"📡 Broadcasting to {len(peer_urls)} peer URLs: {peer_urls}")

        if not peer_urls:
            self.logger.warning("⚠️ No peer instances configured for broadcast - OTP will not be synced to other nodes")
            return

        async with httpx.AsyncClient(timeout=5.0) as client:
            tasks = []
            for peer_url in peer_urls:
                tasks.append(self._send_to_peer(client, peer_url, event))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for peer_url, result in zip(peer_urls, results):
                if isinstance(result, Exception):
                    self.logger.warning(f"Failed to broadcast to {peer_url}: {result}")

    async def _send_to_peer(
        self,
        client: httpx.AsyncClient,
        peer_url: str,
        event: Dict[str, Any]
    ) -> None:
        """Send event to single peer."""
        try:
            json_event = serialize_datetime_dict(event)
            self.logger.info(f"🔌 Broadcasting to {peer_url}: event_type={event.get('event_type')}")

            headers = InternalSignature.create_signature_headers(
                self.server_private_key,
                self.server_public_key
            )

            response = await client.post(
                f"{peer_url}/api/internal/otp/sync",
                json=json_event,
                timeout=5.0,
                headers=headers
            )
            if response.status_code == 200:
                self.logger.info(f"✅ OTP event broadcasted successfully to {peer_url}")
            else:
                self.logger.error(f"❌ OTP event broadcast to {peer_url} failed: status {response.status_code}")
        except Exception as e:
            self.logger.error(f"❌ Failed to broadcast OTP event to {peer_url}: {type(e).__name__}: {e}")
            raise


# Singleton instance
otp_broadcast_service = OTPBroadcastService()

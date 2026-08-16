"""
Job Broadcast Service

Broadcasts job lifecycle events (created / result / failed) to peer instances
via signed HTTP, mirroring the OTP broadcast design:

1. When a processing origin receives a client document, it pushes a shadow
   copy of the job record to peers (they store it unprocessed).
2. When the origin completes the job, it pushes the result; peers store the
   submission and delete their shadow row.
3. If the job fails permanently, peers mark their shadow row failed.

Delivery is fire-and-forget: peers that are down are caught up by the startup
recovery pull (shadow_job_recovery_service) and the shadow TTL safety net.
"""

import asyncio
import threading
from typing import Any, Dict, Optional

import httpx

from app.config.instance_config import instance_config
from app.core.logger import get_logger
from app.core.key.internal_signature import InternalSignature
from app.utils.json_serializer import serialize_datetime_dict

logger = get_logger()


class JobBroadcastService:
    """Broadcast job events to peer instances via HTTP."""

    def __init__(self):
        self.logger = logger
        self.server_public_key, self.server_private_key = InternalSignature.get_server_keys()

    @property
    def instance_id(self) -> str:
        return instance_config.instance_public_key[:16]

    # ------------------------------------------------------------------
    # Public fire-and-forget entry points.
    # Safe to call from async endpoints AND from the synchronous worker
    # thread: each call spawns a daemon thread running its own event loop,
    # so the caller never blocks on peer HTTP.
    # ------------------------------------------------------------------

    def broadcast_job_created(
        self,
        job_id: str,
        request_data_stripped: Dict[str, Any],
        client_public_key: Optional[str] = None,
        user_identity_id: Optional[str] = None,
        callback_url: Optional[str] = None
    ) -> None:
        """
        Push a shadow copy of a newly created job to all peers.

        Not gated on the LLM role: LLM-dependent jobs are silently dropped
        before creation on shadow instances, so anything created here is
        processable by this instance - and non-LLM jobs (selfie liveness,
        key recovery) processed on shadows must replicate to peers like any
        other job so all instances converge.
        """
        if not instance_config.has_peers():
            self.logger.debug("No peer instances configured - skipping job_created broadcast")
            return

        event = {
            "event_type": "job_created",
            "instance_id": self.instance_id,
            "processing_server": instance_config.instance_url,
            "job": {
                "id": job_id,
                "status": "pending",
                "request_data": request_data_stripped,
                "callback_url": callback_url,
                "client_public_key": client_public_key,
                "user_identity_id": user_identity_id
            }
        }
        self._spawn(self._broadcast_event(event, "/api/internal/jobs/sync"))
        self.logger.info(f"Broadcasting job_created for job {job_id} to peers")

    def broadcast_job_result(
        self,
        job_id: str,
        response_data: Dict[str, Any],
        request_data_decrypted_stripped: Dict[str, Any],
        user_key_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Push a completed job's result to all peers.

        Not gated on the LLM role: whichever instance processed the job must
        propagate its result, even if its role was flipped mid-flight.
        """
        if not instance_config.has_peers():
            self.logger.debug("No peer instances configured - skipping job_result broadcast")
            return

        event = {
            "event_type": "job_result",
            "instance_id": self.instance_id,
            "job_id": job_id,
            "response_data": response_data,
            "request_data": request_data_decrypted_stripped,
            "user_key_info": user_key_info or {}
        }
        self._spawn(self._broadcast_event(event, "/api/internal/jobs/sync"))
        self.logger.info(f"Broadcasting job_result for job {job_id} to peers")

    def broadcast_job_failed(self, job_id: str, error_message: str) -> None:
        """Push a permanent job failure to all peers so they mark their shadow row failed."""
        if not instance_config.has_peers():
            self.logger.debug("No peer instances configured - skipping job_failed broadcast")
            return

        event = {
            "event_type": "job_failed",
            "instance_id": self.instance_id,
            "job_id": job_id,
            "error_message": error_message
        }
        self._spawn(self._broadcast_event(event, "/api/internal/jobs/sync"))
        self.logger.info(f"Broadcasting job_failed for job {job_id} to peers")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _spawn(self, coro) -> None:
        """Run a broadcast coroutine in a daemon thread (fire-and-forget)."""
        def _run():
            try:
                asyncio.run(coro)
            except Exception as e:
                self.logger.error(f"Job broadcast task failed: {type(e).__name__}: {e}")
        threading.Thread(target=_run, daemon=True, name="JobBroadcast").start()

    async def _broadcast_event(self, event: Dict[str, Any], path: str) -> None:
        """Send an event to all peer instances."""
        peer_urls = instance_config.get_peer_urls()
        if not peer_urls:
            return

        timeout = float(instance_config.job_replication_timeout)
        async with httpx.AsyncClient(timeout=timeout) as client:
            tasks = [self._send_to_peer(client, peer_url, event, path) for peer_url in peer_urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for peer_url, result in zip(peer_urls, results):
                if isinstance(result, Exception):
                    self.logger.warning(f"Failed to broadcast job event to {peer_url}: {result}")

    async def _send_to_peer(
        self,
        client: httpx.AsyncClient,
        peer_url: str,
        event: Dict[str, Any],
        path: str
    ) -> None:
        """Send an event to a single peer."""
        json_event = serialize_datetime_dict(event)

        headers = InternalSignature.create_signature_headers(
            self.server_private_key,
            self.server_public_key
        )

        response = await client.post(
            f"{peer_url}{path}",
            json=json_event,
            timeout=float(instance_config.job_replication_timeout),
            headers=headers
        )
        if response.status_code == 200:
            self.logger.debug(f"Job event broadcast to {peer_url}: event_type={event.get('event_type')}")
        else:
            self.logger.error(
                f"Job event broadcast to {peer_url} failed: "
                f"status {response.status_code} event_type={event.get('event_type')}"
            )


# Singleton instance
job_broadcast_service = JobBroadcastService()

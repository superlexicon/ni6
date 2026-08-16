"""
Shadow Job Recovery Service

Startup recovery for replicated (shadow) job rows: pending rows whose
processing_server is set are resolved by asking that server for the job's
status via POST /api/internal/jobs/status.

Possible outcomes per job:
- completed : the origin finished but the result push was missed (or this
              instance was down) - finalize locally exactly like the
              job_result push would have (submission + user_keys upsert +
              shadow row deletion).
- pending / processing : still running on the origin - keep the shadow row.
- failed    : mark the shadow row failed.
- unknown   : the origin has neither the job nor a submission - the job was
              dropped. Mark the shadow row failed; the client must resubmit
              (peers cannot reprocess: shadow payloads are stripped).

Runs on every instance that has shadow rows, regardless of LLM role.
"""

import asyncio
import threading
from collections import defaultdict
from typing import Dict, List

import httpx

from app.config.instance_config import instance_config
from app.core.logger import get_logger
from app.core.key.internal_signature import InternalSignature
from app.repositories.job_repository import JobRepository

logger = get_logger()


class ShadowJobRecoveryService:
    """Resolve pending shadow rows by pulling job status from their processing server."""

    def __init__(self):
        self.job_repo = JobRepository()
        self._thread = None
        self.logger = logger
        self.server_public_key, self.server_private_key = InternalSignature.get_server_keys()

    def start(self) -> None:
        """Start recovery in a background thread (non-blocking)."""
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="ShadowJobRecovery"
        )
        self._thread.start()
        self.logger.info("Shadow job recovery thread started")

    def _run(self) -> None:
        try:
            asyncio.run(self._recover())
        except Exception as e:
            self.logger.error(f"Shadow job recovery failed: {e}", exc_info=True)

    async def _recover(self) -> None:
        shadow_jobs = self.job_repo.get_pending_replicated_jobs(limit=500)
        if not shadow_jobs:
            self.logger.info("Shadow job recovery: no pending replicated jobs")
            return

        # Group by the server that owns each job so one HTTP call resolves many
        by_server: Dict[str, List[str]] = defaultdict(list)
        for job in shadow_jobs:
            by_server[job.processing_server].append(job.id)

        self.logger.info(
            f"Shadow job recovery: resolving {len(shadow_jobs)} job(s) across "
            f"{len(by_server)} processing server(s)"
        )

        async with httpx.AsyncClient(
            timeout=float(instance_config.job_replication_timeout)
        ) as client:
            tasks = [
                self._recover_from_server(client, server, job_ids)
                for server, job_ids in by_server.items()
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _recover_from_server(
        self,
        client: httpx.AsyncClient,
        server_url: str,
        job_ids: List[str]
    ) -> None:
        from app.services.replication_handlers import (
            handle_job_result_event, handle_job_failed_event
        )

        headers = InternalSignature.create_signature_headers(
            self.server_private_key,
            self.server_public_key
        )

        try:
            response = await client.post(
                f"{server_url.rstrip('/')}/api/internal/jobs/status",
                json={"job_ids": job_ids},
                headers=headers,
                timeout=float(instance_config.job_replication_timeout)
            )
            if response.status_code != 200:
                self.logger.warning(
                    f"Shadow job recovery: {server_url} returned status "
                    f"{response.status_code} - shadow rows kept for retry"
                )
                return

            results = response.json().get("jobs", [])
        except Exception as e:
            self.logger.warning(
                f"Shadow job recovery: failed to reach {server_url} "
                f"({type(e).__name__}: {e}) - shadow rows kept for retry"
            )
            return

        for entry in results:
            job_id = entry.get("job_id")
            state = entry.get("state")

            if state == "completed":
                result = entry.get("result") or {}
                outcome = handle_job_result_event({
                    "job_id": job_id,
                    "response_data": result.get("response_data") or {},
                    "request_data": result.get("request_data") or {},
                    "user_key_info": {}
                })
                self.logger.info(
                    f"Shadow job recovery: job {job_id} finalized from {server_url}: {outcome.get('status')}"
                )
            elif state == "failed":
                handle_job_failed_event({
                    "job_id": job_id,
                    "error_message": f"Job failed on processing server {server_url}"
                })
            elif state == "unknown":
                handle_job_failed_event({
                    "job_id": job_id,
                    "error_message": (
                        "Dropped: processing server has no record of this job - resubmit"
                    )
                })
            else:
                # pending / processing on the origin - result push still expected
                self.logger.debug(f"Shadow job recovery: job {job_id} still {state} on {server_url}")

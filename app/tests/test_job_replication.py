"""
Unit tests for peer job replication + result sharing.

Covers:
- LLM role detection (LLM_API_URL presence -> origin vs shadow-only)
- Replication event handlers (job_created / job_result / job_failed)
- Peer submission creation (identity/state taken from origin, not local tables)
- Role-aware startup enqueue (shadow rows never enqueued locally)
- Broadcast gating (no peers / no LLM -> no-op)

All database access is mocked; no MySQL or HTTP is required.
"""

import unittest
import sys
import os
from datetime import datetime
from unittest.mock import MagicMock, patch, call

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestLLMRoleDetection(unittest.TestCase):
    """Role detection: LLM_API_URL set -> origin; unset/empty -> shadow-only."""

    def test_class_defaults_are_empty(self):
        """The OpenAI fallback defaults must stay removed (unset = shadow-only)."""
        from app.config.llm_config import LLMSettings

        # app package import loads .env into os.environ; isolate both sources
        saved = {}
        for var in ("LLM_API_URL", "LLM_MODEL"):
            if var in os.environ:
                saved[var] = os.environ.pop(var)
        try:
            s = LLMSettings(_env_file=None)
            self.assertEqual(s.api_url, "")
            self.assertEqual(s.model, "")
        finally:
            os.environ.update(saved)

    def test_configured_when_url_set(self):
        from app.config.llm_config import llm_settings, is_llm_server_configured

        with patch.object(llm_settings, "api_url", "http://gpu-host:1177/v1"):
            self.assertTrue(is_llm_server_configured())

    def test_not_configured_when_empty_or_whitespace(self):
        from app.config.llm_config import llm_settings, is_llm_server_configured

        for value in ("", "   ", None):
            with patch.object(llm_settings, "api_url", value):
                self.assertFalse(
                    is_llm_server_configured(),
                    f"api_url={value!r} must mean shadow-only",
                )


class TestReplicationHandlers(unittest.TestCase):
    """Event handlers used by /api/internal/jobs/sync and the recovery pull."""

    def _handler_module(self):
        from app.services import replication_handlers
        return replication_handlers

    def test_job_created_stores_shadow_row(self):
        from app.repositories.job_repository import JobRepository
        handlers = self._handler_module()

        with patch.object(JobRepository, "create_replicated_job", return_value=True) as m:
            result = handlers.handle_job_created_event({
                "processing_server": "http://origin:12410",
                "job": {
                    "id": "job-1",
                    "request_data": {"client_public_key": "pk"},
                    "client_public_key": "pk",
                    "callback_url": None,
                },
            })
        self.assertEqual(result["status"], "created")
        self.assertEqual(
            m.call_args.kwargs["processing_server"], "http://origin:12410"
        )

    def test_job_created_skips_missing_fields(self):
        handlers = self._handler_module()
        result = handlers.handle_job_created_event({"job": {"id": "job-1"}})
        self.assertEqual(result["status"], "skipped")

    def test_job_result_idempotent_when_submission_exists(self):
        from app.repositories.job_repository import JobRepository
        from app.repositories.document_submission_repository import DocumentSubmissionRepository
        handlers = self._handler_module()

        with patch.object(
            DocumentSubmissionRepository, "get_submission_by_job_id", return_value={"id": "s1"}
        ), patch.object(
            DocumentSubmissionRepository, "create_submission_from_peer"
        ) as create_mock, patch.object(JobRepository, "delete_job", return_value=True) as del_mock:
            result = handlers.handle_job_result_event({
                "job_id": "job-1",
                "response_data": {"result": True},
                "request_data": {},
                "user_key_info": {},
            })

        self.assertEqual(result["status"], "exists")
        create_mock.assert_not_called()
        del_mock.assert_called_once_with("job-1")

    def test_job_result_creates_submission_and_deletes_shadow(self):
        from app.repositories.job_repository import JobRepository
        from app.repositories.document_submission_repository import DocumentSubmissionRepository
        handlers = self._handler_module()

        with patch.object(
            DocumentSubmissionRepository, "get_submission_by_job_id", return_value=None
        ), patch.object(
            DocumentSubmissionRepository,
            "create_submission_from_peer",
            return_value=(True, ""),
        ) as create_mock, patch.object(JobRepository, "delete_job", return_value=True) as del_mock, \
             patch.object(handlers, "_upsert_user_key_info") as upsert_mock:
            result = handlers.handle_job_result_event({
                "job_id": "job-1",
                "response_data": {"result": True, "user_identity_id": "ident-1"},
                "request_data": {"client_public_key": "pk"},
                "user_key_info": {
                    "client_public_key": "pk",
                    "user_identity_id": "ident-1",
                    "verification_state": 2,
                    "sequence_no": 2,
                },
            })

        self.assertEqual(result["status"], "created")
        create_mock.assert_called_once()
        del_mock.assert_called_once_with("job-1")
        upsert_mock.assert_called_once()

    def test_job_result_keeps_shadow_on_storage_error(self):
        from app.repositories.job_repository import JobRepository
        from app.repositories.document_submission_repository import DocumentSubmissionRepository
        handlers = self._handler_module()

        with patch.object(
            DocumentSubmissionRepository, "get_submission_by_job_id", return_value=None
        ), patch.object(
            DocumentSubmissionRepository,
            "create_submission_from_peer",
            return_value=(False, "db error"),
        ), patch.object(JobRepository, "delete_job", return_value=True) as del_mock:
            result = handlers.handle_job_result_event({
                "job_id": "job-1",
                "response_data": {},
                "request_data": {},
                "user_key_info": {},
            })

        self.assertEqual(result["status"], "error")
        del_mock.assert_not_called()

    def test_job_failed_marks_replica(self):
        from app.repositories.job_repository import JobRepository
        handlers = self._handler_module()

        with patch.object(JobRepository, "mark_job_failed_replica", return_value=True) as m:
            result = handlers.handle_job_failed_event({
                "job_id": "job-1",
                "error_message": "boom",
            })
        self.assertEqual(result["status"], "failed")
        m.assert_called_once_with("job-1", "boom")

    def test_user_key_upsert_uses_origin_values(self):
        from app.repositories.user_key_repository import UserKeyRepository
        handlers = self._handler_module()

        with patch.object(
            UserKeyRepository, "get_key_by_public_key",
            return_value={"user_identity_id": "ident-old"},
        ) as get_mock, patch.object(
            UserKeyRepository, "update_key_by_public_key", return_value=True
        ) as update_mock, patch.object(
            UserKeyRepository, "update_state_and_sequence", return_value=True
        ) as state_mock, patch.object(
            UserKeyRepository, "create_key"
        ) as create_mock:
            handlers._upsert_user_key_info(
                {"client_public_key": "pk", "user_identity_id": "ident-new",
                 "verification_state": 1, "sequence_no": 2},
                {"user_identity_id": "ident-new"},
                {"client_public_key": "pk"},
            )

        get_mock.assert_called_once_with("pk")
        create_mock.assert_not_called()
        update_mock.assert_called_once_with("pk", {"user_identity_id": "ident-new"})
        state_mock.assert_called_once_with("pk", 1, 2)

    def test_user_key_upsert_creates_missing_row(self):
        from app.repositories.user_key_repository import UserKeyRepository
        handlers = self._handler_module()

        with patch.object(
            UserKeyRepository, "get_key_by_public_key", return_value=None,
        ), patch.object(
            UserKeyRepository, "create_key", return_value={"id": "k"}
        ) as create_mock, patch.object(
            UserKeyRepository, "update_state_and_sequence", return_value=True
        ) as state_mock:
            handlers._upsert_user_key_info(
                {"client_public_key": "pk", "user_identity_id": "ident-1",
                 "verification_state": 0, "sequence_no": 0},
                {},
                {"client_public_key": "pk"},
            )

        create_mock.assert_called_once()
        self.assertEqual(create_mock.call_args.args[0]["user_public_key"], "pk")
        self.assertEqual(create_mock.call_args.args[0]["user_identity_id"], "ident-1")
        state_mock.assert_called_once_with("pk", 0, 0)

    def test_user_key_upsert_skips_invalid_state(self):
        from app.repositories.user_key_repository import UserKeyRepository
        handlers = self._handler_module()

        with patch.object(
            UserKeyRepository, "get_key_by_public_key", return_value=None,
        ), patch.object(
            UserKeyRepository, "create_key", return_value={"id": "k"}
        ), patch.object(
            UserKeyRepository, "update_state_and_sequence"
        ) as state_mock:
            handlers._upsert_user_key_info(
                {"client_public_key": "pk"},
                {},
                {"client_public_key": "pk"},
            )

        state_mock.assert_not_called()


class TestCreateSubmissionFromPeer(unittest.TestCase):
    """Peer submission creation: origin values, envelope pass-through."""

    def _run_insert(self, response_data, request_data, existing_row=None):
        """Run create_submission_from_peer against a fake DB, return captured params."""
        from app.repositories.document_submission_repository import DocumentSubmissionRepository
        from app.core.db import database

        captured = {}

        class FakeCursor:
            def __init__(self, *a, **k):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
            def execute(self, query, params=None):
                captured["query"] = query
                captured["params"] = params

        class FakeConn:
            def cursor(self, *a, **k):
                return FakeCursor()
            def commit(self):
                pass

        class FakeCtx:
            def __enter__(self):
                return FakeConn()
            def __exit__(self, *a):
                return False

        repo = DocumentSubmissionRepository()
        with patch.object(database, "get_db_connection_context", return_value=FakeCtx()), \
             patch.object(DocumentSubmissionRepository, "get_submission_by_job_id",
                          return_value=existing_row):
            ok, err = repo.create_submission_from_peer(response_data, request_data, "job-1")
        return ok, err, captured

    def test_fields_come_from_origin_not_local_tables(self):
        ok, err, captured = self._run_insert(
            response_data={
                "user_identity_id": "origin-identity",
                "verification_state": 2,
                "sequence_no": 3,
                "result": True,
                "processing_time_seconds": 1.5,
                "docs_auth_score": 90.0,
            },
            request_data={
                "client_public_key": "pk",
                "files": [{"filename": "passport.jpg", "document_type": "passport"}],
            },
        )
        self.assertTrue(ok, err)
        params = captured["params"]
        self.assertEqual(params["user_identity_id"], "origin-identity")
        self.assertEqual(params["verification_state"], 2)  # NOT re-read from local user_keys
        self.assertEqual(params["sequence_no"], 3)
        self.assertEqual(params["document_type"], "passport")
        self.assertEqual(params["filename"], "passport.jpg")
        self.assertEqual(params["job_id"], "job-1")
        self.assertTrue(params["result_status"])

    def test_envelope_pass_through(self):
        """Recovery path: origin's ECIES envelope is stored verbatim (no plaintext)."""
        envelope = {"version": "ECIES-v1", "ciphertext": "abc"}
        ok, err, captured = self._run_insert(
            response_data={
                "user_identity_id": "ident",
                "extracted_data_encrypted": envelope,
            },
            request_data={"client_public_key": "pk", "files": []},
        )
        self.assertTrue(ok, err)
        import json as _json
        stored = _json.loads(captured["params"]["extracted_data_encrypted"])
        self.assertEqual(stored, {"version": "ECIES-v1", "ciphertext": "abc"})

    def test_error_message_stored(self):
        ok, err, captured = self._run_insert(
            response_data={"result": False, "error": "extraction failed"},
            request_data={"client_public_key": "pk", "files": []},
        )
        self.assertTrue(ok, err)
        self.assertFalse(captured["params"]["result_status"])
        self.assertEqual(captured["params"]["error_message"], "extraction failed")


class TestRoleAwareStartup(unittest.TestCase):
    """load_pending_jobs_on_startup: shadow rows never enqueued; demotion fails leftovers."""

    def _make_job(self, job_id, processing_server=None):
        from app.dto.job_models import JobDatabaseRecord, JobStatus
        return JobDatabaseRecord(
            id=job_id,
            status=JobStatus.PENDING,
            request_data={"target_server_public_key": "route-key"},
            created_at=datetime.utcnow(),
            processing_server=processing_server,
        )

    def _make_manager(self, pending_jobs):
        from app.services.job_manager import JobManager

        queue = MagicMock()
        queue.is_job_tracked.return_value = False
        queue.put = MagicMock()

        manager = JobManager.__new__(JobManager)  # skip __init__ (DB connection)
        manager.job_repo = MagicMock()
        manager.job_repo.fail_stale_replicated_jobs.return_value = 0
        manager.job_repo.reset_stale_jobs.return_value = 0
        manager.job_repo.get_pending_jobs.return_value = pending_jobs
        manager.job_queue = queue
        manager._worker = None
        manager.logger = MagicMock()
        return manager, queue

    def test_origin_skips_shadow_rows(self):
        from app.services import job_manager as jm_module

        own = self._make_job("own-1")
        shadow = self._make_job("shadow-1", processing_server="http://origin:12410")
        manager, queue = self._make_manager([own, shadow])

        with patch.object(jm_module, "is_llm_server_configured", return_value=True):
            loaded = manager.load_pending_jobs_on_startup()

        self.assertEqual(loaded, 1)
        self.assertEqual(queue.put.call_count, 1)
        self.assertEqual(queue.put.call_args.args[0].id, "own-1")

    def test_shadow_only_instance_enqueues_nothing_and_fails_own_jobs(self):
        from app.services import job_manager as jm_module
        from app.dto.job_models import JobStatus

        own = self._make_job("own-1")
        shadow = self._make_job("shadow-1", processing_server="http://origin:12410")
        manager, queue = self._make_manager([own, shadow])
        manager.update_job_status = MagicMock(return_value=True)

        with patch.object(jm_module, "is_llm_server_configured", return_value=False):
            loaded = manager.load_pending_jobs_on_startup()

        self.assertEqual(loaded, 0)
        queue.put.assert_not_called()
        # Own leftover job failed loudly; shadow row left for the recovery pull
        self.assertEqual(manager.update_job_status.call_count, 1)
        args = manager.update_job_status.call_args.args
        self.assertEqual(args[0], "own-1")
        self.assertEqual(args[1], JobStatus.FAILED)


class TestBroadcastGating(unittest.TestCase):
    """Broadcasts must no-op without peers, and job_created without the origin role."""

    def test_no_spawn_without_peers(self):
        from app.services import job_broadcast_service as jbs_module
        from app.services.job_broadcast_service import JobBroadcastService

        svc = JobBroadcastService()
        svc._spawn = MagicMock()

        with patch.object(jbs_module, "instance_config") as cfg:
            cfg.has_peers.return_value = False
            cfg.get_peer_urls.return_value = []
            svc.broadcast_job_created("j1", {})
            svc.broadcast_job_result("j1", {}, {})
            svc.broadcast_job_failed("j1", "err")
        svc._spawn.assert_not_called()

    def test_job_created_requires_llm_role(self):
        from app.services import job_broadcast_service as jbs_module
        from app.services.job_broadcast_service import JobBroadcastService

        svc = JobBroadcastService()
        svc._spawn = MagicMock()

        with patch.object(jbs_module, "instance_config") as cfg, \
             patch("app.config.llm_config.is_llm_server_configured", return_value=False):
            cfg.has_peers.return_value = True
            cfg.instance_url = "http://me:12410"
            svc.broadcast_job_created("j1", {})
        svc._spawn.assert_not_called()

    def test_job_result_pushes_regardless_of_role(self):
        """An instance that processed a job propagates results even if demoted mid-flight."""
        from app.services import job_broadcast_service as jbs_module
        from app.services.job_broadcast_service import JobBroadcastService

        svc = JobBroadcastService()
        svc._spawn = MagicMock()

        with patch.object(jbs_module, "instance_config") as cfg, \
             patch("app.config.llm_config.is_llm_server_configured", return_value=False):
            cfg.has_peers.return_value = True
            cfg.get_peer_urls.return_value = ["http://peer:12411"]
            svc.broadcast_job_result("j1", {"result": True}, {})

        svc._spawn.assert_called_once()
        # Close the un-awaited coroutine to avoid RuntimeWarning
        svc._spawn.call_args.args[0].close()


if __name__ == "__main__":
    unittest.main()

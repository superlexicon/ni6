"""
RethinkDB Job Queue Service

Provides persistent job queue using RethinkDB with changefeed support.
Jobs are indexed by instance_public_key for multi-instance coordination.
"""

import time
import threading
from typing import Dict, Any, Optional
from datetime import datetime

from rethinkdb import RethinkDB
from app.core import logger
from app.config.rethinkdb_config import rethinkdb_settings

r = RethinkDB()


class RethinkDBJobQueue:
    """
    RethinkDB-based job queue service.

    Provides persistent storage for async jobs with changefeed support.
    Jobs are tagged with instance_public_key for multi-instance coordination.
    """

    def __init__(self, rethinkdb_connection, instance_public_key: str):
        """
        Initialize RethinkDB job queue.

        Args:
            rethinkdb_connection: RethinkDB connection manager
            instance_public_key: Server public key for this instance
        """
        self.rethinkdb_connection = rethinkdb_connection
        self.instance_public_key = instance_public_key
        self.db_name = rethinkdb_settings.db
        self.table_name = rethinkdb_settings.jobs_table
        self.logger = logger

    def create_job(self, job_data: Dict[str, Any]) -> Optional[str]:
        """
        Insert a new job into RethinkDB.

        Args:
            job_data: Job data dict (must include id, status, request_data, etc.)

        Returns:
            Job ID if successful, None otherwise
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                conn = self.rethinkdb_connection.new_connection()
                try:
                    # instance_public_key should already be set by job_manager (target_server_public_key)
                    # Do NOT override it here!
                    if "instance_public_key" not in job_data:
                        self.logger.warning("Job missing instance_public_key, using receiving instance key")
                        job_data["instance_public_key"] = self.instance_public_key
                    job_data["created_at"] = r.now()

                    # Insert job
                    result = r.db(self.db_name).table(self.table_name).insert(
                        job_data
                    ).run(conn)

                    if result.get("generated_keys"):
                        job_id = result["generated_keys"][0]
                        self.logger.info(f"Job {job_id} created in RethinkDB for instance {self.instance_public_key[:16]}...")
                        return job_id
                    elif job_data.get("id"):
                        job_id = job_data["id"]
                        self.logger.info(f"Job {job_id} created in RethinkDB for instance {self.instance_public_key[:16]}...")
                        return job_id
                    else:
                        self.logger.error("Failed to get job ID from RethinkDB insert")
                        return None

                finally:
                    conn.close()

            except Exception as e:
                self.logger.error(f"Error creating job in RethinkDB (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                else:
                    self.logger.error(f"Failed to create job after {max_retries} attempts")
                    return None

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a job by ID.

        Args:
            job_id: Job ID to retrieve

        Returns:
            Job data dict or None if not found
        """
        try:
            conn = self.rethinkdb_connection.new_connection()
            try:
                job = r.db(self.db_name).table(self.table_name).get(job_id).run(conn)
                return job
            finally:
                conn.close()
        except Exception as e:
            self.logger.error(f"Error getting job {job_id}: {e}")
            return None

    def update_job_status(self, job_id: str, status: str, **kwargs) -> bool:
        """
        Update job status in RethinkDB.

        Args:
            job_id: Job ID to update
            status: New status (pending, processing, completed, failed)
            **kwargs: Additional fields to update (response_data, error_message, etc.)

        Returns:
            True if successful
        """
        try:
            conn = self.rethinkdb_connection.new_connection()
            try:
                update_data = {
                    "status": status,
                    "updated_at": r.now()
                }

                # Add optional fields
                if "response_data" in kwargs:
                    update_data["response_data"] = kwargs["response_data"]
                if "error_message" in kwargs:
                    update_data["error_message"] = kwargs["error_message"]
                if "started_at" in kwargs:
                    update_data["started_at"] = kwargs["started_at"]
                if "completed_at" in kwargs:
                    update_data["completed_at"] = kwargs["completed_at"]

                result = r.db(self.db_name).table(self.table_name).get(job_id).update(
                    update_data
                ).run(conn)

                if result.get("replaced") or result.get("unchanged"):
                    self.logger.debug(f"Job {job_id} status updated to {status}")
                    return True
                else:
                    self.logger.warning(f"Job {job_id} not found or not updated")
                    return False

            finally:
                conn.close()

        except Exception as e:
            self.logger.error(f"Error updating job {job_id}: {e}")
            return False

    def delete_job(self, job_id: str) -> bool:
        """
        Delete a job from RethinkDB.

        Args:
            job_id: Job ID to delete

        Returns:
            True if successful
        """
        try:
            conn = self.rethinkdb_connection.new_connection()
            try:
                result = r.db(self.db_name).table(self.table_name).get(job_id).delete().run(conn)

                if result.get("deleted"):
                    self.logger.info(f"Job {job_id} deleted from RethinkDB")
                    return True
                else:
                    self.logger.warning(f"Job {job_id} not found for deletion")
                    return False

            finally:
                conn.close()

        except Exception as e:
            self.logger.error(f"Error deleting job {job_id}: {e}")
            return False

    def get_pending_jobs(self, instance_public_key: Optional[str] = None) -> list:
        """
        Get pending jobs for an instance.

        Args:
            instance_public_key: Instance public key (defaults to this instance)

        Returns:
            List of pending job dicts
        """
        if instance_public_key is None:
            instance_public_key = self.instance_public_key

        try:
            conn = self.rethinkdb_connection.new_connection()
            try:
                jobs = r.db(self.db_name).table(self.table_name).filter({
                    "instance_public_key": instance_public_key,
                    "status": "pending"
                }).run(conn)

                return list(jobs)

            finally:
                conn.close()

        except Exception as e:
            self.logger.error(f"Error getting pending jobs: {e}")
            return []

    def get_pending_jobs_with_types(self, instance_public_key: Optional[str] = None) -> list:
        """
        Get pending jobs with document_type for queue time estimation.

        Returns jobs ordered by creation time (oldest first) so we can
        calculate position in queue accurately.

        Args:
            instance_public_key: Instance public key (defaults to this instance)

        Returns:
            List of pending job dicts with document_type, ordered by created_at
        """
        if instance_public_key is None:
            instance_public_key = self.instance_public_key

        try:
            conn = self.rethinkdb_connection.new_connection()
            try:
                # Get pending jobs ordered by creation time
                jobs = r.db(self.db_name).table(self.table_name).filter({
                    "instance_public_key": instance_public_key,
                    "status": "pending"
                }).order_by("created_at").run(conn)

                pending_jobs = []
                for job in list(jobs):
                    # Extract document_type from request_data
                    request_data = job.get("request_data", {})
                    files = request_data.get("files", [])

                    document_type = None
                    if files and len(files) > 0:
                        # Get document_type from first file
                        file_obj = files[0]
                        document_type = file_obj.get("document_type") or file_obj.get("file_type")

                    pending_jobs.append({
                        "id": job.get("id"),
                        "document_type": document_type,
                        "created_at": job.get("created_at")
                    })

                return pending_jobs

            finally:
                conn.close()

        except Exception as e:
            self.logger.error(f"Error getting pending jobs with types: {e}")
            return []

    def setup_table(self) -> bool:
        """
        Create RethinkDB jobs table and indexes.

        Returns:
            True if successful
        """
        try:
            conn = self.rethinkdb_connection.new_connection()
            try:
                # Check if database exists
                db_list = r.db_list().run(conn)
                if self.db_name not in db_list:
                    r.db_create(self.db_name).run(conn)
                    self.logger.info(f"Created RethinkDB database: {self.db_name}")

                # Check if table exists
                table_list = r.db(self.db_name).table_list().run(conn)
                if self.table_name not in table_list:
                    r.db(self.db_name).table_create(self.table_name).run(conn)
                    self.logger.info(f"Created RethinkDB table: {self.table_name}")

                # Create indexes
                index_list = r.db(self.db_name).table(self.table_name).index_list().run(conn)

                # instance_public_key index
                if "instance_public_key" not in index_list:
                    r.db(self.db_name).table(self.table_name).index_create(
                        "instance_public_key"
                    ).run(conn)
                    self.logger.info("Created index: instance_public_key")

                # status index
                if "status" not in index_list:
                    r.db(self.db_name).table(self.table_name).index_create(
                        "status"
                    ).run(conn)
                    self.logger.info("Created index: status")

                # Compound index for efficient filtering
                if "instance_status" not in index_list:
                    r.db(self.db_name).table(self.table_name).index_create(
                        "instance_status",
                        lambda doc: [doc["instance_public_key"], doc["status"]]
                    ).run(conn)
                    self.logger.info("Created index: instance_status")

                # Wait for indexes to be ready
                r.db(self.db_name).table(self.table_name).index_wait().run(conn)

                self.logger.info("RethinkDB jobs table setup completed")
                return True

            finally:
                conn.close()

        except Exception as e:
            self.logger.error(f"Failed to setup RethinkDB jobs table: {e}")
            return False

from typing import Optional, List, Dict, Any
import json
from datetime import datetime
from app.dto.job_models import JobStatus, JobDatabaseRecord
from app.core.logger import get_logger


class JobRepository:
    """Repository for job database operations"""

    # Fields to exclude from database storage due to size limits
    LARGE_FIELDS = {
        'encrypted_payload',      # Base64 encrypted video/image files (can be 50MB+)
        'encrypted_archive',      # Legacy encrypted zip archive (can be large)
        'encrypted_key',          # AES key (not needed in DB, kept for worker)
    }

    def __init__(self):
        self.logger = get_logger()

    def _strip_large_fields(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Strip large fields from request_data before storing in database.

        This prevents max_allowed_packet errors when storing video files.
        The actual file data is kept in memory by JobManager and passed to workers.
        """
        stripped_data = {}

        for key, value in request_data.items():
            if key in self.LARGE_FIELDS:
                # Replace with placeholder indicating data exists but not stored
                if value:
                    stripped_data[key] = f"<{key}_omitted_size_{len(str(value))}_bytes>"
                else:
                    stripped_data[key] = None
            elif key == 'files' and isinstance(value, list):
                # Strip file_data from each file, keep metadata only
                stripped_files = []
                for file_obj in value:
                    if isinstance(file_obj, dict):
                        file_metadata = {
                            'filename': file_obj.get('filename', ''),
                            'file_type': file_obj.get('file_type', ''),
                            'document_type': file_obj.get('document_type'),
                            'file_data': f"<file_data_omitted_size_{len(str(file_obj.get('file_data', '')))}_bytes>"
                        }
                        stripped_files.append(file_metadata)
                    else:
                        stripped_files.append(file_obj)
                stripped_data[key] = stripped_files
            else:
                stripped_data[key] = value

        return stripped_data

    def create_job(self, job_id: str, request_data: Dict[str, Any],
                  callback_url: Optional[str] = None, max_retries: int = 3,
                  user_identity_id: Optional[str] = None):
        """Create a new job record in the database. Returns dict if job exists, bool True on success."""
        from app.core.db.database import get_db_connection_context
        cursor = None
        try:
            # Check if job already exists (idempotency check)
            existing = self.get_job_by_id(job_id)
            if existing:
                self.logger.info(f"Job {job_id} already exists with status {existing.status.value}")
                return {"exists": True, "status": existing.status.value, "job_id": job_id}

            with get_db_connection_context() as conn:
                # Use buffered cursor with specific configuration
                cursor = conn.cursor(buffered=True, dictionary=False)

                # Strip out large fields before storing to avoid max_allowed_packet errors
                # Keep only metadata needed for job tracking and processing
                request_data_for_db = self._strip_large_fields(request_data)

                # Safely serialize JSON data
                try:
                    request_json = json.dumps(request_data_for_db, ensure_ascii=False)
                    self.logger.debug(f"Serialized request_data for job {job_id}: {request_json[:200]}...")
                except (TypeError, ValueError) as e:
                    self.logger.error(f"Failed to serialize request_data for job {job_id}: {e}")
                    # Fallback to safe serialization
                    request_json = json.dumps(str(request_data_for_db), ensure_ascii=False)

                # Prepare values with explicit type casting and validation
                try:
                    # Ensure all string parameters are actually strings and not None
                    job_id_str = str(job_id) if job_id is not None else ""
                    status_str = str(JobStatus.PENDING.value) if JobStatus.PENDING.value is not None else "pending"
                    request_json_str = str(request_json) if request_json is not None else "{}"
                    callback_url_str = str(callback_url) if callback_url is not None else ""
                    max_retries_int = int(max_retries) if max_retries is not None else 3

                    # Extract client_public_key from request_data for indexed lookup
                    client_public_key = request_data.get('client_public_key', '')

                    self.logger.debug(f"Job {job_id} prepared parameters: {[(type(p).__name__ + ': ' + str(p)[:50]) for p in [job_id_str, status_str, request_json_str, callback_url_str, max_retries_int]]}")

                    # Build SQL with escaped values to avoid parameterized query issues
                    # Using MySQL's STRING_ESCAPE function for safe string escaping
                    escaped_callback_url = "NULL" if callback_url_str == "" else f"'{self._mysql_escape(cursor, callback_url_str)}'"
                    escaped_request_json = f"'{self._mysql_escape(cursor, request_json_str)}'"
                    escaped_client_public_key = "NULL" if not client_public_key else f"'{self._mysql_escape(cursor, client_public_key)}'"
                    escaped_user_identity_id = "NULL" if not user_identity_id else f"'{self._mysql_escape(cursor, str(user_identity_id))}'"

                    sql = f"""
                        INSERT INTO document_analysis_jobs
                        (id, client_public_key, user_identity_id, status, request_data, callback_url, max_retries)
                        VALUES ('{self._mysql_escape(cursor, job_id_str)}',
                               {escaped_client_public_key},
                               {escaped_user_identity_id},
                               '{self._mysql_escape(cursor, status_str)}',
                               {escaped_request_json},
                               {escaped_callback_url},
                               {max_retries_int})
                    """

                    self.logger.debug(f"Executing SQL for job {job_id}: {sql[:200]}...")

                except Exception as e:
                    self.logger.error(f"Failed to prepare parameters for job {job_id}: {e}")
                    raise

                # Execute with escaped SQL
                cursor.execute(sql)
                affected_rows = cursor.rowcount
                self.logger.info(f"INSERT affected {affected_rows} rows for job {job_id}")

                conn.commit()
                self.logger.info(f"Created job {job_id} in database")
                return True

        except Exception as e:
            self.logger.error(f"Failed to create job {job_id}: {type(e).__name__}")
            if hasattr(e, 'errno'):
                self.logger.error(f"MySQL errno: {e.errno}")
            if hasattr(e, 'sqlstate'):
                self.logger.error(f"MySQL SQLSTATE: {e.sqlstate}")

            try:
                conn.rollback()
            except Exception as rollback_err:
                self.logger.error(f"Error during rollback: {type(rollback_err).__name__}")
            return False
        finally:
            if cursor:
                try:
                    cursor.close()
                except Exception as close_err:
                    self.logger.error(f"Error closing cursor: {close_err}")

    def _mysql_escape(self, cursor, value: str) -> str:
        """Helper method to safely escape MySQL strings"""
        if value is None:
            return ""
        # Use MySQL connector's escape function
        try:
            # Convert to bytes if needed, then use connection's escape method
            if hasattr(cursor, 'connection') and hasattr(cursor.connection, 'converter') and cursor.connection.converter:
                escaped_value = cursor.connection.converter.escape(value)
            else:
                # Fallback: basic escaping
                escaped_value = str(value).replace('\\', '\\\\').replace("'", "\\'").replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
            return escaped_value
        except Exception as e:
            self.logger.warning(f"Error escaping value, using fallback: {e}")
            return str(value).replace('\\', '\\\\').replace("'", "\\'").replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')

    def get_job_by_id(self, job_id: str) -> Optional[JobDatabaseRecord]:
        """Retrieve job record by ID"""
        from app.core.db.database import get_db_connection_context
        try:
            with get_db_connection_context() as conn:
                cursor = conn.cursor(dictionary=True)
                sql = """
                SELECT * FROM document_analysis_jobs WHERE id = %s
                """
                cursor.execute(sql, (job_id,))
                result = cursor.fetchone()
                cursor.close()

                if result:
                    return JobDatabaseRecord(
                        id=result['id'],
                        status=JobStatus(result['status']),
                        request_data=json.loads(result['request_data']) if result['request_data'] else {},
                        response_data=json.loads(result['response_data']) if result['response_data'] else None,
                        error_message=result['error_message'],
                        callback_url=result['callback_url'],
                        retry_count=result['retry_count'],
                        max_retries=result['max_retries'],
                        created_at=result['created_at'],
                        updated_at=result['updated_at'],
                        started_at=result['started_at'],
                        completed_at=result['completed_at'],
                        callback_attempted_at=result['callback_attempted_at']
                    )
                return None
        except Exception as e:
            self.logger.error(f"Failed to get job {job_id}: {str(e)}")
            return None

    def get_job_by_public_key(self, public_key: str) -> Optional[JobDatabaseRecord]:
        """Retrieve most recent job record by client public key"""
        from app.core.db.database import get_db_connection_context
        try:
            with get_db_connection_context() as conn:
                cursor = conn.cursor(dictionary=True)
                sql = """
                SELECT * FROM document_analysis_jobs
                WHERE client_public_key = %s
                ORDER BY created_at DESC
                LIMIT 1
                """
                cursor.execute(sql, (public_key,))
                result = cursor.fetchone()
                cursor.close()

                if result:
                    return JobDatabaseRecord(
                        id=result['id'],
                        status=JobStatus(result['status']),
                        request_data=json.loads(result['request_data']) if result['request_data'] else {},
                        response_data=json.loads(result['response_data']) if result['response_data'] else None,
                        error_message=result['error_message'],
                        callback_url=result['callback_url'],
                        retry_count=result['retry_count'],
                        max_retries=result['max_retries'],
                        created_at=result['created_at'],
                        updated_at=result['updated_at'],
                        started_at=result['started_at'],
                        completed_at=result['completed_at'],
                        callback_attempted_at=result['callback_attempted_at']
                    )
                return None
        except Exception as e:
            self.logger.error(f"Failed to get job by public_key: {str(e)}")
            return None

    def get_in_progress_jobs_by_user_identity_id(
        self, user_identity_id: str, limit: int = 10
    ) -> List[JobDatabaseRecord]:
        """Retrieve in-progress jobs by user identity ID."""
        from app.core.db.database import get_db_connection_context
        try:
            with get_db_connection_context() as conn:
                cursor = conn.cursor(dictionary=True)
                sql = """
                SELECT * FROM document_analysis_jobs
                WHERE user_identity_id = %s
                AND status IN ('pending', 'processing')
                ORDER BY created_at DESC
                LIMIT %s
                """
                cursor.execute(sql, (user_identity_id, limit))
                results = cursor.fetchall()
                cursor.close()

                return [JobDatabaseRecord(
                    id=r['id'],
                    status=JobStatus(r['status']),
                    request_data=json.loads(r['request_data']) if r['request_data'] else {},
                    response_data=json.loads(r['response_data']) if r['response_data'] else None,
                    error_message=r['error_message'],
                    callback_url=r['callback_url'],
                    retry_count=r['retry_count'],
                    max_retries=r['max_retries'],
                    created_at=r['created_at'],
                    updated_at=r['updated_at'],
                    started_at=r['started_at'],
                    completed_at=r['completed_at'],
                    callback_attempted_at=r['callback_attempted_at']
                ) for r in results]
        except Exception as e:
            self.logger.error(f"Error getting in-progress jobs by user_identity_id: {e}")
            return []

    def update_job_user_identity_id(self, job_id: str, user_identity_id: str) -> bool:
        """Update job with user_identity_id."""
        from app.core.db.database import get_db_connection_context
        try:
            with get_db_connection_context() as conn:
                cursor = conn.cursor()
                sql = """
                UPDATE document_analysis_jobs
                SET user_identity_id = %s, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """
                cursor.execute(sql, (user_identity_id, job_id))
                conn.commit()
                cursor.close()

                self.logger.info(f"Updated job {job_id} with user_identity_id {user_identity_id}")
                return True

        except Exception as e:
            self.logger.error(f"Failed to update job {job_id} with user_identity_id: {str(e)}")
            return False

    def update_job_status(self, job_id: str, status: JobStatus,
                         error_message: Optional[str] = None,
                         response_data: Optional[Dict[str, Any]] = None) -> bool:
        """Update job status and related fields"""
        from app.core.db.database import get_db_connection_context
        try:
            with get_db_connection_context() as conn:
                cursor = conn.cursor()

                # Build dynamic SQL based on what needs to be updated
                update_fields = ["status = %s", "updated_at = CURRENT_TIMESTAMP"]
                params = [status.value]

                if status == JobStatus.PROCESSING:
                    update_fields.append("started_at = CURRENT_TIMESTAMP")
                elif status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CALLBACK_FAILED]:
                    update_fields.append("completed_at = CURRENT_TIMESTAMP")

                if error_message:
                    update_fields.append("error_message = %s")
                    params.append(error_message)

                if response_data:
                    update_fields.append("response_data = %s")
                    params.append(json.dumps(response_data))

                params.append(job_id)

                sql = f"""
                UPDATE document_analysis_jobs
                SET {', '.join(update_fields)}
                WHERE id = %s
                """

                cursor.execute(sql, params)
                conn.commit()
                cursor.close()

                self.logger.info(f"Updated job {job_id} status to {status.value}")
                return True

        except Exception as e:
            self.logger.error(f"Failed to update job {job_id} status: {str(e)}")
            return False

    def increment_job_retry(self, job_id: str) -> bool:
        """Increment retry count for a job"""
        from app.core.db.database import get_db_connection_context
        try:
            with get_db_connection_context() as conn:
                cursor = conn.cursor()
                sql = """
                UPDATE document_analysis_jobs
                SET retry_count = retry_count + 1, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """
                cursor.execute(sql, (job_id,))
                conn.commit()
                cursor.close()

                self.logger.info(f"Incremented retry count for job {job_id}")
                return True

        except Exception as e:
            self.logger.error(f"Failed to increment retry count for job {job_id}: {str(e)}")
            return False

    def mark_callback_attempted(self, job_id: str) -> bool:
        """Mark that callback was attempted for a job"""
        from app.core.db.database import get_db_connection_context
        try:
            with get_db_connection_context() as conn:
                cursor = conn.cursor()
                sql = """
                UPDATE document_analysis_jobs
                SET callback_attempted_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """
                cursor.execute(sql, (job_id,))
                conn.commit()
                cursor.close()

                self.logger.info(f"Marked callback attempted for job {job_id}")
                return True

        except Exception as e:
            self.logger.error(f"Failed to mark callback attempted for job {job_id}: {str(e)}")
            return False

    def get_pending_jobs(self, limit: int = 100) -> List[JobDatabaseRecord]:
        """Get pending jobs for processing"""
        from app.core.db.database import get_db_connection_context
        try:
            with get_db_connection_context() as conn:
                cursor = conn.cursor(dictionary=True)
                sql = """
                SELECT * FROM document_analysis_jobs
                WHERE status = %s
                ORDER BY created_at ASC
                LIMIT %s
                """
                cursor.execute(sql, (JobStatus.PENDING.value, limit))
                results = cursor.fetchall()
                cursor.close()

                jobs = []
                for result in results:
                    jobs.append(JobDatabaseRecord(
                        id=result['id'],
                        status=JobStatus(result['status']),
                        request_data=json.loads(result['request_data']) if result['request_data'] else {},
                        response_data=json.loads(result['response_data']) if result['response_data'] else None,
                        error_message=result['error_message'],
                        callback_url=result['callback_url'],
                        retry_count=result['retry_count'],
                        max_retries=result['max_retries'],
                        created_at=result['created_at'],
                        updated_at=result['updated_at'],
                        started_at=result['started_at'],
                        completed_at=result['completed_at'],
                        callback_attempted_at=result['callback_attempted_at']
                    ))

                return jobs

        except Exception as e:
            self.logger.error(f"Failed to get pending jobs: {str(e)}")
            return []

    def get_pending_jobs_for_instance(self, instance_public_key: str, limit: int = 100) -> List[JobDatabaseRecord]:
        """
        Get pending jobs for a specific instance.

        Jobs are routed via target_server_public_key in the request_data.
        This method returns jobs where target_server_public_key matches the given instance.

        Args:
            instance_public_key: The instance's public key
            limit: Maximum number of jobs to return

        Returns:
            List of pending JobDatabaseRecord objects
        """
        from app.core.db.database import get_db_connection_context
        try:
            with get_db_connection_context() as conn:
                cursor = conn.cursor(dictionary=True)
                # Jobs are stored with target_server_public_key in request_data JSON
                # We need to use JSON extraction to filter
                sql = """
                SELECT * FROM document_analysis_jobs
                WHERE status = %s
                AND JSON_EXTRACT(request_data, '$.target_server_public_key') = %s
                ORDER BY created_at ASC
                LIMIT %s
                """
                cursor.execute(sql, (JobStatus.PENDING.value, instance_public_key, limit))
                results = cursor.fetchall()
                cursor.close()

                jobs = []
                for result in results:
                    jobs.append(JobDatabaseRecord(
                        id=result['id'],
                        status=JobStatus(result['status']),
                        request_data=json.loads(result['request_data']) if result['request_data'] else {},
                        response_data=json.loads(result['response_data']) if result['response_data'] else None,
                        error_message=result['error_message'],
                        callback_url=result['callback_url'],
                        retry_count=result['retry_count'],
                        max_retries=result['max_retries'],
                        created_at=result['created_at'],
                        updated_at=result['updated_at'],
                        started_at=result['started_at'],
                        completed_at=result['completed_at'],
                        callback_attempted_at=result['callback_attempted_at']
                    ))

                return jobs

        except Exception as e:
            self.logger.error(f"Failed to get pending jobs for instance {instance_public_key[:16]}...: {str(e)}")
            return []

    def delete_completed_job(self, job_id: str) -> bool:
        """Delete a completed job from database"""
        from app.core.db.database import get_db_connection_context
        try:
            with get_db_connection_context() as conn:
                cursor = conn.cursor()
                sql = "DELETE FROM document_analysis_jobs WHERE id = %s AND status = %s"
                cursor.execute(sql, (job_id, JobStatus.COMPLETED.value))

                if cursor.rowcount > 0:
                    conn.commit()
                    self.logger.info(f"Deleted completed job {job_id}")
                    cursor.close()
                    return True
                else:
                    cursor.close()
                    self.logger.warning(f"Job {job_id} not found or not completed, cannot delete")
                    return False

        except Exception as e:
            self.logger.error(f"Failed to delete job {job_id}: {str(e)}")
            return False

    def delete_job(self, job_id: str) -> bool:
        """Delete any job from database"""
        from app.core.db.database import get_db_connection_context
        try:
            with get_db_connection_context() as conn:
                cursor = conn.cursor()
                sql = "DELETE FROM document_analysis_jobs WHERE id = %s"
                cursor.execute(sql, (job_id,))

                if cursor.rowcount > 0:
                    conn.commit()
                    self.logger.info(f"Deleted job {job_id}")
                    cursor.close()
                    return True
                else:
                    cursor.close()
                    self.logger.warning(f"Job {job_id} not found, cannot delete")
                    return False

        except Exception as e:
            self.logger.error(f"Failed to delete job {job_id}: {str(e)}")
            return False

    def reset_stale_jobs(self, stale_threshold_hours: int = 1) -> int:
        """
        Reset jobs that have been processing for too long back to pending.
        """
        from app.core.db.database import get_db_connection_context
        cursor = None
        try:
            with get_db_connection_context() as fresh_conn:
                # Use buffered cursor to avoid "Commands out of sync" errors
                cursor = fresh_conn.cursor(buffered=True)
                sql = """
                    UPDATE document_analysis_jobs
                    SET status = %s, started_at = NULL, updated_at = CURRENT_TIMESTAMP
                    WHERE status = %s
                    AND started_at < DATE_SUB(NOW(), INTERVAL %s HOUR)
                """
                cursor.execute(sql, (JobStatus.PENDING.value, JobStatus.PROCESSING.value, stale_threshold_hours))
                affected_rows = cursor.rowcount
                fresh_conn.commit()

                if affected_rows > 0:
                    self.logger.info(f"Reset {affected_rows} stale jobs to pending")

                return affected_rows

        except Exception as e:
            self.logger.error(f"Failed to reset stale jobs: {str(e)}")
            return 0
        finally:
            if cursor:
                try:
                    cursor.close()
                except Exception:
                    pass

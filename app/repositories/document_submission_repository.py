import json
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta, timezone
from mysql.connector.errors import Error as MySQLError
from app.core import logger
from .base_repository import BaseRepository

# Multi-Device State Management:
# Document submissions store verification_state and sequence_no from user_keys (per-device)
# instead of user_identity_index (overall identity state). This ensures each device's
# submissions reflect that device's verification progress.
# Lazy import get_ecies_encryption_service to avoid circular import with app.services


# Fields to exclude from database storage due to size limits
LARGE_FIELDS = {
    'encrypted_payload',      # Base64 encrypted video/image files (can be 50MB+)
    'encrypted_archive',      # Legacy encrypted zip archive (can be large)
    'encrypted_key',          # AES key (not needed in DB, kept for worker)
}


def strip_large_fields(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Strip large fields from request_data before storing in database.

    This prevents max_allowed_packet errors when storing video files.
    The actual file data is kept in RethinkDB and passed to workers.

    Args:
        request_data: Original request data with full file data

    Returns:
        Request data with large fields replaced by size placeholders
    """
    if not request_data:
        return {}

    stripped_data = {}

    for key, value in request_data.items():
        if key in LARGE_FIELDS:
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


class DocumentSubmissionRepository(BaseRepository):
    def __init__(self):
        super().__init__('document_submissions')
        # Lazy load verification settings to avoid circular import
        self._verification_settings = None

    @property
    def verification_settings(self):
        """Lazy-load verification settings."""
        if self._verification_settings is None:
            from app.config.verification_config import verification_settings
            self._verification_settings = verification_settings
        return self._verification_settings

    def _construct_response_data(
        self,
        row: Dict[str, Any],
        client_public_key: Optional[str] = None,
        decrypt_extracted_data: bool = False
    ) -> Dict[str, Any]:
        """
        Construct response_data dict from individual columns.

        This method builds the complete response_data structure that was previously
        stored in the response_data JSON column, now constructed from individual
        columns.

        IMPORTANT: With ECIES encryption, server-side decryption is NOT performed.
        The encrypted envelope is returned as-is for client-side decryption.

        Args:
            row: Database row with all columns
            client_public_key: Client public key (NOT used for decryption with ECIES)
            decrypt_extracted_data: Deprecated parameter - kept for API compatibility,
                                   server-side decryption is NOT performed with ECIES

        Returns:
            Constructed response_data dict (PII remains encrypted in envelope)
        """
        response_data: Dict[str, Any] = {}

        # Document metadata fields (included for other_results to identify documents)
        if row.get('document_type'):
            response_data['document_type'] = row['document_type']
        if row.get('filename'):
            response_data['filename'] = row['filename']
        if row.get('job_id'):
            response_data['job_id'] = row['job_id']
        if row.get('processing_time_seconds') is not None:
            response_data['processing_time_seconds'] = row['processing_time_seconds']
        if row.get('verification_state') is not None:
            response_data['verification_state'] = row['verification_state']
        if row.get('sequence_no') is not None:
            response_data['sequence_no'] = row['sequence_no']
        if row.get('result_status') is not None:
            response_data['result'] = row['result_status']
        if row.get('error_message'):
            response_data['error'] = row['error_message']
        if row.get('error_code'):
            response_data['error_code'] = row['error_code']

        # Forgery checks (PhotoHolmes results)
        if row.get('forgery_checks_summary'):
            try:
                if isinstance(row['forgery_checks_summary'], str):
                    response_data['forgery_checks'] = json.loads(row['forgery_checks_summary'])
                else:
                    response_data['forgery_checks'] = row['forgery_checks_summary']
            except (json.JSONDecodeError, TypeError):
                response_data['forgery_checks'] = {}

        # Other checks (validation results)
        if row.get('other_checks_summary'):
            try:
                if isinstance(row['other_checks_summary'], str):
                    response_data['other_checks'] = json.loads(row['other_checks_summary'])
                else:
                    response_data['other_checks'] = row['other_checks_summary']
            except (json.JSONDecodeError, TypeError):
                response_data['other_checks'] = {}

        # Score summaries
        if row.get('docs_auth_score') is not None:
            response_data['docs_auth_score'] = row['docs_auth_score']
        if row.get('id_veri_score') is not None:
            response_data['id_veri_score'] = row['id_veri_score']

        # Handle extracted_data_encrypted (ECIES envelope)
        # With ECIES, server CANNOT decrypt - return envelope for client-side decryption
        if row.get('extracted_data_encrypted'):
            try:
                # Parse encrypted envelope and return as-is for client to decrypt
                envelope = json.loads(row['extracted_data_encrypted'])
                response_data['extracted_data_encrypted'] = envelope
                logger.debug(f"Included encrypted data envelope (version: {envelope.get('version', 'unknown')})")
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to parse encrypted envelope: {e}")
                response_data['extracted_data_encrypted'] = None

        return response_data

    def _extract_submission_fields(
        self,
        response_data: Dict[str, Any],
        request_data: Dict[str, Any],
        client_public_key: Optional[str]
    ) -> Dict[str, Any]:
        """
        Extract and encrypt fields from response_data for storage in individual columns.

        Args:
            response_data: Full response data from analysis
            request_data: Original request data
            client_public_key: Client public key for encryption

        Returns:
            Dict with all fields ready for database insertion
        """
        result = {
            'extracted_data_encrypted': None,
            'processing_time_seconds': None,
            'verification_state': None,
            'sequence_no': None,
            'docs_auth_score': None,
            'id_veri_score': None,
            'forgery_checks_summary': None,
            'other_checks_summary': None,
            'result_status': None,
            'error_message': None,
            'error_code': None
        }

        if not response_data:
            return result

        # Encrypt extracted_data PII with ECIES (user-only decryption)
        extracted_data = response_data.get('extracted_data')
        if extracted_data and client_public_key:
            try:
                from app.services.ecies_encryption_service import get_ecies_encryption_service
                encryption_service = get_ecies_encryption_service()
                result['extracted_data_encrypted'] = encryption_service.create_encryption_envelope(
                    extracted_data, client_public_key
                )
                logger.info(f"Encrypted extracted_data with ECIES for user-only decryption")
            except Exception as e:
                logger.warning(f"Failed to encrypt extracted_data with ECIES: {e}")

        # Extract summary metrics
        result['processing_time_seconds'] = response_data.get('processing_time_seconds')

        # Get verification_state and sequence_no from user_keys (per-device state)
        # instead of from response_data (which may contain overall identity state)
        if client_public_key:
            from app.repositories.user_key_repository import UserKeyRepository
            user_key_repo = UserKeyRepository()
            result['verification_state'] = user_key_repo.get_verification_state(client_public_key)
            result['sequence_no'] = user_key_repo.get_sequence_no(client_public_key)
        else:
            # Fallback to response_data if client_public_key not available
            result['verification_state'] = response_data.get('verification_state')
            result['sequence_no'] = response_data.get('sequence_no')

        result['docs_auth_score'] = response_data.get('docs_auth_score')
        result['id_veri_score'] = response_data.get('id_veri_score')
        result['result_status'] = response_data.get('result')
        result['error_message'] = response_data.get('error')
        result['error_code'] = response_data.get('error_code')

        # Store forgery_checks and other_checks as JSON
        forgery_checks = response_data.get('forgery_checks')
        if forgery_checks:
            result['forgery_checks_summary'] = json.dumps(forgery_checks)

        other_checks = response_data.get('other_checks')
        if other_checks:
            result['other_checks_summary'] = json.dumps(other_checks)

        return result

    def create_submission_record(
        self,
        user_identity_id: Optional[str],
        client_public_key: Optional[str] = None,
        filename: Optional[str] = None,
        document_type: Optional[str] = None,
        response_data: Optional[Dict[str, Any]] = None,
        request_data: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Create a document submission record.

        Note: document_hash removed - uniqueness enforced by face biometrics trigger.
        Same document can be submitted with different encryption (multi-device).

        Args:
            user_identity_id: User identity ID (can be None for selfie-first flow)
            client_public_key: Client public key for lookup
            filename: Document filename (optional)
            document_type: Type of document (optional)
            response_data: Response data (PII will be encrypted)
            request_data: Original request data (optional)

        Returns:
            Created record or None if failed
        """
        from app.core.db.database import get_db_connection_context
        if not client_public_key and request_data:
            client_public_key = request_data.get('client_public_key')
        fields = self._extract_submission_fields(response_data, request_data, client_public_key)

        # Strip large fields from request_data to avoid max_allowed_packet errors
        request_data_for_db = strip_large_fields(request_data)

        query = """
            INSERT INTO document_submissions
            (id, user_identity_id, client_public_key, filename, document_type,
             request_data, extracted_data_encrypted, processing_time_seconds,
             verification_state, sequence_no, docs_auth_score, id_veri_score,
             forgery_checks_summary, other_checks_summary, result_status, error_message, error_code)
            VALUES (UUID(), %(user_identity_id)s, %(client_public_key)s, %(filename)s, %(document_type)s,
                    %(request_data)s, %(extracted_data_encrypted)s, %(processing_time_seconds)s,
                    %(verification_state)s, %(sequence_no)s, %(docs_auth_score)s, %(id_veri_score)s,
                    %(forgacy_checks_summary)s, %(other_checks_summary)s, %(result_status)s, %(error_message)s, %(error_code)s)
        """

        params = {
            'user_identity_id': user_identity_id,
            'client_public_key': client_public_key,
            'filename': filename,
            'document_type': document_type,
            'request_data': json.dumps(request_data_for_db) if request_data_for_db else None,
            'extracted_data_encrypted': fields['extracted_data_encrypted'],
            'processing_time_seconds': fields['processing_time_seconds'],
            'verification_state': fields['verification_state'],
            'sequence_no': fields['sequence_no'],
            'docs_auth_score': fields['docs_auth_score'],
            'id_veri_score': fields['id_veri_score'],
            'forgacy_checks_summary': fields['forgery_checks_summary'],
            'other_checks_summary': fields['other_checks_summary'],
            'result_status': fields['result_status'],
            'error_message': fields['error_message'],
            'error_code': fields['error_code']
        }

        try:
            with get_db_connection_context() as conn:
                with conn.cursor(dictionary=True) as cursor:
                    cursor.execute(query, params)
                    conn.commit()
                    logger.info(
                        f"Created submission record for user {user_identity_id}: {filename}"
                    )
                    return {'id': str(cursor.lastrowid)} if cursor.lastrowid else None
        except MySQLError as e:
            logger.error(f"Error creating submission record: {e}")
            return None

    def create_submission(
        self,
        response_data: Dict[str, Any],
        request_data: Dict[str, Any],
        job_id: Optional[str] = None
    ) -> tuple[bool, str]:
        """
        Create a document submission record.
        PII is encrypted with user's public key and stored in individual columns.

        Note: document_hash removed - uniqueness enforced by face biometrics trigger.
        Same document can be submitted with different encryption (multi-device).

        Args:
            response_data: Response data from analysis (including extracted_data)
            request_data: Original request data
            job_id: Optional job ID to link submission to original job

        Returns:
            Tuple of (success, error_message):
            - (True, "") if successful
            - (False, error_message) if failed
        """
        from app.core.db.database import get_db_connection_context
        # Extract user identity from response data first (set by sequential services)
        client_public_key = request_data.get('client_public_key')
        user_identity_id = None

        # Check response_data for user_identity_id (sequential services set this at top level via model_dump)
        if response_data and response_data.get('user_identity_id'):
            user_identity_id = response_data['user_identity_id']
            logger.info(f"Got user_identity_id from response: {user_identity_id[:16] if user_identity_id else 'None'}...")

        # Fallback: look up from user_keys table
        if not user_identity_id and client_public_key:
            from app.repositories.user_key_repository import UserKeyRepository
            user_key_repo = UserKeyRepository()
            user_key = user_key_repo.get_key_by_public_key(client_public_key)
            if user_key and user_key.get('user_identity_id'):
                user_identity_id = user_key['user_identity_id']
                logger.info(f"Got user_identity_id from user_keys: {user_identity_id[:16] if user_identity_id else 'None'}...")

        # Extract file info from request data
        files = request_data.get('files', [])
        filename = None
        document_type = None

        if files and len(files) > 0:
            filename = files[0].get('filename')
            # Flutter sends 'file_type', normalize it to lowercase
            raw_type = files[0].get('document_type') or files[0].get('file_type')
            if raw_type:
                document_type = raw_type.lower().replace(' ', '_')

        # Extract and encrypt fields for storage
        fields = self._extract_submission_fields(response_data, request_data, client_public_key)

        # Strip large fields from request_data to avoid max_allowed_packet errors
        request_data_for_db = strip_large_fields(request_data)

        query = """
            INSERT INTO document_submissions
            (id, user_identity_id, client_public_key, filename, document_type,
             request_data, job_id, extracted_data_encrypted, processing_time_seconds,
             verification_state, sequence_no, docs_auth_score, id_veri_score,
             forgery_checks_summary, other_checks_summary, result_status, error_message)
            VALUES (UUID(), %(user_identity_id)s, %(client_public_key)s,
                    %(filename)s, %(document_type)s, %(request_data)s, %(job_id)s,
                    %(extracted_data_encrypted)s, %(processing_time_seconds)s, %(verification_state)s,
                    %(sequence_no)s, %(docs_auth_score)s, %(id_veri_score)s, %(forgery_checks_summary)s,
                    %(other_checks_summary)s, %(result_status)s, %(error_message)s)
        """

        params = {
            'user_identity_id': user_identity_id,
            'client_public_key': client_public_key,
            'filename': filename,
            'document_type': document_type,
            'request_data': json.dumps(request_data_for_db) if request_data_for_db else None,
            'job_id': job_id,
            'extracted_data_encrypted': fields['extracted_data_encrypted'],
            'processing_time_seconds': fields['processing_time_seconds'],
            'verification_state': fields['verification_state'],
            'sequence_no': fields['sequence_no'],
            'docs_auth_score': fields['docs_auth_score'],
            'id_veri_score': fields['id_veri_score'],
            'forgery_checks_summary': fields['forgery_checks_summary'],
            'other_checks_summary': fields['other_checks_summary'],
            'result_status': fields['result_status'],
            'error_message': fields['error_message']
        }

        try:
            with get_db_connection_context() as conn:
                with conn.cursor(dictionary=True) as cursor:
                    cursor.execute(query, params)
                    conn.commit()
                    # Success - return success with empty error message
                    logger.info(
                        f"Created submission record for {document_type}: {filename}"
                    )
                    return (True, "")
        except MySQLError as e:
            error_msg = f"Error creating submission record: {e}"
            logger.error(error_msg)
            return (False, error_msg)

    def get_submission_by_public_key(
        self,
        client_public_key: str,
        decrypt_extracted_data: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Get the most recent document submission by client public key.
        Constructs response_data from individual columns.

        Args:
            client_public_key: Client public key
            decrypt_extracted_data: Whether to decrypt extracted_data (server-side)

        Returns:
            Most recent document submission with constructed response_data, or None if not found
        """
        from app.core.db.database import get_db_connection_context
        query = """
            SELECT id, user_identity_id, client_public_key,
                   filename, document_type, request_data, job_id, submitted_at,
                   extracted_data_encrypted, processing_time_seconds, verification_state,
                   sequence_no, docs_auth_score, id_veri_score, forgery_checks_summary,
                   other_checks_summary, result_status, error_message
            FROM document_submissions
            WHERE client_public_key = %s
            ORDER BY submitted_at DESC
            LIMIT 1
        """

        try:
            with get_db_connection_context() as conn:
                with conn.cursor(dictionary=True) as cursor:
                    cursor.execute(query, (client_public_key,))
                    result = cursor.fetchone()

                    if result:
                        # Parse request_data
                        if result.get('request_data'):
                            try:
                                result['request_data'] = json.loads(result['request_data'])
                            except (json.JSONDecodeError, TypeError):
                                result['request_data'] = {}

                        # Construct response_data from individual columns
                        result['response_data'] = self._construct_response_data(
                            result, client_public_key, decrypt_extracted_data
                        )

                    return result
        except MySQLError as e:
            logger.error(f"Error getting submission by public key: {e}")
            return None

    def get_user_document_submissions(
        self,
        user_identity_id: str,
        limit: int = 50,
        client_public_key: Optional[str] = None,
        decrypt_extracted_data: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get the most recent document submission for each document type for a user.
        Uses ROW_NUMBER() to return only the latest submission per document type.

        In multi-device scenarios, only returns submissions encrypted with the
        requesting device's public key to ensure the response can be decrypted.

        Args:
            user_identity_id: User identity ID
            limit: Unused parameter (kept for API compatibility)
            client_public_key: Client public key to filter submissions by.
                              When provided, only returns submissions encrypted
                              with this key. This ensures devices can decrypt responses.
            decrypt_extracted_data: Whether to decrypt extracted_data (server-side)

        Returns:
            List of document submissions with constructed response_data (max one per document type)
        """
        from app.core.db.database import get_db_connection_context

        # Build WHERE clause based on whether client_public_key is provided
        # When filtering by client_public_key, each device only sees its own submissions
        if client_public_key:
            where_clause = "WHERE user_identity_id = %s AND client_public_key = %s"
            query_params = (user_identity_id, client_public_key)
        else:
            where_clause = "WHERE user_identity_id = %s"
            query_params = (user_identity_id,)

        query = f"""
            WITH ranked_submissions AS (
                SELECT id, filename, document_type, request_data,
                       submitted_at, client_public_key, job_id,
                       extracted_data_encrypted,
                       processing_time_seconds, verification_state, sequence_no,
                       docs_auth_score, id_veri_score, forgery_checks_summary,
                       other_checks_summary, result_status, error_message,
                       ROW_NUMBER() OVER (PARTITION BY document_type ORDER BY submitted_at DESC) as rn
                FROM document_submissions
                {where_clause}
            )
            SELECT id, filename, document_type, request_data,
                   submitted_at, client_public_key, job_id,
                   extracted_data_encrypted,
                   processing_time_seconds, verification_state, sequence_no,
                   docs_auth_score, id_veri_score, forgery_checks_summary,
                   other_checks_summary, result_status, error_message
            FROM ranked_submissions
            WHERE rn = 1
        """

        try:
            with get_db_connection_context() as conn:
                with conn.cursor(dictionary=True) as cursor:
                    cursor.execute(query, query_params)
                    results = cursor.fetchall()

                    # Process each result
                    for result in results:
                        # Parse request_data
                        if result.get('request_data'):
                            try:
                                result['request_data'] = json.loads(result['request_data'])
                            except (json.JSONDecodeError, TypeError):
                                result['request_data'] = {}

                        # Construct response_data from individual columns
                        # When filtered by client_public_key, all results have the same key
                        # Use the request's client_public_key to ensure correct encryption context
                        pk = client_public_key or result.get('client_public_key')
                        result['response_data'] = self._construct_response_data(
                            result, pk, decrypt_extracted_data
                        )

                    return results
        except MySQLError as e:
            logger.error(f"Error fetching document submissions for user {user_identity_id}: {e}")
            return []

    def get_user_submission_by_type(
        self,
        user_identity_id: str,
        document_type: str,
        client_public_key: Optional[str] = None,
        decrypt_extracted_data: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Get the most recent document submission of a specific type for a user.
        Constructs response_data from individual columns.

        Used to detect resubmissions - if a user already has a document of
        the same type, the new submission is a resubmission.

        Args:
            user_identity_id: User identity ID
            document_type: Type of document (selfie, passport, bank_statement)
            client_public_key: Client public key for decryption (if needed)
            decrypt_extracted_data: Whether to decrypt extracted_data (server-side)

        Returns:
            Most recent submission of that type with constructed response_data, or None if not found
        """
        from app.core.db.database import get_db_connection_context
        query = """
            SELECT id, user_identity_id, client_public_key,
                   filename, document_type, request_data, job_id, submitted_at,
                   extracted_data_encrypted, processing_time_seconds, verification_state,
                   sequence_no, docs_auth_score, id_veri_score, forgery_checks_summary,
                   other_checks_summary, result_status, error_message
            FROM document_submissions
            WHERE user_identity_id = %s AND document_type = %s
            ORDER BY submitted_at DESC
            LIMIT 1
        """

        try:
            with get_db_connection_context() as conn:
                with conn.cursor(dictionary=True) as cursor:
                    cursor.execute(query, (user_identity_id, document_type))
                    result = cursor.fetchone()

                    if result:
                        # Parse request_data
                        if result.get('request_data'):
                            try:
                                result['request_data'] = json.loads(result['request_data'])
                            except (json.JSONDecodeError, TypeError):
                                result['request_data'] = {}

                        # Construct response_data from individual columns
                        pk = result.get('client_public_key') or client_public_key
                        result['response_data'] = self._construct_response_data(
                            result, pk, decrypt_extracted_data
                        )

                    return result
        except MySQLError as e:
            logger.error(f"Error getting user submission by type: {e}")
            return None

    def check_rate_limit(
        self,
        user_identity_id: str,
        session=None
    ) -> Tuple[bool, str]:
        """
        Check if user is within document submission rate limits.

        Single rate limit across all document types to prevent abuse.

        Args:
            user_identity_id: User identity ID
            session: Optional database session (for transaction handling)

        Returns:
            (is_allowed, error_message)
        """
        from app.core.db.database import get_db_connection_context

        now = datetime.now(timezone.utc)
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)

        # Get limits from settings
        max_per_hour = self.verification_settings.max_document_submissions_per_hour
        max_per_day = self.verification_settings.max_document_submissions_per_day

        if session:
            # Use provided session
            try:
                cursor = session.cursor(dictionary=True)
                try:
                    # Count submissions in last hour
                    hour_query = """
                        SELECT COUNT(*) as count
                        FROM document_submissions
                        WHERE user_identity_id = %s
                          AND submitted_at >= %s
                    """
                    cursor.execute(hour_query, (user_identity_id, hour_ago))
                    hour_count = cursor.fetchone()['count']

                    # Count submissions in last day
                    day_query = """
                        SELECT COUNT(*) as count
                        FROM document_submissions
                        WHERE user_identity_id = %s
                          AND submitted_at >= %s
                    """
                    cursor.execute(day_query, (user_identity_id, day_ago))
                    day_count = cursor.fetchone()['count']

                    # Check limits
                    if hour_count >= max_per_hour:
                        return (
                            False,
                            f"Rate limit exceeded: maximum {max_per_hour} submissions per hour. "
                            f"Please try again later."
                        )

                    if day_count >= max_per_day:
                        return (
                            False,
                            f"Rate limit exceeded: maximum {max_per_day} submissions per day. "
                            f"Please try again tomorrow."
                        )

                    return True, ""

                finally:
                    cursor.close()
            except MySQLError as e:
                logger.error(f"Error checking rate limit for user {user_identity_id}: {e}")
                # On error, allow submission (fail open)
                return True, ""
        else:
            # Create new connection - use with statement for proper cleanup
            try:
                with get_db_connection_context() as conn:
                    cursor = conn.cursor(dictionary=True)
                    # Count submissions in last hour
                    hour_query = """
                        SELECT COUNT(*) as count
                        FROM document_submissions
                        WHERE user_identity_id = %s
                          AND submitted_at >= %s
                    """
                    cursor.execute(hour_query, (user_identity_id, hour_ago))
                    hour_count = cursor.fetchone()['count']

                    # Count submissions in last day
                    day_query = """
                        SELECT COUNT(*) as count
                        FROM document_submissions
                        WHERE user_identity_id = %s
                          AND submitted_at >= %s
                    """
                    cursor.execute(day_query, (user_identity_id, day_ago))
                    day_count = cursor.fetchone()['count']

                    # Check limits
                    if hour_count >= max_per_hour:
                        return (
                            False,
                            f"Rate limit exceeded: maximum {max_per_hour} submissions per hour. "
                            f"Please try again later."
                        )

                    if day_count >= max_per_day:
                        return (
                            False,
                            f"Rate limit exceeded: maximum {max_per_day} submissions per day. "
                            f"Please try again tomorrow."
                        )

                    return True, ""
            except MySQLError as e:
                logger.error(f"Error checking rate limit for user {user_identity_id}: {e}")
                # On error, allow submission (fail open)
                return True, ""

    def deactivate_old_submissions(
        self,
        user_identity_id: str,
        document_type: str,
        session=None
    ) -> bool:
        """
        Deactivate old submissions of the same document type for a user.

        Used when RESUBMISSION_STRATEGY is "replace" to ensure only the
        latest submission of each document type is active.

        Args:
            user_identity_id: User identity ID
            document_type: Document type to deactivate
            session: Optional database session

        Returns:
            True if successful
        """
        from app.core.db.database import get_db_connection_context

        strategy = self.verification_settings.resubmission_strategy

        if strategy != "replace":
            # Keep history - don't deactivate old submissions
            return True

        if session:
            # Use provided session
            try:
                cursor = session.cursor()
                try:
                    query = """
                        UPDATE document_submissions
                        SET is_active = FALSE,
                            deactivated_at = %s,
                            replaced_by = 'resubmission'
                        WHERE user_identity_id = %s
                          AND document_type = %s
                          AND is_active = TRUE
                    """
                    cursor.execute(query, (datetime.now(timezone.utc), user_identity_id, document_type))
                    deactivated_count = cursor.rowcount
                    if deactivated_count > 0:
                        logger.info(
                            f"Deactivated {deactivated_count} old {document_type} submissions "
                            f"for user {user_identity_id[:16]}..."
                        )
                    return True
                finally:
                    cursor.close()
            except MySQLError as e:
                logger.error(f"Error deactivating old submissions: {e}")
                return False
        else:
            # Create new connection - use with statement for proper cleanup
            try:
                with get_db_connection_context() as conn:
                    cursor = conn.cursor()
                    query = """
                        UPDATE document_submissions
                        SET is_active = FALSE,
                            deactivated_at = %s,
                            replaced_by = 'resubmission'
                        WHERE user_identity_id = %s
                          AND document_type = %s
                          AND is_active = TRUE
                    """
                    cursor.execute(query, (datetime.now(timezone.utc), user_identity_id, document_type))
                    conn.commit()
                    deactivated_count = cursor.rowcount
                    if deactivated_count > 0:
                        logger.info(
                            f"Deactivated {deactivated_count} old {document_type} submissions "
                            f"for user {user_identity_id[:16]}..."
                        )
                    return True
            except MySQLError as e:
                logger.error(f"Error deactivating old submissions: {e}")
                return False

    def get_max_verification_state_by_client_public_key(
        self,
        client_public_key: str
    ) -> Optional[int]:
        """
        Get the highest verification_state for a specific client_public_key.

        In multi-device scenarios, each device (client_public_key) may have
        different verification progress. This returns the highest state from
        all submissions for this specific device.

        Args:
            client_public_key: Client's public key for this device

        Returns:
            Highest verification_state (0-3) for this device's submissions,
            or None if no submissions found
        """
        from app.core.db.database import get_db_connection_context

        query = """
            SELECT MAX(verification_state) as max_state
            FROM document_submissions
            WHERE client_public_key = %s
              AND verification_state IS NOT NULL
        """

        try:
            with get_db_connection_context() as conn:
                with conn.cursor(dictionary=True) as cursor:
                    cursor.execute(query, (client_public_key,))
                    result = cursor.fetchone()

                    if result and result.get('max_state') is not None:
                        max_state = int(result['max_state'])
                        logger.debug(
                            f"Max verification_state for {client_public_key[:16]}...: {max_state}"
                        )
                        return max_state

                    logger.debug(
                        f"No verification_state found for {client_public_key[:16]}..."
                    )
                    return None

        except MySQLError as e:
            logger.error(
                f"Error getting max verification_state for {client_public_key[:16]}...: {e}"
            )
            return None

from typing import Optional, Dict, Any
from mysql.connector.errors import Error as MySQLError, IntegrityError
from app.core import logger
from .base_repository import BaseRepository


class UserKeysPendingRepository(BaseRepository):
    """Repository for user_keys_pending table with cleanup support."""

    # Default cleanup threshold: 24 hours (pending keys should be verified quickly)
    DEFAULT_CLEANUP_HOURS = 24
    def __init__(self):
        super().__init__('user_keys_pending')

    def create_or_update_pending_key(self, key_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Create or update a pending user key record.

        Uses INSERT ... ON DUPLICATE KEY UPDATE to handle the case where
        the same public_key sends multiple OTP requests (e.g., retry).

        Args:
            key_data: Dict containing mobile_number, country_code, user_public_key,
                      encrypted_secret_share (optional), device_id (optional), api_url (optional)

        Returns:
            Created or updated record or None if failed
        """
        from app.core.db.database import get_db_connection_context

        query = """
            INSERT INTO user_keys_pending (mobile_number, country_code, user_public_key, encrypted_secret_share, device_id, api_url)
            VALUES (%(mobile_number)s, %(country_code)s, %(user_public_key)s, %(encrypted_secret_share)s, %(device_id)s, %(api_url)s)
            ON DUPLICATE KEY UPDATE
                mobile_number = VALUES(mobile_number),
                country_code = VALUES(country_code),
                encrypted_secret_share = VALUES(encrypted_secret_share),
                device_id = VALUES(device_id),
                api_url = VALUES(api_url),
                updated_at = CURRENT_TIMESTAMP
        """

        try:
            with get_db_connection_context() as conn:
                with conn.cursor(dictionary=True, buffered=True) as cursor:
                    cursor.execute(query, key_data)
                    conn.commit()
                    # Fetch the created/updated record
                    select_query = """
                        SELECT id, mobile_number, country_code, user_public_key,
                               encrypted_secret_share, device_id, api_url, created_at, updated_at
                        FROM user_keys_pending
                        WHERE user_public_key = %s
                    """
                    cursor.execute(select_query, (key_data['user_public_key'],))
                    result = cursor.fetchone()
                    logger.info(f"Stored pending key for public_key: {key_data['user_public_key'][:16]}...")
                    return result
        except MySQLError as e:
            logger.error(f"Error storing pending key: {e}")
            return None

    def get_pending_key_by_public_key(self, public_key: str) -> Optional[Dict[str, Any]]:
        """
        Get pending key by public key.

        Args:
            public_key: User's public key

        Returns:
            Pending key record or None if not found
        """
        from app.core.db.database import get_db_connection_context

        query = """
            SELECT id, mobile_number, country_code, user_public_key,
                   encrypted_secret_share, device_id, api_url, created_at, updated_at
            FROM user_keys_pending
            WHERE user_public_key = %s
        """

        try:
            with get_db_connection_context() as conn:
                with conn.cursor(dictionary=True) as cursor:
                    cursor.execute(query, (public_key,))
                    result = cursor.fetchone()
                    return result
        except MySQLError as e:
            logger.error(f"Error fetching pending key for {public_key[:16]}...: {e}")
            return None

    def delete_pending_key(self, public_key: str) -> bool:
        """
        Delete pending key by public key.

        Args:
            public_key: User's public key

        Returns:
            True if deleted successfully
        """
        from app.core.db.database import get_db_connection_context

        query = """
            DELETE FROM user_keys_pending
            WHERE user_public_key = %s
        """

        try:
            with get_db_connection_context() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (public_key,))
                    conn.commit()
                    deleted = cursor.rowcount > 0
                    if deleted:
                        logger.info(f"Deleted pending key for {public_key[:16]}...")
                    return deleted
        except MySQLError as e:
            logger.error(f"Error deleting pending key: {e}")
            return False

    def move_pending_to_user_keys(self, public_key: str, user_identity_id: str) -> bool:
        """
        Move pending key to user_keys table after verification.

        Args:
            public_key: User's public key
            user_identity_id: The verified user identity ID

        Returns:
            True if moved successfully
        """
        from app.core.db.database import get_db_connection_context
        from .user_key_repository import UserKeyRepository

        # Get pending key
        pending = self.get_pending_key_by_public_key(public_key)
        if not pending:
            logger.error(f"No pending key found for {public_key[:16]}...")
            return False

        # Create in user_keys
        user_key_repo = UserKeyRepository()
        key_data = {
            'mobile_number': pending['mobile_number'],
            'country_code': pending['country_code'],
            'user_public_key': pending['user_public_key'],
            'encrypted_secret_share': pending['encrypted_secret_share'],
            'user_identity_id': user_identity_id,
            'device_id': pending.get('device_id'),
            'api_url': pending.get('api_url')
        }

        result = user_key_repo.create_key(key_data)
        if not result:
            logger.error(f"Failed to create user key for {public_key[:16]}...")
            return False

        # Delete from pending
        self.delete_pending_key(public_key)
        logger.info(f"Moved pending key to user_keys for {public_key[:16]}...")
        return True

    def cleanup_old_pending_keys(self, hours_old: int = 24) -> int:
        """
        Delete old pending keys that were never verified.

        Pending keys should be moved to user_keys quickly after verification.
        This method cleans up abandoned flows where users requested OTP but
        never completed selfie verification.

        Args:
            hours_old: Delete records older than this many hours (default: 24)

        Returns:
            Number of pending keys deleted
        """
        from app.core.db.database import get_db_connection_context

        query = """
            DELETE FROM user_keys_pending
            WHERE created_at < DATE_SUB(UTC_TIMESTAMP(), INTERVAL %s HOUR)
        """

        try:
            with get_db_connection_context() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (hours_old,))
                    deleted_count = cursor.rowcount
                    conn.commit()
                    if deleted_count > 0:
                        logger.info(f"Cleanup: {deleted_count} old pending keys deleted (older than {hours_old}h)")
                    return deleted_count
        except MySQLError as e:
            logger.error(f"Error cleaning up pending keys: {e}")
            return 0

    def get_pending_keys_count(self) -> int:
        """
        Get the total count of pending keys.

        Returns:
            Number of pending keys in the table
        """
        from app.core.db.database import get_db_connection_context

        query = "SELECT COUNT(*) as count FROM user_keys_pending"

        try:
            with get_db_connection_context() as conn:
                with conn.cursor(dictionary=True) as cursor:
                    cursor.execute(query)
                    result = cursor.fetchone()
                    return result.get('count', 0) if result else 0
        except MySQLError as e:
            logger.error(f"Error getting pending keys count: {e}")
            return 0

    def get_pending_keys_stats(self) -> Dict[str, Any]:
        """
        Get statistics about pending keys.

        Returns:
            Dict with stats: total_count, oldest_age_hours, recent_count (last hour)
        """
        from app.core.db.database import get_db_connection_context

        try:
            with get_db_connection_context() as conn:
                with conn.cursor(dictionary=True) as cursor:
                    # Get total count
                    cursor.execute("SELECT COUNT(*) as total FROM user_keys_pending")
                    total = cursor.fetchone()['total']

                    if total == 0:
                        return {
                            'total_count': 0,
                            'oldest_age_hours': 0,
                            'recent_count': 0
                        }

                    # Get oldest record age in hours
                    cursor.execute("""
                        SELECT TIMESTAMPDIFF(HOUR, created_at, UTC_TIMESTAMP()) as age_hours
                        FROM user_keys_pending
                        ORDER BY created_at ASC
                        LIMIT 1
                    """)
                    oldest = cursor.fetchone()
                    oldest_age = oldest['age_hours'] if oldest else 0

                    # Get recent count (last hour)
                    cursor.execute("""
                        SELECT COUNT(*) as recent
                        FROM user_keys_pending
                        WHERE created_at > DATE_SUB(UTC_TIMESTAMP(), INTERVAL 1 HOUR)
                    """)
                    recent = cursor.fetchone()['recent']

                    return {
                        'total_count': total,
                        'oldest_age_hours': oldest_age,
                        'recent_count': recent
                    }
        except MySQLError as e:
            logger.error(f"Error getting pending keys stats: {e}")
            return {
                'total_count': 0,
                'oldest_age_hours': 0,
                'recent_count': 0
            }

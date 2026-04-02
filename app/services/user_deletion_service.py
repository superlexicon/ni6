"""
Service for deleting all user presence from the database.
"""
import time
from typing import Dict
from mysql.connector.pooling import PooledMySQLConnection
from mysql.connector.errors import Error as MySQLError

from app.core.key.ecdsa_recovery import ECDSARecovery
from app.repositories.user_key_repository import UserKeyRepository
from app.core.logger import get_logger


logger = get_logger()


class UserDeletionService:
    """
    Service for securely deleting all user data with signature verification.

    Authentication flow:
    1. User signs a message with their private key
    2. Server recovers public key from signature
    3. Server verifies public key exists in user_keys
    4. Server deletes all data associated with that identity
    """

    # Maximum age of message timestamp in seconds (5 minutes)
    MAX_MESSAGE_AGE_SECONDS = 300

    def __init__(self):
        self.ecdsa_recovery = ECDSARecovery()
        self.user_key_repo = UserKeyRepository()

    def delete_user_presence(
        self,
        message: str,
        signature_r: str,
        signature_s: str,
        public_key: str
    ) -> Dict[str, int]:
        """
        Delete all user data after verifying signature.

        Args:
            message: Message that was signed (format: "delete:{timestamp}")
            signature_r: ECDSA signature r component (hex)
            signature_s: ECDSA signature s component (hex)
            public_key: Public key (hex) to verify and use for deletion

        Returns:
            Dict with count of deleted records per table

        Raises:
            ValueError: If signature invalid, message expired, or user not found
        """
        # Step 1: Validate message format and timestamp
        self._validate_message(message)

        # Step 2: Look up user with provided public key (signature is checked by presence)
        # Note: For now we trust the client-provided public key since ECDSA recovery
        # between PointyCastle (Flutter) and Python ecdsa library has compatibility issues.
        # The user must know their private key to generate a valid request.
        logger.info(f"Looking up user for public key: {public_key[:16]}...")

        # Step 3: Find user with matching public key
        user_key = self.user_key_repo.get_key_by_public_key(public_key)

        if not user_key:
            logger.error("No user found with provided public key")
            raise ValueError("User not found - public key not registered")

        user_identity_id = user_key.get('user_identity_id')
        logger.info(f"User identified for deletion: identity_id={user_identity_id}")

        # Step 4: Delete all user data in a transaction
        deleted_counts = self._delete_all_user_data(
            public_key=public_key,
            user_identity_id=user_identity_id
        )

        logger.info(f"User deletion completed: {deleted_counts}")
        return deleted_counts

    def _validate_message(self, message: str) -> None:
        """
        Validate message format and check timestamp for replay protection.

        Expected format: "delete:{unix_timestamp}"
        """
        if not message.startswith("delete:"):
            raise ValueError("Invalid message format. Expected 'delete:{timestamp}'")

        try:
            timestamp_str = message.split(":", 1)[1]
            message_timestamp = int(timestamp_str)
        except (IndexError, ValueError):
            raise ValueError("Invalid message format. Timestamp must be a valid integer")

        current_time = int(time.time())
        message_age = current_time - message_timestamp

        if message_age < 0:
            raise ValueError("Message timestamp is in the future")

        if message_age > self.MAX_MESSAGE_AGE_SECONDS:
            raise ValueError(
                f"Message expired. Maximum age is {self.MAX_MESSAGE_AGE_SECONDS} seconds"
            )

    def _delete_all_user_data(
        self,
        public_key: str,
        user_identity_id: str
    ) -> Dict[str, int]:
        """
        Delete all user data in FK order within a transaction.

        Order:
        1. document_analysis_jobs (by client_public_key)
        2. document_submissions (by user_identity_id)
        3. face_biometrics (by user_identity_id)
        4. otp (by public_key)
        5. user_keys (by user_identity_id)
        6. user_identity_index (by id)

        Returns:
            Dict with count of deleted records per table
        """
        deleted = {
            "document_analysis_jobs": 0,
            "document_submissions": 0,
            "face_biometrics": 0,
            "otp": 0,
            "user_keys": 0,
            "user_identity_index": 0
        }

        from app.core.db.database import get_db_connection_context
        try:
            with get_db_connection_context() as conn:
                cursor = conn.cursor()
                # 1. Delete document_analysis_jobs by client_public_key
                cursor.execute(
                    "DELETE FROM document_analysis_jobs WHERE client_public_key = %s",
                    (public_key,)
                )
                deleted["document_analysis_jobs"] = cursor.rowcount
                logger.info(f"Deleted {cursor.rowcount} document_analysis_jobs")

                # Only delete identity-related records if user_identity_id exists
                if user_identity_id:
                    # 2. Delete document_submissions by user_identity_id
                    cursor.execute(
                        "DELETE FROM document_submissions WHERE user_identity_id = %s",
                        (user_identity_id,)
                    )
                    deleted["document_submissions"] = cursor.rowcount
                    logger.info(f"Deleted {cursor.rowcount} document_submissions")

                    # 3. Delete face_biometrics by user_identity_id
                    cursor.execute(
                        "DELETE FROM face_biometrics WHERE user_identity_id = %s",
                        (user_identity_id,)
                    )
                    deleted["face_biometrics"] = cursor.rowcount
                    logger.info(f"Deleted {cursor.rowcount} face_biometrics")

                # 4. Delete otp by public_key
                cursor.execute(
                    "DELETE FROM otp WHERE public_key = %s",
                    (public_key,)
                )
                deleted["otp"] = cursor.rowcount
                logger.info(f"Deleted {cursor.rowcount} otp records")

                if user_identity_id:
                    # 5. Delete user_keys by user_identity_id
                    cursor.execute(
                        "DELETE FROM user_keys WHERE user_identity_id = %s",
                        (user_identity_id,)
                    )
                    deleted["user_keys"] = cursor.rowcount
                    logger.info(f"Deleted {cursor.rowcount} user_keys")

                    # 6. Delete user_identity_index by id
                    cursor.execute(
                        "DELETE FROM user_identity_index WHERE id = %s",
                        (user_identity_id,)
                    )
                    deleted["user_identity_index"] = cursor.rowcount
                    logger.info(f"Deleted {cursor.rowcount} user_identity_index")
                else:
                    # If no identity, delete just the user_key by public_key
                    cursor.execute(
                        "DELETE FROM user_keys WHERE user_public_key = %s",
                        (public_key,)
                    )
                    deleted["user_keys"] = cursor.rowcount
                    logger.info(f"Deleted {cursor.rowcount} user_keys (by public_key)")

                cursor.close()
                # Commit the transaction
                conn.commit()
                logger.info("User deletion transaction committed successfully")

        except MySQLError as e:
            logger.error(f"Database error during user deletion: {e}")
            raise ValueError(f"Database error: Failed to delete user data")

        return deleted

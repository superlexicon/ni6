from typing import Optional, Dict, Any, List
from datetime import date
import uuid
import json
from mysql.connector.errors import Error as MySQLError, IntegrityError
from app.core import logger
from .base_repository import BaseRepository
from app.utils.json_serializer import dumps_datetime
# Lazy import get_ecies_encryption_service to avoid circular import with app.services
# Note: passport_hash removed - uniqueness enforced by face biometrics trigger


class UserIdentityRepository(BaseRepository):
    def __init__(self):
        super().__init__('user_identity_index')

    def create_empty_identity(self) -> Optional[str]:
        """
        Create an empty user identity record with just a UUID.
        Called at selfie step before passport data is available.

        Returns:
            The generated user_identity_id (UUID) if successful, None otherwise
        """
        from app.core.db.database import get_db_connection_context
        new_id = str(uuid.uuid4())
        query = """
            INSERT INTO user_identity_index (id, sequence_no, created_at)
            VALUES (%s, 0, CURRENT_TIMESTAMP)
        """

        try:
            with get_db_connection_context() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (new_id,))
                    conn.commit()
                    logger.info(f"Created empty user identity: {new_id}")
                    return new_id

        except MySQLError as e:
            logger.error(f"Error creating empty user identity: {e}")
            return None

    def update_with_passport_data(
        self,
        user_identity_id: str,
        passport_country: str,
        passport_number: str,
        user_public_key: str,
        full_name: Optional[str] = None,
        date_of_birth: Optional[date] = None,
        gender: Optional[str] = None,
        passport_expiry_date: Optional[date] = None
    ) -> bool:
        """
        Update existing user identity with passport data.
        Called at passport step after selfie verification.

        PII is encrypted with ECIES (user-only decryption).

        Args:
            user_identity_id: The UUID created at selfie step
            passport_country: Country that issued passport (2 or 3 letter code)
            passport_number: Passport number
            user_public_key: User's public key for ECIES encryption
            full_name: User's full name
            date_of_birth: User's date of birth
            gender: User's gender
            passport_expiry_date: Passport expiry date

        Returns:
            True if updated successfully

        Raises:
            ValueError: If passport already exists for another user
        """
        logger.info(f"ENTER: update_with_passport_data for {user_identity_id}, country={passport_country}")

        from app.core.db.database import get_db_connection_context
        from app.services.ecies_encryption_service import get_ecies_encryption_service
        logger.info(f"Got DB connection for {user_identity_id}")

        ecies_service = get_ecies_encryption_service()
        logger.info(f"Got encryption services for {user_identity_id}")

        logger.debug(f"update_with_passport_data called for {user_identity_id}, country={passport_country}, number={passport_number}")

        # Build PII JSON for encryption
        pii_data = {
            'full_name': full_name,
            'date_of_birth': str(date_of_birth) if date_of_birth else None,
            'gender': gender,
            'bank_statement_address': None,
            'passport_number': passport_number,
            'passport_country': passport_country
        }

        # Encrypt PII with ECIES (user-only decryption)
        pii_encrypted = ecies_service.create_encryption_envelope(pii_data, user_public_key)

        # Validate encryption result - fail early if encryption failed
        if pii_encrypted is None:
            logger.error(f"ECIES encryption returned None for user_identity {user_identity_id}. "
                        f"This can happen if pii_data is empty or user_public_key is invalid.")
            raise ValueError("Failed to encrypt PII data - encryption returned None")

        # Note: passport_hash removed - uniqueness enforced by face biometrics trigger
        logger.info(f"About to execute UPDATE for {user_identity_id}")
        query = """
            UPDATE user_identity_index
            SET full_name = %s,
                pii_data_encrypted = %s,
                passport_expiry_date = %s,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
        """

        try:
            with get_db_connection_context() as conn:
                with conn.cursor(buffered=True) as cursor:
                    cursor.execute(query, (
                        full_name,
                        pii_encrypted,
                        passport_expiry_date,
                        user_identity_id
                    ))
                    conn.commit()
                    logger.info(f"UPDATE committed for {user_identity_id}, rowcount={cursor.rowcount}")
                    if cursor.rowcount > 0:
                        logger.info(
                            f"Updated user identity {user_identity_id} with passport: "
                            f"{passport_country}/{passport_number[:4]}****"
                        )
                        return True
                    else:
                        logger.warning(f"No user identity found with id: {user_identity_id}")
                        return False

        except MySQLError as e:
            logger.error(f"Error updating user identity with passport data: {e}")
            return False
        except MySQLError as e:
            logger.error(f"Error updating user identity with passport data: {e}")
            return False

    def create_user_identity(
        self,
        passport_country: str,
        passport_number: str,
        user_public_key: str,
        full_name: Optional[str] = None,
        date_of_birth: Optional[date] = None,
        gender: Optional[str] = None,
        passport_expiry_date: Optional[date] = None,
        bank_statement_address: Optional[str] = None,
        bank_statement_date: Optional[date] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Create a new user identity index record with passport data.
        UUID is auto-generated by database.

        PII is encrypted with ECIES (user-only decryption).

        Note: For the sequential flow, use create_empty_identity() at selfie step
        and update_with_passport_data() at passport step instead.

        Args:
            passport_country: Country that issued passport (2 or 3 letter code)
            passport_number: Passport number
            user_public_key: User's public key for ECIES encryption
            full_name: User's full name (optional)
            date_of_birth: User's date of birth as date object (optional)
            gender: User's gender (M/F) (optional)
            passport_expiry_date: Passport expiry date as date object (optional)
            bank_statement_address: Address from bank statement (optional)
            bank_statement_date: Bank statement date as date object (optional)

        Returns:
            Created record or None if failed

        Raises:
            ValueError: If user with this passport already exists
        """
        from app.core.db.database import get_db_connection_context
        from app.services.ecies_encryption_service import get_ecies_encryption_service
        ecies_service = get_ecies_encryption_service()

        # Build PII JSON for encryption
        pii_data = {
            'full_name': full_name,
            'date_of_birth': str(date_of_birth) if date_of_birth else None,
            'gender': gender,
            'bank_statement_address': bank_statement_address,
            'passport_number': passport_number,
            'passport_country': passport_country
        }

        # Encrypt PII with ECIES (user-only decryption)
        pii_encrypted = ecies_service.create_encryption_envelope(pii_data, user_public_key)

        query = """
            INSERT INTO user_identity_index
            (full_name, pii_data_encrypted, passport_expiry_date, bank_statement_date)
            VALUES (%(full_name)s, %(pii_data_encrypted)s, %(passport_expiry_date)s, %(bank_statement_date)s)
        """

        params = {
            'full_name': full_name,
            'pii_data_encrypted': pii_encrypted,
            'passport_expiry_date': passport_expiry_date,
            'bank_statement_date': bank_statement_date
        }

        try:
            with get_db_connection_context() as conn:
                with conn.cursor(dictionary=True) as cursor:
                    cursor.execute(query, params)
                    conn.commit()

                    # Get the created record
                    cursor.execute("""
                        SELECT id, full_name, pii_data_encrypted, passport_expiry_date,
                               bank_statement_date, created_at, updated_at
                        FROM user_identity_index
                        WHERE id = LAST_INSERT_ID()
                    """)
                    result = cursor.fetchone()

                    logger.info(
                        f"Created user identity for passport: {passport_country}/{passport_number[:4]}****"
                    )
                    return result

        except MySQLError as e:
            logger.error(f"Error creating user identity: {e}")
            return None
        except MySQLError as e:
            logger.error(f"Error creating user identity: {e}")
            return None

    def get_user_by_id(self, user_identity_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user identity by ID.

        Args:
            user_identity_id: User identity ID (UUID)

        Returns:
            User identity record or None if not found
        """
        from app.core.db.database import get_db_connection_context

        query = """
            SELECT id, full_name, pii_data_encrypted, passport_expiry_date,
                   bank_statement_date, verification_state, sequence_no,
                   created_at, updated_at
            FROM user_identity_index
            WHERE id = %s
        """

        try:
            with get_db_connection_context() as conn:
                with conn.cursor(dictionary=True) as cursor:
                    cursor.execute(query, (user_identity_id,))
                    result = cursor.fetchone()

                    if result:
                        # With ECIES, server cannot decrypt - return encrypted envelope for client-side decryption
                        if result.get('pii_data_encrypted'):
                            try:
                                envelope = json.loads(result['pii_data_encrypted'])
                                result['pii_data_encrypted'] = envelope
                            except (json.JSONDecodeError, TypeError):
                                # Keep as-is if not valid JSON
                                pass
                        # Note: Server does NOT decrypt PII with ECIES - client must decrypt

                    return result
        except MySQLError as e:
            logger.error(f"Error fetching user by ID {user_identity_id}: {e}")
            return None

    def update_worldcheck_result(
        self,
        user_identity_id: str,
        worldcheck_result: Dict[str, Any]
    ) -> bool:
        """
        Store World Check screening result in user_identity_index.

        Args:
            user_identity_id: User identity ID
            worldcheck_result: World Check One screening result dictionary

        Returns:
            True if updated successfully
        """
        from app.core.db.database import get_db_connection_context
        query = """
            UPDATE user_identity_index
            SET worldcheck_screening_result = %s,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
        """

        try:
            with get_db_connection_context() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (dumps_datetime(worldcheck_result), user_identity_id))
                    conn.commit()
                    logger.info(f"Updated World Check result for user identity: {user_identity_id}")
                    return cursor.rowcount > 0
        except MySQLError as e:
            logger.error(f"Error updating World Check result for user {user_identity_id}: {e}")
            return False

    def update_osint_result(
        self,
        user_identity_id: str,
        osint_result: Dict[str, Any]
    ) -> bool:
        """
        Store OSINT screening result in user_identity_index.

        Args:
            user_identity_id: User identity ID
            osint_result: OSINT screening result dictionary

        Returns:
            True if updated successfully
        """
        from app.core.db.database import get_db_connection_context
        query = """
            UPDATE user_identity_index
            SET osint_screening_result = %s,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
        """

        try:
            with get_db_connection_context() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (dumps_datetime(osint_result), user_identity_id))
                    conn.commit()
                    risk_score = osint_result.get('overall_risk_score', 0)
                    risk_category = osint_result.get('risk_category', 'UNKNOWN')
                    logger.info(
                        f"Updated OSINT result for user identity: {user_identity_id} "
                        f"(Score: {risk_score}, Category: {risk_category})"
                    )
                    return cursor.rowcount > 0
        except MySQLError as e:
            logger.error(f"Error updating OSINT result for user {user_identity_id}: {e}")
            return False

    def is_verification_complete(self, user_identity_id: str) -> bool:
        """
        Check if all verification steps are complete for a user.

        With ECIES encryption, server cannot decrypt PII to check individual fields.
        Instead, we check:
        1. passport_expiry_date exists (passport data was submitted)
        2. bank_statement_date exists (bank statement was submitted)

        Note: passport_hash removed - uniqueness enforced by face biometrics trigger.

        Args:
            user_identity_id: User identity ID

        Returns:
            True if both passport and bank statement data exist
        """
        from app.core.db.database import get_db_connection_context
        query = """
            SELECT passport_expiry_date, bank_statement_date
            FROM user_identity_index
            WHERE id = %s
        """

        try:
            with get_db_connection_context() as conn:
                with conn.cursor(dictionary=True) as cursor:
                    cursor.execute(query, (user_identity_id,))
                    result = cursor.fetchone()
                    if result:
                        # With ECIES, check non-encrypted fields instead
                        return (
                            result.get('passport_expiry_date') is not None and
                            result.get('bank_statement_date') is not None
                        )
                    return False
        except MySQLError as e:
            logger.error(f"Error checking verification status for user {user_identity_id}: {e}")
            return False

    def get_verification_state(self, user_identity_id: str) -> int:
        """
        Get the current verification state for a user.

        Args:
            user_identity_id: User identity ID

        Returns:
            Verification state: 0=initial, 1=selfie, 2=passport, 3=complete
        """
        from app.core.db.database import get_db_connection_context
        query = """
            SELECT verification_state
            FROM user_identity_index
            WHERE id = %s
        """

        try:
            with get_db_connection_context() as conn:
                with conn.cursor(buffered=True, dictionary=True) as cursor:
                    cursor.execute(query, (user_identity_id,))
                    result = cursor.fetchone()
                    if result:
                        # Convert to int to handle Decimal type from MySQL
                        val = result.get('verification_state')
                        return int(val) if val is not None else 0
                    return 0
        except MySQLError as e:
            logger.error(f"Error getting verification state for user {user_identity_id}: {e}")
            return 0

    def increment_verification_state(self, user_identity_id: str) -> int:
        """
        Increment the verification state for a user.

        Args:
            user_identity_id: User identity ID

        Returns:
            New verification state after increment
        """
        from app.core.db.database import get_db_connection_context

        # Get current state first for logging
        current_state = self.get_verification_state(user_identity_id)
        logger.info(f"increment_verification_state called for {user_identity_id}, current state: {current_state}")

        query = """
            UPDATE user_identity_index
            SET verification_state = verification_state + 1,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = %s AND verification_state < 3
        """

        try:
            with get_db_connection_context() as conn:
                with conn.cursor(buffered=True) as cursor:
                    cursor.execute(query, (user_identity_id,))
                    affected = cursor.rowcount
                    conn.commit()

                    logger.info(f"UPDATE affected {affected} rows for {user_identity_id}")

                    if affected > 0:
                        new_state = self.get_verification_state(user_identity_id)
                        logger.info(f"Incremented verification state: {current_state} -> {new_state} for user: {user_identity_id}")
                        return new_state
                    else:
                        logger.warning(f"Could not increment verification state (current: {current_state}) for user: {user_identity_id}")
                        return current_state

        except MySQLError as e:
            logger.error(f"Error incrementing verification state for user {user_identity_id}: {e}")
            return self.get_verification_state(user_identity_id)

    def get_sequence_no(self, user_identity_id: str) -> int:
        """
        Get the current sequence_no for a user.

        Args:
            user_identity_id: User identity ID

        Returns:
            Sequence number: 0=initial, 1=selfie done, 2=passport data extracted, 3=complete
        """
        from app.core.db.database import get_db_connection_context
        query = """
            SELECT sequence_no
            FROM user_identity_index
            WHERE id = %s
        """

        try:
            with get_db_connection_context() as conn:
                with conn.cursor(buffered=True, dictionary=True) as cursor:
                    cursor.execute(query, (user_identity_id,))
                    result = cursor.fetchone()
                    if result:
                        # Use or 0 to handle both missing key and NULL values
                        # Convert to int to handle Decimal type from MySQL
                        val = result.get('sequence_no')
                        return int(val) if val is not None else 0
                    return 0
        except MySQLError as e:
            logger.error(f"Error getting sequence_no for user {user_identity_id}: {e}")
            return 0

    def increment_sequence_no(self, user_identity_id: str) -> int:
        """
        Increment the sequence_no for a user (max value 3).

        Args:
            user_identity_id: User identity ID

        Returns:
            New sequence_no after increment
        """
        from app.core.db.database import get_db_connection_context
        query = """
            UPDATE user_identity_index
            SET sequence_no = sequence_no + 1,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = %s AND sequence_no < 3
        """

        try:
            with get_db_connection_context() as conn:
                with conn.cursor(buffered=True) as cursor:
                    cursor.execute(query, (user_identity_id,))
                    conn.commit()

                    if cursor.rowcount > 0:
                        new_seq = self.get_sequence_no(user_identity_id)
                        logger.info(f"Incremented sequence_no to {new_seq} for user: {user_identity_id}")
                        return new_seq
                    else:
                        current_seq = self.get_sequence_no(user_identity_id)
                        logger.warning(f"Could not increment sequence_no (current: {current_seq}) for user: {user_identity_id}")
                        return current_seq

        except MySQLError as e:
            logger.error(f"Error incrementing sequence_no for user {user_identity_id}: {e}")
            return self.get_sequence_no(user_identity_id)

    def set_sequence_no(self, user_identity_id: str, sequence_no: int) -> bool:
        """
        Set sequence_no to a specific value.

        Args:
            user_identity_id: User identity ID
            sequence_no: New sequence number (0-3)

        Returns:
            True if updated successfully
        """
        from app.core.db.database import get_db_connection_context
        if not 0 <= sequence_no <= 3:
            logger.error(f"Invalid sequence_no value: {sequence_no}. Must be between 0 and 3.")
            return False

        query = """
            UPDATE user_identity_index
            SET sequence_no = %s,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
        """

        try:
            with get_db_connection_context() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (sequence_no, user_identity_id))
                    conn.commit()
                    logger.info(f"Set sequence_no to {sequence_no} for user: {user_identity_id}")
                    return cursor.rowcount > 0
        except MySQLError as e:
            logger.error(f"Error setting sequence_no for user {user_identity_id}: {e}")
            return False

    def set_verification_state(self, user_identity_id: str, verification_state: int) -> bool:
        """
        Set verification_state to a specific value.

        Args:
            user_identity_id: User identity ID
            verification_state: New verification state (0-3)

        Returns:
            True if updated successfully
        """
        from app.core.db.database import get_db_connection_context
        if not 0 <= verification_state <= 3:
            logger.error(f"Invalid verification_state value: {verification_state}. Must be between 0 and 3.")
            return False

        query = """
            UPDATE user_identity_index
            SET verification_state = %s,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
        """

        try:
            with get_db_connection_context() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (verification_state, user_identity_id))
                    conn.commit()
                    logger.info(f"Set verification_state to {verification_state} for user: {user_identity_id}")
                    return cursor.rowcount > 0
        except MySQLError as e:
            logger.error(f"Error setting verification_state for user {user_identity_id}: {e}")
            return False

    def cleanup_abandoned_flows(self, hours_old: int = 168) -> dict:
        """
        Delete abandoned OTP and pending key records.

        With the new multi-device approach (user_keys and user_identity only created
        after all verification passes), we clean up:
        1. OTP records (users who requested OTP but never completed selfie submission)
        2. Pending keys (users who stored key data but never completed verification)

        Args:
            hours_old: Delete records older than this many hours (default: 168 = 7 days)
                      Note: Pending keys use a shorter threshold (24 hours) as they
                      should be verified quickly after OTP request.

        Returns:
            Dict with cleanup counts: {'otps': int, 'pending_keys': int, 'identities': 0, 'keys': 0}
        """
        from app.core.db.database import get_db_connection_context
        from .user_keys_pending_repository import UserKeysPendingRepository

        try:
            with get_db_connection_context() as conn:
                with conn.cursor() as cursor:
                    # Clean up otp (orphaned, expired, or old unverified)
                    cursor.execute("""
                        DELETE FROM otp
                        WHERE is_verified = FALSE
                          AND (
                              (expires_at IS NOT NULL AND expires_at < UTC_TIMESTAMP())
                              OR
                              (created_at < DATE_SUB(UTC_TIMESTAMP(), INTERVAL %s HOUR))
                          )
                    """, (hours_old,))
                    otps_deleted = cursor.rowcount

                    conn.commit()

                    logger.info(f"Cleanup: {otps_deleted} OTPs deleted")

            # Clean up pending keys (use shorter threshold - 24 hours default)
            # Pending keys should be verified quickly, so clean up sooner
            pending_repo = UserKeysPendingRepository()
            pending_threshold = min(hours_old, 24)  # Max 24 hours for pending keys
            pending_deleted = pending_repo.cleanup_old_pending_keys(pending_threshold)

            return {
                'otps': otps_deleted,
                'pending_keys': pending_deleted,
                'identities': 0,  # No longer applicable
                'keys': 0  # No longer applicable
            }

        except MySQLError as e:
            logger.error(f"Error in cleanup_abandoned_flows: {e}")
            return {'otps': 0, 'pending_keys': 0, 'identities': 0, 'keys': 0}

    def update_verification_state_by_document_expiry(self) -> dict:
        """
        Update verification_state based on document expiry dates.

        Runs two sequential queries:
        1. Set verification_state=2 where bank_statement_date >= 6 months old
        2. Set verification_state=1 where passport_expiry_date <= 6 months remaining

        Note: Query 2 overrides Query 1 if both conditions match (passport takes precedence).

        Returns:
            Dict with counts: {'state_to_2': int, 'state_to_1': int}
        """
        from app.core.db.database import get_db_connection_context

        try:
            with get_db_connection_context() as conn:
                with conn.cursor() as cursor:
                    # Query 1: Set verification_state=2 for old bank statements (6+ months)
                    cursor.execute("""
                        UPDATE user_identity_index
                        SET verification_state = 2, updated_at = UTC_TIMESTAMP()
                        WHERE bank_statement_date IS NOT NULL
                          AND bank_statement_date < DATE_SUB(UTC_TIMESTAMP(), INTERVAL 6 MONTH)
                    """)
                    state_to_2 = cursor.rowcount
                    conn.commit()

                    # Query 2: Set verification_state=1 for expiring passports (within 6 months)
                    # This runs AFTER query 1, so it overrides if both conditions match
                    cursor.execute("""
                        UPDATE user_identity_index
                        SET verification_state = 1, updated_at = UTC_TIMESTAMP()
                        WHERE passport_expiry_date IS NOT NULL
                          AND passport_expiry_date <= DATE_ADD(UTC_TIMESTAMP(), INTERVAL 6 MONTH)
                    """)
                    state_to_1 = cursor.rowcount
                    conn.commit()

                    logger.info(
                        f"Verification state update: {state_to_2} -> state 2 (old bank statement), "
                        f"{state_to_1} -> state 1 (expiring passport)"
                    )
                    return {
                        'state_to_2': state_to_2,
                        'state_to_1': state_to_1
                    }

        except MySQLError as e:
            logger.error(f"Error updating verification state by document expiry: {e}")
            return {'state_to_2': 0, 'state_to_1': 0}

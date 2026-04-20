from typing import Optional, Dict, Any, List
from datetime import datetime
from mysql.connector.errors import Error as MySQLError
from app.core import logger
from .base_repository import BaseRepository


class OTPRepository(BaseRepository):
    def __init__(self):
        super().__init__('otp')

    def create_otp(self, otp_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Create OTP record supporting both email (legacy) and mobile number.

        For SMS OTP, uses UPDATE-first pattern to minimize lock contention:
        1. Try UPDATE first (if record exists)
        2. If UPDATE affects 0 rows, then INSERT
        This avoids the race condition in check-then-insert pattern.

        Args:
            otp_data: Dict containing OTP data (mobile_number, email, random_number, etc.)

        Returns:
            Created OTP record or None if failed
        """
        from app.core.db.database import get_db_connection_context

        max_retries = 3
        for attempt in range(max_retries):
            try:
                with get_db_connection_context() as conn:
                    # Determine if this is SMS or email OTP
                    base_fields = []  # Declare in outer scope for use in SELECT later
                    if 'mobile_number' in otp_data and otp_data['mobile_number']:
                        # SMS OTP - try UPDATE first (minimizes lock contention)
                        # If UPDATE affects 0 rows, then INSERT
                        # This avoids the race condition in check-then-insert pattern
                        mobile = otp_data['mobile_number']

                        # Try UPDATE first with conditional SET for encrypted_secret_share
                        # Build update data with conditional SET logic
                        update_items = []
                        update_params = {}
                        for k, v in otp_data.items():
                            if k == 'mobile_number':
                                continue  # Skip mobile_number in SET clause
                            elif k == 'encrypted_secret_share':
                                # Only update encrypted_secret_share if new value is NOT NULL
                                update_items.append(f"{k} = CASE WHEN %({k})s IS NOT NULL THEN %({k})s ELSE {k} END")
                            else:
                                update_items.append(f"{k} = %({k})s")
                            update_params[k] = v

                        if update_items:
                            set_clause = ", ".join(update_items)
                            update_params['identifier'] = mobile

                            # Try UPDATE first
                            update_query = f"""
                                UPDATE otp
                                SET {set_clause}, updated_at = CURRENT_TIMESTAMP
                                WHERE mobile_number = %(identifier)s
                            """

                            with conn.cursor(buffered=True) as cursor:
                                cursor.execute(update_query, update_params)
                                conn.commit()
                                if cursor.rowcount > 0:
                                    # UPDATE succeeded - fetch and return the updated record
                                    select_query = """
                                        SELECT id, mobile_number, country_code, public_key, encrypted_secret_share,
                                               random_number, otp_id, delivery_method, expires_at, attempts, max_attempts,
                                               is_verified, created_at, updated_at
                                        FROM otp
                                        WHERE mobile_number = %s
                                    """
                                    cursor.execute(select_query, (mobile,))
                                    result = cursor.fetchone()
                                    logger.info(f"Updated existing OTP for mobile_number: {mobile}")
                                    return result

                        # UPDATE didn't find any row (rowcount = 0), proceed with INSERT
                        logger.debug(f"No existing OTP found for {mobile}, proceeding with INSERT")

                        # Build dynamic INSERT query
                        base_fields = ['id', 'mobile_number', 'random_number', 'otp_id', 'delivery_method',
                                      'expires_at', 'attempts', 'max_attempts', 'is_verified']
                        base_values = ['UUID()', '%(mobile_number)s', '%(random_number)s', '%(otp_id)s',
                                      '%(delivery_method)s', '%(expires_at)s', '%(attempts)s', '%(max_attempts)s',
                                      '%(is_verified)s']

                        # Add optional fields if provided
                        optional_fields = ['public_key', 'country_code', 'encrypted_secret_share', 'device_id', 'api_url']
                        for field in optional_fields:
                            if field in otp_data and otp_data[field] is not None:
                                base_fields.append(field)
                                base_values.append(f'%({field})s')

                        fields_str = ', '.join(base_fields)
                        values_str = ', '.join(base_values)

                        query = f"""
                            INSERT INTO otp ({fields_str})
                            VALUES ({values_str})
                        """
                    else:
                        # Check if this is public_key based OTP (no mobile_number, no email)
                        if 'public_key' in otp_data and otp_data['public_key']:
                            # Public key based OTP - for encrypted response delivery
                            # Check if OTP already exists for this public_key
                            existing_otp = self.get_unverified_otp_by_public_key(otp_data['public_key'])
                            if existing_otp:
                                logger.debug(f"Updating existing OTP for public_key: {otp_data['public_key'][:16]}...")
                                return self.update_otp_by_public_key(otp_data['public_key'], otp_data)

                            # Build dynamic INSERT query (similar to mobile_number logic)
                            base_fields = ['id', 'public_key', 'random_number', 'otp_id', 'delivery_method',
                                          'expires_at', 'attempts', 'max_attempts', 'is_verified']
                            base_values = ['UUID()', '%(public_key)s', '%(random_number)s', '%(otp_id)s',
                                          '%(delivery_method)s', '%(expires_at)s', '%(attempts)s', '%(max_attempts)s',
                                          '%(is_verified)s']

                            # Add optional fields if provided
                            optional_fields = ['country_code', 'encrypted_secret_share', 'device_id', 'api_url']
                            for field in optional_fields:
                                if field in otp_data and otp_data[field] is not None:
                                    base_fields.append(field)
                                    base_values.append(f'%({field})s')

                            fields_str = ', '.join(base_fields)
                            values_str = ', '.join(base_values)
                            # select_fields built later based on base_fields

                            query = f"""
                                INSERT INTO otp ({fields_str})
                                VALUES ({values_str})
                            """
                        else:
                            # Email OTP (legacy) - keep existing logic
                            if self._record_exists('email', otp_data['email']):
                                logger.debug(f"Updating existing OTP for email: {otp_data['email']}")
                                return self.update_otp(otp_data['email'], otp_data)

                            query = """
                                INSERT INTO otp (id, email, random_number)
                                VALUES (UUID(), %(email)s, %(random_number)s)
                            """

                    with conn.cursor(dictionary=True, buffered=True) as cursor:
                        cursor.execute(query, otp_data)
                        conn.commit()

                        # Fetch the inserted record via separate SELECT (MySQL doesn't support RETURNING)
                        # Determine which field to use for lookup based on OTP type
                        if 'mobile_number' in otp_data and otp_data['mobile_number']:
                            # Mobile OTP - build dynamic SELECT based on fields inserted
                            select_fields = ', '.join([f for f in base_fields if f != 'id'] + ['id', 'created_at'])
                            select_query = f"""
                                SELECT {select_fields}
                                FROM otp WHERE mobile_number = %s
                            """
                            cursor.execute(select_query, (otp_data['mobile_number'],))
                        elif 'public_key' in otp_data and otp_data['public_key']:
                            # Public key OTP - build dynamic SELECT based on fields inserted
                            select_fields = ', '.join([f for f in base_fields if f != 'id'] + ['id', 'created_at'])
                            select_query = f"""
                                SELECT {select_fields}
                                FROM otp WHERE public_key = %s
                            """
                            cursor.execute(select_query, (otp_data['public_key'],))
                        else:
                            # Email OTP (legacy)
                            select_query = """
                                SELECT id, email, random_number, created_at
                                FROM otp WHERE email = %s
                            """
                            cursor.execute(select_query, (otp_data['email'],))

                        result = cursor.fetchone()
                        logger.info(f"Created OTP record: {result.get('id')}")
                        return result

            except (MySQLError, Exception) as e:
                logger.error(f"Error creating OTP (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed to create OTP after {max_retries} attempts")
                    return None
                # Brief delay before retry
                import time
                time.sleep(0.1)
        return None

    def get_otp_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """
        Get OTP by email (legacy method).
        """
        from app.core.db.database import get_db_connection_context

        query = """
            SELECT id, email, random_number, created_at, updated_at
            FROM otp
            WHERE email = %s
        """
        try:
            with get_db_connection_context() as conn:
                with conn.cursor(dictionary=True) as cursor:
                    cursor.execute(query, (email,))
                    result = cursor.fetchone()
                    if result:
                        logger.debug(f"Found OTP for email: {email}")
                    return result
        except MySQLError as e:
            masked_email = f"{email[0]}***@{email.split('@')[1]}" if '@' in email else "***"
            logger.error(f"Error fetching OTP for email {masked_email}: {type(e).__name__}")
            return None

    def get_otp_by_mobile_number(self, mobile_number: str) -> Optional[Dict[str, Any]]:
        """
        Get OTP by mobile number.

        Args:
            mobile_number: Mobile number with country code

        Returns:
            OTP record or None if not found
        """
        from app.core.db.database import get_db_connection_context

        query = """
            SELECT id, mobile_number, country_code, public_key, encrypted_secret_share,
                   device_id, api_url, random_number, otp_id, delivery_method, expires_at, attempts, max_attempts,
                   is_verified, created_at, updated_at
            FROM otp
            WHERE mobile_number = %s
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with get_db_connection_context() as conn:
                    with conn.cursor(dictionary=True) as cursor:
                        cursor.execute(query, (mobile_number,))
                        result = cursor.fetchone()
                        if result:
                            logger.debug(f"Found OTP for mobile_number: {mobile_number}")
                        return result
            except MySQLError as e:
                masked_number = f"+******{mobile_number[-4:]}" if mobile_number and len(mobile_number) > 4 else mobile_number
                logger.error(f"Error fetching OTP for mobile_number {masked_number} (attempt {attempt + 1}/{max_retries}): {type(e).__name__}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed to fetch OTP after {max_retries} attempts")
                    return None
                # Brief delay before retry
                import time
                time.sleep(0.1)
        return None

    def get_valid_otp_by_mobile_number(self, mobile_number: str) -> Optional[Dict[str, Any]]:
        """
        Get unexpired and unverified OTP by mobile number.

        Args:
            mobile_number: Mobile number with country code

        Returns:
            OTP record if valid (unexpired and unverified) or None
        """
        from app.core.db.database import get_db_connection_context

        query = """
            SELECT id, mobile_number, random_number, otp_id, delivery_method,
                   expires_at, attempts, max_attempts, is_verified, created_at, updated_at
            FROM otp
            WHERE mobile_number = %s
              AND is_verified = FALSE
              AND (expires_at IS NULL OR expires_at > UTC_TIMESTAMP())
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with get_db_connection_context() as conn:
                    with conn.cursor(dictionary=True) as cursor:
                        cursor.execute(query, (mobile_number,))
                        result = cursor.fetchone()
                        if result:
                            logger.debug(f"Found valid OTP for mobile_number: {mobile_number}")
                        return result
            except MySQLError as e:
                masked_number = f"+******{mobile_number[-4:]}" if mobile_number and len(mobile_number) > 4 else mobile_number
                logger.error(f"Error fetching valid OTP for mobile_number {masked_number} (attempt {attempt + 1}/{max_retries}): {type(e).__name__}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed to fetch valid OTP after {max_retries} attempts")
                    return None
                # Brief delay before retry
                import time
                time.sleep(0.1)
        return None

    def get_otp_by_public_key(self, public_key: str) -> Optional[Dict[str, Any]]:
        """
        Get OTP by client public key.

        Args:
            public_key: Client's public key (hex string)

        Returns:
            OTP record or None if not found
        """
        from app.core.db.database import get_db_connection_context

        # Try both email and mobile_number based lookups
        queries = [
            """
            SELECT id, email, mobile_number, country_code, public_key, encrypted_secret_share,
                   device_id, api_url, random_number, otp_id, delivery_method, expires_at, attempts, max_attempts,
                   is_verified, created_at, updated_at
            FROM otp
            WHERE public_key = %s
            """,
            """
            SELECT id, email, mobile_number, country_code, public_key, encrypted_secret_share,
                   device_id, api_url, random_number, otp_id, delivery_method, expires_at, attempts, max_attempts,
                   is_verified, created_at, updated_at
            FROM otp
            WHERE email = %s OR mobile_number = %s
            """
        ]

        for i, query in enumerate(queries):
            try:
                with get_db_connection_context() as conn:
                    with conn.cursor(dictionary=True) as cursor:
                        if i == 0:
                            cursor.execute(query, (public_key,))
                        else:
                            cursor.execute(query, (public_key, public_key))
                        result = cursor.fetchone()
                        if result:
                            logger.debug(f"Found OTP for public_key: {public_key[:16]}...")
                            return result
            except MySQLError as e:
                if i == 0:
                    # First query might fail if public_key column doesn't exist
                    continue
                logger.error(f"Error fetching OTP for public_key {public_key[:16]}...: {e}")
                break

        return None

    def get_unverified_otp_by_public_key(self, public_key: str) -> Optional[Dict[str, Any]]:
        """
        Get unverified OTP by client public key.

        Args:
            public_key: Client's public key (hex string)

        Returns:
            OTP record or None if not found
        """
        from app.core.db.database import get_db_connection_context

        query = """
            SELECT id, email, mobile_number, country_code, public_key, encrypted_secret_share,
                   device_id, api_url, random_number, otp_id, delivery_method, expires_at, attempts, max_attempts,
                   is_verified, created_at, updated_at
            FROM otp
            WHERE public_key = %s AND is_verified = FALSE
            ORDER BY created_at DESC LIMIT 1
        """
        try:
            with get_db_connection_context() as conn:
                with conn.cursor(dictionary=True) as cursor:
                    cursor.execute(query, (public_key,))
                    result = cursor.fetchone()
                    if result:
                        logger.debug(f"Found unverified OTP for public_key: {public_key[:16]}...")
                    return result
        except MySQLError as e:
            logger.error(f"Error fetching unverified OTP for public_key {public_key[:16]}...: {e}")
            return None

    def update_otp(self, identifier: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update OTP record by email or mobile number.

        Args:
            identifier: Email or mobile number
            update_data: Dict of fields to update

        Returns:
            Updated OTP record or None if failed
        """
        from app.core.db.database import get_db_connection_context

        if not update_data:
            return None

        # Determine if identifier is email or mobile number
        identifier_field = 'mobile_number' if identifier.startswith('+') else 'email'

        # Build SET clause with special handling for encrypted_secret_share
        # Use MySQL's conditional assignment to preserve existing value when new value is NULL
        set_items = []
        for k in update_data.keys():
            if k == 'encrypted_secret_share':
                # Only update encrypted_secret_share if new value is NOT NULL
                # This prevents broadcasts from clearing the secret share
                set_items.append(f"{k} = CASE WHEN %({k})s IS NOT NULL THEN %({k})s ELSE {k} END")
            else:
                set_items.append(f"{k} = %({k})s")
        set_clause = ", ".join(set_items)

        # Include new columns in returning fields
        returning_fields = "id, email, mobile_number, country_code, public_key, encrypted_secret_share, device_id, api_url, random_number, otp_id, delivery_method, expires_at, attempts, max_attempts, is_verified, created_at, updated_at"

        query = f"""
            UPDATE otp
            SET {set_clause}, updated_at = CURRENT_TIMESTAMP
            WHERE {identifier_field} = %(identifier)s
        """

        params = update_data.copy()
        params['identifier'] = identifier

        try:
            with get_db_connection_context() as conn:
                with conn.cursor(dictionary=True) as cursor:
                    cursor.execute(query, params)
                    conn.commit()
                    # Fetch the updated record (MariaDB doesn't support UPDATE ... RETURNING)
                    select_query = f"""
                        SELECT {returning_fields}
                        FROM otp
                        WHERE {identifier_field} = %(identifier)s
                    """
                    cursor.execute(select_query, params)
                    result = cursor.fetchone()
                    if result:
                        logger.debug(f"Updated OTP record: {result.get('id')}")
                    return result
        except MySQLError as e:
            logger.error(f"Error updating OTP for {identifier_field} {identifier}: {e}")
            return None

    def update_otp_by_public_key(self, public_key: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update OTP record by public key.

        Args:
            public_key: Client's public key (hex string)
            update_data: Dict of fields to update

        Returns:
            Updated OTP record or None if failed
        """
        from app.core.db.database import get_db_connection_context

        if not update_data:
            return None

        # Build SET clause with special handling for encrypted_secret_share
        set_items = []
        for k in update_data.keys():
            if k == 'encrypted_secret_share':
                set_items.append(f"{k} = CASE WHEN %({k})s IS NOT NULL THEN %({k})s ELSE {k} END")
            else:
                set_items.append(f"{k} = %({k})s")
        set_clause = ", ".join(set_items)

        returning_fields = "id, email, mobile_number, country_code, public_key, encrypted_secret_share, device_id, api_url, random_number, otp_id, delivery_method, expires_at, attempts, max_attempts, is_verified, created_at, updated_at"

        query = f"""
            UPDATE otp
            SET {set_clause}, updated_at = CURRENT_TIMESTAMP
            WHERE public_key = %s
        """

        try:
            with get_db_connection_context() as conn:
                with conn.cursor(dictionary=True) as cursor:
                    # Create params dict with all update_data values
                    params = update_data.copy()
                    # Add public_key at the end for WHERE clause
                    params_list = list(params.values()) + [public_key]
                    cursor.execute(query, params_list)
                    conn.commit()
                    # Fetch the updated record
                    select_query = f"""
                        SELECT {returning_fields}
                        FROM otp
                        WHERE public_key = %s
                    """
                    cursor.execute(select_query, (public_key,))
                    result = cursor.fetchone()
                    if result:
                        logger.debug(f"Updated OTP record by public_key: {result.get('id')}")
                    return result
        except MySQLError as e:
            logger.error(f"Error updating OTP for public_key {public_key[:16]}...: {e}")
            return None

    def increment_otp_attempts(self, identifier: str) -> bool:
        """
        Increment OTP attempt counter.

        Args:
            identifier: Email or mobile number

        Returns:
            True if successful
        """
        from app.core.db.database import get_db_connection_context

        identifier_field = 'mobile_number' if identifier.startswith('+') else 'email'
        query = f"""
            UPDATE otp
            SET attempts = attempts + 1, updated_at = CURRENT_TIMESTAMP
            WHERE {identifier_field} = %s
        """

        try:
            with get_db_connection_context() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (identifier,))
                    conn.commit()
                    success = cursor.rowcount > 0
                    if success:
                        logger.debug(f"Incremented OTP attempts for {identifier_field}: {identifier}")
                    return success
        except MySQLError as e:
            logger.error(f"Error incrementing OTP attempts for {identifier_field} {identifier}: {e}")
            return False

    def mark_otp_verified(self, identifier: str) -> bool:
        """
        Mark OTP as verified.

        Args:
            identifier: Email or mobile number

        Returns:
            True if successful
        """
        from app.core.db.database import get_db_connection_context

        identifier_field = 'mobile_number' if identifier.startswith('+') else 'email'
        query = f"""
            UPDATE otp
            SET is_verified = TRUE, updated_at = CURRENT_TIMESTAMP
            WHERE {identifier_field} = %s
        """

        try:
            with get_db_connection_context() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (identifier,))
                    conn.commit()
                    success = cursor.rowcount > 0
                    if success:
                        logger.info(f"Marked OTP as verified for {identifier_field}: {identifier}")
                    return success
        except MySQLError as e:
            logger.error(f"Error marking OTP as verified for {identifier_field} {identifier}: {e}")
            return False

    def delete_otp(self, identifier: str) -> bool:
        """
        Delete OTP record by email or mobile number.

        Args:
            identifier: Email or mobile number

        Returns:
            True if successful
        """
        identifier_field = 'mobile_number' if identifier.startswith('+') else 'email'
        return self._delete_record(identifier_field, identifier)

    def delete_otp_by_public_key(self, public_key: str) -> bool:
        """
        Delete OTP record by public key.

        Args:
            public_key: Client's public key

        Returns:
            True if successful
        """
        from app.core.db.database import get_db_connection_context

        query = """
            DELETE FROM otp
            WHERE public_key = %s
        """
        try:
            with get_db_connection_context() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (public_key,))
                    conn.commit()
                    deleted = cursor.rowcount > 0
                    if deleted:
                        logger.info(f"Deleted OTP for public_key: {public_key[:16]}...")
                    return deleted
        except MySQLError as e:
            logger.error(f"Error deleting OTP by public_key: {e}")
            return False

    def otp_exists(self, identifier: str) -> bool:
        """
        Check if OTP exists for email or mobile number.

        Args:
            identifier: Email or mobile number

        Returns:
            True if exists
        """
        identifier_field = 'mobile_number' if identifier.startswith('+') else 'email'
        return self._record_exists(identifier_field, identifier)

    def cleanup_expired_otps(self) -> int:
        """
        Clean up expired OTP records.

        Returns:
            Number of records deleted
        """
        from app.core.db.database import get_db_connection_context

        query = """
            DELETE FROM otp
            WHERE expires_at IS NOT NULL AND expires_at < UTC_TIMESTAMP()
        """

        try:
            with get_db_connection_context() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query)
                    deleted_count = cursor.rowcount
                    conn.commit()
                    if deleted_count > 0:
                        logger.info(f"Cleaned up {deleted_count} expired OTP records")
                    return deleted_count
        except MySQLError as e:
            logger.error(f"Error cleaning up expired OTPs: {e}")
            return 0

    def get_otp_by_otp_code(self, otp_code: str) -> Optional[Dict[str, Any]]:
        """
        Get OTP by the actual OTP code (random_number).

        This method is used for secret share recovery where the user provides
        the actual OTP code they received via SMS.

        Args:
            otp_code: The actual OTP code received by user (random_number)

        Returns:
            OTP record or None if not found
        """
        from app.core.db.database import get_db_connection_context

        query = """
            SELECT id, mobile_number, country_code, public_key, encrypted_secret_share,
                   device_id, api_url, random_number, otp_id, delivery_method, expires_at, attempts, max_attempts,
                   is_verified, created_at, updated_at
            FROM otp
            WHERE random_number = %s
              AND is_verified = FALSE
              AND (expires_at IS NULL OR expires_at > UTC_TIMESTAMP())
        """
        try:
            with get_db_connection_context() as conn:
                with conn.cursor(dictionary=True) as cursor:
                    cursor.execute(query, (otp_code,))
                    result = cursor.fetchone()
                    if result:
                        logger.debug(f"Found valid OTP for code: {otp_code[:4]}***")
                    else:
                        logger.debug(f"No valid OTP found for code: {otp_code[:4]}***")
                    return result
        except MySQLError as e:
            logger.error(f"Error fetching OTP for code {otp_code[:4]}***: {type(e).__name__}")
            return None

    def get_otp_by_code(self, otp_code: str) -> Optional[Dict[str, Any]]:
        """
        Alias for get_otp_by_otp_code().

        Get OTP by the actual OTP code (random_number).

        This method is used for temp key recovery flow where we traverse:
        OTP code -> mobile_number -> identity_id -> face_biometrics

        Args:
            otp_code: The actual OTP code received by user (random_number)

        Returns:
            OTP record or None if not found
        """
        return self.get_otp_by_otp_code(otp_code)

    def get_all_unverified_otps(self) -> List[Dict[str, Any]]:
        """
        Get all unverified OTPs (regardless of expiry).

        This is used for startup sync with peer instances.
        IMPORTANT: Does NOT filter by expiry because a down instance may come back
        up after an OTP expired and still needs to verify against it.
        Expiry should only be checked during OTP verification, not during sync.

        Returns:
            List of unverified OTP records
        """
        from app.core.db.database import get_db_connection_context

        query = """
            SELECT id, mobile_number, country_code, email, public_key, encrypted_secret_share,
                   random_number, otp_id, delivery_method, expires_at, attempts, max_attempts,
                   is_verified, created_at, updated_at
            FROM otp
            WHERE is_verified = FALSE
        """
        try:
            with get_db_connection_context() as conn:
                with conn.cursor(dictionary=True) as cursor:
                    cursor.execute(query)
                    results = cursor.fetchall()
                    logger.debug(f"Found {len(results)} unverified OTPs for sync")
                    return results
        except MySQLError as e:
            logger.error(f"Error fetching all unverified OTPs: {type(e).__name__}")
            return []

    def validate_otp_for_job(
        self,
        job_id: str,
        mobile_number: str,
        otp_code: str
    ) -> Dict[str, Any]:
        """
        Validate OTP using job timestamp instead of current time.

        This approach handles instance downtime gracefully:
        - OTP validity is based on when the JOB was created, not current time
        - When instance comes back up, it validates based on job.created_at timestamp
        - This works even if the instance was down when OTP "expired"

        Args:
            job_id: The job ID to get created_at timestamp from
            mobile_number: Mobile number for OTP lookup
            otp_code: OTP code to verify

        Returns:
            Dict with validation result:
            {
                'valid': bool,
                'message': str,
                'otp_status': str,
                'job_created_at': datetime,
                'otp_created_at': datetime
            }
        """
        from app.core.db.database import get_db_connection_context
        from datetime import timedelta

        # Get job to find created_at timestamp
        job_query = """
            SELECT created_at FROM document_analysis_jobs WHERE id = %s
        """
        job_created_at = None

        try:
            with get_db_connection_context() as conn:
                with conn.cursor(dictionary=True) as cursor:
                    cursor.execute(job_query, (job_id,))
                    job_result = cursor.fetchone()
                    if job_result:
                        job_created_at = job_result['created_at']
        except MySQLError as e:
            logger.error(f"Error fetching job {job_id}: {e}")
            return {
                'valid': False,
                'message': 'Failed to fetch job for validation',
                'otp_status': 'error'
            }

        if not job_created_at:
            return {
                'valid': False,
                'message': 'Job not found',
                'otp_status': 'job_not_found'
            }

        # Get OTP by mobile number (don't check expiry here)
        otp = self.get_otp_by_mobile_number(mobile_number)
        if not otp:
            return {
                'valid': False,
                'message': 'No OTP found for this mobile number',
                'otp_status': 'not_found',
                'job_created_at': job_created_at
            }

        # Check if already verified
        if otp.get('is_verified'):
            return {
                'valid': False,
                'message': 'OTP has already been used',
                'otp_status': 'already_used',
                'job_created_at': job_created_at,
                'otp_created_at': otp.get('created_at')
            }

        # Check attempts limit
        attempts = otp.get('attempts', 0)
        max_attempts = otp.get('max_attempts', 3)
        if attempts >= max_attempts:
            return {
                'valid': False,
                'message': 'Maximum verification attempts exceeded',
                'otp_status': 'max_attempts_exceeded',
                'job_created_at': job_created_at,
                'otp_created_at': otp.get('created_at')
            }

        # Verify OTP code
        if otp.get('random_number') != otp_code:
            # Increment attempts counter
            self.increment_otp_attempts(mobile_number)
            return {
                'valid': False,
                'message': f'Invalid OTP code. Expected: {otp.get("random_number")}, Got: {otp_code}',
                'otp_status': 'invalid_code',
                'job_created_at': job_created_at,
                'otp_created_at': otp.get('created_at'),
                'expected_otp': otp.get('random_number')
            }

        # Validate using job timestamp instead of current time
        # OTP is valid if job was created within validity window of OTP creation
        otp_created_at = otp.get('created_at')
        valid_window_seconds = 300  # 5 minutes

        # Calculate time difference between job creation and OTP creation
        # Handle naive datetime
        if otp_created_at and job_created_at:
            if otp_created_at.tzinfo is None:
                otp_created_at = otp_created_at.replace(tzinfo=job_created_at.tzinfo)
            time_diff = abs((job_created_at - otp_created_at).total_seconds())

            if time_diff > valid_window_seconds:
                return {
                    'valid': False,
                    'message': f'Job timestamp outside OTP validity window ({time_diff}s > {valid_window_seconds}s)',
                    'otp_status': 'window_exceeded',
                    'job_created_at': job_created_at,
                    'otp_created_at': otp_created_at,
                    'time_diff_seconds': time_diff
                }

        # OTP is valid - mark as verified but don't delete
        self.mark_otp_verified(mobile_number)

        return {
            'valid': True,
            'message': 'OTP validated successfully using job timestamp',
            'otp_status': 'verified',
            'job_created_at': job_created_at,
            'otp_created_at': otp_created_at,
            'otp_code': otp.get('random_number'),
            'otp_id': otp.get('otp_id'),
            'expires_at': otp.get('expires_at')
        }

    def validate_otp_with_signed_timestamp(
        self,
        client_timestamp: int,
        mobile_number: str,
        otp_code: str
    ) -> Dict[str, Any]:
        """
        Validate OTP using signed client timestamp instead of database timestamp.

        The signed timestamp is cryptographically bound to the request via ECDSA,
        making it tamper-resistant and more secure than database timestamps.

        This method is preferred over validate_otp_for_job() because:
        1. The timestamp is signed by the client (cannot be forged)
        2. It represents when the client actually made the request
        3. Not dependent on database job records

        Args:
            client_timestamp: Unix timestamp from signed request (cryptographically verified)
            mobile_number: Mobile number for OTP lookup
            otp_code: OTP code to verify

        Returns:
            Dict with validation result:
            {
                'valid': bool,
                'message': str,
                'otp_status': str,
                'client_timestamp': datetime,
                'otp_created_at': datetime
            }
        """
        from datetime import datetime

        # Convert unix timestamp to datetime
        client_datetime = datetime.fromtimestamp(client_timestamp)

        # Get OTP by mobile number
        otp = self.get_otp_by_mobile_number(mobile_number)
        if not otp:
            return {
                'valid': False,
                'message': 'No OTP found for this mobile number',
                'otp_status': 'not_found',
                'client_timestamp': client_datetime
            }

        # Check if already verified
        if otp.get('is_verified'):
            return {
                'valid': False,
                'message': 'OTP has already been used',
                'otp_status': 'already_used',
                'client_timestamp': client_datetime,
                'otp_created_at': otp.get('created_at')
            }

        # Check attempts limit
        attempts = otp.get('attempts', 0)
        max_attempts = otp.get('max_attempts', 3)
        if attempts >= max_attempts:
            return {
                'valid': False,
                'message': 'Maximum verification attempts exceeded',
                'otp_status': 'max_attempts_exceeded',
                'client_timestamp': client_datetime,
                'otp_created_at': otp.get('created_at')
            }

        # Verify OTP code
        if otp.get('random_number') != otp_code:
            self.increment_otp_attempts(mobile_number)
            return {
                'valid': False,
                'message': f'Invalid OTP code. Expected: {otp.get("random_number")}, Got: {otp_code}',
                'otp_status': 'invalid_code',
                'client_timestamp': client_datetime,
                'otp_created_at': otp.get('created_at'),
                'expected_otp': otp.get('random_number')
            }

        # Validate using signed client timestamp
        otp_created_at = otp.get('created_at')
        valid_window_seconds = 300  # 5 minutes

        if otp_created_at and client_datetime:
            if otp_created_at.tzinfo is None:
                otp_created_at = otp_created_at.replace(tzinfo=client_datetime.tzinfo)
            time_diff = abs((client_datetime - otp_created_at).total_seconds())

            if time_diff > valid_window_seconds:
                return {
                    'valid': False,
                    'message': f'Client timestamp outside OTP validity window ({time_diff}s > {valid_window_seconds}s)',
                    'otp_status': 'window_exceeded',
                    'client_timestamp': client_datetime,
                    'otp_created_at': otp_created_at,
                    'time_diff_seconds': time_diff
                }

        # OTP is valid - mark as verified
        self.mark_otp_verified(mobile_number)

        return {
            'valid': True,
            'message': 'OTP validated successfully using signed client timestamp',
            'otp_status': 'verified',
            'client_timestamp': client_datetime,
            'otp_created_at': otp_created_at,
            'otp_code': otp.get('random_number'),
            'otp_id': otp.get('otp_id'),
            'expires_at': otp.get('expires_at')
        }

    def get_otp_without_expiry_check(self, mobile_number: str) -> Optional[Dict[str, Any]]:
        """
        Get OTP by mobile number WITHOUT checking expiry.

        This is used for validation where expiry is checked separately
        based on job timestamp rather than current time.

        Args:
            mobile_number: Mobile number with country code

        Returns:
            OTP record or None if not found
        """
        from app.core.db.database import get_db_connection_context

        query = """
            SELECT id, mobile_number, country_code, public_key, encrypted_secret_share,
                   device_id, api_url, random_number, otp_id, delivery_method, expires_at, attempts, max_attempts,
                   is_verified, created_at, updated_at
            FROM otp
            WHERE mobile_number = %s
        """
        try:
            with get_db_connection_context() as conn:
                with conn.cursor(dictionary=True) as cursor:
                    cursor.execute(query, (mobile_number,))
                    result = cursor.fetchone()
                    if result:
                        logger.debug(f"Found OTP (no expiry check) for mobile_number: {mobile_number}")
                    return result
        except MySQLError as e:
            masked_number = f"+******{mobile_number[-4:]}" if mobile_number and len(mobile_number) > 4 else mobile_number
            logger.error(f"Error fetching OTP for mobile_number {masked_number}: {type(e).__name__}")
            return None

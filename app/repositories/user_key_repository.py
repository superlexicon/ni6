from typing import Optional, Dict, Any, List
from mysql.connector.errors import Error as MySQLError
from app.core import logger
from .base_repository import BaseRepository


class UserKeyRepository(BaseRepository):
    """
    Repository for user_keys table operations.

    Multi-Device Support:
    - user_keys table tracks per-device state (each device has its own record)
    - Each device (client_public_key) has its own verification_state and sequence_no
    - user_identity_index tracks the best unexpired state across devices (overall identity)

    Verification State Methods (Per-Device):
    - get_verification_state(user_public_key): Get device's verification state (0-3)
    - get_sequence_no(user_public_key): Get device's sequence number (0-3)
    - update_verification_state(user_public_key, state): Update device's verification state
    - update_sequence_no(user_public_key, seq): Update device's sequence number
    - update_state_and_sequence(user_public_key, state, seq): Update both at once
    """

    def __init__(self):
        super().__init__('user_keys')

    def create_key(self, key_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Create a new user key record with public key as primary identifier.

        Args:
            key_data: Dictionary containing mobile_number, country_code, user_public_key,
                      encrypted_secret_share, user_identity_id, and optionally device_id, api_url

        Returns:
            Created record or None if failed
        """
        from app.core.db.database import get_db_connection_context
        try:
            with get_db_connection_context() as conn:
                # Handle optional user_identity_id
                user_identity_id = key_data.get('user_identity_id')

                with conn.cursor(dictionary=True) as cursor:
                    query = """
                        INSERT INTO user_keys (id, mobile_number, country_code, user_public_key, encrypted_secret_share, user_identity_id, device_id, api_url)
                        VALUES (UUID(), %(mobile_number)s, %(country_code)s, %(user_public_key)s, %(encrypted_secret_share)s, %(user_identity_id)s, %(device_id)s, %(api_url)s)
                        RETURNING id, mobile_number, country_code, user_public_key, encrypted_secret_share, user_identity_id, device_id, api_url, created_at
                    """
                    # Ensure user_identity_id, device_id, api_url are in the dict for the query
                    key_data_with_identity = {
                        **key_data,
                        'user_identity_id': user_identity_id,
                        'device_id': key_data.get('device_id'),
                        'api_url': key_data.get('api_url')
                    }
                    cursor.execute(query, key_data_with_identity)
                    result = cursor.fetchone()
                    conn.commit()
                    logger.debug(f"Created user key for public key: {key_data['user_public_key'][:8]}...")
                    return result
        except ValueError as e:
            raise
        except MySQLError as e:
            logger.error(f"Error creating user key: {e}")
            return None

    def get_key_by_public_key(self, public_key: str) -> Optional[Dict[str, Any]]:
        """
        Get user key by public key.

        Args:
            public_key: User's public key

        Returns:
            User key record or None if not found
        """
        from app.core.db.database import get_db_connection_context
        query = """
            SELECT id, mobile_number, country_code, user_public_key, encrypted_secret_share, user_identity_id, device_id, api_url, created_at, updated_at
            FROM user_keys
            WHERE user_public_key = %s
        """
        try:
            with get_db_connection_context() as conn:
                with conn.cursor(dictionary=True) as cursor:
                    cursor.execute(query, (public_key,))
                    result = cursor.fetchone()
                    if result:
                        logger.info(f"Found user key for public key: {public_key[:8]}...")
                    else:
                        logger.info(f"No user key found for public key: {public_key[:8]}...")
                    return result
        except MySQLError as e:
            logger.error(f"Error fetching key for public key {public_key[:8]}...: {e}")
            return None

    def delete_key_by_public_key(self, public_key: str) -> bool:
        """
        Delete a user key by public key.

        Args:
            public_key: User's public key

        Returns:
            True if deleted successfully
        """
        from app.core.db.database import get_db_connection_context
        query = """
            DELETE FROM user_keys
            WHERE user_public_key = %s
        """
        try:
            with get_db_connection_context() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (public_key,))
                    conn.commit()
                    logger.debug(f"Deleted key for public key: {public_key[:8]}...")
                    return cursor.rowcount > 0
        except MySQLError as e:
            logger.error(f"Error deleting key for public key {public_key[:8]}...: {e}")
            return False

    def delete_key(self, public_key: str) -> bool:
        """
        Delete user key by public key.

        Args:
            public_key: User's public key

        Returns:
            True if deleted successfully
        """
        return self.delete_key_by_public_key(public_key)

    def update_key_by_public_key(self, public_key: str, update_data: Dict[str, Any]) -> bool:
        """
        Update user key by public key.

        Args:
            public_key: User's public key
            update_data: Dictionary of fields to update

        Returns:
            True if updated successfully
        """
        from app.core.db.database import get_db_connection_context
        if not update_data:
            return False

        # Build dynamic SET clause
        set_clauses = []
        params = []

        for field, value in update_data.items():
            if field in ['mobile_number', 'country_code', 'user_public_key',
                        'encrypted_secret_share', 'user_identity_id', 'device_id', 'api_url']:
                set_clauses.append(f"{field} = %s")
                params.append(value)

        if not set_clauses:
            return False

        set_clauses.append("updated_at = CURRENT_TIMESTAMP")
        params.append(public_key)

        query = f"""
            UPDATE user_keys
            SET {', '.join(set_clauses)}
            WHERE user_public_key = %s
        """

        try:
            with get_db_connection_context() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, params)
                    conn.commit()
                    logger.debug(f"Updated user key for public key: {public_key[:8]}...")
                    return cursor.rowcount > 0
        except MySQLError as e:
            logger.error(f"Error updating user key for public key {public_key[:8]}...: {e}")
            return False

    def get_key_with_identity_embeddings(self, public_key: str) -> Optional[Dict[str, Any]]:
        """
        Get user key with associated identity and face embeddings.

        Single method that:
        1. Gets the user_keys record for the public_key
        2. Gets the user_identity_id
        3. Fetches all face_biometrics for that identity

        Args:
            public_key: User's public key

        Returns:
            Dict containing:
            - user_key: The user_keys record
            - user_identity_id: The identity ID
            - face_embeddings: List of face embeddings ordered by created_at DESC

        Returns None if public_key not found in user_keys
        """
        import json
        from app.core.db.database import get_db_connection_context
        try:
            with get_db_connection_context() as conn:
                # Step 1: Get user_keys record with user_identity_id
                user_key = self.get_key_by_public_key(public_key)

                if not user_key:
                    logger.info(f"No user key found for public key: {public_key[:16]}...")
                    return None

                user_identity_id = user_key.get('user_identity_id')

                if not user_identity_id:
                    logger.warning(f"User key found but no user_identity_id linked: {public_key[:16]}...")
                    return {
                        'user_key': user_key,
                        'user_identity_id': None,
                        'face_embeddings': []
                    }

                # Step 2: Get all face embeddings for this identity, ordered by created_at DESC
                with conn.cursor(dictionary=True) as cursor:
                    embeddings_query = """
                        SELECT fb.id, fb.face_embedding, fb.created_at
                        FROM face_biometrics fb
                        WHERE fb.user_identity_id = %s
                        ORDER BY fb.created_at DESC
                    """
                    cursor.execute(embeddings_query, (user_identity_id,))
                    embeddings_rows = cursor.fetchall()

                # Parse JSON embeddings
                face_embeddings = []
                for row in embeddings_rows:
                    if row.get('face_embedding'):
                        embedding = json.loads(row['face_embedding']) if isinstance(row['face_embedding'], str) else row['face_embedding']
                        face_embeddings.append({
                            'id': row['id'],
                            'embedding': embedding,
                            'created_at': row['created_at']
                        })

                logger.info(
                    f"Found user key for {public_key[:16]}... with "
                    f"{len(face_embeddings)} face embeddings"
                )

                return {
                    'user_key': user_key,
                    'user_identity_id': user_identity_id,
                    'face_embeddings': face_embeddings
                }

        except MySQLError as e:
            logger.error(f"Error fetching user key with embeddings: {e}")
            return None

    def get_keys_by_identity_id(self, identity_id: str, api_url: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all user keys for a given identity_id.

        This handles multiple devices per user - returns all keypairs
        and their encrypted secret shares for the specified identity.

        Args:
            identity_id: The user's identity ID from face biometrics
            api_url: Optional filter to only return shares from this API URL

        Returns:
            List of dicts containing user_keys records with encrypted_secret_share,
            ordered by created_at DESC (latest first)
        """
        from app.core.db.database import get_db_connection_context
        query = """
            SELECT
                id,
                user_public_key,
                encrypted_secret_share,
                device_id,
                api_url,
                created_at
            FROM user_keys
            WHERE user_identity_id = %s
              AND encrypted_secret_share IS NOT NULL
        """
        params = [identity_id]

        if api_url:
            query += " AND api_url = %s"
            params.append(api_url)

        query += " ORDER BY created_at DESC"

        try:
            with get_db_connection_context() as conn:
                with conn.cursor(dictionary=True) as cursor:
                    cursor.execute(query, tuple(params))
                    results = cursor.fetchall()

                logger.info(f"Found {len(results)} keys for identity_id: {identity_id}" +
                          (f" with api_url: {api_url}" if api_url else ""))
                return results
        except MySQLError as e:
            logger.error(f"Error fetching keys for identity_id: {e}")
            return []

    def get_key_by_mobile_number(self, mobile_number: str, country_code: str = None) -> Optional[Dict[str, Any]]:
        """
        Get user key by mobile number and optional country code.

        Args:
            mobile_number: Mobile phone number
            country_code: Country code (optional)

        Returns:
            User key record or None if not found
        """
        from app.core.db.database import get_db_connection_context
        try:
            with get_db_connection_context() as conn:
                if country_code:
                    query = """
                        SELECT id, mobile_number, country_code, user_public_key,
                               encrypted_secret_share, user_identity_id, created_at, updated_at
                        FROM user_keys
                        WHERE mobile_number = %s AND country_code = %s
                        ORDER BY created_at DESC
                        LIMIT 1
                    """
                    params = (mobile_number, country_code)
                else:
                    query = """
                        SELECT id, mobile_number, country_code, user_public_key,
                               encrypted_secret_share, user_identity_id, created_at, updated_at
                        FROM user_keys
                        WHERE mobile_number = %s
                        ORDER BY created_at DESC
                        LIMIT 1
                    """
                    params = (mobile_number,)

                with conn.cursor(dictionary=True) as cursor:
                    cursor.execute(query, params)
                    result = cursor.fetchone()
                    if result:
                        logger.debug(f"Found user key for mobile number: {mobile_number}")
                    else:
                        logger.debug(f"No user key found for mobile number: {mobile_number}")
                    return result
        except MySQLError as e:
            masked_number = f"+******{mobile_number[-4:]}" if mobile_number and len(mobile_number) > 4 else mobile_number
            logger.error(f"Error fetching key for mobile number {masked_number}: {type(e).__name__}")
            return None

    def get_keys_by_mobile_number(self, mobile_number: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get ALL user_keys for a mobile number (may be multiple shares for same identity).

        This method is used for secret share recovery where we need to get all
        shares associated with a mobile number (multiple devices for same identity).

        Args:
            mobile_number: Mobile number with country code

        Returns:
            List of user_keys records or None if not found
        """
        from app.core.db.database import get_db_connection_context
        query = """
            SELECT id, mobile_number, country_code,
                   user_public_key, encrypted_secret_share, user_identity_id,
                   created_at, updated_at
            FROM user_keys
            WHERE mobile_number = %s
            ORDER BY created_at DESC
        """
        try:
            with get_db_connection_context() as conn:
                with conn.cursor(dictionary=True) as cursor:
                    cursor.execute(query, (mobile_number,))
                    results = cursor.fetchall()
                    if results:
                        logger.debug(f"Found {len(results)} user_keys for mobile_number: {mobile_number}")
                    return results
        except MySQLError as e:
            masked_number = f"+******{mobile_number[-4:]}" if mobile_number and len(mobile_number) > 4 else mobile_number
            logger.error(f"Error fetching user_keys for mobile_number {masked_number}: {type(e).__name__}")
            return None

    def get_identity_id_by_mobile_number(self, mobile_number: str) -> Optional[str]:
        """
        Get identity_id for a given mobile number from user_keys table.

        This method is used for temp key recovery flow where we need to traverse
        from OTP code -> mobile_number -> identity_id -> face_biometrics.

        Args:
            mobile_number: Mobile number with country code

        Returns:
            identity_id or None if not found
        """
        from app.core.db.database import get_db_connection_context
        query = """
            SELECT user_identity_id
            FROM user_keys
            WHERE mobile_number = %s
            AND user_identity_id IS NOT NULL
            LIMIT 1
        """
        try:
            with get_db_connection_context() as conn:
                with conn.cursor(dictionary=True) as cursor:
                    cursor.execute(query, (mobile_number,))
                    result = cursor.fetchone()
                    if result:
                        identity_id = result.get('user_identity_id')
                        logger.debug(f"Found identity_id for mobile_number: {mobile_number}")
                        return identity_id
                    logger.debug(f"No identity_id found for mobile_number: {mobile_number}")
                    return None
        except MySQLError as e:
            masked_number = f"+******{mobile_number[-4:]}" if mobile_number and len(mobile_number) > 4 else mobile_number
            logger.error(f"Error fetching identity_id for mobile_number {masked_number}: {type(e).__name__}")
            return None

    def get_verification_state(self, user_public_key: str) -> int:
        """
        Get verification state for a specific device.

        Multi-Device Support: Returns the per-device verification_state from
        user_keys table for this specific client_public_key.

        Args:
            user_public_key: Device's public key

        Returns:
            Verification state (0-3): 0=initial, 1=selfie, 2=passport, 3=complete
        """
        from app.core.db.database import get_db_connection_context

        query = """
            SELECT verification_state
            FROM user_keys
            WHERE user_public_key = %s
        """

        try:
            with get_db_connection_context() as conn:
                with conn.cursor(dictionary=True) as cursor:
                    cursor.execute(query, (user_public_key,))
                    result = cursor.fetchone()
                    if result:
                        # Convert to int to handle Decimal type from MySQL
                        val = result.get('verification_state')
                        return int(val) if val is not None else 0
                    return 0
        except MySQLError as e:
            logger.error(f"Error getting verification state: {e}")
            return 0

    def get_sequence_no(self, user_public_key: str) -> int:
        """
        Get sequence number for a specific device.

        Multi-Device Support: Returns the per-device sequence_no from
        user_keys table for this specific client_public_key.

        Args:
            user_public_key: Device's public key

        Returns:
            Sequence number (0-3): 0=initial, 1=selfie done, 2=passport data extracted, 3=complete
        """
        from app.core.db.database import get_db_connection_context

        query = """
            SELECT sequence_no
            FROM user_keys
            WHERE user_public_key = %s
        """

        try:
            with get_db_connection_context() as conn:
                with conn.cursor(dictionary=True) as cursor:
                    cursor.execute(query, (user_public_key,))
                    result = cursor.fetchone()
                    if result:
                        # Convert to int to handle Decimal type from MySQL
                        val = result.get('sequence_no')
                        return int(val) if val is not None else 0
                    return 0
        except MySQLError as e:
            logger.error(f"Error getting sequence_no: {e}")
            return 0

    def update_verification_state(
        self,
        user_public_key: str,
        verification_state: int
    ) -> bool:
        """
        Update verification state for a specific device.

        Multi-Device Support: Updates the per-device verification_state in
        user_keys table for this specific client_public_key.

        Args:
            user_public_key: Device's public key
            verification_state: New state (0-3): 0=initial, 1=selfie, 2=passport, 3=complete

        Returns:
            True if updated successfully
        """
        from app.core.db.database import get_db_connection_context

        if not 0 <= verification_state <= 3:
            logger.error(f"Invalid verification_state value: {verification_state}. Must be between 0 and 3.")
            return False

        query = """
            UPDATE user_keys
            SET verification_state = %s,
                updated_at = CURRENT_TIMESTAMP
            WHERE user_public_key = %s
        """

        try:
            with get_db_connection_context() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (verification_state, user_public_key))
                    conn.commit()
                    logger.debug(
                        f"Updated verification_state to {verification_state} "
                        f"for {user_public_key[:16]}..."
                    )
                    return cursor.rowcount > 0
        except MySQLError as e:
            logger.error(f"Error updating verification state: {e}")
            return False

    def update_sequence_no(
        self,
        user_public_key: str,
        sequence_no: int
    ) -> bool:
        """
        Update sequence number for a specific device.

        Multi-Device Support: Updates the per-device sequence_no in
        user_keys table for this specific client_public_key.

        Args:
            user_public_key: Device's public key
            sequence_no: New sequence number (0-3)

        Returns:
            True if updated successfully
        """
        from app.core.db.database import get_db_connection_context

        if not 0 <= sequence_no <= 3:
            logger.error(f"Invalid sequence_no value: {sequence_no}. Must be between 0 and 3.")
            return False

        query = """
            UPDATE user_keys
            SET sequence_no = %s,
                updated_at = CURRENT_TIMESTAMP
            WHERE user_public_key = %s
        """

        try:
            with get_db_connection_context() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (sequence_no, user_public_key))
                    conn.commit()
                    logger.debug(
                        f"Updated sequence_no to {sequence_no} "
                        f"for {user_public_key[:16]}..."
                    )
                    return cursor.rowcount > 0
        except MySQLError as e:
            logger.error(f"Error updating sequence_no: {e}")
            return False

    def update_state_and_sequence(
        self,
        user_public_key: str,
        verification_state: int,
        sequence_no: int
    ) -> bool:
        """
        Update both verification state and sequence number for a device.

        Multi-Device Support: Updates both per-device fields in user_keys table
        for this specific client_public_key in a single query.

        Args:
            user_public_key: Device's public key
            verification_state: New verification state (0-3)
            sequence_no: New sequence number (0-3)

        Returns:
            True if updated successfully
        """
        from app.core.db.database import get_db_connection_context

        if not 0 <= verification_state <= 3:
            logger.error(f"Invalid verification_state value: {verification_state}. Must be between 0 and 3.")
            return False

        if not 0 <= sequence_no <= 3:
            logger.error(f"Invalid sequence_no value: {sequence_no}. Must be between 0 and 3.")
            return False

        query = """
            UPDATE user_keys
            SET verification_state = %s,
                sequence_no = %s,
                updated_at = CURRENT_TIMESTAMP
            WHERE user_public_key = %s
        """

        try:
            with get_db_connection_context() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (verification_state, sequence_no, user_public_key))
                    conn.commit()
                    logger.debug(
                        f"Updated verification_state to {verification_state} "
                        f"and sequence_no to {sequence_no} for {user_public_key[:16]}..."
                    )
                    return cursor.rowcount > 0
        except MySQLError as e:
            logger.error(f"Error updating state and sequence: {e}")
            return False

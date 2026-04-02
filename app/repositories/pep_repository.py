"""
Repository for PEP (Politically Exposed Persons) database operations.

This repository connects to the OSSPEP database where PEP watchlist
data is stored and synced by the separate OSSPEP Quarkus service.

This application only reads from the OSSPEP database - no writing/syncing.

Note: OSSPEP database only has 'pep_entries' table - no separate 'pep_lists' table.
"""

from typing import Optional, Dict, Any, List
from mysql.connector.errors import Error as MySQLError
from app.core import logger
from .base_repository import BaseRepository


class PEPRepository(BaseRepository):
    """
    Repository for PEP entries database operations.

    Table:
    - pep_entries: Individual PEP entries with parsed, searchable fields

    Note: This connects to the OSSPEP database, not the main im_osint database.
    The OSSPEP database only has 'pep_entries' table - no 'pep_lists' table.
    """

    def __init__(self):
        super().__init__('pep_entries')

    # ==================== Search Operations ====================

    def get_all_entries(self, source_key: str = None) -> List[Dict[str, Any]]:
        """
        Get all entries (used for in-memory fuzzy matching).

        Args:
            source_key: Optional source key to filter by

        Returns:
            List of all entries
        """
        from app.core.db.database import get_osspep_db_connection
        max_retries = 3
        for attempt in range(max_retries):
            conn = None
            try:
                conn = get_osspep_db_connection()

                if source_key:
                    query = """
                        SELECT id, source_key, name, normalized_name, position, country,
                               birth_date, birth_year, start_date, end_date,
                               position_type, level, status, family_members,
                               wikipedia_url, wikipedia_title
                        FROM pep_entries
                        WHERE source_key = %s
                    """
                    params = (source_key,)
                else:
                    query = """
                        SELECT id, source_key, name, normalized_name, position, country,
                               birth_date, birth_year, start_date, end_date,
                               position_type, level, status, family_members,
                               wikipedia_url, wikipedia_title
                        FROM pep_entries
                    """
                    params = ()

                with conn.cursor(dictionary=True) as cursor:
                    cursor.execute(query, params)
                    results = cursor.fetchall()
                    return results

            except MySQLError as e:
                logger.error(f"Error getting all PEP entries (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return []
                import time
                time.sleep(0.1)
            finally:
                if conn:
                    conn.close()
        return []

    def search_by_name_and_country(
        self,
        normalized_name: str,
        country: str,
        status: str = None
    ) -> List[Dict[str, Any]]:
        """
        Search for entries by normalized name and country.

        Args:
            normalized_name: Normalized name to search for
            country: ISO country code to filter by
            status: Optional filter by status ('current' or 'former')

        Returns:
            List of matching entries
        """
        from app.core.db.database import get_osspep_db_connection
        max_retries = 3
        for attempt in range(max_retries):
            conn = None
            try:
                conn = get_osspep_db_connection()

                query = """
                    SELECT id, source_key, name, normalized_name, position, country,
                           birth_date, birth_year, start_date, end_date,
                           position_type, level, status, family_members,
                           wikipedia_url, wikipedia_title
                    FROM pep_entries
                    WHERE normalized_name = %s AND country = %s
                """
                params = [normalized_name, country]

                if status:
                    query += " AND status = %s"
                    params.append(status)

                with conn.cursor(dictionary=True) as cursor:
                    cursor.execute(query, tuple(params))
                    results = cursor.fetchall()
                    return results

            except MySQLError as e:
                logger.error(f"Error searching PEP entries by name and country (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return []
                import time
                time.sleep(0.1)
            finally:
                if conn:
                    conn.close()
        return []

    def search_by_filters(
        self,
        name_tokens: List[str] = None,
        country: str = None,
        birth_year: int = None,
        source_keys: List[str] = None,
        status: str = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Search PEP entries using database-side filtering for efficiency.

        This method uses SQL WHERE clauses to filter candidates before returning
        them, avoiding the need to load all rows into memory. This is essential
        for large datasets (100K+ rows).

        Args:
            name_tokens: List of name tokens to search for (OR conditions with LIKE)
            country: Filter by country (optional)
            birth_year: Filter by birth year (optional)
            source_keys: Filter by source keys (optional)
            status: Filter by status ('current' or 'former', optional)
            limit: Maximum results to return (default 1000)

        Returns:
            List of matching entries (reduced set for fuzzy matching)
        """
        from app.core.db.database import get_osspep_db_connection
        max_retries = 3
        for attempt in range(max_retries):
            conn = None
            try:
                conn = get_osspep_db_connection()

                # Build dynamic WHERE clauses
                conditions = []
                params = []

                # Add name token filters (OR conditions - match ANY token)
                if name_tokens:
                    token_conditions = []
                    for token in name_tokens:
                        token_conditions.append("normalized_name LIKE %s")
                        params.append(f"%{token}%")
                    # Wrap OR conditions in parentheses
                    conditions.append(f"({' OR '.join(token_conditions)})")

                # Add country filter (country column is single value)
                if country:
                    conditions.append("country = %s")
                    params.append(country)

                # Add birth_year filter
                if birth_year is not None:
                    conditions.append("birth_year = %s")
                    params.append(birth_year)

                # Add source_keys filter
                if source_keys:
                    placeholders = ', '.join(['%s'] * len(source_keys))
                    conditions.append(f"source_key IN ({placeholders})")
                    params.extend(source_keys)

                # Add status filter
                if status:
                    conditions.append("status = %s")
                    params.append(status)

                # Build the base query
                query = """
                    SELECT id, source_key, name, normalized_name, position, country,
                           birth_date, birth_year, start_date, end_date,
                           position_type, level, status, family_members,
                           wikipedia_url, wikipedia_title
                    FROM pep_entries
                """

                # Add WHERE clause if we have conditions
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)

                # Add LIMIT to prevent excessive results
                query += f" LIMIT {int(limit)}"

                with conn.cursor(dictionary=True) as cursor:
                    cursor.execute(query, tuple(params))
                    results = cursor.fetchall()

                    logger.info(
                        f"search_by_filters returned {len(results)} candidates "
                        f"(name_tokens={name_tokens}, country={country}, "
                        f"birth_year={birth_year}, source_keys={source_keys}, "
                        f"status={status}, limit={limit})"
                    )
                    return results

            except MySQLError as e:
                logger.error(f"Error in search_by_filters (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return []
                import time
                time.sleep(0.1)
            finally:
                if conn:
                    conn.close()
        return []

    # ==================== Utility Operations ====================

    def get_entry_count(self, source_key: str = None) -> int:
        """
        Get count of entries.

        Args:
            source_key: Optional source key to filter by

        Returns:
            Number of entries
        """
        from app.core.db.database import get_osspep_db_connection
        max_retries = 3
        for attempt in range(max_retries):
            conn = None
            try:
                conn = get_osspep_db_connection()

                if source_key:
                    query = "SELECT COUNT(*) as count FROM pep_entries WHERE source_key = %s"
                    params = (source_key,)
                else:
                    query = "SELECT COUNT(*) as count FROM pep_entries"
                    params = ()

                with conn.cursor(dictionary=True) as cursor:
                    cursor.execute(query, params)
                    result = cursor.fetchone()
                    return result['count'] if result else 0

            except MySQLError as e:
                logger.error(f"Error getting PEP entry count (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return 0
                import time
                time.sleep(0.1)
            finally:
                if conn:
                    conn.close()
        return 0


# Global instance (initialized after database connection is established)
pep_repository = None

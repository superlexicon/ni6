"""
Repository for Crime database operations.

This repository connects to the OSSPEP database where crime watchlist
data is stored and synced by the separate OSSPEP Quarkus service.

This application only reads from the OSSPEP database - no writing/syncing.

Note: The crime_entries table includes sanctioned individuals (no separate sanctions table).
"""

from typing import Optional, Dict, Any, List
from mysql.connector.errors import Error as MySQLError
from app.core import logger
from .base_repository import BaseRepository


class CrimeRepository(BaseRepository):
    """
    Repository for crime entries database operations.

    Table:
    - crime_entries: Individual crime entries with parsed, searchable fields
      Includes sanctioned individuals (no separate sanctions table)

    Note: This connects to the OSSPEP database, not the main im_osint database.
    The OSSPEP database only has 'crime_entries' table - no 'crime_lists' table.
    """

    def __init__(self):
        super().__init__('crime_entries')

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
                        SELECT id, source_key, name, normalized_name, crime_details, country,
                               birth_date, birth_year, arrest_date, conviction_date,
                               sentence, status, summary, details, charges, court,
                               case_number, organization, position, opensanctions_url
                        FROM crime_entries
                        WHERE source_key = %s
                    """
                    params = (source_key,)
                else:
                    query = """
                        SELECT id, source_key, name, normalized_name, crime_details, country,
                               birth_date, birth_year, arrest_date, conviction_date,
                               sentence, status, summary, details, charges, court,
                               case_number, organization, position, opensanctions_url
                        FROM crime_entries
                    """
                    params = ()

                with conn.cursor(dictionary=True) as cursor:
                    cursor.execute(query, params)
                    results = cursor.fetchall()
                    return results

            except MySQLError as e:
                logger.error(f"Error getting all crime entries (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return []
                import time
                time.sleep(0.1)
            finally:
                if conn:
                    conn.close()
        return []

    def search_by_name(
        self,
        normalized_name: str,
        source_keys: List[str] = None,
        crime_details: str = None
    ) -> List[Dict[str, Any]]:
        """
        Search for entries by normalized name.

        Args:
            normalized_name: Normalized name to search for
            source_keys: Optional list of source keys to filter by
            crime_details: Optional crime type to filter by (e.g., 'sanctions_evasion')

        Returns:
            List of matching entries
        """
        from app.core.db.database import get_osspep_db_connection
        max_retries = 3
        for attempt in range(max_retries):
            conn = None
            try:
                conn = get_osspep_db_connection()

                # Build query with optional filters
                conditions = ["normalized_name = %s"]
                params = [normalized_name]

                if source_keys:
                    placeholders = ', '.join(['%s'] * len(source_keys))
                    conditions.append(f"source_key IN ({placeholders})")
                    params.extend(source_keys)

                if crime_details:
                    conditions.append("crime_details = %s")
                    params.append(crime_details)

                query = f"""
                    SELECT id, source_key, name, normalized_name, crime_details, country,
                           birth_date, birth_year, arrest_date, conviction_date,
                           sentence, status, summary, details, charges, court,
                           case_number, organization, position, opensanctions_url
                    FROM crime_entries
                    WHERE {' AND '.join(conditions)}
                """

                with conn.cursor(dictionary=True) as cursor:
                    cursor.execute(query, tuple(params))
                    results = cursor.fetchall()
                    return results

            except MySQLError as e:
                logger.error(f"Error searching crime entries by name (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return []
                import time
                time.sleep(0.1)
            finally:
                if conn:
                    conn.close()
        return []

    def search_sanctions_by_name(
        self,
        normalized_name: str,
        source_keys: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for sanctioned individuals by normalized name.

        Filters crime_entries for sanctions-related crime types.

        Args:
            normalized_name: Normalized name to search for
            source_keys: Optional list of source keys to filter by

        Returns:
            List of matching sanctioned individuals
        """
        from app.core.db.database import get_osspep_db_connection
        max_retries = 3
        for attempt in range(max_retries):
            conn = None
            try:
                conn = get_osspep_db_connection()

                # Sanctions-related crime types
                sanctions_types = ['sanctions_evasion', 'financial_crime', 'corruption']

                # Build query with optional filters
                # Build dynamic placeholders for crime_details IN clause
                crime_details_placeholders = ', '.join(['%s'] * len(sanctions_types))
                conditions = ["normalized_name = %s", f"crime_details IN ({crime_details_placeholders})"]
                params = [normalized_name] + sanctions_types

                if source_keys:
                    source_key_placeholders = ', '.join(['%s'] * len(source_keys))
                    conditions.append(f"source_key IN ({source_key_placeholders})")
                    params.extend(source_keys)

                query = f"""
                    SELECT id, source_key, name, normalized_name, crime_details, country,
                           birth_date, birth_year, arrest_date, conviction_date,
                           sentence, status, summary, details, charges, court,
                           case_number, organization, position, opensanctions_url
                    FROM crime_entries
                    WHERE {' AND '.join(conditions)}
                """

                with conn.cursor(dictionary=True) as cursor:
                    cursor.execute(query, tuple(params))
                    results = cursor.fetchall()
                    return results

            except MySQLError as e:
                logger.error(f"Error searching sanctions by name (attempt {attempt + 1}/{max_retries}): {e}")
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
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Search crime entries using database-side filtering for efficiency.

        This method uses SQL WHERE clauses to filter candidates before returning
        them, avoiding the need to load all rows into memory. This is essential
        for large datasets (100K+ rows).

        Args:
            name_tokens: List of name tokens to search for (OR conditions with LIKE)
            country: Filter by country (optional)
            birth_year: Filter by birth year (optional)
            source_keys: Filter by source keys (optional)
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

                # Add country filter (countries column is comma-separated, use FIND_IN_SET)
                if country:
                    conditions.append("FIND_IN_SET(%s, countries) > 0")
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

                # Build the base query
                query = """
                    SELECT id, source_key, name, normalized_name, crime_details, countries,
                           birth_date, birth_year, entry_type, sanctions_details
                    FROM crime_entries
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
                        f"birth_year={birth_year}, source_keys={source_keys}, limit={limit})"
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
                    query = "SELECT COUNT(*) as count FROM crime_entries WHERE source_key = %s"
                    params = (source_key,)
                else:
                    query = "SELECT COUNT(*) as count FROM crime_entries"
                    params = ()

                with conn.cursor(dictionary=True) as cursor:
                    cursor.execute(query, params)
                    result = cursor.fetchone()
                    return result['count'] if result else 0

            except MySQLError as e:
                logger.error(f"Error getting crime entry count (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return 0
                import time
                time.sleep(0.1)
            finally:
                if conn:
                    conn.close()
        return 0

    def is_source_empty(self, source_key: str) -> bool:
        """
        Check if a source has no entries.

        Args:
            source_key: Source key to check

        Returns:
            True if source has no entries, False otherwise
        """
        return self.get_entry_count(source_key) == 0


# Global instance (initialized after database connection is established)
crime_repository = None

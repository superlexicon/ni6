"""
Database-backed bank lookup service.

Replaces JSON-based bank_lookup.py with database queries.
Maintains same API for compatibility with existing code.
Uses raw SQL with existing mysql-connector-python driver.
"""

import re
import json
from dataclasses import dataclass
from typing import Optional, List, Any, Dict

from app.core.db.database import get_db_connection
from app.core.logger import get_logger

logger = get_logger()

# Country name to ISO code mapping for normalization
_COUNTRY_NAME_TO_ISO = {
    "Singapore": "SG",
    "SINGAPORE": "SG",
    "India": "IN",
    "INDIA": "IN",
    "United States": "US",
    "USA": "US",
    "U.S.A.": "US",
    "United Kingdom": "GB",
    "UK": "GB",
    "GREAT BRITAIN": "GB",
    "UAE": "AE",
    "United Arab Emirates": "AE",
    "Malaysia": "MY",
    "Thailand": "TH",
    "Australia": "AU",
    "Hong Kong": "HK",
    "China": "CN",
    "Japan": "JP",
    "South Korea": "KR",
    "Korea": "KR",
    "Taiwan": "TW",
    "Vietnam": "VN",
    "Macau": "MO",
    "Macao": "MO",
}


def _normalize_country_code(country_hint: str) -> str:
    """Normalize country name to ISO code, or return as-is if already 2-letter code."""
    if not country_hint:
        return None

    country_upper = country_hint.upper().strip()

    # If already 2-letter ISO code, return as-is
    if len(country_upper) == 2:
        return country_upper

    # Look up in mapping
    return _COUNTRY_NAME_TO_ISO.get(country_upper, country_upper)


@dataclass
class BankInfo:
    """Bank information with SWIFT code."""
    abbreviation: str
    full_name: str
    country: str
    swift_code: str  # Single SWIFT code per row
    bank_id: int = 0  # Database ID for compatibility with related tables


class BankDatabaseLookup:
    """
    Database-backed bank lookup.

    Provides the same API as BankLookup but queries the database instead of
    reading from JSON files.

    Uses optimized single-query lookups for better performance:
    - lookup_by_name: Single UNION query instead of 3-4 sequential queries
    - detect_bank_in_text: Database-side word filtering instead of fetching all identifiers
    - lookup_by_domain: Database-side LIKE filtering
    """

    def __init__(self):
        # No caching - using optimized queries instead
        pass

    def lookup_by_name(
        self,
        bank_name: str,
        country: str = None,
    ) -> Optional[BankInfo]:
        """
        Look up bank by full name or abbreviation.

        Uses the new simplified banks table with FULLTEXT search.

        Args:
            bank_name: Bank name to look up (e.g., "DBS Bank", "HDFC", "Emirates NBD")
            country: Optional ISO country code for disambiguation

        Returns:
            BankInfo if found, None otherwise
        """
        if not bank_name:
            return None

        bank_name_upper = bank_name.upper().strip()
        bank_name_lower = bank_name.lower().strip()

        conn = get_db_connection()
        try:
            cursor = conn.cursor(dictionary=True)

            # Try exact match in abbreviations first (fastest)
            base_query = """
                SELECT id, swift_code, country_code, legal_name, abbreviations, common_names
                FROM banks
                WHERE is_active = 1
                  AND abbreviations LIKE %s
            """
            params = [f"%{bank_name_upper}%"]

            # Add country filter if provided
            country_upper = _normalize_country_code(country)
            if country_upper:
                base_query += " AND country_code = %s"
                params.append(country_upper)

            base_query += " LIMIT 1"

            cursor.execute(base_query, tuple(params))
            result = cursor.fetchone()

            if result:
                return self._build_bank_info_from_row(result)

            # Try FULLTEXT search for partial name match
            # Split search term into meaningful words (4+ chars) for filtering
            search_words = set(w for w in bank_name_lower.split() if len(w) >= 4)

            # Generic words that appear in many bank names - reject if ONLY these are present
            generic_words = {'bank', 'ltd', 'limited', 'corp', 'corporation', 'plc', 'co', 'company'}
            is_generic_only = search_words and search_words.issubset(generic_words)

            try:
                fulltext_query = """
                    SELECT id, swift_code, country_code, legal_name, abbreviations, common_names
                    FROM banks
                    WHERE is_active = 1
                      AND MATCH(abbreviations, common_names) AGAINST(%s IN NATURAL LANGUAGE MODE)
                """
                ft_params = [bank_name_lower]

                if country_upper:
                    fulltext_query += " AND country_code = %s"
                    ft_params.append(country_upper)

                # Fetch all matches (not just 1) for filtering
                cursor.execute(fulltext_query, tuple(ft_params))
                results = cursor.fetchall()

                if results:
                    # If search has only generic words, skip FULLTEXT results (too many matches)
                    if is_generic_only:
                        logger.debug(f"Search contains only generic words {search_words}, skipping FULLTEXT results")
                    # If search has 2+ meaningful words, filter to matches with multiple word overlap
                    elif len(search_words) >= 2:
                        for result in results:
                            # Check how many search words appear in this bank's names
                            bank_text = f"{result['abbreviations']} {result['common_names']}".lower()
                            word_matches = sum(1 for w in search_words if w in bank_text)
                            # Require at least 2 word matches OR 50% of words
                            if word_matches >= max(2, len(search_words) // 2):
                                return self._build_bank_info_from_row(result)
                        # No good match found
                        logger.debug(
                            f"FULLTEXT had {len(results)} results but none met word match criteria. "
                            f"Search words: {search_words}"
                        )
                    else:
                        # Single specific word search - return first result (existing behavior)
                        return self._build_bank_info_from_row(results[0])
            except Exception as e:
                # FULLTEXT might fail if index doesn't exist or word too short
                logger.debug(f"FULLTEXT search failed: {e}, falling back to LIKE search")

            # Fallback to LIKE search for partial matching
            # Skip if search has only generic words (already handled above)
            if not is_generic_only:
                fallback_query = """
                    SELECT id, swift_code, country_code, legal_name, abbreviations, common_names
                    FROM banks
                    WHERE is_active = 1
                      AND common_names LIKE %s
                """
                fallback_params = [f"%{bank_name_lower}%"]

                if country_upper:
                    fallback_query += " AND country_code = %s"
                    fallback_params.append(country_upper)

                fallback_query += " LIMIT 1"

                cursor.execute(fallback_query, tuple(fallback_params))
                result = cursor.fetchone()

                if result:
                    return self._build_bank_info_from_row(result)

        finally:
            conn.close()

        logger.debug(f"Bank not found: {bank_name}")
        return None

    def lookup_by_domain(self, text: str) -> Optional[BankInfo]:
        """
        Look up bank by domain patterns found in text.

        With the new schema, domains are stored in common_names TEXT field.
        Uses database-side LIKE filtering.

        Args:
            text: Text that may contain URLs or email domains

        Returns:
            BankInfo if found, None otherwise
        """
        if not text:
            return None

        text_lower = text.lower()

        conn = get_db_connection()
        try:
            cursor = conn.cursor(dictionary=True)

            # Try to extract domain from text
            # Look for patterns like "dbs.com", "dbs.com.sg", etc.
            import re
            domain_pattern = r'\b([a-z0-9-]+(\.[a-z]{2,})+)\b'
            domains = re.findall(domain_pattern, text_lower)

            if not domains:
                return None

            # Try each domain found (longest first for more specific matches)
            for domain_match in sorted(domains, key=lambda x: len(x[0]), reverse=True):
                domain = domain_match[0]

                # Search in common_names for domain match
                query = """
                    SELECT id, swift_code, country_code, legal_name, abbreviations, common_names
                    FROM banks
                    WHERE is_active = 1
                      AND common_names LIKE %s
                    LIMIT 1
                """

                cursor.execute(query, (f"%{domain}%",))
                result = cursor.fetchone()

                if result:
                    return self._build_bank_info_from_row(result)

        finally:
            conn.close()

        return None

    def detect_bank_in_text(
        self,
        text: str,
        country_hint: str = None,
    ) -> Optional[BankInfo]:
        """
        Detect bank in text using the new simplified schema.

        Searches for bank names and abbreviations in the text using
        the abbreviations and common_names fields.

        Args:
            text: Text to search for bank references
            country_hint: Optional country code for disambiguation

        Returns:
            BankInfo if found, None otherwise
        """
        if not text:
            return None

        text_lower = text.lower()

        conn = get_db_connection()
        try:
            cursor = conn.cursor(dictionary=True)

            # Extract meaningful words (4+ characters) for search
            words = set(w for w in text_lower.split() if len(w) >= 4)

            if words:
                # Build LIKE query for each word
                # Search in both abbreviations and common_names
                placeholders = ' OR '.join(['(abbreviations LIKE %s OR common_names LIKE %s)'] * len(words))
                params = []
                for word in words:
                    params.extend([f"%{word}%", f"%{word}%"])

                base_query = f"""
                    SELECT id, swift_code, country_code, legal_name, abbreviations, common_names
                    FROM banks
                    WHERE is_active = 1
                      AND ({placeholders})
                """
                query_params = params.copy()

                # Add country filter if provided
                country_upper = _normalize_country_code(country_hint)
                if country_upper:
                    base_query += " AND country_code = %s"
                    query_params.append(country_upper)

                base_query += " LIMIT 1"

                cursor.execute(base_query, tuple(query_params))
                result = cursor.fetchone()

                if result:
                    return self._build_bank_info_from_row(result)

        finally:
            conn.close()

        # Fallback to domain search
        return self.lookup_by_domain(text)

    def extract_country_from_tld(self, text: str) -> Optional[str]:
        """
        Extract country code from website TLD (Top-Level Domain).

        For example: "hsbc.com.sg" -> "SG", "dbs.com.in" -> "IN"

        Args:
            text: Text that may contain website URLs or email addresses

        Returns:
            ISO country code if found, None otherwise
        """
        if not text:
            return None

        # Common TLD to country mapping
        tld_country_map = {
            ".sg": "SG",
            ".in": "IN",
            ".ae": "AE",
            ".th": "TH",
            ".my": "MY",
            ".hk": "HK",
            ".au": "AU",
            ".jp": "JP",
            ".uk": "GB",
            ".gb": "GB",
            ".us": "US",
            ".ca": "CA",
            ".de": "DE",
            ".fr": "FR",
            ".it": "IT",
            ".nl": "NL",
            ".es": "ES",
            ".ch": "CH",
        }

        # Find TLDs in text (e.g., .sg, .in, .ae)
        text_lower = text.lower()
        for tld, country in tld_country_map.items():
            if re.search(r'\b' + re.escape(tld) + r'\b', text_lower):
                return country

        return None

    def _build_bank_info_from_row(self, row: dict) -> BankInfo:
        """
        Build BankInfo from a single database row.

        Args:
            row: Database row with bank info

        Returns:
            BankInfo
        """
        # Extract primary abbreviation from abbreviations CSV
        abbrev = row['abbreviations'].split(',')[0].strip() if row['abbreviations'] else row['swift_code'][:4]

        return BankInfo(
            abbreviation=abbrev,
            full_name=row['legal_name'],
            country=row['country_code'],
            swift_code=row['swift_code'],
            bank_id=row['id']
        )

    def _build_bank_info_from_results(
        self,
        results: list,
        country_hint: str = None,
    ) -> Optional[BankInfo]:
        """
        Build BankInfo from query results with country disambiguation.

        With the new schema, each result is a different SWIFT code (country).
        This method handles country disambiguation.

        Args:
            results: List of result rows with bank info
            country_hint: Optional country code for disambiguation

        Returns:
            BankInfo if found, None otherwise
        """
        if not results:
            return None

        # Select country operation based on hint or use first result
        country_upper = _normalize_country_code(country_hint)

        if country_upper:
            # Find matching country
            for row in results:
                if row['country_code'] == country_upper:
                    return self._build_bank_info_from_row(row)
            # No matching country found
            logger.warning(
                f"Bank not found in country {country_upper}. "
                f"Available countries: {[r['country_code'] for r in results]}"
            )
            return None
        elif len(results) == 1:
            # Single country - return it
            return self._build_bank_info_from_row(results[0])

        # Multi-country bank without country hint - return first result
        # With new schema, there's no is_primary_country, so just pick the first
        logger.debug(
            f"Bank has multiple countries {[r['country_code'] for r in results]} "
            f"and no country hint provided. Using first result."
        )
        return self._build_bank_info_from_row(results[0])

    def get_swift_code(
        self,
        bank_name: str,
        country: str = None,
    ) -> Optional[str]:
        """
        Get SWIFT code for a bank.

        Args:
            bank_name: Bank name or abbreviation
            country: Optional ISO country code

        Returns:
            SWIFT code if found, None otherwise
        """
        info = self.lookup_by_name(bank_name, country)
        if info and info.swift_code:
            return info.swift_code
        return None

    def get_country(
        self,
        bank_name: str,
    ) -> Optional[str]:
        """
        Get default country for a bank.

        Args:
            bank_name: Bank name or abbreviation

        Returns:
            ISO country code if found, None otherwise
        """
        info = self.lookup_by_name(bank_name, None)
        if info:
            return info.country
        return None

    def lookup_by_iban(self, text: str) -> Optional[BankInfo]:
        """
        Look up bank by IBAN found in text.

        Extracts IBAN from text and uses the bank code to identify the bank.
        Currently supports UAE IBANs.

        UAE IBAN format: AE + 2 check digits + 3-digit bank code + 16-digit account

        Args:
            text: Text that may contain an IBAN

        Returns:
            BankInfo if found, None otherwise
        """
        if not text:
            return None

        # UAE IBAN mapping (bank code -> abbreviation)
        UAE_IBAN_BANK_CODES = {
            '001': 'CBUAE', '002': 'FAB', '003': 'CITI', '007': 'ENBD', '008': 'ENBD',
            '010': 'ADCB', '014': 'ADCB', '017': 'MASHREQ', '019': 'DIB', '022': 'CBD',
            '023': 'ARABANK', '024': 'FAB', '025': 'ADCB', '030': 'ENBD', '031': 'NOF',
            '032': 'ADCB', '033': 'CBD', '035': 'RAKBANK', '040': 'HSBC', '041': 'ABCB',
            '042': 'BBME', '050': 'FAB', '060': 'ENBD', '201': 'ISDB', '301': 'ADIB',
            '302': 'DIB', '303': 'AJB', '304': 'ALHILAL', '305': 'UNB', '306': 'NOOR',
            '307': 'ADIB',
        }

        # Find UAE IBAN pattern in text
        iban_pattern = r'AE\d{2}(\d{3})\d{16}'
        match = re.search(iban_pattern, text.replace(' ', '').replace('-', ''))

        if match:
            bank_code = match.group(1)
            logger.debug(f"Found UAE IBAN with bank code: {bank_code}")

            abbrev = UAE_IBAN_BANK_CODES.get(bank_code)
            if abbrev:
                info = self.lookup_by_name(abbrev, 'AE')
                if info:
                    logger.info(f"IBAN lookup found bank: {info.full_name} (code: {bank_code})")
                    return info

        return None

    def lookup_by_ifsc(self, text: str) -> Optional[BankInfo]:
        """
        Look up bank by IFSC code found in text.

        Extracts IFSC from text and uses the bank code (first 4 chars) to identify the bank.
        Supports Indian IFSC codes.

        IFSC format: 4-letter bank code + '0' + 6-digit branch code
        Example: "UTIB0005157" -> "UTIB" -> Axis Bank

        Args:
            text: Text that may contain an IFSC code

        Returns:
            BankInfo if found, None otherwise
        """
        if not text:
            return None

        # India IFSC bank code mapping (first 4 characters -> abbreviation)
        INDIA_IFSC_BANK_CODES = {
            'UTIB': 'AXIS', 'HDFC': 'HDFC', 'SBIN': 'SBIN', 'ICIC': 'ICIC',
            'PUNB': 'PUNB', 'UBIN': 'UBIN', 'BKID': 'BKID', 'MAHB': 'MAHB',
            'CANR': 'CANR', 'CORP': 'CORP', 'ALLA': 'ALLA', 'ANDA': 'ANDA',
            'BARB': 'BARB', 'CABB': 'CABB', 'CBIN': 'CBIN', 'CIUB': 'CIUB',
            'DEUT': 'DEUT', 'DLXB': 'DLXB', 'DSKB': 'DSKB', 'FEDR': 'FEDR',
            'INDB': 'INDB', 'IOBA': 'IOBA', 'JAKA': 'JAKA', 'KKBK': 'KKBK',
            'KVBL': 'KVBL', 'LAVB': 'LAVB', 'NRBL': 'NRBL', 'RATN': 'RATN',
            'SVCB': 'SVCB', 'TMBL': 'TMBL', 'UCBA': 'UCBA', 'VIJB': 'VIJB',
            'YESB': 'YESB',
        }

        # Find IFSC pattern in text
        ifsc_pattern = r'\b([A-Z]{4})0\d{6}\b'
        match = re.search(ifsc_pattern, text.replace(' ', '').replace('-', ''))

        if match:
            bank_code = match.group(1)
            logger.debug(f"Found IFSC with bank code: {bank_code}")

            abbrev = INDIA_IFSC_BANK_CODES.get(bank_code)
            if abbrev:
                info = self.lookup_by_name(abbrev, 'IN')
                if info:
                    logger.info(f"IFSC lookup found bank: {info.full_name} (code: {bank_code})")
                    return info

        return None

    def get_all_bank_identifiers(self) -> List[Dict[str, Any]]:
        """
        Get all bank identifiers for unified map pattern matching.

        Returns all bank identifiers including abbreviations and common names.
        Used by build_unified_map() for Pass 1 validation.

        With the new schema, identifiers are stored in abbreviations and common_names
        TEXT fields. This method parses them and returns a unified list.

        Args:
            None

        Returns:
            List of dicts with keys: 'identifier' (str), 'identifier_type' (str)
            All identifiers are lowercase for case-insensitive matching
        """
        conn = get_db_connection()
        try:
            cursor = conn.cursor(dictionary=True)

            # Get all active banks and parse their identifiers
            query = """
                SELECT abbreviations, common_names
                FROM banks
                WHERE is_active = 1
            """

            cursor.execute(query)
            results = cursor.fetchall()

            # Parse CSV fields and return lowercase identifiers
            identifiers = []
            for row in results:
                # Add abbreviations
                if row['abbreviations']:
                    for abbrev in row['abbreviations'].split(','):
                        abbrev = abbrev.strip()
                        if abbrev:
                            identifiers.append({
                                'identifier': abbrev.lower(),
                                'identifier_type': 'abbreviation'
                            })

                # Add common names
                if row['common_names']:
                    for name in row['common_names'].split(','):
                        name = name.strip()
                        if name and len(name) >= 4:  # Filter out very short names
                            identifiers.append({
                                'identifier': name.lower(),
                                'identifier_type': 'full_name'
                            })

            logger.info(f"Loaded {len(identifiers)} bank identifiers for unified map")
            return identifiers

        finally:
            conn.close()


# Singleton instance
_instance = None


def get_bank_database_lookup() -> BankDatabaseLookup:
    """Get the singleton BankDatabaseLookup instance."""
    global _instance
    if _instance is None:
        _instance = BankDatabaseLookup()
    return _instance


# ==================== Convenience Functions ====================
# These provide the same API as the old bank_lookup.py module

def detect_bank_in_text(text: str, country_hint: str = None) -> Optional[BankInfo]:
    """Detect bank in text using unified mapping."""
    lookup = get_bank_database_lookup()
    return lookup.detect_bank_in_text(text, country_hint)


def lookup_bank_by_name(bank_name: str, country: str = None) -> Optional[BankInfo]:
    """Look up bank by name."""
    lookup = get_bank_database_lookup()
    return lookup.lookup_by_name(bank_name, country)


def get_swift_code_for_bank(bank_name: str, country: str = None) -> Optional[str]:
    """Get SWIFT code for a bank."""
    lookup = get_bank_database_lookup()
    return lookup.get_swift_code(bank_name, country)


def get_country_for_bank(bank_name: str) -> Optional[str]:
    """Get default country for a bank."""
    lookup = get_bank_database_lookup()
    return lookup.get_country(bank_name)


def lookup_bank_by_iban(text: str) -> Optional[BankInfo]:
    """Look up bank by IBAN patterns in text."""
    lookup = get_bank_database_lookup()
    return lookup.lookup_by_iban(text)


def lookup_bank_by_ifsc(text: str) -> Optional[BankInfo]:
    """Look up bank by IFSC patterns in text."""
    lookup = get_bank_database_lookup()
    return lookup.lookup_by_ifsc(text)


def lookup_bank_by_domain(text: str) -> Optional[BankInfo]:
    """Look up bank by domain patterns in text."""
    lookup = get_bank_database_lookup()
    return lookup.lookup_by_domain(text)


def extract_country_from_tld(text: str) -> Optional[str]:
    """Extract country code from website TLD."""
    lookup = get_bank_database_lookup()
    return lookup.extract_country_from_tld(text)

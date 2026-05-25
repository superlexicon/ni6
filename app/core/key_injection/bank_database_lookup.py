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


@dataclass
class BankInfo:
    """Bank information with SWIFT code."""
    abbreviation: str
    full_name: str
    country: str
    swift_codes: list
    primary_swift: str = ""


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

        Uses a single UNION query for optimal performance.

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

            # Single optimized UNION query - checks abbrev and identifiers together
            query = """
                SELECT b.id, b.abbrev, b.full_name, bco.country_code, bco.swift_codes
                FROM banks b
                JOIN bank_country_operations bco ON b.id = bco.bank_id
                WHERE b.abbrev = %s AND b.is_active = 1 AND bco.is_active = 1

                UNION

                SELECT b.id, b.abbrev, b.full_name, bco.country_code, bco.swift_codes
                FROM bank_identifiers bi
                JOIN banks b ON bi.bank_id = b.id
                JOIN bank_country_operations bco ON b.id = bco.bank_id
                WHERE bi.identifier = %s AND b.is_active = 1 AND bi.is_validated = 1 AND bco.is_active = 1
            """

            cursor.execute(query, (bank_name_upper, bank_name_lower))
            results = cursor.fetchall()

            if results:
                return self._build_bank_info_from_results(results, country)

            # Try partial matching for multi-word bank names (secondary path)
            words = bank_name_lower.split()
            if len(words) > 1:
                for word in words:
                    if len(word) >= 4:
                        cursor.execute(
                            """SELECT b.id, b.abbrev, b.full_name, bco.country_code, bco.swift_codes
                               FROM bank_identifiers bi
                               JOIN banks b ON bi.bank_id = b.id
                               JOIN bank_country_operations bco ON b.id = bco.bank_id
                               WHERE bi.identifier = %s AND bi.is_validated = 1
                                 AND b.is_active = 1 AND bco.is_active = 1""",
                            (word,)
                        )
                        results = cursor.fetchall()
                        if results:
                            return self._build_bank_info_from_results(results, country)

        finally:
            conn.close()

        logger.debug(f"Bank not found: {bank_name}")
        return None

    def lookup_by_domain(self, text: str) -> Optional[BankInfo]:
        """
        Look up bank by domain patterns found in text.

        Uses database-side LIKE filtering for optimal performance.

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

            # Database-side LIKE filtering - single query
            query = """
                SELECT b.id, b.abbrev, b.full_name, bco.country_code, bco.swift_codes
                FROM bank_identifiers bi
                JOIN banks b ON bi.bank_id = b.id
                JOIN bank_country_operations bco ON b.id = bco.bank_id
                WHERE bi.identifier_type IN ('domain', 'email_domain')
                  AND %s LIKE CONCAT('%%', bi.identifier, '%%')
                  AND b.is_active = 1 AND bco.is_active = 1
                ORDER BY LENGTH(bi.identifier) DESC
                LIMIT 1
            """

            cursor.execute(query, (text_lower,))
            results = cursor.fetchall()

            if results:
                return self._build_bank_info_from_results(results, None)

        finally:
            conn.close()

        return None

    def detect_bank_in_text(
        self,
        text: str,
        country_hint: str = None,
    ) -> Optional[BankInfo]:
        """
        Detect bank in text using database-side word filtering.

        Instead of fetching ALL 63K identifiers and filtering in Python,
        this method extracts words from the text and only fetches matching
        identifiers from the database.

        Args:
            text: Text to search for bank references
            country_hint: Optional country code for disambiguation

        Returns:
            BankInfo if found, None otherwise
        """
        if not text:
            return None

        text_lower = text.lower()
        # Extract meaningful words (4+ characters)
        words = set(w for w in text_lower.split() if len(w) >= 4)

        if not words:
            # Try domain search as fallback
            return self.lookup_by_domain(text)

        conn = get_db_connection()
        try:
            cursor = conn.cursor(dictionary=True)

            # Only fetch identifiers that match words in the text - database-side filtering
            placeholders = ', '.join(['%s'] * len(words))
            query = f"""
                SELECT b.id, b.abbrev, b.full_name, bco.country_code, bco.swift_codes
                FROM bank_identifiers bi
                JOIN banks b ON bi.bank_id = b.id
                JOIN bank_country_operations bco ON b.id = bco.bank_id
                WHERE bi.identifier IN ({placeholders})
                  AND bi.identifier_type NOT IN ('domain', 'email_domain')
                  AND b.is_active = 1 AND bi.is_validated = 1 AND bco.is_active = 1
                ORDER BY LENGTH(bi.identifier) DESC
                LIMIT 1
            """

            cursor.execute(query, list(words))
            results = cursor.fetchall()

            if results:
                return self._build_bank_info_from_results(results, country_hint)

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

    def _build_bank_info_from_results(
        self,
        results: list,
        country_hint: str = None,
    ) -> Optional[BankInfo]:
        """
        Build BankInfo from query results with country disambiguation.

        Args:
            results: List of result rows with bank info and country operations
            country_hint: Optional country code for disambiguation

        Returns:
            BankInfo if found, None otherwise
        """
        if not results:
            return None

        # Select country operation based on hint or use first result
        country_upper = country_hint.upper() if country_hint and country_hint.strip() else None

        if country_upper:
            # Find matching country
            for row in results:
                if row['country_code'] == country_upper:
                    swift_codes = json.loads(row['swift_codes']) if isinstance(row['swift_codes'], str) else row['swift_codes']
                    return BankInfo(
                        abbreviation=row['abbrev'],
                        full_name=row['full_name'],
                        country=row['country_code'],
                        swift_codes=swift_codes,
                        primary_swift=swift_codes[0] if swift_codes else ""
                    )
            # No matching country found
            logger.warning(
                f"Bank '{results[0]['abbrev']}' not found in country {country_upper}. "
                f"Available countries: {[r['country_code'] for r in results]}"
            )
            return None
        elif len(results) == 1:
            # Single country - return it
            row = results[0]
            swift_codes = json.loads(row['swift_codes']) if isinstance(row['swift_codes'], str) else row['swift_codes']
            return BankInfo(
                abbreviation=row['abbrev'],
                full_name=row['full_name'],
                country=row['country_code'],
                swift_codes=swift_codes,
                primary_swift=swift_codes[0] if swift_codes else ""
            )

        # Multi-country bank without country hint
        logger.warning(
            f"Bank '{results[0]['abbrev']}' has multiple countries "
            f"{[r['country_code'] for r in results]} and no country hint provided"
        )
        return None

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
        if info and info.swift_codes:
            return info.swift_codes[0]
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

    def get_bank_by_name_and_country(
        self,
        bank_name: str,
        country_code: str
    ) -> Optional[Dict[str, Any]]:
        """
        Look up bank by name and country, return bank info and prompts.

        This is used for bank-specific GLiNER2 prompt retrieval.
        Returns both bank information and any custom prompts available.

        Args:
            bank_name: Bank name (full name or abbreviation)
            country_code: ISO country code

        Returns:
            Dictionary with bank info and prompts, or None if not found:
            {
                "bank_id": int,
                "bank_abbrev": str,
                "bank_name": str,
                "country_code": str,
                "prompts": {...}  # Bank-specific prompts if available, else None
            }
        """
        conn = get_db_connection()
        try:
            cursor = conn.cursor(dictionary=True)

            # First find the bank
            bank_query = """
                SELECT id, abbrev, full_name
                FROM banks
                WHERE (full_name = %s OR abbrev = %s)
                  AND is_active = 1
                LIMIT 1
            """
            cursor.execute(bank_query, (bank_name, bank_name.upper()))
            bank_result = cursor.fetchone()

            if not bank_result:
                return None

            bank_id = bank_result['id']

            # Check for prompts
            prompt_query = """
                SELECT entity_type, prompt_description, entity_category,
                       threshold, examples, validation_pattern
                FROM bank_gliner_prompts
                WHERE bank_id = %s AND country_code = %s AND is_active = 1
            """
            cursor.execute(prompt_query, (bank_id, country_code.upper()))
            prompt_results = cursor.fetchall()

            prompts = None
            if prompt_results:
                # Convert to GLiNER2 schema format
                prompts = {}
                for p in prompt_results:
                    prompts[p['entity_type']] = {
                        'description': p['prompt_description'],
                        'entity': p['entity_category'],
                        'threshold': float(p['threshold']),
                        'examples': json.loads(p['examples']) if p['examples'] else [],
                        'pattern': p['validation_pattern']
                    }
                logger.info(
                    f"Found {len(prompts)} custom prompts for "
                    f"{bank_result['abbrev']}/{country_code}"
                )

            return {
                'bank_id': bank_id,
                'bank_abbrev': bank_result['abbrev'],
                'bank_name': bank_result['full_name'],
                'country_code': country_code,
                'prompts': prompts
            }

        finally:
            conn.close()

    def get_all_bank_identifiers(self) -> List[Dict[str, Any]]:
        """
        Get all bank identifiers for unified map pattern matching.

        Returns all validated bank identifiers including abbreviations, full names,
        and alternate names. Used by build_unified_map() for Pass 1 validation.

        Args:
            None

        Returns:
            List of dicts with keys: 'identifier' (str), 'identifier_type' (str)
            All identifiers are lowercase for case-insensitive matching
        """
        conn = get_db_connection()
        try:
            cursor = conn.cursor(dictionary=True)

            # Get all validated bank identifiers
            query = """
                SELECT bi.identifier, bi.identifier_type
                FROM bank_identifiers bi
                JOIN banks b ON bi.bank_id = b.id
                WHERE b.is_active = 1
                  AND bi.is_validated = 1
                  AND bi.identifier_type IN ('abbreviation', 'full_name', 'alternate_name')
            """

            cursor.execute(query)
            results = cursor.fetchall()

            # Return lowercase identifiers for case-insensitive matching
            identifiers = []
            for row in results:
                identifiers.append({
                    'identifier': row['identifier'].lower(),
                    'identifier_type': row['identifier_type']
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

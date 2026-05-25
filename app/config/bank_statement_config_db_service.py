"""
Bank Statement Configuration Database Service

Database-backed configuration service for bank statement processing.
Replaces JSON-based config loading with database queries.

Provides methods to retrieve:
- Currency information and account number validation rules
- Field labels for extraction (account_number, iban, currency, etc.)
- Address format rules by country
- State-to-country mappings for inference
- Currency name to code mappings

Usage:
    service = get_bank_statement_config_db_service()
    currency_info = service.get_currency_info("SGD")
    labels = service.get_field_labels("account_number")
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from app.core.db.database import get_db_connection
from app.core.logger import get_logger

logger = get_logger()

# Global singleton instance
_instance: Optional["BankStatementConfigDBService"] = None


class BankStatementConfigDBService:
    """
    Database-backed bank statement configuration service.

    Loads configuration data from database tables instead of JSON files.
    Provides the same interface as the old JSON-based config for compatibility.
    """

    def __init__(self):
        """Initialize the database configuration service."""
        self._currency_cache: Dict[str, Dict[str, Any]] = {}
        self._field_labels_cache: Dict[str, List[str]] = {}
        self._currency_name_map_cache: Optional[Dict[str, str]] = None
        self._address_formats_cache: Dict[str, Dict[str, Any]] = {}
        self._state_country_cache: Optional[Dict[str, str]] = None
        self._address_exceptions_cache: Optional[List[str]] = None
        self._cache_loaded = False

    # ============================================================
    # CURRENCY METHODS
    # ============================================================

    def get_currency_info(self, currency_code: str) -> Optional[Dict[str, Any]]:
        """
        Get currency information including account number length specs.

        Args:
            currency_code: ISO 4217 currency code (e.g., "SGD", "INR")

        Returns:
            Currency info dict with keys: currency_name, country_code,
            account_number_min, account_number_max, or None if not found
        """
        if not currency_code:
            return None

        currency_code_upper = currency_code.upper()

        # Check cache first
        if currency_code_upper in self._currency_cache:
            return self._currency_cache[currency_code_upper]

        conn = get_db_connection()
        try:
            cursor = conn.cursor(dictionary=True)

            query = """
                SELECT currency_code, currency_name, country_code,
                       account_number_min, account_number_max
                FROM currencies
                WHERE currency_code = %s AND is_active = 1
            """

            cursor.execute(query, (currency_code_upper,))
            row = cursor.fetchone()

            if row:
                result = {
                    'currency_code': row['currency_code'],
                    'name': row['currency_name'],
                    'country': row['country_code'],
                    'account_number_length': {
                        'min': row['account_number_min'],
                        'max': row['account_number_max']
                    }
                }
                self._currency_cache[currency_code_upper] = result
                return result

        finally:
            conn.close()

        return None

    def get_account_number_length_range(self, currency: str) -> Tuple[int, int]:
        """
        Get min/max account number length for a currency.

        Args:
            currency: Currency code

        Returns:
            Tuple of (min_length, max_length), defaults to (8, 20)
        """
        currency_info = self.get_currency_info(currency)
        if currency_info:
            length_spec = currency_info.get("account_number_length", {})
            return (length_spec.get("min", 8), length_spec.get("max", 20))
        return (8, 20)

    def get_currency_for_country(self, country: str) -> Optional[str]:
        """
        Get the default currency code for a given country.

        Args:
            country: ISO 2-letter country code (e.g., "IN", "SG", "AE")

        Returns:
            Currency code (e.g., "INR", "SGD") or None if not found
        """
        if not country:
            return None

        conn = get_db_connection()
        try:
            cursor = conn.cursor(dictionary=True)

            query = """
                SELECT currency_code
                FROM currencies
                WHERE country_code = %s AND is_active = 1
                LIMIT 1
            """

            cursor.execute(query, (country.upper(),))
            row = cursor.fetchone()

            if row:
                return row['currency_code']

        finally:
            conn.close()

        return None

    def get_currency_name_map(self) -> Dict[str, str]:
        """
        Get mapping of currency names to ISO codes.

        Returns:
            Dict mapping names like "UAE DIRHAM" to codes like "AED"
        """
        if self._currency_name_map_cache is not None:
            return self._currency_name_map_cache

        result = {}

        conn = get_db_connection()
        try:
            cursor = conn.cursor(dictionary=True)

            query = """
                SELECT display_name, currency_code
                FROM currency_name_map
                ORDER BY is_common DESC, display_name
            """

            cursor.execute(query)
            for row in cursor.fetchall():
                result[row['display_name']] = row['currency_code']

        finally:
            conn.close()

        self._currency_name_map_cache = result
        return result

    # ============================================================
    # FIELD LABELS METHODS
    # ============================================================

    def get_field_labels(self, label_type: str) -> List[str]:
        """
        Get field labels by type.

        Args:
            label_type: Type of label ('account_number', 'account_holder_name',
                      'currency', 'iban', 'statement_date', 'opening_balance',
                      'closing_balance', 'address')

        Returns:
            List of label strings
        """
        # Check cache first
        if label_type in self._field_labels_cache:
            return self._field_labels_cache[label_type]

        result = []

        conn = get_db_connection()
        try:
            cursor = conn.cursor(dictionary=True)

            query = """
                SELECT label_text
                FROM field_labels
                WHERE label_type = %s AND is_active = 1
                ORDER BY priority DESC, label_text
            """

            cursor.execute(query, (label_type,))
            for row in cursor.fetchall():
                result.append(row['label_text'])

        finally:
            conn.close()

        self._field_labels_cache[label_type] = result
        return result

    def get_account_number_labels(self) -> List[str]:
        """Return account number labels."""
        return self.get_field_labels('account_number')

    def get_account_holder_name_labels(self) -> List[str]:
        """Return account holder name labels."""
        return self.get_field_labels('account_holder_name')

    def get_currency_labels(self) -> List[str]:
        """Return currency field labels."""
        return self.get_field_labels('currency')

    def get_iban_labels(self) -> List[str]:
        """Return IBAN field labels."""
        return self.get_field_labels('iban')

    def get_statement_date_labels(self) -> List[str]:
        """Return statement date field labels."""
        return self.get_field_labels('statement_date')

    def get_opening_balance_labels(self) -> List[str]:
        """Return opening balance field labels."""
        return self.get_field_labels('opening_balance')

    def get_closing_balance_labels(self) -> List[str]:
        """Return closing balance field labels."""
        return self.get_field_labels('closing_balance')

    def get_address_labels(self) -> List[str]:
        """Return address field labels."""
        return self.get_field_labels('address')

    # ============================================================
    # ADDRESS FORMAT METHODS
    # ============================================================

    def get_address_format(self, country_code: str) -> Optional[Dict[str, Any]]:
        """
        Get address format rules for a country.

        Args:
            country_code: ISO 2-letter country code

        Returns:
            Dict with keys: postal_code_pattern, postal_code_length, is_required
        """
        if not country_code:
            return None

        country_code_upper = country_code.upper()

        # Check cache first
        if country_code_upper in self._address_formats_cache:
            return self._address_formats_cache[country_code_upper]

        conn = get_db_connection()
        try:
            cursor = conn.cursor(dictionary=True)

            query = """
                SELECT postal_code_pattern, postal_code_length, is_required
                FROM address_formats
                WHERE country_code = %s
            """

            cursor.execute(query, (country_code_upper,))
            row = cursor.fetchone()

            if row:
                result = {
                    'postal_code_pattern': row['postal_code_pattern'],
                    'postal_code_length': row['postal_code_length'],
                    'is_required': bool(row['is_required'])
                }
                self._address_formats_cache[country_code_upper] = result
                return result

        finally:
            conn.close()

        return None

    def is_postal_code_required(self, country_code: str) -> bool:
        """
        Check if postal code is required for a country.

        Args:
            country_code: ISO 2-letter country code

        Returns:
            True if postal code is required, False if optional
        """
        address_format = self.get_address_format(country_code)
        if address_format:
            return address_format.get('is_required', True)
        return True  # Default to required

    def get_postal_code_pattern(self, country_code: str) -> Optional[str]:
        """
        Get regex pattern for postal code extraction.

        Args:
            country_code: ISO 2-letter country code

        Returns:
            Regex pattern string or None if not configured
        """
        address_format = self.get_address_format(country_code)
        if address_format:
            return address_format.get('postal_code_pattern')
        return None

    def get_postal_code_length(self, country_code: str) -> Optional[int]:
        """
        Get expected postal code length for a country.

        Args:
            country_code: ISO 2-letter country code

        Returns:
            Expected length or None if not configured
        """
        address_format = self.get_address_format(country_code)
        if address_format:
            return address_format.get('postal_code_length')
        return None

    # ============================================================
    # STATE TO COUNTRY METHODS
    # ============================================================

    def infer_country_from_state(self, state: str) -> Optional[str]:
        """
        Infer country code from state name.

        Args:
            state: State name (e.g., "Maharashtra", "Dubai")

        Returns:
            ISO country code or None
        """
        if not state:
            return None

        state_map = self.get_state_to_country_map()
        state_clean = state.strip().title()

        return state_map.get(state_clean)

    def get_state_to_country_map(self) -> Dict[str, str]:
        """
        Get complete state to country mapping.

        Returns:
            Dict mapping state names to country codes
        """
        if self._state_country_cache is not None:
            return self._state_country_cache

        result = {}

        conn = get_db_connection()
        try:
            cursor = conn.cursor(dictionary=True)

            query = """
                SELECT state_name, country_code
                FROM state_to_country_map
            """

            cursor.execute(query)
            for row in cursor.fetchall():
                result[row['state_name']] = row['country_code']

        finally:
            conn.close()

        self._state_country_cache = result
        return result

    # ============================================================
    # ADDRESS EXTRACTION EXCEPTIONS
    # ============================================================

    def get_address_extraction_exceptions(self) -> Dict[str, List[str]]:
        """
        Get banks where address field should be skipped.

        Returns:
            Dict with 'skip_field_label' list of bank abbreviations
        """
        if self._address_exceptions_cache is not None:
            return {'skip_field_label': self._address_exceptions_cache}

        result = []

        conn = get_db_connection()
        try:
            cursor = conn.cursor(dictionary=True)

            query = """
                SELECT bank_abbrev
                FROM address_extraction_exceptions
            """

            cursor.execute(query)
            for row in cursor.fetchall():
                result.append(row['bank_abbrev'])

        finally:
            conn.close()

        self._address_exceptions_cache = result
        return {'skip_field_label': result}

    def is_address_extraction_exception(self, bank_abbrev: str) -> bool:
        """
        Check if a bank is an address extraction exception.

        Args:
            bank_abbrev: Bank abbreviation

        Returns:
            True if address extraction should be skipped for this bank
        """
        exceptions = self.get_address_extraction_exceptions()
        return bank_abbrev.upper() in [e.upper() for e in exceptions.get('skip_field_label', [])]

    # ============================================================
    # UTILITY METHODS
    # ============================================================

    def clear_cache(self) -> None:
        """Clear all internal caches."""
        self._currency_cache.clear()
        self._field_labels_cache.clear()
        self._currency_name_map_cache = None
        self._address_formats_cache.clear()
        self._state_country_cache = None
        self._address_exceptions_cache = None
        logger.debug("Configuration cache cleared")

    def warm_cache(self) -> None:
        """Load commonly used data into cache."""
        # Preload all currencies
        conn = get_db_connection()
        try:
            cursor = conn.cursor(dictionary=True)

            query = """
                SELECT currency_code, currency_name, country_code,
                       account_number_min, account_number_max
                FROM currencies
                WHERE is_active = 1
            """

            cursor.execute(query)
            for row in cursor.fetchall():
                self._currency_cache[row['currency_code']] = {
                    'currency_code': row['currency_code'],
                    'name': row['currency_name'],
                    'country': row['country_code'],
                    'account_number_length': {
                        'min': row['account_number_min'],
                        'max': row['account_number_max']
                    }
                }

            logger.info(f"Warmed cache with {len(self._currency_cache)} currencies")

        finally:
            conn.close()


def get_bank_statement_config_db_service() -> BankStatementConfigDBService:
    """
    Get the global BankStatementConfigDBService instance.

    Creates the instance on first call (lazy initialization).

    Returns:
        BankStatementConfigDBService singleton instance
    """
    global _instance
    if _instance is None:
        _instance = BankStatementConfigDBService()
    return _instance


def reset_instance() -> None:
    """
    Reset the global instance (useful for testing).
    """
    global _instance
    _instance = None

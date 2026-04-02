"""
Bank Lookup Utility

Provides centralized bank lookup using the comprehensive bank configuration.
Uses reference_templates/bank_statements/config.json which contains:
- banks: Bank abbreviation → Countries → SWIFT codes
- bank_identifiers_map: Bank names, domains, email domains → Abbreviation
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BankInfo:
    """Bank information with SWIFT code."""
    abbreviation: str      # Bank abbreviation (e.g., "DBS", "HDFC")
    full_name: str         # Full bank name (e.g., "DBS Bank", "HDFC Bank")
    country: str           # ISO 3166-1 alpha-2 country code
    swift_codes: List[str] # List of SWIFT codes for this bank/country


class BankLookup:
    """
    Centralized bank lookup using comprehensive configuration.

    Uses a unified mapping containing both bank full names and domain names.
    Lookup strategies:
    1. By full name or alternate name (e.g., "DBS Bank", "Development Bank of Singapore")
    2. By abbreviation (e.g., "DBS", "HDFC")
    3. By domain pattern (e.g., "dbs.com.sg", "hdfcbank.com")

    The unified map ensures that domain matches are prioritized (longer patterns)
    while maintaining a single, consistent lookup strategy.
    """

    _instance = None
    _config = None

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize BankLookup with optional custom config path.

        Args:
            config_path: Path to config.json. If None, uses default location.
        """
        if config_path:
            self._load_config(config_path)
        else:
            self._load_default_config()

        # Build unified reverse mapping
        self._build_unified_mapping()

    def _load_default_config(self):
        """Load config from default location."""
        default_path = Path(__file__).parent.parent.parent / "reference_templates" / "bank_statements" / "config.json"
        self._load_config(str(default_path))

    def _load_config(self, config_path: str):
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config = json.load(f)
            logger.info(f"Loaded bank config from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load bank config from {config_path}: {e}")
            self._config = {"banks": {}, "bank_identifiers_map": {}}

    def _build_unified_mapping(self):
        """Build unified reverse mapping for efficient lookup.

        Single map containing bank names, domains, and email domains.
        Sorted by length to prioritize longer matches (more specific patterns).
        """
        # Unified map containing all bank identifiers
        self._unified_bank_map: Dict[str, str] = {}

        # Add all bank identifiers (names, domains, email domains)
        for identifier, abbrev in self._config.get("bank_identifiers_map", {}).items():
            self._unified_bank_map[identifier.lower()] = abbrev

        # Build abbreviation set for direct matching
        self._abbreviations = set(self._config.get("banks", {}).keys())

        # Create a sorted list of patterns by length (longest first)
        # This ensures more specific patterns (like full domains) match first
        self._sorted_patterns = sorted(
            self._unified_bank_map.keys(),
            key=len,
            reverse=True
        )

        logger.debug(
            f"Built unified mapping: {len(self._unified_bank_map)} patterns, "
            f"{len(self._abbreviations)} banks"
        )

    @classmethod
    def get_instance(cls) -> 'BankLookup':
        """Get singleton instance of BankLookup."""
        if cls._instance is None:
            cls._instance = BankLookup()
        return cls._instance

    def lookup_by_name(self, bank_name: str, country: str = None) -> Optional[BankInfo]:
        """
        Look up bank by full name or abbreviation.

        Args:
            bank_name: Bank name to look up (e.g., "DBS Bank", "HDFC", "Emirates NBD")
            country: Optional ISO country code for disambiguation

        Returns:
            BankInfo if found, None otherwise
        """
        if not bank_name:
            return None

        bank_name_lower = bank_name.lower().strip()
        banks = self._config.get("banks", {})

        # Check if bank_name is an abbreviation
        if bank_name.upper() in self._abbreviations:
            abbrev = bank_name.upper()
            if abbrev in banks:
                return self._get_bank_info_for_abbrev(abbrev, banks, country)

        # Check unified map for any pattern match (longest first)
        for pattern in self._sorted_patterns:
            if pattern in bank_name_lower:
                abbrev = self._unified_bank_map[pattern]
                if abbrev in banks:
                    return self._get_bank_info_for_abbrev(abbrev, banks, country)

        logger.debug(f"Bank not found: {bank_name}")
        return None

    def lookup_by_domain(self, text: str) -> Optional[BankInfo]:
        """
        Look up bank by domain patterns found in text.

        Searches for bank domains (e.g., "dbs.com.sg", "hdfcbank.com") in URLs
        or email addresses within the text.

        Args:
            text: Text that may contain URLs or email domains

        Returns:
            BankInfo if found, None otherwise
        """
        if not text:
            return None

        text_lower = text.lower()
        banks = self._config.get("banks", {})

        # Check sorted patterns (longest first) - domains will match due to being longer
        for pattern in self._sorted_patterns:
            if pattern in text_lower:
                abbrev = self._unified_bank_map[pattern]
                if abbrev in banks:
                    return self._get_bank_info_for_abbrev(abbrev, banks)

        return None

    def _get_full_name_for_abbrev(self, abbrev: str) -> Optional[str]:
        """Get the primary full name for a bank abbreviation."""
        for name, abbr in self._config.get("alternate_names_map", {}).items():
            if abbr == abbrev:
                return name
        return None

    def detect_bank_by_full_name(self, text: str, country_hint: str = None) -> Optional[BankInfo]:
        """
        Detect bank by full name using unified mapping.

        Checks if any pattern from the unified map (bank names or domains)
        appears in the text. Longer patterns are checked first.

        Args:
            text: Text to search for bank references
            country_hint: Optional country code for disambiguation

        Returns:
            BankInfo if found, None otherwise
        """
        if not text:
            return None

        text_lower = text.lower().strip()
        banks = self._config.get("banks", {})

        # Check sorted patterns (longest first)
        for pattern in self._sorted_patterns:
            if pattern in text_lower:
                abbrev = self._unified_bank_map[pattern]
                if abbrev in banks:
                    return self._get_bank_info_for_abbrev(abbrev, banks, country_hint)

        return None

    def detect_bank_in_text(self, text: str, country_hint: str = None) -> Optional[BankInfo]:
        """
        Detect bank in text using unified mapping.

        Checks if any pattern from the unified map (bank names or domains)
        appears in the text. Longer patterns are checked first, so more
        specific matches (like full domains) take priority.

        Args:
            text: Text to search for bank references
            country_hint: Optional country code for disambiguation

        Returns:
            BankInfo if found, None otherwise
        """
        if not text:
            return None

        text_lower = text.lower()
        banks = self._config.get("banks", {})

        # Find first matching pattern (longest first due to _sorted_patterns)
        for pattern in self._sorted_patterns:
            if pattern in text_lower:
                abbrev = self._unified_bank_map[pattern]
                if abbrev in banks:
                    return self._get_bank_info_for_abbrev(abbrev, banks, country_hint)

        return None

    def _get_bank_info_for_abbrev(self, abbrev: str, banks: dict, country_hint: str = None) -> Optional[BankInfo]:
        """Helper method to get BankInfo for an abbreviation."""
        bank_countries = banks[abbrev]
        if not bank_countries:
            return None

        # Determine which country to use
        country_upper = country_hint.upper() if country_hint else None

        if country_upper and country_upper in bank_countries:
            swift_codes = bank_countries[country_upper]
            full_name = self._get_full_name_for_abbrev(abbrev)
            return BankInfo(
                abbreviation=abbrev,
                full_name=full_name or abbrev,
                country=country_upper,
                swift_codes=swift_codes if isinstance(swift_codes, list) else [swift_codes]
            )

        # Fallback: Use first available country (default)
        default_country = list(bank_countries.keys())[0]
        swift_codes = bank_countries[default_country]
        full_name = self._get_full_name_for_abbrev(abbrev)
        return BankInfo(
            abbreviation=abbrev,
            full_name=full_name or abbrev,
            country=default_country,
            swift_codes=swift_codes if isinstance(swift_codes, list) else [swift_codes]
        )

    def get_swift_code(self, bank_name: str, country: str = None) -> Optional[str]:
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

    def get_country(self, bank_name: str) -> Optional[str]:
        """
        Get default country for a bank.

        Args:
            bank_name: Bank name or abbreviation

        Returns:
            ISO country code if found, None otherwise
        """
        info = self.lookup_by_name(bank_name)
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
        # Source: UAE Central Bank
        UAE_IBAN_BANK_CODES = {
            '001': 'CBUAE',   # Central Bank of UAE
            '002': 'FAB',     # National Bank of Abu Dhabi (merged into FAB)
            '003': 'CITI',    # Citibank
            '007': 'ENBD',    # National Bank of Dubai (merged into ENBD)
            '008': 'ENBD',    # Emirates Bank International (merged into ENBD)
            '010': 'ADCB',    # Abu Dhabi Commercial Bank
            '014': 'ADCB',    # Union National Bank (merged into ADCB)
            '017': 'MASHREQ', # Mashreq Bank
            '019': 'DIB',     # Dubai Islamic Bank
            '022': 'CBD',     # Commercial Bank of Dubai
            '023': 'ARABANK', # Arab Bank
            '024': 'FAB',     # First Abu Dhabi Bank (NBAD + FGB)
            '025': 'ADCB',    # Abu Dhabi Commercial Bank
            '030': 'ENBD',    # Emirates NBD
            '031': 'NOF',     # National Bank of Fujairah
            '032': 'ADCB',    # Abu Dhabi Commercial Bank
            '033': 'CBD',     # Commercial Bank of Dubai
            '035': 'RAKBANK', # RAK Bank
            '040': 'HSBC',    # HSBC Bank Middle East
            '041': 'ABCB',    # Al Bathani Arab Bank
            '042': 'BBME',    # Bank of Baroda (now HSBC UAE)
            '050': 'FAB',     # First Gulf Bank (merged into FAB)
            '060': 'ENBD',    # Emirates NBD (alternate code)
            '201': 'ISDB',    # Islamic Development Bank
            '301': 'ADIB',    # Abu Dhabi Islamic Bank
            '302': 'DIB',     # Dubai Islamic Bank
            '303': 'AJB',     # Ajman Bank
            '304': 'ALHILAL', # Al Hilal Bank
            '305': 'UNB',     # Union National Bank
            '306': 'NOOR',    # Noor Bank
            '307': 'ADIB',    # Abu Dhabi Islamic Bank
        }

        # Find UAE IBAN pattern in text
        # UAE IBAN: AE + 2 digits + 3 digit bank code + 16 digits = 23 chars total
        iban_pattern = r'AE\d{2}(\d{3})\d{16}'
        match = re.search(iban_pattern, text.replace(' ', '').replace('-', ''))

        if match:
            bank_code = match.group(1)
            logger.debug(f"Found UAE IBAN with bank code: {bank_code}")

            abbrev = UAE_IBAN_BANK_CODES.get(bank_code)
            if abbrev:
                banks = self._config.get("banks", {})
                if abbrev in banks:
                    bank_countries = banks[abbrev]
                    if 'AE' in bank_countries:
                        swift_codes = bank_countries['AE']
                        full_name = self._get_full_name_for_abbrev(abbrev)
                        logger.info(f"IBAN lookup found bank: {full_name} (code: {bank_code})")
                        return BankInfo(
                            abbreviation=abbrev,
                            full_name=full_name or abbrev,
                            country='AE',
                            swift_codes=swift_codes if isinstance(swift_codes, list) else [swift_codes]
                        )

        return None


# Module-level convenience functions using singleton
def get_bank_lookup() -> BankLookup:
    """Get the singleton BankLookup instance."""
    return BankLookup.get_instance()


def detect_bank_in_text(text: str, country_hint: str = None) -> Optional[BankInfo]:
    """
    Detect bank in text using multiple strategies.

    Convenience function using singleton instance.
    """
    return get_bank_lookup().detect_bank_in_text(text, country_hint)


def lookup_bank_by_name(bank_name: str, country: str = None) -> Optional[BankInfo]:
    """
    Look up bank by name.

    Convenience function using singleton instance.
    """
    return get_bank_lookup().lookup_by_name(bank_name, country)


def lookup_bank_by_domain(text: str) -> Optional[BankInfo]:
    """
    Look up bank by domain patterns in text.

    Convenience function using singleton instance.
    """
    return get_bank_lookup().lookup_by_domain(text)


def get_swift_code_for_bank(bank_name: str, country: str = None) -> Optional[str]:
    """
    Get SWIFT code for a bank.

    Convenience function using singleton instance.
    """
    return get_bank_lookup().get_swift_code(bank_name, country)


def get_country_for_bank(bank_name: str) -> Optional[str]:
    """
    Get default country for a bank.

    Convenience function using singleton instance.
    """
    return get_bank_lookup().get_country(bank_name)


def lookup_bank_by_iban(text: str) -> Optional[BankInfo]:
    """
    Look up bank by IBAN patterns in text.

    Convenience function using singleton instance.
    """
    return get_bank_lookup().lookup_by_iban(text)

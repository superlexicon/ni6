"""
Bank Statement Country Configuration Loader

Loads country-specific configuration for bank statement processing from JSON config.
Consolidates hardcoded country logic into a single configuration-driven system.

This allows adding new countries by only modifying JSON configuration files,
without touching Python code.

Usage:
    loader = get_country_config_loader()
    config = loader.get_country_config("AE")  # Returns UAE-specific config
    subdivisions = loader.get_subdivisions("AE")  # Returns UAE emirates
    is_required = loader.is_postal_code_required("AE")  # Returns False
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Global singleton instance
_instance: Optional["BankStatementCountryConfigLoader"] = None


class BankStatementCountryConfigLoader:
    """
    Load and cache country-specific configuration from JSON config.

    Provides methods to retrieve:
    - Country name aliases (e.g., "uae" -> "AE")
    - Postal code requirements and patterns
    - Subdivisions (states/provinces/emirates)
    - Address extraction patterns
    - City name aliases for common spelling variations
    - Bank regulatory patterns
    - Bank abbreviations
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the country configuration loader.

        Args:
            config_path: Path to the bank_statement_country_config.json file.
                        If None, uses default path in app/config/.
        """
        self._config: Dict[str, Any] = {}
        self._config_loaded = False

        # Determine config path
        if config_path is None:
            config_path = str(
                Path(__file__).parent / "bank_statement_country_config.json"
            )

        self._load_config(config_path)

    def _load_config(self, config_path: str) -> None:
        """
        Load country configuration from the config file.

        Args:
            config_path: Path to bank_statement_country_config.json
        """
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                logger.warning(f"Country config not found at {config_path}, using empty config")
                return

            with open(config_file, 'r', encoding='utf-8') as f:
                self._config = json.load(f)

            self._config_loaded = True
            supported = self._config.get("supported_countries", [])
            logger.info(f"Loaded country config for {len(supported)} countries: {supported}")

        except Exception as e:
            logger.error(f"Failed to load country config: {type(e).__name__}: {e}")
            self._config = {}

    def get_country_config(self, country_code: str) -> Dict[str, Any]:
        """
        Get full configuration for a country.

        Args:
            country_code: ISO 2-letter country code (e.g., "AE", "SG", "IN")

        Returns:
            Dict containing all configuration for the country, or empty dict if not found
        """
        if not self._config_loaded:
            return {}

        country_code_upper = country_code.upper()
        countries = self._config.get("countries", {})
        return countries.get(country_code_upper, {})

    def get_subdivisions(self, country_code: str) -> List[str]:
        """
        Get states/provinces/emirates for a country.

        Args:
            country_code: ISO 2-letter country code

        Returns:
            List of subdivision names, or empty list if not configured
        """
        country_config = self.get_country_config(country_code)
        subdivisions = country_config.get("subdivisions", {})

        # Return the list if configured
        if "list" in subdivisions:
            return subdivisions["list"]

        return []

    def is_postal_code_required(self, country_code: str) -> bool:
        """
        Check if postal code is required for validation.

        Args:
            country_code: ISO 2-letter country code

        Returns:
            True if postal code is required, False if optional
        """
        country_config = self.get_country_config(country_code)
        postal_config = country_config.get("postal_code", {})

        # If explicitly configured, use that value
        if "required" in postal_config:
            return postal_config["required"]

        # Default: postal code is required unless in optional list
        optional_countries = self._config.get("postal_code_optional_countries", [])
        return country_code.upper() not in optional_countries

    def get_postal_code_pattern(self, country_code: str) -> Optional[str]:
        """
        Get regex pattern for postal code extraction.

        Args:
            country_code: ISO 2-letter country code

        Returns:
            Regex pattern string or None if not configured
        """
        country_config = self.get_country_config(country_code)
        postal_config = country_config.get("postal_code", {})
        return postal_config.get("pattern")

    def get_address_patterns(self, country_code: str) -> Dict[str, List[str]]:
        """
        Get country-specific address extraction patterns.

        Args:
            country_code: ISO 2-letter country code

        Returns:
            Dict with keys like 'building', 'unit_tower', 'tower_name', 'street',
            'extraction_patterns' containing lists of regex patterns
        """
        country_config = self.get_country_config(country_code)
        return country_config.get("address_patterns", {})

    def get_unit_patterns(self, country_code: str) -> List[str]:
        """
        Get regex patterns for unit number recognition.

        Args:
            country_code: ISO 2-letter country code

        Returns:
            List of regex pattern strings
        """
        country_config = self.get_country_config(country_code)
        return country_config.get("unit_patterns", [])

    def get_city_aliases(self, country_code: str) -> Dict[str, str]:
        """
        Get city name aliases for common spelling variations.

        Maps common OCR spellings to official city names.

        Args:
            country_code: ISO 2-letter country code

        Returns:
            Dict mapping aliases to official names (e.g., {"BANGALORE": "BENGALURU"})
        """
        country_config = self.get_country_config(country_code)
        return country_config.get("city_aliases", {})

    def get_country_name_aliases(self, country_code: str) -> List[str]:
        """
        Get name aliases for a country code.

        Returns variations like "uae", "united arab emirates" for "AE".

        Args:
            country_code: ISO 2-letter country code

        Returns:
            List of name alias strings
        """
        country_config = self.get_country_config(country_code)
        return country_config.get("name_aliases", [])

    def get_bank_regulatory_patterns(self, country_code: str) -> List[str]:
        """
        Get bank regulatory patterns for a country.

        These patterns help identify bank's registered country from
        regulatory text in documents (e.g., "Central Bank of UAE").

        Args:
            country_code: ISO 2-letter country code

        Returns:
            List of regex pattern strings
        """
        country_config = self.get_country_config(country_code)
        return country_config.get("bank_regulatory_patterns", [])

    def get_bank_abbreviations(self, country_code: str) -> List[str]:
        """
        Get bank abbreviations for a country.

        Args:
            country_code: ISO 2-letter country code

        Returns:
            List of bank abbreviation strings (e.g., ["FAB", "ENBD", "ADCB"] for UAE)
        """
        country_config = self.get_country_config(country_code)
        return country_config.get("bank_abbreviations", [])

    def get_header_patterns(self, country_code: str) -> List[str]:
        """
        Get patterns for detecting country in bank document headers.

        Args:
            country_code: ISO 2-letter country code

        Returns:
            List of regex pattern strings for header detection
        """
        country_config = self.get_country_config(country_code)
        return country_config.get("header_patterns", [])

    def get_inference_patterns(self, country_code: str) -> List[str]:
        """
        Get patterns for inferring country from address text.

        Args:
            country_code: ISO 2-letter country code

        Returns:
            List of pattern strings for country inference
        """
        country_config = self.get_country_config(country_code)
        return country_config.get("inference_patterns", [])

    def get_supported_countries(self) -> List[str]:
        """
        Get list of supported country codes.

        Returns:
            List of ISO 2-letter country codes
        """
        if not self._config_loaded:
            return []
        return self._config.get("supported_countries", [])

    def get_dynamic_location_countries(self) -> Set[str]:
        """
        Get countries for which we dynamically load cities/states from library.

        Returns:
            Set of ISO 2-letter country codes
        """
        if not self._config_loaded:
            return set()
        countries = self._config.get("dynamic_location_countries", [])
        return set(countries)

    def is_config_loaded(self) -> bool:
        """Check if the config file was successfully loaded."""
        return self._config_loaded

    def get_all_country_inference_patterns(self) -> Dict[str, List[str]]:
        """
        Get all country inference patterns for address matching.

        Returns a dict mapping country codes to their inference patterns.
        Useful for the _infer_country_from_address method.

        Returns:
            Dict mapping country codes to lists of pattern strings
        """
        if not self._config_loaded:
            return {}

        result = {}
        for country_code in self.get_supported_countries():
            patterns = self.get_inference_patterns(country_code)
            if patterns:
                result[country_code] = patterns
        return result

    def get_currency_country_map(self) -> Dict[str, str]:
        """
        Get mapping of ISO currency codes to ISO country codes.

        Used to infer the bank's country from the currency used in the statement.
        For example, AED -> AE (UAE Dirham to United Arab Emirates).

        Returns:
            Dict mapping currency codes (e.g., "AED", "SGD") to country codes (e.g., "AE", "SG")
        """
        # Standard currency to country mapping
        # Note: Some currencies are used by multiple countries (e.g., USD by US, PA, etc.)
        # This mapping uses the primary/most common country for each currency.
        return {
            # UAE
            "AED": "AE",
            # Singapore
            "SGD": "SG",
            # India
            "INR": "IN",
            # United States
            "USD": "US",
            # United Kingdom
            "GBP": "GB",
            # Malaysia
            "MYR": "MY",
            # Thailand
            "THB": "TH",
            # Hong Kong
            "HKD": "HK",
            # Myanmar
            "MMK": "MM",
            # Other common currencies
            "EUR": "DE",  # Eurozone - use Germany as default
            "JPY": "JP",  # Japan
            "CNY": "CN",  # China
            "AUD": "AU",  # Australia
            "CAD": "CA",  # Canada
            "CHF": "CH",  # Switzerland
            "NZD": "NZ",  # New Zealand
            "KRW": "KR",  # South Korea
            "PHP": "PH",  # Philippines
            "IDR": "ID",  # Indonesia
            "VND": "VN",  # Vietnam
            "SAR": "SA",  # Saudi Arabia
            "QAR": "QA",  # Qatar
            "KWD": "KW",  # Kuwait
            "BHD": "BH",  # Bahrain
            "OMR": "OM",  # Oman
            "PKR": "PK",  # Pakistan
            "LKR": "LK",  # Sri Lanka
            "NPR": "NP",  # Nepal
            "BDT": "BD",  # Bangladesh
            "ZAR": "ZA",  # South Africa
            "NGN": "NG",  # Nigeria
            "EGP": "EG",  # Egypt
            "ILS": "IL",  # Israel
            "TRY": "TR",  # Turkey
            "RUB": "RU",  # Russia
            "BRL": "BR",  # Brazil
            "MXN": "MX",  # Mexico
            "ARS": "AR",  # Argentina
            "COP": "CO",  # Colombia
            "CLP": "CL",  # Chile
            "PEN": "PE",  # Peru
            "CZK": "CZ",  # Czech Republic
            "PLN": "PL",  # Poland
            "SEK": "SE",  # Sweden
            "NOK": "NO",  # Norway
            "DKK": "DK",  # Denmark
            "HUF": "HU",  # Hungary
            "RON": "RO",  # Romania
            "BGN": "BG",  # Bulgaria
            "HRK": "HR",  # Croatia
            "RSD": "RS",  # Serbia
            "UAH": "UA",  # Ukraine
            "GEL": "GE",  # Georgia
            "KZT": "KZ",  # Kazakhstan
            "UZS": "UZ",  # Uzbekistan
            "AZN": "AZ",  # Azerbaijan
            "KGS": "KG",  # Kyrgyzstan
            "TJS": "TJ",  # Tajikistan
            "TMT": "TM",  # Turkmenistan
            "MNT": "MN",  # Mongolia
        }


def get_country_config_loader() -> BankStatementCountryConfigLoader:
    """
    Get the global BankStatementCountryConfigLoader instance.

    Creates the instance on first call (lazy initialization).

    Returns:
        BankStatementCountryConfigLoader singleton instance
    """
    global _instance
    if _instance is None:
        _instance = BankStatementCountryConfigLoader()
    return _instance


def reset_instance() -> None:
    """
    Reset the global instance (useful for testing).
    """
    global _instance
    _instance = None

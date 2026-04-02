"""
Address Keywords Loader

Loads country-specific address keywords from JSON config.
Dynamically loads cities and states from countrystatecity-countries library.
Used by bank statement extractor to detect addresses using
country-appropriate patterns (e.g., UAE uses TOWER, PO BOX,
while Singapore uses BLK, STREET, etc.).

Usage:
    loader = get_address_keywords_loader()
    keywords = loader.get_keywords('AE')  # Returns UAE-specific keywords
    unit_patterns = loader.get_unit_patterns('AE')
    postal_patterns = loader.get_postal_patterns('AE')
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Set

logger = logging.getLogger(__name__)

# Lazy import for countrystatecity library
_csc_cache: Dict[str, Any] = {}


def _get_csc_module():
    """Lazy load the countrystatecity_countries module."""
    if "module" not in _csc_cache:
        try:
            from countrystatecity_countries import (
                get_states_of_country,
                get_cities_of_country,
            )
            _csc_cache["module"] = True
            _csc_cache["get_states_of_country"] = get_states_of_country
            _csc_cache["get_cities_of_country"] = get_cities_of_country
        except ImportError:
            logger.warning("countrystatecity-countries not installed. City/state keywords will be limited.")
            _csc_cache["module"] = None
    return _csc_cache


def _load_country_locations(country_code: str) -> Set[str]:
    """
    Load all states and cities for a country from the countrystatecity library.

    Args:
        country_code: ISO 2-letter country code (e.g., "IN", "US")

    Returns:
        Set of uppercase location names (states + cities)
    """
    locations: Set[str] = set()
    csc = _get_csc_module()

    if csc.get("module") is None:
        return locations

    try:
        # Load states
        states = csc["get_states_of_country"](country_code)
        for state in states:
            # Add state name (handle multi-word states like "ANDHRA PRADESH")
            state_name = state.name.upper()
            locations.add(state_name)

        # Load cities (for India this will be 7000+ cities)
        cities = csc["get_cities_of_country"](country_code)
        for city in cities:
            city_name = city.name.upper()
            locations.add(city_name)

        logger.info(f"Loaded {len(states)} states and {len(cities)} cities for {country_code}")

    except Exception as e:
        logger.error(f"Failed to load locations for {country_code}: {type(e).__name__}")

    return locations


class AddressKeywordsLoader:
    """
    Load and cache country-specific address keywords from JSON config.
    Dynamically loads cities and states from countrystatecity-countries library.

    Provides methods to retrieve:
    - Address keywords for a country (TOWER, PO BOX for UAE, etc.)
    - Unit number patterns (standalone digits for UAE unit numbers)
    - Postal code patterns (P.O. Box for UAE, 6-digit for SG/IN, etc.)
    """

    # Default keywords used when country not found
    DEFAULT_KEYWORDS = [
        "STREET", "ST", "ROAD", "RD", "LANE", "LN", "AVENUE", "AVE",
        "BLOCK", "BLK", "BUILDING", "UNIT", "FLAT", "APARTMENT",
        "PO BOX", "POSTAL", "ZIP", "NEAR", "OPPOSITE", "OPP"
    ]

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the address keywords loader.

        Args:
            config_path: Path to the address_keywords.json file.
                        If None, uses default path in app/config/.
        """
        self.keywords_by_country: Dict[str, Dict[str, Any]] = {}
        self._config_loaded = False
        self._dynamic_locations: Dict[str, Set[str]] = {}  # Cache for loaded locations

        # Determine config path
        if config_path is None:
            config_path = str(
                Path(__file__).parent / "address_keywords.json"
            )

        self._load_config(config_path)

    def _load_config(self, config_path: str) -> None:
        """
        Load address keywords from the config file.

        Args:
            config_path: Path to address_keywords.json
        """
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                logger.warning(f"Address keywords config not found at {config_path}, using defaults")
                return

            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # Load default keywords
            if "default" in config:
                self.keywords_by_country["DEFAULT"] = config["default"]

            # Load country-specific keywords
            countries = config.get("countries", {})
            for country_code, country_data in countries.items():
                self.keywords_by_country[country_code.upper()] = country_data
                logger.debug(f"Loaded address keywords for {country_code}")

            self._config_loaded = True
            logger.info(f"Loaded address keywords for {len(self.keywords_by_country)} countries/regions")

        except Exception as e:
            logger.error(f"Failed to load address keywords config: {type(e).__name__}")
            # Continue with default keywords only

    def get_keywords(self, country_code: Optional[str] = None) -> List[str]:
        """
        Get address keywords for a country.

        Combines static keywords from JSON config with dynamically loaded
        cities and states from the countrystatecity-countries library.

        Falls back to default keywords if country not defined.

        Args:
            country_code: ISO 2-letter country code (e.g., "AE", "SG", "IN")

        Returns:
            List of keyword strings to match against
        """
        country_code = (country_code or "DEFAULT").upper()

        # Start with static keywords from config
        static_keywords: List[str] = []
        if country_code in self.keywords_by_country:
            static_keywords = self.keywords_by_country[country_code].get("keywords", [])
        elif country_code == "DEFAULT":
            default_data = self.keywords_by_country.get("DEFAULT", {})
            static_keywords = default_data.get("keywords", self.DEFAULT_KEYWORDS)

        if not static_keywords:
            default_data = self.keywords_by_country.get("DEFAULT", {})
            static_keywords = default_data.get("keywords", self.DEFAULT_KEYWORDS)

        # For countries with dynamic location loading, combine with cities/states
        # Get from bank statement country config
        from app.config.bank_statement_country_loader import get_country_config_loader
        country_loader = get_country_config_loader()
        dynamic_location_countries = country_loader.get_dynamic_location_countries()

        if country_code in dynamic_location_countries:
            if country_code not in self._dynamic_locations:
                # Lazy load locations on first request
                self._dynamic_locations[country_code] = _load_country_locations(country_code)

            dynamic_locations = self._dynamic_locations[country_code]
            # Combine static keywords with dynamic locations (avoiding duplicates)
            all_keywords = list(static_keywords) + [loc for loc in dynamic_locations if loc not in static_keywords]
            return all_keywords

        return static_keywords

    def get_unit_patterns(self, country_code: Optional[str] = None) -> List[str]:
        """
        Get regex patterns for unit number recognition.

        Some countries have specific unit number patterns:
        - UAE: standalone 3-5 digit numbers (e.g., "3109")
        - Singapore: #11-25 format

        Args:
            country_code: ISO 2-letter country code

        Returns:
            List of regex pattern strings
        """
        country_code = (country_code or "DEFAULT").upper()

        if country_code in self.keywords_by_country:
            return self.keywords_by_country[country_code].get("unit_patterns", [])

        return []

    def get_postal_patterns(self, country_code: Optional[str] = None) -> List[str]:
        """
        Get regex patterns for postal code extraction.

        Different countries use different postal formats:
        - UAE: P.O. Box 38103
        - Singapore/India: 6-digit (123456)
        - Malaysia/Thailand/US: 5-digit (12345)
        - UK: SW1A 1AA

        Args:
            country_code: ISO 2-letter country code

        Returns:
            List of regex pattern strings
        """
        country_code = (country_code or "DEFAULT").upper()

        if country_code in self.keywords_by_country:
            # Get patterns for this country (may be empty list)
            country_data = self.keywords_by_country[country_code]
            if "postal_patterns" in country_data:
                # Return patterns even if empty (explicitly no postal patterns)
                return country_data["postal_patterns"]

        # Fallback to default patterns
        default_data = self.keywords_by_country.get("DEFAULT", {})
        return default_data.get("postal_patterns", [r"\b\d{5,6}\b"])

    def matches_address_keyword(self, text: str, country_code: Optional[str] = None) -> bool:
        """
        Check if text contains any address keyword for the country.

        Uses word boundary matching to avoid false positives
        (e.g., "STATEMENT" should not match "STATE").

        Args:
            text: Text to check
            country_code: ISO 2-letter country code

        Returns:
            True if text contains an address keyword
        """
        if not text:
            return False

        text_upper = text.upper()
        keywords = self.get_keywords(country_code)

        for keyword in keywords:
            # Escape special regex characters in keyword
            escaped_keyword = re.escape(keyword)
            # Use word boundaries to avoid partial matches
            if re.search(r'\b' + escaped_keyword + r'\b', text_upper):
                return True

        return False

    def matches_unit_pattern(self, text: str, country_code: Optional[str] = None) -> bool:
        """
        Check if text matches a unit number pattern for the country.

        Args:
            text: Text to check
            country_code: ISO 2-letter country code

        Returns:
            True if text matches a unit pattern
        """
        if not text:
            return False

        text_clean = text.strip()
        unit_patterns = self.get_unit_patterns(country_code)

        for pattern in unit_patterns:
            try:
                if re.match(pattern, text_clean):
                    return True
            except re.error:
                logger.warning(f"Invalid unit pattern: {pattern}")

        return False

    def extract_postal_code(self, text: str, country_code: Optional[str] = None) -> Optional[str]:
        """
        Extract postal code from text using country-specific patterns.

        Args:
            text: Text to extract postal code from
            country_code: ISO 2-letter country code

        Returns:
            Extracted postal code or None
        """
        if not text:
            return None

        postal_patterns = self.get_postal_patterns(country_code)

        for pattern in postal_patterns:
            try:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return match.group(0)
            except re.error:
                logger.warning(f"Invalid postal pattern: {pattern}")

        return None

    def is_config_loaded(self) -> bool:
        """Check if the config file was successfully loaded."""
        return self._config_loaded

    def get_supported_countries(self) -> List[str]:
        """Get list of countries with custom address keywords defined."""
        return [c for c in self.keywords_by_country.keys() if c != "DEFAULT"]


# Global singleton instance
_instance: Optional[AddressKeywordsLoader] = None


def get_address_keywords_loader() -> AddressKeywordsLoader:
    """
    Get the global AddressKeywordsLoader instance.

    Creates the instance on first call (lazy initialization).

    Returns:
        AddressKeywordsLoader singleton instance
    """
    global _instance
    if _instance is None:
        _instance = AddressKeywordsLoader()
    return _instance


def reset_instance() -> None:
    """
    Reset the global instance (useful for testing).
    """
    global _instance
    _instance = None

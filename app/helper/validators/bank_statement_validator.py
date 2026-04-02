"""
Bank Statement Validator

Validates bank statement fields against reference data from config.
Includes credit card detection using Luhn algorithm.
Uses countrystatecity-countries library for city/state data.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from app.config.address_keywords_loader import get_address_keywords_loader
from app.config.bank_statement_country_loader import get_country_config_loader

logger = logging.getLogger(__name__)

# Cache for dynamically loaded locations
_location_cache: Dict[str, Dict[str, Set[str]]] = {}
# Cache for city-to-state mapping
_city_to_state_cache: Dict[str, Dict[str, str]] = {}


def _get_state_from_city(city_name: str, country_code: str = None) -> Optional[str]:
    """
    Get state from city name using countrystatecity library.

    Args:
        city_name: Name of the city
        country_code: ISO 2-letter country code (optional, searches all countries if not provided)

    Returns:
        State name if found, None otherwise
    """
    city_upper = city_name.upper()

    # Determine which countries to search
    if country_code:
        search_countries = [country_code]
    else:
        # Get supported countries from config
        country_loader = get_country_config_loader()
        search_countries = country_loader.get_supported_countries()

    for country in search_countries:
        if not country:
            continue

        # Build city-to-state mapping for this country if not cached
        if country not in _city_to_state_cache:
            _city_to_state_cache[country] = {}
            try:
                from countrystatecity_countries import get_cities_of_country, get_states_of_country

                cities = get_cities_of_country(country)
                states = get_states_of_country(country)
                # Create state_id to state_name mapping
                state_map = {s.id: s.name for s in states}

                for city in cities:
                    # city.state_id is the ID of the state this city belongs to
                    if hasattr(city, 'state_id') and city.state_id in state_map:
                        _city_to_state_cache[country][city.name.upper()] = state_map[city.state_id].upper()
                    elif hasattr(city, 'state_name'):
                        # Fallback to state_name attribute if available
                        _city_to_state_cache[country][city.name.upper()] = city.state_name.upper()

                logger.debug(f"Loaded city-to-state mapping for {country}: {len(_city_to_state_cache[country])} cities")
            except Exception as e:
                logger.warning(f"Failed to load city-to-state mapping for {country}: {type(e).__name__}: {e}")

        # Look up the city
        if city_upper in _city_to_state_cache[country]:
            return _city_to_state_cache[country][city_upper]

    return None


def _load_country_locations(country_code: str) -> Dict[str, Set[str]]:
    """
    Load states and cities for a country from the countrystatecity library.

    Args:
        country_code: ISO 2-letter country code

    Returns:
        Dict with 'states' and 'cities' sets
    """
    if country_code in _location_cache:
        return _location_cache[country_code]

    result: Dict[str, Set[str]] = {"states": set(), "cities": set()}

    try:
        from countrystatecity_countries import (
            get_states_of_country,
            get_cities_of_country,
        )

        # Load states
        states = get_states_of_country(country_code)
        for state in states:
            result["states"].add(state.name.upper())

        # Load cities
        cities = get_cities_of_country(country_code)
        for city in cities:
            result["cities"].add(city.name.upper())

        logger.debug(f"Loaded {len(states)} states and {len(cities)} cities for {country_code}")

    except ImportError:
        logger.warning("countrystatecity-countries not installed, city/state detection limited")
    except Exception as e:
        logger.error(f"Failed to load locations for {country_code}: {type(e).__name__}")

    _location_cache[country_code] = result
    return result


def _get_all_known_cities(countries: List[str] = None) -> Set[str]:
    """
    Get all known cities for specified countries.

    Args:
        countries: List of country codes, or None for all supported countries

    Returns:
        Set of uppercase city names
    """
    if countries is None:
        # Get supported countries from config
        country_loader = get_country_config_loader()
        countries = country_loader.get_supported_countries()

    all_cities: Set[str] = set()
    for country in countries:
        locations = _load_country_locations(country)
        all_cities.update(locations["cities"])

    return all_cities


def _get_all_known_states(countries: List[str] = None) -> Set[str]:
    """
    Get all known states for specified countries.

    Args:
        countries: List of country codes, or None for all supported countries

    Returns:
        Set of uppercase state names
    """
    if countries is None:
        # Get supported countries from config
        country_loader = get_country_config_loader()
        countries = country_loader.get_supported_countries()

    all_states: Set[str] = set()
    for country in countries:
        locations = _load_country_locations(country)
        all_states.update(locations["states"])

    return all_states


def _infer_country_from_state_dynamic(state: str, countries: List[str] = None) -> Optional[str]:
    """
    Infer country from state name using the countrystatecity library.

    Args:
        state: State name to look up
        countries: List of country codes to search, or None for all

    Returns:
        ISO country code or None
    """
    if not state:
        return None

    if countries is None:
        # Get supported countries from config
        country_loader = get_country_config_loader()
        countries = country_loader.get_supported_countries()

    state_upper = state.strip().upper()

    for country in countries:
        locations = _load_country_locations(country)
        if state_upper in locations["states"]:
            return country

    return None


class BankStatementValidator:
    """
    Validator for bank statement fields.

    Provides validation for:
    - Account number format and length (per currency)
    - Credit card detection (Luhn algorithm)
    - Address component parsing and validation
    - Country inference from state
    """

    _instance = None
    _config = None

    def __new__(cls):
        """Singleton pattern to load config only once."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self) -> None:
        """Load configuration from JSON file."""
        config_path = Path(__file__).parent.parent.parent / "reference_templates" / "bank_statements" / "config.json"

        try:
            with open(config_path, 'r') as f:
                self._config = json.load(f)
            logger.info(f"Loaded bank statement config from {config_path}")
        except FileNotFoundError:
            logger.warning(f"Bank statement config not found at {config_path}, using defaults")
            self._config = self._get_default_config()
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in bank statement config: {e}")
            self._config = self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration if file not found."""
        return {
            "currencies": {
                "SGD": {"name": "Singapore Dollar", "country": "SG", "account_number_length": {"min": 8, "max": 20}},
                "INR": {"name": "Indian Rupee", "country": "IN", "account_number_length": {"min": 8, "max": 20}},
                "MYR": {"name": "Malaysian Ringgit", "country": "MY", "account_number_length": {"min": 8, "max": 20}},
                "THB": {"name": "Thai Baht", "country": "TH", "account_number_length": {"min": 8, "max": 20}},
                "IDR": {"name": "Indonesian Rupiah", "country": "ID", "account_number_length": {"min": 8, "max": 20}},
                "USD": {"name": "US Dollar", "country": "US", "account_number_length": {"min": 8, "max": 20}},
                "GBP": {"name": "British Pound", "country": "GB", "account_number_length": {"min": 8, "max": 20}},
                "HKD": {"name": "Hong Kong Dollar", "country": "HK", "account_number_length": {"min": 8, "max": 20}},
                "AED": {"name": "UAE Dirham", "country": "AE", "account_number_length": {"min": 8, "max": 20}},
            },
            "state_to_country_map": {},
            "account_number_labels": [
                "ACCOUNT NUMBER", "ACCOUNT NO", "ACCOUNT NO.", "A/C NO", "A/C NO.", "AC NO",
                "SAVINGS A/C", "CURRENT A/C", "SAVINGS ACCOUNT", "CURRENT ACCOUNT"
            ],
            "address_labels": [
                "ADDRESS", "RESIDENTIAL ADDRESS", "CUSTOMER ADDRESS", "MAILING ADDRESS",
                "PERMANENT ADDRESS", "COMMUNICATION ADDRESS"
            ]
        }

    # ============================================================
    # LABEL ACCESSORS
    # ============================================================

    def get_account_number_labels(self) -> List[str]:
        """Return account number labels from config."""
        return self._config.get("account_number_labels", [])

    def get_address_labels(self) -> List[str]:
        """Return address labels from config."""
        return self._config.get("address_labels", [])

    def get_address_extraction_exceptions(self) -> Dict[str, List[str]]:
        """Return address extraction exceptions from config."""
        return self._config.get("address_extraction_exceptions", {})

    # ============================================================
    # CURRENCY VALIDATION
    # ============================================================

    def get_currency_info(self, currency: str) -> Optional[Dict[str, Any]]:
        """
        Get currency information including account number length specs.

        Args:
            currency: Currency code (e.g., "SGD", "INR")

        Returns:
            Currency info dict or None if not found
        """
        if not currency:
            return None

        currencies = self._config.get("currencies", {})
        return currencies.get(currency.upper())

    def get_currency_name_map(self) -> Dict[str, str]:
        """
        Get mapping of currency names to ISO codes.

        Returns:
            Dict mapping names like "UAE DIRHAM" to codes like "AED"
        """
        return self._config.get("currency_name_map", {})

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

        Uses the currencies mapping in config.json to find which currency
        is associated with the given country code.

        Args:
            country: ISO 2-letter country code (e.g., "IN", "SG", "AE")

        Returns:
            Currency code (e.g., "INR", "SGD") or None if not found
        """
        if not country:
            return None

        currencies = self._config.get("currencies", {})
        country_upper = country.upper()

        for currency_code, currency_info in currencies.items():
            if currency_info.get("country") == country_upper:
                return currency_code

        return None

    # ============================================================
    # COUNTRY INFERENCE
    # ============================================================

    def infer_country_from_state(self, state: str) -> Optional[str]:
        """
        Infer country code from state name.

        Uses dynamic lookup from countrystatecity library first,
        falls back to static state_to_country_map from config.

        Args:
            state: State name (e.g., "Maharashtra", "Selangor")

        Returns:
            ISO country code or None
        """
        if not state:
            return None

        # Try dynamic lookup first
        result = _infer_country_from_state_dynamic(state)
        if result:
            return result

        # Fallback to static map for edge cases (alternate spellings, etc.)
        state_clean = state.strip().title()
        state_map = self._config.get("state_to_country_map", {})
        return state_map.get(state_clean)

    # ============================================================
    # ADDRESS PARSING
    # ============================================================

    def parse_address_components(
        self,
        address: str,
        country_hint: str = None
    ) -> Dict[str, Optional[str]]:
        """
        Parse address into structured components.

        Args:
            address: Full address string
            country_hint: Optional country code for format hints

        Returns:
            Dict with keys: street_number, street_name, city, postal_code, state, country
        """
        import logging
        logger = logging.getLogger(__name__)

        if not address:
            logger.warning("parse_address_components: empty address provided")
            return {
                "street_number": None,
                "street_name": None,
                "city": None,
                "postal_code": None,
                "state": None,
                "country": None
            }

        logger.info(f"parse_address_components: address='{address[:100]}...', country_hint='{country_hint}'")
        components = {
            "street_number": None,
            "street_name": None,
            "city": None,
            "postal_code": None,
            "state": None,
            "country": None
        }

        # Get country-specific postal patterns from loader
        loader = get_address_keywords_loader()
        country_loader = get_country_config_loader()

        # Check if postal codes are used/required for this country
        # For countries where postal codes are optional/unused, skip extraction entirely
        # This prevents false positives from generic pattern matching (e.g., UAE PO Box numbers)
        is_postal_optional = country_hint is not None and not country_loader.is_postal_code_required(country_hint)

        # For countries where postal codes are optional/unused, skip extraction entirely
        # This prevents false positives from generic pattern matching
        if is_postal_optional and not loader.get_postal_patterns(country_hint):
            # Country explicitly has no postal patterns and postal is optional
            # Skip postal code extraction to avoid PO Box/unit number misidentification
            components["postal_code"] = None
        else:
            # Continue with normal postal code extraction
            postal_patterns = loader.get_postal_patterns(country_hint)

            # Extract postal code using country-specific patterns
            for pattern in postal_patterns:
                try:
                    match = re.search(pattern, address, re.IGNORECASE)
                    if match:
                        components["postal_code"] = match.group(0)
                        break
                except re.error:
                    continue

            # Fallback to generic patterns if no country-specific match
            if not components["postal_code"]:
                generic_postal_patterns = [
                    r'\b(\d{6})\b',  # 6-digit (SG, IN)
                    r'\b(\d{5})\b',  # 5-digit (MY, TH, US)
                    r'\b(\d{4})\b',  # 4-digit (AU, NZ)
                ]

                # Infer country from address context for unit pattern exclusion
                inferred_country = country_hint or components.get("country")
                if not inferred_country:
                    # Try to infer from address content
                    address_upper = address.upper()

                    # Check UAE emirates from config
                    uae_subdivisions = country_loader.get_subdivisions("AE")
                    if any(emirate.upper() in address_upper for emirate in uae_subdivisions):
                        inferred_country = "AE"
                    elif 'HONG KONG' in address_upper:
                        inferred_country = "HK"

                # Get unit patterns for the inferred country
                unit_patterns = []
                if inferred_country:
                    unit_patterns = loader.get_unit_patterns(inferred_country)

                for pattern in generic_postal_patterns:
                    match = re.search(pattern, address)
                    if match:
                        potential_postal_code = match.group(1)
                        skip_this_code = False

                        # Check if this is part of a PO Box pattern - exclude it
                        # PO Box formats: "P.O. BOX 12345", "PO BOX 12345", "BOX 12345", "P.O.BOX 12345"
                        po_box_context = re.search(
                            r'(?:P\.?O\.?\s*(?:BOX)?\s*|BOX\s*|POST\s+OFFICE\s*BOX)\s*\d+',
                            address,
                            re.IGNORECASE
                        )
                        if po_box_context:
                            # Check if the matched number is part of the PO Box
                            po_box_numbers = re.findall(r'\d+', po_box_context.group(0))
                            if potential_postal_code in po_box_numbers:
                                # This is a PO Box number, not a postal code - skip it
                                skip_this_code = True

                        # Check if this is a unit number pattern for the country
                        # For UAE, standalone 3-5 digit numbers are unit numbers, not postal codes
                        if not skip_this_code and unit_patterns:
                            for unit_pattern in unit_patterns:
                                try:
                                    # Check if the potential postal code matches a unit pattern
                                    unit_pattern_clean = unit_pattern
                                    if unit_pattern_clean.startswith('^'):
                                        unit_pattern_clean = unit_pattern_clean[1:]
                                    if unit_pattern_clean.endswith('$'):
                                        unit_pattern_clean = unit_pattern_clean[:-1]
                                    if re.fullmatch(unit_pattern_clean, potential_postal_code):
                                        # This is a unit number, not a postal code - skip it
                                        skip_this_code = True
                                        break
                                except re.error:
                                    pass

                        # Check if this looks like a phone number (5-8 digits often appear near phone labels)
                        # Phone numbers shouldn't be extracted as postal codes
                        if not skip_this_code:
                            phone_context = re.search(
                                r'(?:TEL|TELEPHONE|PHONE|MOBILE|FAX|CONTACT|\+971|\+65|\+91)\s*:?\s*(\d{' + str(len(potential_postal_code)) + r'})',
                                address,
                                re.IGNORECASE
                            )
                            if phone_context:
                                # The matched number appears in a phone context
                                skip_this_code = True

                        if not skip_this_code:
                            components["postal_code"] = potential_postal_code
                            logger.info(f"Set postal_code to '{potential_postal_code}' using generic pattern")
                            break
                        else:
                            logger.info(f"Skipped potential postal code '{potential_postal_code}' (skip_reason: phone_context={bool(phone_context)}, po_box_context={bool(po_box_context)}, unit_pattern_match={bool(unit_patterns)})")

        # Try to infer country from state
        # Use dynamic lookup from countrystatecity library
        address_upper = address.upper()
        country_loader = get_country_config_loader()

        # UAE emirates (can appear as city or state)
        uae_subdivisions = country_loader.get_subdivisions("AE")
        for subdivision in uae_subdivisions:
            if subdivision.upper() in address_upper:
                components["state"] = subdivision.title()
                components["country"] = "AE"
                # Also set city if not already set (emirates often serve as both)
                if not components["city"]:
                    components["city"] = subdivision.title()
                break

        # Try dynamic state lookup for common countries
        if not components["country"]:
            search_countries = country_loader.get_supported_countries()
            for country in search_countries:
                locations = _load_country_locations(country)
                for state in locations["states"]:
                    # Try exact match first
                    if state in address_upper:
                        components["state"] = state.title()
                        components["country"] = country
                        break
                    # Try match without spaces
                    state_no_space = state.replace(' ', '')
                    address_no_space = address_upper.replace(' ', '')
                    if state_no_space in address_no_space:
                        components["state"] = state.title()
                        components["country"] = country
                        break
                if components["country"]:
                    break

        # Fallback to static map for edge cases (alternate spellings, etc.)
        if not components["country"]:
            state_map = self._config.get("state_to_country_map", {})
            for state_name, country_code in state_map.items():
                state_upper = state_name.upper()
                # Try exact match first
                if state_upper in address_upper:
                    components["state"] = state_name
                    components["country"] = country_code
                    break
                # Try match without spaces
                state_no_space = state_upper.replace(' ', '')
                address_no_space = address_upper.replace(' ', '')
                if state_no_space in address_no_space:
                    components["state"] = state_name
                    components["country"] = country_code
                    break

        # If country hint provided and no country inferred
        if country_hint and not components["country"]:
            components["country"] = country_hint.upper()

        # Try to extract street number and name
        # Standard pattern - search anywhere in address, not just at start
        # House number can include letters (e.g., "24B", "12A")
        # Street suffixes include common abbreviations like "SQUA" for "SQUARE"
        street_suffixes = r'(?:STREET|ST|ROAD|RD|AVENUE|AVE|LANE|LN|DRIVE|DR|CRESCENT|CRES|WAY|PLACE|PL|CLOSE|TERRACE|TERR|SQUARE|SQ|SQUA|BOULEVARD|BLVD|COURT|CT)'
        street_pattern = rf'(\d+[A-Z]?)\s*[-,]?\s*([A-Za-z][A-Za-z\s]+?)\s+{street_suffixes}'
        street_match = re.search(street_pattern, address, re.IGNORECASE)
        if street_match:
            components["street_number"] = street_match.group(1)
            components["street_name"] = street_match.group(2).strip()

        # Get country-specific address patterns from config
        country_loader = get_country_config_loader()

        # Check for inferred country for pattern selection
        inferred_country = country_hint or components.get("country")

        # UAE-style patterns: Villa X, Building X, Plot X, Tower X, Block X
        if (not components["street_number"] or not components["street_name"]) and inferred_country == "AE":
            address_patterns = country_loader.get_address_patterns("AE")
            uae_patterns = address_patterns.get("building", [])

            for pattern in uae_patterns:
                match = re.search(pattern, address, re.IGNORECASE)
                if match:
                    # For UAE patterns, the first group is the building/unit, second is location info
                    if not components["street_number"]:
                        # Extract number from the villa/building identifier
                        num_match = re.search(r'\d+', match.group(1))
                        if num_match:
                            components["street_number"] = num_match.group(0)
                    if not components["street_name"]:
                        components["street_name"] = match.group(2).strip()
                    break

        # Additional UAE pattern: Unit number followed by Tower name
        # e.g., "3109, TAMOUH TOWER" or "3109 TAMOUH TOWER"
        if (not components["street_number"] or not components["street_name"]) and inferred_country == "AE":
            address_patterns = country_loader.get_address_patterns("AE")
            unit_tower_patterns = address_patterns.get("unit_tower", [])

            for pattern in unit_tower_patterns:
                unit_tower_match = re.search(pattern, address, re.IGNORECASE)
                if unit_tower_match:
                    if not components["street_number"]:
                        components["street_number"] = unit_tower_match.group(1)
                    if not components["street_name"]:
                        components["street_name"] = unit_tower_match.group(2).strip()
                    break

        # Additional UAE pattern: Any tower/building name containing "TOWER" or "BUILDING"
        if not components["street_name"] and inferred_country == "AE":
            address_patterns = country_loader.get_address_patterns("AE")
            tower_name_patterns = address_patterns.get("tower_name", [])

            for pattern in tower_name_patterns:
                tower_name_match = re.search(pattern, address, re.IGNORECASE)
                if tower_name_match:
                    components["street_name"] = tower_name_match.group(1).strip()
                    break

        # Additional UAE pattern: "Street X" or "St X" anywhere in address
        if not components["street_name"] and inferred_country == "AE":
            address_patterns = country_loader.get_address_patterns("AE")
            street_patterns = address_patterns.get("street", [])

            for pattern in street_patterns:
                street_match = re.search(pattern, address, re.IGNORECASE)
                if street_match:
                    components["street_name"] = street_match.group(1)
                    break

        # Indian address patterns: "H NO X", "PLOT NO X", "DOOR NO X", "FLAT NO X"
        if not components["street_name"] and inferred_country == "IN":
            address_patterns = country_loader.get_address_patterns("IN")
            indian_building_patterns = address_patterns.get("building", [])

            for pattern in indian_building_patterns:
                match = re.search(pattern, address, re.IGNORECASE)
                if match:
                    street = match.group(1).strip().rstrip(',')
                    if len(street) > 3 and not street.upper().startswith('INDIA'):
                        components["street_name"] = street
                        break

        # Try to find city (common patterns)
        # Use dynamically loaded cities from countrystatecity library
        # Prioritize cities from the hinted country first
        all_known_cities = _get_all_known_cities()

        # City name aliases for common spelling variations
        # Get from config for the inferred country
        city_aliases = country_loader.get_city_aliases(inferred_country) if inferred_country else {}

        # Apply city name aliases to address before matching
        address_upper = address.upper()
        for alias, official_name in city_aliases.items():
            # Replace alias with official name using word boundaries
            address_upper = re.sub(r'\b' + re.escape(alias) + r'\b', official_name, address_upper)

        # Get city extraction patterns from config
        city_patterns = []
        if inferred_country:
            address_patterns = country_loader.get_address_patterns(inferred_country)
            city_patterns = address_patterns.get("extraction_patterns", [])

        # Add generic city patterns as fallback
        city_patterns.extend([
            r',\s*([A-Z][a-z]+)\s+\d{5,6}',  # City before postal code
            r',\s*([A-Z][a-z]+)\s*$',  # City at end
        ])

        # First try cities from the hinted country (if provided)
        # Note: address_upper already has alias replacements applied above
        if country_hint:
            country_locations = _load_country_locations(country_hint.upper())
            country_cities = country_locations["cities"]
            # Sort cities by length (longest first) to match compound names first
            sorted_country_cities = sorted(country_cities, key=len, reverse=True)
            for city in sorted_country_cities:
                # Use word boundary matching to avoid partial matches
                if re.search(r'\b' + re.escape(city) + r'\b', address_upper):
                    components["city"] = city.title()
                    break

        # If no city found, try all known cities
        if not components["city"]:
            # Sort cities by length (longest first) to match compound names first
            sorted_cities = sorted(all_known_cities, key=len, reverse=True)
            for city in sorted_cities:
                # Use word boundary matching to avoid partial matches
                if re.search(r'\b' + re.escape(city) + r'\b', address_upper):
                    components["city"] = city.title()
                    break

        # Infer state from city if state not found
        if components["city"] and not components["state"]:
            state_from_city = _get_state_from_city(components["city"], components.get("country") or country_hint)
            if state_from_city:
                components["state"] = state_from_city.title()
                logger.debug(f"Inferred state '{state_from_city}' from city '{components['city']}'")

        # Then try pattern-based matching
        if not components["city"]:
            for pattern in city_patterns:
                match = re.search(pattern, address)
                if match:
                    potential_city = match.group(1)
                    # Don't use state names as cities
                    if potential_city.title() not in state_map:
                        components["city"] = potential_city
                        break

        logger.info(f"parse_address_components result: postal_code='{components.get('postal_code')}', city='{components.get('city')}', state='{components.get('state')}', country='{components.get('country')}'")
        return components

    def validate_address_components(
        self,
        components: Dict[str, Optional[str]]
    ) -> Tuple[bool, List[str]]:
        """
        Validate that address has all required components.

        Args:
            components: Dict from parse_address_components()

        Returns:
            Tuple of (is_valid, missing_components)
        """
        # Check if postal code is required using country config
        country_loader = get_country_config_loader()

        required = ["street_number", "street_name", "city", "postal_code", "state", "country"]
        missing = []

        country_code = components.get("country", "")

        # Check if country has subdivisions (states/provinces)
        # City-states like Singapore (SG) have no subdivisions
        country_config = country_loader.get_country_config(country_code)
        subdivisions = country_config.get("subdivisions", {})
        has_subdivisions = (
            subdivisions.get("type") != "none" and
            (subdivisions.get("list") or subdivisions.get("use_dynamic"))
        )

        for field in required:
            if not components.get(field):
                # street_number is optional in some countries
                if field == "street_number":
                    continue
                # Postal code optional for UAE and Gulf countries (config-driven)
                if field == "postal_code" and not country_loader.is_postal_code_required(country_code):
                    continue
                # State is optional for city-states (countries with no subdivisions)
                if field == "state" and not has_subdivisions:
                    continue
                missing.append(field.replace("_", " "))

        return len(missing) == 0, missing

    # ============================================================
    # ACCOUNT NUMBER VALIDATION
    # ============================================================

    def validate_account_number(
        self,
        account_number: str,
        currency: str = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate account number format.

        Checks:
        - Digits only
        - Length within range for currency
        - Not a credit card (Luhn check)

        Args:
            account_number: The account number to validate
            currency: Currency code for length validation

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not account_number:
            return False, "Account number is empty"

        # Remove spaces and hyphens
        cleaned = re.sub(r'[\s\-]', '', account_number)

        # Check for masked account numbers (containing masking characters)
        # Common masking patterns: X, *, •, -, _, #
        masking_pattern = r'[Xx\*•\-\_#]'
        has_masking = bool(re.search(masking_pattern, cleaned))

        # Reject masked account numbers - they don't contain the full account number needed for verification
        if has_masking:
            # If it has masking characters but also has non-digit characters, it's masked
            non_digit_chars = re.sub(r'[0-9\s]', '', cleaned)
            if non_digit_chars:
                return False, "Account number: Masked account numbers are not accepted. Please provide a document with the full account number visible."

        # Must be all digits
        if not cleaned.isdigit():
            return False, "Account number must contain only digits"

        # Check length
        min_len, max_len = self.get_account_number_length_range(currency)
        actual_len = len(cleaned)

        if actual_len < min_len or actual_len > max_len:
            return False, f"Account number: Invalid length ({actual_len}), expected {min_len}-{max_len} for {currency}"

        # Check if it's a credit card (16 digits + passes Luhn)
        if len(cleaned) == 16 and self._passes_luhn(cleaned):
            return False, "Account number: Appears to be a credit card number"

        return True, None

    def _passes_luhn(self, number: str) -> bool:
        """
        Check if a number passes the Luhn algorithm (credit card check).

        The Luhn algorithm is used to validate credit card numbers.
        If a 16-digit number passes this check, it's likely a credit card.

        Args:
            number: String of digits to check

        Returns:
            True if the number passes the Luhn check
        """
        if not number or not number.isdigit():
            return False

        # Luhn algorithm implementation
        total = 0
        reverse_digits = number[::-1]

        for i, digit in enumerate(reverse_digits):
            n = int(digit)

            # Double every second digit
            if i % 2 == 1:
                n *= 2
                # If doubling results in > 9, subtract 9
                if n > 9:
                    n -= 9

            total += n

        # Valid if total is divisible by 10
        return total % 10 == 0

    # ============================================================
    # GEOGRAPHIC DETECTION HELPER METHODS
    # ============================================================

    def text_contains_country(
        self,
        text: str,
        countries: List[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if text contains a known country name or ISO code.

        Args:
            text: Text to search
            countries: List of country codes to search, or None for all supported

        Returns:
            Tuple of (found, country_code)
        """
        if not text:
            return False, None

        country_loader = get_country_config_loader()
        search_countries = countries or country_loader.get_supported_countries()

        text_upper = text.upper()

        # First check for ISO codes (exact match)
        for country in search_countries:
            if country in text_upper:
                # Check word boundaries to avoid partial matches
                if re.search(r'\b' + re.escape(country) + r'\b', text_upper):
                    return True, country

        # Then check for full country names
        for country in search_countries:
            country_config = country_loader.get_country_config(country)
            if country_config:
                # Try both 'name' and 'country_name' keys
                country_name = country_config.get("name") or country_config.get("country_name", "")
                country_name = country_name.upper()
                if country_name and country_name in text_upper:
                    if re.search(r'\b' + re.escape(country_name) + r'\b', text_upper):
                        return True, country

                # Also check name_aliases
                name_aliases = country_config.get("name_aliases", [])
                for alias in name_aliases:
                    alias_upper = alias.upper()
                    if alias_upper and alias_upper in text_upper:
                        if re.search(r'\b' + re.escape(alias_upper) + r'\b', text_upper):
                            return True, country

        return False, None

    def text_contains_state(
        self,
        text: str,
        countries: List[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if text contains a known state/province name.

        Args:
            text: Text to search
            countries: List of country codes to search, or None for all supported

        Returns:
            Tuple of (found, state_name)
        """
        if not text:
            return False, None

        country_loader = get_country_config_loader()
        search_countries = countries or country_loader.get_supported_countries()

        text_upper = text.upper()
        text_no_space = text_upper.replace(' ', '')

        for country in search_countries:
            # Use dynamically loaded states from countrystatecity
            locations = _load_country_locations(country)
            states = locations["states"]

            # Sort by length (longest first) to match compound names
            sorted_states = sorted(states, key=len, reverse=True)

            for state in sorted_states:
                # Try exact match first
                if re.search(r'\b' + re.escape(state) + r'\b', text_upper):
                    return True, state

                # Try match without spaces (for "TamilNadu" style)
                state_no_space = state.replace(' ', '')
                if state_no_space and re.search(r'\b' + re.escape(state_no_space) + r'\b', text_no_space):
                    return True, state

        # Fallback to static map for edge cases
        state_map = self._config.get("state_to_country_map", {})
        for state_name in state_map.keys():
            state_upper = state_name.upper()
            if re.search(r'\b' + re.escape(state_upper) + r'\b', text_upper):
                return True, state_name
            if state_upper.replace(' ', '') in text_no_space:
                return True, state_name

        return False, None

    def text_contains_city(
        self,
        text: str,
        countries: List[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if text contains a known city name.

        Args:
            text: Text to search
            countries: List of country codes to search, or None for all supported

        Returns:
            Tuple of (found, city_name)
        """
        if not text:
            return False, None

        country_loader = get_country_config_loader()
        search_countries = countries or country_loader.get_supported_countries()

        # Get city aliases for better matching
        city_aliases = {}
        for country in search_countries:
            aliases = country_loader.get_city_aliases(country)
            city_aliases.update(aliases)

        # Apply aliases to text before matching
        text_upper = text.upper()
        for alias, official_name in city_aliases.items():
            text_upper = re.sub(r'\b' + re.escape(alias) + r'\b', official_name, text_upper)

        # Get all known cities
        all_cities = _get_all_known_cities(search_countries)

        # Sort by length (longest first) to match compound names
        sorted_cities = sorted(all_cities, key=len, reverse=True)

        for city in sorted_cities:
            if re.search(r'\b' + re.escape(city) + r'\b', text_upper):
                return True, city

        return False, None

    def get_geographic_matches(
        self,
        text: str,
        countries: List[str] = None
    ) -> Dict[str, Any]:
        """
        Check text for all geographic entities (city, state, country).

        Returns a dict with all found entities and their metadata.

        Args:
            text: Text to search
            countries: List of country codes to search, or None for all supported

        Returns:
            Dict with keys: 'country', 'state', 'city', 'inferred_country'
            Each value is a tuple of (found, entity_name) or None
        """
        result = {
            'country': self.text_contains_country(text, countries),
            'state': self.text_contains_state(text, countries),
            'city': self.text_contains_city(text, countries),
        }

        # Infer country from state if found
        if result['state'][0]:
            state_name = result['state'][1]
            inferred = self.infer_country_from_state(state_name)
            result['inferred_country'] = (True, inferred) if inferred else (False, None)
        else:
            result['inferred_country'] = (False, None)

        return result

    # ============================================================
    # BANK VALIDATION
    # ============================================================

    def validate_bank_in_swift_codes(
        self,
        bank_name: str,
        country: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate that bank exists in SWIFT codes database.

        Uses the new BankLookup with comprehensive bank configuration including
        alternate names and domain mappings.

        Args:
            bank_name: Bank name to look up
            country: ISO country code

        Returns:
            Tuple of (found, swift_code_or_error)
        """
        if not bank_name:
            return False, "Bank name is empty"

        try:
            from app.core.key_injection.bank_lookup import lookup_bank_by_name

            # Look up bank using new BankLookup (handles alternate names, abbreviations)
            bank_info = lookup_bank_by_name(bank_name, country)
            if bank_info and bank_info.swift_codes:
                return True, bank_info.swift_codes[0]

            # If not found with country, try without country (for default)
            if country:
                bank_info = lookup_bank_by_name(bank_name)
                if bank_info and bank_info.swift_codes:
                    # Bank exists but not for this specific country
                    return False, f"Bank '{bank_name}' found ({bank_info.country}) but not for {country}"

            return False, f"Bank '{bank_name}' not recognized"

        except ImportError:
            logger.warning("bank_lookup module not available")
            return True, None  # Don't fail if module unavailable


# Singleton instance for convenience
_validator_instance = None


def get_bank_statement_validator() -> BankStatementValidator:
    """Get the singleton BankStatementValidator instance."""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = BankStatementValidator()
    return _validator_instance


def get_geographic_detector() -> BankStatementValidator:
    """Get the singleton BankStatementValidator for geographic detection."""
    return get_bank_statement_validator()

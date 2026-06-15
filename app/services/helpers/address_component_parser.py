"""
Address Component Parser

Parses address components (street, city, state, postal_code, country) from
extracted address text. Supports country-specific parsing patterns.

Supported countries:
- IN (India): 6-digit postal codes, state detection
- US (United States): 5-digit ZIP codes, state abbreviations
- GB (United Kingdom): UK postal patterns
- SG (Singapore): 6-digit postal codes
- AE (UAE): No postal codes
- TH (Thailand): 5-digit postal codes, province detection
"""

import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from app.core.logger import get_logger


logger = get_logger()


# Country name to ISO code mapping (for common country names extracted by LLMs)
COUNTRY_NAME_TO_ISO = {
    "INDIA": "IN",
    "INDIAN": "IN",
    "SINGAPORE": "SG",
    "UNITED STATES": "US",
    "USA": "US",
    "UNITED STATES OF AMERICA": "US",
    "UNITED KINGDOM": "GB",
    "UK": "GB",
    "GREAT BRITAIN": "GB",
    "UAE": "AE",
    "UNITED ARAB EMIRATES": "AE",
    "THAILAND": "TH",
    "THAI": "TH",
}


@dataclass
class AddressComponents:
    """Parsed address components."""
    street_address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    postal_code: Optional[str] = None
    country: Optional[str] = None


# Indian states mapping (common variations)
INDIA_STATES = {
    "ANDHRA PRADESH": "AP",
    "ARUNACHAL PRADESH": "AR",
    "ASSAM": "AS",
    "BIHAR": "BR",
    "CHHATTISGARH": "CG",
    "GOA": "GA",
    "GUJARAT": "GJ",
    "HARYANA": "HR",
    "HIMACHAL PRADESH": "HP",
    "JHARKHAND": "JH",
    "KARNATAKA": "KA",
    "KERALA": "KL",
    "MADHYA PRADESH": "MP",
    "MAHARASHTRA": "MH",
    "MANIPUR": "MN",
    "MEGHALAYA": "ML",
    "MIZORAM": "MZ",
    "NAGALAND": "NL",
    "ODISHA": "OD",
    "PUNJAB": "PB",
    "RAJASTHAN": "RJ",
    "SIKKIM": "SK",
    "TAMIL NADU": "TN",
    "TELANGANA": "TG",
    "TRIPURA": "TR",
    "UTTARAKHAND": "UK",
    "UTTAR PRADESH": "UP",
    "WEST BENGAL": "WB",
    "DELHI": "DL",
    "CHANDIGARH": "CH",
    "PUDUCHERRY": "PY",
    "JAMMU AND KASHMIR": "JK",
    "LADAKH": "LA",
    "ANDAMAN AND NICOBAR": "AN",
    "LAKSHADWEEP": "LD",
}

# Indian state name normalizations (handles missing spaces, etc.)
INDIA_STATE_NORMALIZATIONS = {
    "ANDHRAPRADESH": "ANDHRA PRADESH",
    "TAMILNADU": "TAMIL NADU",
    "UTTARPRADESH": "UTTAR PRADESH",
    "MADHYAPRADESH": "MADHYA PRADESH",
    "HIMACHALPRADESH": "HIMACHAL PRADESH",
    "ARUNACHALPRADESH": "ARUNACHAL PRADESH",
}


# US states mapping
US_STATES = {
    "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR",
    "CALIFORNIA": "CA", "COLORADO": "CO", "CONNECTICUT": "CT", "DELAWARE": "DE",
    "FLORIDA": "FL", "GEORGIA": "GA", "HAWAII": "HI", "IDAHO": "ID",
    "ILLINOIS": "IL", "INDIANA": "IN", "IOWA": "IA", "KANSAS": "KS",
    "KENTUCKY": "KY", "LOUISIANA": "LA", "MAINE": "ME", "MARYLAND": "MD",
    "MASSACHUSETTS": "MA", "MICHIGAN": "MI", "MINNESOTA": "MN", "MISSISSIPPI": "MS",
    "MISSOURI": "MO", "MONTANA": "MT", "NEBRASKA": "NE", "NEVADA": "NV",
    "NEW HAMPSHIRE": "NH", "NEW JERSEY": "NJ", "NEW MEXICO": "NM", "NEW YORK": "NY",
    "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND", "OHIO": "OH", "OKLAHOMA": "OK",
    "OREGON": "OR", "PENNSYLVANIA": "PA", "RHODE ISLAND": "RI", "SOUTH CAROLINA": "SC",
    "SOUTH DAKOTA": "SD", "TENNESSEE": "TN", "TEXAS": "TX", "UTAH": "UT",
    "VERMONT": "VT", "VIRGINIA": "VA", "WASHINGTON": "WA", "WEST VIRGINIA": "WV",
    "WISCONSIN": "WI", "WYOMING": "WY",
    "DISTRICT OF COLUMBIA": "DC",
}


class AddressComponentParser:
    """
    Parse address components from raw address text.

    Supports country-specific parsing patterns for:
    - India (IN): 6-digit PIN codes, state names
    - United States (US): ZIP codes, state abbreviations
    - United Kingdom (GB): Postcode patterns
    - Singapore (SG): 6-digit postal codes
    - United Arab Emirates (AE): No postal codes
    - Thailand (TH): 5-digit postal codes, province names
    """

    def __init__(self):
        """Initialize the address parser."""
        self.india_state_pattern = self._build_india_state_pattern()
        self.us_state_pattern = self._build_us_state_pattern()
        # Build reverse mapping for normalization
        self.india_state_normalizations = INDIA_STATE_NORMALIZATIONS

    def _normalize_state_in_text(self, text: str) -> str:
        """
        Normalize state names in text by adding missing spaces.

        For example, converts "ANDHRAPRADESH" to "ANDHRA PRADESH".

        Args:
            text: Text to normalize

        Returns:
            Text with normalized state names
        """
        text_upper = text.upper()

        # Apply each normalization (replace in reverse order of length to avoid partial matches)
        for unnormalized, normalized in sorted(
            self.india_state_normalizations.items(),
            key=lambda x: len(x[0]),
            reverse=True
        ):
            # Only replace if found as whole word
            pattern = rf'\b{re.escape(unnormalized)}\b'
            if re.search(pattern, text_upper):
                text = re.sub(pattern, normalized, text, flags=re.IGNORECASE)
                text_upper = text.upper()

        return text

    def parse_address(
        self,
        address: Optional[str],
        country_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Parse address into components based on country-specific rules.

        Args:
            address: Raw address text (may contain newlines)
            country_hint: Optional ISO country code for parsing rules

        Returns:
            Dictionary with parsed components using address_* field names:
            {
                "street_address": {"value": "...", "confidence": 0.9},
                "address_city": {"value": "...", "confidence": 0.9},
                "address_state": {"value": "...", "confidence": 0.9},
                "address_postal": {"value": "...", "confidence": 0.9},
                "address_country": {"value": "...", "confidence": 0.9}
            }
        """
        if not address or not address.strip():
            return self._empty_components()

        # Normalize address
        address = address.strip()
        # Replace newlines with spaces for parsing
        address_normalized = re.sub(r'[\n\r]+', ' ', address)

        # Parse based on country
        # Convert country name to ISO code if necessary
        country_upper = (country_hint or "").upper()
        # Normalize country names to ISO codes (e.g., "INDIA" -> "IN")
        country_iso = COUNTRY_NAME_TO_ISO.get(country_upper, country_upper)

        if country_iso == "IN":
            components = self._parse_indian_address(address_normalized)
        elif country_iso == "US":
            components = self._parse_us_address(address_normalized)
        elif country_iso == "GB":
            components = self._parse_uk_address(address_normalized)
        elif country_iso == "SG":
            components = self._parse_singapore_address(address_normalized)
        elif country_iso == "AE":
            components = self._parse_uae_address(address_normalized)
        elif country_iso == "TH":
            components = self._parse_thai_address(address_normalized)
        else:
            # Generic parsing (no country-specific rules)
            components = self._parse_generic_address(address_normalized)

        # Add country hint if available (use ISO code, not full country name)
        if country_hint and not components.get("address_country"):
            components["address_country"] = {
                "value": country_iso,  # Use ISO code (e.g., "IN" not "INDIA")
                "confidence": 0.95,
                "source": "country_hint"
            }

        return components

    def _parse_indian_address(self, address: str) -> Dict[str, Any]:
        """
        Parse Indian address with 6-digit PIN code and state detection.

        Indian address format:
        Street/Locality, City, State PIN
        Example: "123 Main St, Mumbai, Maharashtra 400001"

        Correct parsing order:
        1. Extract PIN code and remove it
        2. Normalize state names (fix missing spaces)
        3. Extract country and remove it
        4. Extract state FULL NAME (not code) and remove it
        5. What remains is the city

        Args:
            address: Normalized address text

        Returns:
            Parsed address components with address_* field names
        """
        components = self._empty_components()

        # Start with original address, progressively remove components
        address_part = address.upper()

        # Step 1: Extract 6-digit PIN code (usually at end) and remove it
        pin_match = re.search(r'\b(\d{6})\b', address_part)
        if pin_match:
            pin_code = pin_match.group(1)
            # Validate PIN code (starts with 1-9 for valid Indian PINs)
            if pin_code[0] in '123456789':
                components["address_postal"] = {
                    "value": pin_code,
                    "confidence": 0.95,
                    "source": "parsed"
                }
                # Remove PIN from address_part
                address_part = address_part.replace(pin_code, "")

        # Step 2: Normalize state names (fix "ANDHRAPRADESH" -> "ANDHRA PRADESH", etc.)
        address_part = self._normalize_state_in_text(address_part)

        # Step 4: Extract country and remove it
        # India can appear as "INDIA" or "IN" (as a whole word, not part of another word)
        country_indicators = ["INDIA", "IN"]
        country_found = False
        for indicator in country_indicators:
            # Use word boundary regex to avoid matching "IN" inside words like "SHIVAJINAGAR"
            pattern = rf'\b{indicator}\b'
            if re.search(pattern, address_part):
                components["address_country"] = {
                    "value": "IN",
                    "confidence": 0.95,
                    "source": "parsed"
                }
                # Remove country indicator from address_part using word boundary
                address_part = re.sub(pattern, '', address_part)
                country_found = True
                break

        # Step 5: Extract state FULL NAME (not code) and remove it
        # Sort by length (longest first) to match multi-word states first
        sorted_states = sorted(INDIA_STATES.keys(), key=len, reverse=True)
        state_found = None
        for state_name in sorted_states:
            # Use word boundary regex for consistent matching
            pattern = rf'\b{state_name}\b'
            if re.search(pattern, address_part):
                # Store the FULL state name, not the code
                components["address_state"] = {
                    "value": state_name.title(),
                    "confidence": 0.90,
                    "source": "parsed"
                }
                # Remove just THIS state name from address_part using word boundary
                address_part = re.sub(pattern, '', address_part)
                state_found = state_name
                break

        # Step 6: What remains is the city (last meaningful word after cleaning)
        # Clean up the remaining address part
        address_part = address_part.strip(", \t\n-")

        if address_part:
            # Split on commas FIRST to handle comma-separated values properly
            comma_parts = [p.strip() for p in address_part.split(',') if p.strip()]
            if comma_parts:
                # Get the last comma-separated part
                last_part = comma_parts[-1]
                # Split on whitespace to get individual words
                words = [w for w in last_part.split() if w and len(w) > 1]
                if words:
                    # Get the last word as potential city
                    potential_city = words[-1].strip(".,/-:;")
                    if len(potential_city) >= 3:  # Minimum city name length
                        components["address_city"] = {
                            "value": potential_city.title(),
                            "confidence": 0.75,
                            "source": "parsed"
                        }

        # FALLBACK: If state/country extraction failed but city was found,
        # try reverse lookup of state/country from city
        if not state_found and components.get("address_city") and not components.get("address_country"):
            city_name = components["address_city"]["value"]
            try:
                from countrystatecity_countries import get_cities_of_country, get_states_of_country

                # Try to find the city in Indian cities
                indian_cities = get_cities_of_country("IN")
                indian_states = get_states_of_country("IN")
                state_map = {s.id: s.name for s in indian_states}

                for city in indian_cities:
                    if city.name.upper() == city_name.upper():
                        # Found the city, get its state
                        if hasattr(city, 'state_id') and city.state_id in state_map:
                            state_name = state_map[city.state_id]
                            components["address_state"] = {
                                "value": state_name.title(),
                                "confidence": 0.70,
                                "source": "reverse_lookup"
                            }
                            components["address_country"] = {
                                "value": "IN",
                                "confidence": 0.95,
                                "source": "reverse_lookup"
                            }
                            logger.info(f"Reverse lookup from city '{city_name}': state='{state_name}', country='IN'")
                        break
            except Exception as e:
                logger.debug(f"Failed reverse lookup from city '{city_name}': {e}")

        # Everything before city is street address
        street_part = address
        if components.get("address_city"):
            city_value = components["address_city"]["value"]
            street_part = street_part.replace(city_value, "")
            street_part = street_part.replace(city_value.upper(), "")
            street_part = street_part.replace(city_value.lower(), "")
        if state_found:
            street_part = street_part.replace(state_found, "")
            street_part = street_part.replace(state_found.title(), "")
        if pin_match:
            street_part = street_part.replace(pin_match.group(1), "")
        if country_found:
            for indicator in country_indicators:
                street_part = street_part.replace(indicator, "")
                street_part = street_part.replace(indicator.lower(), "")

        street_part = street_part.strip(", \t\n-")
        if street_part and len(street_part) >= 5:
            components["street_address"] = {
                "value": street_part,
                "confidence": 0.80,
                "source": "parsed"
            }

        return components

    def _parse_us_address(self, address: str) -> Dict[str, Any]:
        """
        Parse US address with ZIP code and state abbreviation.

        US address format:
        Street, City, State ZIP
        Example: "123 Main St, New York, NY 10001"

        Correct parsing order:
        1. Extract ZIP code and remove it
        2. Extract country and remove it
        3. Extract state abbreviation (US uses 2-letter codes) and remove it
        4. What remains is the city

        Args:
            address: Normalized address text

        Returns:
            Parsed address components with address_* field names
        """
        components = self._empty_components()

        # Start with original address, progressively remove components
        address_part = address.upper()

        # Step 1: Extract 5-digit ZIP code (or ZIP+4) and remove it
        zip_match = re.search(r'\b(\d{5})(?:-\d{4})?\b', address_part)
        if zip_match:
            components["address_postal"] = {
                "value": zip_match.group(1),
                "confidence": 0.95,
                "source": "parsed"
            }
            # Remove ZIP from address_part
            address_part = address_part.replace(zip_match.group(1), "")
            address_part = re.sub(r'-\d{4}', '', address_part)

        # Step 2: Extract country and remove it
        # USA can appear as "USA", "US", "UNITED STATES", "UNITED STATES OF AMERICA" (as whole words)
        country_indicators = ["UNITED STATES OF AMERICA", "UNITED STATES", "USA", "US"]
        country_found = False
        for indicator in country_indicators:
            # Use word boundary regex to avoid matching substrings
            pattern = rf'\b{indicator}\b'
            if re.search(pattern, address_part):
                components["address_country"] = {
                    "value": "US",
                    "confidence": 0.95,
                    "source": "parsed"
                }
                # Remove country indicator from address_part using word boundary
                address_part = re.sub(pattern, '', address_part)
                country_found = True
                break

        # Step 3: Extract state abbreviation and remove it
        # US states use 2-letter codes
        state_found = None
        for state_code in US_STATES.values():
            if f" {state_code} " in f" {address_part} " or address_part.endswith(f" {state_code}") or address_part.startswith(f"{state_code} "):
                components["address_state"] = {
                    "value": state_code,
                    "confidence": 0.90,
                    "source": "parsed"
                }
                # Remove state code from address_part
                address_part = address_part.replace(state_code, "")
                state_found = state_code
                break

        # Step 4: What remains is the city (last meaningful word after cleaning)
        address_part = address_part.strip(", \t\n-")

        if address_part:
            # Filter out empty strings and punctuation-only tokens
            words = [w for w in address_part.split() if w and len(w) > 1 and not w.strip('.,/-:;') == '']
            if words:
                # Get last meaningful word as potential city
                potential_city = words[-1].strip(", ")
                if len(potential_city) >= 3:
                    components["address_city"] = {
                        "value": potential_city.title(),
                        "confidence": 0.75,
                        "source": "parsed"
                    }

        # Street address is everything before city
        street_part = address
        if components.get("address_city"):
            city_value = components["address_city"]["value"]
            street_part = street_part.replace(city_value, "")
            street_part = street_part.replace(city_value.upper(), "")
        if state_found:
            street_part = street_part.replace(state_found, "")
        if zip_match:
            street_part = street_part.replace(zip_match.group(1), "")
        if country_found:
            for indicator in country_indicators:
                street_part = street_part.replace(indicator, "")

        street_part = street_part.strip(", \t\n-")
        if street_part and len(street_part) >= 5:
            components["street_address"] = {
                "value": street_part,
                "confidence": 0.80,
                "source": "parsed"
            }

        return components

    def _parse_uk_address(self, address: str) -> Dict[str, Any]:
        """
        Parse UK address with postcode.

        UK postcode format: AA9A 9AA or similar
        Example: "123 High St, London SW1A 1AA"

        Correct parsing order:
        1. Extract postcode and remove it
        2. Extract country and remove it
        3. What remains is the city

        Args:
            address: Normalized address text

        Returns:
            Parsed address components with address_* field names
        """
        components = self._empty_components()

        # Start with original address, progressively remove components
        address_part = address.upper()

        # Step 1: Extract UK postcode and remove it
        # Pattern: outward code (area+district) + inward code (sector+unit)
        postcode_pattern = r'\b([A-Z]{1,2}\d[A-Z\d]?\s\d[A-Z]{2})\b'
        postcode_match = re.search(postcode_pattern, address_part)
        if postcode_match:
            components["address_postal"] = {
                "value": postcode_match.group(1),
                "confidence": 0.95,
                "source": "parsed"
            }
            # Remove postcode from address_part
            address_part = address_part.replace(postcode_match.group(1), "")

        # Step 2: Extract country and remove it
        # UK can appear as "UK", "GB", "UNITED KINGDOM", "GREAT BRITAIN" (as whole words)
        country_indicators = ["UNITED KINGDOM", "GREAT BRITAIN", "UK", "GB"]
        country_found = False
        for indicator in country_indicators:
            # Use word boundary regex to avoid matching substrings
            pattern = rf'\b{indicator}\b'
            if re.search(pattern, address_part):
                components["address_country"] = {
                    "value": "GB",
                    "confidence": 0.95,
                    "source": "parsed"
                }
                # Remove country indicator from address_part using word boundary
                address_part = re.sub(pattern, '', address_part)
                country_found = True
                break

        # Step 3: What remains is the city (last meaningful word after cleaning)
        address_part = address_part.strip(", \t\n-")

        if address_part:
            # Filter out empty strings and punctuation-only tokens
            words = [w for w in address_part.split() if w and len(w) > 1 and not w.strip('.,/-:;') == '']
            if words:
                # Get last meaningful word as potential city
                potential_city = words[-1].strip(", ")
                if len(potential_city) >= 3:
                    components["address_city"] = {
                        "value": potential_city.title(),
                        "confidence": 0.75,
                        "source": "parsed"
                    }

        # Street address
        street_part = address
        if components.get("address_city"):
            city_value = components["address_city"]["value"]
            street_part = street_part.replace(city_value, "")
            street_part = street_part.replace(city_value.upper(), "")
        if postcode_match:
            street_part = street_part.replace(postcode_match.group(1), "")
        if country_found:
            for indicator in country_indicators:
                street_part = street_part.replace(indicator, "")

        street_part = street_part.strip(", \t\n-")
        if street_part and len(street_part) >= 5:
            components["street_address"] = {
                "value": street_part,
                "confidence": 0.80,
                "source": "parsed"
            }

        return components

    def _parse_singapore_address(self, address: str) -> Dict[str, Any]:
        """
        Parse Singapore address with 6-digit postal code.

        Singapore address format:
        Block Street, Area Singapore ZIP
        Example: "123 Main St #12-34, Singapore 238896"

        Correct parsing order:
        1. Extract postal code and remove it
        2. Extract country and remove it
        3. What remains is the area/city

        Args:
            address: Normalized address text

        Returns:
            Parsed address components with address_* field names
        """
        components = self._empty_components()

        # Start with original address, progressively remove components
        address_part = address.upper()

        # Step 1: Extract 6-digit postal code and remove it
        postal_match = re.search(r'\b(\d{6})\b', address_part)
        if postal_match:
            components["address_postal"] = {
                "value": postal_match.group(1),
                "confidence": 0.95,
                "source": "parsed"
            }
            # Remove postal code from address_part
            address_part = address_part.replace(postal_match.group(1), "")

        # Step 2: Extract country and remove it
        # Singapore can appear as "SINGAPORE" or "SG" (as whole words)
        country_indicators = ["SINGAPORE", "SG"]
        country_found = False
        for indicator in country_indicators:
            # Use word boundary regex to avoid matching substrings
            pattern = rf'\b{indicator}\b'
            if re.search(pattern, address_part):
                components["address_country"] = {
                    "value": "SG",
                    "confidence": 0.95,
                    "source": "parsed"
                }
                # Remove country indicator from address_part using word boundary
                address_part = re.sub(pattern, '', address_part)
                country_found = True
                break

        # Step 3: What remains is the area/city (last meaningful word after cleaning)
        address_part = address_part.strip(", \t\n-")

        if address_part:
            # Filter out empty strings and punctuation-only tokens
            words = [w for w in address_part.split() if w and len(w) > 1 and not w.strip('.,/-:;') == '']
            if words:
                # Get last meaningful word as potential area/city
                potential_area = words[-1].strip(", ")
                if len(potential_area) >= 3:
                    components["address_city"] = {
                        "value": potential_area.title(),
                        "confidence": 0.75,
                        "source": "parsed"
                    }

        # Street address is everything before area
        street_part = address
        if components.get("address_city"):
            city_value = components["address_city"]["value"]
            street_part = street_part.replace(city_value, "")
            street_part = street_part.replace(city_value.upper(), "")
        if country_found:
            for indicator in country_indicators:
                street_part = street_part.replace(indicator, "")
        if postal_match:
            street_part = street_part.replace(postal_match.group(1), "")

        street_part = street_part.strip(", \t\n-")
        if street_part and len(street_part) >= 5:
            components["street_address"] = {
                "value": street_part,
                "confidence": 0.80,
                "source": "parsed"
            }

        return components

    def _parse_uae_address(self, address: str) -> Dict[str, Any]:
        """
        Parse UAE address (no postal codes).

        UAE address format:
        Street, Area, City, Country
        Example: "123 Sheikh Zayed Rd, Dubai, UAE"

        Correct parsing order:
        1. Extract country and remove it
        2. Extract city and remove it
        3. What remains is the street address

        Args:
            address: Normalized address text

        Returns:
            Parsed address components with address_* field names (no postal code)
        """
        components = self._empty_components()

        # Start with original address, progressively remove components
        address_part = address.upper()

        # Step 1: Extract country and remove it
        # UAE can appear as "UAE", "UNITED ARAB EMIRATES" (as whole words)
        country_indicators = ["UNITED ARAB EMIRATES", "UAE"]
        country_found = False
        for indicator in country_indicators:
            # Use word boundary regex to avoid matching substrings
            pattern = rf'\b{indicator}\b'
            if re.search(pattern, address_part):
                components["address_country"] = {
                    "value": "AE",
                    "confidence": 0.95,
                    "source": "parsed"
                }
                # Remove country indicator from address_part using word boundary
                address_part = re.sub(pattern, '', address_part)
                country_found = True
                break

        # Step 2: Extract city and remove it (UAE cities are distinctive)
        # Sort by length (longest first) to match multi-word cities first
        cities = ["RAS AL KHAIMAH", "ABU DHABI", "UMM AL QUWAIN", "AL AIN",
                  "DUBAI", "SHARJAH", "AJMAN", "FUJAIRAH"]
        city_found = None
        for city in cities:
            if city in address_part:
                components["address_city"] = {
                    "value": city.title(),
                    "confidence": 0.85,
                    "source": "parsed"
                }
                # Remove city from address_part
                address_part = address_part.replace(city, "")
                city_found = city
                break

        # Step 3: What remains is the street address
        street_part = address_part.strip(", \t\n-")

        if street_part and len(street_part) >= 5:
            components["street_address"] = {
                "value": street_part,
                "confidence": 0.75,
                "source": "parsed"
            }

        return components

    def _parse_thai_address(self, address: str) -> Dict[str, Any]:
        """
        Parse Thai address with 5-digit postal code and province detection.

        Uses countrystatecity-countries library for dynamic province loading.
        Thai address format: District, Province Postal Code
        Example: "Lam Sai, Wang Noi, Phra Nakhon Si Ayutthaya 13170"

        Correct parsing order:
        1. Extract 5-digit postal code and remove it
        2. Load Thai states/provinces from countrystatecity-countries library
        3. Extract province (state) and remove it
        4. What remains is the city/district
        5. Everything before is street address

        Args:
            address: Normalized address text

        Returns:
            Parsed address components with address_* field names
        """
        components = self._empty_components()

        # Start with original address, progressively remove components
        address_part = address.upper()
        state_found = None

        # Step 1: Extract 5-digit postal code (usually at end) and remove it
        postal_match = re.search(r'\b(\d{5})\b', address_part)
        if postal_match:
            components["address_postal"] = {
                "value": postal_match.group(1),
                "confidence": 0.95,
                "source": "parsed"
            }
            # Remove postal code from address_part
            address_part = address_part.replace(postal_match.group(1), "")

        # Step 2: Extract country and remove it
        # Thailand can appear as "THAILAND", "THAI", or "TH" (as whole words)
        country_indicators = ["THAILAND", "THAI", "TH"]
        country_found = False
        for indicator in country_indicators:
            # Use word boundary regex to avoid matching substrings
            pattern = rf'\b{indicator}\b'
            if re.search(pattern, address_part):
                components["address_country"] = {
                    "value": "TH",
                    "confidence": 0.95,
                    "source": "parsed"
                }
                # Remove country indicator from address_part using word boundary
                address_part = re.sub(pattern, '', address_part)
                country_found = True
                break

        # Step 3: Load Thai states/provinces from countrystatecity-countries library
        # and extract province (state) from address
        try:
            from countrystatecity_countries import get_states_of_country

            thai_states = get_states_of_country("TH")
            # Sort by length (longest first) to match multi-word provinces first
            sorted_states = sorted(thai_states, key=lambda s: len(s.name), reverse=True)

            address_upper = address_part
            for state in sorted_states:
                state_name_upper = state.name.upper()
                # Use word boundary regex for consistent matching
                pattern = rf'\b{re.escape(state_name_upper)}\b'
                if re.search(pattern, address_upper):
                    # Found province/state
                    components["address_state"] = {
                        "value": state.name.title(),
                        "confidence": 0.90,
                        "source": "parsed"
                    }
                    # Remove just THIS state name from address_part using word boundary
                    address_part = re.sub(pattern, '', address_part)
                    state_found = state.name
                    break
        except Exception as e:
            logger.debug(f"Failed to load Thai states from library: {e}")

        # Step 4: What remains is the city/district (last meaningful word after cleaning)
        address_part = address_part.strip(", \t\n-")

        if address_part:
            # Split on commas FIRST to handle comma-separated values properly
            comma_parts = [p.strip() for p in address_part.split(',') if p.strip()]
            if comma_parts:
                # Get the last comma-separated part
                last_part = comma_parts[-1]
                # Split on whitespace to get individual words
                words = [w for w in last_part.split() if w and len(w) > 1]
                if words:
                    # Get the last word as potential city
                    potential_city = words[-1].strip(".,/-:;")
                    if len(potential_city) >= 3:  # Minimum city name length
                        components["address_city"] = {
                            "value": potential_city.title(),
                            "confidence": 0.75,
                            "source": "parsed"
                        }

        # Step 5: Everything before city is street address
        street_part = address
        if components.get("address_city"):
            city_value = components["address_city"]["value"]
            street_part = street_part.replace(city_value, "")
            street_part = street_part.replace(city_value.upper(), "")
            street_part = street_part.replace(city_value.lower(), "")
        if state_found:
            street_part = street_part.replace(state_found, "")
            street_part = street_part.replace(state_found.title(), "")
            street_part = street_part.replace(state_found.upper(), "")
        if postal_match:
            street_part = street_part.replace(postal_match.group(1), "")
        if country_found:
            for indicator in country_indicators:
                street_part = street_part.replace(indicator, "")
                street_part = street_part.replace(indicator.lower(), "")

        street_part = street_part.strip(", \t\n-")
        if street_part and len(street_part) >= 5:
            components["street_address"] = {
                "value": street_part,
                "confidence": 0.80,
                "source": "parsed"
            }

        return components

    def _parse_generic_address(self, address: str) -> Dict[str, Any]:
        """
        Parse generic address without country-specific rules.

        Attempts basic extraction of street, city, and postal code.

        Correct parsing order:
        1. Extract postal code and remove it
        2. What remains is the city (last meaningful word)

        Args:
            address: Normalized address text

        Returns:
            Parsed address components with address_* field names and lower confidence
        """
        components = self._empty_components()

        # Start with original address, progressively remove components
        address_part = address.upper()

        # Step 1: Try to extract postal code (various formats) and remove it
        postal_patterns = [
            r'\b([A-Z]{1,2}\d[A-Z\d]?\s\d[A-Z]{2})\b',  # UK format
            r'\b(\d{6})\b',  # 6-digit
            r'\b(\d{5})\b',  # 5-digit
        ]

        postal_found = None
        for pattern in postal_patterns:
            match = re.search(pattern, address_part)
            if match:
                components["address_postal"] = {
                    "value": match.group(1),
                    "confidence": 0.70,
                    "source": "parsed"
                }
                # Remove postal code from address_part
                address_part = address_part.replace(match.group(1), "")
                postal_found = match.group(1)
                break

        # Step 2: What remains is the city (last meaningful word after cleaning)
        address_part = address_part.strip(", \t\n-")

        if address_part:
            # Filter out empty strings and punctuation-only tokens
            words = [w for w in address_part.split() if w and len(w) > 1 and not w.strip('.,/-:;') == '']
            if words:
                # Get last meaningful word as potential city
                potential_city = words[-1].strip(", ")
                if len(potential_city) >= 3:
                    components["address_city"] = {
                        "value": potential_city.title(),
                        "confidence": 0.60,
                        "source": "parsed"
                    }

        # Street address
        street_part = address
        if components.get("address_city"):
            city_value = components["address_city"]["value"]
            street_part = street_part.replace(city_value, "")
            street_part = street_part.replace(city_value.upper(), "")
        if postal_found:
            street_part = street_part.replace(postal_found, "")

        street_part = street_part.strip(", \t\n-")
        if street_part and len(street_part) >= 5:
            components["street_address"] = {
                "value": street_part,
                "confidence": 0.60,
                "source": "parsed"
            }

        return components

    def _empty_components(self) -> Dict[str, Any]:
        """Return empty components dictionary with address_* field names."""
        return {
            "street_address": None,
            "address_city": None,
            "address_state": None,
            "address_postal": None,
            "address_country": None
        }

    def _build_india_state_pattern(self) -> str:
        """Build regex pattern for Indian states."""
        # Pattern for state codes or full names
        state_codes = "|".join(INDIA_STATES.values())
        state_names = "|".join(INDIA_STATES.keys())
        return rf'\b({state_codes}|{state_names})\b'

    def _build_us_state_pattern(self) -> str:
        """Build regex pattern for US states."""
        state_codes = "|".join(US_STATES.values())
        return rf'\b({state_codes})\b'


# Singleton instance
_instance = None


def get_address_component_parser() -> AddressComponentParser:
    """Get the singleton address parser instance."""
    global _instance
    if _instance is None:
        _instance = AddressComponentParser()
    return _instance

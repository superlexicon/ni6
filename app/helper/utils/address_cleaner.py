"""Address cleaning utility for bank statement extraction.

Provides shared address cleaning logic used by both spatial and GLiNER extractors.
"""

import re
from typing import Dict, Optional


def _get_state_normalizations(country_code: str) -> Dict[str, str]:
    """Get state name normalizations for a country.

    Handles common variations like missing spaces (e.g., "ANDHRAPRADESH" -> "ANDHRA PRADESH").

    Args:
        country_code: ISO country code (e.g., "IN" for India)

    Returns:
        Dictionary mapping incorrect state names to correct state names
    """
    normalizations = {}

    if country_code == "IN":
        # Indian state name normalizations
        normalizations = {
            "ANDHRAPRADESH": "ANDHRA PRADESH",
            "TAMILNADU": "TAMIL NADU",
            "UTTARPRADESH": "UTTAR PRADESH",
            "MADHYAPRADESH": "MADHYA PRADESH",
            "HIMACHALPRADESH": "HIMACHAL PRADESH",
            "ARUNACHALPRADESH": "ARUNACHAL PRADESH",
        }

    return normalizations


def _get_state_abbreviations(country_code: str) -> Dict[str, str]:
    """Get state abbreviation to full name mapping for a country.

    Args:
        country_code: ISO country code (e.g., "IN" for India, "AE" for UAE)

    Returns:
        Dictionary mapping state abbreviations to full state names
    """
    abbreviations = {}

    if country_code == "IN":
        # Indian state abbreviations
        abbreviations = {
            "AP": "ANDHRA PRADESH",
            "TN": "TAMIL NADU",
            "KA": "KARNATAKA",
            "MH": "MAHARASHTRA",
            "DL": "DELHI",
            "TS": "TELANGANA",
            "KL": "KERALA",
        }
    elif country_code == "AE":
        # UAE emirates
        abbreviations = {
            "DU": "DUBAI",
            "AB": "ABU DHABI",
            "SH": "SHARJAH",
            "AJ": "AJMAN",
            "UM": "UMM AL QUWAIN",
            "RAK": "RAS AL KHAIMAH",
            "FUJ": "FUJAIRAH",
        }

    return abbreviations


def _clean_street_address(
    street_address: str,
    city: str,
    state: Optional[str] = None,
    postal_code: Optional[str] = None,
    country_code: Optional[str] = None
) -> str:
    """
    Clean street address by removing location components.

    Removes city, state, country, and postal code from the street address string.
    Used for multi-span addresses where components may be in separate spans.

    Args:
        street_address: The joined street address string
        city: City name to remove
        state: State name to remove
        postal_code: Postal code to remove
        country_code: ISO country code for country name removal

    Returns:
        Cleaned street address
    """
    if not street_address:
        return ""

    text = street_address.strip()
    text_upper = text.upper()

    # Remove postal code
    if postal_code:
        text = re.sub(r'\b' + re.escape(postal_code) + r'\b', '', text, flags=re.IGNORECASE).strip()
        text_upper = text.upper()

    # Remove country name
    if country_code:
        from app.config.bank_statement_country_loader import get_country_config_loader
        config_loader = get_country_config_loader()
        config = config_loader.get_country_config(country_code)
        if config:
            country_name = config.get("country_name", "").upper()
            name_aliases = [alias.upper() for alias in config.get("name_aliases", [])]
            for alias in [country_name] + name_aliases:
                if alias in text_upper:
                    text = re.sub(r'\b' + re.escape(alias) + r'\b', '', text, flags=re.IGNORECASE).strip()
                    text_upper = text.upper()
                    break

    # Remove state
    if state:
        state_upper = state.upper()
        text = re.sub(r'\b' + re.escape(state_upper) + r'\b', '', text, flags=re.IGNORECASE).strip()

    # Remove city - remove ALL occurrences (not just the last)
    if city:
        city_upper = city.upper()
        # Remove all occurrences using regex
        text = re.sub(r'\b' + re.escape(city_upper) + r'\b', '', text, flags=re.IGNORECASE).strip()

    # Also remove state abbreviations (for India, UAE, etc.)
    if state and country_code:
        state_upper = state.upper()
        abbreviations = _get_state_abbreviations(country_code)
        for abbrev, full_name in abbreviations.items():
            if state_upper == full_name.upper() or state_upper == abbrev.upper():
                # Remove the abbreviation from text
                text = re.sub(r'\b' + re.escape(abbrev.upper()) + r'\b', '', text, flags=re.IGNORECASE).strip()
                break

    # Clean up: remove extra commas and whitespace
    text = re.sub(r',\s*,', ',', text)
    text = re.sub(r'^,\s*', '', text)
    text = re.sub(r',\s*$', '', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def _remove_duplicate_words(text: str, city: str = None) -> str:
    """
    Remove duplicate words and location names from address.

    Handles cases like:
    - "1-21 SETTYGIRIPALLE, SETTYGIRIPALLE" -> "1-21 SETTYGIRIPALLE"
    - "CHITTOOR, Chittoor" -> "" (both are city name)

    Preserves house number prefixes like "1-21" while removing standalone duplicates.

    Args:
        text: Address text to clean
        city: City name to identify and remove duplicates

    Returns:
        Text with duplicates removed
    """
    if not text:
        return ""

    # Split by comma and clean each part
    parts = [p.strip() for p in text.split(',')]
    if not parts:
        return ""

    seen_words = set()
    seen_parts = []
    unique_parts = []

    for part in parts:
        if not part:
            continue

        part_upper = part.upper()

        # Check if this is a city name variant
        is_city_variant = False
        if city:
            city_upper = city.upper()
            # Direct match or case variation
            if part_upper == city_upper or part_upper in city_upper or city_upper in part_upper:
                is_city_variant = True

        if is_city_variant:
            # Skip city name variants
            continue

        # Check for duplicate location names
        # Split part into words to check for standalone duplicates
        words = part_upper.split()
        is_duplicate = False

        # If part contains house number pattern, keep it
        has_house_number = bool(re.match(r'^\d+[-\s]', part))

        if not has_house_number and len(words) == 1:
            # Single word - check if already seen
            if words[0] in seen_words:
                is_duplicate = True
            else:
                seen_words.add(words[0])
        elif not has_house_number:
            # Multi-word part - check if all words were seen before
            all_words_seen = all(w in seen_words for w in words)
            if all_words_seen:
                is_duplicate = True
            else:
                for w in words:
                    seen_words.add(w)

        # Also check for exact matches in seen_parts
        for seen in seen_parts:
            if seen == part_upper:
                is_duplicate = True
                break
            # Check if one is contained in the other (partial duplicates)
            if len(seen) > 3 and seen in part_upper:
                is_duplicate = True
                break
            if len(part_upper) > 3 and part_upper in seen:
                is_duplicate = True
                break

        if not is_duplicate:
            seen_parts.append(part_upper)
            unique_parts.append(part)

    result = ', '.join(unique_parts).strip()
    # Clean up any trailing/leading commas
    result = re.sub(r',\s*,', ',', result)
    result = re.sub(r'^,\s*', '', result)
    result = re.sub(r',\s*$', '', result)

    return result


def _normalize_state_name(state: Optional[str], country_code: Optional[str] = None) -> Optional[str]:
    """
    Normalize state name by applying country-specific corrections.

    Handles common issues like missing spaces (e.g., "ANDHRAPRADESH" -> "ANDHRA PRADESH").

    Args:
        state: Raw state name to normalize
        country_code: ISO country code for country-specific normalizations

    Returns:
        Normalized state name or original if no normalization needed
    """
    if not state or not country_code:
        return state

    state_upper = state.upper().strip()
    normalizations = _get_state_normalizations(country_code)

    # Return normalized version if found, otherwise return original
    return normalizations.get(state_upper, state)


def clean_gliner_address(
    address: str,
    city: Optional[str] = None,
    state: Optional[str] = None,
    postal_code: Optional[str] = None,
    country_code: Optional[str] = None
) -> str:
    """
    Clean a GLiNER-extracted address using spatial extractor's cleaning logic.

    Removes city names (all occurrences), state names/abbreviations, postal codes,
    country names, and duplicate location names.

    This is the main entry point for cleaning GLiNER-extracted addresses.
    It combines both cleaning steps:
    1. Normalize state name (fix missing spaces, etc.)
    2. Remove city/state/postal/country components
    3. Remove duplicate words and location names

    Args:
        address: Raw GLiNER-extracted address
        city: Extracted city name (for removal)
        state: Extracted state name (for removal)
        postal_code: Extracted postal code (for removal)
        country_code: ISO country code (for state abbreviations and country name removal)

    Returns:
        Cleaned address with location components and duplicates removed

    Example:
        >>> clean_gliner_address(
        ...     "1-21 SETTYGIRIPALLE,NEAR TEMPLE,SETTYGIRIPALLE,CHITTOOR,Chittoor,517419",
        ...     city="Chittoor",
        ...     state=None,
        ...     postal_code="517419",
        ...     country_code="IN"
        ... )
        "1-21 SETTYGIRIPALLE, NEAR TEMPLE"
    """
    if not address:
        return ""

    # Step 1: Normalize state name (fix "ANDHRAPRADESH" -> "ANDHRA PRADESH", etc.)
    normalized_state = _normalize_state_name(state, country_code)

    # Step 2: Remove city, state, postal code, country name
    cleaned = _clean_street_address(
        address,
        city=city or "",
        state=normalized_state,
        postal_code=postal_code,
        country_code=country_code
    )

    # Step 3: Remove duplicate words and location names
    cleaned = _remove_duplicate_words(cleaned, city=city)

    return cleaned.strip()

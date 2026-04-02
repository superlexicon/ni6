"""
SWIFT Code Lookup Utility

Provides functions to look up SWIFT codes for banks based on bank name.
Uses the hardcoded bank_swift_codes.json mapping.
"""

import json
import os
from typing import Optional
from pathlib import Path

# Cache for loaded SWIFT codes
_swift_codes_cache = None


def _load_swift_codes() -> dict:
    """Load SWIFT codes from JSON file."""
    global _swift_codes_cache
    if _swift_codes_cache is not None:
        return _swift_codes_cache

    # Path to SWIFT codes file
    swift_codes_path = Path(__file__).parent / "bank_swift_codes.json"

    with open(swift_codes_path, 'r') as f:
        data = json.load(f)
        _swift_codes_cache = data
        return data


def get_swift_code_for_bank(bank_name: Optional[str]) -> Optional[str]:
    """
    Get SWIFT code for a given bank name.

    Args:
        bank_name: Name of the bank (case-insensitive, partial match supported)

    Returns:
        SWIFT code if found, None otherwise
    """
    if not bank_name:
        return None

    bank_name_lower = bank_name.lower().strip()

    # Load SWIFT codes
    data = _load_swift_codes()

    # First try exact match by name
    for bank_entry in data.get("banks", []):
        entry_name = bank_entry.get("name", "").lower()
        if entry_name == bank_name_lower or entry_name in bank_name_lower or bank_name_lower in entry_name:
            return bank_entry.get("swift_code")

    # Second try match by key
    for bank_entry in data.get("banks", []):
        entry_key = bank_entry.get("key", "").lower()
        if entry_key in bank_name_lower:
            return bank_entry.get("swift_code")

    return None


def get_swift_code_for_bank_with_country(bank_name: Optional[str], country: Optional[str] = None) -> Optional[str]:
    """
    Get SWIFT code for a given bank name and country.

    When country is provided, it first tries to find an exact match with both
    bank name and country. This is more accurate for multinational banks like
    DBS, OCBC, UOB which have different SWIFT codes per country.

    Args:
        bank_name: Name of the bank (case-insensitive, partial match supported)
        country: ISO 3166-1 alpha-2 country code (e.g., "SG", "AE", "IN")

    Returns:
        SWIFT code if found, None otherwise
    """
    if not bank_name:
        return None

    bank_name_lower = bank_name.lower().strip()
    country_upper = country.upper() if country else None

    # Load SWIFT codes
    data = _load_swift_codes()

    # If country is provided, first try exact match with both name AND country
    if country_upper:
        for bank_entry in data.get("banks", []):
            entry_name = bank_entry.get("name", "").lower()
            entry_country = bank_entry.get("country", "").upper()

            # Exact country match required
            if entry_country != country_upper:
                continue

            # Check name match
            if entry_name == bank_name_lower or entry_name in bank_name_lower or bank_name_lower in entry_name:
                return bank_entry.get("swift_code")

        # Second try with country: match by key
        for bank_entry in data.get("banks", []):
            entry_key = bank_entry.get("key", "").lower()
            entry_country = bank_entry.get("country", "").upper()

            if entry_country != country_upper:
                continue

            if entry_key in bank_name_lower:
                return bank_entry.get("swift_code")

    # Fallback: Try without country (original behavior)
    return get_swift_code_for_bank(bank_name)


def get_country_for_bank(bank_name: Optional[str]) -> Optional[str]:
    """
    Get country code for a given bank name.

    Args:
        bank_name: Name of the bank (case-insensitive, partial match supported)

    Returns:
        Country code if found, None otherwise
    """
    if not bank_name:
        return None

    bank_name_lower = bank_name.lower().strip()

    # Load SWIFT codes
    data = _load_swift_codes()

    # First try exact match by name
    for bank_entry in data.get("banks", []):
        entry_name = bank_entry.get("name", "").lower()
        if entry_name == bank_name_lower or entry_name in bank_name_lower or bank_name_lower in entry_name:
            return bank_entry.get("country")

    # Second try match by key
    for bank_entry in data.get("banks", []):
        entry_key = bank_entry.get("key", "").lower()
        if entry_key in bank_name_lower:
            return bank_entry.get("country")

    return None


def get_country_for_bank_with_country(bank_name: Optional[str], country: Optional[str] = None) -> Optional[str]:
    """
    Get country code for a given bank name, with optional country hint for disambiguation.

    When country is provided, it validates that the bank entry matches the expected country.
    This is useful for confirming the bank's country when it's already known from document extraction.

    Args:
        bank_name: Name of the bank (case-insensitive, partial match supported)
        country: Expected ISO country code (optional, for validation)

    Returns:
        Country code if found, None otherwise
    """
    if not bank_name:
        return None

    bank_name_lower = bank_name.lower().strip()
    country_upper = country.upper() if country else None

    # Load SWIFT codes
    data = _load_swift_codes()

    # If country is provided, first try to match with that country
    if country_upper:
        for bank_entry in data.get("banks", []):
            entry_name = bank_entry.get("name", "").lower()
            entry_country = bank_entry.get("country", "").upper()

            if entry_country != country_upper:
                continue

            if entry_name == bank_name_lower or entry_name in bank_name_lower or bank_name_lower in entry_name:
                return entry_country

        # Second try with country: match by key
        for bank_entry in data.get("banks", []):
            entry_key = bank_entry.get("key", "").lower()
            entry_country = bank_entry.get("country", "").upper()

            if entry_country != country_upper:
                continue

            if entry_key in bank_name_lower:
                return entry_country

    # Fallback: Return the first match (default country for the bank)
    return get_country_for_bank(bank_name)

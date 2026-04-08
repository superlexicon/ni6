"""
Country-specific field labels for passport extraction.

This module loads and caches field labels from the passport config,
allowing extraction to use country-specific labels (including native languages)
instead of hardcoded global patterns.

Usage:
    labels = CountryFieldLabels(config_path)
    labels.get_labels('MM', 'full_name')  # Returns ['name', 'full name', 'အမည်', ...]
    labels.find_field_for_label('MM', 'အမည်')  # Returns 'full_name'
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any

from app.core import logger


class CountryFieldLabels:
    """
    Manage country-specific field label lookup for passport extraction.

    Loads labels from config.json and provides efficient lookup methods
    with fallback to global FIELD_PATTERNS when country labels not defined.
    """

    # Default field labels from FIELD_PATTERNS (fallback when country not defined)
    DEFAULT_LABELS: Dict[str, List[str]] = {
        "passport_number": ["passport no", "passport number", "passport#", "no passport", "passport no."],
        "id_number": ["id number", "identity card", "national id", "id no", "nric", "id card no"],
        "full_name": ["name", "full name", "given names", "surname", "surame", "last name", "first name"],
        "date_of_birth": ["date of birth", "dob", "birth date", "born", "birthday"],
        "date_of_expiry": ["date of expiry", "expiry date", "valid until", "expiration", "exp date", "valid thru"],
        "date_of_issue": ["date of issue", "issue date", "issued on", "iss date"],
        "sex": ["sex", "gender", "male", "female"],
        "place_of_birth": ["place of birth", "birth place", "birthplace", "pob", "born in"],
        "issuing_authority": ["issuing authority", "issuing office", "issued by", "authority"],
        "issuing_country": ["issuing country", "country", "nationality"],
        "nationality": ["nationality", "country code"],
        "address": ["address", "resident address", "current address", "home address", "permanent address"],
        "nrc_number": ["nrc", "nrc no", "national registration card"],
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the country field labels loader.

        Args:
            config_path: Path to the passport config.json file.
                        If None, uses default path.
        """
        self.logger = logger
        self.labels_by_country: Dict[str, Dict[str, List[str]]] = {}
        self.name_extraction_rules: Dict[str, Dict[str, Any]] = {}
        self._config_loaded = False

        # Determine config path
        if config_path is None:
            # Default path relative to this file
            config_path = str(
                Path(__file__).parent.parent.parent / "reference_templates" / "passports" / "config.json"
            )

        self._load_config(config_path)

    def _load_config(self, config_path: str) -> None:
        """
        Load field labels from the config file.

        Args:
            config_path: Path to config.json
        """
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                self.logger.warning(f"Passport config not found at {config_path}, using default labels only")
                return

            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # Extract field_labels from each country
            countries = config.get("countries", {})
            for country_code, country_data in countries.items():
                if "field_labels" in country_data:
                    self.labels_by_country[country_code.upper()] = country_data["field_labels"]
                    self.logger.debug(f"Loaded field labels for {country_code}")

                # Extract name extraction rules
                if "name_extraction" in country_data:
                    self.name_extraction_rules[country_code.upper()] = country_data["name_extraction"]
                    self.logger.debug(f"Loaded name extraction rules for {country_code}")

            self._config_loaded = True
            self.logger.info(f"Loaded field labels for {len(self.labels_by_country)} countries")

        except Exception as e:
            self.logger.error(f"Failed to load passport config: {type(e).__name__}")
            # Continue with default labels only

    def get_labels(self, country_code: str, field_name: str) -> List[str]:
        """
        Get labels for a field in a specific country.

        Falls back to global FIELD_PATTERNS if not defined for the country.

        Args:
            country_code: ISO 2-letter country code (e.g., "SG", "MM")
            field_name: Name of the field (e.g., "full_name", "date_of_birth")

        Returns:
            List of label strings to match against
        """
        country_code = country_code.upper() if country_code else ""

        # Try country-specific labels first
        if country_code in self.labels_by_country:
            country_labels = self.labels_by_country[country_code]
            if field_name in country_labels:
                return country_labels[field_name]

        # Fallback to default labels
        return self.DEFAULT_LABELS.get(field_name, [])

    def find_field_for_label(self, country_code: str, label_text: str) -> Optional[str]:
        """
        Reverse lookup: find which field a label refers to.

        Args:
            country_code: ISO 2-letter country code (e.g., "SG", "MM")
            label_text: The label text to match (e.g., "အမည်", "name")

        Returns:
            Field name if found, None otherwise
        """
        if not label_text:
            return None

        country_code = country_code.upper() if country_code else ""
        label_lower = label_text.lower().strip()

        # Build merged labels for this country (country-specific + defaults)
        merged_labels = self._get_merged_labels(country_code)

        # Search for matching field
        for field_name, labels in merged_labels.items():
            for label in labels:
                if label.lower() in label_lower or label_lower in label.lower():
                    return field_name

        return None

    def is_field_label(self, country_code: str, text: str, field_name: str) -> bool:
        """
        Check if text is a label for a specific field.

        Args:
            country_code: ISO 2-letter country code
            text: Text to check
            field_name: Field name to match against

        Returns:
            True if text matches a label for the field
        """
        if not text:
            return False

        labels = self.get_labels(country_code, field_name)
        text_lower = text.lower().strip()

        for label in labels:
            # Check for substring match (handles cases like "Name:" or "Full Name")
            if label.lower() in text_lower:
                return True

        return False

    def get_name_labels(self, country_code: str) -> List[str]:
        """
        Get all labels that indicate a name field.

        Args:
            country_code: ISO 2-letter country code

        Returns:
            List of name-related labels
        """
        return self.get_labels(country_code, "full_name")

    def get_date_labels(self, country_code: str, date_type: str = "any") -> List[str]:
        """
        Get labels for date fields.

        Args:
            country_code: ISO 2-letter country code
            date_type: "birth", "expiry", "issue", or "any"

        Returns:
            List of date-related labels
        """
        if date_type == "birth":
            return self.get_labels(country_code, "date_of_birth")
        elif date_type == "expiry":
            return self.get_labels(country_code, "date_of_expiry")
        elif date_type == "issue":
            return self.get_labels(country_code, "date_of_issue")
        else:
            # Return all date labels
            return (
                self.get_labels(country_code, "date_of_birth") +
                self.get_labels(country_code, "date_of_expiry") +
                self.get_labels(country_code, "date_of_issue")
            )

    def is_name_label(self, country_code: str, text: str) -> bool:
        """
        Check if text is a name-related label.

        Args:
            country_code: ISO 2-letter country code
            text: Text to check

        Returns:
            True if text matches a name label
        """
        return self.is_field_label(country_code, text, "full_name")

    def _get_merged_labels(self, country_code: str) -> Dict[str, List[str]]:
        """
        Get merged labels (country-specific + defaults) for a country.

        Args:
            country_code: ISO 2-letter country code

        Returns:
            Dict mapping field names to merged label lists
        """
        merged = {}

        # Start with defaults
        for field_name, labels in self.DEFAULT_LABELS.items():
            merged[field_name] = list(labels)

        # Overlay country-specific labels
        if country_code in self.labels_by_country:
            for field_name, labels in self.labels_by_country[country_code].items():
                if field_name in merged:
                    # Merge and deduplicate
                    merged[field_name] = list(set(merged[field_name] + labels))
                else:
                    merged[field_name] = list(labels)

        return merged

    def is_config_loaded(self) -> bool:
        """Check if the config file was successfully loaded."""
        return self._config_loaded

    def get_supported_countries(self) -> List[str]:
        """Get list of countries with custom field labels defined."""
        return list(self.labels_by_country.keys())

    # Name extraction rules methods

    def has_separate_surname_given_names(self, country_code: str) -> bool:
        """
        Check if a country uses separate surname and given names labels.

        Args:
            country_code: ISO 2-letter country code

        Returns:
            True if the country uses separate surname and given names labels
        """
        country_code = country_code.upper() if country_code else ""
        if country_code in self.name_extraction_rules:
            return self.name_extraction_rules[country_code].get("separate_surname_given_names", False)
        return False

    def get_surname_labels(self, country_code: str) -> List[str]:
        """
        Get surname labels for a country.

        Args:
            country_code: ISO 2-letter country code

        Returns:
            List of surname label strings
        """
        country_code = country_code.upper() if country_code else ""
        if country_code in self.name_extraction_rules:
            return self.name_extraction_rules[country_code].get("surname_labels", [])
        return []

    def get_given_names_labels(self, country_code: str) -> List[str]:
        """
        Get given names labels for a country.

        Args:
            country_code: ISO 2-letter country code

        Returns:
            List of given names label strings
        """
        country_code = country_code.upper() if country_code else ""
        if country_code in self.name_extraction_rules:
            return self.name_extraction_rules[country_code].get("given_names_labels", [])
        return []

    def require_both_name_parts(self, country_code: str) -> bool:
        """
        Check if a country requires both surname and given names to be present.

        Args:
            country_code: ISO 2-letter country code

        Returns:
            True if both parts are required
        """
        country_code = country_code.upper() if country_code else ""
        if country_code in self.name_extraction_rules:
            return self.name_extraction_rules[country_code].get("require_both_parts", False)
        return False

    def is_surname_label(self, country_code: str, text: str) -> bool:
        """
        Check if text is a surname label for a country.

        Args:
            country_code: ISO 2-letter country code
            text: Text to check

        Returns:
            True if text matches a surname label
        """
        if not text:
            return False

        surname_labels = self.get_surname_labels(country_code)
        text_lower = text.lower().strip()

        for label in surname_labels:
            if label.lower() == text_lower:
                return True

        return False

    def is_given_names_label(self, country_code: str, text: str) -> bool:
        """
        Check if text is a given names label for a country.

        Args:
            country_code: ISO 2-letter country code
            text: Text to check

        Returns:
            True if text matches a given names label
        """
        if not text:
            return False

        given_names_labels = self.get_given_names_labels(country_code)
        text_lower = text.lower().strip()

        for label in given_names_labels:
            if label.lower() == text_lower:
                return True

        return False


# Global singleton instance
_instance: Optional[CountryFieldLabels] = None


def get_country_field_labels() -> CountryFieldLabels:
    """
    Get the global CountryFieldLabels instance.

    Creates the instance on first call (lazy initialization).

    Returns:
        CountryFieldLabels singleton instance
    """
    global _instance
    if _instance is None:
        _instance = CountryFieldLabels()
    return _instance


def reset_instance() -> None:
    """
    Reset the global instance (useful for testing).
    """
    global _instance
    _instance = None

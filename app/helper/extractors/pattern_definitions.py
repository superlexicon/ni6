"""
Country and field pattern definitions for unified identity document extraction.

This module defines patterns for extracting fields from passports and ID cards
across multiple countries. Patterns are layout-independent and work based on
content recognition rather than spatial positioning.

Key Design:
- Each country has separate patterns for passport and id_card
- Fields are marked as required or optional per document type
- Missing optional fields (e.g., expiry on Singapore NRIC) don't affect confidence
"""

from typing import Dict, List, Optional, Any


# ============================================================================
# ISO COUNTRY CODE MAPPINGS
# ============================================================================

# 3-letter ISO to 2-letter ISO mapping
ISO3_TO_ISO2: Dict[str, str] = {
    "SGP": "SG",  # Singapore
    "THA": "TH",  # Thailand
    "IND": "IN",  # India
    "MYS": "MY",  # Malaysia
    "IDN": "ID",  # Indonesia
    "PHL": "PH",  # Philippines
    "VNM": "VN",  # Vietnam
    "CHN": "CN",  # China
    "MMR": "MM",  # Myanmar
    "ARE": "AE",  # United Arab Emirates
    # Add more as needed
}


# ============================================================================
# COUNTRY PATTERNS
# ============================================================================

COUNTRY_PATTERNS: Dict[str, Dict[str, Any]] = {
    "SG": {
        "name": "Singapore",
        "passport": {
            "number": r"^[EK]\d{7}[A-Z]$",
            "required_fields": ["number", "full_name", "dob", "sex", "expiry", "issuing_country"],
            "optional_fields": ["place_of_birth", "issuing_authority"],
        },
        "id_card": {
            "number": r"^[STFG]\d{7}[A-Z]$",
            "required_fields": ["number", "full_name", "dob", "sex"],
            "optional_fields": ["issue_date"],
            # NO expiry date for Singapore NRIC (valid for life)
        },
        "date_format": ["DD-MM-YY", "DD/MM/YYYY"],
        "date_label": ["date of birth", "dob", "birth date"],
    },
    "TH": {
        "name": "Thailand",
        "passport": {
            "number": r"^[A-Z]{2}\d{7}$",
            "required_fields": ["number", "full_name", "dob", "sex", "expiry"],
            "optional_fields": ["place_of_birth", "issuing_authority"],
        },
        "id_card": {
            "number": r"^\d{13}$",
            "required_fields": ["number", "full_name", "dob", "address"],
            "optional_fields": ["issue_date", "expiry"],  # Some Thai IDs may have expiry
        },
        "date_format": ["DD MMM YYYY", "DD-MM-YYYY"],
        "date_label": ["date of birth", "dob", "birth date"],
    },
    "MM": {
        "name": "Myanmar",
        "passport": {
            "number": r"^\w{8,9}$",
            "required_fields": ["number", "full_name", "dob", "sex", "expiry"],
            "optional_fields": ["nrc_number", "place_of_birth"],
        },
        "id_card": {
            "number": r"^\d{1,2}/\w{6}\(\w\)\d{6}$",  # NRC format: 12/MaMaNa(Pa)123456
            "required_fields": ["number", "full_name", "dob"],
            "optional_fields": ["address", "sex"],  # May vary by NRC type
        },
        "date_format": ["DD-MM-YYYY"],
        "date_label": ["date of birth", "dob", "birth date"],
    },
    "MY": {
        "name": "Malaysia",
        "passport": {
            "number": r"^[A-Z]\d{8}$",
            "required_fields": ["number", "full_name", "dob", "sex", "expiry"],
            "optional_fields": ["place_of_birth", "issuing_authority"],
        },
        "id_card": {
            "number": r"^\d{6}-\d{2}-\d{4}$",  # MyKad: 123456-01-5678
            "required_fields": ["number", "full_name", "dob", "address"],
            "optional_fields": ["sex", "religion"],
        },
        "date_format": ["DD-MM-YYYY"],
        "date_label": ["date of birth", "dob", "birth date"],
    },
    "ID": {
        "name": "Indonesia",
        "passport": {
            "number": r"^[A-Z]\d{7}$",
            "required_fields": ["number", "full_name", "dob", "sex", "expiry"],
            "optional_fields": ["place_of_birth", "issuing_authority"],
        },
        "id_card": {
            "number": r"^\d{16}$",  # KTP: 16 digits
            "required_fields": ["number", "full_name", "dob", "address"],
            "optional_fields": ["sex", "religion"],
        },
        "date_format": ["DD-MM-YYYY"],
        "date_label": ["date of birth", "dob", "birth date"],
    },
    "PH": {
        "name": "Philippines",
        "passport": {
            "number": r"^[A-Z]{2}\d{7}$",
            "required_fields": ["number", "full_name", "dob", "sex", "expiry"],
            "optional_fields": ["place_of_birth", "issuing_authority"],
        },
        "id_card": {
            "number": r"^\d{4}-\d{7}-\d{1}$",  # UMID: XXXX-XXXXXXX-X
            "required_fields": ["number", "full_name", "dob"],
            "optional_fields": ["sex", "address"],
        },
        "date_format": ["DD-MM-YYYY"],
        "date_label": ["date of birth", "dob", "birth date"],
    },
    "VN": {
        "name": "Vietnam",
        "passport": {
            "number": r"^[A-Z]\d{8}$",
            "required_fields": ["number", "full_name", "dob", "sex", "expiry"],
            "optional_fields": ["place_of_birth", "issuing_authority"],
        },
        "id_card": {
            "number": r"^\d{9}$",  # CMND: 9 digits
            "required_fields": ["number", "full_name", "dob", "sex"],
            "optional_fields": ["address", "issue_date", "expiry"],
        },
        "date_format": ["DD-MM-YYYY"],
        "date_label": ["date of birth", "dob", "birth date"],
    },
    "CN": {
        "name": "China",
        "passport": {
            "number": r"^[EG]\d{8}$",
            "required_fields": ["number", "full_name", "dob", "sex", "expiry"],
            "optional_fields": ["place_of_birth", "issuing_authority"],
        },
        "id_card": {
            "number": r"^\d{18}$",  # Resident ID: 18 digits
            "required_fields": ["number", "full_name", "dob", "address"],
            "optional_fields": ["sex", "ethnicity"],
        },
        "date_format": ["YYYY-MM-DD", "YYYYMMDD"],
        "date_label": ["date of birth", "dob", "birth date"],
    },
    "IN": {
        "name": "India",
        "passport": {
            "number": r"^[A-Z]\d{8}$",
            "required_fields": ["number", "full_name", "dob", "sex", "expiry"],
            "optional_fields": ["place_of_birth", "issuing_authority"],
        },
        "id_card": {
            "number": r"^\d{12}$",  # Aadhaar: 12 digits
            "required_fields": ["number", "full_name", "dob"],
            "optional_fields": ["sex", "address"],
        },
        "date_format": ["DD-MM-YYYY", "DD/MM/YYYY"],
        "date_label": ["date of birth", "dob", "birth date"],
    },
    "AE": {
        "name": "United Arab Emirates",
        "passport": {
            "number": r"^[A-Z]\d{8}$",
            "required_fields": ["number", "full_name", "dob", "sex", "expiry"],
            "optional_fields": ["place_of_birth", "issuing_authority"],
        },
        "id_card": {
            "number": r"^\d{3}-\d{4}-\d{7}-\d{1}$",  # Emirates ID: 784-XXXX-XXXXXXX-X
            "required_fields": ["number", "full_name", "dob"],
            "optional_fields": ["sex", "nationality", "expiry"],
        },
        "date_format": ["DD-MM-YYYY", "DD/MM/YYYY"],
        "date_label": ["date of birth", "dob", "birth date"],
    },
    # Add more countries as needed
}


# ============================================================================
# FIELD PATTERNS
# ============================================================================

FIELD_PATTERNS: Dict[str, Dict[str, Any]] = {
    "passport_number": {
        "labels": ["passport no", "passport number", "passport#", "no passport", "passport no."],
        "regex": r"\b[A-Z0-9]{6,12}\b",
        "position_hints": ["near country code", "after type"],
        "case_sensitive": False,
    },
    "id_number": {
        "labels": ["id number", "identity card", "national id", "id no", "nric", "id card no", "id card #", "mykad", "ktp", "umid", "cmnd", "aadhaar"],
        "regex": r"\b[A-Z0-9-]{6,20}\b",
        "position_hints": ["near country", "near date of birth"],
        "case_sensitive": False,
    },
    "full_name": {
        "labels": ["name", "full name", "given names", "surname", "last name", "first name"],
        "regex": r"[A-Z\s\.\-]{3,50}",
        "case_sensitive": False,
        "can_multiline": True,
    },
    "date_of_birth": {
        "labels": ["date of birth", "dob", "birth date", "born", "birthday"],
        "formats": ["DD-MM-YYYY", "DD/MM/YYYY", "DD MMM YYYY", "YYYY-MM-DD", "YYYYMMDD"],
        "optional": False,
        "case_sensitive": False,
    },
    "date_of_expiry": {
        "labels": ["date of expiry", "expiry date", "valid until", "expiration", "exp date", "valid thru"],
        "formats": ["DD-MM-YYYY", "DD/MM/YYYY", "DD MMM YYYY", "YYYY-MM-DD", "YYYYMMDD"],
        "optional": True,  # NOT present in Singapore NRIC
        "case_sensitive": False,
    },
    "date_of_issue": {
        "labels": ["date of issue", "issue date", "issued on", "iss date"],
        "formats": ["DD-MM-YYYY", "DD/MM/YYYY", "DD MMM YYYY", "YYYY-MM-DD", "YYYYMMDD"],
        "optional": True,
        "case_sensitive": False,
    },
    "sex": {
        "labels": ["sex", "gender", "male", "female"],
        "values": ["M", "F", "Male", "Female"],
        "optional": False,
        "case_sensitive": False,
    },
    "place_of_birth": {
        "labels": ["place of birth", "birth place", "birthplace", "pob", "born in"],
        "regex": r"[A-Z\s\.\-]{3,100}",
        "optional": True,
        "case_sensitive": False,
        "can_multiline": True,
    },
    "issuing_authority": {
        "labels": ["issuing authority", "issuing office", "issued by", "authority"],
        "regex": r"[A-Z0-9\s\.\-]{3,100}",
        "optional": True,
        "case_sensitive": False,
    },
    "issuing_country": {
        "labels": ["issuing country", "country", "nationality"],
        "regex": r"\b[A-Z]{3}\b",
        "optional": True,
        "case_sensitive": False,
    },
    "address": {
        "labels": ["address", "resident address", "current address", "home address", "permanent address"],
        "regex": r"[A-Z0-9\s\.\-,#()]{10,200}",
        "optional": True,
        "case_sensitive": False,
        "can_multiline": True,
    },
    "nrc_number": {
        "labels": ["nrc", "nrc no", "national registration card"],
        "regex": r"^\d{1,2}/\w{6}\(\w\)\d{6}$",
        "optional": True,
        "case_sensitive": False,
    },
}


# ============================================================================
# COUNTRY CODE MAPPINGS
# ============================================================================

COUNTRY_NAME_TO_CODE: Dict[str, str] = {
    "singapore": "SG",
    "thailand": "TH",
    "myanmar": "MM",
    "malaysia": "MY",
    "indonesia": "ID",
    "philippines": "PH",
    "vietnam": "VN",
    "china": "CN",
    "india": "IN",
    "united arab emirates": "AE",
    "uae": "AE",
    "dubai": "AE",
    "abu dhabi": "AE",
    # Add more mappings as needed
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_country_pattern(country_code: str) -> Optional[Dict[str, Any]]:
    """
    Get pattern definition for a country.

    Args:
        country_code: ISO 3166-1 alpha-2 country code (e.g., "SG", "TH")

    Returns:
        Country pattern dict or None if not found
    """
    return COUNTRY_PATTERNS.get(country_code.upper())


def get_field_pattern(field_name: str) -> Optional[Dict[str, Any]]:
    """
    Get pattern definition for a field.

    Args:
        field_name: Name of the field (e.g., "passport_number", "full_name")

    Returns:
        Field pattern dict or None if not found
    """
    return FIELD_PATTERNS.get(field_name)


def detect_country_from_text(text: str) -> Optional[str]:
    """
    Detect country from OCR text by searching for 3-letter ISO codes or country names.

    Handles concatenated text like "PA SGP" by splitting on spaces.

    Args:
        text: OCR text from document

    Returns:
        ISO country code (2-letter) or None if not detected
    """
    if not text:
        return None

    text_lower = text.lower()

    # First check for country names (more specific matches)
    for country_name, country_code in COUNTRY_NAME_TO_CODE.items():
        if country_name in text_lower:
            return country_code

    # Then check for ISO3 codes
    text_upper = text.upper()
    words = text_upper.split()

    for word in words:
        # Clean word of common punctuation
        word = word.strip(".,-/")
        # Check if it's a valid 3-letter ISO code
        if word in ISO3_TO_ISO2:
            return ISO3_TO_ISO2[word]

    return None


def detect_document_type_from_patterns(text: str, country_code: str) -> Optional[str]:
    """
    Detect document type (passport vs id_card) based on number patterns.

    Args:
        text: OCR text from document
        country_code: ISO country code

    Returns:
        "passport" or "id_card" or None if not detected
    """
    if not text or not country_code:
        return None

    country_pattern = get_country_pattern(country_code)
    if not country_pattern:
        return None

    text_upper = text.upper()

    # Check if passport number pattern matches
    if "passport" in country_pattern:
        passport_pattern = country_pattern["passport"]["number"]
        # Look for text matching the passport pattern
        words = text_upper.split()
        for word in words:
            word = word.strip(".,-")
            if word == passport_pattern or len(word) >= 6:
                return "passport"

    # Check if ID card number pattern matches
    if "id_card" in country_pattern:
        id_pattern = country_pattern["id_card"]["number"]
        words = text_upper.split()
        for word in words:
            word = word.strip(".,-")
            if len(word) >= 6:  # Basic heuristic for ID numbers
                return "id_card"

    # Default: assume passport if document type can't be determined
    return "passport"


def is_optional_field(field_name: str, country_code: str, document_type: str) -> bool:
    """
    Check if a field is optional for a given country and document type.

    Args:
        field_name: Name of the field (e.g., "date_of_expiry")
        country_code: ISO country code
        document_type: "passport" or "id_card"

    Returns:
        True if field is optional, False if required
    """
    country_pattern = get_country_pattern(country_code)
    if not country_pattern:
        return True  # Assume optional for unknown countries

    if document_type not in country_pattern:
        return True  # Assume optional for unknown document types

    doc_config = country_pattern[document_type]

    # Check if field is in optional_fields
    if "optional_fields" in doc_config:
        if field_name in doc_config["optional_fields"]:
            return True

    # Check if field is required
    if "required_fields" in doc_config:
        if field_name in doc_config["required_fields"]:
            return False

    # Default to optional if not explicitly listed
    return True

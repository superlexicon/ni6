"""
Address validation using rule-based patterns for international addresses.

Provides tiered trust system combining GLiNER confidence with structural validation.
This module implements rule-based validation as a fallback since pypostal/libpostal
has build issues on many systems.
"""

from typing import Tuple, Optional, Dict
from logging import getLogger
import re
import unicodedata

logger = getLogger(__name__)

# Country-specific postal code patterns
POSTAL_CODE_PATTERNS = {
    'SG': r'\b\d{6}\b',                    # Singapore: 6 digits
    'TH': r'\b\d{5}\b',                    # Thailand: 5 digits
    'MY': r'\b\d{5}\b',                    # Malaysia: 5 digits
    'IN': r'\b\d{6}\b',                    # India: 6 digits
    'AE': r'\b\d{5,6}\b',                  # UAE: 5-6 digits (varies)
    'MM': r'\b\d{5}\b',                    # Myanmar: 5 digits
    'VN': r'\b\d{6}\b',                    # Vietnam: 6 digits
    'PH': r'\b\d{4}\b',                    # Philippines: 4 digits
    'ID': r'\b\d{5}\b',                    # Indonesia: 5 digits
    'US': r'\b\d{5}(-\d{4})?\b',          # US: 5 digits or 5+4
    'UK': r'\b[A-Z]{1,2}\d[A-Z\d]? \d[A-Z]{2}\b',  # UK: SW1A 1AA format
    'default': r'\b\d{4,7}\b'              # Default: 4-7 digits
}

# Address keyword indicators (used when validation is uncertain)
ADDRESS_KEYWORDS = {
    'english': [
        'street', 'st', 'road', 'rd', 'lane', 'ln', 'drive', 'dr',
        'avenue', 'ave', 'place', 'pl', 'court', 'way', 'walk',
        'crescent', 'block', 'blk', 'flat', 'unit', 'building',
        'po box', 'postal', 'zip', 'jalan', 'lorong', 'soi',
        'nagar', 'colony', 'extension', 'sector', 'phase',
        'moo', 'baan', 'tambon', 'amphoe', 'khet', 'kwaeng'
    ],
    'thai': [
        'ซอย', 'ถนน', 'แยก', 'ตำบล', 'อำเภอ'
    ]
}

# Negative patterns - text that looks like address but isn't
NEGATIVE_PATTERNS = [
    r'^page\s+\d+',
    r'^\d+\s+of\s+\d+',
    r'^account\s+statement',
    r'^statement\s+period',
    r'^consolidated\s+statement',
    r'summary',
    r'balance\s+(summary|sheet)',
    r'transaction\s+(history|list)',
    r'^\w+\s+bank',  # Bank names without address info
    r'reduce\s+and\s+save',
    r'learn\s+more',
    r'^(all|total)\s+transactions?',
]

# Minimum quality thresholds (country-specific)
MIN_ADDRESS_LENGTH_DEFAULT = 10  # characters
MIN_ADDRESS_LENGTH_BY_COUNTRY = {
    'TH': 6,   # Thailand - short place names like "Lam Sai", "Bang Na"
    'MY': 6,   # Myanmar - short place names like "Thanlyin"
    'VN': 6,   # Vietnam - short place names like "Ha Dong"
    'KH': 6,   # Cambodia
    'LA': 6,   # Laos
    # Default for other countries
    'default': MIN_ADDRESS_LENGTH_DEFAULT
}
MIN_ADDRESS_WORDS = 2    # words

# GLiNER confidence thresholds (country-specific)
# SE Asian countries often have transliterated addresses (Latin script)
# that need lower GLiNER confidence thresholds
GLINER_MIN_CONFIDENCE_BY_COUNTRY = {
    'TH': 0.35,  # Thailand - transliterated addresses like "Lam Sai"
    'MY': 0.35,  # Myanmar - transliterated addresses
    'VN': 0.35,  # Vietnam - transliterated addresses
    'KH': 0.35,  # Cambodia
    'LA': 0.35,  # Laos
    'default': 0.5  # Default minimum GLiNER confidence
}


def _contains_non_latin_script(text: str) -> bool:
    """
    Check if text contains non-Latin script characters (Thai, Myanmar, Arabic, Chinese, etc.).

    These scripts should be validated more leniently since our keyword patterns
    are primarily English-based.
    """
    if not text:
        return False

    # Count non-ASCII characters (likely non-Latin script)
    non_ascii_count = 0
    total_chars = 0

    for char in text:
        if char.isalpha():
            total_chars += 1
            # Check if character is outside Latin range
            if ord(char) > 0x024F:  # Beyond Latin Extended-B
                non_ascii_count += 1

    # If more than 30% of alpha characters are non-Latin, consider it non-Latin text
    if total_chars > 0 and non_ascii_count / total_chars > 0.3:
        return True

    return False


def _is_garbled_ocr(text: str) -> bool:
    """
    Check if text appears to be garbled OCR output from non-Latin scripts.

    Garbled OCR often has many consecutive consonants or unusual character patterns
    that don't match typical address formats.
    """
    if not text or len(text) < 5:
        return False

    # Check for unusual consonant clusters (common in garbled Thai/Myanmar OCR)
    # These scripts when mis-OCR'd often produce strings with many lowercase vowels missing
    # Strip punctuation before checking word length (commas, colons, etc. can combine words)
    import string
    words = text.split()
    for word in words:
        # Strip punctuation from both ends
        cleaned_word = word.strip(string.punctuation)
        if len(cleaned_word) > 20:  # Very long word without spaces is suspicious (increased from 15)
            return True

    # Check ratio of consonants to vowels - garbled OCR often has very few vowels
    alpha_chars = [c for c in text if c.isalpha()]
    if len(alpha_chars) > 10:
        vowel_count = sum(1 for c in alpha_chars if c.lower() in 'aeiou')
        # If less than 10% vowels, likely garbled
        if vowel_count / len(alpha_chars) < 0.1:
            return True

    return False


def should_keep_address(
    gliner_confidence: float,
    addr_text: str,
    country_code: Optional[str] = None
) -> Tuple[bool, float, str]:
    """
    Decide whether to keep GLiNER-extracted address using tiered trust system.

    Args:
        gliner_confidence: GLiNER's confidence score (0-1)
        addr_text: The extracted address text
        country_code: ISO 3166-1 alpha-2 country code (e.g., 'SG', 'TH')

    Returns:
        (keep: bool, final_confidence: float, reason: str)

    Decision Logic:
    - Non-Latin script (Thai, Arabic, etc.): Trust GLiNER more (keywords don't apply)
    - Structural valid (has postal code or street keywords): Always keep, boost confidence
    - GLiNER >= 0.7: Keep even if structurally invalid (GLiNER might be right)
    - GLiNER 0.5-0.7: Keep only if has address keywords OR is non-Latin script
    - GLiNER < 0.5: Filter if structurally invalid (unless very high confidence non-Latin)
    - For SE Asian countries (TH, MY, VN, etc.): Use lower confidence thresholds due to transliteration
    """
    try:
        # Get country-specific minimum GLiNER confidence
        min_gliner_confidence = GLINER_MIN_CONFIDENCE_BY_COUNTRY.get(
            country_code or 'default',
            GLINER_MIN_CONFIDENCE_BY_COUNTRY['default']
        )

        # Check if text contains non-Latin script (Thai, Myanmar, Arabic, etc.)
        is_non_latin = _contains_non_latin_script(addr_text)
        is_garbled = _is_garbled_ocr(addr_text)

        # For non-Latin scripts, be more lenient - our keyword patterns don't apply
        if is_non_latin or is_garbled:
            # For non-Latin, trust GLiNER more since we can't validate with English keywords
            if gliner_confidence >= 0.5:
                return True, gliner_confidence, "non_latin_script_trusted"
            elif gliner_confidence >= 0.4 and _meets_minimum_quality(addr_text, country_code):
                return True, gliner_confidence, "non_latin_medium_confidence"
            else:
                return False, 0.0, "non_latin_low_confidence"

        # Standard validation for Latin scripts
        postal_valid = _has_postal_code(addr_text, country_code)
        has_keywords = _has_address_keywords(addr_text)
        min_quality = _meets_minimum_quality(addr_text, country_code)

        if not min_quality:
            return False, 0.0, "below_minimum_quality"

        # Check for negative patterns
        if _matches_negative_pattern(addr_text):
            return False, 0.0, "negative_pattern"

        # Strong structural indicator: has postal code
        if postal_valid:
            return True, min(gliner_confidence + 0.1, 1.0), "postal_code_valid"

        # Good indicator: has street keywords
        if has_keywords:
            if gliner_confidence >= 0.4:
                return True, min(gliner_confidence + 0.05, 1.0), "keywords_present"
            else:
                return False, 0.0, "low_confidence_with_keywords"

        # No structural indicators - trust GLiNER more
        if gliner_confidence >= 0.7:
            # Trust GLiNER for high confidence
            return True, gliner_confidence, "gliner_high_confidence"

        elif gliner_confidence >= min_gliner_confidence:
            # Medium confidence - needs at least some structure
            # For SE Asian countries, use lower threshold due to transliteration
            # Check for basic address characteristics (numbers, mixed case)
            if _has_basic_address_structure(addr_text):
                return True, gliner_confidence, "medium_confidence_basic_structure"
            # For SE Asian countries, be even more lenient - just check minimum quality
            # This allows short place names like "Lam Sai" to pass
            is_se_asian_country = country_code in ['TH', 'MY', 'VN', 'KH', 'LA']
            if is_se_asian_country and min_quality:
                return True, gliner_confidence, "se_asian_short_place_name"
            return False, 0.0, "medium_confidence_no_structure"

        else:
            # Low confidence + no structural indicators = filter it
            return False, 0.0, "low_confidence_filtered"

    except Exception as e:
        logger.warning(f"Address validation failed: {e}")
        # On error, keep high confidence extractions
        if gliner_confidence >= 0.7:
            return True, gliner_confidence, "validation_error_fallback"
        return False, 0.0, "validation_error_low_confidence"


def _has_postal_code(text: str, country_code: Optional[str] = None) -> bool:
    """Check if text contains a postal code for the given country."""
    pattern = POSTAL_CODE_PATTERNS.get(country_code or 'default',
                                       POSTAL_CODE_PATTERNS['default'])
    return bool(re.search(pattern, text, re.IGNORECASE))


def _has_address_keywords(text: str) -> bool:
    """Check if text contains address-related keywords."""
    text_lower = text.lower()
    for keyword in ADDRESS_KEYWORDS['english']:
        if keyword in text_lower:
            return True
    return False


def _meets_minimum_quality(text: str, country_code: Optional[str] = None) -> bool:
    """Check if text meets minimum quality requirements."""
    # Get country-specific minimum length
    min_length = MIN_ADDRESS_LENGTH_BY_COUNTRY.get(
        country_code or 'default',
        MIN_ADDRESS_LENGTH_DEFAULT
    )

    if len(text) < min_length:
        return False
    if len(text.split()) < MIN_ADDRESS_WORDS:
        return False
    return True


def _matches_negative_pattern(text: str) -> bool:
    """Check if text matches negative patterns (non-address content)."""
    text_lower = text.lower().strip()
    for pattern in NEGATIVE_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False


def _has_basic_address_structure(text: str) -> bool:
    """Check if text has basic address characteristics."""
    # Should contain at least one number
    has_digit = bool(re.search(r'\d', text))
    # Should have mixed content (letters and numbers/digits)
    words = text.split()
    if len(words) < 2:
        return False
    # Check for some uppercase (common in addresses)
    has_upper = any(c.isupper() for c in text)
    return has_digit and has_upper


def normalize_address(addr_text: str, country_code: Optional[str] = None) -> str:
    """
    Normalize address by standardizing common abbreviations.

    Args:
        addr_text: The address text to normalize
        country_code: ISO 3166-1 alpha-2 country code

    Returns:
        Normalized address text
    """
    if not addr_text:
        return addr_text

    # Common abbreviations to expand
    abbreviations = {
        'blk': 'block',
        'st': 'street',
        'rd': 'road',
        'ln': 'lane',
        'dr': 'drive',
        'ave': 'avenue',
        'pl': 'place',
        'sq': 'square',
        'cir': 'circle',
        'ct': 'court',
        'way': 'way',
        'bldg': 'building',
        'apt': 'apartment',
        'unit': 'unit',
        'fl': 'floor',
        'ph': 'penthouse',
    }

    # Tokenize and expand abbreviations (case-insensitive)
    words = addr_text.split()
    normalized_words = []

    for word in words:
        # Remove punctuation for matching
        word_clean = word.rstrip('.,;:').lower()
        if word_clean in abbreviations:
            # Expand abbreviation
            expanded = abbreviations[word_clean]
            # Preserve original capitalization pattern
            if word.isupper():
                normalized_words.append(expanded.upper())
            elif word[0].isupper():
                normalized_words.append(expanded.capitalize())
            else:
                normalized_words.append(expanded)
        else:
            normalized_words.append(word)

    return ' '.join(normalized_words)


def validate_with_required_components(
    addr_text: str,
    country_code: Optional[str] = None
) -> Tuple[bool, Dict[str, str]]:
    """
    Validate address has required components for its country.

    This is a simplified version that checks for basic components.
    Returns:
        (is_valid: bool, components: dict)
    """
    components = {}

    # Extract postal code if present
    pattern = POSTAL_CODE_PATTERNS.get(country_code or 'default',
                                       POSTAL_CODE_PATTERNS['default'])
    postal_match = re.search(pattern, addr_text)
    if postal_match:
        components['postal_code'] = postal_match.group(0)

    # Extract street keywords
    text_lower = addr_text.lower()
    for keyword in ADDRESS_KEYWORDS['english']:
        if keyword in text_lower:
            components['street_type'] = keyword
            break

    # Basic validation: should have at least one component
    is_valid = (
        _meets_minimum_quality(addr_text) and
        not _matches_negative_pattern(addr_text) and
        (bool(components) or _has_address_keywords(addr_text))
    )

    return is_valid, components


def is_valid_address(text: str) -> bool:
    """
    Quick validation check for address text.
    Returns True if text appears to be a valid address.
    """
    return (
        _meets_minimum_quality(text) and
        not _matches_negative_pattern(text) and
        (_has_postal_code(text) or _has_address_keywords(text))
    )


def looks_like_bank_address(text: str) -> bool:
    """
    Check if address text looks like a bank address rather than a personal address.

    Bank addresses often contain:
    - "Branch", "Bank", "Head Office", "HQ", "Operations Center"
    - Well-known street names that are typically commercial areas
    - Post office boxes for banks

    Returns True if the address appears to be a bank address.
    """
    if not text:
        return False

    text_lower = text.lower()

    # Keywords that strongly suggest bank/commercial address
    bank_keywords = [
        'branch', 'head office', 'h.o.', 'hq', 'operations center',
        'corporate office', 'registered office', 'business center',
        'tower', 'plaza', 'complex', 'centre', 'center'
    ]

    for keyword in bank_keywords:
        if keyword in text_lower:
            return True

    # Check if the address contains bank-related context
    # (e.g., "Krung Thai Bank" mentioned nearby in address)
    bank_names = ['bank', 'financial', 'credit', 'investment', 'capital']
    if any(name in text_lower for name in bank_names):
        # Only flag if combined with other commercial indicators
        if any(indicator in text_lower for indicator in ['road', 'street', 'building', 'tower']):
            return True

    return False

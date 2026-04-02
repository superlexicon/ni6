"""
Date extraction utility for documents.

Extracts dates by pattern matching rather than spatial/LLM-based detection.
Finds all dates matching common formats. Shared across all document extractors
(passports, bank statements, tax statements, etc.).
"""

import re
from datetime import date, datetime
from typing import List, Optional, Tuple


# Month name to number mapping
MONTH_MAP = {
    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4,
    'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8,
    'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
}


# Common date patterns with their strptime formats
DATE_PATTERNS = [
    # DDMMMYYYY (no spaces, e.g., "30DEC2025", "03JAN2026" - common in bank statements)
    (r'\b(\d{1,2})(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d{4})\b', '%d %b %Y'),
    # DD MMM YYYY (most reliable for bank statements)
    (r'\b(\d{1,2})\s+(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+(\d{4})\b', '%d %b %Y'),
    # DD MMM. YYYY (with period after abbreviation, e.g., "30 Nov. 2025")
    (r'\b(\d{1,2})\s+(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\.\s+(\d{4})\b', '%d %b %Y'),
    # DD MONTH YYYY (full month names, e.g., "18 January 2026")
    (r'\b(\d{1,2})\s+(JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s+(\d{4})\b', '%d %B %Y'),
    # Month DD, YYYY (full month names with comma, e.g., "January 18, 2026")
    (r'\b(JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s+(\d{1,2}),\s+(\d{4})\b', '%B %d, %Y'),
    # DD/MM/YYYY
    (r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b', '%d/%m/%Y'),
    # DD-MM-YYYY
    (r'\b(\d{1,2})-(\d{1,2})-(\d{4})\b', '%d-%m-%Y'),
    # YYYY-MM-DD (ISO)
    (r'\b(\d{4})-(\d{2})-(\d{2})\b', '%Y-%m-%d'),
    # YYYY/MM/DD
    (r'\b(\d{4})/(\d{2})/(\d{2})\b', '%Y/%m/%d'),
]


def extract_all_dates(text: str, max_age_days: int = 730) -> List[date]:
    """
    Extract all valid dates from text.

    Args:
        text: Text to extract dates from
        max_age_days: Maximum age of dates to include (default 2 years)

    Returns:
        List of date objects found in the text
    """
    if not text:
        return []

    dates = []
    text_upper = text.upper()
    today = date.today()

    for pattern, fmt in DATE_PATTERNS:
        for match in re.finditer(pattern, text_upper, re.IGNORECASE):
            try:
                # Build date string from captured groups to handle variations
                # (e.g., "30 Nov. 2025" -> groups are (30, NOV, 2025))
                groups = match.groups()
                if len(groups) == 3:
                    # Reconstruct clean date string with proper formatting
                    if fmt == '%d %b %Y':
                        date_str = f"{groups[0]} {groups[1]} {groups[2]}"
                    elif fmt == '%d %B %Y':
                        date_str = f"{groups[0]} {groups[1]} {groups[2]}"
                    elif fmt == '%B %d, %Y':
                        # Month DD, YYYY format - preserve the comma
                        date_str = f"{groups[0]} {groups[1]}, {groups[2]}"
                    elif fmt in ('%d/%m/%Y', '%d-%m-%Y'):
                        date_str = match.group(0).upper()
                    elif fmt in ('%Y-%m-%d', '%Y/%m/%d'):
                        date_str = match.group(0).upper()
                    else:
                        date_str = match.group(0).upper()
                else:
                    date_str = match.group(0).upper()

                # Parse date string with format
                parsed = datetime.strptime(date_str, fmt).date()

                # Filter: must be within max_age_days and not in future
                if parsed <= today and (today - parsed).days <= max_age_days:
                    dates.append(parsed)
            except ValueError:
                continue

    return dates


def get_statement_date(text: str) -> Optional[date]:
    """
    Get statement date by finding the most recent valid date in the text.

    Bank statements typically show the most recent date as the statement date
    or end-of-period date.

    Args:
        text: Text to extract statement date from

    Returns:
        Most recent valid date found, or None if no valid dates
    """
    dates = extract_all_dates(text)
    if not dates:
        return None
    # Return most recent date
    return max(dates)


def get_statement_date_str(text: str, fmt: str = "%d %b %Y") -> Optional[str]:
    """
    Get statement date as formatted string.

    Args:
        text: Text to extract statement date from
        fmt: Output date format (default: "DD MMM YYYY")

    Returns:
        Formatted date string, or None if no valid dates
    """
    statement_date = get_statement_date(text)
    if statement_date:
        return statement_date.strftime(fmt).upper()
    return None


def extract_and_remove_dates(text: str) -> Tuple[List[Tuple[datetime, str]], str]:
    """
    Extract embedded dates from string and return remaining text.

    Handles merged OCR values like "10 NOV 1992SINGAPORE" where date has
    internal spaces but is concatenated with other values.

    Supports formats:
    - DD MMM YYYY (e.g., "10 NOV 1992")
    - DD/MM/YYYY or DD-MM-YYYY (e.g., "10/11/1992", "10-11-1992")
    - YYYY-MM-DD or YYYY/MM/DD (e.g., "1992-11-10")

    Args:
        text: Input string potentially containing embedded dates

    Returns:
        Tuple of:
        - List of (datetime, original_text) tuples for found dates
        - Remaining string after dates are removed

    Examples:
        "10 NOV 1992 SINGAPORE" → ([(datetime(1992,11,10), "10 NOV 1992")], "SINGAPORE")
        "PA SGP 25 DEC 2030" → ([(datetime(2030,12,25), "25 DEC 2030")], "PA SGP")
        "DOB:10/11/1992END" → ([(datetime(1992,11,10), "10/11/1992")], "DOB:END")
    """
    remaining = text.upper()
    found_dates = []

    # Collect all matches with their positions for unified removal
    all_matches = []  # [(start, end, datetime, original_text), ...]

    # Pattern 1: DD MMM YYYY (e.g., "10 NOV 1992", "10NOV 1992", "10NOV1992")
    # Use \s* (zero or more spaces) to handle tight OCR spacing in passport MRZ/VIZ
    pattern1 = r'(\d{1,2})\s*([A-Z]{3})\s+(\d{4})'
    for match in re.finditer(pattern1, remaining):
        day_str, month_name, year_str = match.groups()
        month = MONTH_MAP.get(month_name)
        if month:
            try:
                dt = datetime(int(year_str), month, int(day_str))
                all_matches.append((match.start(), match.end(), dt, match.group(0)))
            except ValueError:
                pass

    # Pattern 2: DD/MM/YYYY or DD-MM-YYYY (e.g., "10/11/1992")
    pattern2 = r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})'
    for match in re.finditer(pattern2, remaining):
        day_str, month_str, year_str = match.groups()
        try:
            dt = datetime(int(year_str), int(month_str), int(day_str))
            all_matches.append((match.start(), match.end(), dt, match.group(0)))
        except ValueError:
            pass

    # Pattern 3: YYYY-MM-DD or YYYY/MM/DD (e.g., "1992-11-10")
    pattern3 = r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})'
    for match in re.finditer(pattern3, remaining):
        year_str, month_str, day_str = match.groups()
        try:
            dt = datetime(int(year_str), int(month_str), int(day_str))
            all_matches.append((match.start(), match.end(), dt, match.group(0)))
        except ValueError:
            pass

    # Sort by position and remove duplicates (same position = same date)
    all_matches.sort(key=lambda x: x[0])

    # Remove overlapping matches (keep first occurrence)
    filtered_matches = []
    last_end = -1
    for start, end, dt, original in all_matches:
        if start >= last_end:
            filtered_matches.append((start, end, dt, original))
            last_end = end

    # Remove dates from string in reverse order to preserve indices
    for start, end, dt, original in reversed(filtered_matches):
        found_dates.insert(0, (dt, original))
        remaining = remaining[:start] + remaining[end:]

    return found_dates, remaining.strip()

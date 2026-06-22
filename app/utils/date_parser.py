"""
Date Parser Utility

Provides consistent date parsing for MariaDB DATE columns across the application.
Handles various date formats from passport extraction, bank statements, and other sources.
"""

from datetime import datetime, date
from typing import Optional, Union
import re
from dateutil import parser as dateutil_parser


def parse_date_to_mariadb(date_input: Optional[Union[str, date, datetime]]) -> Optional[date]:
    """
    Parse various date formats and return a Python date object suitable for MariaDB DATE column.

    Supported formats:
    - DD MMM YYYY (e.g., "10 NOV 1982") - from passport extractor
    - YYYY-MM-DD (e.g., "1982-11-10") - ISO format
    - DD-MM-YYYY (e.g., "10-11-1982")
    - DD/MM/YYYY (e.g., "10/11/1982")
    - YYYYMMDD (e.g., "19821110")
    - DDMMYYYY (e.g., "10111982")
    - Python date or datetime objects

    Args:
        date_input: Date string, date object, or datetime object to parse

    Returns:
        Python date object or None if parsing fails
    """
    if date_input is None:
        return None

    # Handle date objects directly
    if isinstance(date_input, date) and not isinstance(date_input, datetime):
        return date_input

    # Handle datetime objects
    if isinstance(date_input, datetime):
        return date_input.date()

    # Handle string inputs
    if not isinstance(date_input, str):
        return None

    date_str = date_input.strip().upper()

    if not date_str:
        return None

    # Handle date ranges (e.g., "From : 02-02-2026 To : 03-02-2026")
    # Extract the end date (most recent) for statement_date
    range_patterns = [
        # "From : DD-MM-YYYY To : DD-MM-YYYY" or "From : DD/MM/YYYY To : DD/MM/YYYY"
        r'from\s*:\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\s+to\s*:\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
        # "From: DD-MM-YYYY to DD-MM-YYYY" (without spaces after colons)
        r'from\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\s+to\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
        # "DD-MM-YYYY to DD-MM-YYYY" (without From/To labels)
        r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\s+to\s+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
    ]

    for pattern in range_patterns:
        match = re.search(pattern, date_str, re.IGNORECASE)
        if match:
            end_date_str = match.group(2)  # Get the second date (end date)
            # Try parsing with common formats
            for fmt in ["%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%Y/%m/%d"]:
                try:
                    return datetime.strptime(end_date_str, fmt).date()
                except ValueError:
                    continue
            # If formats don't match, try dateutil as fallback
            try:
                return dateutil_parser.parse(end_date_str, dayfirst=True).date()
            except Exception:
                pass
            break  # Found a range pattern but couldn't parse, don't try other patterns

    # Month name mapping for DD MMM YYYY format
    months = {
        'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4,
        'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8,
        'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
    }

    # Try DD MMM YYYY format first (most common from passport extraction)
    match = re.match(r'(\d{1,2})\s*([A-Z]{3})\s*(\d{4})', date_str)
    if match:
        day, month_name, year = match.groups()
        month = months.get(month_name)
        if month:
            try:
                return date(int(year), month, int(day))
            except ValueError:
                pass

    # Try explicit date formats
    formats = [
        "%Y-%m-%d",    # 1990-01-15 (ISO)
        "%d-%m-%Y",    # 15-01-1990
        "%Y/%m/%d",    # 1990/01/15
        "%d/%m/%Y",    # 15/01/1990
        "%Y%m%d",      # 19900115
        "%d%m%Y",      # 15011990
        "%m/%d/%Y",    # 01/15/1990 (US format)
        "%m-%d-%Y",    # 01-15-1990 (US format)
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue

    # Last resort: use dateutil parser (handles many formats)
    try:
        # Use dayfirst=True for DD/MM/YYYY preference (non-US format)
        return dateutil_parser.parse(date_str, dayfirst=True).date()
    except Exception:
        pass

    return None


def format_date_for_display(date_obj: Optional[date], format_str: str = "%Y-%m-%d") -> Optional[str]:
    """
    Format a date object for display.

    Args:
        date_obj: Date object to format
        format_str: Format string (default: ISO format YYYY-MM-DD)

    Returns:
        Formatted date string or None if input is None
    """
    if date_obj is None:
        return None

    return date_obj.strftime(format_str)


def format_date_for_passport(date_obj: Optional[Union[date, datetime]]) -> Optional[str]:
    """
    Format a date object in standard passport format: DD MMM YYYY.

    Args:
        date_obj: Date or datetime object to format

    Returns:
        Formatted date string like "11 AUG 1996" or None if input is None

    Examples:
        date(1996, 8, 11) -> "11 AUG 1996"
        datetime(1996, 8, 11, 12, 30) -> "11 AUG 1996"
    """
    if date_obj is None:
        return None

    # Convert datetime to date if needed
    if isinstance(date_obj, datetime):
        date_obj = date_obj.date()

    # Month abbreviations in uppercase (as used in passports)
    months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
              'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

    return f"{date_obj.day} {months[date_obj.month - 1]} {date_obj.year}"

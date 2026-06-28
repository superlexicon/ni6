"""
ICAO 9303 MRZ (Machine Readable Zone) Parser

Implements position-based parsing of passport MRZ with check digit validation.
Uses the 7-3-1 weighting algorithm for check digit computation.
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class MRZData:
    """Parsed MRZ data with validation status."""
    passport_number: str
    country_code: str  # 3-letter code
    surname: str
    given_names: str
    date_of_birth: str  # YYMMDD format
    sex: str  # M/F/X
    date_of_expiry: str  # YYMMDD format
    optional_field: str  # Personal number (optional)
    passport_number_valid: bool  # Check digit validation
    dob_valid: bool
    expiry_valid: bool
    composite_valid: bool
    all_valid: bool


class MRZParser:
    """
    ICAO 9303 MRZ parser with check digit validation.

    The MRZ uses check digits to verify data integrity.
    Check digit algorithm: 7-3-1 weighting on alpha-numeric values.
    """

    # Character mappings for check digit computation
    # Maps characters to their numeric values (0-9 letters → 0-35, < → 0)
    CHAR_MAP = {str(i): i for i in range(10)}
    CHAR_MAP.update({chr(65 + i): 10 + i for i in range(26)})  # A-Z
    CHAR_MAP['<'] = 0  # Filler character

    @classmethod
    def char_value(cls, char: str) -> int:
        """Get numeric value of a character for check digit computation."""
        return cls.CHAR_MAP.get(char.upper(), 0)

    @classmethod
    def compute_check_digit(cls, data: str) -> int:
        """
        Compute check digit using 7-3-1 weighting algorithm.

        Args:
            data: String to compute check digit for

        Returns:
            Check digit (0-9)
        """
        total = 0
        weights = [7, 3, 1] * ((len(data) // 3) + 1)

        for i, char in enumerate(data):
            total += cls.char_value(char) * weights[i]

        return total % 10

    @classmethod
    def parse_td3(cls, line1: str, line2: str) -> Optional[MRZData]:
        """
        Parse TD3 format MRZ (2 lines, 44 chars each).

        TD3 Format:
        Line 1 (44 chars):
        - Position 0: Document type (P)
        - Position 1: Filler (<)
        - Position 2-4: Country code (3 letters)
        - Position 5-44: Names (SURNAME<<GIVEN<NAMES) + Passport number
        - Last character before end: Check digit for passport number

        Line 2 (44 chars):
        - Position 0-5: DOB (YYMMDD)
        - Position 6: DOB check digit
        - Position 7: Sex (M/F/X)
        - Position 8-13: Expiry date (YYMMDD)
        - Position 14: Expiry check digit
        - Position 15-42: Optional/Personal number
        - Position 43: Composite check digit

        Args:
            line1: First line of MRZ (44 chars)
            line2: Second line of MRZ (44 chars)

        Returns:
            MRZData with parsed fields and validation status
        """
        # Clean and validate format
        line1 = line1.strip().replace(' ', '').upper()
        line2 = line2.strip().replace(' ', '').upper()

        if len(line1) != 44 or len(line2) != 44:
            return None

        if not line1.startswith('P'):
            return None

        # Parse Line 1
        # Position 0: Document type
        doc_type = line1[0]

        # Position 1: Filler
        filler = line1[1]

        # Position 2-4: Country code (3 letters)
        country_code = line1[2:5]

        # Position 5-42: Names and passport number (38 characters)
        # Format: SURNAME<<GIVEN<NAMES<<<<<<<<<...<<<<PASSPORT_NUMBER
        # - Names are separated by '<<'
        # - Filler '<' characters fill space after names
        # - Passport number is at the end ( alphanumeric, typically ending with digits)
        # - Check digit for passport number is at position 43

        remainder = line1[5:43]  # 38 characters (positions 5-42)

        # Extract passport number from the end
        # Passport number is typically 8-9 alphanumeric chars at the end
        # It's preceded by '<' filler characters
        passport_number = ""
        names_section = remainder

        # Work backwards from the end to find the passport number
        # Passport number is alphanumeric and typically ends with digits
        temp = remainder.rstrip('<')
        if temp:
            # Find where the passport number starts (last sequence of alphanumeric chars)
            # Remove '<' fillers and work backwards
            i = len(temp) - 1
            while i >= 0 and temp[i].isalnum():
                i -= 1

            if i < len(temp) - 1:
                # Found passport number at the end
                passport_number = temp[i + 1:]
                names_section = temp[:i + 1]

        # Check digit at position 43
        passport_check_digit = cls.char_value(line1[43])

        # Parse names from names_section
        # Format: SURNAME<<GIVEN<NAMES
        # Primary surname is between first '<' and '<<'
        # Given names are after '<<'

        # Remove leading '<' if present
        if names_section.startswith('<'):
            names_section = names_section[1:]

        # Find the first '<<' separator
        double_separator = names_section.find('<<')

        if double_separator != -1:
            # Surname is from first character up to '<<'
            surname_part = names_section[:double_separator]
            surname = surname_part.replace('<', ' ').strip()

            # Given names are after '<<'
            given_names_part = names_section[double_separator + 2:]
            given_names = given_names_part.replace('<', ' ').strip()
        else:
            # No '<<' found, try to parse differently
            # Split by '<' and take first non-empty part as surname
            parts = [p for p in names_section.split('<') if p.strip()]
            if len(parts) >= 2:
                surname = parts[0].strip()
                given_names = ' '.join(parts[1:]).strip()
            elif len(parts) == 1:
                surname = parts[0].strip()
                given_names = ""
            else:
                surname = ""
                given_names = ""

        # Parse Line 2
        # Position 0-5: DOB (YYMMDD)
        dob = line2[0:6]

        # Position 6: DOB check digit
        dob_check_digit = cls.char_value(line2[6])

        # Position 7: Sex (M/F/X)
        sex = line2[7]

        # Position 8-13: Expiry date (YYMMDD)
        expiry = line2[8:14]

        # Position 14: Expiry check digit
        expiry_check_digit = cls.char_value(line2[14])

        # Position 15-42: Optional/Personal number (28 characters)
        optional = line2[15:43]

        # Position 43: Composite check digit
        composite_check_digit = cls.char_value(line2[43])

        # Validate check digits
        passport_valid = cls.compute_check_digit(passport_number) == passport_check_digit if passport_number else False
        dob_valid = cls.compute_check_digit(dob) == dob_check_digit
        expiry_valid = cls.compute_check_digit(expiry) == expiry_check_digit

        # Composite check digit covers positions 0-42 of line 2 (DOB, sex, expiry, optional)
        composite_data = line2[0:43]
        composite_valid = cls.compute_check_digit(composite_data) == composite_check_digit

        all_valid = passport_valid and dob_valid and expiry_valid and composite_valid

        return MRZData(
            passport_number=passport_number,
            country_code=country_code,
            surname=surname,
            given_names=given_names,
            date_of_birth=dob,
            sex=sex,
            date_of_expiry=expiry,
            optional_field=optional,
            passport_number_valid=passport_valid,
            dob_valid=dob_valid,
            expiry_valid=expiry_valid,
            composite_valid=composite_valid,
            all_valid=all_valid
        )

    @classmethod
    def normalize_date(cls, yyymmdd: str) -> Optional[str]:
        """
        Convert YYMMDD to YYYY-MM-DD format.

        Args:
            yyymmdd: 6-digit date string

        Returns:
            Date in YYYY-MM-DD format, or None if invalid
        """
        if len(yyymmdd) != 6 or not yyymmdd.isdigit():
            return None

        yy = int(yyymmdd[0:2])
        mm = yyymmdd[2:4]
        dd = yyymmdd[4:6]

        # Convert 2-digit year (if >= 50, assume 19YY; else 20YY)
        yyyy = 1900 + yy if yy >= 50 else 2000 + yy

        return f"{yyyy}-{mm}-{dd}"

    @classmethod
    def find_and_parse_mrz(cls, text_blocks: list) -> Optional[MRZData]:
        """
        Find MRZ lines in OCR output and parse them.

        MRZ is identified by:
        - Two consecutive lines
        - Each line is 44 characters
        - Contains many '<' characters
        - Line 1 starts with 'P'

        Args:
            text_blocks: List of OCR text blocks with geometry

        Returns:
            MRZData if found and parsed, None otherwise
        """
        # Extract plain text from blocks
        texts = [block.get('text', '').strip() for block in text_blocks]

        # Find potential MRZ lines
        mrz_candidates = []
        for text in texts:
            # MRZ lines are typically all uppercase with many '<'
            if len(text) >= 30 and text.count('<') >= 3:
                cleaned = text.replace(' ', '').upper()
                mrz_candidates.append(cleaned)

        # Try to find TD3 format (2 lines of 44 chars)
        for i in range(len(mrz_candidates) - 1):
            line1 = mrz_candidates[i]
            line2 = mrz_candidates[i + 1]

            if len(line1) == 44 and len(line2) == 44 and line1.startswith('P'):
                # Try to parse as TD3
                result = cls.parse_td3(line1, line2)
                if result and result.all_valid:
                    return result

        return None

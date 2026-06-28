"""
Unit tests for MRZ (Machine Readable Zone) Parser.

Tests the ICAO 9303-compliant MRZ parser with check digit validation.
"""

import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import MRZParser directly to avoid app module issues
import importlib.util
spec = importlib.util.spec_from_file_location("mrz_parser", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "utils", "mrz_parser.py"))
mrz_parser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mrz_parser)

MRZParser = mrz_parser.MRZParser
MRZData = mrz_parser.MRZData


class TestMRZParser(unittest.TestCase):
    """Test cases for MRZ parser."""

    def test_char_value(self):
        """Test character to numeric value conversion."""
        # Digits 0-9 map to 0-9
        self.assertEqual(MRZParser.char_value('0'), 0)
        self.assertEqual(MRZParser.char_value('9'), 9)

        # Letters A-Z map to 10-35
        self.assertEqual(MRZParser.char_value('A'), 10)
        self.assertEqual(MRZParser.char_value('Z'), 35)

        # Filler character maps to 0
        self.assertEqual(MRZParser.char_value('<'), 0)

        # Case insensitive
        self.assertEqual(MRZParser.char_value('a'), 10)
        self.assertEqual(MRZParser.char_value('z'), 35)

    def test_compute_check_digit(self):
        """Test check digit computation using 7-3-1 weighting algorithm."""
        # Test with passport number A1234567
        # Expected check digit: 6
        self.assertEqual(MRZParser.compute_check_digit('A1234567'), 6)

        # Test with date 900115
        # Expected check digit: 8
        self.assertEqual(MRZParser.compute_check_digit('900115'), 8)

        # Test with date 250620
        # Expected check digit: 7
        self.assertEqual(MRZParser.compute_check_digit('250620'), 7)

    def test_parse_td3_valid(self):
        """Test parsing valid TD3 format MRZ."""
        # US Passport MRZ
        line1 = 'P<USADOE<<JOHN<<<A1234567<<<<<<<<<<<<<<<<<<6'
        line2 = '9001158M2506207<<<<<<<<<<<<<<<<<<<<<<<<<<<<4'

        result = MRZParser.parse_td3(line1, line2)

        self.assertIsNotNone(result)
        self.assertTrue(result.all_valid)
        self.assertEqual(result.passport_number, 'A1234567')
        self.assertEqual(result.country_code, 'USA')
        self.assertEqual(result.surname, 'DOE')
        self.assertEqual(result.given_names, 'JOHN')
        self.assertEqual(result.date_of_birth, '900115')
        self.assertEqual(result.sex, 'M')
        self.assertEqual(result.date_of_expiry, '250620')
        self.assertTrue(result.passport_number_valid)
        self.assertTrue(result.dob_valid)
        self.assertTrue(result.expiry_valid)
        self.assertTrue(result.composite_valid)

    def test_parse_td3_invalid_format(self):
        """Test parsing TD3 with invalid format."""
        # Line 1 not starting with P
        line1 = 'X<USADOE<<JOHN<<<A1234567<<<<<<<<<<<<<<<<<<6'
        line2 = '9001158M2506207<<<<<<<<<<<<<<<<<<<<<<<<<<<<4'

        result = MRZParser.parse_td3(line1, line2)
        self.assertIsNone(result)

        # Wrong length
        line1 = 'P<USADOE<<JOHN<<<A1234567<<<<<<<<<<<<<<<<<<6'
        line2 = '9001158M2506207<<<<<<<<<<<<<<<<<<<<<<<<<<'  # 43 chars

        result = MRZParser.parse_td3(line1, line2)
        self.assertIsNone(result)

    def test_parse_td3_invalid_check_digits(self):
        """Test parsing TD3 with invalid check digits."""
        # Valid format but wrong check digits
        line1 = 'P<USADOE<<JOHN<<<A1234567<<<<<<<<<<<<<<<<<<9'  # Wrong check digit
        line2 = '9001158M2506207<<<<<<<<<<<<<<<<<<<<<<<<<<<<4'

        result = MRZParser.parse_td3(line1, line2)

        self.assertIsNotNone(result)
        self.assertFalse(result.all_valid)
        self.assertFalse(result.passport_number_valid)

    def test_normalize_date(self):
        """Test date normalization from YYMMDD to YYYY-MM-DD."""
        # Year >= 50 should be 19YY
        self.assertEqual(MRZParser.normalize_date('900115'), '1990-01-15')
        self.assertEqual(MRZParser.normalize_date('851231'), '1985-12-31')

        # Year < 50 should be 20YY
        self.assertEqual(MRZParser.normalize_date('250620'), '2025-06-20')
        self.assertEqual(MRZParser.normalize_date('050101'), '2005-01-01')

        # Invalid dates
        self.assertIsNone(MRZParser.normalize_date('12345'))  # Too short
        self.assertIsNone(MRZParser.normalize_date('abcdef'))  # Non-numeric
        self.assertIsNone(MRZParser.normalize_date(''))  # Empty

    def test_find_and_parse_mrz(self):
        """Test finding and parsing MRZ in OCR output."""
        # Simulated OCR output with MRZ
        text_blocks = [
            {'text': 'Some other text above MRZ'},
            {'text': 'P<USADOE<<JOHN<<<A1234567<<<<<<<<<<<<<<<<<<6'},
            {'text': '9001158M2506207<<<<<<<<<<<<<<<<<<<<<<<<<<<<4'},
            {'text': 'Some text below MRZ'}
        ]

        result = MRZParser.find_and_parse_mrz(text_blocks)

        self.assertIsNotNone(result)
        self.assertTrue(result.all_valid)
        self.assertEqual(result.passport_number, 'A1234567')

    def test_find_and_parse_mrz_not_found(self):
        """Test when MRZ is not found in OCR output."""
        # OCR output without MRZ
        text_blocks = [
            {'text': 'Some random text'},
            {'text': 'More random text without MRZ format'},
            {'text': 'No MRZ here'}
        ]

        result = MRZParser.find_and_parse_mrz(text_blocks)

        self.assertIsNone(result)

    def test_parse_td3_indian_passport(self):
        """Test parsing Indian passport MRZ with S/O pattern."""
        # Indian Passport with "S/O" (Son Of) in given names
        # Using shorter names to fit 44-char MRZ format
        line1 = 'P<INDSHARMA<<RAKESH<S/O<<<<<<<<<<<<X12345677'
        line2 = '8508155M3008155<<<<<<<<<<<<<<<<<<<<<<<<<<<<4'

        result = MRZParser.parse_td3(line1, line2)

        self.assertIsNotNone(result)
        self.assertEqual(result.country_code, 'IND')
        self.assertEqual(result.surname, 'SHARMA')
        # The given names should preserve the S/O pattern
        self.assertIn('RAKESH', result.given_names.upper())
        self.assertIn('S/O', result.given_names.upper())


class TestVisionLLMMRZExtraction(unittest.TestCase):
    """Test cases for vision LLM MRZ extraction integration."""

    def test_parse_vision_llm_mrz_response(self):
        """Test parsing MRZ from vision LLM response format."""
        # Mock LLM response with MRZ lines
        mock_response = """LINE1: P<USADOE<<JOHN<<<A1234567<<<<<<<<<<<<<<<<<<6
LINE2: 9001158M2506207<<<<<<<<<<<<<<<<<<<<<<<<<<<<4"""

        # Parse the response as the vision LLM would
        lines = mock_response.strip().split('\n')
        line1 = None
        line2 = None

        for line in lines:
            line = line.strip()
            if line.startswith('LINE1:'):
                line1 = line.split(':', 1)[1].strip().upper()
            elif line.startswith('LINE2:'):
                line2 = line.split(':', 1)[1].strip().upper()

        # Verify parsing worked
        self.assertIsNotNone(line1)
        self.assertIsNotNone(line2)
        self.assertEqual(len(line1), 44)
        self.assertEqual(len(line2), 44)

        # Parse MRZ using the existing parser
        result = MRZParser.parse_td3(line1, line2)

        # Verify MRZ data
        self.assertIsNotNone(result)
        self.assertTrue(result.all_valid)
        self.assertEqual(result.passport_number, 'A1234567')
        self.assertEqual(result.country_code, 'USA')
        self.assertEqual(result.surname, 'DOE')
        self.assertEqual(result.given_names, 'JOHN')

    def test_parse_vision_llm_mrz_response_indian(self):
        """Test parsing Indian passport MRZ from vision LLM response."""
        # Mock LLM response for Indian passport
        mock_response = """LINE1: P<INDSHARMA<<RAKESH<S/O<<<<<<<<<<<<X12345677
LINE2: 8508155M3008155<<<<<<<<<<<<<<<<<<<<<<<<<<<<4"""

        # Parse the response
        lines = mock_response.strip().split('\n')
        line1 = None
        line2 = None

        for line in lines:
            line = line.strip()
            if line.startswith('LINE1:'):
                line1 = line.split(':', 1)[1].strip().upper()
            elif line.startswith('LINE2:'):
                line2 = line.split(':', 1)[1].strip().upper()

        # Verify parsing worked
        self.assertIsNotNone(line1)
        self.assertIsNotNone(line2)
        self.assertEqual(len(line1), 44)
        self.assertEqual(len(line2), 44)

        # Parse MRZ
        result = MRZParser.parse_td3(line1, line2)

        # Verify MRZ data
        self.assertIsNotNone(result)
        self.assertEqual(result.country_code, 'IND')
        self.assertEqual(result.surname, 'SHARMA')
        # Verify S/O pattern is preserved
        self.assertIn('RAKESH', result.given_names.upper())
        self.assertIn('S/O', result.given_names.upper())


if __name__ == '__main__':
    unittest.main()

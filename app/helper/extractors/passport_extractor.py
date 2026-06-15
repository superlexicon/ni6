import re
import bisect
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

from app.schemas.passport_schema import PassportData
from app.helper.doctr.document_text_extractor import DocumentTextExtractor
from app.core import logger, doctr_model
from app.utils.country_code_converter import validate_iso_country_code
from app.utils.date_extractor import extract_and_remove_dates


class PassportExtractor:
    """Extract all standard fields from passport documents"""

    # Label text that should NOT be considered as name candidates
    # These are common field labels on passports that OCR might incorrectly pick as names
    LABEL_TEXT_TO_EXCLUDE = {
        'surname', 'given names', 'first name', 'last name', 'family name',
        'name', 'full name', 'nationality', 'date of birth', 'place of birth',
        'sex', 'issue date', 'expiry date', 'passport no', 'passport number',
        'country code', 'type', 'code', 'birth', 'place', 'issued', 'valid',
        # OCR misreads
        'given name', 'given name(s)'
    }

    # Minimum characters for a valid name part
    MIN_NAME_PART_LENGTH = 2

    # Minimum OCR confidence for name candidates (0.0-1.0)
    # Text with lower confidence is likely garbage
    MIN_NAME_CONFIDENCE = 0.6

    def __init__(self):
        self.logger = logger
        self.text_extractor = DocumentTextExtractor()

    async def extract(self, content: bytes, is_pdf: bool = False) -> PassportData:
        """
        Extract all standard passport fields using streamlined 4-step process:
        1. EXIF auto-rotation (handled in DocumentTextExtractor)
        2. DocTR OCR (single pass with geometry)
        3. Detect document type and country/organization
        4. Sequential schema-based field extraction

        Args:
            content: Document bytes
            is_pdf: Whether content is PDF

        Returns:
            PassportData with extracted fields and confidence scores
        """
        try:
            self.logger.info("="*80)
            self.logger.info("PASSPORT EXTRACTION - 4-STEP PROCESS")
            self.logger.info("="*80)

            # Step 1 & 2: EXIF auto-rotation + DocTR OCR (single pass)
            self.logger.info("Step 1-2: EXIF auto-rotation + DocTR OCR")
            text_blocks = await self.text_extractor.extract_text_with_geometry_enhanced(content, is_pdf=is_pdf)
            self.logger.info(f"  Extracted {len(text_blocks)} text lines with geometry and OCR confidence")

            # Step 3: Logic-based field extraction (O(N) single-pass)
            self.logger.info("Step 3: Logic-based field extraction (O(N) single-pass)")
            extracted_fields = self._extract_fields_logic_based(text_blocks)

            if not extracted_fields:
                self.logger.error("Extraction failed - no fields extracted")
                return PassportData(confidence_scores={})

            self.logger.info(f"  Extracted {len(extracted_fields)} fields")

            # Populate PassportData from extracted fields
            passport_data = PassportData()
            passport_data.passport_number = extracted_fields.get('passport_number')
            passport_data.full_name = extracted_fields.get('full_name')
            passport_data.sex = extracted_fields.get('sex')
            passport_data.nationality = extracted_fields.get('nationality')
            passport_data.date_of_birth = extracted_fields.get('date_of_birth')
            passport_data.place_of_birth = extracted_fields.get('place_of_birth')
            passport_data.date_of_issue = extracted_fields.get('date_of_issue')
            passport_data.issuing_authority = extracted_fields.get('issuing_authority')
            passport_data.date_of_expiry = extracted_fields.get('date_of_expiry')
            passport_data.passport_country = extracted_fields.get('country_code')
            passport_data.extraction_source = 'logic_based'
            passport_data.raw_data = "\n".join([block.get('text', '') for block in text_blocks])

            # Calculate confidence based on OCR confidence scores from DocTR
            field_confidences = self._calculate_field_confidences(text_blocks, extracted_fields)
            avg_confidence = sum(field_confidences.values()) / len(field_confidences) if field_confidences else 85.0

            passport_data.overall_confidence = round(avg_confidence, 2)
            passport_data.field_confidences = field_confidences
            self.logger.info(f"Overall confidence: {passport_data.overall_confidence:.2f}% (based on OCR scores)")

            # Log individual field confidences
            for field, confidence in field_confidences.items():
                self.logger.info(f"  {field}: {confidence:.2f}%")

            # Log recommendation
            if passport_data.overall_confidence >= 85:
                self.logger.info("✓ High confidence - extraction successful")
            elif passport_data.overall_confidence >= 70:
                self.logger.warning("⚠ Medium confidence - manual review recommended")
            else:
                self.logger.warning("✗ Low confidence - rescan required")

            self.logger.info("="*80)
            return passport_data

        except Exception as e:
            self.logger.error(f"Passport extraction failed: {type(e).__name__}")
            return PassportData(confidence_scores={})

    def _calculate_field_confidences(self, text_blocks: list, extracted_fields: dict) -> dict:
        """
        Calculate OCR confidence scores for each extracted field based on DocTR confidence data.

        Args:
            text_blocks: List of dicts with OCR results including confidence scores
            extracted_fields: Dict of extracted field values

        Returns:
            Dict mapping field names to their OCR confidence percentages (0-100)
        """
        field_confidences = {}

        # Create a map of text to confidence for quick lookup
        text_confidence_map = {}
        for block in text_blocks:
            text = block.get('text', '').strip()
            confidence = block.get('confidence', 0.9)  # Default to 90% if not provided
            text_confidence_map[text] = confidence

        self.logger.info(f"Created confidence map for {len(text_confidence_map)} unique text blocks")

        # Calculate confidence for each extracted field
        for field_name, field_value in extracted_fields.items():
            if field_value:
                # Find the best matching OCR confidence for this field value
                field_confidence = self._find_best_confidence_match(field_value, text_confidence_map, field_name)
                field_confidences[field_name] = field_confidence

                self.logger.debug(f"Field '{field_name}' value '{field_value}' -> confidence: {field_confidence:.2f}%")

        self.logger.info(f"Calculated OCR confidence for {len(field_confidences)} fields")
        return field_confidences

    def _find_best_confidence_match(self, field_value: str, text_confidence_map: dict, field_name: str = None) -> float:
        """
        Find the best OCR confidence match for a given field value.

        Args:
            field_value: The extracted field value
            text_confidence_map: Map of text strings to their OCR confidences
            field_name: Name of the field (used for special field handling)

        Returns:
            Confidence percentage (0-100)
        """
        # Check if this is a date field and use simple OCR confidence aggregation
        if field_name and 'date' in field_name.lower():
            return self._calculate_date_confidence_simple(field_value, text_confidence_map)

        if not field_value or not text_confidence_map:
            return 85.0  # Default fallback

        field_value_clean = field_value.strip().upper()
        best_confidence = 0.0

        # Try exact match first
        if field_value_clean in text_confidence_map:
            confidence = text_confidence_map[field_value_clean]
            return confidence * 100  # Convert from decimal to percentage

        # Try partial matches for field values that might be substrings
        for text, confidence in text_confidence_map.items():
            text_clean = text.strip().upper()

            # Check if field value is contained in OCR text or vice versa
            if (field_value_clean in text_clean or text_clean in field_value_clean):
                # Use fuzzy matching to find best similarity
                similarity = self._calculate_text_similarity(field_value_clean, text_clean)
                if similarity > 0.7:  # 70% similarity threshold
                    weighted_confidence = confidence * similarity
                    if weighted_confidence > best_confidence:
                        best_confidence = weighted_confidence

        # If we found a good match, return it, otherwise use default
        return best_confidence * 100 if best_confidence > 0 else 85.0

    def _calculate_date_confidence_simple(self, date_value: str, text_confidence_map: dict) -> float:
        """
        Calculate confidence for date fields by finding best OCR confidence matches.
        Simple approach: find the highest confidence OCR text that contains date components.

        Args:
            date_value: The extracted date value (e.g., "10 NOV 1982")
            text_confidence_map: Map of OCR text to their confidence scores

        Returns:
            Confidence percentage (0-100)
        """
        if not date_value or not text_confidence_map:
            return 85.0  # Keep existing fallback

        date_upper = date_value.upper()
        best_confidence = 0.0

        # Find OCR text blocks that contain parts of our date
        for ocr_text, confidence in text_confidence_map.items():
            if any(word in ocr_text.upper() for word in date_upper.split() if len(word) > 1):
                # This OCR block contains some part of our date
                if confidence > best_confidence:
                    best_confidence = confidence

        return best_confidence * 100 if best_confidence > 0 else 85.0

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity using simple character matching.

        Args:
            text1: First text string
            text2: Second text string

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not text1 or not text2:
            return 0.0

        # Simple character-based similarity (more sophisticated methods could be used)
        common_chars = set(text1) & set(text2)
        total_chars = set(text1) | set(text2)

        if not total_chars:
            return 0.0

        return len(common_chars) / len(total_chars)

    def _parse_date_string(self, date_str: str) -> Optional[str]:
        """Parse various date formats and return YYYY-MM-DD"""
        from datetime import datetime
        import re

        date_str = date_str.strip().upper()

        # Month name mapping
        months = {
            'JAN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04',
            'MAY': '05', 'JUN': '06', 'JUL': '07', 'AUG': '08',
            'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12'
        }

        try:
            # Try DD MMM YYYY format (e.g., "12 NOV 1992")
            match = re.match(r'(\d{1,2})\s*([A-Z]{3})\s*(\d{4})', date_str)
            if match:
                day, month_name, year = match.groups()
                month = months.get(month_name)
                if month:
                    return f"{year}-{month}-{day.zfill(2)}"

            # Try DD/MM/YYYY or DD-MM-YYYY
            match = re.match(r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', date_str)
            if match:
                day, month, year = match.groups()
                return f"{year}-{month.zfill(2)}-{day.zfill(2)}"

            # Try YYYY-MM-DD or YYYY/MM/DD
            match = re.match(r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})', date_str)
            if match:
                year, month, day = match.groups()
                return f"{year}-{month.zfill(2)}-{day.zfill(2)}"

        except Exception as e:
            self.logger.warning(f"Failed to parse date '{date_str}': {str(e)}")

        return None

    # ========================================================================
    # LOGIC-BASED EXTRACTION (O(N) single-pass with early exit)
    # ========================================================================

    def _is_type(self, text: str) -> bool:
        """Check if text is a passport type (1-2 chars starting with P/D/S)."""
        text = text.strip().upper()
        if len(text) < 1 or len(text) > 2:
            return False
        return text[0] in ('P', 'D', 'S') and text.isalpha()

    def _is_country_code(self, text: str) -> bool:
        """Check if text is a valid 3-letter country code."""
        text = text.strip().upper()
        return len(text) == 3 and validate_iso_country_code(text)

    def _is_passport_number(self, text: str) -> bool:
        """Check if text is a passport number (8-9 alphanumeric chars with ≥6 digits)."""
        text = text.strip().upper()
        if len(text) < 8 or len(text) > 9:
            return False
        if not text.isalnum():
            return False
        digit_count = sum(c.isdigit() for c in text)
        return digit_count >= 6

    def _is_sex(self, text: str) -> bool:
        """Check if text is a sex indicator (standalone M or F)."""
        text = text.strip().upper()
        return text == 'M' or text == 'F'

    def _is_name_label(self, text: str, country_code: Optional[str] = None) -> bool:
        """
        Check if text is a name-related label.

        Uses country-specific labels when country is known, falls back
        to global patterns otherwise.

        Args:
            text: Text to check
            country_code: ISO 2-letter country code (e.g., "SG", "MM")

        Returns:
            True if text matches a name label
        """
        if not text:
            return False

        # Exclude "nationality" - it's a country field, not a name field
        text_lower = text.lower()
        if 'nationality' in text_lower:
            return False

        # Check for "name" substring (fallback behavior)
        return 'name' in text_lower

    def _is_date_text(self, text: str) -> Optional[datetime]:
        """
        Check if text is a date and parse it.
        Returns datetime object if valid, None otherwise.
        Supports: DD MMM YYYY, DD/MM/YYYY, DD-MM-YYYY, YYYY-MM-DD
        """
        text = text.strip().upper()

        months = {
            'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4,
            'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8,
            'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
        }

        try:
            # Try DD MMM YYYY format (e.g., "12 NOV 1992")
            match = re.match(r'^(\d{1,2})\s*([A-Z]{3})\s*(\d{4})$', text)
            if match:
                day, month_name, year = match.groups()
                month = months.get(month_name)
                if month:
                    return datetime(int(year), month, int(day))

            # Try DD/MM/YYYY or DD-MM-YYYY
            match = re.match(r'^(\d{1,2})[/-](\d{1,2})[/-](\d{4})$', text)
            if match:
                day, month, year = match.groups()
                return datetime(int(year), int(month), int(day))

            # Try YYYY-MM-DD or YYYY/MM/DD
            match = re.match(r'^(\d{4})[/-](\d{1,2})[/-](\d{1,2})$', text)
            if match:
                year, month, day = match.groups()
                return datetime(int(year), int(month), int(day))

        except (ValueError, TypeError):
            pass

        return None

    def _format_date(self, dt: datetime) -> str:
        """Format datetime to DD MMM YYYY string."""
        months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                  'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
        return f"{dt.day} {months[dt.month - 1]} {dt.year}"

    def _extract_fields_logic_based(self, text_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract passport fields using logic-based single-pass algorithm.

        Algorithm:
        1. Single pass through OCR blocks
        2. Check in priority order: type → country → passport → alt_id → "name" → dates → sex
        3. Skip already-found fields (early exit per field)
        4. Post-pass: resolve dates (sort) and names (x-proximity)

        Returns:
            Dict with extracted fields
        """
        self.logger.info("=" * 80)
        self.logger.info("LOGIC-BASED PASSPORT EXTRACTION (O(N) single-pass)")
        self.logger.info("=" * 80)

        # Initialize state
        found = {
            'type': None,
            'country_code': None,
            'passport_number': None,
            'alternative_id': None,
            'sex': None
        }

        # Dates stored as (datetime, original_text) tuples, kept sorted
        dates: List[Tuple[datetime, str]] = []

        # Name labels: (x1, y2, label_text) - store bottom-left corner for matching
        name_labels: List[Tuple[float, float, str]] = []

        # Country-specific separate surname and given names labels
        # For countries like India that have separate "Surname" and "Given Names" labels
        surname_labels: List[Tuple[float, float, str]] = []
        given_names_labels: List[Tuple[float, float, str]] = []

        # Unprocessed blocks for name extraction: (text, x_coord, y_coord, confidence)
        unprocessed: List[Tuple[str, float, float, float]] = []

        # Track processed texts to avoid double-counting
        processed_texts = set()

        self.logger.info(f"Processing {len(text_blocks)} OCR blocks...")

        # SINGLE PASS through all OCR blocks
        for block in text_blocks:
            text = block.get('text', '').strip()
            x = block.get('x1', 0)
            y = block.get('y1', 0)

            if not text:
                continue

            matched = False
            text_upper = text.upper()

            # Check in priority order, skip already-found fields

            # 1. Type (1-2 chars starting with P/D/S)
            if not found['type'] and self._is_type(text):
                found['type'] = text_upper
                processed_texts.add(text)
                matched = True
                self.logger.info(f"  ✓ Type: {text_upper}")

            # 2. Country code (3-letter ISO code)
            elif not found['country_code'] and self._is_country_code(text):
                found['country_code'] = text_upper
                processed_texts.add(text)
                matched = True
                self.logger.info(f"  ✓ Country: {text_upper}")

            # 3. Passport number (8-9 alphanumeric with ≥6 digits)
            elif not found['passport_number'] and self._is_passport_number(text):
                found['passport_number'] = text_upper
                processed_texts.add(text)
                matched = True
                self.logger.info(f"  ✓ Passport Number: {text_upper}")

            # 4. Alternative ID (second passport-number-like string)
            elif found['passport_number'] and not found['alternative_id'] and self._is_passport_number(text):
                found['alternative_id'] = text_upper
                processed_texts.add(text)
                matched = True
                self.logger.info(f"  ✓ Alternative ID: {text_upper}")

            # 5. Name labels - collect for post-pass matching (simplified, Qwen handles complex cases)
            elif self._is_name_label(text, found.get('country_code')):
                y1 = y  # Use top of label box for comparison
                name_labels.append((x, y1, text))
                processed_texts.add(text)
                matched = True
                self.logger.info(f"  ✓ Name label at (x1={x:.3f}, y1={y1:.3f}): {text}")

            # 6. Dates (max 3, auto-sorted)
            elif len(dates) < 3:
                parsed_date = self._is_date_text(text)
                if parsed_date:
                    # Insert sorted by date
                    bisect.insort(dates, (parsed_date, text))
                    processed_texts.add(text)
                    matched = True
                    self.logger.info(f"  ✓ Date: {text} -> {parsed_date.date()}")

            # 7. Sex (standalone M or F)
            if not matched and not found['sex'] and self._is_sex(text):
                found['sex'] = text_upper
                processed_texts.add(text)
                matched = True
                self.logger.info(f"  ✓ Sex: {text_upper}")

            # Track unprocessed blocks for name extraction (include confidence)
            if not matched:
                confidence = block.get('confidence', 0.9)
                unprocessed.append((text, x, y, confidence))

        # ============================================================
        # SECOND PASS: Substring matching for merged OCR values
        # ============================================================
        self.logger.info("-" * 40)
        self.logger.info("SECOND PASS: Extracting from merged OCR values...")

        # Track strings that get fully processed in second pass
        still_unprocessed = []

        for text, x, y, confidence in unprocessed:
            remaining_text = text

            # STEP 1: Extract embedded dates first (dates have internal spaces)
            if len(dates) < 3:
                embedded_dates, remaining_text = extract_and_remove_dates(text)
                for dt_tuple in embedded_dates:
                    if len(dates) < 3:
                        bisect.insort(dates, dt_tuple)
                        self.logger.info(f"  ✓ Embedded date from '{text}': {dt_tuple[1]} -> {dt_tuple[0].date()}")

            # STEP 2: Split remaining string by spaces, check each word
            words = remaining_text.split()
            word_matched_any = False

            for word in words:
                word_upper = word.upper()
                matched_word = False

                # Same if/elif priority as first pass
                if not found['type'] and self._is_type(word):
                    found['type'] = word_upper
                    matched_word = True
                    self.logger.info(f"  ✓ Type from '{text}': {word_upper}")

                elif not found['country_code'] and self._is_country_code(word):
                    found['country_code'] = word_upper
                    matched_word = True
                    self.logger.info(f"  ✓ Country from '{text}': {word_upper}")

                elif not found['passport_number'] and self._is_passport_number(word):
                    found['passport_number'] = word_upper
                    matched_word = True
                    self.logger.info(f"  ✓ Passport number from '{text}': {word_upper}")

                elif found['passport_number'] and not found['alternative_id'] and self._is_passport_number(word):
                    found['alternative_id'] = word_upper
                    matched_word = True
                    self.logger.info(f"  ✓ Alternative ID from '{text}': {word_upper}")

                elif not found['sex'] and self._is_sex(word):
                    found['sex'] = word_upper
                    matched_word = True
                    self.logger.info(f"  ✓ Sex from '{text}': {word_upper}")

                if matched_word:
                    word_matched_any = True

            # Keep track of strings that still have unmatched content (for name extraction)
            if not word_matched_any and remaining_text.strip():
                still_unprocessed.append((remaining_text, x, y, confidence))

        # Update unprocessed list for name extraction
        unprocessed = still_unprocessed

        # POST-PASS: Resolve dates
        self.logger.info("-" * 40)
        self.logger.info("Resolving dates (earliest=DOB, latest=Expiry, middle=Issue)...")

        if len(dates) >= 1:
            found['date_of_birth'] = self._format_date(dates[0][0])
            self.logger.debug(f"  DOB (earliest): {found['date_of_birth']}")

        if len(dates) >= 3:
            found['date_of_issue'] = self._format_date(dates[1][0])
            found['date_of_expiry'] = self._format_date(dates[2][0])
            self.logger.info(f"  Issue (middle): {found['date_of_issue']}")
            self.logger.info(f"  Expiry (latest): {found['date_of_expiry']}")
        elif len(dates) == 2:
            # Only 2 dates: assume DOB and Expiry
            found['date_of_expiry'] = self._format_date(dates[1][0])
            self.logger.info(f"  Expiry (latest): {found['date_of_expiry']}")

        # POST-PASS: Name extraction (simplified, Qwen handles complex cases)
        country_code = found.get('country_code')
        if country_code and (surname_labels or given_names_labels):
            self.logger.info("-" * 40)
            self.logger.info(f"{country_code} PASSPORT: Matching surname and given names separately...")

            def is_label_text_for_country(text: str) -> bool:
                """Check if text is a label or contains label keywords."""
                text_lower = text.lower().strip()
                if text_lower in self.LABEL_TEXT_TO_EXCLUDE:
                    return True
                for label in self.LABEL_TEXT_TO_EXCLUDE:
                    if label in text_lower:
                        return True
                return False

            # Filter candidates for name extraction
            candidates = [
                (text, x, y, conf) for text, x, y, conf in unprocessed
                if not is_label_text_for_country(text) and conf >= self.MIN_NAME_CONFIDENCE
            ]
            self.logger.info(f"  Filtered to {len(candidates)} name candidates")

            surname = None
            given_names = None

            # Match surname label to value
            if surname_labels and candidates:
                for label_x, label_y, label_text in surname_labels:
                    candidates_below = [
                        (text, x, y, conf) for text, x, y, conf in candidates
                        if y > label_y - 0.05
                    ]
                    if candidates_below:
                        candidates_below.sort(key=lambda t: len(t[0].split()), reverse=True)
                        closest = min(candidates_below, key=lambda t: abs(t[1] - label_x) + abs(t[2] - label_y))
                        surname = closest[0]
                        candidates.remove(closest)
                        self.logger.info(f"  Surname '{label_text}' → '{surname}'")
                        break

            # Match given names label to value
            if given_names_labels and candidates:
                for label_x, label_y, label_text in given_names_labels:
                    candidates_below = [
                        (text, x, y, conf) for text, x, y, conf in candidates
                        if y > label_y - 0.05
                    ]
                    if candidates_below:
                        candidates_below.sort(key=lambda t: len(t[0].split()), reverse=True)
                        closest = min(candidates_below, key=lambda t: abs(t[1] - label_x) + abs(t[2] - label_y))
                        given_names = closest[0]
                        candidates.remove(closest)
                        self.logger.info(f"  Given Names '{label_text}' → '{given_names}'")
                        break

            # Concatenate given_names + surname for full_name
            if given_names and surname:
                found['full_name'] = f"{given_names} {surname}"
            elif given_names:
                found['full_name'] = given_names
            elif surname:
                found['full_name'] = surname
            self.logger.info(f"  ✓ {country_code} passport full_name: {found['full_name']}")

        # POST-PASS: Match name values to labels using y1→y1 spatial proximity (non-Indian passports)
        elif name_labels:
            self.logger.info("-" * 40)
            self.logger.info(f"Matching {len(name_labels)} name label(s) to values using y1→y1 proximity...")

            name_parts = []

            def is_label_text(text: str) -> bool:
                """Check if text is a label or contains label keywords."""
                text_lower = text.lower().strip()

                # Exact match check
                if text_lower in self.LABEL_TEXT_TO_EXCLUDE:
                    return True

                # Substring check - filter out candidates containing label keywords
                for label in self.LABEL_TEXT_TO_EXCLUDE:
                    if label in text_lower:
                        return True

                return False

            # Filter candidates: exclude label text, garbage, and low-confidence OCR
            # Include only candidates with confidence >= MIN_NAME_CONFIDENCE
            candidates = [
                (text, x, y, conf) for text, x, y, conf in unprocessed
                if not is_label_text(text) and conf >= self.MIN_NAME_CONFIDENCE
            ]
            self.logger.info(f"  Filtered to {len(candidates)} candidates (excluded label text, confidence >= {self.MIN_NAME_CONFIDENCE})")

            for label_x1, label_y1, label_text in name_labels:
                if not candidates:
                    break

                # Filter candidates to only those BELOW this label (y1 > label's y1)
                # with some tolerance for OCR geometry variations
                candidates_below = [
                    (text, x, y, conf) for text, x, y, conf in candidates
                    if y > label_y1 - 0.05  # Small tolerance for OCR inaccuracies
                ]

                if not candidates_below:
                    self.logger.info(f"  No candidates below '{label_text}', skipping...")
                    continue

                # Find closest value where x1 aligns and y is below label
                # Delta = |x1_value - x1_label| + |y1_value - y1_label|
                # Prefer multi-word candidates by sorting them first
                candidates_below.sort(key=lambda t: len(t[0].split()), reverse=True)
                closest = min(candidates_below, key=lambda t: abs(t[1] - label_x1) + abs(t[2] - label_y1))
                delta = abs(closest[1] - label_x1) + abs(closest[2] - label_y1)
                name_parts.append(closest[0])
                candidates.remove(closest)
                self.logger.info(f"  '{label_text}' → '{closest[0]}' (delta={delta:.3f}, conf={closest[3]:.2f}, words={len(closest[0].split())})")

            if name_parts:
                found['full_name'] = ' '.join(name_parts)
                self.logger.debug(f"  Full name: {found['full_name']}")

        # Set nationality from country code
        found['nationality'] = found.get('country_code')

        # Log summary (DEBUG level to avoid PII in production logs)
        self.logger.debug("=" * 80)
        self.logger.debug("EXTRACTION SUMMARY:")
        for field, value in found.items():
            if value:
                self.logger.debug(f"  {field}: {value}")
        self.logger.debug("=" * 80)

        return found

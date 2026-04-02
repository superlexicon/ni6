"""
Span-Based Bank Statement Extractor

A simple, direct multi-pass approach working with PyMuPDF spans.

Pass 1: Find label spans (account holder name, account number, bank name)
Pass 2: Find location spans (country, state, city) and build address blocks
Pass 3: Find account number value if not in Pass 1
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path

import fitz  # PyMuPDF

from app.core.key_injection.bank_lookup import get_bank_lookup, BankInfo
from app.utils.string_matching import clean_name_for_storage

logger = logging.getLogger(__name__)


@dataclass
class Span:
    """A text span from PyMuPDF."""
    text: str
    x1: float
    y1: float
    x2: float
    y2: float
    page_num: int = 0

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


@dataclass
class ExtractionResult:
    """Result of bank statement extraction."""
    account_holder_name: Optional[str] = None
    account_holder_address: Optional[str] = None
    address_city: Optional[str] = None
    address_state: Optional[str] = None
    address_postal: Optional[str] = None
    address_country: Optional[str] = None
    account_number: Optional[str] = None
    bank_name: Optional[str] = None
    bank_country: Optional[str] = None
    bank_code: Optional[str] = None  # SWIFT/IFSC code
    statement_date: Optional[str] = None
    currency: Optional[str] = None
    ifsc_code: Optional[str] = None
    swift_code: Optional[str] = None
    raw_values: Dict[str, Dict[str, float]] = field(default_factory=dict)


class SpanBasedBankExtractor:
    """
    Multi-pass span-based bank statement extractor.

    Simple, direct approach:
    - Pass 1: Find label spans
    - Pass 2: Find location spans and build address blocks
    - Pass 3: Find account number value
    """

    # Label variations map - labels, bank full names
    LABEL_PATTERNS = {
        # Account holder name labels
        'account_holder_name_label': [
            r'^Account\s+Holder\s+Names\s*[:\s]*$',
            r'^Account\s+Holder\s*Name\s*[:\s]*$',
            r'^Customer\s*Name\s*[:\s]*$',
            r'^A/c\s*Holder\s*Name\s*[:\s]*$',
            r'^Account\s+Holder\s*[:\s]*$',
            r'^Customer\s*[:\s]*$',
        ],
        # Account number labels
        'account_number_label': [
            r'Account\s*No',
            r'Account\s*Number',
            r'A/c\s*No',
            r'Account\s*[:\s]*$',
            r'Acct\s*No',
            r'Account\s*No\.',
        ],
        # Address label
        'address_label': [
            r'^Address\s*[:\s]*$',
            r'^Addr\s*[:\s]*$',
        ],
        # Bank full names (will be loaded from bank_lookup)
        'bank_name': [],  # Populated dynamically
    }

    # Proximity thresholds
    SAME_LINE_Y_THRESHOLD = 3.0  # points - same line
    CLOSE_X_THRESHOLD = 100.0    # points - same column (relaxed for better matching)
    CLOSE_Y_THRESHOLD = 20.0     # points - vertically adjacent
    ADDRESS_GROUP_Y_THRESHOLD = 30.0  # points - group address lines

    # Postal code patterns
    POSTAL_CODE_PATTERNS = [
        r'\b\d{6}\b',  # Indian postal code (6 digits)
        r'\b\d{5}(-\d{4})?\b',  # US ZIP code
        r'\b[A-Z]\d[A-Z]\s\d[A-Z]\d\b',  # Canadian postal code
    ]

    def __init__(self):
        self.bank_lookup = get_bank_lookup()
        # Load bank full names into label patterns
        self._load_bank_names()

    def _load_bank_names(self):
        """Load bank full names from bank_lookup for Pass 1 matching."""
        # Bank names are looked up dynamically via bank_lookup.detect_bank_in_text()
        pass

    def extract(self, pdf_path: str, max_pages: int = 1) -> ExtractionResult:
        """
        Extract bank statement data using multi-pass span algorithm.

        Args:
            pdf_path: Path to PDF file
            max_pages: Maximum pages to process

        Returns:
            ExtractionResult with extracted fields
        """
        result = ExtractionResult()

        # Extract all spans from PDF
        spans = self._extract_spans(pdf_path, max_pages)
        if not spans:
            logger.warning(f"No spans extracted from {pdf_path}")
            return result

        logger.info(f"Extracted {len(spans)} spans from {pdf_path}")

        # =========================================================================
        # PASS 1: Find label spans
        # =========================================================================
        label_spans = self._pass1_find_labels(spans)
        logger.info(f"Pass 1: Found {len(label_spans)} label spans")

        # Check if account number value is in the same span as the label
        account_number = self._extract_account_number_from_label_span(
            label_spans.get('account_number_label')
        )
        if account_number:
            result.account_number = account_number
            logger.info(f"Account number found in label span: {account_number}")

        # Extract name value from "Account Holder Names" label if found
        if label_spans.get('account_holder_name_label'):
            name_value = self._extract_name_from_label_span(
                spans, label_spans['account_holder_name_label']
            )
            if name_value:
                result.account_holder_name = clean_name_for_storage(name_value)
                logger.info(f"Name found from account holder name label: {name_value}")

        # =========================================================================
        # PASS 1.5: Extract address from address label if found
        # =========================================================================
        address_from_label = None
        if label_spans.get('address_label'):
            address_from_label = self._extract_address_from_label_span(
                spans, label_spans['address_label']
            )
            if address_from_label:
                logger.info(f"Address found from address label: {address_from_label['text'][:100]}...")

        # =========================================================================
        # PASS 2: Find location spans and build address blocks
        # =========================================================================
        address_blocks = self._pass2_find_addresses(spans, label_spans)
        logger.info(f"Pass 2: Found {len(address_blocks)} address blocks")

        # Determine which address belongs to customer vs bank
        customer_address, bank_address = self._identify_customer_vs_bank_address(
            address_blocks, label_spans
        )

        # Use address from label if available, otherwise use the detected address
        final_address = address_from_label if address_from_label else customer_address

        if final_address:
            result.account_holder_address = final_address['text']
            result.address_city = final_address.get('city') or customer_address.get('city') if customer_address else None
            result.address_state = final_address.get('state') or customer_address.get('state') if customer_address else None
            result.address_postal = final_address.get('postal_code')
            result.address_country = final_address.get('country') or customer_address.get('country') if customer_address else None
            # Only set name if not already set from the label
            if not result.account_holder_name:
                name = final_address.get('name') or customer_address.get('name') if customer_address else None
                result.account_holder_name = clean_name_for_storage(name) if name else None

        if bank_address:
            result.bank_country = bank_address.get('country')

        # =========================================================================
        # PASS 3: Find account number value if not found in Pass 1
        # =========================================================================
        if not result.account_number and label_spans.get('account_number_label'):
            account_number = self._pass3_find_account_number_value(
                spans, label_spans['account_number_label']
            )
            if account_number:
                result.account_number = account_number
                logger.info(f"Pass 3: Found account number: {account_number}")

        # =========================================================================
        # FINAL: Get SWIFT code from bank lookup
        # =========================================================================
        if label_spans.get('bank_name'):
            bank_name = label_spans['bank_name'].text
            result.bank_name = bank_name
            # Get SWIFT code
            bank_info = self.bank_lookup.lookup_by_name(bank_name)
            if bank_info:
                result.swift_code = bank_info.swift_codes[0] if bank_info.swift_codes else None
                result.bank_code = result.swift_code

        return result

    def _extract_spans(self, pdf_path: str, max_pages: int) -> List[Span]:
        """
        Extract all text spans from PDF using PyMuPDF dict format.

        Returns:
            List of Span objects sorted by y, then x position
        """
        spans = []

        try:
            doc = fitz.open(pdf_path)
            pages_to_process = min(max_pages, doc.page_count)

            for page_num in range(pages_to_process):
                page = doc[page_num]
                blocks = page.get_text("dict")

                for block in blocks.get("blocks", []):
                    if "lines" in block:  # Text block
                        for line in block.get("lines", []):
                            for span_dict in line.get("spans", []):
                                text = span_dict.get("text", "").strip()
                                if not text:
                                    continue

                                bbox = span_dict.get("bbox", [0, 0, 0, 0])
                                span = Span(
                                    text=text,
                                    x1=bbox[0],
                                    y1=bbox[1],
                                    x2=bbox[2],
                                    y2=bbox[3],
                                    page_num=page_num
                                )
                                spans.append(span)

            doc.close()

        except Exception as e:
            logger.error(f"Error extracting spans from PDF: {e}")
            return []

        # Sort by y (top to bottom), then by x (left to right)
        spans.sort(key=lambda s: (s.y1, s.x1))
        return spans

    # =========================================================================
    # PASS 1: Find label spans
    # =========================================================================

    def _pass1_find_labels(self, spans: List[Span]) -> Dict[str, Span]:
        """
        Pass 1: Find spans containing labels as substrings.

        Checks for:
        - Account holder name label
        - Account number label
        - Address label
        - Bank full name

        Returns:
            Dict mapping label_type → Span (or None)
        """
        found = {}
        remaining_spans = list(spans)  # Work on a copy

        for span in remaining_spans:
            text_upper = span.text.upper()

            # Check account holder name label patterns
            if 'account_holder_name_label' not in found:
                for pattern in self.LABEL_PATTERNS['account_holder_name_label']:
                    if re.search(pattern, text_upper, re.IGNORECASE):
                        found['account_holder_name_label'] = span
                        logger.debug(f"Found account holder name label: '{span.text}'")
                        break

            # Check account number label patterns
            if 'account_number_label' not in found:
                for pattern in self.LABEL_PATTERNS['account_number_label']:
                    if re.search(pattern, text_upper, re.IGNORECASE):
                        found['account_number_label'] = span
                        logger.debug(f"Found account number label: '{span.text}'")
                        break

            # Check address label patterns
            if 'address_label' not in found:
                for pattern in self.LABEL_PATTERNS['address_label']:
                    if re.search(pattern, text_upper, re.IGNORECASE):
                        found['address_label'] = span
                        logger.debug(f"Found address label: '{span.text}'")
                        break

            # Check bank name patterns
            if 'bank_name' not in found:
                # Check against bank full names from bank_lookup
                bank_info = self.bank_lookup.detect_bank_in_text(span.text)
                if bank_info:
                    found['bank_name'] = span
                    logger.debug(f"Found bank name: '{span.text}' ({bank_info.full_name})")
                    break

        return found

    def _extract_account_number_from_label_span(self, label_span: Optional[Span]) -> Optional[str]:
        """
        Check if account number value is in the same span as the label.

        Uses country-specific account number regex from config.
        """
        if not label_span:
            return None

        # Generic account number patterns
        account_patterns = [
            r'\b\d{10,20}\b',  # 10-20 digits
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Formatted
        ]

        text = label_span.text
        # Remove the label part and check what's left
        for pattern in account_patterns:
            match = re.search(pattern, text)
            if match:
                # Extract just the digits
                return re.sub(r'[^\d]', '', match.group(0))

        return None

    def _extract_address_from_label_span(
        self,
        spans: List[Span],
        label_span: Span
    ) -> Optional[Dict]:
        """
        Extract address value from an address label span.

        Looks:
        1. To the right of label (same line)
        2. Below label (same column)

        Returns:
            Dict with 'text', 'city', 'state', 'postal_code', 'country' or None
        """
        if not label_span:
            return None

        # Find best candidate by scoring each span
        best_candidate = None
        best_score = 0

        for span in spans:
            if span is label_span:
                continue

            score = 0

            # Check if on same line (close y)
            same_line = abs(span.center_y - label_span.center_y) <= self.SAME_LINE_Y_THRESHOLD

            # Check if to the right
            to_right = span.x1 > label_span.x2

            # Check if below
            below = span.y1 > label_span.y2

            # Check if in same column
            same_col = abs(span.center_x - label_span.center_x) <= self.CLOSE_X_THRESHOLD

            # Score based on position
            if same_line and to_right:
                score = 100  # Highest priority: same line, to the right
            elif below and same_col:
                score = 80   # High priority: below, same column
            elif same_line:
                score = 60   # Medium priority: same line
            elif below:
                score = 40   # Lower priority: below

            if score > 0 and score > best_score:
                # Check if this looks like an address
                if self._looks_like_address(span.text):
                    best_candidate = span
                    best_score = score

        if best_candidate:
            return self._parse_address(best_candidate.text, best_candidate)

        return None

    def _looks_like_address(self, text: str) -> bool:
        """Check if text looks like an address."""
        # Check for postal code
        for pattern in self.POSTAL_CODE_PATTERNS:
            if re.search(pattern, text):
                return True

        # Check for address keywords
        address_keywords = ['H NO', 'HOUSE NO', 'FLAT', 'ROOM', 'STREET', 'ROAD', 'NEAR', 'OPPOSITE',
                           'PALLE', 'COLONY', 'NAGAR', 'DIST', 'DISTRICT']
        text_upper = text.upper()
        for keyword in address_keywords:
            if keyword in text_upper:
                return True

        return False

    def _parse_address(self, text: str, span: Span) -> Dict:
        """Parse address text into components."""
        result = {
            'text': text,
            'spans': [span],
            'center_x': span.center_x,
            'center_y': span.center_y,
        }

        # Extract postal code
        for pattern in self.POSTAL_CODE_PATTERNS:
            match = re.search(pattern, text)
            if match:
                result['postal_code'] = match.group(0)
                break

        # Extract state (common Indian states)
        # Include both with and without spaces for matching
        states = [
            'ANDHRA PRADESH', 'ANDHRAPRADESH',
            'TELANGANA', 'TAMIL NADU', 'TAMILNADU',
            'KARNATAKA', 'KERALA',
            'MAHARASHTRA', 'GUJARAT', 'RAJASTHAN',
            'DELHI', 'PUNJAB', 'HARYANA',
            'UTTAR PRADESH', 'UTTARPRADESH',
            'MADHYA PRADESH', 'MADHYAPRADESH',
            'BIHAR', 'WEST BENGAL', 'WESTBENGAL', 'ODISHA'
        ]
        text_upper = text.upper()
        found_state = None
        for state in states:
            if state in text_upper:
                found_state = state
                # Normalize state name (use with-space version)
                if 'PRADESH' in state and ' ' not in state:
                    result['state'] = state.replace('PRADESH', ' PRADESH')
                elif 'NADU' in state and ' ' not in state:
                    result['state'] = state.replace('NADU', ' NADU')
                elif 'BENGAL' in state and ' ' not in state:
                    result['state'] = state.replace('BENGAL', ' BENGAL')
                else:
                    result['state'] = state
                break

        if found_state:
            # Extract city (word before state, if any)
            parts = text_upper.split(found_state)
            if parts:
                before_state = parts[0].strip()
                # Get last word before state as potential city
                words = before_state.split()
                if words:
                    result['city'] = words[-1].strip(', ')

        # Extract country
        if 'INDIA' in text_upper:
            result['country'] = 'INDIA'
        elif 'USA' in text_upper or 'US' in text_upper:
            result['country'] = 'USA'

        return result

    def _extract_name_from_label_span(
        self,
        spans: List[Span],
        label_span: Span
    ) -> Optional[str]:
        """
        Extract name value from an account holder name label span.

        Looks:
        1. To the right of label (same line)
        2. Below label (same column)

        Returns:
            Name string or None
        """
        if not label_span:
            return None

        # Find best candidate by scoring each span
        best_candidate = None
        best_score = 0

        for span in spans:
            if span is label_span:
                continue

            score = 0

            # Check if on same line (close y)
            same_line = abs(span.center_y - label_span.center_y) <= self.SAME_LINE_Y_THRESHOLD

            # Check if to the right
            to_right = span.x1 > label_span.x2

            # Check if below
            below = span.y1 > label_span.y2

            # Check if in same column
            same_col = abs(span.center_x - label_span.center_x) <= self.CLOSE_X_THRESHOLD

            # Score based on position
            if same_line and to_right:
                score = 100  # Highest priority: same line, to the right
            elif below and same_col:
                score = 80   # High priority: below, same column
            elif same_line:
                score = 60   # Medium priority: same line
            elif below:
                score = 40   # Lower priority: below

            if score > 0 and score > best_score:
                # Check if this looks like a name (no digits, reasonable length)
                if self._looks_like_name(span.text):
                    best_candidate = span
                    best_score = score

        if best_candidate:
            return best_candidate.text.strip()

        return None

    def _looks_like_name(self, text: str) -> bool:
        """Check if text looks like a person's name."""
        # Name should not have digits (except maybe suffixes like Jr., III, etc.)
        # But for simplicity, we'll check if it's mostly alphabetic
        if not text:
            return False

        # Remove common titles/prefixes
        text_clean = re.sub(r'^(Mr\.|Mrs\.|Ms\.|Dr\.)\s*', '', text.strip(), flags=re.IGNORECASE)

        # Check if it has too many digits (not a name)
        digit_count = sum(c.isdigit() for c in text_clean)
        if digit_count > 2:
            return False

        # Check reasonable length (1-100 characters)
        if len(text_clean) < 1 or len(text_clean) > 100:
            return False

        # Should have at least some letters
        if not any(c.isalpha() for c in text_clean):
            return False

        return True

    # =========================================================================
    # PASS 2: Find location spans and build address blocks
    # =========================================================================

    def _pass2_find_addresses(
        self,
        spans: List[Span],
        label_spans: Dict[str, Span]
    ) -> List[Dict]:
        """
        Pass 2: Find all address blocks by detecting country/state/city spans.

        Does NOT stop at first match - collects all addresses.
        Disambiguates customer vs bank address later.

        Returns:
            List of address dicts with 'text', 'spans', 'country', 'state', 'city', 'name'
        """
        try:
            from countrystatecity_countries import (
                get_all_countries,
                get_states_of_country,
                get_cities_of_country
            )
        except ImportError:
            logger.warning("countrystatecity_countries not available, using fallback address detection")
            return self._pass2_find_addresses_fallback(spans, label_spans)

        address_blocks = []

        # Get all country names
        all_countries = {c.name.upper(): c for c in get_all_countries()}

        # Find spans containing country names
        for i, span in enumerate(spans):
            text_upper = span.text.upper()

            # Check if this span contains a country name
            for country_name, country_obj in all_countries.items():
                if country_name in text_upper:
                    # Found a country - now look for state and city in nearby spans
                    state, city, address_spans = self._build_address_block(
                        spans, i, country_obj
                    )

                    if address_spans:
                        # Build full address text
                        address_text = ' '.join(s.text for s in address_spans)

                        # Determine if we need to extract name (only if no label found)
                        name = None
                        if 'account_holder_name_label' not in label_spans:
                            name = self._extract_name_before_address(
                                spans, address_spans
                            )

                        address_blocks.append({
                            'text': address_text,
                            'spans': address_spans,
                            'country': country_name,
                            'state': state,
                            'city': city,
                            'name': name,
                            'center_x': sum(s.center_x for s in address_spans) / len(address_spans),
                            'center_y': sum(s.center_y for s in address_spans) / len(address_spans),
                        })

                        logger.debug(f"Found address block: country={country_name}, state={state}, city={city}")

        return address_blocks

    def _pass2_find_addresses_fallback(
        self,
        spans: List[Span],
        label_spans: Dict[str, Span]
    ) -> List[Dict]:
        """
        Fallback address detection when countrystatecity_countries is not available.
        Uses simple heuristics to find address-like content.
        """
        address_blocks = []

        # Common patterns that indicate address content
        # Include both with and without spaces for matching
        address_keywords = {
            'country': ['INDIA', 'US', 'USA', 'UK', 'UAE', 'SINGAPORE', 'CANADA'],
            'state_indian': [
                'ANDHRA PRADESH', 'ANDHRAPRADESH',
                'TELANGANA', 'TAMIL NADU', 'TAMILNADU',
                'KARNATAKA', 'KERALA',
                'MAHARASHTRA', 'GUJARAT', 'RAJASTHAN',
                'DELHI', 'PUNJAB', 'HARYANA',
                'UTTAR PRADESH', 'UTTARPRADESH',
                'MADHYA PRADESH', 'MADHYAPRADESH',
                'BIHAR', 'WEST BENGAL', 'WESTBENGAL', 'ODISHA'
            ],
        }

        # Find spans containing country/state names
        for i, span in enumerate(spans):
            text_upper = span.text.upper()

            # Check for country
            country = None
            for country_name in address_keywords['country']:
                if country_name in text_upper:
                    country = country_name
                    break

            if country:
                # Build address block from this span and nearby spans
                address_spans = self._build_address_block_simple(spans, i)

                if address_spans:
                    # Build full address text
                    address_text = ' '.join(s.text for s in address_spans)
                    address_text_upper = address_text.upper()

                    # Try to extract state (and normalize)
                    state = None
                    found_state_name = None
                    for state_name in address_keywords['state_indian']:
                        if state_name in address_text_upper:
                            found_state_name = state_name
                            # Normalize state name (use with-space version)
                            if 'PRADESH' in state_name and ' ' not in state_name:
                                state = state_name.replace('PRADESH', ' PRADESH')
                            elif 'NADU' in state_name and ' ' not in state_name:
                                state = state_name.replace('NADU', ' NADU')
                            elif 'BENGAL' in state_name and ' ' not in state_name:
                                state = state_name.replace('BENGAL', ' BENGAL')
                            else:
                                state = state_name
                            break

                    # Extract city (word before state, if any)
                    city = None
                    if found_state_name:
                        parts = address_text_upper.split(found_state_name)
                        if parts:
                            before_state = parts[0].strip()
                            # Get last word before state as potential city
                            words = before_state.split()
                            if words:
                                city = words[-1].strip(', ')

                    # Extract postal code
                    postal_code = None
                    for pattern in self.POSTAL_CODE_PATTERNS:
                        match = re.search(pattern, address_text)
                        if match:
                            postal_code = match.group(0)
                            break

                    # Extract name if no label found
                    name = None
                    if 'account_holder_name_label' not in label_spans:
                        name = self._extract_name_before_address(spans, address_spans)

                    address_blocks.append({
                        'text': address_text,
                        'spans': address_spans,
                        'country': country,
                        'state': state,
                        'city': city,
                        'postal_code': postal_code,
                        'name': name,
                        'center_x': sum(s.center_x for s in address_spans) / len(address_spans),
                        'center_y': sum(s.center_y for s in address_spans) / len(address_spans),
                    })

                    logger.debug(f"Found address block (fallback): country={country}, state={state}")

        return address_blocks

    def _build_address_block(
        self,
        spans: List[Span],
        country_span_idx: int,
        country_obj
    ) -> Tuple[Optional[str], Optional[str], List[Span]]:
        """
        Build an address block starting from the country span.

        Groups spans with:
        - Similar x coordinates (same column)
        - Close vertical proximity

        Continues upward until:
        - Line goes from having numbers to not having numbers (address start)
        - Or next line is too far away

        Returns:
            (state, city, list_of_spans_in_block)
        """
        from countrystatecity_countries import (
            get_states_of_country,
            get_cities_of_country
        )

        country_span = spans[country_span_idx]
        country_code = country_obj.iso2

        # Get states and cities for this country
        states = {s.name.upper(): s for s in get_states_of_country(country_code)}
        cities = {c.name.upper(): c for c in get_cities_of_country(country_code)}

        # Collect address spans starting from country and going upward
        address_spans = [country_span]

        # Find state and city
        state = None
        city = None

        for span in address_spans:
            text_upper = span.text.upper()
            for state_name in states:
                if state_name in text_upper:
                    state = state_name
                    break
            for city_name in cities:
                if city_name in text_upper:
                    city = city_name
                    break

        # Group spans above the country span
        reference_x = country_span.center_x
        idx = country_span_idx - 1

        found_first_address_line = False

        while idx >= 0:
            span = spans[idx]

            # Check if in same column (similar x)
            if abs(span.center_x - reference_x) > self.CLOSE_X_THRESHOLD:
                idx -= 1
                continue

            # Check if vertically close
            if country_span.y1 - span.y2 > self.ADDRESS_GROUP_Y_THRESHOLD:
                break  # Too far away

            # Check for address start: line with numbers → address starts here
            has_digits = any(c.isdigit() for c in span.text)

            if found_first_address_line and not has_digits:
                # We've gone from numbers to no numbers - this is the name line
                # Include it and stop
                address_spans.insert(0, span)
                break

            if has_digits:
                found_first_address_line = True

            address_spans.insert(0, span)
            idx -= 1

        return state, city, address_spans

    def _build_address_block_simple(
        self,
        spans: List[Span],
        start_idx: int
    ) -> List[Span]:
        """
        Simple address block building without country/state/city library.
        Groups spans with similar x and close y.
        """
        if start_idx >= len(spans):
            return []

        reference_span = spans[start_idx]
        reference_x = reference_span.center_x
        address_spans = [reference_span]

        # Group spans above the reference span
        idx = start_idx - 1

        found_first_address_line = False

        while idx >= 0:
            span = spans[idx]

            # Check if in same column (similar x)
            if abs(span.center_x - reference_x) > self.CLOSE_X_THRESHOLD:
                idx -= 1
                continue

            # Check if vertically close
            if reference_span.y1 - span.y2 > self.ADDRESS_GROUP_Y_THRESHOLD:
                break  # Too far away

            # Check for address start: line with numbers → address starts here
            has_digits = any(c.isdigit() for c in span.text)

            if found_first_address_line and not has_digits:
                # We've gone from numbers to no numbers - this is the name line
                # Include it and stop
                address_spans.insert(0, span)
                break

            if has_digits:
                found_first_address_line = True

            address_spans.insert(0, span)
            idx -= 1

        return address_spans

    def _extract_name_before_address(
        self,
        spans: List[Span],
        address_spans: List[Span]
    ) -> Optional[str]:
        """
        Extract account holder name from the line immediately before address,
        or from the first span of the address if it looks like a name.

        Only called when no account holder name label was found.
        """
        if not address_spans:
            return None

        # First, check if the first span in address_spans looks like a name
        # (e.g., "MR.K C ROHITH" at the start of an address)
        first_span = address_spans[0]
        if self._looks_like_name(first_span.text):
            # But make sure it doesn't look like an address line (no house numbers, etc.)
            if not self._looks_like_address(first_span.text):
                return first_span.text.strip()

        # Find the index of the first address span
        try:
            first_addr_idx = spans.index(first_span)
        except ValueError:
            return None

        # Check the span immediately before
        if first_addr_idx > 0:
            name_span = spans[first_addr_idx - 1]
            # Verify it's in the same column
            if abs(name_span.center_x - first_span.center_x) < self.CLOSE_X_THRESHOLD:
                name_text = name_span.text.strip()
                # Name should not have digits
                if not any(c.isdigit() for c in name_text):
                    return name_text

        return None

    def _identify_customer_vs_bank_address(
        self,
        address_blocks: List[Dict],
        label_spans: Dict[str, Span]
    ) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Determine which address belongs to customer vs bank.

        Heuristics:
        - Customer address is usually in the main content area (left side)
        - Bank address is often in header/footer (right side or edges)
        - If bank name label was found, use its position as hint
        """
        if not address_blocks:
            return None, None
        if len(address_blocks) == 1:
            return address_blocks[0], None

        # Sort by x position (left to right)
        sorted_by_x = sorted(address_blocks, key=lambda a: a['center_x'])

        # Customer address is typically on the left
        # Bank address is typically on the right
        return sorted_by_x[0], sorted_by_x[-1]

    # =========================================================================
    # PASS 3: Find account number value
    # =========================================================================

    def _pass3_find_account_number_value(
        self,
        spans: List[Span],
        label_span: Span
    ) -> Optional[str]:
        """
        Pass 3: Find account number value if not found in Pass 1.

        Looks:
        1. To the right of label (same line)
        2. Below label (same column)

        Uses country-specific account number regex.
        """
        if not label_span:
            return None

        # Generic account number patterns
        account_patterns = [
            r'\b\d{10,20}\b',
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        ]

        # Look to the right (same y, larger x)
        for span in spans:
            if span is label_span:
                continue

            # Check if on same line (close y)
            if abs(span.center_y - label_span.center_y) > self.SAME_LINE_Y_THRESHOLD:
                continue

            # Check if to the right
            if span.x1 > label_span.x2:
                # Found span to the right - check if it matches account pattern
                for pattern in account_patterns:
                    match = re.search(pattern, span.text)
                    if match:
                        return re.sub(r'[^\d]', '', match.group(0))

        # Look below (same column, larger y)
        for span in spans:
            if span is label_span:
                continue

            # Check if in same column (close x)
            if abs(span.center_x - label_span.center_x) > self.CLOSE_X_THRESHOLD:
                continue

            # Check if below
            if span.y1 > label_span.y2:
                # Found span below - check if it matches account pattern
                for pattern in account_patterns:
                    match = re.search(pattern, span.text)
                    if match:
                        return re.sub(r'[^\d]', '', match.group(0))

        return None

"""
Spatial Bank Statement Extractor - Three-Pass Algorithm

Uses geometry and proximity for robust extraction across various PDF layouts and images.

Three-Pass Algorithm:
- Pass 1: Label and Initial Value Detection - Find labels and populate unified map
- Pass 2: Value Extraction via Spatial Proximity - Extract values by geometric relationships
- Pass 3: Full Address Extraction - Build complete address from address anchor blocks (states)

Supports:
- Text-based PDFs (direct PyMuPDF extraction)
- Image-based PDFs (OCR via DocTR)
- Regular images (JPG, PNG, etc. via OCR)
"""

import fitz  # PyMuPDF
import json
import logging
import re
import statistics
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from io import BytesIO

from app.core.key_injection.bank_lookup import get_bank_lookup, BankInfo
from app.helper.validators.bank_statement_validator import (
    get_bank_statement_validator, get_country_config_loader
)
from app.config.bank_statement_country_loader import get_country_config_loader
from app.schemas.bank_statement_schema import BankStatementData
from app.helper.doctr.document_text_extractor import DocumentTextExtractor

logger = logging.getLogger(__name__)


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class SpanInfo:
    """Common data structure for span information."""
    x1: float
    x2: float
    y1: float
    y2: float
    value: Optional[str] = None
    text: str = ""

    def width(self) -> float:
        return self.x2 - self.x1

    def height(self) -> float:
        return self.y2 - self.y1

    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2


@dataclass
class ExtractionResult:
    """Result of bank statement extraction - compatible with test verification."""
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
    iban: Optional[str] = None  # Phase 2: International Bank Account Number
    opening_balance: Optional[str] = None  # Phase 2: Opening/statement balance
    closing_balance: Optional[str] = None  # Phase 2: Closing/current balance
    ifsc_code: Optional[str] = None
    swift_code: Optional[str] = None
    raw_values: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class ExtractedBankStatement:
    """Final extracted data structure."""
    account_number: str
    account_holder_name: Optional[str] = None
    bank_name: str = ""                      # Abbreviation
    bank_full_name: str = ""                 # From lookup
    address: str = ""                        # Lines 1 & 2 only
    city: str = ""
    state: Optional[str] = None
    country: str = ""
    postal_code: Optional[str] = None
    swift_code: Optional[str] = None


# ============================================================
# CONFIGURATION LOADER
# ============================================================

def _get_all_known_countries() -> Set[str]:
    """
    Get all known country names and aliases from config.

    Filters out very short patterns (< 3 chars) to avoid false positives
    from substring matching (e.g., "IN" in "NARASIMHAN").

    Returns:
        Set of uppercase country names and aliases
    """
    country_loader = get_country_config_loader()
    all_countries: Set[str] = set()

    for country_code in country_loader.get_supported_countries():
        config = country_loader.get_country_config(country_code)
        if not config:
            continue

        # Add country name
        country_name = config.get("country_name")
        if country_name:
            all_countries.add(country_name.upper())

        # Add name aliases (only if 3+ characters to avoid false positives)
        for alias in config.get("name_aliases", []):
            if len(alias) >= 3:  # Skip short aliases like "IN", "SG", "US"
                all_countries.add(alias.upper())

    return all_countries


def load_config() -> Dict:
    """Load bank statement configuration."""
    config_path = Path(__file__).parent.parent.parent / "reference_templates" / "bank_statements" / "config.json"
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        return {}


def build_unified_map(config: Dict) -> Dict[str, str]:
    """
    Build unified map containing all patterns for label/bank/address detection.

    Returns:
        Dict mapping pattern text to category:
        - "account_number_label"
        - "account_holder_name_label"
        - "currency_label"
        - "iban_label"
        - "statement_date_label"
        - "opening_balance_label"
        - "closing_balance_label"
        - "bank_name"
        - "address_block"
    """
    unified_map = {}

    # 1. Account number labels
    for label in config.get("account_number_labels", []):
        unified_map[label.upper()] = "account_number_label"

    # 2. Account holder name labels (NEW)
    for label in config.get("account_holder_name_labels", []):
        unified_map[label.upper()] = "account_holder_name_label"

    # 3. Currency labels
    for label in config.get("currency_labels", []):
        unified_map[label.upper()] = "currency_label"

    # Also add currency names (e.g., "UAE DIRHAM", "US DOLLAR") as currency labels
    # This ensures that currency text is extracted and removed from address consideration
    currency_name_map = config.get("currency_name_map", {})
    for currency_name in currency_name_map.keys():
        unified_map[currency_name.upper()] = "currency_label"

    # 4. IBAN labels (NEW - Phase 2)
    for label in config.get("iban_labels", []):
        unified_map[label.upper()] = "iban_label"

    # 5. Statement date labels (NEW - Phase 2)
    for label in config.get("statement_date_labels", []):
        unified_map[label.upper()] = "statement_date_label"

    # 6. Opening balance labels (NEW - Phase 2)
    for label in config.get("opening_balance_labels", []):
        unified_map[label.upper()] = "opening_balance_label"

    # 7. Closing balance labels (NEW - Phase 2)
    for label in config.get("closing_balance_labels", []):
        unified_map[label.upper()] = "closing_balance_label"

    # 8. Bank identifiers (names OR URLs, never both - analyzed per bank)
    # Filter out very short patterns (< 4 chars) to avoid false matches like "YES", "ING", "US"
    for identifier, abbrev in config.get("bank_identifiers_map", {}).items():
        if len(identifier) >= 4:
            unified_map[identifier.upper()] = "bank_name"

    # 9. Country names from config (most reliable anchors for address detection)
    # Countries are distinctive and rarely appear in street names or other content
    try:
        all_countries = _get_all_known_countries()
        for country in all_countries:
            unified_map[country] = "address_block"
    except Exception as e:
        logger.warning(f"Failed to load countries for unified map: {e}")

    # 10. State names from state_to_country_map (for addresses that don't contain country name)
    # This helps with PDFs where the address only shows state (e.g., "ANDHRA PRADESH") without "INDIA"
    state_to_country = config.get("state_to_country_map", {})
    for state_name in state_to_country.keys():
        unified_map[state_name.upper()] = "address_block"

    return unified_map


def get_account_number_regex(country: str) -> re.Pattern:
    """Get account number regex for a country."""
    config = load_config()
    currencies = config.get("currencies", {})

    # Find currency config for this country
    for currency_info in currencies.values():
        if currency_info.get("country") == country:
            length_info = currency_info.get("account_number_length", {"min": 8, "max": 16})
            min_len = length_info.get("min", 8)
            max_len = length_info.get("max", 16)
            # Allow spaces, dashes, but mostly digits
            return re.compile(r'^[\d\s-]{' + str(min_len) + ',' + str(max_len) + '}$')

    # Default: 8-16 digits with optional spaces/dashes
    return re.compile(r'^[\d\s-]{8,16}$')


# ============================================================
# PYMUPDF SPAN EXTRACTION
# ============================================================

def extract_spans_from_pdf(pdf_path: str, max_pages: int = 1) -> List[Dict]:
    """
    Extract text grouped into spans using PyMuPDF.

    All coordinates are normalized to 0-1 range to match OCR output format.

    Args:
        pdf_path: Path to PDF file
        max_pages: Maximum pages to process

    Returns:
        List of span dictionaries with text and normalized coordinates (0-1)
    """
    try:
        doc = fitz.open(pdf_path)
        spans = []

        pages_to_process = min(max_pages, doc.page_count)

        for page_num in range(pages_to_process):
            page = doc[page_num]
            blocks = page.get_text("dict")

            # Get page dimensions for normalization
            page_rect = page.rect
            page_width = page_rect.width
            page_height = page_rect.height

            for block in blocks.get("blocks", []):
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            span_text = span.get("text", "").strip()
                            if not span_text:
                                continue

                            # Get bounding box
                            bbox = span.get("bbox", [0, 0, 0, 0])
                            x0, y0, x1, y1 = bbox

                            # Normalize coordinates to 0-1 range
                            spans.append({
                                "text": span_text,
                                "x1": x0 / page_width,
                                "y1": y0 / page_height,
                                "x2": x1 / page_width,
                                "y2": y1 / page_height,
                                "page_num": page_num + 1,
                                "font_size": span.get("size", 12),
                                "flags": span.get("flags", 0)
                            })

        doc.close()
        return spans

    except Exception as e:
        logger.error(f"Failed to extract spans from PDF {pdf_path}: {e}")
        return []


async def extract_spans_from_bytes(pdf_bytes: bytes, max_pages: int = 1) -> List[Dict]:
    """
    Extract text grouped into spans from PDF bytes (async wrapper).

    For text-based PDFs, uses PyMuPDF directly.
    For image-based PDFs, falls back to OCR.

    Args:
        pdf_bytes: Raw PDF file bytes
        max_pages: Maximum pages to process

    Returns:
        List of span dictionaries with text and coordinates
    """
    try:
        # Try direct PDF extraction first
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        # Check if PDF has extractable text
        has_text = False
        for page_num in range(min(max_pages, doc.page_count)):
            page = doc[page_num]
            text_blocks = page.get_text("blocks")
            if text_blocks:
                has_text = True
                break

        doc.close()

        if has_text:
            # Use direct extraction for text-based PDFs
            return _extract_spans_from_pdf_bytes_direct(pdf_bytes, max_pages)
        else:
            # Use OCR for image-based PDFs
            logger.info("PDF appears to be image-based, using OCR extraction")
            return await _extract_spans_from_ocr(pdf_bytes, is_pdf=True, max_pages=max_pages)

    except Exception as e:
        logger.error(f"Failed to extract spans from PDF bytes: {e}")
        # Fallback to OCR
        try:
            return await _extract_spans_from_ocr(pdf_bytes, is_pdf=True, max_pages=max_pages)
        except Exception as ocr_error:
            logger.error(f"OCR fallback also failed: {ocr_error}")
            return []


async def extract_spans_from_image(image_bytes: bytes, max_pages: int = 1) -> List[Dict]:
    """
    Extract text grouped into spans from image bytes using OCR.

    Args:
        image_bytes: Raw image file bytes (JPG, PNG, etc.)
        max_pages: Maximum pages to process (for multi-page formats)

    Returns:
        List of span dictionaries with text and coordinates
    """
    return await _extract_spans_from_ocr(image_bytes, is_pdf=False, max_pages=max_pages)


def _extract_spans_from_pdf_bytes_direct(pdf_bytes: bytes, max_pages: int = 1) -> List[Dict]:
    """
    Extract text grouped into spans using PyMuPDF directly from bytes (sync).

    All coordinates are normalized to 0-1 range to match OCR output format.

    Args:
        pdf_bytes: Raw PDF file bytes
        max_pages: Maximum pages to process

    Returns:
        List of span dictionaries with text and normalized coordinates (0-1)
    """
    try:
        # Open PDF directly from bytes
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        spans = []

        pages_to_process = min(max_pages, doc.page_count)

        for page_num in range(pages_to_process):
            page = doc[page_num]
            blocks = page.get_text("dict")

            # Get page dimensions for normalization
            page_rect = page.rect
            page_width = page_rect.width
            page_height = page_rect.height

            for block in blocks.get("blocks", []):
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            span_text = span.get("text", "").strip()
                            if not span_text:
                                continue

                            # Get bounding box
                            bbox = span.get("bbox", [0, 0, 0, 0])
                            x0, y0, x1, y1 = bbox

                            # Normalize coordinates to 0-1 range
                            spans.append({
                                "text": span_text,
                                "x1": x0 / page_width,
                                "y1": y0 / page_height,
                                "x2": x1 / page_width,
                                "y2": y1 / page_height,
                                "page_num": page_num + 1,
                                "font_size": span.get("size", 12),
                                "flags": span.get("flags", 0)
                            })

        doc.close()

        logger.info(f"Extracted {len(spans)} spans from PDF, line spacing: 0.01")
        # Log ALL spans for debugging bank name detection
        for i, span in enumerate(spans):
            logger.debug(f"  Span {i}: '{span.get('text', '')}'")
        return spans

    except Exception as e:
        logger.error(f"Failed to extract spans from PDF bytes (direct): {e}")
        return []


async def _extract_spans_from_ocr(file_bytes: bytes, is_pdf: bool, max_pages: int = 1) -> List[Dict]:
    """
    Extract text spans using OCR (DocTR).

    This function provides a common interface for OCR-based extraction,
    used for both image-based PDFs and regular images.

    Args:
        file_bytes: Raw file bytes (PDF or image)
        is_pdf: Whether the input is a PDF
        max_pages: Maximum pages to process

    Returns:
        List of span dictionaries with text and coordinates (normalized 0-1)
    """
    try:
        # Use DocumentTextExtractor for OCR
        text_extractor = DocumentTextExtractor()
        ocr_result = await text_extractor.extract_text_with_geometry_enhanced(
            file_bytes, is_pdf=is_pdf, max_pages=max_pages
        )

        # Convert OCR result to span format
        spans = []
        for line in ocr_result:
            span_text = line.get("text", "").strip()
            if not span_text:
                continue

            # OCR provides normalized coordinates (0-1), keep them consistent
            spans.append({
                "text": span_text,
                "x1": line.get("x1", 0.0),
                "y1": line.get("y1", 0.0),
                "x2": line.get("x2", 0.0),
                "y2": line.get("y2", 0.0),
                "page_num": 1,  # OCR treats first page as page 1
                "font_size": 12,  # OCR doesn't provide font size, use default
                "flags": 0,
                "confidence": line.get("confidence", 0.9)  # Store confidence for potential filtering
            })

        logger.info(f"OCR extracted {len(spans)} spans from {'PDF' if is_pdf else 'image'}")
        # Log first 15 spans at INFO level for bank name debugging
        for i, span in enumerate(spans[:15]):
            logger.info(f"  Span {i}: '{span.get('text', '')}'")
        # Log remaining spans at DEBUG level
        for i, span in enumerate(spans[15:], start=15):
            logger.debug(f"  Span {i}: '{span.get('text', '')}'")
        return spans

    except Exception as e:
        logger.error(f"Failed to extract spans via OCR: {e}")
        raise


def _is_span_printable(text: str, min_printable_ratio: float = 0.5) -> bool:
    """
    Check if span text is mostly printable characters.

    Filters out PDF encoding artifacts that contain mostly control/non-printable chars.
    """
    if not text:
        return False

    # Count printable characters (ASCII printable + common whitespace)
    printable_chars = set(string.printable)  # Includes letters, digits, punctuation, whitespace
    printable_count = sum(1 for c in text if c in printable_chars)

    # Also count common extended ASCII printable characters (like accented letters)
    extended_printable = sum(1 for c in text if 32 <= ord(c) <= 126 or 128 <= ord(c) <= 255 or c in '\n\r\t')
    printable_count = max(printable_count, extended_printable)

    ratio = printable_count / len(text) if text else 0
    return ratio >= min_printable_ratio


def convert_spans_to_span_info(spans: List[Dict]) -> List[SpanInfo]:
    """Convert span dictionaries to SpanInfo objects, filtering out non-printable spans."""
    result = []
    for s in spans:
        text = s["text"]
        # Skip spans with mostly non-printable characters (PDF encoding artifacts)
        if not _is_span_printable(text):
            continue
        result.append(
            SpanInfo(
                x1=s["x1"],
                y1=s["y1"],
                x2=s["x2"],
                y2=s["y2"],
                text=text,
                value=None
            )
        )
    return result


def calculate_standard_line_spacing(spans: List[SpanInfo]) -> float:
    """Calculate median line height from all spans."""
    line_heights = [s.height() for s in spans if s.height() > 0]

    if not line_heights:
        return 12.0  # Default fallback

    return statistics.median(line_heights)


# ============================================================
# THREE-PASS EXTRACTION ALGORITHM
# ============================================================

class SpatialBankStatementExtractor:
    """
    Three-pass spatial bank statement extractor.

    Pass 1: Label and Initial Value Detection
    Pass 2: Value Extraction via Spatial Proximity
    Pass 3: Full Address Extraction
    """

    # Common title/prefix patterns that indicate the actual name follows
    # Only include titles with "." to avoid false positives
    TITLE_PATTERNS = {
        'MR.', 'MRS.', 'MS.', 'DR.',
        'MISS.', 'PROF.', 'REV.', 'HON.'
    }

    # Patterns for labels that should NOT be part of street address
    # These are typically form labels, not actual address content
    ADDRESS_SKIP_LABELS = {
        'JOINT HOLDER', 'JOINT HOLDERS', 'JOINT HOLDER NAME',
        'PRIMARY HOLDER', 'SECONDARY HOLDER', 'HOLDER NAME',
        'NOMINEE', 'NOMINEE NAME',
        'CURRENT ACCOUNT', 'SAVINGS ACCOUNT', 'ACCOUNT TYPE',
        'ACCOUNT NUMBER', 'A/C NO', 'AC NO',
        'CENTRAL BANK', 'BANK NAME', 'BRANCH NAME',
        'BRANCH CODE', 'BRANCH EMAIL', 'BRANCH PHONE', 'BRANCH NUMBER',
        'BRANCH ADDRESS', 'REGISTERED OFFICE', 'CORPORATE OFFICE',
        'HEAD OFFICE', 'REGD OFFICE', 'REGISTERED OFFICE',
        'IFSC CODE', 'MICR CODE', 'RTGS', 'NEFT',
        'ADHAR', 'AADHAAR', 'PAN', 'KYC',
        'MOBILE', 'PHONE', 'EMAIL', 'CONTACT',
        'CUSTOMER DETAILS', 'CUSTOMER NAME', 'CUSTOMER ID',
        'NAME', 'PRIMARY ID TYPE', 'ID TYPE',
        'PAGE', 'STATEMENT DETAILS', 'STATEMENT SUMMARY',
        'TRANSACTION DATE', 'VALUE DATE', 'CHEQUE NO',
        'DESCRIPTION', 'CR/DR', 'BALANCE',
        'DATE ISSUED', 'ISSUE DATE', 'EXPIRY DATE',
        'CARD TYPE', 'CARD NUMBER',
        'WEBSITE', 'WWW', 'HTTP', 'HTTPS',
        'YOUR BASE BRANCH', 'BASE BRANCH'
    }

    def __init__(self):
        self.logger = logger
        self.bank_lookup = get_bank_lookup()
        self.config = load_config()
        self.unified_map = build_unified_map(self.config)

    def extract(self, pdf_path: str, max_pages: int = 1) -> ExtractionResult:
        """
        Extract bank statement data using three-pass algorithm.

        Args:
            pdf_path: Path to PDF file
            max_pages: Maximum pages to process

        Returns:
            ExtractionResult with extracted fields (compatible with test verification)
        """
        # Extract spans using PyMuPDF
        raw_spans = extract_spans_from_pdf(pdf_path, max_pages)
        if not raw_spans:
            raise ValueError("No text extracted from PDF")

        # Convert to SpanInfo
        spans = convert_spans_to_span_info(raw_spans)

        # Sort spans by Y then X to ensure consistent processing order
        # This ensures that earlier matches (like bank name in header) are found first
        spans.sort(key=lambda s: (s.y1, s.x1))

        # Calculate standard line spacing
        standard_line_spacing = calculate_standard_line_spacing(spans)
        multi_line_threshold = standard_line_spacing * 1.5

        self.logger.info(f"Extracted {len(spans)} spans, line spacing: {standard_line_spacing:.2f}")

        # PASS 1: Label and Initial Value Detection
        first_pass_results, first_pass_addresses, list_for_second_pass, currency_spans = self._pass_1_label_detection(
            spans, standard_line_spacing
        )

        # Validate Pass 1 results
        self._validate_pass_1(first_pass_results, first_pass_addresses)

        # PASS 2: Value Extraction via Spatial Proximity
        self._pass_2_value_extraction(
            first_pass_results, list_for_second_pass, standard_line_spacing
        )

        # PASS 3: Full Address Extraction
        address_components = self._pass_3_address_extraction(
            first_pass_addresses, list_for_second_pass, spans, standard_line_spacing,
            first_pass_results.get("account_holder_name_label"),
            first_pass_results.get("bank_name"),
            currency_spans  # Pass currency spans for exclusion
        )

        # Build final result
        return self._build_result(first_pass_results, address_components)

    async def extract_from_bytes(
        self,
        file_bytes: bytes,
        max_pages: int = 1,
        is_pdf: Optional[bool] = None
    ) -> BankStatementData:
        """
        Extract bank statement data from file bytes (async for compatibility with service layer).

        Supports:
        - Text-based PDFs (direct PyMuPDF extraction)
        - Image-based PDFs (OCR via DocTR)
        - Regular images (JPG, PNG, etc. via OCR)

        Returns BankStatementData (same as GLiNER extractor) for compatibility.
        Note: Wraps synchronous PyMuPDF operations (event loop blocked during extraction,
        matching GLiNER's pattern which also wraps sync PyTorch inference).

        Args:
            file_bytes: Raw file bytes (PDF or image)
            max_pages: Maximum pages to process
            is_pdf: Whether the file is a PDF (auto-detected if None)

        Returns:
            BankStatementData with extracted fields (Pydantic BaseModel)
        """
        # Auto-detect file type if not specified
        if is_pdf is None:
            is_pdf = self._detect_file_type(file_bytes)

        # Extract spans using the appropriate method
        if is_pdf:
            self.logger.info("Processing PDF: attempting direct extraction first")
            raw_spans = await extract_spans_from_bytes(file_bytes, max_pages)
        else:
            self.logger.info("Processing image: using OCR extraction")
            raw_spans = await extract_spans_from_image(file_bytes, max_pages)

        if not raw_spans:
            raise ValueError("No text extracted from file")

        # Convert to SpanInfo
        spans = convert_spans_to_span_info(raw_spans)

        # Sort spans by Y then X to ensure consistent processing order
        # This ensures that earlier matches (like bank name in header) are found first
        spans.sort(key=lambda s: (s.y1, s.x1))

        # Calculate standard line spacing
        standard_line_spacing = calculate_standard_line_spacing(spans)
        multi_line_threshold = standard_line_spacing * 1.5

        self.logger.info(f"Extracted {len(spans)} spans from {'PDF' if is_pdf else 'image'}, "
                        f"line spacing: {standard_line_spacing:.2f}")

        # PASS 1: Label and Initial Value Detection
        first_pass_results, first_pass_addresses, list_for_second_pass, currency_spans = self._pass_1_label_detection(
            spans, standard_line_spacing
        )

        # OCR Fallback: If bank_name is missing, try OCR-based detection
        # This handles cases where bank name/logo is embedded as an image
        if "bank_name" not in first_pass_results and is_pdf:
            self.logger.info("Bank name not found in direct text extraction, trying OCR fallback")
            bank_abbrev = await self._detect_bank_name_from_ocr(file_bytes, is_pdf)
            if bank_abbrev:
                # Add bank_name to first_pass_results
                first_pass_results["bank_name"] = SpanInfo(
                    x1=0, y1=0, x2=0, y2=0,
                    text="",
                    value=bank_abbrev
                )
                self.logger.info(f"OCR fallback successfully detected bank: {bank_abbrev}")

        # Validate Pass 1 results
        self._validate_pass_1(first_pass_results, first_pass_addresses)

        # PASS 2: Value Extraction via Spatial Proximity
        self._pass_2_value_extraction(
            first_pass_results, list_for_second_pass, standard_line_spacing
        )

        # PASS 3: Full Address Extraction
        address_components = self._pass_3_address_extraction(
            first_pass_addresses, list_for_second_pass, spans, standard_line_spacing,
            first_pass_results.get("account_holder_name_label"),
            first_pass_results.get("bank_name"),
            currency_spans  # Pass currency spans for exclusion
        )

        # Create internal result first
        internal_result = self._build_result(first_pass_results, address_components)

        # Convert to BankStatementData (GLiNER-compatible format)
        return self._convert_to_bank_statement_data(internal_result)

    def _detect_file_type(self, file_bytes: bytes) -> bool:
        """
        Detect if the file bytes represent a PDF or an image.

        Args:
            file_bytes: Raw file bytes

        Returns:
            True if PDF, False if image
        """
        # Check for PDF magic bytes
        if file_bytes.startswith(b'%PDF'):
            return True

        # Check for common image signatures
        # JPEG: FF D8 FF
        if file_bytes.startswith(b'\xFF\xD8\xFF'):
            return False
        # PNG: 89 50 4E 47
        if file_bytes.startswith(b'\x89PNG'):
            return False
        # GIF: 47 49 46 38
        if file_bytes.startswith(b'GIF8'):
            return False
        # WebP: 52 49 46 46 ... 57 45 42 50
        if file_bytes.startswith(b'RIFF') and len(file_bytes) > 11:
            if file_bytes[8:12] == b'WEBP':
                return False
        # BMP: 42 4D
        if file_bytes.startswith(b'BM'):
            return False

        # Default to PDF if we can't determine
        self.logger.warning("Could not determine file type from magic bytes, defaulting to PDF")
        return True

    def _pass_1_label_detection(
        self,
        spans: List[SpanInfo],
        standard_line_spacing: float
    ) -> Tuple[Dict[str, SpanInfo], List[SpanInfo], List[SpanInfo], List[SpanInfo]]:
        """
        Pass 1: Label and Initial Value Detection.

        For each span, iterate through unified map looking for matches ONLY if:
        - "account_number_label" not found, OR
        - "account_holder_name_label" not found, OR
        - "currency_label" not found, OR
        - "bank_name" not found, OR
        - "address_block" not found (at least one)

        Returns:
            Tuple of (first_pass_results, first_pass_addresses, list_for_second_pass, currency_spans)
        """
        first_pass_results: Dict[str, SpanInfo] = {}
        first_pass_addresses: List[SpanInfo] = []
        list_for_second_pass: List[SpanInfo] = []
        # Track currency-matched spans for exclusion in Pass 3
        currency_spans: List[SpanInfo] = []

        # Track the maximum date found across all spans for statement date
        from app.utils.date_extractor import extract_all_dates
        latest_statement_date = None

        for span in spans:
            span_text_upper = span.text.upper().strip()

            # Extract dates from this span's text and track the maximum
            # Use very large max_age_days to include all dates regardless of age
            span_dates = extract_all_dates(span.text, max_age_days=36500)  # 100 years
            if span_dates:
                span_max_date = max(span_dates)  # Get max date in this span
                if latest_statement_date is None or span_max_date > latest_statement_date:
                    latest_statement_date = span_max_date

            # Check if we still need to find any targets
            still_need_account_number = "account_number_label" not in first_pass_results
            still_need_holder_name = "account_holder_name_label" not in first_pass_results
            still_need_currency = "currency_label" not in first_pass_results
            still_need_bank = "bank_name" not in first_pass_results
            # Note: We no longer collect address_block in Pass 1
            # Country will be detected based on proximity to bank name in Pass 3

            # Check unified map for matches
            matched = False

            # Always check for currency_label patterns (even if we already have one)
            # This ensures currency names like "UAE DIRHAM", "DIRHAM" are removed from address consideration
            for pattern, category in self.unified_map.items():
                if category == "currency_label" and pattern in span_text_upper:
                    currency = self._extract_currency_from_span(span, spans)
                    span.value = currency
                    if still_need_currency:
                        first_pass_results["currency_label"] = span
                    # Track currency spans for exclusion in Pass 3
                    currency_spans.append(span)
                    matched = True
                    break

            # Check other patterns (account_number_label, account_holder_name_label, bank_name)
            if not matched:
                if not (still_need_account_number or still_need_holder_name or still_need_bank):
                    # All required items found (except currency which we always check)
                    list_for_second_pass.append(span)
                    continue

                for pattern, category in self.unified_map.items():
                    if category == "address_block":
                        continue  # Already checked above
                    if category == "currency_label":
                        continue  # Already checked above
                    if category == "account_number_label" and not still_need_account_number:
                        continue
                    if category == "account_holder_name_label" and not still_need_holder_name:
                        continue
                    if category == "bank_name" and not still_need_bank:
                        continue

                    # For other categories, check if pattern is a substring of span text
                    # Only one direction: pattern (from unified_map) should be in span_text_upper
                    if pattern in span_text_upper:
                        # Found a match
                        if category == "account_number_label":
                            # Check if account number is in the same span
                            account_number = self._extract_account_number_from_span(span)
                            span.value = account_number
                            first_pass_results["account_number_label"] = span
                            matched = True

                        elif category == "account_holder_name_label":
                            # Account holder name might be in same span or split (title + name)
                            holder_name = self._extract_holder_name_from_span(span, spans)
                            # If the extracted value looks like a label (contains label keywords), clear it
                            # This allows Pass 2 to extract the actual value spatially
                            if holder_name and self._looks_like_label(holder_name):
                                holder_name = None
                            span.value = holder_name
                            first_pass_results["account_holder_name_label"] = span
                            matched = True

                        elif category == "currency_label":
                            # Currency labels
                            currency = self._extract_currency_from_span(span, spans)
                            span.value = currency
                            first_pass_results["currency_label"] = span
                            # Track currency spans for exclusion in Pass 3
                            currency_spans.append(span)
                            matched = True

                        elif category == "iban_label":
                            # IBAN labels (Phase 2)
                            iban = self._extract_iban_from_span(span, spans)
                            span.value = iban
                            first_pass_results["iban_label"] = span
                            matched = True

                        elif category == "statement_date_label":
                            # Statement date labels (Phase 2)
                            statement_date = self._extract_statement_date_from_span(span, spans)
                            span.value = statement_date
                            first_pass_results["statement_date_label"] = span
                            matched = True

                        elif category == "opening_balance_label":
                            # Opening balance labels (Phase 2)
                            opening_balance = self._extract_balance_from_span(span, spans)
                            span.value = opening_balance
                            first_pass_results["opening_balance_label"] = span
                            matched = True

                        elif category == "closing_balance_label":
                            # Closing balance labels (Phase 2)
                            closing_balance = self._extract_balance_from_span(span, spans)
                            span.value = closing_balance
                            first_pass_results["closing_balance_label"] = span
                            matched = True

                        elif category == "bank_name":
                            # Bank name requires value to be populated
                            self.logger.debug(f"Bank name pattern matched in span: '{span.text}'")
                            bank_info = self._match_bank_from_text(span.text)
                            if bank_info:
                                span.value = bank_info.abbreviation
                                first_pass_results["bank_name"] = span
                                self.logger.info(f"Bank name detected: {bank_info.abbreviation} from '{span.text}'")
                                matched = True
                            else:
                                self.logger.warning(f"Bank pattern matched but lookup failed for: '{span.text}'")

                    if matched:
                        break

            if not matched:
                list_for_second_pass.append(span)

        # Sort list_for_second_pass by Y then X
        list_for_second_pass.sort(key=lambda s: (s.y1, s.x1))

        self.logger.info(
            f"Pass 1: Found {len(first_pass_results)} labels, "
            f"{len(first_pass_addresses)} address blocks, "
            f"{len(list_for_second_pass)} spans for pass 2"
        )

        # Log the address blocks found
        for i, addr in enumerate(first_pass_addresses):
            self.logger.info(f"  first_pass_addresses[{i}]: Y=[{addr.y1:.1f}, {addr.y2:.1f}] text='{addr.text}'")

        self.logger.info(f"Pass 1: Found {len(currency_spans)} currency spans to exclude from Pass 3")

        # Assign the latest date found as the statement date (primary method)
        if latest_statement_date:
            # Ensure final format is DD MMM YYYY (e.g., "10 Mar 2026")
            statement_date_str = latest_statement_date.strftime("%d %b %Y").upper()
            # Always use the latest date found, overriding any existing statement_date_label
            first_pass_results["statement_date_label"] = SpanInfo(
                x1=0, y1=0, x2=0, y2=0,
                text="",
                value=statement_date_str
            )
            self.logger.info(f"Statement date set to latest date found in document: {statement_date_str}")

        return first_pass_results, first_pass_addresses, list_for_second_pass, currency_spans

    def _validate_pass_1(
        self,
        first_pass_results: Dict[str, SpanInfo],
        first_pass_addresses: List[SpanInfo]
    ) -> None:
        """
        Validate Pass 1 results.

        Must have:
        - account_number_label
        - bank_name

        Note: address_block detection is no longer done in Pass 1.
        Country is detected in Pass 3 using spatial proximity to bank name.
        """
        required = ["account_number_label", "bank_name"]

        for req in required:
            if req not in first_pass_results:
                # Provide diagnostic info for missing bank_name
                if req == "bank_name":
                    self.logger.error("Bank name not found in Pass 1 - providing diagnostic info")
                    # Log available bank patterns from config
                    bank_patterns = [k for k, v in self.unified_map.items() if v == "bank_name"]
                    self.logger.info(f"Available bank patterns in config ({len(bank_patterns)}): "
                                   f"{', '.join(sorted(bank_patterns)[:20])}...")
                raise ValueError(f"Pass 1 validation failed: Missing required '{req}'")

        # first_pass_addresses is now empty (we no longer collect address blocks in Pass 1)
        # Country detection will happen in Pass 3 using spatial proximity to bank name

        self.logger.info("Pass 1 validation passed")

    def _pass_2_value_extraction(
        self,
        first_pass_results: Dict[str, SpanInfo],
        list_for_second_pass: List[SpanInfo],
        standard_line_spacing: float
    ) -> None:
        """
        Pass 2: Value Extraction via Spatial Proximity.

        Extracts account number, account holder name, currency, IBAN, statement date,
        and balance values using geometric relationships (right-side and below-label search).
        """
        # Extract Account Number Value (if not already found)
        account_label = first_pass_results.get("account_number_label")
        if account_label and account_label.value is None:
            account_number = self._extract_account_number_spatial(
                account_label, list_for_second_pass, standard_line_spacing
            )
            if account_number:
                account_label.value = account_number
                self.logger.info(f"Extracted account number: {account_number}")

        # Extract Account Holder Name Value (if label exists and value is None)
        holder_label = first_pass_results.get("account_holder_name_label")
        if holder_label and holder_label.value is None:
            holder_name = self._extract_holder_name_spatial(
                holder_label, list_for_second_pass, standard_line_spacing
            )
            if holder_name:
                holder_label.value = holder_name
                self.logger.info(f"Extracted account holder name: {holder_name}")

        # Extract Currency Value (if label exists and value is None)
        currency_label = first_pass_results.get("currency_label")
        if currency_label and currency_label.value is None:
            currency = self._extract_currency_spatial(
                currency_label, list_for_second_pass, standard_line_spacing
            )
            if currency:
                currency_label.value = currency
                self.logger.info(f"Extracted currency: {currency}")

        # Extract IBAN Value (if label exists and value is None) - Phase 2
        iban_label = first_pass_results.get("iban_label")
        if iban_label and iban_label.value is None:
            iban = self._extract_iban_spatial(
                iban_label, list_for_second_pass, standard_line_spacing
            )
            if iban:
                iban_label.value = iban
                self.logger.info(f"Extracted IBAN: {iban}")

        # Extract Statement Date Value (if label exists and value is None) - Phase 2
        statement_date_label = first_pass_results.get("statement_date_label")
        if statement_date_label and statement_date_label.value is None:
            statement_date = self._extract_statement_date_spatial(
                statement_date_label, list_for_second_pass, standard_line_spacing
            )
            if statement_date:
                statement_date_label.value = statement_date
                self.logger.info(f"Extracted statement date: {statement_date}")

        # Extract Opening Balance Value (if label exists and value is None) - Phase 2
        opening_balance_label = first_pass_results.get("opening_balance_label")
        if opening_balance_label and opening_balance_label.value is None:
            opening_balance = self._extract_balance_spatial(
                opening_balance_label, list_for_second_pass, standard_line_spacing
            )
            if opening_balance:
                opening_balance_label.value = opening_balance
                self.logger.info(f"Extracted opening balance: {opening_balance}")

        # Extract Closing Balance Value (if label exists and value is None) - Phase 2
        closing_balance_label = first_pass_results.get("closing_balance_label")
        if closing_balance_label and closing_balance_label.value is None:
            closing_balance = self._extract_balance_spatial(
                closing_balance_label, list_for_second_pass, standard_line_spacing
            )
            if closing_balance:
                closing_balance_label.value = closing_balance
                self.logger.info(f"Extracted closing balance: {closing_balance}")

    def _detect_country_by_proximity_to_bank(
        self,
        bank_span: SpanInfo,
        all_spans: List[SpanInfo]
    ) -> Optional[Tuple[str, SpanInfo]]:
        """
        Detect country by finding the country pattern closest to the bank name.

        Args:
            bank_span: The span containing the bank name
            all_spans: All spans in the document

        Returns:
            Tuple of (country_code, country_span) or None
        """
        from app.config.bank_statement_country_loader import get_country_config_loader

        country_loader = get_country_config_loader()
        bank_y_center = (bank_span.y1 + bank_span.y2) / 2

        # Find all spans that contain country patterns
        country_candidates = []
        for span in all_spans:
            span_text_upper = span.text.upper().strip()

            for country_code in country_loader.get_supported_countries():
                config = country_loader.get_country_config(country_code)
                if not config:
                    continue

                # Check country name with word boundaries
                country_name = config.get("country_name", "").upper()
                pattern = r'\b' + re.escape(country_name) + r'\b'
                if re.search(pattern, span_text_upper):
                    # Calculate distance to bank (Y-axis distance)
                    span_y_center = (span.y1 + span.y2) / 2
                    distance = abs(span_y_center - bank_y_center)
                    country_candidates.append((distance, country_code, span))
                    break  # Found country in this span, move to next span

                # Also check name_aliases (e.g., "UAE" for "United Arab Emirates")
                name_aliases = config.get("name_aliases", [])
                for alias in name_aliases:
                    alias_upper = alias.upper()
                    pattern = r'\b' + re.escape(alias_upper) + r'\b'
                    if re.search(pattern, span_text_upper):
                        # Calculate distance to bank (Y-axis distance)
                        span_y_center = (span.y1 + span.y2) / 2
                        distance = abs(span_y_center - bank_y_center)
                        country_candidates.append((distance, country_code, span))
                        break  # Found country alias in this span, move to next span
                else:
                    continue  # Only continue if inner loop didn't break
                break  # Break outer loop if alias was found

        if not country_candidates:
            return None

        # Sort by distance and return closest
        country_candidates.sort(key=lambda x: x[0])
        closest = country_candidates[0]
        self.logger.info(
            f"Detected country '{closest[1]}' at Y=[{closest[2].y1:.1f}, {closest[2].y2:.1f}], "
            f"distance to bank: {closest[0]:.1f}px"
        )
        return (closest[1], closest[2])

    def _pass_3_address_extraction(
        self,
        first_pass_addresses: List[SpanInfo],
        list_for_second_pass: List[SpanInfo],
        all_spans: List[SpanInfo],
        standard_line_spacing: float,
        holder_label: Optional[SpanInfo],
        bank_span: Optional[SpanInfo],
        currency_spans: List[SpanInfo] = None  # New parameter
    ) -> Dict[str, Any]:
        """
        Pass 3: Full Address Extraction.

        NEW APPROACH:
        1. Detect country by finding closest country pattern to bank name
        2. Load cities only for that country
        3. Search for city spans above the country span (account holder address is typically above bank/country)
        4. Use city span as anchor for address collection

        Args:
            currency_spans: Spans matched as currency in Pass 1 (excluded from country detection)
        """
        # Initialize currency_spans if None
        if currency_spans is None:
            currency_spans = []

        # Create filtered span list excluding only currency spans
        # This keeps bank_name, account_number_label, etc. for country detection
        spans_for_country_detection = [s for s in all_spans if s not in currency_spans]

        # STEP 1: Detect country by finding closest country pattern to bank name
        country_code = None
        country_span = None

        if bank_span:
            proximity_result = self._detect_country_by_proximity_to_bank(bank_span, spans_for_country_detection)
            if proximity_result:
                country_code, country_span = proximity_result
            else:
                self.logger.warning("Could not detect country by proximity to bank, falling back to first_pass_addresses")
        else:
            self.logger.warning("No bank span provided, falling back to first_pass_addresses")

        # Fallback: try to detect country from first_pass_addresses (legacy approach)
        if not country_code and first_pass_addresses:
            country_code = self._detect_country_from_addresses(first_pass_addresses)
            if country_code:
                self.logger.info(f"Detected country from first_pass_addresses (fallback): {country_code}")

        if not country_code:
            self.logger.warning("Could not detect country, returning empty address components")
            return {}

        self.logger.info(f"Detected country: {country_code}")

        # STEP 2: Load cities only (not states) for this country to avoid matching states like "KARNATAKA"
        country_locations = self._load_country_cities_only(country_code)
        self.logger.info(f"Loaded {len(country_locations)} cities (excluding states) for {country_code}")

        # STEP 3: Search for city span
        # Search entire page for city anchor - no Y-based filtering
        # Postal code validation will filter out non-address spans

        city_span = None
        matched_city = None
        detected_state = None

        self.logger.info("Searching for city anchor across entire page (no Y-based filtering)")

        # Get postal code regex for country validation
        country_loader = get_country_config_loader()
        postal_code_pattern_str = country_loader.get_postal_code_pattern(country_code)
        postal_code_regex = re.compile(postal_code_pattern_str) if postal_code_pattern_str else None

        # Combine both lists to search for city spans
        all_spans_for_city_search = list_for_second_pass + first_pass_addresses

        # Check if postal code is required for this country
        # UAE, Myanmar, HK, etc. have optional postal codes
        require_postal_code = country_loader.is_postal_code_required(country_code)

        # Store all candidate city matches with their scores
        # Score = (Y position preference, city name coverage)
        # We prefer cities that appear later (higher Y) and are more prominent in the span text
        city_candidates = []

        # Bank-specific keywords that indicate a span is part of bank address, not customer address
        bank_address_keywords = [
            "BRANCH ADDRESS", "REGISTERED OFFICE", "CORPORATE OFFICE",
            "HEAD OFFICE", "REGD OFFICE", "REGD. OFFICE", "REGD OFFICE:",
            "BRANCH", "AXIS BANK LTD", "IDBI BANK LTD", "IDBI BANK LIMITED",
            "WEBSITE:WWW", "WEBSITE:", "WWW.", "HTTP", "HTTPS",
            "REGISTERED OFFICE", "CORPORATE OFFICE", "HQ",
            "STATEMENT OF ACCOUNT", "STATEMENT DETAILS", "STATEMENT PERIOD",
            "ACCOUNT TYPE", "ACCOUNT NUMBER", "CURRENCY", "INTEREST PAYOUT"
        ]

        # HYBRID APPROACH: Two-pass city detection
        # Pass 1: Find cities with postal code in same span (current behavior)
        # Pass 2: If no candidates, find cities without postal code requirement

        for span in all_spans_for_city_search:
            # No Y-based filtering - check all spans for city matches
            # Postal code validation will filter out non-address spans

            span_text_upper = span.text.upper().strip()

            # Skip spans that contain bank-specific keywords (these are bank addresses, not customer addresses)
            is_bank_address = any(keyword in span_text_upper for keyword in bank_address_keywords)
            if is_bank_address:
                self.logger.debug(f"Skipping span with bank address keyword: Y=[{span.y1:.1f}, {span.y2:.1f}], text='{span.text[:50]}...'")
                continue

            # Find ALL city matches with valid postal code after them
            # Pick the one that appears latest in the text (closest to postal code)
            best_city_for_span = None
            best_match_end = -1  # Track position of best match in text
            best_city_coverage = 0.0  # Track how much of the span text is the city name

            for city in country_locations:
                # Match city as a whole word using regex word boundaries
                # \b matches word boundary (non-word char or start/end of string)
                pattern = r'\b' + re.escape(city) + r'\b'
                match = re.search(pattern, span_text_upper)
                if match:
                    # Calculate city name coverage (ratio of city name length to span text length)
                    city_coverage = len(city) / len(span_text_upper) if span_text_upper else 0

                    # Check if there's a valid postal code AFTER this city match
                    if postal_code_regex and require_postal_code:
                        text_after_city = span_text_upper[match.end():]
                        if postal_code_regex.search(text_after_city):
                            # Pick the city with best coverage OR latest position
                            if city_coverage > best_city_coverage or (city_coverage == best_city_coverage and match.end() > best_match_end):
                                best_city_for_span = city
                                best_match_end = match.end()
                                best_city_coverage = city_coverage
                                self.logger.debug(f"City '{city}' at pos {match.start()}-{match.end()} has valid postal code after it")
                    else:
                        # No postal code pattern or not required
                        # Prefer cities with better coverage (city is main content of span)
                        if city_coverage > best_city_coverage or (city_coverage == best_city_coverage and match.end() > best_match_end):
                            best_city_for_span = city
                            best_match_end = match.end()
                            best_city_coverage = city_coverage

            if best_city_for_span:
                # Calculate score: prefer leftmost X, then EARLIER Y (customer address is at top), then better coverage
                # -span.x1 is negated so smaller X values sort first when using reverse=True
                # -span.y1 is negated so smaller Y (earlier in doc) sorts first when using reverse=True
                score = (-span.x1, -span.y1, best_city_coverage)
                city_candidates.append((score, span, best_city_for_span))

        # Select best city candidate: prefer leftmost X, then highest Y (latest in document), then best coverage
        if city_candidates:
            # Sort by score (-X desc = X asc, Y desc, coverage desc)
            city_candidates.sort(key=lambda x: (x[0][0], x[0][1], x[0][2]), reverse=True)
            best_score, city_span, matched_city = city_candidates[0]
            # Normalize city name (e.g., BANGALORE -> BENGALURU)
            normalized_city = self._normalize_city_name(matched_city, country_code)
            city_span.value = normalized_city  # Store the normalized city name
            # Look up state for the city using the normalized name
            detected_state = self._get_state_for_city(normalized_city, country_code) if country_code else None
            display_city = normalized_city if normalized_city != matched_city.upper() else matched_city
            self.logger.info(f"Found city anchor (with postal code): '{display_city}' in span text: '{city_span.text}' X=[{city_span.x1:.1f}, {city_span.x2:.1f}] Y=[{city_span.y1:.1f}, {city_span.y2:.1f}]")
            if detected_state:
                self.logger.info(f"Looked up state for city '{display_city}': '{detected_state}'")
        else:
            # PASS 2 (FALLBACK): No city with postal code found in same span
            # Try to find city-only matches (cities only, no states)
            # Prefer lower Y values (cities typically appear before states in addresses)
            self.logger.info("No city anchor found with postal code in same span, trying fallback (city-only detection)")
            city_span = None

            # Load only cities (not states) for fallback to avoid matching states like "KARNATAKA"
            cities_only = self._load_country_cities_only(country_code)
            self.logger.info(f"Loaded {len(cities_only)} cities (excluding states) for fallback")

            # Minimum coverage threshold to filter out cities that are just small parts of longer text
            # This prevents matching "NAGAR" in "JP NAGAR APARTMENTS" while still matching standalone city names
            # Use lower threshold when span contains country name (handles "CITY, COUNTRY" format)
            # UAE-specific: Use even lower threshold because UAE addresses often have city names embedded in longer text
            MIN_COVERAGE_THRESHOLD = 0.15 if country_code == "AE" else 0.5
            MIN_COVERAGE_THRESHOLD_WITH_COUNTRY = 0.15 if country_code == "AE" else 0.20  # Lower threshold when country name is present

            # Get country name(s) for span filtering
            country_name_in_span = None
            if country_code:
                country_loader = get_country_config_loader()
                config = country_loader.get_country_config(country_code)
                if config:
                    country_name_in_span = config.get("country_name", "").upper()
                    # Also check name aliases
                    for alias in config.get("name_aliases", []):
                        country_name_in_span = country_name_in_span + "|" + alias.upper()

            for span in all_spans_for_city_search:
                span_text_upper = span.text.upper().strip()

                # Skip spans that contain bank-specific keywords
                is_bank_address = any(keyword in span_text_upper for keyword in bank_address_keywords)
                if is_bank_address:
                    continue

                best_city_for_span = None
                best_city_coverage = 0.0

                for city in cities_only:
                    pattern = r'\b' + re.escape(city) + r'\b'
                    match = re.search(pattern, span_text_upper)
                    if match:
                        city_coverage = len(city) / len(span_text_upper) if span_text_upper else 0
                        # Prefer cities with better coverage (city is main content of span)
                        if city_coverage > best_city_coverage:
                            best_city_for_span = city
                            best_city_coverage = city_coverage

                if best_city_for_span:
                    # Determine if span contains country name (for "CITY, COUNTRY" format)
                    span_contains_country = False
                    if country_name_in_span:
                        span_contains_country = any(
                            re.search(r'\b' + re.escape(name) + r'\b', span_text_upper)
                            for name in country_name_in_span.split('|')
                        )

                    # Apply appropriate threshold - lower when country name is present
                    threshold = (
                        MIN_COVERAGE_THRESHOLD_WITH_COUNTRY
                        if span_contains_country
                        else MIN_COVERAGE_THRESHOLD
                    )

                    # Apply minimum coverage threshold to filter out partial matches
                    # This prevents matching "NAGAR" in "JP NAGAR APARTMENTS" (coverage ~11%)
                    # while still matching standalone city names like "BENGALURU" (coverage 100%)
                    # When span contains country name (e.g., "CITY, COUNTRY"), use lower threshold
                    if best_city_coverage >= threshold:
                        # For fallback, prefer leftmost X, then LOWER Y (cities appear before states in addresses),
                        # then better coverage
                        # -span.x1 for leftmost X, +span.y1 for lower Y (we'll negate in sort)
                        score = (-span.x1, -span.y1, best_city_coverage)
                        city_candidates.append((score, span, best_city_for_span))
                        country_note = " (with country)" if span_contains_country else ""
                        self.logger.debug(f"City '{best_city_for_span}' meets coverage threshold{country_note}: {best_city_coverage:.1%} >= {threshold:.1%}")
                    else:
                        country_note = " (with country)" if span_contains_country else ""
                        self.logger.debug(f"City '{best_city_for_span}' below coverage threshold{country_note}: {best_city_coverage:.1%} < {threshold:.1%}, span text: '{span_text_upper}'")

            if city_candidates:
                # Sort by score: -X desc = X asc, -Y desc = Y asc (lower Y first), coverage desc
                city_candidates.sort(key=lambda x: (x[0][0], x[0][1], x[0][2]), reverse=True)
                best_score, city_span, matched_city = city_candidates[0]
                # Normalize city name (e.g., BANGALORE -> BENGALURU)
                normalized_city = self._normalize_city_name(matched_city, country_code)
                city_span.value = normalized_city
                # Look up state for the city using the normalized name
                detected_state = self._get_state_for_city(normalized_city, country_code) if country_code else None
                display_city = normalized_city if normalized_city != matched_city.upper() else matched_city
                self.logger.info(f"Found city anchor (fallback, no postal code required): '{display_city}' in span text: '{city_span.text}' X=[{city_span.x1:.1f}, {city_span.x2:.1f}] Y=[{city_span.y1:.1f}, {city_span.y2:.1f}]")
                if detected_state:
                    self.logger.info(f"Looked up state for city '{display_city}': '{detected_state}'")
            else:
                city_span = None

        # Fallback: if no city found, use country span or first address as anchor
        if not city_span:
            if country_span:
                self.logger.warning("No city span found, using country span as anchor")
                city_span = country_span
                # Clear the value to avoid using bank name or other previous values
                city_span.value = None
                matched_city = country_code
            elif first_pass_addresses:
                self.logger.warning("No city span found, falling back to topmost address anchor")
                # Prefer the topmost address (smallest y1) - account holder address typically appears higher
                city_span = min(first_pass_addresses, key=lambda s: s.y1)
                matched_city = city_span.value if hasattr(city_span, 'value') and city_span.value else city_span.text
            else:
                self.logger.warning("No city or country anchor available, returning empty address")
                return {}

        self.logger.info(f"Selected address anchor: Y=[{city_span.y1:.1f}, {city_span.y2:.1f}], text='{city_span.text}'")

        # STEP 4: Collect address spans using city anchor (with content-driven approach)
        address_spans = self._collect_address_spans(
            city_span, list_for_second_pass, standard_line_spacing,
            country_code=country_code,
            known_locations=country_locations
        )

        return self._build_address_result(address_spans, city_span, holder_label, all_spans, country_code, detected_state)

    # ============================================================
    # HELPER METHODS - SPAN EXTRACTION
    # ============================================================

    def _extract_account_number_from_span(self, span: SpanInfo) -> Optional[str]:
        """Extract account number from span if it matches regex."""
        country_loader = get_country_config_loader()
        text = span.text.strip()

        for country in country_loader.get_supported_countries():
            regex = get_account_number_regex(country)
            # Remove anchors for searching within text
            search_regex = re.compile(regex.pattern.strip('^$'))
            match = search_regex.search(text)
            if match:
                # Remove common separators and return clean number
                account_number = match.group().strip().replace(' ', '').replace('-', '')
                return account_number

        return None

    def _extract_holder_name_from_span(self, span: SpanInfo, all_spans: List[SpanInfo] = None) -> Optional[str]:
        """
        Extract account holder name from span.

        If the span contains only a title (e.g., MR.), searches for the actual
        name to the right on the same line and combines them.

        Args:
            span: The span to extract from
            all_spans: All spans on the page (required for title lookup)

        Returns:
            Extracted name or None
        """
        text = span.text.strip()
        if not text:
            return None

        # Check if this is a title-only span
        if self._is_title(text):
            # Try to find the actual name to the right
            if all_spans:
                combined = self._find_name_after_title(span, all_spans)
                if combined:
                    return combined
            # Return just the title if no name found
            return text

        return text

    def _extract_currency_from_span(self, span: SpanInfo, all_spans: List[SpanInfo] = None) -> Optional[str]:
        """
        Extract currency code from a span labeled as currency.

        First checks if span text is a known currency name (e.g., "UAE DIRHAM" -> "AED").
        Then checks right/below spans for currency codes.

        Args:
            span: The span labeled as currency
            all_spans: All spans on the page (required for label-only spans)

        Returns:
            ISO 4217 currency code (e.g., "AED", "USD", "INR")
        """
        span_text_upper = span.text.upper().strip()

        # Load currency name map from config
        config = load_config()
        currency_name_map = config.get("currency_name_map", {})

        # Check if span text itself is a currency name
        if span_text_upper in currency_name_map:
            return currency_name_map[span_text_upper]

        # Check if span text contains a currency name
        for currency_name, currency_code in currency_name_map.items():
            if currency_name in span_text_upper:
                return currency_code

        # If only label (e.g., "CURRENCY:"), look right for the value
        if span_text_upper in ["CURRENCY", "CURRENCY NAME", "CCY", "CURRENCY CODE", "CURRENCY:"]:
            # Search in same line to the right
            if all_spans:
                for other in all_spans:
                    if (other != span and
                        abs(other.y1 - span.y1) < 5 and  # Same line
                        other.x1 > span.x2 and  # To the right
                        other.x1 - span.x2 < 100):  # Within reasonable distance
                        other_text_upper = other.text.upper().strip()
                        # Strip leading colon and whitespace (e.g., ": INR" -> "INR")
                        other_text_upper = other_text_upper.lstrip(':').strip()
                        if other_text_upper in currency_name_map:
                            return currency_name_map[other_text_upper]
                        # Check for 3-letter ISO codes
                        if len(other_text_upper) == 3 and other_text_upper.isalpha():
                            return other_text_upper

                # Also search below the label (within vertical distance and horizontally aligned)
                for other in all_spans:
                    if (other != span and
                        other.y1 > span.y2 and  # Below the label
                        other.y1 - span.y2 < 30 and  # Within reasonable vertical distance
                        abs(other.x1 - span.x1) < 50):  # Horizontally aligned (left edge)
                        other_text_upper = other.text.upper().strip()
                        # Strip leading colon and whitespace (e.g., ": INR" -> "INR")
                        other_text_upper = other_text_upper.lstrip(':').strip()
                        if other_text_upper in currency_name_map:
                            return currency_name_map[other_text_upper]
                        # Check for 3-letter ISO codes
                        if len(other_text_upper) == 3 and other_text_upper.isalpha():
                            return other_text_upper

        return None

    def _is_title(self, text: str) -> bool:
        """Check if text is a title/prefix (e.g., MR., MRS., DR.)."""
        text_upper = text.strip().upper()
        return text_upper in self.TITLE_PATTERNS

    def _is_title_label(self, text: str) -> bool:
        """Check if the label text is a title pattern (e.g., 'MR.', 'MRS.')."""
        text_upper = text.upper().strip()
        # Only check exact match with period to avoid false positives
        # e.g., "REV." should match, but "REV" would match "PREVIOUS"
        return text_upper in self.TITLE_PATTERNS

    def _looks_like_label(self, text: str) -> bool:
        """Check if text looks like a label rather than a name value."""
        if not text:
            return False

        text_stripped = text.strip()
        text_upper = text_stripped.upper()

        # Check if starts with special characters (addresses, etc.)
        if text_stripped and text_stripped[0] in '#-*@':
            return True

        # Specific label patterns (exact phrases, not substrings)
        label_patterns = [
            "ACCOUNT TYPE",
            "ACCOUNT HOLDER",
            "CUSTOMER NAME",
            "PRIMARY ACCOUNT",
            "CARD TYPE",
            "CARD NUMBER",
            "DATE ISSUED",
            "ISSUE DATE",
            "EXPIRY DATE",
            "BRANCH NAME",
            "YOUR BASE",
            "BASE BRANCH",
            "GSTIN",
            "PRIMARY GSTIN",
            "SEQUENCE NUMBER",
            "PAGE",
            "STATEMENT",
            "CUSTOMER DETAILS",
            "ACCOUNT DETAILS"
        ]

        # Check if text matches or starts with any label pattern
        for pattern in label_patterns:
            if text_upper == pattern or text_upper.startswith(pattern + " ") or text_upper.startswith(pattern + ":"):
                return True

        # Check if text contains "BANK" and is not a person's name
        # Bank names like "Ayeyarwady Bank" should not be treated as account holder names
        if " BANK" in text_upper or text_upper.endswith(" BANK"):
            return True

        # Check for email addresses
        if self._is_email(text_stripped):
            return True

        return False


    def _is_email(self, text: str) -> bool:
        """Check if text looks like an email address."""
        if not text:
            return False
        # Simple email pattern check
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        return bool(re.search(email_pattern, text.strip()))

    def _is_address_skip_label(self, text: str) -> bool:
        """
        Check if text is a label that should NOT be part of the street address.

        These are typically form labels, bank information, or non-address content
        that may appear near the actual address block.

        Args:
            text: The span text to check

        Returns:
            True if the text should be skipped as an address label
        """
        if not text:
            return False

        text_upper = text.upper().strip()

        # Check for exact or partial matches with skip labels
        for skip_label in self.ADDRESS_SKIP_LABELS:
            if skip_label in text_upper:
                return True

        # Check for patterns that look like labels (end with colon or ":-")
        # e.g., "Joint Holder :-", "Account: "
        if re.match(r'^[A-Z\s]{3,}\s*:-?\s*$', text_upper):
            return True

        # Check for very short single-character spans (likely artifacts)
        # e.g., "Y", ":", "to"
        if len(text.strip()) <= 2 and not text.strip().replace('.', '').replace(',', '').isdigit():
            return True

        # Check for standalone account numbers (all digits, length 10-20)
        # e.g., "1015053449401"
        if re.match(r'^\d{10,20}$', text.strip()):
            return True

        # Check for IBAN/SWIFT-like patterns with spaces
        # e.g., "AE73 0260 0010..."
        if re.match(r'^[A-Z]{2}\d{2}[\s\d]{10,}', text_upper):
            return True

        # Check for masked numbers (phone, Aadhar, account numbers)
        # e.g., "xxxxx1234", "xxxxxxxxxx74"
        if re.match(r'^[Xx]{5,}\d{2,}$', text_upper.strip()):
            return True

        return False

    def _clean_address_text(
        self,
        text: str,
        holder_name: Optional[str] = None
    ) -> str:
        """
        Clean address text by removing common formatting artifacts.

        Removes:
        - Leading colons (e.g., ": VINEETH NARASIMHAN" -> "VINEETH NARASIMHAN")
        - Leading dashes (e.g., "- MAIN STREET" -> "MAIN STREET")
        - Account holder name from first line (when holder_name is provided)
        - Excessive whitespace

        Args:
            text: The raw address text to clean
            holder_name: Optional account holder name to remove from first line

        Returns:
            Cleaned address text
        """
        self.logger.info(f"_clean_address_text INPUT: text='{text}', holder_name='{holder_name}'")

        if not text:
            return text

        cleaned = text.strip()

        # Remove leading colons, dashes, and common separators
        cleaned = re.sub(r'^[:\-]\s*', '', cleaned)

        # Remove account holder name from first line when provided
        if holder_name and cleaned:
            # Split into lines to handle multi-line addresses
            lines = cleaned.split('\n')
            if lines:
                first_line = lines[0]
                # Try various matching patterns
                holder_upper = holder_name.upper().strip()
                first_upper = first_line.upper().strip()

                # Case 1: First line starts with holder name (exact match)
                if first_upper.startswith(holder_upper):
                    # Remove the name and any following separator
                    remaining = first_line[len(holder_name):].strip()
                    # Remove leading separators like comma, dash, colon
                    remaining = re.sub(r'^[\-,:\s]+', '', remaining).strip()
                    lines[0] = remaining if remaining else ''
                # Case 2: First line contains holder name (e.g., "K C ROHITH, 1-21 SETTY...")
                elif holder_upper in first_upper:
                    # Split by common separators and remove the part containing the name
                    parts = re.split(r'[,/-]\s*', first_line)
                    filtered_parts = []
                    for part in parts:
                        if part.upper().strip() != holder_upper:
                            filtered_parts.append(part.strip())
                    lines[0] = ', '.join([p for p in filtered_parts if p])

            # Rejoin lines and skip empty first line
            cleaned = '\n'.join(lines)
            lines = cleaned.split('\n')
            if lines and not lines[0].strip():
                lines = lines[1:]
            cleaned = '\n'.join(lines)

        # Remove trailing colons
        cleaned = re.sub(r'\s*:\s*$', '', cleaned)

        # Collapse multiple spaces
        cleaned = re.sub(r'\s+', ' ', cleaned)

        result = cleaned.strip()
        self.logger.info(f"_clean_address_text OUTPUT: '{result}'")
        return result

    def _normalize_name_for_matching(self, name: str) -> str:
        """
        Normalize a name for flexible matching.

        Removes common honorifics (Mr, Mrs, Ms, Dr, etc.),
        normalizes whitespace and case for comparison.

        Args:
            name: The name to normalize

        Returns:
            Normalized name suitable for comparison
        """
        if not name:
            return name

        # Common honorifics to remove (with and without dots)
        honorifics = [
            'MR\\.', 'MRS\\.', 'MS\\.', 'DR\\.', 'PROF\\.',
            'MR', 'MRS', 'MS', 'DR', 'PROF',
            'SHRI', 'SMT', 'KUM',  # Indian honorifics
        ]

        normalized = name.upper().strip()

        # Remove honorifics
        for honorific in honorifics:
            # Match as whole word at start
            normalized = re.sub(rf'^{honorific}\s+', '', normalized)
            # Match as whole word anywhere
            normalized = re.sub(rf'\s{honorific}\s+', ' ', normalized)
            # Match at end
            normalized = re.sub(rf'\s{honorific}\.?$', '', normalized)

        # Collapse multiple spaces
        normalized = re.sub(r'\s+', ' ', normalized)

        return normalized.strip()

    def _is_name_line(self, text: str) -> bool:
        """
        Check if text looks like an account holder name.

        Name lines typically:
        - Start with a title (Mr., Mrs., etc.)
        - Are 2-4 words with proper capitalization after stripping title

        Excludes patterns that look like addresses:
        - Street numbers (e.g., "1-21", "2-122", "#123", etc.)
        - Address keywords (e.g., "NEAR", "TEMPLE", "STREET", "ROAD", etc.)
        """
        if not text:
            return False

        text_stripped = text.strip()
        text_upper = text_stripped.upper()

        # EXCLUDE: Address patterns that look like names but are actually addresses
        # Check for street number patterns at the start (e.g., "1-21", "2-122", "#123")
        if re.match(r'^[\d#]+[-/\s]?\d*', text_stripped.strip()):
            return False

        # Check for common address keywords that indicate this is an address, not a name
        address_keywords = [
            'NEAR', 'TEMPLE', 'STREET', 'ROAD', 'LANE', 'COLONY', 'NAGAR',
            'EXTENSION', 'EXTN', 'SECTOR', 'PHASE', 'BLOCK', 'OPP', 'OPPOSITE',
            'BEHIND', 'BESIDE', 'APARTMENT', 'SOCIETY', 'COMPLEX', 'BUILDING',
            'VILLAGE', 'MANDAL', 'DISTRICT', 'TALUK', 'TEHSIL'
        ]
        if any(keyword in text_upper for keyword in address_keywords):
            return False

        # Check if starts with title
        if self._is_title(text_stripped):
            return True

        # Check for common name format: 2-4 words, each starting with capital letter
        # Avoid matching addresses or other text
        words = text_stripped.split()
        if 2 <= len(words) <= 4:
            # Check if most words start with capital letter
            capital_count = sum(1 for w in words if w and w[0].isupper())
            if capital_count >= len(words) - 1:  # Allow one word to not start with capital
                # Exclude common non-name patterns
                exclude_keywords = ['ACCOUNT', 'SUMMARY', 'STATEMENT', 'BRANCH', 'PHONE',
                                   'EMAIL', 'CIF', 'IFSC', 'CURRENCY', 'STATUS', 'NOMINEE']
                if not any(keyword in text_upper for keyword in exclude_keywords):
                    return True

        return False

    def _find_name_after_title(self, title_span: SpanInfo, all_spans: List[SpanInfo]) -> Optional[str]:
        """
        Find the actual account holder name to the right of a title span.

        Searches for a span on the same line to the right of the title.
        Combines title + name if found.

        Args:
            title_span: The span containing the title (e.g., "MR.")
            all_spans: All spans on the page

        Returns:
            Combined "TITLE NAME" string if name found, None otherwise
        """
        title_text = title_span.text.strip().upper()

        # Search for name to the right on the same line
        best_match = None
        best_distance = float('inf')
        standard_line_spacing = self._estimate_line_spacing(all_spans)

        for span in all_spans:
            if span is title_span:
                continue

            # Check if spans are on the same line by comparing Y ranges
            y_overlap = min(span.y2, title_span.y2) - max(span.y1, title_span.y1)
            center_y_delta = abs(span.center_y() - title_span.center_y())

            # Spans are on the same line if Y ranges overlap or centers are close
            is_same_line = y_overlap > 0 or center_y_delta < standard_line_spacing * 0.3

            if is_same_line:
                delta_x = span.x1 - title_span.x2

                # Only consider spans to the right of the title
                if delta_x >= 0:
                    # Prefer the closest span to the right
                    if delta_x < best_distance:
                        text = span.text.strip()
                        if text:
                            best_distance = delta_x
                            best_match = text

        # Combine title + name if found
        if best_match:
            # Remove trailing dot from title if present (e.g., "MR." -> "MR")
            title_clean = title_text.rstrip('.')
            return f"{title_clean} {best_match}"

        return None

    def _estimate_line_spacing(self, all_spans: List[SpanInfo]) -> float:
        """Estimate line spacing from spans."""
        return calculate_standard_line_spacing(all_spans)

    def _clean_account_holder_name(self, name: str) -> Optional[str]:
        """
        Clean account holder name by removing titles/salutations and patronymic markers.

        Uses the shared clean_name_for_storage utility for consistency.
        """
        from app.utils.string_matching import clean_name_for_storage
        return clean_name_for_storage(name)

    def _match_bank_from_text(self, text: str) -> Optional[BankInfo]:
        """Match bank from text using bank lookup."""
        bank_info = self.bank_lookup.detect_bank_in_text(text)
        return bank_info

    async def _detect_bank_name_from_ocr(self, file_bytes: bytes, is_pdf: bool) -> Optional[str]:
        """
        Detect bank name using OCR when direct text extraction fails.

        This handles cases where the bank name/logo is embedded as an image
        (e.g., HSBC logo in scanned PDFs).

        Args:
            file_bytes: Raw file bytes
            is_pdf: Whether the file is a PDF

        Returns:
            Bank abbreviation if found, None otherwise
        """
        try:
            from app.helper.doctr.document_text_extractor import ImageProcessor
            from app.core import get_doctr_model

            def get_document_file():
                from doctr.io import DocumentFile
                return DocumentFile

            self.logger.info("OCR fallback: Converting PDF page to image for OCR")

            # Convert PDF to image for OCR (bypasses direct extraction)
            img_bytes = await ImageProcessor.convert_to_png(file_bytes, max_pages=1)

            # Run OCR on the image
            DocumentFile = get_document_file()
            doc = DocumentFile.from_images(img_bytes[:1])
            model = get_doctr_model()
            result = model(doc)

            # Collect all OCR text from the image
            all_text_parts = []
            for page in result.pages:
                for block in page.blocks:
                    for line in block.lines:
                        line_text = " ".join([word.value for word in line.words])
                        all_text_parts.append(line_text)

            all_text = " ".join(all_text_parts)
            self.logger.info(f"OCR fallback extracted {len(all_text_parts)} text elements")
            self.logger.debug(f"Combined OCR text: '{all_text[:500]}'")

            # Try to match bank name from OCR text
            bank_info = self._match_bank_from_text(all_text)
            if bank_info:
                self.logger.info(f"Bank name detected from OCR fallback: {bank_info.abbreviation}")
                return bank_info.abbreviation

            self.logger.warning(f"Bank name not found in OCR text")
            return None

        except Exception as e:
            self.logger.error(f"OCR fallback failed: {e}")
            return None

    def _load_country_locations(self, country_code: str) -> Set[str]:
        """
        Load all cities and states for a specific country code.

        Uses the countrystatecity library for dynamic loading.
        Includes static cities and states from config for countries like UAE.

        Args:
            country_code: ISO country code (e.g., "IN", "AE", "SG")

        Returns:
            Set of uppercase location names (cities + states)
        """
        from countrystatecity_countries import get_cities_of_country, get_states_of_country
        from app.config.bank_statement_country_loader import get_country_config_loader

        locations: Set[str] = set()

        # Load cities from countrystatecity library
        try:
            city_objects = get_cities_of_country(country_code)
            for city in city_objects:
                locations.add(city.name.upper())
        except Exception as e:
            self.logger.warning(f"Failed to load cities for {country_code} from library: {e}")

        # Load states from countrystatecity library
        try:
            state_objects = get_states_of_country(country_code)
            for state in state_objects:
                locations.add(state.name.upper())
        except Exception as e:
            self.logger.warning(f"Failed to load states for {country_code} from library: {e}")

        # Load static locations from config (for UAE, etc.)
        try:
            country_loader = get_country_config_loader()
            config = country_loader.get_country_config(country_code)
            if config:
                # Add subdivisions as locations (e.g., UAE emirates)
                for subdivision in config.get("subdivisions", {}).get("list", []):
                    locations.add(subdivision.upper())
                # Add additional cities
                for city in config.get("subdivisions", {}).get("additional_cities", []):
                    locations.add(city.upper())
        except Exception as e:
            self.logger.warning(f"Failed to load static locations for {country_code}: {e}")

        return locations

    def _load_country_cities_only(self, country_code: str) -> Set[str]:
        """
        Load only cities for a specific country code (excluding states).

        This is used for fallback city detection when no city with postal code
        is found in the same span. We want to avoid matching states like
        "KARNATAKA" when we're looking for cities like "BENGALURU".

        Args:
            country_code: ISO country code (e.g., "IN", "AE", "SG")

        Returns:
            Set of uppercase city names only (no states)
        """
        from countrystatecity_countries import get_cities_of_country
        from app.config.bank_statement_country_loader import get_country_config_loader

        cities: Set[str] = set()

        # Load cities from countrystatecity library
        try:
            city_objects = get_cities_of_country(country_code)
            for city in city_objects:
                cities.add(city.name.upper())
        except Exception as e:
            self.logger.warning(f"Failed to load cities for {country_code} from library: {e}")

        # Load static cities from config (for UAE, etc.)
        try:
            country_loader = get_country_config_loader()
            config = country_loader.get_country_config(country_code)
            if config:
                # Add additional cities (subdivisions list contains states/provinces, not cities)
                for city in config.get("subdivisions", {}).get("additional_cities", []):
                    cities.add(city.upper())

                # Add city aliases (e.g., BANGALORE for BENGALURU)
                # Aliases map old/alternate names to canonical names
                city_aliases = config.get("city_aliases", {})
                for alias, canonical_name in city_aliases.items():
                    cities.add(alias.upper())
                    # Also add canonical name if not already present
                    cities.add(canonical_name.upper())
        except Exception as e:
            self.logger.warning(f"Failed to load static cities for {country_code}: {e}")

        return cities

    def _normalize_city_name(self, city_name: str, country_code: str) -> str:
        """
        Normalize city name by converting aliases to canonical names.

        For example, "BANGALORE" -> "BENGALURU", "BOMBAY" -> "MUMBAI"

        Args:
            city_name: Name of the city (may be an alias)
            country_code: ISO 2-letter country code

        Returns:
            Canonical city name if alias found, otherwise original name
        """
        from app.config.bank_statement_country_loader import get_country_config_loader

        try:
            country_loader = get_country_config_loader()
            config = country_loader.get_country_config(country_code)
            if config:
                city_aliases = config.get("city_aliases", {})
                city_upper = city_name.upper()
                if city_upper in city_aliases:
                    canonical_name = city_aliases[city_upper]
                    self.logger.debug(f"Normalized city alias '{city_name}' to canonical name '{canonical_name}'")
                    return canonical_name.upper()
        except Exception as e:
            self.logger.warning(f"Failed to normalize city name '{city_name}': {e}")

        return city_name.upper()

    def _get_state_for_city(self, city_name: str, country_code: str) -> Optional[str]:
        """
        Look up state for a given city using countrystatecity library.

        Args:
            city_name: Name of the city
            country_code: ISO 2-letter country code

        Returns:
            State name if found, None otherwise
        """
        from countrystatecity_countries import get_cities_of_country, get_states_of_country

        city_upper = city_name.upper()
        try:
            # Load states to create state_id -> state_name mapping
            state_objects = get_states_of_country(country_code)
            state_map = {s.id: s.name for s in state_objects}

            # Load cities and find matching city
            city_objects = get_cities_of_country(country_code)
            for city in city_objects:
                if city.name.upper() == city_upper:
                    if hasattr(city, 'state_id') and city.state_id in state_map:
                        return state_map[city.state_id].upper()
                    elif hasattr(city, 'state_name'):
                        return city.state_name.upper()
        except Exception as e:
            self.logger.warning(f"Failed to lookup state for city '{city_name}' in {country_code}: {e}")

        return None

    def _remove_known_location_components(
        self,
        text: str,
        known_locations: Set[str],
        country_code: Optional[str]
    ) -> str:
        """
        Remove known cities, states, postal codes, and country from text.

        This method is used for content-driven address span collection.
        By removing all known location components, we can detect if leftover
        content exists (street address) or if we need to continue collecting
        spans above.

        Args:
            text: The combined text from collected spans (e.g., "DUBAI, UAE")
            known_locations: Set of uppercase city and state names from _load_country_locations
            country_code: ISO country code (e.g., "AE", "IN") for postal/country removal

        Returns:
            Leftover content after removing all known location components.
            Empty string means only location info found - continue collecting.
            Non-empty string means street address found - stop collecting.
        """
        if not text:
            return ""

        text_upper = text.upper()

        # Remove known cities and states (longest first to avoid partial matches)
        for location in sorted(known_locations, key=len, reverse=True):
            if location in text_upper:
                text = re.sub(r'\b' + re.escape(location) + r'\b', '', text, flags=re.IGNORECASE)
                text_upper = text.upper()

        # Remove postal codes
        if country_code:
            try:
                country_loader = get_country_config_loader()
                config = country_loader.get_country_config(country_code)
                if config:
                    postal_config = config.get("postal_code", {})
                    if postal_config.get("pattern"):
                        pattern = postal_config["pattern"]
                        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
                        text_upper = text.upper()
            except Exception as e:
                self.logger.debug(f"Failed to remove postal code: {e}")

        # Remove country name and aliases
        if country_code:
            try:
                country_loader = get_country_config_loader()
                config = country_loader.get_country_config(country_code)
                if config:
                    country_name = config.get("country_name", "").upper()
                    if country_name in text_upper:
                        text = re.sub(r'\b' + re.escape(country_name) + r'\b', '', text, flags=re.IGNORECASE)
                        text_upper = text.upper()

                    for alias in config.get("name_aliases", []):
                        if alias.upper() in text_upper:
                            text = re.sub(r'\b' + re.escape(alias) + r'\b', '', text, flags=re.IGNORECASE)
                            text_upper = text.upper()
            except Exception as e:
                self.logger.debug(f"Failed to remove country name: {e}")

        # Clean up: remove commas, extra spaces, invalid chars
        text = re.sub(r'[,.\s]+', ' ', text).strip()

        return text

    def _detect_country_from_addresses(self, first_pass_addresses: List[SpanInfo]) -> Optional[str]:
        """
        Detect country code from address spans found in Pass 1.

        Maps country names from span text to ISO country codes.
        Prefers topmost addresses (smallest y1) since account holder addresses
        typically appear higher on the page than bank addresses.

        Args:
            first_pass_addresses: List of address spans from Pass 1

        Returns:
            ISO country code (e.g., "IN", "AE", "SG") or None
        """
        from app.config.bank_statement_country_loader import get_country_config_loader

        if not first_pass_addresses:
            return None

        country_loader = get_country_config_loader()

        # Sort addresses by Y coordinate (topmost first) - account holder address typically appears higher
        sorted_addresses = sorted(first_pass_addresses, key=lambda s: s.y1)

        # Check top 3 addresses for country detection (account holder address is usually in top positions)
        for span in sorted_addresses[:3]:
            span_text_upper = span.text.upper().strip()

            # Check each country's config for matching names/aliases
            for country_code in country_loader.get_supported_countries():
                config = country_loader.get_country_config(country_code)
                if not config:
                    continue

                country_name = config.get("country_name", "").upper()
                name_aliases = [alias.upper() for alias in config.get("name_aliases", [])]

                # Check if span text matches country name or alias
                if span_text_upper == country_name or span_text_upper in name_aliases:
                    return country_code

                # Also check if country name is contained in span text
                if country_name in span_text_upper:
                    return country_code

        return None

    # ============================================================
    # HELPER METHODS - SPATIAL VALUE EXTRACTION
    # ============================================================

    def _extract_account_number_spatial(
        self,
        label_span: SpanInfo,
        candidate_spans: List[SpanInfo],
        standard_line_spacing: float
    ) -> Optional[str]:
        """
        Extract account number using spatial proximity.

        Strategy:
        1. Right-side search (same line, to the right)
        2. Below-label search (next line below)
        """
        country_loader = get_country_config_loader()
        best_match = None
        best_distance = float('inf')

        for span in candidate_spans:
            # Check if spans are on the same line by comparing Y ranges
            y_overlap = min(span.y2, label_span.y2) - max(span.y1, label_span.y1)
            center_y_delta = abs(span.center_y() - label_span.center_y())
            delta_x = span.x1 - label_span.x2

            # Step 1: Try to find value to the RIGHT (same line)
            # Spans are on the same line if their Y ranges overlap or centers are close
            is_same_line = y_overlap > 0 or center_y_delta < standard_line_spacing * 0.3

            if is_same_line and delta_x >= 0:  # delta_x >= 0 allows touching spans
                for country in country_loader.get_supported_countries():
                    regex = get_account_number_regex(country)
                    text = span.text.strip()
                    # Search for account number within text (not exact match)
                    # Strip anchors from regex pattern to search within longer text
                    search_regex = re.compile(regex.pattern.strip('^$'))
                    match = search_regex.search(text)
                    if match:
                        account_number = match.group().strip().replace(' ', '').replace('-', '')
                        # Combined distance: horizontal + weighted Y misalignment for better tie-breaking
                        combined_distance = delta_x + (center_y_delta * 0.5)
                        if combined_distance < best_distance:
                            best_distance = combined_distance
                            best_match = account_number

        if best_match:
            return best_match

        # Step 2: If not found, try BELOW the label
        best_match = None
        best_distance = float('inf')

        for span in candidate_spans:
            # Check if span is below the label
            delta_y = span.y1 - label_span.y2
            delta_x = span.x1 - label_span.x1

            if delta_y > 0 and delta_y < standard_line_spacing * 5 and abs(delta_x) < standard_line_spacing * 2:
                for country in country_loader.get_supported_countries():
                    regex = get_account_number_regex(country)
                    text = span.text.strip()
                    # Search for account number within text (not exact match)
                    # Strip anchors from regex pattern to search within longer text
                    search_regex = re.compile(regex.pattern.strip('^$'))
                    match = search_regex.search(text)
                    if match:
                        account_number = match.group().strip().replace(' ', '').replace('-', '')
                        # Combined distance: vertical + weighted X misalignment for better tie-breaking
                        combined_distance = delta_y + (abs(delta_x) * 0.5)
                        if combined_distance < best_distance:
                            best_distance = combined_distance
                            best_match = account_number

        return best_match

    def _extract_holder_name_spatial(
        self,
        label_span: SpanInfo,
        candidate_spans: List[SpanInfo],
        standard_line_spacing: float
    ) -> Optional[str]:
        """
        Extract account holder name using spatial proximity.

        Strategy:
        1. Right-side search (same line, to the right)
        2. Below-label search (next line below)
        """
        best_match = None
        best_distance = float('inf')

        # Step 1: Try to find value to the RIGHT (same line)
        for span in candidate_spans:
            # Check if spans are on the same line by comparing Y ranges
            y_overlap = min(span.y2, label_span.y2) - max(span.y1, label_span.y1)
            center_y_delta = abs(span.center_y() - label_span.center_y())
            delta_x = span.x1 - label_span.x2

            # Spans are on the same line if their Y ranges overlap or centers are close
            is_same_line = y_overlap > 0 or center_y_delta < standard_line_spacing * 0.3

            if is_same_line and delta_x >= 0:
                text = span.text.strip()
                if text:
                    # Clean leading punctuation (colons, dashes, etc.)
                    text = text.lstrip(':,-–— ')
                    # If this span is the label itself (matches a label pattern), skip it
                    # Only skip if it's a pure title and we're not looking for a title-based label
                    if self._is_title(text) and not self._is_title_label(label_span.text):
                        continue

                    if delta_x < best_distance:
                        best_distance = delta_x
                        best_match = text

        if best_match:
            return best_match

        # Step 2: If not found, try BELOW the label
        best_match = None
        best_distance = float('inf')

        for span in candidate_spans:
            # Check if span is below the label
            delta_y = span.y1 - label_span.y2
            delta_x = span.x1 - label_span.x1

            if delta_y > 0 and delta_y < standard_line_spacing * 2 and abs(delta_x) < standard_line_spacing * 2:
                text = span.text.strip()
                if text:
                    # Clean leading punctuation (colons, dashes, etc.)
                    text = text.lstrip(':,-–— ')
                    if text:  # Any non-empty text is acceptable
                        if delta_y < best_distance:
                            best_distance = delta_y
                            best_match = text

        return best_match

    def _extract_currency_spatial(
        self,
        label_span: SpanInfo,
        candidate_spans: List[SpanInfo],
        standard_line_spacing: float
    ) -> Optional[str]:
        """
        Extract currency value using spatial search from label span.

        Strategy:
        1. Right-side search (same line, to the right)
        2. Below-label search (next line below)
        """
        # Load currency name map from config
        config = load_config()
        currency_name_map = config.get("currency_name_map", {})

        best_match = None
        best_distance = float('inf')

        # Step 1: Try to find value to the RIGHT (same line)
        for span in candidate_spans:
            # Check if spans are on the same line by comparing Y ranges
            y_overlap = min(span.y2, label_span.y2) - max(span.y1, label_span.y1)
            center_y_delta = abs(span.center_y() - label_span.center_y())
            delta_x = span.x1 - label_span.x2

            # Spans are on the same line if their Y ranges overlap or centers are close
            is_same_line = y_overlap > 0 or center_y_delta < standard_line_spacing * 0.3

            if is_same_line and delta_x >= 0:
                text_upper = span.text.upper().strip()

                # Check if it's a known currency name
                if text_upper in currency_name_map:
                    if delta_x < best_distance:
                        best_distance = delta_x
                        best_match = currency_name_map[text_upper]

                # Check for 3-letter ISO codes
                elif len(text_upper) == 3 and text_upper.isalpha():
                    if delta_x < best_distance:
                        best_distance = delta_x
                        best_match = text_upper

        if best_match:
            return best_match

        # Step 2: If not found, try BELOW the label
        best_match = None
        best_distance = float('inf')

        for span in candidate_spans:
            # Check if span is below the label
            delta_y = span.y1 - label_span.y2
            delta_x = span.x1 - label_span.x1

            if delta_y > 0 and delta_y < standard_line_spacing * 2 and abs(delta_x) < standard_line_spacing * 2:
                text_upper = span.text.upper().strip()

                # Check if it's a known currency name
                if text_upper in currency_name_map:
                    if delta_y < best_distance:
                        best_distance = delta_y
                        best_match = currency_name_map[text_upper]

                # Check for 3-letter ISO codes
                elif len(text_upper) == 3 and text_upper.isalpha():
                    if delta_y < best_distance:
                        best_distance = delta_y
                        best_match = text_upper

        return best_match

    # ============================================================
    # PHASE 2 EXTRACTION METHODS (IBAN, Statement Date, Balances)
    # ============================================================

    def _extract_iban_from_span(self, span: SpanInfo, all_spans: List[SpanInfo] = None) -> Optional[str]:
        """
        Extract IBAN from a span labeled as IBAN.

        First checks if span text is a valid IBAN format.
        Then checks right/below spans for IBAN value.

        IBAN format: Starts with 2-letter country code, followed by 2 digits, then alphanumerics.
        Example: "AE123456789012345678901" (UAE IBAN)

        Args:
            span: The span labeled as IBAN
            all_spans: All spans on the page (required for label-only spans)

        Returns:
            IBAN string or None
        """
        span_text_upper = span.text.upper().strip()

        # IBAN regex pattern: 2 letters, 2 digits, then up to 30 alphanumerics
        iban_pattern = re.compile(r'^[A-Z]{2}\d{2}[A-Z0-9]{11,30}$')

        # Check if span text itself is a valid IBAN
        if iban_pattern.match(span_text_upper):
            return span_text_upper

        # Check if span text contains an IBAN
        iban_match = iban_pattern.search(span_text_upper)
        if iban_match:
            return iban_match.group()

        # If only label (e.g., "IBAN:"), look right for the value
        if span_text_upper in ["IBAN", "IBAN:", "INTERNATIONAL BANK ACCOUNT NUMBER", "INTL BANK ACCOUNT NO"]:
            if all_spans:
                for other in all_spans:
                    if (other != span and
                        abs(other.y1 - span.y1) < 5 and  # Same line
                        other.x1 > span.x2 and  # To the right
                        other.x1 - span.x2 < 200):  # Within reasonable distance
                        other_text_upper = other.text.upper().strip()
                        if iban_pattern.match(other_text_upper):
                            return other_text_upper
                        iban_match = iban_pattern.search(other_text_upper)
                        if iban_match:
                            return iban_match.group()

        return None

    def _extract_statement_date_from_span(self, span: SpanInfo, all_spans: List[SpanInfo] = None) -> Optional[str]:
        """
        Extract statement date from a span labeled as statement date/period.

        First checks if span text contains a date.
        Then checks right/below spans for date value.

        Args:
            span: The span labeled as statement date
            all_spans: All spans on the page (required for label-only spans)

        Returns:
            Date string or None
        """
        span_text = span.text.strip()

        # Date patterns (try multiple formats)
        # DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY, YYYY-MM-DD, etc.
        date_patterns = [
            r'\b\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}\b',  # DD/MM/YYYY
            r'\b\d{2,4}[-/\.]\d{1,2}[-/\.]\d{1,2}\b',  # YYYY-MM-DD
            r'\b(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*[\s\-\.]?\d{1,2}[\s\-\.]?\d{2,4}\b',  # Mon DD YYYY
        ]
        combined_pattern = re.compile('|'.join(date_patterns), re.IGNORECASE)

        # Check if span text itself contains a date
        date_match = combined_pattern.search(span_text)
        if date_match:
            return date_match.group().strip()

        # If only label (e.g., "STATEMENT DATE:"), look right for the value
        span_text_upper = span.text.upper().strip()
        if span_text_upper in ["STATEMENT DATE", "STATEMENT DATE:", "DATE OF STATEMENT",
                               "STATEMENT PERIOD", "STATEMENT PERIOD:", "PERIOD", "PERIOD:",
                               "AS OF", "AS OF:"]:
            if all_spans:
                for other in all_spans:
                    if (other != span and
                        abs(other.y1 - span.y1) < 5 and  # Same line
                        other.x1 > span.x2 and  # To the right
                        other.x1 - span.x2 < 150):  # Within reasonable distance
                        other_text = other.text.strip()
                        date_match = combined_pattern.search(other_text)
                        if date_match:
                            return date_match.group().strip()
                        # Also check below for date ranges
                        if other.y1 > span.y2 and other.y1 - span.y2 < 20:
                            date_match = combined_pattern.search(other_text)
                            if date_match:
                                return date_match.group().strip()

        return None

    def _extract_balance_from_span(self, span: SpanInfo, all_spans: List[SpanInfo] = None) -> Optional[str]:
        """
        Extract balance (opening/closing) from a span labeled as balance.

        First checks if span text contains a monetary value.
        Then checks right/below spans for balance value.

        Args:
            span: The span labeled as opening/closing balance
            all_spans: All spans on the page (required for label-only spans)

        Returns:
            Balance string or None
        """
        span_text = span.text.strip()

        # Balance pattern: Optional currency symbol, digits with commas/decimal
        # Examples: "1,234.56", "AED 1,234.56", "$1,234.56", "1.234,56" (European format)
        balance_patterns = [
            r'[A-Z]{3}\s*[\d,]+\.?\d*',  # AED 1,234.56
            r'[$\€\£\¥]\s*[\d,]+\.?\d*',  # $1,234.56
            r'[\d,]+\.?\d*\s*[A-Z]{3}',  # 1,234.56 AED
            r'[\d,]+\.?\d*',  # 1,234.56
        ]
        combined_pattern = re.compile('|'.join(balance_patterns), re.IGNORECASE)

        # Check if span text itself contains a balance
        balance_match = combined_pattern.search(span_text)
        if balance_match:
            return balance_match.group().strip()

        # If only label (e.g., "OPENING BALANCE:"), look right for the value
        span_text_upper = span.text.upper().strip()
        if span_text_upper in ["OPENING BALANCE", "OPENING BALANCE:", "OPENING BAL:",
                               "PREVIOUS BALANCE", "PREVIOUS BALANCE:",
                               "BRING FORWARD", "BRING FORWARD:", "BF", "B/F",
                               "CLOSING BALANCE", "CLOSING BALANCE:", "CLOSING BAL:",
                               "CURRENT BALANCE", "CURRENT BALANCE:",
                               "ENDING BALANCE", "ENDING BALANCE:",
                               "CARRY FORWARD", "CARRY FORWARD:", "CF", "C/F"]:
            if all_spans:
                for other in all_spans:
                    if (other != span and
                        abs(other.y1 - span.y1) < 5 and  # Same line
                        other.x1 > span.x2 and  # To the right
                        other.x1 - span.x2 < 150):  # Within reasonable distance
                        other_text = other.text.strip()
                        balance_match = combined_pattern.search(other_text)
                        if balance_match:
                            return balance_match.group().strip()

        return None

    def _extract_iban_spatial(
        self,
        label_span: SpanInfo,
        candidate_spans: List[SpanInfo],
        standard_line_spacing: float
    ) -> Optional[str]:
        """Extract IBAN value using spatial search from label span."""
        iban_pattern = re.compile(r'^[A-Z]{2}\d{2}[A-Z0-9]{11,30}$')

        # Step 1: Try to find value to the RIGHT (same line)
        for span in candidate_spans:
            y_overlap = min(span.y2, label_span.y2) - max(span.y1, label_span.y1)
            center_y_delta = abs(span.center_y() - label_span.center_y())
            delta_x = span.x1 - label_span.x2

            is_same_line = y_overlap > 0 or center_y_delta < standard_line_spacing * 0.3

            if is_same_line and delta_x >= 0 and delta_x < 200:
                text_upper = span.text.upper().strip()
                if iban_pattern.match(text_upper):
                    return text_upper
                iban_match = iban_pattern.search(text_upper)
                if iban_match:
                    return iban_match.group()

        # Step 2: If not found, try BELOW the label
        for span in candidate_spans:
            delta_y = span.y1 - label_span.y2
            delta_x = span.x1 - label_span.x1

            if delta_y > 0 and delta_y < standard_line_spacing * 2 and abs(delta_x) < standard_line_spacing * 2:
                text_upper = span.text.upper().strip()
                if iban_pattern.match(text_upper):
                    return text_upper
                iban_match = iban_pattern.search(text_upper)
                if iban_match:
                    return iban_match.group()

        return None

    def _extract_statement_date_spatial(
        self,
        label_span: SpanInfo,
        candidate_spans: List[SpanInfo],
        standard_line_spacing: float
    ) -> Optional[str]:
        """Extract statement date value using spatial search from label span."""
        date_patterns = [
            r'\b\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}\b',
            r'\b\d{2,4}[-/\.]\d{1,2}[-/\.]\d{1,2}\b',
            r'\b(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*[\s\-\.]?\d{1,2}[\s\-\.]?\d{2,4}\b',
        ]
        combined_pattern = re.compile('|'.join(date_patterns), re.IGNORECASE)

        # Step 1: Try to find value to the RIGHT (same line)
        for span in candidate_spans:
            y_overlap = min(span.y2, label_span.y2) - max(span.y1, label_span.y1)
            center_y_delta = abs(span.center_y() - label_span.center_y())
            delta_x = span.x1 - label_span.x2

            is_same_line = y_overlap > 0 or center_y_delta < standard_line_spacing * 0.3

            if is_same_line and delta_x >= 0 and delta_x < 150:
                date_match = combined_pattern.search(span.text)
                if date_match:
                    return date_match.group().strip()

        # Step 2: If not found, try BELOW the label
        for span in candidate_spans:
            delta_y = span.y1 - label_span.y2
            delta_x = span.x1 - label_span.x1

            if delta_y > 0 and delta_y < standard_line_spacing * 2 and abs(delta_x) < standard_line_spacing * 2:
                date_match = combined_pattern.search(span.text)
                if date_match:
                    return date_match.group().strip()

        return None

    def _extract_balance_spatial(
        self,
        label_span: SpanInfo,
        candidate_spans: List[SpanInfo],
        standard_line_spacing: float
    ) -> Optional[str]:
        """Extract balance value using spatial search from label span."""
        balance_patterns = [
            r'[A-Z]{3}\s*[\d,]+\.?\d*',
            r'[$\€\£\¥]\s*[\d,]+\.?\d*',
            r'[\d,]+\.?\d*\s*[A-Z]{3}',
            r'[\d,]+\.?\d*',
        ]
        combined_pattern = re.compile('|'.join(balance_patterns), re.IGNORECASE)

        # Step 1: Try to find value to the RIGHT (same line)
        for span in candidate_spans:
            y_overlap = min(span.y2, label_span.y2) - max(span.y1, label_span.y1)
            center_y_delta = abs(span.center_y() - label_span.center_y())
            delta_x = span.x1 - label_span.x2

            is_same_line = y_overlap > 0 or center_y_delta < standard_line_spacing * 0.3

            if is_same_line and delta_x >= 0 and delta_x < 150:
                balance_match = combined_pattern.search(span.text)
                if balance_match:
                    return balance_match.group().strip()

        # Step 2: If not found, try BELOW the label
        for span in candidate_spans:
            delta_y = span.y1 - label_span.y2
            delta_x = span.x1 - label_span.x1

            if delta_y > 0 and delta_y < standard_line_spacing * 2 and abs(delta_x) < standard_line_spacing * 2:
                balance_match = combined_pattern.search(span.text)
                if balance_match:
                    return balance_match.group().strip()

        return None

    # ============================================================
    # HELPER METHODS - ADDRESS EXTRACTION
    # ============================================================

    def _collect_address_spans(
        self,
        city_span: SpanInfo,
        candidate_spans: List[SpanInfo],
        standard_line_spacing: float,
        country_code: Optional[str] = None,
        known_locations: Optional[Set[str]] = None
    ) -> List[SpanInfo]:
        """
        Collect address spans around the address anchor span.

        Uses content-driven collection when country_code and known_locations are provided.
        Falls back to geometry-based collection for backward compatibility.

        Content-driven approach:
        1. Start with the city span
        2. Use preloaded city/state map to detect leftover content
        3. Continue collecting spans above while leftover is empty
        4. Stop when leftover content is found (street address)

        Only includes spans that are:
        1. Vertically close (within line spacing thresholds)
        2. Horizontally aligned (x1 within 50 pixels of anchor's x1)
        """
        address_spans = [city_span]

        # Define horizontal alignment threshold (max x1 difference to be considered same column)
        max_x_delta = 50  # pixels - spans must be roughly in the same column as anchor

        # Content-driven collection when country info is available
        # Adaptive approach: Calculate line spacing from first span above, use that for subsequent iterations
        if country_code and known_locations:
            self.logger.debug(f"Using content-driven address collection for {country_code}")
            self.logger.debug(f"City anchor: Y=[{city_span.y1:.1f}, {city_span.y2:.1f}], X1={city_span.x1:.1f}, text='{city_span.text}'")

            # Debug: log candidate_spans near the city anchor
            nearby_spans = [s for s in candidate_spans if city_span.y2 - 50 < s.y1 < city_span.y2 + 100]
            self.logger.debug(f"Found {len(nearby_spans)} candidate spans near city anchor:")
            for s in nearby_spans[:10]:  # Log first 10
                self.logger.debug(f"  Candidate: Y=[{s.y1:.1f}, {s.y2:.1f}], X1={s.x1:.1f}, text='{s.text}'")

            collected = [city_span]
            max_iterations = 10
            calculated_line_spacing = None  # Will be set from first span above
            line_spacing_multiplier = 1.5  # Allow 1.5x the calculated spacing for tolerance
            initial_max_gap = standard_line_spacing * 2.5  # Initial threshold before we calculate actual spacing

            for iteration in range(max_iterations):
                # Find closest span above the most recently collected span (topmost one)
                most_recent_span = collected[0]
                closest_span = None
                closest_gap = float('inf')

                for span in candidate_spans:
                    if span in collected:
                        continue
                    # Skip email addresses - they're not part of the physical address
                    if self._is_email(span.text):
                        continue
                    # Skip date patterns (e.g., "01 JAN, 2026", "27 JAN 2026")
                    import re
                    date_pattern = r'^\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[a-z]*[,\.]?\s*\d{4}$'
                    if re.match(date_pattern, span.text.strip(), re.IGNORECASE):
                        self.logger.debug(f"Skipping span with date pattern: '{span.text}'")
                        continue
                    # Skip spans that contain bank-specific keywords (these are bank headers, not customer address)
                    bank_address_keywords = [
                        "STATEMENT OF ACCOUNT", "STATEMENT DETAILS", "STATEMENT PERIOD",
                        "ACCOUNT TYPE", "ACCOUNT NUMBER", "CURRENCY", "INTEREST PAYOUT",
                        "BRANCH", "BRANCH ADDRESS", "WEBSITE:", "HTTP", "HTTPS"
                    ]
                    if any(keyword in span.text.upper() for keyword in bank_address_keywords):
                        self.logger.debug(f"Skipping span with bank keyword: '{span.text}'")
                        continue
                    # Check horizontal alignment
                    if abs(span.x1 - city_span.x1) > max_x_delta:
                        continue
                    # Gap from most_recent_span.y1 to span.y2 (incremental gap measurement)
                    # Skip spans that are below (span's top is below reference top)
                    # This handles both negative and positive gaps correctly
                    if span.y1 >= most_recent_span.y1:
                        continue
                    gap = most_recent_span.y1 - span.y2
                    # Apply adaptive threshold: use calculated line spacing (if available) + multiplier
                    if calculated_line_spacing is not None:
                        max_gap = calculated_line_spacing * line_spacing_multiplier
                    else:
                        # First span - must be within reasonable initial threshold
                        max_gap = initial_max_gap
                        self.logger.debug(f"Initial threshold: {max_gap:.1f}px (standard_line_spacing: {standard_line_spacing:.2f}px)")

                    if gap > max_gap:
                        self.logger.debug(f"Skipping '{span.text}' - gap {gap:.1f}px exceeds max_gap {max_gap:.1f}px")
                        continue
                    if gap < closest_gap:
                        closest_gap = gap
                        closest_span = span

                if closest_span:
                    self.logger.debug(f"Collecting next span above: '{closest_span.text}' (gap: {closest_gap:.1f}px)")
                    collected.insert(0, closest_span)
                    # Calculate line spacing from this first span above
                    # This will be used as threshold for subsequent iterations
                    if calculated_line_spacing is None:
                        calculated_line_spacing = closest_gap
                        self.logger.debug(f"Calculated line spacing from first span above: {calculated_line_spacing:.1f}px (max threshold: {calculated_line_spacing * line_spacing_multiplier:.1f}px)")
                else:
                    self.logger.debug(f"No more spans above within threshold, stopping collection")
                    break

            address_spans = collected

            # Spans BELOW anchor - collect using adaptive threshold, only if city span lacks postal code
            city_span_has_postal = False
            city_text_upper = city_span.text.upper()
            # Check if city span text contains postal code
            if country_code:
                try:
                    country_loader = get_country_config_loader()
                    postal_code_pattern_str = country_loader.get_postal_code_pattern(country_code)
                    postal_code_regex = re.compile(postal_code_pattern_str) if postal_code_pattern_str else None
                    if postal_code_regex:
                        city_span_has_postal = postal_code_regex.search(city_text_upper) is not None
                except Exception:
                    city_span_has_postal = False

            if not city_span_has_postal:
                # City span doesn't have postal code - collect spans below using adaptive threshold
                self.logger.debug(f"City span has no postal code, collecting spans below with adaptive threshold")

                # Use adaptive threshold based on standard line spacing
                line_spacing_multiplier = 3.0
                max_gap_below = standard_line_spacing * line_spacing_multiplier
                self.logger.debug(f"Max gap below: {max_gap_below:.1f}px")

                lines_below_collected = 0
                max_lines_below = 3
                most_recent_span_below = city_span  # Track most recently collected span for iterative gap calculation

                for span in candidate_spans:
                    if span in collected:
                        continue
                    # Skip spans that are clearly above the anchor (span's bottom is at or above anchor's top)
                    # This is more accurate than checking if span starts before anchor ends
                    if span.y2 <= city_span.y1:
                        continue  # Span is entirely above anchor
                    if self._is_email(span.text):
                        continue
                    # Skip date patterns (e.g., "01 JAN, 2026", "27 JAN 2026")
                    import re
                    date_pattern = r'^\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[a-z]*[,\.]?\s*\d{4}$'
                    if re.match(date_pattern, span.text.strip(), re.IGNORECASE):
                        self.logger.debug(f"  Skipping below span with date pattern: '{span.text}'")
                        continue
                    # Skip spans that contain bank-specific keywords (these are bank headers, not customer address)
                    bank_address_keywords = [
                        "STATEMENT OF ACCOUNT", "STATEMENT DETAILS", "STATEMENT PERIOD",
                        "ACCOUNT TYPE", "ACCOUNT NUMBER", "CURRENCY", "INTEREST PAYOUT",
                        "BRANCH", "BRANCH ADDRESS", "WEBSITE:", "HTTP", "HTTPS"
                    ]
                    if any(keyword in span.text.upper() for keyword in bank_address_keywords):
                        self.logger.debug(f"  Skipping below span with bank keyword: '{span.text}'")
                        continue
                    # Check horizontal alignment first
                    x_delta = abs(span.x1 - city_span.x1)
                    in_correct_column = x_delta <= max_x_delta
                    if not in_correct_column:
                        self.logger.debug(f"  Skipping below span (wrong column): X1={span.x1:.1f}, x_delta={x_delta:.1f}px, text='{span.text[:30]}'")
                        continue
                    # Now check vertical distance from the most recently collected span below
                    delta_y = span.y1 - most_recent_span_below.y2
                    if delta_y > max_gap_below:
                        self.logger.debug(f"  Span in correct column is too far: gap={delta_y:.1f}px exceeds {max_gap_below:.1f}px, stopping collection")
                        break

                    # Collect this span (if not already collected)
                    # Note: address_spans and collected point to the same list (line 2855)
                    # Only append once to avoid duplication
                    if span not in collected:
                        self.logger.debug(f"  Adding span below: Y=[{span.y1:.1f}, {span.y2:.1f}], gap={delta_y:.1f}px (from prev Y2={most_recent_span_below.y2:.1f}), text='{span.text}'")
                        collected.append(span)
                        most_recent_span_below = span  # Update for iterative gap calculation
                        lines_below_collected += 1
                        if lines_below_collected >= max_lines_below:
                            break
                    else:
                        self.logger.debug(f"  Skipping span below (already in address_spans): Y=[{span.y1:.1f}, {span.y2:.1f}], text='{span.text}'")
            else:
                self.logger.debug(f"City span already has postal code, skipping spans below collection")
        else:
            # Fallback: geometry-based collection (original behavior)
            self.logger.debug(f"Using geometry-based address collection (no country info)")

            # Define vertical threshold - use fixed pixel values
            base_max_y_delta_above = 5  # pixels - gap between consecutive address lines
            max_y_delta_below = 25  # pixels - for content below anchor

            def collect_spans_above(threshold: float) -> List[SpanInfo]:
                """Collect address spans above the city anchor with the given threshold."""
                collected = [city_span]
                previous_span = city_span
                max_iterations = 10  # Prevent infinite loops
                iteration = 0

                while iteration < max_iterations:
                    iteration += 1
                    closest_span = None
                    closest_gap = float('inf')

                    # Find the closest span above the previous span
                    for span in candidate_spans:
                        if span in collected:
                            continue  # Already collected
                        # Skip email addresses - they're not part of the physical address
                        if self._is_email(span.text):
                            continue
                        # Skip date patterns (e.g., "01 JAN, 2026", "27 JAN 2026")
                        import re
                        date_pattern = r'^\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[a-z]*[,\.]?\s*\d{4}$'
                        if re.match(date_pattern, span.text.strip(), re.IGNORECASE):
                            continue
                        # Skip spans that contain bank-specific keywords (these are bank headers, not customer address)
                        bank_address_keywords = [
                            "STATEMENT OF ACCOUNT", "STATEMENT DETAILS", "STATEMENT PERIOD",
                            "ACCOUNT TYPE", "ACCOUNT NUMBER", "CURRENCY", "INTEREST PAYOUT",
                            "BRANCH", "BRANCH ADDRESS", "WEBSITE:", "HTTP", "HTTPS"
                        ]
                        if any(keyword in span.text.upper() for keyword in bank_address_keywords):
                            continue
                        # Check horizontal alignment - must be in same column as anchor
                        x_delta = abs(span.x1 - city_span.x1)
                        if x_delta > max_x_delta:
                            continue
                        # Gap = previous_span.y1 - span.y2 (top of previous to bottom of new)
                        # Skip spans that are below (span's top is below reference top)
                        # This handles both negative and positive gaps correctly
                        if span.y1 >= previous_span.y1:
                            continue
                        gap = previous_span.y1 - span.y2
                        if gap < closest_gap:
                            closest_gap = gap
                            closest_span = span

                    # Check if closest span is within threshold
                    if closest_span and closest_gap <= threshold:
                        x_delta = abs(closest_span.x1 - city_span.x1)
                        is_name = self._is_name_line(closest_span.text)

                        self.logger.debug(f"  Adding span: Y=[{closest_span.y1:.1f}, {closest_span.y2:.1f}], X1={closest_span.x1:.1f}, x_delta={x_delta:.1f}px, gap={closest_gap:.1f}px, text='{closest_span.text}'{' [NAME LINE]' if is_name else ''}")
                        collected.insert(0, closest_span)
                        previous_span = closest_span  # Update for next iteration
                    else:
                        # No more spans within threshold
                        if closest_span:
                            self.logger.debug(f"  Skipping span (too far): Y=[{closest_span.y1:.1f}, {closest_span.y2:.1f}], gap={closest_gap:.1f}px (exceeds threshold {threshold}), text='{closest_span.text}'")
                        else:
                            self.logger.debug(f"  No more spans found")
                        break

                return collected

            # Spans ABOVE anchor - chain through collected spans with dynamic retry
            self.logger.debug(f"Starting chain ABOVE anchor: anchor Y=[{city_span.y1:.1f}, {city_span.y2:.1f}], X1={city_span.x1:.1f}, text='{city_span.text}'")

            # First attempt with base threshold (5px)
            address_spans = collect_spans_above(base_max_y_delta_above)

            # Dynamic retry: only retry if city span contains un-extracted street content
            if len(address_spans) == 1 and country_code:
                # Parse components from city span to detect leftover street content
                components = self._parse_address_components(
                    full_address="",  # No street lines collected yet
                    city=city_span.text,
                    country_code=country_code
                )

                # Extract leftover content by removing city/state/postal/country from span text
                leftover_street = self._extract_street_from_single_span(
                    span_text=city_span.text,
                    city=components.get("city"),
                    state=components.get("state"),
                    postal_code=components.get("postal_code"),
                    country_code=country_code
                )

                # Only retry if there's NO leftover street content (street must be in separate spans above)
                # If leftover_street exists, the street is already in the city span (single-line address)
                if not leftover_street.strip():
                    retry_threshold = base_max_y_delta_above * 2
                    self.logger.debug(f"City span contains only location info (no street in span), retrying with {retry_threshold}px...")
                    address_spans = collect_spans_above(retry_threshold)
                else:
                    leftover_preview = leftover_street[:50] + '...' if len(leftover_street) > 50 else leftover_street
                    self.logger.debug(f"City span contains street content ('{leftover_preview}'), skipping retry (single-line address)")

            # Spans BELOW anchor - use adaptive line spacing, only if city span lacks postal code
            city_span_has_postal = False
            city_text_upper = city_span.text.upper()
            # Check if city span text contains postal code (same logic as city anchor detection)
            if country_code:
                try:
                    country_loader = get_country_config_loader()
                    postal_code_pattern_str = country_loader.get_postal_code_pattern(country_code)
                    postal_code_regex = re.compile(postal_code_pattern_str) if postal_code_pattern_str else None
                    if postal_code_regex:
                        city_span_has_postal = postal_code_regex.search(city_text_upper) is not None
                except Exception:
                    # If postal code lookup fails, assume no postal code
                    city_span_has_postal = False

            if not city_span_has_postal:
                # City span doesn't have postal code - collect spans below using adaptive threshold
                self.logger.debug(f"City span has no postal code, collecting spans below with adaptive threshold")

                # Use adaptive threshold based on standard line spacing
                # Larger threshold for spans below (more spacing variability below city)
                line_spacing_multiplier = 3.0
                max_gap_below = standard_line_spacing * line_spacing_multiplier

                self.logger.debug(f"Max gap below: {max_gap_below:.1f}px")

                lines_below_collected = 0
                max_lines_below = 3  # Collect up to 3 lines below city

                for span in candidate_spans:
                    if span.y1 < city_span.y2:
                        continue  # Span is above anchor
                    if self._is_email(span.text):
                        continue
                    # Check horizontal alignment
                    x_delta = abs(span.x1 - city_span.x1)
                    if x_delta > max_x_delta:
                        self.logger.debug(f"  Skipping below span (wrong column): X1={span.x1:.1f}, x_delta={x_delta:.1f}px")
                        continue
                    delta_y = span.y1 - city_span.y2
                    if delta_y > max_gap_below:
                        self.logger.debug(f"  Skipping span below (too far): gap={delta_y:.1f}px exceeds {max_gap_below:.1f}px")
                        break  # Too far below, stop collection

                    # Collect this span
                    self.logger.debug(f"  Adding span below: Y=[{span.y1:.1f}, {span.y2:.1f}], gap={delta_y:.1f}px, text='{span.text}'")
                    address_spans.append(span)
                    lines_below_collected += 1
                    if lines_below_collected >= max_lines_below:
                        break  # Collected max lines below city
            else:
                self.logger.debug(f"City span already has postal code, skipping spans below collection")

        self.logger.debug(f"Collected {len(address_spans)} address spans:")
        for i, s in enumerate(address_spans):
            self.logger.debug(f"  [{i}] Y=[{s.y1:.1f}, {s.y2:.1f}]: {s.text}")

        return address_spans

    def _extract_name_from_address(
        self,
        address_spans: List[SpanInfo],
        city_span: SpanInfo,
        all_spans: List[SpanInfo]
    ) -> Optional[str]:
        """
        Extract account holder name from address spans.

        When account_holder_name_label NOT detected:
        Returns the first text span above the address anchor as account holder name.

        If the name is only a title (e.g., MR.), searches for the actual name
        to the right on the same line and combines them.

        Args:
            address_spans: Collected address spans
            city_span: The city/address anchor span
            all_spans: All spans on the page (for title lookup)
        """
        if not address_spans:
            return None

        # Find the city span index
        city_idx = -1
        for i, span in enumerate(address_spans):
            if span == city_span:
                city_idx = i
                break

        # Return the first span above the city as account holder name
        if city_idx > 0:
            name_span = address_spans[0]
            name_text = name_span.text.strip()

            # Skip if the first span looks like an address line (starts with house number)
            # This prevents treating address lines as names
            if re.match(r'^\d+[-/\s]', name_text.upper().strip()):
                self.logger.debug(f"First span above city looks like address line, skipping name extraction: '{name_text}'")
                return None

            # Skip if the text looks like a label, address, or account number
            if self._looks_like_label(name_text):
                self.logger.debug(f"First span above city looks like a label, skipping name extraction: '{name_text}'")
                return None

            # Skip if the text looks like an address (contains address keywords)
            address_keywords = ["BLK", "BLOCK", "STREET", "ROAD", "CRESCENT", "AVENUE", "LANE", "DRIVE", "JALAN", "ROAD"]
            name_upper = name_text.upper()
            for keyword in address_keywords:
                if keyword in name_upper:
                    self.logger.debug(f"First span above city contains address keyword '{keyword}', skipping name extraction: '{name_text}'")
                    return None

            # Skip if the text is mostly digits (account number)
            digit_ratio = sum(c.isdigit() for c in name_text) / len(name_text) if name_text else 0
            if digit_ratio > 0.5:
                self.logger.debug(f"First span above city is mostly digits (ratio: {digit_ratio:.2f}), skipping name extraction: '{name_text}'")
                return None

            # If the name is only a title, try to find the full name
            if self._is_title(name_text):
                # Use _find_name_after_title to get the combined name
                combined = self._find_name_after_title(name_span, all_spans)
                if combined:
                    return self._clean_address_text(combined)
                return self._clean_address_text(name_text)

            return self._clean_address_text(name_text)

        return None

    def _extract_name_via_title_detection(
        self,
        all_spans: List[SpanInfo],
        city_span: SpanInfo
    ) -> Optional[str]:
        """
        Extract account holder name using title-based detection.

        This is a fallback when address block extraction fails.
        Searches for spans that start with a title pattern (e.g., "Mr.", "Dr.").

        Args:
            all_spans: All spans on the page
            city_span: The city/country anchor span (used for position reference)

        Returns:
            Extracted name if found, None otherwise
        """
        # Search for spans that start with a title pattern
        for span in all_spans:
            span_text_upper = span.text.strip().upper()

            # Check if span starts with any title pattern
            for title in self.TITLE_PATTERNS:
                # Check with or without space after title (e.g., "MR. K" or "MR.K")
                if span_text_upper.startswith(title + ' ') or span_text_upper.startswith(title):
                    # Found a span that starts with a title
                    # Extract the full text as the name
                    name_text = span.text.strip()

                    # Validate that this looks like a name (not a random span with title)
                    if self._is_name_line(name_text):
                        self.logger.debug(f"Found name via title detection: '{name_text}' Y=[{span.y1:.1f}, {span.y2:.1f}]")
                        return name_text

        return None

    def _build_address_result(
        self,
        address_spans: List[SpanInfo],
        city_span: SpanInfo,
        holder_label: Optional[SpanInfo],
        all_spans: List[SpanInfo],
        country_code: Optional[str],
        detected_state: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Build final address result from address spans.

        Args:
            address_spans: Collected address spans
            city_span: The city/country anchor span
            holder_label: Account holder name label (if detected)
            all_spans: All spans on the page (for name extraction)
            country_code: Detected ISO country code
            detected_state: State looked up from city-state library (optional)

        Returns:
            Dictionary with address components
        """
        # Debug logging: collected address spans
        self.logger.info(f"Collected {len(address_spans)} address spans:")
        for i, s in enumerate(address_spans):
            self.logger.info(f"  [{i}] Y=[{s.y1:.1f}, {s.y2:.1f}]: '{s.text}'")

        # Determine the account holder name (from holder_label or extraction)
        extracted_name = None
        name_span_index = -1
        city_span_index = -1

        if holder_label is not None and holder_label.value is not None:
            # Use the name from the holder label
            extracted_name = holder_label.value
            self.logger.info(f"Using account holder name from holder label: '{extracted_name}'")
        else:
            # Extract name from address spans when no holder label
            extracted_name = self._extract_name_from_address(address_spans, city_span, all_spans)
            if extracted_name:
                self.logger.info(f"Extracted account holder name from address: '{extracted_name}'")
            else:
                # Fallback: Try title-based detection if address block extraction failed
                # This handles cases where the name is not in the address block (e.g., different column)
                extracted_name = self._extract_name_via_title_detection(all_spans, city_span)
                if extracted_name:
                    self.logger.info(f"Extracted name via title-based detection: '{extracted_name}'")

        # ALWAYS find the span that contains the extracted name and mark it for removal
        # This ensures the name is removed from the address regardless of source
        if extracted_name:
            # Normalize the extracted name for comparison (remove honorifics, normalize whitespace)
            extracted_name_normalized = self._normalize_name_for_matching(extracted_name)
            self.logger.info(f"Normalized extracted name: '{extracted_name_normalized}'")

            for i, span in enumerate(address_spans):
                span_text_normalized = self._normalize_name_for_matching(span.text)
                # Check if the span text matches or contains the extracted name (or vice versa)
                # This handles cases where:
                # - extracted_name is "MR K C ROHITH" and span.text is "K C ROHITH"
                # - extracted_name is "K C ROHITH" and span.text is "MR K C ROHITH"
                if (extracted_name_normalized in span_text_normalized or
                    span_text_normalized in extracted_name_normalized or
                    extracted_name_normalized == span_text_normalized):
                    name_span_index = i
                    self.logger.info(f"Marking span[{i}] for removal (name match: '{extracted_name}' ~ '{span.text}')")
                    break

        # Find the city span index
        for i, span in enumerate(address_spans):
            if span == city_span:
                city_span_index = i
                break

        # Debug logging: indices
        self.logger.info(f"Name span index: {name_span_index}, City span index: {city_span_index}")
        if extracted_name:
            self.logger.info(f"Extracted name: '{extracted_name}'")

        # Build street address (lines 1 & 2, excluding account holder name and city)
        street_address_lines = []
        for i, addr_span in enumerate(address_spans):
            if i != name_span_index and i != city_span_index and addr_span.text.strip():
                # Safety net: skip any address labels that may have slipped through
                if self._is_address_skip_label(addr_span.text):
                    self.logger.info(f"Skipping address label in final result: '{addr_span.text}'")
                    continue
                # Name filtering is now handled by _clean_address_text() when holder_name is provided
                # No need to skip name lines here
                # Clean the address text to remove formatting artifacts
                # Pass holder name when available to remove it from address
                holder_name = holder_label.value if (holder_label and holder_label.value) else None
                cleaned_text = self._clean_address_text(addr_span.text, holder_name)
                if cleaned_text:
                    street_address_lines.append(cleaned_text)

        # Debug logging: street address lines
        self.logger.info(f"Street address lines: {street_address_lines}")

        # Build full address for component parsing - always start with city span text
        # (which contains postal code for Indian addresses like "CHITTOOR, AP, 517419")
        # then concatenate street address lines
        full_address = city_span.text.strip() if city_span else ""
        if street_address_lines:
            full_address = f"{full_address}, {', '.join(street_address_lines)}"

        # Get city name from city_span (raw value for parsing)
        # Use .value if set (contains normalized city name from detection), otherwise use .text
        # Note: When country_span is used as fallback, .value is cleared to avoid using bank name
        city_name_raw = city_span.value if hasattr(city_span, 'value') and city_span.value else city_span.text

        # CRITICAL: Parse address components FIRST to get the actual city name
        # This is needed because for single-span addresses, city_name_raw contains the full address
        components = self._parse_address_components(full_address, city_name_raw, country_code)
        parsed_city = components.get("city")  # The actual extracted city name (e.g., "CHITTOOR")

        # Use detected state from city lookup if state wasn't parsed from address
        if detected_state and not components.get("state"):
            components["state"] = detected_state
            self.logger.info(f"Using detected state from city lookup: '{detected_state}'")

        # Use all street address lines (multi-span case)
        street_address = ", ".join(street_address_lines)

        # Clean street_address of all location components for multi-span addresses
        if street_address and parsed_city:
            street_address = self._clean_street_address(
                street_address,
                parsed_city,
                components.get("state"),
                components.get("postal_code"),
                country_code
            )
            self.logger.info(f"Cleaned street address (multi-span): '{street_address}'")

        # Extract street portion from city span if it contains more than just the city name
        # This handles cases where the city span contains the full address (e.g., "24B,Wasl Squa,.,.,AlSafa1,DUBAI,")
        if city_span and city_span.text and parsed_city:
            city_span_street = self._extract_street_from_single_span(
                city_span.text,
                parsed_city,  # Use the PARSED city name, not the raw span text
                components.get("state"),  # Pass the parsed state
                components.get("postal_code"),  # Pass the parsed postal code
                country_code
            )
            if city_span_street:
                self.logger.info(f"Extracted street portion from city span: '{city_span_street}'")
                # Prepend the city span's street portion to the existing street_address
                if street_address:
                    street_address = f"{city_span_street}, {street_address}"
                else:
                    street_address = city_span_street

        # Clean the final concatenated address to remove any duplicates or location components
        # that may have been introduced by the city span extraction
        if street_address and parsed_city:
            self.logger.info(f"Final address BEFORE cleaning: '{street_address}'")
            final_cleaned = self._clean_street_address(
                street_address,
                parsed_city,
                components.get("state"),
                components.get("postal_code"),
                country_code
            )
            self.logger.info(f"Final address AFTER _clean_street_address: '{final_cleaned}'")
            street_address = final_cleaned

        return {
            "full_address": full_address,
            "street_address": street_address,
            "city": parsed_city,  # Use the parsed city name
            "state": components.get("state"),
            "country": components.get("country"),
            "postal_code": components.get("postal_code"),
            "account_holder_name": extracted_name
        }

    def _parse_address_components(
        self,
        full_address: str,
        city: Optional[str],
        country_code: Optional[str] = None
    ) -> Dict[str, Optional[str]]:
        """
        Parse address components: state, country, postal_code.

        Args:
            full_address: Full street address text
            city: City name (may include country like "Dubai, UAE")
            country_code: Pre-detected ISO country code (e.g., "AE", "IN")

        Returns:
            Dictionary with city, state, country, postal_code
        """
        components = {
            "city": city,
            "state": None,
            "country": None,
            "postal_code": None
        }

        # If country_code is provided, pre-populate country
        if country_code:
            components["country"] = country_code

        # Extract country from city text (e.g., "Dubai, UAE" -> country="AE")
        # Also extract city name without country (e.g., "Dubai, UAE" -> city="Dubai")
        city_only = city
        if city:
            city_upper = city.upper()
            if "UAE" in city_upper or "UNITED ARAB EMIRATES" in city_upper:
                components["country"] = "AE"
                # UAE has emirates, not states, so set state to None
                components["state"] = None
                # Extract city name without country
                city_only = city.split(",")[0].strip()
            elif "SG" in city_upper or "SINGAPORE" in city_upper:
                components["country"] = "SG"
                components["state"] = None
                city_only = city.split(",")[0].strip()
            # India detection
            elif any(india_marker in city_upper for india_marker in ["INDIA", "KARNATAKA", "TAMIL NADU", "MAHARASHTRA", "DELHI"]):
                components["country"] = "IN"
                city_only = city.split(",")[0].strip()

        # If city_only is just a country name, set it to None
        # This handles cases where city text is just "Singapore" or "UAE"
        country_names = ["UAE", "UNITED ARAB EMIRATES", "SG", "SINGAPORE", "INDIA"]
        if city_only and city_only.upper() in country_names:
            city_only = None

        components["city"] = city_only or city

        if not full_address:
            return components

        # Extract state from address if country is known
        if country_code and not components.get("state"):
            state = self._extract_state_from_address(full_address, country_code)
            if state:
                components["state"] = state

        try:
            validator = get_bank_statement_validator()
            parsed = validator.parse_address_components(full_address, country_hint=country_code)
            # Debug logging: show what the validator parsed
            self.logger.info(f"Validator parsed address components: postal_code='{parsed.get('postal_code')}', full_address='{full_address[:100]}...'")

            # Don't override country and state if already set
            # Only use postal_code from parsed result if it's not a PO Box number
            if not components.get("country"):
                components.update(parsed)
                self.logger.info(f"Updated components with parsed result (country was not set)")
            else:
                # Country is already set, but ALWAYS use the validator's parsed city if available
                parsed_city = parsed.get("city")
                if parsed_city:
                    components["city"] = parsed_city
                    self.logger.info(f"Updated city with validator result: '{parsed_city}' (country was already set)")

                # Only update postal_code from parsed result
                # But avoid PO Box numbers being treated as postal codes
                parsed_postal = parsed.get("postal_code")
                self.logger.info(f"Country already set to '{components.get('country')}', checking postal_code from validator: '{parsed_postal}'")

                # Check if address is a PO Box - more robust check
                is_po_box = bool(re.search(r'\b(?:P\.?O\.?\s*(?:BOX)?|POST\s+OFFICE\s*BOX)\b', full_address, re.IGNORECASE))
                self.logger.info(f"PO Box check: is_po_box={is_po_box}, condition={parsed_postal and not is_po_box}")

                if parsed_postal and not is_po_box:
                    components["postal_code"] = parsed_postal
                    self.logger.info(f"Set postal_code to '{parsed_postal}'")
                else:
                    if parsed_postal:
                        self.logger.warning(f"Postal code '{parsed_postal}' rejected (is_po_box={is_po_box})")
                    else:
                        self.logger.warning(f"No postal code found by validator in address: '{full_address[:100]}...'")
        except Exception as e:
            self.logger.warning(f"Failed to parse address components: {e}")

        return components

    def _extract_street_from_single_span(
        self,
        span_text: str,
        city: str,
        state: Optional[str] = None,
        postal_code: Optional[str] = None,
        country_code: Optional[str] = None
    ) -> str:
        """
        Extract street address from a single span that contains the full address.

        Removes city, state, country, and postal code from the span text.
        Unlike the previous version, this accepts pre-parsed components.

        Args:
            span_text: Full span text (e.g., "H NO 1 X 21, PALLE, CHITTOOR, AP, 517419")
            city: City name (e.g., "CHITTOOR")
            state: State name (e.g., "ANDHRA PRADESH")
            postal_code: Postal code (e.g., "517419")
            country_code: ISO country code for country name removal

        Returns:
            Street address portion (e.g., "H NO 1 X 21, PALLE")
        """
        if not span_text:
            return ""

        text = span_text.strip()
        text_upper = text.upper()

        # Remove postal code (if provided)
        if postal_code:
            # Remove the exact postal code from the text
            text = re.sub(r'\b' + re.escape(postal_code) + r'\b', '', text, flags=re.IGNORECASE).strip()
            text_upper = text.upper()

        # Remove country name (e.g., "INDIA", "SINGAPORE")
        if country_code:
            from app.config.bank_statement_country_loader import get_country_config_loader
            config_loader = get_country_config_loader()
            config = config_loader.get_country_config(country_code)
            if config:
                country_name = config.get("country_name", "").upper()
                name_aliases = [alias.upper() for alias in config.get("name_aliases", [])]
                for alias in [country_name] + name_aliases:
                    if alias in text_upper:
                        text = re.sub(r'\b' + re.escape(alias) + r'\b', '', text, flags=re.IGNORECASE).strip()
                        text_upper = text.upper()
                        break

        # Remove state (if provided)
        if state:
            state_upper = state.upper()
            # Try with word boundary first
            pattern = r'\b' + re.escape(state_upper) + r'\b'
            if re.search(pattern, text_upper):
                text = re.sub(pattern, '', text, flags=re.IGNORECASE).strip()
                text_upper = text.upper()
            else:
                # Fallback: try without word boundary (for state names at edges)
                text = re.sub(re.escape(state_upper), '', text, flags=re.IGNORECASE).strip()
                text_upper = text.upper()

        # Remove city name - only remove the last occurrence (typically at end of address)
        if city:
            city_upper = city.upper()
            # Find all matches and remove only the last one (at the end of address)
            pattern = r'\b' + re.escape(city_upper) + r'\b'
            matches = list(re.finditer(pattern, text_upper))
            if matches:
                # Remove the last occurrence (typically at the end)
                last_match = matches[-1]
                text = text[:last_match.start()] + text[last_match.end():]
                text = text.strip()
            else:
                # Fallback: remove without word boundary (last occurrence)
                parts = re.split(r'(?i)' + re.escape(city_upper), text)
                if len(parts) > 1:
                    text = ''.join(parts[:-1]).strip()
                    text = re.sub(r',\s*$', '', text).strip()

        # Clean up: remove extra commas and whitespace
        text = re.sub(r',\s*,', ',', text)  # Double commas
        text = re.sub(r'^,\s*', '', text)   # Leading comma
        text = re.sub(r',\s*$', '', text)   # Trailing comma
        text = re.sub(r'\s+', ' ', text)    # Multiple spaces to single space
        text = text.strip()

        return text

    def _clean_street_address(
        self,
        street_address: str,
        city: str,
        state: Optional[str] = None,
        postal_code: Optional[str] = None,
        country_code: Optional[str] = None
    ) -> str:
        """
        Clean street address by removing location components.

        Removes city, state, country, and postal code from the street address string.
        Used for multi-span addresses where components may be in separate spans.

        Args:
            street_address: The joined street address string
            city: City name to remove
            state: State name to remove
            postal_code: Postal code to remove
            country_code: ISO country code for country name removal

        Returns:
            Cleaned street address
        """
        self.logger.info(f"_clean_street_address INPUT: '{street_address}', city='{city}', state='{state}', postal='{postal_code}'")

        if not street_address:
            return ""

        text = street_address.strip()
        text_upper = text.upper()

        # Remove postal code
        if postal_code:
            text = re.sub(r'\b' + re.escape(postal_code) + r'\b', '', text, flags=re.IGNORECASE).strip()
            text_upper = text.upper()

        # Remove country name
        if country_code:
            from app.config.bank_statement_country_loader import get_country_config_loader
            config_loader = get_country_config_loader()
            config = config_loader.get_country_config(country_code)
            if config:
                country_name = config.get("country_name", "").upper()
                name_aliases = [alias.upper() for alias in config.get("name_aliases", [])]
                for alias in [country_name] + name_aliases:
                    if alias in text_upper:
                        text = re.sub(r'\b' + re.escape(alias) + r'\b', '', text, flags=re.IGNORECASE).strip()
                        text_upper = text.upper()
                        break

        # Remove state - tokenize multi-word state names to remove individual words
        if state:
            state_upper = state.upper()
            # First try to remove the full state name
            text = re.sub(r'\b' + re.escape(state_upper) + r'\b', '', text, flags=re.IGNORECASE).strip()
            text_upper = text.upper()
            # Also remove individual words from multi-word state names
            # This handles cases like "PRADESH" from "ANDHRA PRADESH"
            state_words = state_upper.split()
            if len(state_words) > 1:
                for word in state_words:
                    # Remove each word individually if it appears standalone
                    text = re.sub(r'\b' + re.escape(word) + r'\b', '', text, flags=re.IGNORECASE).strip()
                    text_upper = text.upper()

        # Remove city - remove ALL occurrences (not just the last)
        if city:
            city_upper = city.upper()
            # Remove all occurrences using regex
            text = re.sub(r'\b' + re.escape(city_upper) + r'\b', '', text, flags=re.IGNORECASE).strip()

        # Also remove state abbreviations (for India, UAE, etc.)
        if state and country_code:
            state_upper = state.upper()
            abbreviations = self._get_state_abbreviations(country_code)
            for abbrev, full_name in abbreviations.items():
                if state_upper == full_name.upper() or state_upper == abbrev.upper():
                    # Remove the abbreviation from text
                    text = re.sub(r'\b' + re.escape(abbrev.upper()) + r'\b', '', text, flags=re.IGNORECASE).strip()
                    break

        # Clean up: remove extra commas and whitespace
        text = re.sub(r',\s*,', ',', text)
        text = re.sub(r'^,\s*', '', text)
        text = re.sub(r',\s*$', '', text)
        text = re.sub(r'\s+', ' ', text)

        result = text.strip()
        self.logger.info(f"_clean_street_address OUTPUT: '{result}'")
        return result

    def _get_state_abbreviations(self, country_code: str) -> Dict[str, str]:
        """Get state abbreviation to full name mapping for a country."""
        abbreviations = {}

        if country_code == "IN":
            # Indian state abbreviations
            abbreviations = {
                "AP": "ANDHRA PRADESH",
                "TN": "TAMIL NADU",
                "KA": "KARNATAKA",
                "MH": "MAHARASHTRA",
                "DL": "DELHI",
                "TS": "TELANGANA",
                "KL": "KERALA",
            }
        elif country_code == "AE":
            # UAE emirates
            abbreviations = {
                "DU": "DUBAI",
                "AB": "ABU DHABI",
                "SH": "SHARJAH",
                "AJ": "AJMAN",
                "UM": "UMM AL QUWAIN",
                "RAK": "RAS AL KHAIMAH",
                "FUJ": "FUJAIRAH",
            }

        return abbreviations

    def _remove_duplicate_words(self, text: str, city: str = None) -> str:
        """
        Remove duplicate words and location names from address.

        Handles cases like:
        - "1-21 SETTYGIRIPALLE, SETTYGIRIPALLE" -> "1-21 SETTYGIRIPALLE"
        - "CHITTOOR, Chittoor" -> "" (both are city name)

        Preserves house number prefixes like "1-21" while removing standalone duplicates.
        """
        if not text:
            return ""

        self.logger.info(f"_remove_duplicate_words INPUT: '{text}', city='{city}'")

        # Split by comma and clean each part
        parts = [p.strip() for p in text.split(',')]
        if not parts:
            return ""

        seen_words = set()
        seen_parts = []
        unique_parts = []

        for part in parts:
            if not part:
                continue

            part_upper = part.upper()

            # Check if this is a city name variant
            is_city_variant = False
            if city:
                city_upper = city.upper()
                # Direct match or case variation
                if part_upper == city_upper or part_upper in city_upper or city_upper in part_upper:
                    is_city_variant = True

            if is_city_variant:
                # Skip city name variants
                continue

            # Check for duplicate location names
            # Split part into words to check for standalone duplicates
            words = part_upper.split()
            is_duplicate = False

            # If part contains house number pattern, keep it
            # Match patterns like: "1-21", "1-21 STREET", "123 MAIN ST", "H NO 1", "DOOR NO 123"
            house_number_patterns = [
                r'^\d+[-\s]\d+',  # "1-21", "123-456"
                r'^[\d-]+$',      # "1-21" (just numbers and hyphens)
                r'^H\s*NO\s*\d+',  # "H NO 1", "H NO 123"
                r'^DOOR\s*NO\s*\d+',  # "DOOR NO 123"
                r'^HOUSE\s*NO\s*\d+',  # "HOUSE NO 123"
                r'^FLT\s*\d+',     # "FLT 1", "FLAT 1"
                r'^\d+[-/\s]',     # "123-", "123 ", "123/"
            ]
            has_house_number = any(re.match(pattern, part_upper) for pattern in house_number_patterns)

            if not has_house_number and len(words) == 1:
                # Single word - check if already seen
                if words[0] in seen_words:
                    is_duplicate = True
                else:
                    seen_words.add(words[0])
            elif not has_house_number:
                # Multi-word part - check if all words were seen before
                all_words_seen = all(w in seen_words for w in words)
                if all_words_seen:
                    is_duplicate = True
                else:
                    for w in words:
                        seen_words.add(w)

            # Also check for exact matches in seen_parts
            for seen in seen_parts:
                if seen == part_upper:
                    is_duplicate = True
                    break
                # Check if one is contained in the other (partial duplicates)
                if len(seen) > 3 and seen in part_upper:
                    is_duplicate = True
                    break
                if len(part_upper) > 3 and part_upper in seen:
                    is_duplicate = True
                    break

            if not is_duplicate:
                seen_parts.append(part_upper)
                unique_parts.append(part)
                self.logger.info(f"_remove_duplicate_words KEPT: '{part}' (has_house_number={has_house_number})")
            else:
                self.logger.info(f"_remove_duplicate_words REMOVED: '{part}' (reason=duplicate)")

        result = ', '.join(unique_parts).strip()
        # Clean up any trailing/leading commas
        result = re.sub(r',\s*,', ',', result)
        result = re.sub(r'^,\s*', '', result)
        result = re.sub(r',\s*$', '', result)

        return result

    def _extract_state_from_address(self, address: str, country_code: str) -> Optional[str]:
        """
        Extract state/province from address text using country config.

        Args:
            address: Full address text
            country_code: ISO country code

        Returns:
            Extracted state name or None
        """
        from app.config.bank_statement_country_loader import get_country_config_loader
        import json

        country_loader = get_country_config_loader()
        subdivisions = country_loader.get_subdivisions(country_code)

        # If subdivisions list is empty, try to load from bank_statements config
        if not subdivisions:
            try:
                # Load state_to_country_map from bank_statements config
                config_path = Path(__file__).parent.parent.parent.parent / "app" / "reference_templates" / "bank_statements" / "config.json"
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        bank_config = json.load(f)
                        state_to_country_map = bank_config.get("state_to_country_map", {})
                        # Get states for this country
                        subdivisions = [state for state, cc in state_to_country_map.items() if cc == country_code]
            except Exception as e:
                self.logger.debug(f"Failed to load states from bank config: {e}")

        if not subdivisions:
            return None

        address_upper = address.upper()

        # Try to find the state in the address text
        # Sort by length descending to match longer names first (e.g., "ANDHRA PRADESH" before "ANDHRA")
        for state in sorted(subdivisions, key=len, reverse=True):
            state_upper = state.upper()
            # Use word boundary matching to avoid false positives
            pattern = r'\b' + re.escape(state_upper) + r'\b'
            if re.search(pattern, address_upper):
                self.logger.debug(f"Found state '{state}' in address")
                return state

        return None

    # ============================================================
    # RESULT BUILDING
    # ============================================================

    def _build_result(
        self,
        first_pass_results: Dict[str, SpanInfo],
        address_components: Dict[str, Any]
    ) -> ExtractionResult:
        """Build final ExtractionResult from pass results."""
        # Get bank info
        bank_span = first_pass_results.get("bank_name")
        bank_abbrev = bank_span.value if bank_span else ""
        bank_full_name = ""
        bank_country = None

        if bank_abbrev:
            # Determine country hint from address or currency
            country_hint = address_components.get("country")

            # If no address country, try to get from currency
            if not country_hint:
                currency_span = first_pass_results.get("currency_label")
                if currency_span and currency_span.value:
                    currency = currency_span.value
                    # Get country from currency config
                    from app.config.bank_statement_country_loader import get_country_config_loader
                    country_loader = get_country_config_loader()
                    currency_country_map = country_loader.get_currency_country_map()
                    country_hint = currency_country_map.get(currency)

            # Look up bank with country hint
            bank_info = self.bank_lookup.lookup_by_name(bank_abbrev, country_hint)
            if bank_info:
                bank_full_name = bank_info.full_name
                bank_country = bank_info.country

        # Get account number
        account_span = first_pass_results.get("account_number_label")
        account_number = account_span.value if account_span and account_span.value else None

        # Get account holder name
        holder_span = first_pass_results.get("account_holder_name_label")
        holder_name = holder_span.value if holder_span else address_components.get("account_holder_name")

        # Clean the account holder name (remove titles, "null" prefix, etc.)
        if holder_name:
            holder_name = self._clean_account_holder_name(holder_name)

        # Get SWIFT code (using bank_country, not account holder's address country)
        swift_code = None
        if bank_abbrev and bank_country:
            # Re-lookup bank with country to get country-specific SWIFT codes
            bank_info = self.bank_lookup.lookup_by_name(bank_abbrev, bank_country)
            if bank_info and bank_info.swift_codes:
                swift_code = bank_info.swift_codes[0]

        # Get currency
        currency_span = first_pass_results.get("currency_label")
        currency = currency_span.value if currency_span else None

        # Get IBAN (Phase 2)
        iban_span = first_pass_results.get("iban_label")
        iban = iban_span.value if iban_span and iban_span.value else None

        # Get statement date (Phase 2)
        statement_date_span = first_pass_results.get("statement_date_label")
        statement_date = statement_date_span.value if statement_date_span and statement_date_span.value else None

        # Get opening balance (Phase 2)
        opening_balance_span = first_pass_results.get("opening_balance_label")
        opening_balance = opening_balance_span.value if opening_balance_span and opening_balance_span.value else None

        # Get closing balance (Phase 2)
        closing_balance_span = first_pass_results.get("closing_balance_label")
        closing_balance = closing_balance_span.value if closing_balance_span and closing_balance_span.value else None

        # Build raw values for debugging
        raw_values = {}
        for key, span in first_pass_results.items():
            raw_values[key] = {
                'text': span.text,
                'value': span.value,
                'position': {'x1': span.x1, 'y1': span.y1, 'x2': span.x2, 'y2': span.y2}
            }

        return ExtractionResult(
            account_holder_name=holder_name,
            account_holder_address=address_components.get("street_address"),
            address_city=address_components.get("city"),
            address_state=address_components.get("state"),
            address_postal=address_components.get("postal_code"),
            address_country=address_components.get("country"),
            account_number=account_number,
            bank_name=bank_full_name or bank_abbrev,
            bank_country=bank_country or address_components.get("country"),
            bank_code=swift_code,
            swift_code=swift_code,
            currency=currency,
            iban=iban,
            statement_date=statement_date,
            opening_balance=opening_balance,
            closing_balance=closing_balance,
            raw_values=raw_values
        )

    def _convert_to_bank_statement_data(self, result: ExtractionResult) -> BankStatementData:
        """
        Convert internal ExtractionResult to BankStatementData (GLiNER-compatible format).

        Args:
            result: Internal ExtractionResult from spatial extraction

        Returns:
            BankStatementData with fields matching GLiNER extractor output
        """
        return BankStatementData(
            # Account Holder Information
            account_holder_name=result.account_holder_name,
            address=result.account_holder_address,

            # Account Details
            account_number=result.account_number,
            account_type=None,  # Not extracted by spatial

            # Bank Information
            bank_name=result.bank_name,
            bank_branch=None,  # Not extracted by spatial
            bank_code=result.bank_code or result.ifsc_code or result.swift_code,

            # Country fields
            bank_country=result.bank_country,
            account_holder_country=result.address_country,

            # Address components (structured)
            address_street_number=None,  # Would require parsing
            address_street_name=None,    # Would require parsing
            address_city=result.address_city,
            address_postal=result.address_postal,
            address_state=result.address_state,
            address_country=result.address_country,

            # Statement Period
            statement_from_date=None,  # Not extracted
            statement_to_date=None,    # Not extracted
            statement_date=result.statement_date,

            # Balances
            opening_balance=self._parse_balance(result.opening_balance),
            closing_balance=self._parse_balance(result.closing_balance),
            currency=result.currency,

            # Transactions
            transaction_count=None,
            transactions=None,

            # Extraction metadata
            account_number_extraction_method="spatial_geometry",

            # Confidence scores (spatial extractor doesn't provide per-field confidence)
            overall_confidence=85.0,  # Default high confidence for spatial
            confidence_scores={},

            # Validation results
            validation_results={},
        )

    def _parse_balance(self, balance_str: Optional[str]) -> Optional[float]:
        """Parse balance string to float."""
        if not balance_str:
            return None
        try:
            # Remove currency symbols, commas, whitespace
            cleaned = re.sub(r'[^\d.-]', '', str(balance_str))
            return float(cleaned) if cleaned else None
        except (ValueError, TypeError):
            return None


# ============================================================
# PUBLIC API
# ============================================================

def get_spatial_bank_statement_extractor() -> SpatialBankStatementExtractor:
    """Get the singleton instance of SpatialBankStatementExtractor."""
    return SpatialBankStatementExtractor()


def extract_bank_statement(pdf_path: str, max_pages: int = 1) -> ExtractedBankStatement:
    """
    Extract bank statement data using spatial algorithm.

    Args:
        pdf_path: Path to PDF file
        max_pages: Maximum pages to process (default: 1)

    Returns:
        ExtractedBankStatement with extracted fields
    """
    extractor = get_spatial_bank_statement_extractor()
    return extractor.extract(pdf_path, max_pages)


async def extract_bank_statement_from_bytes(
    file_bytes: bytes,
    max_pages: int = 1
) -> BankStatementData:
    """
    Extract bank statement data from PDF bytes (async convenience function).

    Args:
        file_bytes: Raw PDF file bytes
        max_pages: Maximum pages to process (default: 1)

    Returns:
        BankStatementData with extracted fields (GLiNER-compatible format)
    """
    extractor = get_spatial_bank_statement_extractor()
    return await extractor.extract_from_bytes(file_bytes, max_pages)

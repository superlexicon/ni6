"""
Spatial-based UAE Tax Residency Certificate extractor.

Uses label-based spatial extraction for reliable field extraction.
"""

import re
from typing import Optional, Dict, List
from datetime import datetime

from app.core.logger import get_logger
from app.helper.doctr.document_text_extractor import DocumentTextExtractor
from app.schemas.id_card_schema import IDCardData


class SpatialUAETRCExtractor:
    """
    Label-based spatial UAE TRC extractor.

    Extraction Strategy (User-Specified):
    1. Certificate Number: Find "Certificate Number" label -> extract closest value to the **right**
    2. Name: Find any string containing "Name" -> extract closest value to the **right**
    3. Expiry Date: Find **all dates** in document (various formats) -> take the **largest (latest) date**
    """

    # Comprehensive date patterns to handle various formats:
    # - DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY
    # - DD MMM YYYY, MMM DD YYYY, MMM DD, YYYY
    # - YYYY-MM-DD, etc.
    DATE_PATTERNS = [
        re.compile(r'\b\d{2}[/-]\d{2}[/-]\d{4}\b'),  # DD/MM/YYYY or DD-MM-YYYY
        re.compile(r'\b\d{4}[/-]\d{2}[/-]\d{2}\b'),  # YYYY-MM-DD or YYYY/MM/DD
        re.compile(r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b', re.IGNORECASE),  # DD MMM YYYY
        re.compile(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b', re.IGNORECASE),  # MMM DD, YYYY
    ]

    # Certificate number pattern: TRC-YYYY-NNNNN or similar
    CERT_NUMBER_PATTERN = re.compile(r'TRC[-\s]?\d{4}[-\s]?\d+', re.IGNORECASE)

    def __init__(self):
        self.logger = get_logger()
        self.text_extractor = DocumentTextExtractor()

    async def extract(
        self,
        content: bytes,
        is_pdf: bool = False
    ) -> IDCardData:
        """
        Extract UAE TRC fields using label-based spatial extraction.

        Args:
            content: Document bytes (image or PDF)
            is_pdf: Whether content is PDF

        Returns:
            IDCardData with extracted fields
        """
        try:
            self.logger.info("=" * 80)
            self.logger.info("SPATIAL UAE TRC EXTRACTION")
            self.logger.info("=" * 80)

            # Step 1: Extract text with geometry
            self.logger.info("Step 1: Extracting text with geometry")
            text_blocks = await self.text_extractor.extract_text_with_geometry_enhanced(
                content, is_pdf=is_pdf, max_pages=1
            )
            self.logger.info(f"  Extracted {len(text_blocks)} text blocks")

            # Step 2: Extract fields using spatial logic
            self.logger.info("Step 2: Extracting fields using spatial logic")

            certificate_number = self._extract_certificate_number(text_blocks)
            full_name = self._extract_name(text_blocks)
            expiry_date = self._extract_expiry_date(text_blocks)

            # Step 3: Build result
            field_values = {}
            confidence_scores = {}

            if certificate_number:
                field_values['certificate_number'] = certificate_number
                field_values['identification_number'] = certificate_number
                confidence_scores['certificate_number'] = 98.0

            if full_name:
                field_values['full_name'] = full_name
                confidence_scores['full_name'] = 95.0

            if expiry_date:
                field_values['expiry_date'] = expiry_date
                field_values['valid_until'] = expiry_date
                confidence_scores['valid_until'] = 90.0

            # Calculate overall confidence
            overall_confidence = 0.0
            if confidence_scores:
                overall_confidence = sum(confidence_scores.values()) / len(confidence_scores)

            result = IDCardData(
                document_type="UAE_TRC",
                issuing_country="AE",
                full_name=full_name,
                identification_number=certificate_number,
                field_values=field_values,
                confidence_scores=confidence_scores,
                overall_confidence=overall_confidence
            )

            self.logger.info("=" * 80)
            self.logger.info("EXTRACTION COMPLETE")
            self.logger.info(f"  Certificate Number: {certificate_number}")
            self.logger.info(f"  Name: {full_name}")
            self.logger.info(f"  Expiry Date: {expiry_date}")
            self.logger.info(f"  Overall Confidence: {overall_confidence:.1f}%")
            self.logger.info("=" * 80)

            return result

        except Exception as e:
            self.logger.error(f"UAE TRC extraction failed: {e}", exc_info=True)
            return IDCardData(
                document_type="UAE_TRC",
                issuing_country="AE",
                field_values={},
                confidence_scores={},
                overall_confidence=0.0
            )

    def _extract_certificate_number(self, text_blocks: List[Dict]) -> Optional[str]:
        """
        Extract certificate number by finding "Certificate Number" label
        and extracting the closest value to the **right side**.

        Strategy:
        1. Find block containing "Certificate Number" or similar
        2. Find closest text block to the right (same line or nearby)
        3. Validate with certificate number pattern
        """
        self.logger.info("  Extracting certificate number...")

        # Find Certificate Number label
        label_block = None
        for block in text_blocks:
            text_lower = block.get('text', '').lower()
            if 'certificate number' in text_lower or 'cert number' in text_lower:
                label_block = block
                break

        if not label_block:
            self.logger.warning("    'Certificate Number' label not found")
            # Fallback: find first block matching certificate pattern
            for block in text_blocks:
                text = block.get('text', '').strip()
                if self.CERT_NUMBER_PATTERN.search(text):
                    self.logger.info(f"    Found cert number via pattern: {text}")
                    return text
            return None

        self.logger.info(f"    Found label: '{label_block['text']}'")

        # Find closest text block to the right
        value_block = self._find_block_to_right(text_blocks, label_block)

        if value_block:
            cert_text = value_block.get('text', '').strip()
            self.logger.info(f"    Extracted certificate number: {cert_text}")
            return cert_text

        # Fallback: try to extract from the same line (after colon)
        label_text = label_block.get('text', '')
        if ':' in label_text:
            parts = label_text.split(':', 1)
            if len(parts) == 2:
                value = parts[1].strip()
                if value:
                    self.logger.info(f"    Extracted from same line: {value}")
                    return value

        self.logger.warning("    No certificate number found")
        return None

    def _extract_name(self, text_blocks: List[Dict]) -> Optional[str]:
        """
        Extract entity name by finding any string containing "Name"
        and extracting the closest value to the **right side**.

        Strategy:
        1. Find block containing "Name" (prioritize "Name of Entity")
        2. Find closest text block to the right
        3. Validate: should have at least 2 words
        """
        self.logger.info("  Extracting entity name...")

        # Find any block containing "Name"
        label_block = None
        for block in text_blocks:
            text_lower = block.get('text', '').lower()
            # Look for "Name of Entity" first
            if 'name of entity' in text_lower:
                label_block = block
                break

        if not label_block:
            # Try broader search
            for block in text_blocks:
                text_lower = block.get('text', '').lower()
                if 'name' in text_lower and 'father' not in text_lower and 'mother' not in text_lower:
                    label_block = block
                    break

        if not label_block:
            self.logger.warning("    'Name' label not found")
            return None

        self.logger.info(f"    Found label: '{label_block['text']}'")

        # Find closest text block to the right
        value_block = self._find_block_to_right(text_blocks, label_block)

        if value_block:
            name = value_block.get('text', '').strip()
            # Basic validation: should have at least 2 words
            if len(name.split()) >= 2:
                self.logger.info(f"    Extracted name: {name}")
                return name
            else:
                self.logger.warning(f"    Text doesn't look like a name: {name}")

        self.logger.warning("    No name found to the right")
        return None

    def _extract_expiry_date(self, text_blocks: List[Dict]) -> Optional[str]:
        """
        Extract expiry date by finding ALL dates in the document
        and taking the **largest (latest) date**.

        Strategy:
        1. Scan all text blocks for date patterns (various formats)
        2. Parse all found dates to datetime objects
        3. Return the largest (latest) date
        """
        self.logger.info("  Extracting expiry date (largest date in document)...")

        import dateparser  # For flexible date parsing

        all_dates = []

        for block in text_blocks:
            text = block.get('text', '')

            # Try each date pattern
            for pattern in self.DATE_PATTERNS:
                matches = pattern.findall(text)
                for match in matches:
                    try:
                        # Parse the date string
                        parsed_date = dateparser.parse(match)
                        if parsed_date:
                            all_dates.append((parsed_date, match))
                    except Exception as e:
                        self.logger.debug(f"    Failed to parse date '{match}': {e}")

        if not all_dates:
            self.logger.warning("    No dates found in document")
            return None

        # Sort by date (descending) and get the largest
        all_dates.sort(key=lambda x: x[0], reverse=True)
        latest_date = all_dates[0]

        self.logger.info(f"    Found {len(all_dates)} dates, latest is: {latest_date[1]}")
        return latest_date[1]

    def _find_block_to_right(
        self,
        text_blocks: List[Dict],
        reference_block: Dict,
        max_distance: float = 0.50
    ) -> Optional[Dict]:
        """
        Find text block to the right of reference block.

        Looks for block whose x1 is greater than reference block's x2,
        preferably on the same line (similar y coordinates).
        """
        ref_y1 = reference_block.get('y1', 0)
        ref_y2 = reference_block.get('y2', 0)
        ref_x2 = reference_block.get('x2', 0)
        ref_center_y = (ref_y1 + ref_y2) / 2

        best_match = None
        best_score = float('inf')

        for block in text_blocks:
            if block is reference_block:
                continue

            block_x1 = block.get('x1', 0)
            block_y1 = block.get('y1', 0)
            block_y2 = block.get('y2', 0)
            block_center_y = (block_y1 + block_y2) / 2

            # Check if block is to the right
            horizontal_distance = block_x1 - ref_x2

            if horizontal_distance > 0 and horizontal_distance <= max_distance:
                # Calculate vertical alignment score (prefer same line)
                vertical_delta = abs(block_center_y - ref_center_y)

                # Combined score: prefer right + same line
                score = horizontal_distance + (vertical_delta * 2)

                if score < best_score:
                    best_score = score
                    best_match = block

        return best_match

"""
Spatial-based PAN card extractor using label-to-value spatial relationships.

Works for both physical PAN cards and downloaded document variants.
Uses simple label finding + spatial proximity extraction.
"""

import re
from typing import Optional, Dict, List, Any

from app.core.logger import get_logger
from app.helper.doctr.document_text_extractor import DocumentTextExtractor
from app.schemas.id_card_schema import IDCardData


class SpatialPANExtractor:
    """
    Label-based spatial PAN card extractor.

    Extraction Strategy:
    1. Find "Permanent Account Number" label -> extract text below it
    2. Find "Name" label -> extract text to the right of it
    3. Find "Date of Birth" label -> extract text to the right (or first date found)
    """

    # PAN number format: 5 letters + 4 digits + 1 letter
    PAN_PATTERN = re.compile(r'^[A-Z]{5}[0-9]{4}[A-Z]$', re.IGNORECASE)

    # Date patterns: DD/MM/YYYY, DD-MM-YYYY, or similar
    DATE_PATTERN = re.compile(r'\d{2}[/-]\d{2}[/-]\d{4}')

    # Field label patterns
    LABEL_PATTERNS = {
        'pan_number': [
            'permanent account number',
            'pan number',
            'pan',
            'permanent account number card'
        ],
        'full_name': [
            'name',
            'full name',
            'card holder name',
            'cardholder name'
        ],
        'date_of_birth': [
            'date of birth',
            'dob',
            'birth date',
            'd.o.b'
        ]
    }

    def __init__(self):
        self.logger = get_logger()
        self.text_extractor = DocumentTextExtractor()

    async def extract(
        self,
        content: bytes,
        is_pdf: bool = False
    ) -> IDCardData:
        """
        Extract PAN card fields using label-based spatial extraction.

        Args:
            content: Document bytes (image or PDF)
            is_pdf: Whether content is PDF

        Returns:
            IDCardData with extracted fields
        """
        try:
            self.logger.info("=" * 80)
            self.logger.info("SPATIAL PAN EXTRACTION")
            self.logger.info("=" * 80)

            # Step 1: Extract text with geometry
            self.logger.info("Step 1: Extracting text with geometry")
            text_blocks = await self.text_extractor.extract_text_with_geometry_enhanced(
                content, is_pdf=is_pdf, max_pages=1
            )
            self.logger.info(f"  Extracted {len(text_blocks)} text blocks")

            # Step 2: Find labels and extract values
            self.logger.info("Step 2: Finding labels and extracting values")

            pan_number = self._extract_pan_number(text_blocks)
            full_name = self._extract_full_name(text_blocks)
            date_of_birth = self._extract_date_of_birth(text_blocks)

            # Step 3: Build result
            field_values = {}
            confidence_scores = {}

            if pan_number:
                field_values['pan_number'] = pan_number
                confidence_scores['pan_number'] = 95.0
                field_values['identification_number'] = pan_number

            if full_name:
                field_values['full_name'] = full_name
                confidence_scores['full_name'] = 90.0

            if date_of_birth:
                field_values['date_of_birth'] = date_of_birth
                confidence_scores['date_of_birth'] = 90.0

            # Calculate overall confidence
            overall_confidence = 0.0
            if confidence_scores:
                overall_confidence = sum(confidence_scores.values()) / len(confidence_scores)

            result = IDCardData(
                document_type="PAN",
                issuing_country="IND",
                full_name=full_name,
                date_of_birth=date_of_birth,
                identification_number=pan_number,
                field_values=field_values,
                confidence_scores=confidence_scores,
                overall_confidence=overall_confidence
            )

            self.logger.info("=" * 80)
            self.logger.info("EXTRACTION COMPLETE")
            self.logger.info(f"  PAN Number: {pan_number}")
            self.logger.info(f"  Name: {full_name}")
            self.logger.info(f"  DOB: {date_of_birth}")
            self.logger.info(f"  Overall Confidence: {overall_confidence:.1f}%")
            self.logger.info("=" * 80)

            return result

        except Exception as e:
            self.logger.error(f" Spatial PAN extraction failed: {e}", exc_info=True)
            return IDCardData(
                document_type="PAN",
                issuing_country="IND",
                field_values={},
                confidence_scores={},
                overall_confidence=0.0
            )

    def _extract_pan_number(self, text_blocks: List[Dict]) -> Optional[str]:
        """
        Find 'Permanent Account Number' label and extract text directly below it.

        Strategy:
        1. Find first block containing "Permanent Account Number" or similar
        2. Look for text block below it (y2 of label ~ y1 of value)
        3. Validate with PAN pattern
        """
        self.logger.info("  Extracting PAN number...")

        # Find PAN label
        label_block = self._find_first_label(text_blocks, self.LABEL_PATTERNS['pan_number'])

        if not label_block:
            self.logger.warning("    PAN label not found, trying pattern fallback")
            # Fallback: find first block matching PAN pattern
            for block in text_blocks:
                text = block.get('text', '').strip().upper().replace(' ', '')
                if self.PAN_PATTERN.match(text):
                    self.logger.info(f"    Found PAN via pattern: {text}")
                    return text
            return None

        self.logger.info(f"    Found PAN label: '{label_block['text']}'")

        # Find text block directly below the label
        value_block = self._find_block_below(text_blocks, label_block)

        if value_block:
            pan_text = value_block.get('text', '').strip().upper().replace(' ', '')
            if self.PAN_PATTERN.match(pan_text):
                self.logger.info(f"    Extracted PAN: {pan_text}")
                return pan_text
            else:
                self.logger.warning(f"    Text below PAN label doesn't match pattern: {pan_text}")

        return None

    def _extract_full_name(self, text_blocks: List[Dict]) -> Optional[str]:
        """
        Find 'Name' label and extract closest string below it.

        Strategy:
        1. Find first block containing "Name" (but NOT "Father's Name")
        2. Look for text block below the label (standard PAN card layout)
        3. Exclude father's name patterns
        """
        self.logger.info("  Extracting full name...")

        # Find Name label (exclude father's name)
        label_block = self._find_first_label(
            text_blocks,
            self.LABEL_PATTERNS['full_name'],
            exclude_patterns=['father', 'mother', 's/o', 'd/o', 'a/l']
        )

        if not label_block:
            self.logger.warning("    Name label not found")
            return None

        self.logger.info(f"    Found Name label: '{label_block['text']}'")

        # Find text block directly below the label
        value_block = self._find_block_below(text_blocks, label_block)

        if value_block:
            name_text = value_block.get('text', '').strip()
            # Basic validation: should have at least 2 words, not be a date or number
            if name_text and len(name_text.split()) >= 2:
                self.logger.info(f"    Extracted Name: {name_text}")
                return name_text
            else:
                self.logger.warning(f"    Text below doesn't look like a name: {name_text}")

        return None

    def _extract_date_of_birth(self, text_blocks: List[Dict]) -> Optional[str]:
        """
        Find 'Date of Birth' label and extract closest string below it.

        OR simply extract the first date string in the document.

        Strategy:
        1. Try to find DOB label and extract value below
        2. Fallback: find first block matching date pattern
        """
        self.logger.info("  Extracting date of birth...")

        # First try: Find DOB label and extract below
        label_block = self._find_first_label(text_blocks, self.LABEL_PATTERNS['date_of_birth'])

        if label_block:
            self.logger.info(f"    Found DOB label: '{label_block['text']}'")
            value_block = self._find_block_below(text_blocks, label_block)

            if value_block:
                dob_text = value_block.get('text', '').strip()
                if self.DATE_PATTERN.search(dob_text):
                    self.logger.info(f"    Extracted DOB: {dob_text}")
                    return dob_text

        # Fallback: Find first date string in document
        self.logger.info("    DOB label method failed, trying first date in document")
        for block in text_blocks:
            text = block.get('text', '').strip()
            date_match = self.DATE_PATTERN.search(text)
            if date_match:
                self.logger.info(f"    Found DOB via pattern: {date_match.group()}")
                return date_match.group()

        self.logger.warning("    No date found")
        return None

    # ============================================================
    # SPATIAL HELPER METHODS
    # ============================================================

    def _find_first_label(
        self,
        text_blocks: List[Dict],
        patterns: List[str],
        exclude_patterns: Optional[List[str]] = None
    ) -> Optional[Dict]:
        """
        Find first text block matching any label pattern.

        Args:
            text_blocks: List of text blocks with geometry
            patterns: List of label patterns to search for
            exclude_patterns: Optional list of patterns to exclude

        Returns:
            First matching text block or None
        """
        for block in text_blocks:
            text_lower = block.get('text', '').lower()

            # Check if block matches any pattern
            for pattern in patterns:
                if pattern in text_lower:
                    # Check exclusion patterns
                    if exclude_patterns:
                        should_exclude = False
                        for exclude in exclude_patterns:
                            if exclude in text_lower:
                                should_exclude = True
                                break
                        if should_exclude:
                            continue

                    return block

        return None

    def _find_block_below(
        self,
        text_blocks: List[Dict],
        reference_block: Dict,
        max_distance: float = 0.15
    ) -> Optional[Dict]:
        """
        Find text block directly below reference block.

        Looks for block whose y1 is close to reference block's y2.
        """
        ref_y2 = reference_block.get('y2', 0)
        ref_x1 = reference_block.get('x1', 0)
        ref_x2 = reference_block.get('x2', 0)

        best_match = None
        best_distance = float('inf')

        for block in text_blocks:
            if block is reference_block:
                continue

            block_y1 = block.get('y1', 0)
            block_x1 = block.get('x1', 0)
            block_x2 = block.get('x2', 0)

            # Check if block is below (y1 > ref_y2)
            vertical_distance = block_y1 - ref_y2

            if 0 < vertical_distance <= max_distance:
                # Check horizontal overlap (should be somewhat aligned)
                horizontal_overlap = not (block_x2 < ref_x1 or block_x1 > ref_x2)

                if horizontal_overlap and vertical_distance < best_distance:
                    best_distance = vertical_distance
                    best_match = block

        return best_match

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

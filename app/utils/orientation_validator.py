"""
Orientation Validator - Validates document orientation based on OCR extraction results.

Documents (except selfies) must be submitted in the correct upright orientation.
When a document is rotated (90°, 180°, 270°), DocTr fails to extract text properly
but doesn't throw a specific error - it simply returns empty or poor results.

This validator detects such cases by checking that sufficient text was extracted.
"""

from typing import Tuple, Optional
from app.core.logger import get_logger


class OrientationValidator:
    """Validates document orientation based on OCR extraction results."""

    def __init__(self):
        self.logger = get_logger()

    def validate_orientation(
        self,
        text_blocks: list,
        document_type: str,
        image_size: Optional[tuple] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate that document is in correct orientation based on OCR results.

        Args:
            text_blocks: OCR extracted text blocks
            document_type: Type of document (selfie, passport, bank_statement, etc.)
            image_size: Optional (width, height) of image for additional checks

        Returns:
            (is_valid, error_message)
        """
        from app.config.verification_config import verification_settings

        # Skip orientation check for selfies
        if document_type == "selfie":
            return True, None

        # Check if validation is enabled
        if not verification_settings.enable_orientation_validation:
            return True, None

        # Count extracted elements
        block_count = len(text_blocks) if text_blocks else 0

        # Count lines within blocks - handle both flat list and nested structures
        line_count = 0
        if text_blocks:
            # Check if this is a flat list of lines (DocTR format)
            # or a nested list of blocks (legacy format)
            first_item = text_blocks[0] if text_blocks else None

            if first_item and isinstance(first_item, dict):
                # Flat list of lines from DocTR - each item has 'text' key
                if 'text' in first_item:
                    # This is the DocTR format - flat list of lines
                    line_count = block_count  # Each item is a line
                elif 'lines' in first_item:
                    # Legacy format - blocks containing lines
                    for block in text_blocks:
                        lines = block.get('lines', [])
                        line_count += len(lines)
            elif hasattr(first_item, 'lines'):
                # Object with lines attribute
                for block in text_blocks:
                    line_count += len(block.lines)

        # Get thresholds
        min_blocks = verification_settings.min_text_blocks_for_valid_doc
        min_lines = verification_settings.min_text_lines_for_valid_doc

        self.logger.info(
            f"Orientation validation for {document_type}: "
            f"{block_count} blocks (min {min_blocks}), "
            f"{line_count} lines (min {min_lines})"
        )

        # Check thresholds
        if block_count < min_blocks or line_count < min_lines:
            return (
                False,
                f"Document appears to be rotated. Please ensure the document is "
                f"upright (right-side up) before submitting. "
                f"Extracted: {block_count} blocks, {line_count} lines. "
                f"Required: at least {min_blocks} blocks and {min_lines} lines."
            )

        return True, None

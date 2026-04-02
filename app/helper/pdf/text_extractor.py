"""
PDF Text Extractor with Spatial Data
Extracts text from PDFs while preserving spatial coordinates and formatting information.
Compatible with existing key injection system.
"""

import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional, Tuple
import logging


class TextElement:
    """Text element with spatial information, compatible with key injection system."""

    def __init__(self, text: str, confidence: float = 1.0, geometry: Optional[Dict] = None):
        self.text = text
        # Force confidence to 1.0 for direct extraction since it's 100% accurate
        self.confidence = 1.0
        self.geometry = geometry or {
            'x': 0, 'y': 0, 'width': 0, 'height': 0,
            'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0
        }


class PDFTextExtractor:
    """Extracts text from PDFs with spatial data preservation."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def extract_text_with_spatial_data(self, pdf_path: str, max_pages: int = 1) -> List[TextElement]:
        """
        Extract text from PDF with spatial coordinates.

        Args:
            pdf_path: Path to PDF file
            max_pages: Maximum number of pages to process (default: 1 for efficiency)

        Returns:
            List of TextElement objects with positioning data
        """
        try:
            doc = fitz.open(pdf_path)
            all_elements = []

            self.logger.info(f"Starting PDF text extraction from {pdf_path} (max {max_pages} pages)")

            # Limit processing to max_pages for efficiency
            pages_to_process = min(max_pages, doc.page_count)

            for page_num in range(pages_to_process):
                page = doc[page_num]
                page_elements = self._extract_page_text(page, page_num)
                all_elements.extend(page_elements)

                self.logger.debug(f"Page {page_num + 1}: extracted {len(page_elements)} text elements")

            doc.close()

            self.logger.info(f"PDF text extraction complete: {len(all_elements)} total elements from {pages_to_process} page(s)")
            return all_elements

        except Exception as e:
            self.logger.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
            raise

    def _extract_page_text(self, page, page_num: int) -> List[TextElement]:
        """
        Extract text from a single page with spatial data.

        Args:
            page: PyMuPDF page object
            page_num: Page number (0-indexed)

        Returns:
            List of TextElement objects from this page
        """
        elements = []

        try:
            # Get text blocks with their positions
            blocks = page.get_text("dict")

            for block in blocks.get("blocks", []):
                if "lines" in block:  # Text block
                    block_elements = self._process_text_block(block, page_num)
                    elements.extend(block_elements)
                elif "image" in block:  # Image block - skip for text extraction
                    continue

            # Also extract text from form fields (widgets) which may contain labels
            # Some PDFs have labels like "Primary Account Holder Name" in form fields
            widget_elements = self._extract_form_fields(page, page_num)
            elements.extend(widget_elements)

        except Exception as e:
            self.logger.warning(f"Error extracting text from page {page_num + 1}: {str(e)}")

        # Sort elements by Y position (top-to-bottom), then by X position (left-to-right)
        # This ensures tabular layouts with values in right columns are correctly ordered
        elements.sort(key=lambda e: (e.geometry.get('y1', 0), e.geometry.get('x1', 0)))

        return elements

    def _extract_form_fields(self, page, page_num: int) -> List[TextElement]:
        """
        Extract text from form fields (widgets) on the page.

        Some PDFs store labels like "Primary Account Holder Name" in form fields
        rather than regular text blocks. This method extracts those.

        Args:
            page: PyMuPDF page object
            page_num: Page number (0-indexed)

        Returns:
            List of TextElement objects from form fields
        """
        elements = []

        try:
            # Get all widgets (form fields) on the page
            widgets = page.widgets()

            if not widgets:
                return elements

            self.logger.debug(f"Found {len(widgets)} form fields on page {page_num + 1}")

            for widget in widgets:
                # Get the field name (which is often the label)
                field_name = widget.field_name
                if field_name and field_name.strip():
                    # Get the widget's bounding box
                    rect = widget.rect

                    # Create geometry dictionary
                    geometry = {
                        'x': int(rect.x0),
                        'y': int(rect.y0),
                        'width': int(rect.x1 - rect.x0),
                        'height': int(rect.y1 - rect.y0),
                        'x1': int(rect.x0),
                        'y1': int(rect.y0),
                        'x2': int(rect.x1),
                        'y2': int(rect.y1),
                        'font': '',
                        'font_size': 12,
                        'flags': 0,
                        'page_num': page_num,
                        'block_type': 'form_field'
                    }

                    element = TextElement(
                        text=field_name.strip(),
                        confidence=1.0,
                        geometry=geometry
                    )

                    elements.append(element)
                    self.logger.debug(f"Extracted form field: '{field_name}' at ({rect.x0}, {rect.y0})")

        except Exception as e:
            self.logger.debug(f"Error extracting form fields from page {page_num + 1}: {str(e)}")

        return elements

    def _process_text_block(self, block: Dict, page_num: int) -> List[TextElement]:
        """
        Process a text block and extract individual text elements.

        Args:
            block: Text block dictionary from PyMuPDF
            page_num: Page number

        Returns:
            List of TextElement objects
        """
        elements = []

        try:
            for line in block.get("lines", []):
                line_elements = self._process_text_line(line, block, page_num)
                elements.extend(line_elements)

        except Exception as e:
            self.logger.warning(f"Error processing text block on page {page_num + 1}: {str(e)}")

        return elements

    def _process_text_line(self, line: Dict, block: Dict, page_num: int) -> List[TextElement]:
        """
        Process a text line and extract individual spans.

        Args:
            line: Text line dictionary
            block: Parent text block
            page_num: Page number

        Returns:
            List of TextElement objects
        """
        elements = []

        try:
            for span in line.get("spans", []):
                element = self._create_text_element(span, line, block, page_num)
                if element and element.text.strip():
                    elements.append(element)

        except Exception as e:
            self.logger.warning(f"Error processing text line on page {page_num + 1}: {str(e)}")

        return elements

    def _create_text_element(self, span: Dict, line: Dict, block: Dict, page_num: int) -> Optional[TextElement]:
        """
        Create a TextElement from a span with spatial data.

        Args:
            span: Text span dictionary
            line: Parent text line
            block: Parent text block
            page_num: Page number

        Returns:
            TextElement object or None if text is empty
        """
        try:
            text = span.get("text", "").strip()
            if not text:
                return None

            # Get bounding box coordinates
            bbox = span.get("bbox", [0, 0, 0, 0])
            x0, y0, x1, y1 = bbox

            # Convert to normalized coordinates (0-1 range) for compatibility with key injection
            page_height = 842  # Standard letter size height in points
            page_width = 595   # Standard letter size width in points

            # Create geometry dictionary in the format expected by key injection system
            geometry = {
                'x': int(x0),
                'y': int(y0),
                'width': int(x1 - x0),
                'height': int(y1 - y0),
                'x1': int(x0),
                'y1': int(y0),
                'x2': int(x1),
                'y2': int(y1),
                # Additional metadata for enhanced processing
                'font': span.get("font", ""),
                'font_size': span.get("size", 12),
                'flags': span.get("flags", 0),
                'page_num': page_num,
                'block_type': 'text'
            }

            # Create TextElement with perfect confidence (direct extraction)
            element = TextElement(
                text=text,
                confidence=1.0,  # Perfect confidence for direct extraction
                geometry=geometry
            )

            return element

        except Exception as e:
            self.logger.warning(f"Error creating text element: {str(e)}")
            return None

    def extract_text_lines_with_coordinates(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract text as lines with coordinates, useful for debugging.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of dictionaries with text and coordinate information
        """
        text_elements = self.extract_text_with_spatial_data(pdf_path)

        lines_data = []
        for element in text_elements:
            line_data = {
                'text': element.text,
                'x': element.geometry['x'],
                'y': element.geometry['y'],
                'width': element.geometry['width'],
                'height': element.geometry['height'],
                'x1': element.geometry['x1'],
                'y1': element.geometry['y1'],
                'x2': element.geometry['x2'],
                'y2': element.geometry['y2'],
                'font': element.geometry.get('font', ''),
                'font_size': element.geometry.get('font_size', 12),
                'page_num': element.geometry.get('page_num', 0),
                'confidence': element.confidence
            }
            lines_data.append(line_data)

        return lines_data

    def extract_text_as_lines(self, pdf_path: str, normalize: bool = True) -> List[str]:
        """
        Extract text as simple lines without spatial data.

        Args:
            pdf_path: Path to PDF file
            normalize: Whether to normalize whitespace

        Returns:
            List of text lines
        """
        text_elements = self.extract_text_with_spatial_data(pdf_path)

        # Group elements by line (similar y-coordinates)
        lines = {}
        for element in text_elements:
            line_key = round(element.geometry['y'], 1)  # Group by y-coordinate

            if line_key not in lines:
                lines[line_key] = []
            lines[line_key].append(element)

        # Sort by y-coordinate and concatenate text in each line
        sorted_lines = sorted(lines.keys())
        result = []

        for line_key in sorted_lines:
            # Sort elements in line by x-coordinate
            line_elements = sorted(lines[line_key], key=lambda e: e.geometry['x'])
            line_text = " ".join(element.text for element in line_elements)

            if normalize:
                line_text = " ".join(line_text.split())  # Normalize whitespace

            if line_text.strip():
                result.append(line_text)

        return result

    def extract_image_boundaries(
        self,
        pdf_path: str,
        max_pages: int = 1,
        page_width: float = 595,
        page_height: float = 842
    ) -> List[Dict[str, Any]]:
        """
        Extract image bounding boxes from PDF.

        Images act as visual separators in bank statements. This method returns
        their positions so they can be used as "hard breaks" in text grouping.

        Args:
            pdf_path: Path to PDF file
            max_pages: Maximum number of pages to process (default: 1)
            page_width: Page width in points for normalization (default: 595)
            page_height: Page height in points for normalization (default: 842)

        Returns:
            List of dictionaries with normalized image boundaries:
            [{'x1': float, 'y1': float, 'x2': float, 'y2': float, 'page_num': int}, ...]
            Coordinates are normalized to 0-1 range.
        """
        try:
            doc = fitz.open(pdf_path)
            image_boundaries = []

            self.logger.info(f"Starting image boundary extraction from {pdf_path} (max {max_pages} pages)")

            # Limit processing to max_pages for efficiency
            pages_to_process = min(max_pages, doc.page_count)

            for page_num in range(pages_to_process):
                page = doc[page_num]

                # Get all blocks (including images)
                blocks = page.get_text("dict")

                for block in blocks.get("blocks", []):
                    if "image" in block:  # Image block
                        bbox = block.get("bbox", [0, 0, 0, 0])
                        x0, y0, x1, y1 = bbox

                        # Normalize coordinates to 0-1 range
                        image_boundary = {
                            'x1': x0 / page_width,
                            'y1': y0 / page_height,
                            'x2': x1 / page_width,
                            'y2': y1 / page_height,
                            'page_num': page_num
                        }
                        image_boundaries.append(image_boundary)
                        self.logger.debug(
                            f"Found image on page {page_num + 1}: "
                            f"normalized bbox ({image_boundary['x1']:.3f}, "
                            f"{image_boundary['y1']:.3f}, {image_boundary['x2']:.3f}, "
                            f"{image_boundary['y2']:.3f})"
                        )

            doc.close()

            self.logger.info(f"Image boundary extraction complete: {len(image_boundaries)} images found")
            return image_boundaries

        except Exception as e:
            self.logger.error(f"Error extracting image boundaries from PDF {pdf_path}: {str(e)}")
            return []


# Global instance for easy access
pdf_text_extractor = PDFTextExtractor()
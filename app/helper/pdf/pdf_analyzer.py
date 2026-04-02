"""
PDF Content Type Analyzer
Analyzes PDF files to determine if they contain extractable text or are image-based (scanned).
"""

import os
from typing import Tuple, Dict, Any
from pathlib import Path
import fitz  # PyMuPDF


class PDFContentType:
    """Types of PDF content."""
    TEXT_BASED = "text_based"      # PDF contains extractable text
    IMAGE_BASED = "image_based"    # PDF is scanned images only
    MIXED = "mixed"               # PDF has both text and images


class PDFAnalyzer:
    """Analyzes PDF files to determine content type and extract metadata."""

    def __init__(self):
        self.logger = self._get_logger()

    def _get_logger(self):
        """Get logger instance."""
        import logging
        return logging.getLogger(__name__)

    def analyze_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Analyze a PDF file to determine its content type and characteristics.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dictionary containing analysis results
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        try:
            # Open PDF with PyMuPDF
            doc = fitz.open(pdf_path)

            analysis = {
                'file_path': pdf_path,
                'page_count': doc.page_count,
                'content_type': PDFContentType.IMAGE_BASED,
                'text_pages': 0,
                'image_pages': 0,
                'mixed_pages': 0,
                'total_text_blocks': 0,
                'total_images': 0,
                'pages_details': [],
                'has_extractable_text': False,
                'recommended_extraction': 'ocr'  # Default to OCR
            }

            # Analyze each page
            for page_num in range(doc.page_count):
                page = doc[page_num]
                page_analysis = self._analyze_page(page, page_num)

                analysis['pages_details'].append(page_analysis)
                analysis['total_text_blocks'] += page_analysis['text_blocks']
                analysis['total_images'] += page_analysis['images']

                # Categorize page type
                if page_analysis['text_blocks'] > 0 and page_analysis['images'] == 0:
                    analysis['text_pages'] += 1
                elif page_analysis['text_blocks'] == 0 and page_analysis['images'] > 0:
                    analysis['image_pages'] += 1
                elif page_analysis['text_blocks'] > 0 and page_analysis['images'] > 0:
                    analysis['mixed_pages'] += 1

            doc.close()

            # Determine overall content type
            if analysis['text_pages'] == analysis['page_count']:
                analysis['content_type'] = PDFContentType.TEXT_BASED
                analysis['has_extractable_text'] = True
                analysis['recommended_extraction'] = 'direct'
            elif analysis['image_pages'] == analysis['page_count']:
                analysis['content_type'] = PDFContentType.IMAGE_BASED
                analysis['has_extractable_text'] = False
                analysis['recommended_extraction'] = 'ocr'
            else:
                analysis['content_type'] = PDFContentType.MIXED
                analysis['has_extractable_text'] = analysis['total_text_blocks'] > 0
                # For mixed content, prefer direct text extraction if extractable text exists
                # This handles cases where pages have both text (for data) and images (logos/watermarks)
                if analysis['total_text_blocks'] > 0:
                    # Has extractable text - use direct extraction with OCR fallback
                    analysis['recommended_extraction'] = 'direct_with_ocr_fallback'
                else:
                    # No extractable text - use OCR only
                    analysis['recommended_extraction'] = 'ocr'

            self.logger.info(f"PDF Analysis Complete: {analysis['content_type']} "
                           f"({analysis['text_pages']} text, {analysis['image_pages']} image, "
                           f"{analysis['mixed_pages']} mixed pages)")

            return analysis

        except Exception as e:
            self.logger.error(f"Error analyzing PDF {pdf_path}: {str(e)}")
            raise

    def _analyze_page(self, page, page_num: int) -> Dict[str, Any]:
        """
        Analyze a single page for text and image content.

        Args:
            page: PyMuPDF page object
            page_num: Page number (0-indexed)

        Returns:
            Dictionary with page analysis results
        """
        page_analysis = {
            'page_num': page_num,
            'text_blocks': 0,
            'images': 0,
            'has_text': False,
            'has_images': False,
            'text_sample': '',
            'image_details': []
        }

        try:
            # Count text blocks
            text_blocks = page.get_text("blocks")
            page_analysis['text_blocks'] = len(text_blocks)
            page_analysis['has_text'] = len(text_blocks) > 0

            # Get sample text for debugging
            if text_blocks:
                first_block = text_blocks[0]
                if len(first_block) >= 4:
                    page_analysis['text_sample'] = first_block[4][:100]  # First 100 chars of text

            # Count images
            image_list = page.get_images()
            page_analysis['images'] = len(image_list)
            page_analysis['has_images'] = len(image_list) > 0

            # Get image details
            for img_index, img in enumerate(image_list):
                img_info = {
                    'index': img_index,
                    'width': img[2],
                    'height': img[3],
                    'colorspace': img[5],
                    'bpc': img[6]  # bits per component
                }
                page_analysis['image_details'].append(img_info)

            self.logger.debug(f"Page {page_num + 1}: {len(text_blocks)} text blocks, {len(image_list)} images")

        except Exception as e:
            self.logger.warning(f"Error analyzing page {page_num + 1}: {str(e)}")

        return page_analysis

    def is_text_based_pdf(self, pdf_path: str) -> bool:
        """
        Quick check if PDF is text-based (contains extractable text).

        Args:
            pdf_path: Path to PDF file

        Returns:
            True if PDF contains extractable text
        """
        try:
            analysis = self.analyze_pdf(pdf_path)
            return analysis['has_extractable_text']
        except Exception as e:
            self.logger.error(f"Error checking if PDF is text-based {pdf_path}: {str(e)}")
            return False  # Default to OCR if analysis fails

    def get_page_count(self, pdf_path: str) -> int:
        """
        Get the number of pages in a PDF file.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Number of pages
        """
        try:
            doc = fitz.open(pdf_path)
            page_count = doc.page_count
            doc.close()
            return page_count
        except Exception as e:
            self.logger.error(f"Error getting page count for {pdf_path}: {str(e)}")
            return 0

    def has_scanned_content(self, pdf_path: str) -> bool:
        """
        Check if PDF contains scanned content (images without text).

        Args:
            pdf_path: Path to PDF file

        Returns:
            True if PDF contains scanned content
        """
        try:
            analysis = self.analyze_pdf(pdf_path)
            return analysis['image_pages'] > 0
        except Exception as e:
            self.logger.error(f"Error checking for scanned content in {pdf_path}: {str(e)}")
            return True  # Default to treating as scanned if analysis fails

    def detect_orientation(self, pdf_path: str, page_num: int = 0) -> Dict[str, Any]:
        """
        Detect page orientation.

        Args:
            pdf_path: Path to PDF file
            page_num: Page number to analyze (default: 0, first page)

        Returns:
            {
                'is_landscape': bool,
                'rotation': int,  # 0, 90, 180, 270 (from PyMuPDF)
                'width': float,
                'height': float,
                'aspect_ratio': float
            }
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        try:
            doc = fitz.open(pdf_path)
            page = doc[page_num]

            # Get actual page dimensions
            rect = page.rect
            width = rect.width
            height = rect.height

            # Check rotation metadata (some PDFs have explicit rotation)
            rotation = page.rotation  # 0, 90, 180, 270

            # Determine orientation
            # A4 is 210x297mm (ratio ~0.71), Letter is 8.5x11in (ratio ~0.77)
            # Landscape: ratio > 1.0, Portrait: ratio < 1.0
            is_landscape = width > height

            doc.close()

            return {
                'is_landscape': is_landscape,
                'rotation': rotation,
                'width': width,
                'height': height,
                'aspect_ratio': width / height
            }
        except Exception as e:
            self.logger.error(f"Error detecting orientation for {pdf_path}: {str(e)}")
            # Return default portrait orientation on error
            return {
                'is_landscape': False,
                'rotation': 0,
                'width': 595.0,  # Standard A4 width
                'height': 842.0,  # Standard A4 height
                'aspect_ratio': 0.71
            }


# Global instance for easy access
pdf_analyzer = PDFAnalyzer()
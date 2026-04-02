from typing import List, Tuple
from datetime import datetime
import io
from PIL import Image

from app.dto.document_analysis import PDFMetadataAnalysis
from app.core import logger


class PDFAnalysisService:
    """Service for analyzing PDF metadata and structure for forgery detection"""

    # Suspicious PDF producers that might indicate manipulation
    SUSPICIOUS_PRODUCERS = [
        "photoshop",
        "gimp",
        "inkscape",
        "illustrator",
        "scribus"
    ]

    def __init__(self):
        self.logger = logger

    async def convert_pdf_to_images(self, pdf_bytes: bytes, max_pages: int = None) -> List[Image.Image]:
        """
        Convert PDF pages to PIL Images for forgery detection

        Args:
            pdf_bytes: PDF file content as bytes
            max_pages: Maximum number of pages to convert (None for all pages)

        Returns:
            List of PIL Image objects (one per page)
        """
        try:
            import fitz  # PyMuPDF

            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            images = []

            # If max_pages is specified, limit the number of pages
            if max_pages is not None:
                pages_to_process = min(max_pages, pdf_document.page_count)
                self.logger.info(f"Limiting PDF processing to {pages_to_process} pages")
            else:
                pages_to_process = pdf_document.page_count

            for page_num in range(pages_to_process):
                page = pdf_document[page_num]

                # Render page to image at 300 DPI for good quality
                pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))

                # Convert to PIL Image
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                images.append(img)

            pdf_document.close()
            self.logger.info(f"Converted {len(images)} PDF pages to images")
            return images

        except Exception as e:
            self.logger.error(f"Error converting PDF to images: {str(e)}")
            raise ValueError(f"Failed to convert PDF to images: {str(e)}")

    async def analyze_pdf_metadata(self, pdf_bytes: bytes) -> PDFMetadataAnalysis:
        """
        Analyze PDF metadata for suspicious indicators

        Args:
            pdf_bytes: PDF file content as bytes

        Returns:
            PDFMetadataAnalysis with findings and authenticity score
        """
        try:
            import fitz  # PyMuPDF

            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            metadata = pdf_document.metadata

            suspicious_indicators = []
            authenticity_score = 100.0

            # Extract metadata fields
            creation_date = metadata.get("creationDate", "")
            modification_date = metadata.get("modDate", "")
            producer = metadata.get("producer", "")
            creator = metadata.get("creator", "")

            # Check 1: Suspicious producer/creator software
            if producer:
                producer_lower = producer.lower()
                for suspicious in self.SUSPICIOUS_PRODUCERS:
                    if suspicious in producer_lower:
                        suspicious_indicators.append(
                            f"Document created with image editing software: {producer}"
                        )
                        authenticity_score -= 30.0

            if creator:
                creator_lower = creator.lower()
                for suspicious in self.SUSPICIOUS_PRODUCERS:
                    if suspicious in creator_lower:
                        suspicious_indicators.append(
                            f"Document creator is image editing software: {creator}"
                        )
                        authenticity_score -= 30.0

            # Check 2: Recent modification after creation
            if creation_date and modification_date:
                try:
                    creation_dt = self._parse_pdf_date(creation_date)
                    mod_dt = self._parse_pdf_date(modification_date)

                    if creation_dt and mod_dt:
                        time_diff = (mod_dt - creation_dt).total_seconds()

                        # If modified more than 1 hour after creation, suspicious
                        if time_diff > 3600:
                            suspicious_indicators.append(
                                f"Document modified {int(time_diff/3600)} hours after creation"
                            )
                            authenticity_score -= 15.0
                except Exception as e:
                    self.logger.warning(f"Could not parse PDF dates: {e}")

            # Check 3: Missing standard metadata
            if not creation_date:
                suspicious_indicators.append("Missing creation date metadata")
                authenticity_score -= 10.0

            if not producer:
                suspicious_indicators.append("Missing producer metadata")
                authenticity_score -= 10.0

            # Check 4: Analyze embedded objects
            embedded_images = 0
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                image_list = page.get_images()
                embedded_images += len(image_list)

            # Multiple embedded images might indicate composite/manipulated document
            if embedded_images > 5:
                suspicious_indicators.append(
                    f"High number of embedded images ({embedded_images})"
                )
                authenticity_score -= 10.0

            pdf_document.close()

            # Ensure score doesn't go below 0
            authenticity_score = max(0.0, authenticity_score)

            self.logger.info(
                f"PDF metadata analysis complete. Authenticity: {authenticity_score}%, "
                f"Indicators: {len(suspicious_indicators)}"
            )

            return PDFMetadataAnalysis(
                suspicious_indicators=suspicious_indicators,
                authenticity_score=authenticity_score,
                creation_date=creation_date if creation_date else None,
                modification_date=modification_date if modification_date else None,
                producer=producer if producer else None,
                creator=creator if creator else None
            )

        except Exception as e:
            self.logger.error(f"Error analyzing PDF metadata: {str(e)}")
            raise ValueError(f"Failed to analyze PDF metadata: {str(e)}")

    def _parse_pdf_date(self, date_str: str) -> datetime:
        """
        Parse PDF date format (D:YYYYMMDDHHmmSS+HH'mm')

        Args:
            date_str: PDF date string

        Returns:
            datetime object or None if parsing fails
        """
        try:
            # Remove D: prefix if present
            if date_str.startswith("D:"):
                date_str = date_str[2:]

            # Remove timezone info (everything after +/-)
            date_str = date_str.split('+')[0].split('-')[0]

            # Parse the date
            return datetime.strptime(date_str[:14], "%Y%m%d%H%M%S")
        except Exception:
            return None

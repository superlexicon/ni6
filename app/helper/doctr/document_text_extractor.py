import io
from app.core.logger import get_logger
from app.utils.coordinate_transformer import CoordinateTransformer, OrientationInfo
from io import BytesIO
from typing import List, Optional, Dict, Any, Tuple
from PIL import Image, UnidentifiedImageError, ImageOps
# Lazy import to avoid dependency issues
def get_document_file():
    from doctr.io import DocumentFile
    return DocumentFile
from pypdf import PdfReader
import fitz  # PyMuPDF
from app.core import get_doctr_model
from app.core.key_injection import key_injection_manager, DocumentType
from app.core.key_injection.data_structures import OCRLinesContainer
from app.helper.pdf.pdf_analyzer import pdf_analyzer, PDFContentType
from app.helper.pdf.text_extractor import pdf_text_extractor
import tempfile
import os
logger = get_logger()


# Constants
DEFAULT_MAX_SIZE = 1600  # Increased from 800 for better OCR quality
DEFAULT_MAX_PAGES = 3


class DocumentTextExtractor:
    @staticmethod
    async def extract_text(file: bytes, is_pdf: bool = False, max_pages: int = DEFAULT_MAX_PAGES) -> str:
        logger.info(f"Extracting text from {'PDF' if is_pdf else 'image'}")

        if is_pdf:
            text = await DocumentTextExtractor._extract_from_pdf(file)
            if text:
                return text
            logger.debug("No text extracted from PDF, falling back to OCR")

        return await DocumentTextExtractor._extract_with_ocr(file, max_pages)

    @staticmethod
    async def extract_text_with_geometry(file: bytes, is_pdf: bool = False, max_pages: int = DEFAULT_MAX_PAGES) -> list:
        """
        Extract text from image with geometry information for each line.
        Uses EXIF auto-rotation only (no manual rotation attempts).

        Returns:
            List of dicts with keys: text, x1, y1, x2, y2 (bounding box coordinates)
        """
        logger.info(f"Extracting text with geometry from {'PDF' if is_pdf else 'image'}")

        try:
            # Convert to PNG with EXIF auto-rotation (no manual rotation)
            img_bytes = await ImageProcessor.convert_to_png(file, max_pages=max_pages)
            DocumentFile = get_document_file()
            doc = DocumentFile.from_images(img_bytes[:max_pages])

            # Run OCR
            model = get_doctr_model()
            result = model(doc)

            # Log raw block structure from DocTR
            logger.info("DocTR output structure:")
            for page_idx, page in enumerate(result.pages):
                logger.info(f"  Page {page_idx}: {len(page.blocks)} blocks")
                for block_idx, block in enumerate(page.blocks):
                    # Get block geometry
                    block_x1, block_y1 = block.geometry[0]
                    block_x2, block_y2 = block.geometry[1]
                    logger.info(
                        f"    Block {block_idx}: {len(block.lines)} lines, "
                        f"geometry=({block_x1:.3f}, {block_y1:.3f}, {block_x2:.3f}, {block_y2:.3f})"
                    )
                    for line_idx, line in enumerate(block.lines[:5]):  # Show first 5 lines per block
                        line_words = [word.value for word in line.words]
                        line_text = " ".join(line_words)
                        line_x1, line_y1 = line.geometry[0]
                        line_x2, line_y2 = line.geometry[1]
                        logger.info(
                            f"      Line {line_idx}: '{line_text}' "
                            f"({line_x1:.3f}, {line_y1:.3f}, {line_x2:.3f}, {line_y2:.3f})"
                        )
                    if len(block.lines) > 5:
                        logger.info(f"      ... and {len(block.lines) - 5} more lines")

            # Extract text with geometry - use LINES instead of blocks
            lines = []
            for page in result.pages:
                for block in page.blocks:
                    for line in block.lines:
                        # Get all words in the line with confidence scores
                        line_words = []
                        word_confidences = []

                        for word in line.words:
                            line_words.append(word.value)
                            word_confidences.append(word.confidence)

                        if line_words:
                            line_text = " ".join(line_words)
                            # Calculate average confidence for the line
                            line_confidence = sum(word_confidences) / len(word_confidences) if word_confidences else 0.0

                            # Get line geometry (normalized coordinates 0-1)
                            # line.geometry is ((x1, y1), (x2, y2))
                            x1, y1 = line.geometry[0]
                            x2, y2 = line.geometry[1]

                            lines.append({
                                'text': line_text,
                                'confidence': line_confidence,
                                'x1': x1,
                                'y1': y1,
                                'x2': x2,
                                'y2': y2
                            })

            logger.info(f"Extracted {len(lines)} lines from DocTR")

            return lines

        except Exception as e:
            logger.error(f"OCR text extraction with geometry failed: {str(e)}")
            raise ValueError(f"Text extraction with geometry failed: {e}")

    @staticmethod
    async def extract_text_with_geometry_enhanced(
        file: bytes, is_pdf: bool = False, max_pages: int = DEFAULT_MAX_PAGES, document_type: Optional[DocumentType] = None
    ) -> list:
        """
        Extract text from image with enhanced geometry information including word-level details.
        This method includes per-word confidence scores which is essential for key injection.

        For PDFs: Uses intelligent routing - direct text extraction for native PDFs,
        OCR fallback for image-based PDFs.

        Args:
            file: Image or PDF file bytes
            is_pdf: Whether the file is a PDF
            max_pages: Maximum number of pages to process
            document_type: Optional document type to force OCR for specific document types (e.g., bank statements)

        Returns:
            List of dicts with keys: text, x1, y1, x2, y2, confidence, word_details
        """
        logger.info(f"Extracting enhanced text with geometry from {'PDF' if is_pdf else 'image'}")

        try:
            # Intelligent PDF routing
            if is_pdf:
                return await DocumentTextExtractor._extract_pdf_intelligent(file, max_pages, document_type)
            # Convert to PNG with EXIF auto-rotation (no manual rotation)
            img_bytes = await ImageProcessor.convert_to_png(file, max_pages=max_pages)
            DocumentFile = get_document_file()
            doc = DocumentFile.from_images(img_bytes[:max_pages])

            # Run OCR
            model = get_doctr_model()
            result = model(doc)

            # Extract text with enhanced geometry and word details
            lines = []
            for page in result.pages:
                for block in page.blocks:
                    for line in block.lines:
                        # Get all words in the line with their details
                        line_words = []
                        word_details = []
                        total_confidence = 0
                        word_count = 0

                        for word in line.words:
                            line_words.append(word.value)

                            # Get word-level confidence and geometry
                            word_conf = getattr(word, 'confidence', 0.9)
                            total_confidence += word_conf
                            word_count += 1

                            # Word geometry
                            word_x1, word_y1 = word.geometry[0]
                            word_x2, word_y2 = word.geometry[1]

                            word_details.append({
                                'text': word.value,
                                'confidence': word_conf,
                                'x1': word_x1,
                                'y1': word_y1,
                                'x2': word_x2,
                                'y2': word_y2
                            })

                        if line_words:
                            line_text = " ".join(line_words)
                            # Get line geometry (normalized coordinates 0-1)
                            x1, y1 = line.geometry[0]
                            x2, y2 = line.geometry[1]

                            # Calculate average line confidence
                            avg_confidence = total_confidence / word_count if word_count > 0 else 0.9

                            lines.append({
                                'text': line_text,
                                'x1': x1,
                                'y1': y1,
                                'x2': x2,
                                'y2': y2,
                                'confidence': avg_confidence,
                                'word_details': word_details
                            })

            logger.info(f"Extracted {len(lines)} enhanced lines from DocTR")

            # Log detailed extraction results for debugging
            logger.info("=== DOCTR EXTRACTION RESULTS ===")
            for i, line in enumerate(lines[:10]):  # Show first 10 lines to avoid spamming logs
                logger.info(f"Line {i+1}: '{line['text'][:50]}{'...' if len(line['text']) > 50 else ''}' "
                           f"(confidence: {line['confidence']:.3f}, "
                           f"geometry: {line['x1']:.3f},{line['y1']:.3f},{line['x2']:.3f},{line['y2']:.3f})")

            if len(lines) > 10:
                logger.info(f"... and {len(lines) - 10} more lines")

            # Show total text length and sample for validation
            total_text = " ".join([line['text'] for line in lines])
            logger.info(f"Total extracted text length: {len(total_text)} characters")
            logger.info(f"Sample text: {total_text[:200]}{'...' if len(total_text) > 200 else ''}")
            logger.info("=== END DOCTR EXTRACTION RESULTS ===")

            return lines

        except Exception as e:
            logger.error(f"Enhanced OCR text extraction with geometry failed: {str(e)}")
            raise ValueError(f"Enhanced text extraction with geometry failed: {e}")

    @staticmethod
    async def _extract_pdf_intelligent(file: bytes, max_pages: int, document_type: Optional[DocumentType] = None) -> list:
        """
        Extract text from PDF using intelligent routing:
        1. Try direct text extraction (for native PDFs)
        2. Fall back to OCR (for image-based PDFs)

        Handles landscape orientation by transforming coordinates.

        Args:
            file: PDF file bytes
            max_pages: Maximum pages to process
            document_type: Optional document type to force OCR for specific document types

        Returns:
            List of dicts with text, geometry, confidence, word_details
        """
        # Save PDF to temporary file for analysis
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(file)
            temp_pdf_path = temp_file.name

        try:
            # Detect orientation BEFORE processing
            orientation_data = pdf_analyzer.detect_orientation(temp_pdf_path)
            orientation = OrientationInfo(**orientation_data)

            logger.info(f"PDF orientation detected: landscape={orientation.is_landscape}, "
                       f"rotation={orientation.rotation}, aspect_ratio={orientation.aspect_ratio:.2f}")

            # Analyze PDF content type
            pdf_analysis = pdf_analyzer.analyze_pdf(temp_pdf_path)

            logger.info(f"PDF Analysis: {pdf_analysis['content_type']} "
                       f"(recommended: {pdf_analysis['recommended_extraction']})")
            logger.info(f"Document type parameter: {document_type.value if document_type else 'None'}")

            # ============================================================
            # BANK STATEMENT: Validate PDF type and use direct extraction only
            # ============================================================
            # DISABLED: PDF type validation was incorrectly detecting text-based PDFs as image-based
            # TODO: Fix pdf_analyzer content type detection before re-enabling
            if document_type == DocumentType.BANK_STATEMENT:
                logger.info("Bank statement detected - PDF type validation disabled, using direct extraction")
                # if pdf_analysis['content_type'] != PDFContentType.TEXT_BASED:
                #     error_msg = (
                #         "Bank statement must be a text-based PDF with extractable text. "
                #         "Image-based or scanned PDFs are not accepted. "
                #         f"Detected type: {pdf_analysis['content_type']}"
                #     )
                #     logger.error(f"Bank statement validation failed: {error_msg}")
                #     raise ValueError(error_msg)
                # logger.info("Bank statement: validated as text-based PDF, using direct extraction only")

            # Try direct extraction if recommended
            if pdf_analysis['recommended_extraction'] in ['direct', 'direct_with_ocr_fallback']:
                try:
                    logger.info("=== ATTEMPTING DIRECT PDF TEXT EXTRACTION ===")
                    geometry_data = await DocumentTextExtractor._extract_geometry_from_direct_pdf(
                        temp_pdf_path, max_pages
                    )
                    logger.info(f"✅ DIRECT EXTRACTION SUCCESSFUL: {len(geometry_data)} elements")

                    # Add extraction_method metadata as a special marker element
                    # This allows downstream code to verify the extraction method
                    geometry_data.insert(0, {
                        'text': '__EXTRACTION_METADATA__',
                        'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0,
                        'confidence': 1.0,
                        'word_details': [],
                        'extraction_method': 'direct',
                        'orientation': 'landscape' if orientation.is_landscape else 'portrait'
                    })

                    # Add word_details for compatibility (direct extraction has no word-level details)
                    for element in geometry_data[1:]:  # Skip the metadata element
                        element['word_details'] = [{
                            'text': element['text'],
                            'confidence': 1.0,
                            'x1': element['x1'],
                            'y1': element['y1'],
                            'x2': element['x2'],
                            'y2': element['y2']
                        }]

                    return geometry_data

                except Exception as e:
                    logger.warning(f"❌ DIRECT EXTRACTION FAILED: {str(e)}")
                    # For bank statements, reject if direct extraction fails
                    if document_type == DocumentType.BANK_STATEMENT:
                        error_msg = (
                            "Bank statement direct extraction failed. "
                            "This usually indicates the PDF is not a valid text-based bank statement. "
                            "Please ensure you are uploading a digital bank statement PDF "
                            "(not a scan or screenshot)."
                        )
                        logger.error(f"Bank statement extraction failed: {error_msg}")
                        raise ValueError(error_msg)
                    # For other documents, fall back to OCR
                    logger.info("Falling back to OCR for non-bank-statement document")
            else:
                logger.info(f"Non-bank-statement document with '{pdf_analysis['recommended_extraction']}' recommendation")

            # Fall back to OCR with improved quality
            logger.info("=== FALLING BACK TO OCR TEXT EXTRACTION ===")

            # Convert to PNG with improved quality
            img_bytes = await ImageProcessor.convert_to_png(file, max_size=1600, max_pages=max_pages)
            DocumentFile = get_document_file()
            doc = DocumentFile.from_images(img_bytes[:max_pages])

            # Run OCR
            model = get_doctr_model()
            result = model(doc)

            # Extract text with enhanced geometry
            lines = []
            for page in result.pages:
                for block in page.blocks:
                    for line in block.lines:
                        line_words = []
                        word_details = []
                        total_confidence = 0
                        word_count = 0

                        for word in line.words:
                            line_words.append(word.value)
                            word_conf = getattr(word, 'confidence', 0.9)
                            total_confidence += word_conf
                            word_count += 1

                            word_x1, word_y1 = word.geometry[0]
                            word_x2, word_y2 = word.geometry[1]

                            word_details.append({
                                'text': word.value,
                                'confidence': word_conf,
                                'x1': word_x1,
                                'y1': word_y1,
                                'x2': word_x2,
                                'y2': word_y2
                            })

                        if line_words:
                            line_text = " ".join(line_words)
                            x1, y1 = line.geometry[0]
                            x2, y2 = line.geometry[1]
                            avg_confidence = total_confidence / word_count if word_count > 0 else 0.9

                            lines.append({
                                'text': line_text,
                                'x1': x1,
                                'y1': y1,
                                'x2': x2,
                                'y2': y2,
                                'confidence': avg_confidence,
                                'word_details': word_details,
                                'orientation': 'landscape' if orientation.is_landscape else 'portrait'
                            })

            # Apply coordinate transformation if landscape
            if orientation.is_landscape:
                logger.info("Applying coordinate transformation for landscape orientation")
                lines = CoordinateTransformer.transform_text_blocks(lines, orientation)

            # Add extraction_method metadata as a special marker element
            # This allows downstream code to verify the extraction method
            lines.insert(0, {
                'text': '__EXTRACTION_METADATA__',
                'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0,
                'confidence': 1.0,
                'word_details': [],
                'extraction_method': 'ocr',
                'orientation': 'landscape' if orientation.is_landscape else 'portrait'
            })

            logger.info(f"✅ OCR EXTRACTION COMPLETED: {len(lines)} elements")
            return lines

        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_pdf_path)
            except Exception:
                pass

    @staticmethod
    async def extract_text_with_geometry_and_confidence(
        file: bytes, is_pdf: bool = False, max_pages: int = DEFAULT_MAX_PAGES
    ) -> Tuple[List[Dict[str, Any]], OCRLinesContainer]:
        """
        Extract text with geometry AND preserve OCR lines for confidence lookup.

        Args:
            file: Image or PDF file bytes
            is_pdf: Whether the file is a PDF
            max_pages: Maximum number of pages to process

        Returns:
            Tuple of (geometry_data, ocr_lines_container)
        """
        logger.info(f"Extracting text with geometry and confidence preservation from {'PDF' if is_pdf else 'image'}")

        try:
            # Use existing extraction logic
            geometry_data = await DocumentTextExtractor.extract_text_with_geometry_enhanced(file, is_pdf, max_pages)

            # Extract line-level OCR data from geometry_data
            ocr_lines = []
            for line_data in geometry_data:
                ocr_lines.append({
                    'text': line_data['text'],
                    'confidence': line_data['confidence']  # Already calculated by DocTR
                })

            # Create container for storage
            document_hash = str(hash(file))
            ocr_container = OCRLinesContainer(lines=ocr_lines, document_hash=document_hash)

            logger.info(f"Created OCR lines container with {len(ocr_lines)} lines for confidence lookup")

            return geometry_data, ocr_container

        except Exception as e:
            logger.error(f"Text extraction with geometry and confidence preservation failed: {str(e)}")
            raise ValueError(f"Text extraction with geometry and confidence preservation failed: {e}")

    @staticmethod
    async def extract_with_key_injection(file: bytes, document_type: DocumentType,
                                       is_pdf: bool = False, max_pages: int = DEFAULT_MAX_PAGES,
                                       cross_document_data: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
        """
        Extract text with key injection enhancement using intelligent PDF processing.

        Args:
            file: Image or PDF file bytes
            document_type: Type of document being processed
            is_pdf: Whether the file is a PDF
            max_pages: Maximum number of pages to process
            cross_document_data: Values from other documents for cross-validation

        Returns:
            Dict with keys: geometry_data, key_injection_result, summary
        """
        logger.info(f"Extracting text with key injection for {document_type.value}")

        try:
            if is_pdf:
                # Use intelligent PDF processing
                return await DocumentTextExtractor._extract_with_key_injection_intelligent_pdf(
                    file, document_type, max_pages, cross_document_data
                )
            else:
                # Use regular OCR processing for images
                return await DocumentTextExtractor._extract_with_key_injection_ocr(
                    file, document_type, max_pages, cross_document_data
                )

        except Exception as e:
            logger.error(f"Text extraction with key injection failed: {str(e)}")
            raise ValueError(f"Text extraction with key injection failed: {e}")

    @staticmethod
    async def _extract_with_key_injection_intelligent_pdf(file: bytes, document_type: DocumentType,
                                                        max_pages: int,
                                                        cross_document_data: Optional[Dict[str, List[str]]]) -> Dict[str, Any]:
        """
        Extract text with key injection for PDFs using intelligent routing.
        Handles landscape orientation by transforming coordinates.

        Args:
            file: PDF file bytes
            document_type: Type of document being processed
            max_pages: Maximum number of pages to process
            cross_document_data: Values from other documents for cross-validation

        Returns:
            Dict with keys: geometry_data, key_injection_result, summary
        """
        logger.info("Starting intelligent PDF processing with key injection")

        # Save PDF to temporary file for analysis
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(file)
            temp_pdf_path = temp_file.name

        try:
            # Detect orientation for logging
            orientation_data = pdf_analyzer.detect_orientation(temp_pdf_path)
            orientation = OrientationInfo(**orientation_data)

            logger.info(f"PDF orientation detected: landscape={orientation.is_landscape}, "
                       f"rotation={orientation.rotation}, aspect_ratio={orientation.aspect_ratio:.2f}")

            # Analyze PDF content type
            pdf_analysis = pdf_analyzer.analyze_pdf(temp_pdf_path)

            logger.info(f"PDF Analysis: {pdf_analysis['content_type']} "
                       f"(recommended: {pdf_analysis['recommended_extraction']})")

            # ============================================================
            # BANK STATEMENT: Support both text-based and image-based PDFs
            # ============================================================
            if document_type == DocumentType.BANK_STATEMENT:
                logger.info(f"Bank statement: {pdf_analysis['content_type']} PDF - will use "
                           f"{'direct extraction' if pdf_analysis['content_type'] == PDFContentType.TEXT_BASED else 'OCR'}")

            geometry_data = None
            extraction_method = None

            # Route to appropriate extraction method
            if pdf_analysis['recommended_extraction'] in ['direct', 'direct_with_ocr_fallback']:
                try:
                    # Try direct text extraction first
                    logger.info("=== ATTEMPTING DIRECT PDF TEXT EXTRACTION ===")
                    geometry_data = await DocumentTextExtractor._extract_geometry_from_direct_pdf(
                        temp_pdf_path, max_pages
                    )
                    extraction_method = 'direct'
                    logger.info(f"✅ DIRECT EXTRACTION SUCCESSFUL: {len(geometry_data)} elements")
                    logger.info(f"Sample confidence values: {[g.get('confidence', 'N/A') for g in geometry_data[:5]]}")

                except Exception as e:
                    logger.warning(f"❌ DIRECT EXTRACTION FAILED: {str(e)}, falling back to OCR")
                    geometry_data = None

            # Fall back to OCR if direct extraction failed or not recommended
            # Skip OCR fallback for bank statements - they must be text-based
            if geometry_data is None:
                if document_type == DocumentType.BANK_STATEMENT:
                    error_msg = (
                        "Bank statement direct extraction failed. "
                        "This usually indicates the PDF is not a valid text-based bank statement. "
                        "Please ensure you are uploading a digital bank statement PDF "
                        "(not a scan or screenshot)."
                    )
                    logger.error(f"Bank statement extraction failed: {error_msg}")
                    raise ValueError(error_msg)
                else:
                    logger.info("=== FALLING BACK TO OCR TEXT EXTRACTION ===")
                    geometry_data = await DocumentTextExtractor.extract_text_with_geometry_enhanced(
                        file, is_pdf=True, max_pages=max_pages
                    )
                    extraction_method = 'ocr'
                    logger.info(f"✅ OCR EXTRACTION COMPLETED: {len(geometry_data)} elements")
                    logger.info(f"Sample confidence values: {[g.get('confidence', 'N/A') for g in geometry_data[:5]]}")

            # Apply coordinate transformation if landscape and not already applied
            if orientation.is_landscape:
                # Check if orientation marker is already present (from _extract_geometry_from_direct_pdf)
                needs_transform = not any('orientation' in g for g in geometry_data)
                if needs_transform:
                    logger.info("Applying coordinate transformation for landscape orientation")
                    geometry_data = CoordinateTransformer.transform_text_blocks(geometry_data, orientation)

            # Process with key injection
            key_injection_result = key_injection_manager.process_document_with_key_injection(
                ocr_geometry_data=geometry_data,
                document_type=document_type,
                cross_document_data=cross_document_data
            )

            # Generate summary
            summary = {
                'total_lines_extracted': len(geometry_data),
                'key_injection_successful': key_injection_result.success,
                'keys_detected': len(key_injection_result.detected_keys),
                'keys_with_values': sum(1 for key in key_injection_result.detected_keys if key.value_candidate),
                'high_confidence_extractions': len(
                    key_injection_manager.get_high_confidence_extractions(key_injection_result)
                ),
                'pdf_analysis': pdf_analysis,
                'extraction_method': extraction_method,
                'is_direct_extraction': extraction_method == 'direct',
                'is_landscape': orientation.is_landscape
            }

            if not key_injection_result.success:
                logger.error(f"Key injection failed: {key_injection_result.error_message}")
                summary['error'] = key_injection_result.error_message

            return {
                'geometry_data': geometry_data,
                'key_injection_result': key_injection_result,
                'summary': summary
            }

        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_pdf_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file: {str(e)}")

    @staticmethod
    async def _extract_with_key_injection_ocr(file: bytes, document_type: DocumentType,
                                           max_pages: int,
                                           cross_document_data: Optional[Dict[str, List[str]]]) -> Dict[str, Any]:
        """
        Extract text with key injection for images using OCR.

        Args:
            file: Image file bytes
            document_type: Type of document being processed
            max_pages: Maximum number of pages to process
            cross_document_data: Values from other documents for cross-validation

        Returns:
            Dict with keys: geometry_data, key_injection_result, summary
        """
        logger.info("Using OCR text extraction for image")

        # Get enhanced geometry data
        geometry_data = await DocumentTextExtractor.extract_text_with_geometry_enhanced(
            file, is_pdf=False, max_pages=max_pages
        )

        # Process with key injection
        key_injection_result = key_injection_manager.process_document_with_key_injection(
            ocr_geometry_data=geometry_data,
            document_type=document_type,
            cross_document_data=cross_document_data
        )

        # Generate summary
        summary = {
            'total_lines_extracted': len(geometry_data),
            'key_injection_successful': key_injection_result.success,
            'keys_detected': len(key_injection_result.detected_keys),
            'keys_with_values': sum(1 for key in key_injection_result.detected_keys if key.value_candidate),
            'high_confidence_extractions': len(
                key_injection_manager.get_high_confidence_extractions(key_injection_result)
            ),
            'extraction_method': 'ocr'
        }

        if not key_injection_result.success:
            logger.error(f"Key injection failed: {key_injection_result.error_message}")
            summary['error'] = key_injection_result.error_message

        return {
            'geometry_data': geometry_data,
            'key_injection_result': key_injection_result,
            'summary': summary
        }

    @staticmethod
    async def _extract_geometry_from_direct_pdf(pdf_path: str, max_pages: int) -> List[Dict[str, Any]]:
        """
        Extract geometry data directly from PDF without OCR.
        Handles landscape orientation by transforming coordinates.

        Args:
            pdf_path: Path to PDF file
            max_pages: Maximum number of pages to process

        Returns:
            List of geometry data dictionaries compatible with key injection
        """
        logger.info(f"Extracting geometry data directly from PDF: {pdf_path}")

        try:
            # Detect orientation
            orientation_info = pdf_analyzer.detect_orientation(pdf_path)
            logger.info(f"PDF orientation: landscape={orientation_info['is_landscape']}, "
                        f"rotation={orientation_info['rotation']}, "
                        f"aspect_ratio={orientation_info['aspect_ratio']:.2f}")

            # Create OrientationInfo object
            orientation = OrientationInfo(**orientation_info)

            # Extract text elements with spatial data (limit to max_pages for efficiency)
            text_elements = pdf_text_extractor.extract_text_with_spatial_data(pdf_path, max_pages)

            # Convert to geometry data format expected by key injection
            geometry_data = []

            for element in text_elements:
                if element.text.strip():  # Skip empty text
                    # Get raw geometry (before normalization)
                    raw_geometry = element.geometry

                    # Transform geometry if landscape
                    if orientation.is_landscape:
                        transformed_geometry = CoordinateTransformer.transform_geometry_dict(
                            raw_geometry, orientation
                        )
                    else:
                        transformed_geometry = raw_geometry

                    # Use actual page dimensions for normalization
                    page_width = orientation.width
                    page_height = orientation.height

                    geometry_dict = {
                        'text': element.text,
                        'x1': transformed_geometry['x1'] / page_width,
                        'y1': transformed_geometry['y1'] / page_height,
                        'x2': transformed_geometry['x2'] / page_width,
                        'y2': transformed_geometry['y2'] / page_height,
                        'confidence': 1.0,  # 100% confidence for direct PDF extraction
                        # Additional metadata
                        'font': transformed_geometry.get('font', ''),
                        'font_size': transformed_geometry.get('font_size', 12),
                        'page_num': transformed_geometry.get('page_num', 0),
                        'orientation': 'landscape' if orientation.is_landscape else 'portrait'
                    }

                    geometry_data.append(geometry_dict)

            logger.info(f"Direct PDF extraction: {len(geometry_data)} text elements with geometry")

            # Log sample extraction results
            logger.info("=== DIRECT PDF EXTRACTION RESULTS ===")
            for i, element in enumerate(geometry_data[:10]):  # Show first 10 elements
                logger.info(f"Element {i+1}: '{element['text'][:50]}{'...' if len(element['text']) > 50 else ''}' "
                           f"(confidence: {element['confidence']:.3f}, "
                           f"geometry: {element['x1']:.3f},{element['y1']:.3f},{element['x2']:.3f},{element['y2']:.3f})")

            if len(geometry_data) > 10:
                logger.info(f"... and {len(geometry_data) - 10} more elements")

            logger.info("=== END DIRECT PDF EXTRACTION RESULTS ===")

            return geometry_data

        except Exception as e:
            logger.error(f"Direct PDF geometry extraction failed: {str(e)}")
            raise ValueError(f"Direct PDF geometry extraction failed: {e}")

    @staticmethod
    async def extract_simple_key_value_pairs(file: bytes, document_type: DocumentType,
                                          is_pdf: bool = False, max_pages: int = DEFAULT_MAX_PAGES) -> List[Dict[str, str]]:
        """
        Simple key-value extraction without geometry data.
        Useful for quick processing or when geometry is not available.

        Args:
            file: Image or PDF file bytes
            document_type: Type of document being processed
            is_pdf: Whether the file is a PDF
            max_pages: Maximum number of pages to process

        Returns:
            List of dicts with extracted key-value pairs
        """
        logger.info(f"Extracting simple key-value pairs for {document_type.value}")

        try:
            # Get basic text extraction
            text = await DocumentTextExtractor.extract_text(file, is_pdf, max_pages)
            text_lines = text.split('\n')

            # Extract key-value pairs using key injection
            detected_keys = key_injection_manager.extract_simple_key_value_pairs(
                text_lines, document_type
            )

            # Convert to simple key-value format
            key_value_pairs = []
            for key in detected_keys:
                if key.value_candidate:
                    key_value_pairs.append({
                        'key': key.key_name,
                        'value': key.value_candidate,
                        'confidence': key.confidence
                    })

            logger.info(f"Extracted {len(key_value_pairs)} key-value pairs")

            return key_value_pairs

        except Exception as e:
            logger.error(f"Simple key-value extraction failed: {str(e)}")
            raise ValueError(f"Simple key-value extraction failed: {e}")

    @staticmethod
    async def _extract_from_pdf(file: bytes) -> Optional[str]:
        try:
            pdf_reader = PdfReader(BytesIO(file))
            if not pdf_reader.pages:
                raise ValueError("Empty PDF")

            if len(pdf_reader.pages) > 10:
                logger.warning(
                    f"PDF has {len(pdf_reader.pages)} pages, processing first page only")

            return pdf_reader.pages[0].extract_text().strip() or None
        except Exception as e:
            logger.error(f"PDF text extraction failed: {str(e)}")
            return None

    @staticmethod
    async def _extract_with_ocr(file: bytes, max_pages: int) -> str:
        """
        Extract text from image using OCR.
        Uses EXIF auto-rotation only (no manual rotation attempts).
        """
        try:
            # Convert to PNG with EXIF auto-rotation (no manual rotation)
            img_bytes = await ImageProcessor.convert_to_png(file, max_pages=max_pages)
            DocumentFile = get_document_file()
            doc = DocumentFile.from_images(img_bytes[:max_pages])

            # Run OCR - model is initialized with export_as_straight_boxes=True
            # to handle rotated text and export text boxes in reading order
            model = get_doctr_model()
            logger.info(f"DocTR model created: {type(model)}")
            result = model(doc)
            logger.info(f"DocTR result type: {type(result)}")

            # Log DocTR output structure
            logger.info("DocTR output structure:")
            for page_idx, page in enumerate(result.pages):
                logger.info(f"  Page {page_idx}: {len(page.blocks)} blocks")
                if len(page.blocks) == 0:
                    logger.warning(f"  Page {page_idx}: No blocks detected - possible OCR failure")

            # Extract text preserving block structure
            # Each block represents a logical grouping of text
            blocks = []
            total_blocks = 0
            total_lines = 0
            total_words = 0

            for page_idx, page in enumerate(result.pages):
                logger.debug(f"Processing page {page_idx} with {len(page.blocks)} blocks")
                for block_idx, block in enumerate(page.blocks):
                    total_blocks += 1
                    # Get all words in the block
                    block_words = []
                    # Confidence threshold for filtering OCR noise
                    # Words below this threshold are likely misreads and should be excluded
                    OCR_CONFIDENCE_THRESHOLD = 0.6

                    for line in block.lines:
                        total_lines += 1
                        for word in line.words:
                            total_words += 1
                            confidence = getattr(word, 'confidence', 1.0)
                            # Skip low-confidence words that are likely OCR noise
                            if confidence < OCR_CONFIDENCE_THRESHOLD:
                                logger.debug(f"    Filtered low-confidence word: '{word.value}' (confidence: {confidence:.3f})")
                                continue
                            block_words.append(word.value)
                            logger.debug(f"    Word: '{word.value}' (confidence: {confidence:.3f})")

                    if block_words:
                        # Join words in block with space, add block as a line
                        block_text = " ".join(block_words)
                        blocks.append(block_text)
                        logger.debug(f"  Block {block_idx}: '{block_text}'")

            logger.info(f"OCR stats: {total_blocks} blocks, {total_lines} lines, {total_words} words detected")

            # Join blocks with newlines to preserve block structure
            extracted_text = "\n".join(blocks).strip()

            logger.info(f"OCR extracted {len(extracted_text)} characters")

            return extracted_text
        except Exception as e:
            logger.error(f"OCR text extraction failed: {str(e)}")
            raise ValueError(f"Text extraction failed: {e}")


class ImageProcessor:
    @staticmethod
    async def convert_to_png(file: bytes, max_size: int = DEFAULT_MAX_SIZE, max_pages: int = None) -> List[bytes]:
        """
        Convert image to PNG format with EXIF auto-rotation.

        Args:
            file: Image file bytes
            max_size: Maximum dimension size
            max_pages: Maximum number of pages to process (for PDFs only)

        Returns:
            List of PNG image bytes
        """
        try:

            try:
                img = Image.open(io.BytesIO(file)).convert("RGB")
                original_width, original_height = img.size
                logger.info(f"Original image dimensions: {original_width}x{original_height}")

                # Auto-rotate image based on EXIF orientation
                try:
                    img = ImageOps.exif_transpose(img)
                    logger.info("Applied EXIF auto-rotation")
                    # Check dimensions after rotation (they might swap)
                    rotated_width, rotated_height = img.size
                    if (rotated_width, rotated_height) != (original_width, original_height):
                        logger.info(f"Image rotated: dimensions changed to {rotated_width}x{rotated_height}")
                except Exception as e:
                    logger.debug(f"No EXIF rotation needed: {e}")

                # Check if resizing will reduce image quality
                if max(original_width, original_height) > max_size:
                    logger.info(f"Image will be resized from {max(original_width, original_height)}px to {max_size}px")
                else:
                    logger.info(f"Image size {max(original_width, original_height)}px is within limit {max_size}px - no resizing needed")

                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                final_width, final_height = img.size
                logger.info(f"Final image dimensions after processing: {final_width}x{final_height}")

                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG', optimize=True)
                return [img_byte_arr.getvalue()]
            except UnidentifiedImageError:
                pass

            # If not an image, try to process as PDF
            return await ImageProcessor._convert_pdf_to_png(file, max_size, max_pages)

        except Exception as e:
            logger.error(f"Image conversion failed: {str(e)}")
            raise ValueError(f"Failed to convert file to images: {e}")

    @staticmethod
    async def _convert_pdf_to_png(file: bytes, max_size: int, max_pages: int = None) -> List[bytes]:
        """
        Convert PDF pages to PNG images using PyMuPDF for proper rendering.

        Args:
            file: PDF file bytes
            max_size: Maximum dimension for the output images
            max_pages: Maximum number of pages to process (None for all pages)

        Returns:
            List of PNG image bytes
        """
        try:
            # Open PDF with PyMuPDF
            pdf_document = fitz.open(stream=file, filetype="pdf")

            if pdf_document.page_count == 0:
                raise ValueError("Empty PDF")

            # Use provided max_pages, or default to processing all pages if ≤10
            if max_pages is None:
                if pdf_document.page_count > 10:
                    logger.warning(
                        f"PDF has {pdf_document.page_count} pages, processing first page only")
                    max_pages = 1
                else:
                    max_pages = pdf_document.page_count
            else:
                logger.info(f"Limiting PDF processing to {max_pages} pages as requested")

            rendered_images = []

            for page_num in range(min(max_pages, pdf_document.page_count)):
                try:
                    # Get the page
                    page = pdf_document[page_num]

                    # Calculate zoom factor for desired resolution
                    # Higher zoom = better OCR quality but slower processing
                    zoom = 4.0  # Increased from 2.0 for better OCR quality

                    # Render page to pixmap
                    mat = fitz.Matrix(zoom, zoom)  # Zoom factor
                    pix = page.get_pixmap(matrix=mat)

                    # Convert to PIL Image
                    img_data = pix.tobytes("png")

                    # Convert to PIL Image for potential resizing
                    img = Image.open(BytesIO(img_data))

                    # Resize if needed while maintaining aspect ratio
                    if max(img.size) > max_size:
                        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

                    # Convert back to bytes
                    img_byte_arr = BytesIO()
                    img.save(img_byte_arr, format='PNG', quality=95)
                    rendered_images.append(img_byte_arr.getvalue())

                    logger.info(f"Successfully rendered page {page_num + 1} to PNG "
                              f"(size: {img.size[0]}x{img.size[1]})")

                except Exception as e:
                    logger.error(f"Failed to render page {page_num + 1}: {str(e)}")
                    continue

            pdf_document.close()

            if not rendered_images:
                raise ValueError("Failed to render any pages from PDF")

            logger.info(f"Successfully converted PDF to {len(rendered_images)} PNG images")
            return rendered_images

        except Exception as e:
            logger.error(f"PDF to image conversion failed: {str(e)}")
            raise ValueError(f"Failed to convert PDF to images: {e}")

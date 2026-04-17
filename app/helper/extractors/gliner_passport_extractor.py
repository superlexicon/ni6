"""
GLiNER2-based Passport Extractor

Uses GLiNER2 zero-shot NER with schema-based extraction for passport fields.
Falls back to logic-based extraction if confidence is low.
"""

from typing import Optional, Dict, Any
from app.schemas.passport_schema import PassportData
from app.core.gliner_ner_model import get_gliner_ner_model
from app.helper.doctr.document_text_extractor import DocumentTextExtractor
from app.core.logger import get_logger
from app.utils.date_extractor import extract_and_remove_dates
from app.utils.date_parser import format_date_for_passport


class GLiNERPassportExtractor:
    """Extract passport fields using GLiNER2 zero-shot NER with schema-based extraction."""

    def __init__(self):
        self.logger = get_logger()
        self.text_extractor = DocumentTextExtractor()
        self.gliner_model = None

    async def extract(self, content: bytes, is_pdf: bool = False) -> PassportData:
        """
        Extract passport fields using GLiNER2 schema-based extraction.

        Args:
            content: Raw file bytes
            is_pdf: True if input is a PDF file

        Returns:
            PassportData with extracted fields and confidence scores
        """
        try:
            self.logger.info("=" * 80)
            self.logger.info("GLiNER2 PASSPORT EXTRACTION")
            self.logger.info("=" * 80)

            # Step 1: Extract text with geometry
            self.logger.info("Step 1: Extracting text with geometry")
            text_blocks = await self.text_extractor.extract_text_with_geometry_enhanced(
                content, is_pdf=is_pdf, max_pages=1
            )
            self.logger.info(f"  Extracted {len(text_blocks)} text lines")

            # Step 2: Combine text for GLiNER
            all_text = "\n".join([block.get('text', '') for block in text_blocks])
            self.logger.info(f"  Total text length: {len(all_text)} characters")

            # Step 3: Get GLiNER model singleton
            if self.gliner_model is None:
                self.gliner_model = get_gliner_ner_model()

            # Step 4: Extract using GLiNER2 schema
            self.logger.info("Step 2: Extracting entities with GLiNER2 schema")
            gliner_result = await self.gliner_model.extract_passport_with_schema_async(all_text)
            self.logger.info(f"  GLiNER extracted {len([v for v in gliner_result.values() if v])} entities")

            # Step 5: Build result
            self.logger.info("Step 3: Building result")
            result = PassportData()
            confidence_scores = {}

            for field_name, entity_data in gliner_result.items():
                value = entity_data.get('value', '').strip() if entity_data else ''
                confidence = entity_data.get('confidence', 0.0) * 100 if entity_data else 0.0

                if value:
                    setattr(result, field_name, value)
                    confidence_scores[field_name] = {
                        'overall_confidence': confidence / 100,
                        'sources': ['gliner']
                    }

            # POST-PROCESSOR: Apply logic-based date extraction for robust chronological assignment
            self.logger.info("Step 4: Applying logic-based date extraction for chronological sorting")
            all_dates_with_text = []

            # Extract dates from all text blocks
            for block in text_blocks:
                block_text = block.get('text', '')
                if block_text:
                    dates_from_block, _ = extract_and_remove_dates(block_text)
                    all_dates_with_text.extend(dates_from_block)

            # Sort dates chronologically (earliest to latest)
            all_dates_with_text.sort(key=lambda x: x[0])

            # Assign dates based on chronological order
            if len(all_dates_with_text) >= 1:
                dt_obj, original_text = all_dates_with_text[0]
                result.date_of_birth = format_date_for_passport(dt_obj)
                self.logger.info(f"  DOB (earliest): {result.date_of_birth} (from: '{original_text}')")

                if 'date_of_birth' not in confidence_scores:
                    confidence_scores['date_of_birth'] = {
                        'overall_confidence': 0.75,
                        'sources': ['logic_based']
                    }

            if len(all_dates_with_text) >= 3:
                dt_obj_issue, original_text_issue = all_dates_with_text[1]
                dt_obj_expiry, original_text_expiry = all_dates_with_text[2]

                result.date_of_issue = format_date_for_passport(dt_obj_issue)
                result.date_of_expiry = format_date_for_passport(dt_obj_expiry)

                self.logger.info(f"  Issue (middle): {result.date_of_issue} (from: '{original_text_issue}')")
                self.logger.info(f"  Expiry (latest): {result.date_of_expiry} (from: '{original_text_expiry}')")

                if 'date_of_issue' not in confidence_scores:
                    confidence_scores['date_of_issue'] = {
                        'overall_confidence': 0.75,
                        'sources': ['logic_based']
                    }
                if 'date_of_expiry' not in confidence_scores:
                    confidence_scores['date_of_expiry'] = {
                        'overall_confidence': 0.75,
                        'sources': ['logic_based']
                    }
            elif len(all_dates_with_text) == 2:
                # Only 2 dates: assume DOB (already set) and Expiry
                dt_obj_expiry, original_text_expiry = all_dates_with_text[1]
                result.date_of_expiry = format_date_for_passport(dt_obj_expiry)
                self.logger.info(f"  Expiry (latest): {result.date_of_expiry} (from: '{original_text_expiry}')")

                if 'date_of_expiry' not in confidence_scores:
                    confidence_scores['date_of_expiry'] = {
                        'overall_confidence': 0.75,
                        'sources': ['logic_based']
                    }

            # Debug: Log which required fields are missing
            required_fields = ['passport_number', 'full_name', 'date_of_birth', 'nationality']
            present_fields = [f for f in required_fields if getattr(result, f)]
            missing_fields = [f for f in required_fields if not getattr(result, f)]

            if missing_fields:
                self.logger.warning(f"  Missing required fields: {missing_fields}")
                self.logger.debug(f"  GLiNER raw result keys: {list(gliner_result.keys())}")
                for field_name, entity_data in gliner_result.items():
                    value = entity_data.get('value', '').strip() if entity_data else ''
                    self.logger.debug(f"    {field_name}: '{value}' (conf: {entity_data.get('confidence', 0):.2f})")

            # Set confidence data
            result.field_confidences = confidence_scores
            result.overall_confidence = result.calculate_overall_confidence()
            result.extraction_source = 'gliner_ner'
            result.raw_data = all_text

            self.logger.info("=" * 80)
            self.logger.info("EXTRACTION COMPLETE")
            self.logger.info(f"  Overall confidence: {result.overall_confidence:.1f}%")
            self.logger.info(f"  Fields extracted: {len([v for v in [result.passport_number, result.full_name, result.date_of_birth, result.nationality] if v])}")
            self.logger.info("=" * 80)

            return result

        except Exception as e:
            self.logger.error(f"GLiNER2 passport extraction failed: {e}")
            # Return minimal result on error
            return PassportData(
                overall_confidence=0.0,
                extraction_source='gliner_ner'
            )

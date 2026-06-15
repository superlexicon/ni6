"""
Sequential ID Card Service - Handles ID card processing in sequential mode.

ID cards (PAN, national ID, driver's license) use GLiNER-based extraction
and are independent documents - they don't increment verification_state.

State is tracked via verification_state column in user_identity_index.
"""

from typing import Dict, Any, Optional, Tuple, List
from app.services.sequential_document_processor_base import DocumentProcessorBase
from app.helper.extractors.gliner_id_card_extractor import GLiNERIDCardExtractor
from app.core.key_injection import DocumentType
from app.services.extractors.qwen_uae_trc_extractor import QwenUAETrcExtractor
from app.services.extractors.qwen_pan_extractor import QwenPANExtractor
from app.services.extractors.qwen_singapore_nric_extractor import QwenSingaporeNricExtractor


class SequentialIDCardService(DocumentProcessorBase):
    """
    Service for handling ID card processing using GLiNER-based extraction.

    ID cards are independent documents that don't increment verification_state.
    They use GLiNER zero-shot NER for flexible field extraction.
    """

    # ============================================================
    # ABSTRACT METHOD IMPLEMENTATIONS
    # ============================================================

    def get_document_type(self) -> str:
        return "id_card"

    def get_required_fields(self) -> List[str]:
        # Core fields that should be present for most ID cards
        # Not all fields are required for all card types (e.g., some ID cards have no expiry)
        return ['full_name']

    def get_name_field(self) -> str:
        """ID cards use 'full_name' as the name field."""
        return 'full_name'

    def should_increment_state(self) -> bool:
        """
        ID cards are independent documents - no state change.

        Only passport increments state (1 -> 2 -> 3).
        ID cards can be submitted at any time without affecting flow.
        """
        return False

    def should_validate_photoholmes(self) -> bool:
        """
        ID cards should have PhotoHolmes forgery detection.

        Validates authenticity of the ID card document.
        """
        return True

    def get_photoholmes_document_type(self) -> str:
        """ID cards use 'id_card' for PhotoHolmes detection."""
        return "id_card"

    async def extract_fields_from_ocr(
        self, text_blocks: list, raw_text: str, image_bytes: bytes, is_pdf: bool
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Extract ID card fields using Qwen3-VL extractors (preferred) or fallback to GLiNER/Spatial extractors.

        Priority:
        1. UAE TRC -> QwenUAETrcExtractor (with fallback to SpatialUAETRCExtractor)
        2. PAN -> QwenPANExtractor (with fallback to SpatialPANExtractor)
        3. Singapore NRIC -> QwenSingaporeNricExtractor (no fallback)
        4. Other -> GLiNERIDCardExtractor (existing)

        Returns:
            (extracted_data, confidence_data)
        """
        # Detect document type from raw_text
        from app.helper.extractors.gliner_id_card_extractor import GLiNERIDCardExtractor
        temp_extractor = GLiNERIDCardExtractor()
        document_type = temp_extractor._detect_document_type(raw_text, text_blocks)

        # Use Qwen UAE TRC extractor for UAE TRC
        if document_type == 'UAE_TRC':
            self.logger.info("Using Qwen3-VL UAE TRC Extractor for UAE Tax Residency Certificate")
            try:
                extractor = QwenUAETrcExtractor()
                qwen_extracted_data, qwen_confidence_data = await extractor.extract_fields(image_bytes)

                # Convert Qwen format to service format
                # Qwen returns: {"field": {"value": ..., "confidence": ..., "source": ...}}
                # Service expects: flat dict + separate confidence dict

                extracted_data = {
                    "document_type": "UAE_TRC",
                    "issuing_country": "AE",
                }

                confidence_data = {}

                # Flatten the Qwen extracted data
                for field_name, field_data in qwen_extracted_data.items():
                    if isinstance(field_data, dict) and "value" in field_data:
                        value = field_data["value"]
                        confidence = field_data.get("confidence", 1.0)
                        source = field_data.get("source", "vision_llm")

                        # Add to extracted_data
                        if field_name == "certificate_number":
                            extracted_data["identification_number"] = value
                            confidence_data["identification_number"] = {
                                'overall_confidence': confidence,
                                'sources': [source]
                            }
                        else:
                            extracted_data[field_name] = value
                            confidence_data[field_name] = {
                                'overall_confidence': confidence,
                                'sources': [source]
                            }

                self.logger.info(
                    f"Qwen3-VL UAE TRC extraction completed: {len(extracted_data)} fields extracted"
                )

                return extracted_data, confidence_data

            except Exception as e:
                self.logger.warning(f"Qwen3-VL UAE TRC extraction failed: {e}. Falling back to SpatialUAETRCExtractor")
                # Fallback to original spatial extractor
                from app.helper.extractors.spatial_uae_trc_extractor import SpatialUAETRCExtractor
                extractor = SpatialUAETRCExtractor()
                result = await extractor.extract(image_bytes, is_pdf)

                extracted_data = {
                    "document_type": "UAE_TRC",
                    "issuing_country": "AE",
                    "full_name": result.full_name,
                    "identification_number": result.identification_number,
                    "raw_data": result.raw_data,
                    **result.field_values,
                }

                confidence_data = {}
                for field, confidence in result.confidence_scores.items():
                    conf_value = confidence / 100 if confidence > 1 else confidence
                    confidence_data[field] = {
                        'overall_confidence': conf_value,
                        'sources': ['spatial_extraction']
                    }

                self.logger.info(
                    f"Fallback Spatial UAE TRC extraction: document_type={result.document_type}, "
                    f"fields_extracted={len(result.field_values)}"
                )

                return extracted_data, confidence_data

        # Use Qwen PAN extractor for PAN cards
        elif document_type == 'PAN':
            self.logger.info("Using Qwen3-VL PAN Extractor for PAN card")
            try:
                extractor = QwenPANExtractor()
                qwen_extracted_data, qwen_confidence_data = await extractor.extract_fields(image_bytes)

                # Convert Qwen format to service format
                extracted_data = {}

                confidence_data = {}

                # Flatten the Qwen extracted data
                for field_name, field_data in qwen_extracted_data.items():
                    if isinstance(field_data, dict) and "value" in field_data:
                        value = field_data["value"]
                        confidence = field_data.get("confidence", 1.0)
                        source = field_data.get("source", "vision_llm")

                        # Add to extracted_data
                        if field_name == "pan_number":
                            extracted_data["identification_number"] = value
                            confidence_data["identification_number"] = {
                                'overall_confidence': confidence,
                                'sources': [source]
                            }
                        else:
                            extracted_data[field_name] = value
                            confidence_data[field_name] = {
                                'overall_confidence': confidence,
                                'sources': [source]
                            }

                self.logger.info(
                    f"Qwen3-VL PAN extraction completed: {len(extracted_data)} fields extracted"
                )

                return extracted_data, confidence_data

            except Exception as e:
                self.logger.warning(f"Qwen3-VL PAN extraction failed: {e}. Falling back to SpatialPANExtractor")
                # Fallback to original spatial extractor
                from app.helper.extractors.spatial_pan_extractor import SpatialPANExtractor
                extractor = SpatialPANExtractor()
                result = await extractor.extract(image_bytes, is_pdf)

                extracted_data = {
                    "full_name": result.full_name,
                    "date_of_birth": result.date_of_birth,
                    "identification_number": result.identification_number,
                    "raw_data": result.raw_data,
                    **result.field_values,
                }

                confidence_data = {}
                for field, confidence in result.confidence_scores.items():
                    conf_value = confidence / 100 if confidence > 1 else confidence
                    confidence_data[field] = {
                        'overall_confidence': conf_value,
                        'sources': ['spatial_extraction']
                    }

                self.logger.info(
                    f"Fallback Spatial PAN extraction: document_type={result.document_type}, "
                    f"fields_extracted={len(result.field_values)}"
                )

                return extracted_data, confidence_data

        # Use Qwen Singapore NRIC extractor for Singapore NRIC/FIN
        elif document_type == 'SINGAPORE_NRIC':
            self.logger.info("Using Qwen3-VL Singapore NRIC Extractor for Singapore NRIC/FIN")
            try:
                extractor = QwenSingaporeNricExtractor()
                qwen_extracted_data, qwen_confidence_data = await extractor.extract_fields(image_bytes)

                # Convert Qwen format to service format
                extracted_data = {
                    "document_type": "SINGAPORE_NRIC",
                    "issuing_country": "SG",
                }

                confidence_data = {}

                # Flatten the Qwen extracted data
                for field_name, field_data in qwen_extracted_data.items():
                    if isinstance(field_data, dict) and "value" in field_data:
                        value = field_data["value"]
                        confidence = field_data.get("confidence", 1.0)
                        source = field_data.get("source", "vision_llm")

                        # Add to extracted_data
                        if field_name == "nric_fin_number":
                            extracted_data["identification_number"] = value
                            confidence_data["identification_number"] = {
                                'overall_confidence': confidence,
                                'sources': [source]
                            }
                        elif field_name == "sex":
                            extracted_data["gender"] = value
                            confidence_data["gender"] = {
                                'overall_confidence': confidence,
                                'sources': [source]
                            }
                        else:
                            extracted_data[field_name] = value
                            confidence_data[field_name] = {
                                'overall_confidence': confidence,
                                'sources': [source]
                            }

                self.logger.info(
                    f"Qwen3-VL Singapore NRIC extraction completed: {len(extracted_data)} fields extracted"
                )

                return extracted_data, confidence_data

            except Exception as e:
                self.logger.warning(f"Qwen3-VL Singapore NRIC extraction failed: {e}. Falling back to GLiNERIDCardExtractor")
                # Fallback to GLiNER extractor for Singapore NRIC
                self.logger.info("Using GLiNER extractor for Singapore NRIC (fallback)")
                extractor = GLiNERIDCardExtractor()
                result = await extractor.extract(image_bytes, is_pdf)

                # Build extracted_data from IDCardData
                extracted_data = {
                    "document_type": "SINGAPORE_NRIC",
                    "issuing_country": "SG",
                    "full_name": result.full_name,
                    "date_of_birth": result.date_of_birth,
                    "gender": result.gender,
                    "identification_number": result.identification_number,
                    "raw_data": result.raw_data,
                    **result.field_values,
                }

                confidence_data = {}
                for field, confidence in result.confidence_scores.items():
                    conf_value = confidence / 100 if confidence > 1 else confidence
                    confidence_data[field] = {
                        'overall_confidence': conf_value,
                        'sources': ['gliner_ner']
                    }

                self.logger.info(
                    f"Fallback GLiNER Singapore NRIC extraction: document_type={result.document_type}, "
                    f"fields_extracted={len(result.field_values)}"
                )

                return extracted_data, confidence_data

        # Use GLiNER for other ID card types (existing code)
        self.logger.info("Using GLiNER extractor for non-Qwen-supported ID card")
        extractor = GLiNERIDCardExtractor()
        result = await extractor.extract(image_bytes, is_pdf)

        # Build extracted_data from IDCardData
        extracted_data = {
            "document_type": result.document_type,
            "issuing_country": result.issuing_country,
            "full_name": result.full_name,
            "date_of_birth": result.date_of_birth,
            "gender": result.gender,
            "identification_number": result.identification_number,
            "raw_data": result.raw_data,  # Full OCR text for debugging/auditing
            # All field_values as separate fields for flexibility
            **result.field_values,
        }

        # Build confidence_data from confidence_scores
        confidence_data = {}
        for field, confidence in result.confidence_scores.items():
            # Convert to 0-1 range if needed
            conf_value = confidence / 100 if confidence > 1 else confidence
            confidence_data[field] = {
                'overall_confidence': conf_value,
                'sources': ['gliner_ner']
            }

        # Also add core fields to confidence_data
        core_fields = ['full_name', 'date_of_birth', 'gender', 'identification_number']
        for field in core_fields:
            if field in extracted_data and extracted_data[field] and field not in confidence_data:
                confidence_data[field] = {
                    'overall_confidence': 0.85,  # Default confidence for extracted fields
                    'sources': ['gliner_ner']
                }

        self.logger.info(
            f"GLiNER ID card extraction: document_type={result.document_type}, "
            f"fields_extracted={len(result.field_values)}, "
            f"overall_confidence={result.overall_confidence:.2f}%"
        )

        return extracted_data, confidence_data

    def perform_document_specific_validations(
        self, extracted_data: Dict[str, Any], user_identity: Dict[str, Any]
    ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Perform ID card specific validations.

        Validations:
        - Document type detection (should be set)
        - Identification number validation (format varies by type)

        Note: ID cards have flexible validation since different
        card types have different requirements (e.g., PAN vs national ID).
        """
        document_type = extracted_data.get('document_type')

        # Check if document type was detected
        if not document_type:
            return False, "Could not detect ID card type", {}

        additional_checks = {
            'document_type': document_type,
            'issuing_country': extracted_data.get('issuing_country'),
        }

        # PAN-specific validation (if detected)
        if document_type and document_type.upper() == 'PAN':
            pan_number = extracted_data.get('pan_number') or extracted_data.get('identification_number')
            if pan_number:
                import re
                pan_pattern = re.compile(r'^[A-Z]{5}[0-9]{4}[A-Z]$', re.IGNORECASE)
                if not pan_pattern.match(pan_number):
                    additional_checks['pan_format_valid'] = False
                    return False, f"PAN number format invalid: {pan_number}", additional_checks
                additional_checks['pan_format_valid'] = True

        # UAE TRC-specific validation (if detected)
        if document_type and document_type.upper() == 'UAE_TRC':
            # No strict format validation for certificate numbers
            # Accept any certificate number format
            cert_number = extracted_data.get('certificate_number') or extracted_data.get('identification_number')
            if cert_number:
                additional_checks['certificate_format_valid'] = True
                additional_checks['certificate_number'] = cert_number

        # Singapore NRIC-specific validation (if detected)
        if document_type and document_type.upper() == 'SINGAPORE_NRIC':
            nric_number = extracted_data.get('nric_fin_number') or extracted_data.get('identification_number')
            if nric_number:
                import re
                nric_pattern = re.compile(r'^[STFG]\d{7}[A-Z]$', re.IGNORECASE)
                if not nric_pattern.match(nric_number):
                    additional_checks['nric_format_valid'] = False
                    return False, f"Singapore NRIC/FIN number format invalid: {nric_number}", additional_checks
                additional_checks['nric_format_valid'] = True

        # ID cards are generally valid if we got this far
        return True, None, additional_checks

    # ============================================================
    # UNIFIED VALIDATION ENTRY POINT (for both API and test script)
    # ============================================================

    @staticmethod
    async def extract_from_file(
        file_bytes: bytes,
        filename: str
    ) -> Dict[str, Any]:
        """
        SINGLE ENTRY POINT for ID card extraction.

        This is the ONLY method that should be called for ID card processing.
        Both the API endpoint and test script MUST use this method to ensure
        consistent behavior.

        This method:
        1. Extracts all fields using GLiNER ID card extractor
        2. Validates required fields
        3. Performs document-specific validations (e.g., PAN format)

        Supports:
        - Text-based PDFs (direct extraction)
        - Image-based PDFs (OCR extraction)
        - Regular images (JPG, PNG, etc. via OCR)

        Args:
            file_bytes: Raw file bytes (PDF or image)
            filename: Original filename

        Returns:
            Dict with extraction and validation results
        """
        # Create an instance to call validate_from_file
        service = SequentialIDCardService()
        return await service.validate_from_file(
            file_bytes=file_bytes,
            filename=filename,
            skip_photoholmes=True,
        )

    async def validate_from_file(
        self,
        file_bytes: bytes,
        filename: str,
        skip_photoholmes: bool = True,
    ) -> Dict[str, Any]:
        """
        Unified validation entry point for ID cards (PAN, national ID, driver's license).

        This method does everything from file bytes to full validation:
        1. Text extraction (direct PDF or OCR)
        2. Field extraction using GLiNER ID card extractor
        3. Required fields validation
        4. Document-specific validations (e.g., PAN format validation)

        Supports:
        - Text-based PDFs (direct extraction)
        - Image-based PDFs (OCR extraction)
        - Regular images (JPG, PNG, etc. via OCR)

        Args:
            file_bytes: Raw file bytes (PDF or image)
            filename: Original filename (used to determine if PDF)
            skip_photoholmes: Whether to skip forgery detection (default True for testing)

        Returns:
            Dict with keys:
                - success: bool - Overall validation result
                - extracted_data: dict - All extracted fields
                - confidence_data: dict - Confidence scores
                - validation_results: dict - Detailed validation results
                - error_message: str or None - Combined error message if failed
                - elapsed_seconds: float - Processing time
        """
        import time
        from app.helper.doctr.document_text_extractor import DocumentTextExtractor

        start_time = time.time()
        is_pdf = filename.lower().endswith('.pdf')

        # Initialize result structure
        result = {
            'success': False,
            'extracted_data': {},
            'confidence_data': {},
            'validation_results': {},
            'error_message': None,
            'elapsed_seconds': 0.0,
        }

        try:
            # Step 1: Text extraction with geometry
            self.logger.info("=" * 80)
            self.logger.info("UNIFIED ID CARD EXTRACTION PATH")
            self.logger.info(f"Processing {'PDF' if is_pdf else 'image'}")
            self.logger.info("=" * 80)

            ocr = DocumentTextExtractor()
            text_blocks = await ocr.extract_text_with_geometry_enhanced(
                file_bytes, is_pdf=is_pdf, max_pages=1, document_type=DocumentType.ID_CARD
            )

            self.logger.info(f"Text extraction complete: {len(text_blocks)} text blocks extracted")

            if not text_blocks:
                result['error_message'] = "Text extraction failed: no text found"
                result['elapsed_seconds'] = time.time() - start_time
                return result

            raw_text = "\n".join([block.get('text', '') for block in text_blocks])

            # Step 2: Field extraction using GLiNER ID card extractor
            extracted_data, confidence_data = await self.extract_fields_from_ocr(
                text_blocks=text_blocks,
                raw_text=raw_text,
                image_bytes=file_bytes,
                is_pdf=is_pdf
            )

            result['extracted_data'] = extracted_data
            result['confidence_data'] = confidence_data

            # Step 3: Validate required fields
            fields_valid, missing_fields, field_validations = self.validate_required_fields_with_confidence(
                extracted_data, confidence_data
            )

            result['validation_results']['required_fields'] = {
                'valid': fields_valid,
                'missing_fields': missing_fields,
                'field_validations': field_validations,
            }

            if not fields_valid:
                result['error_message'] = f"Missing required fields: {', '.join(missing_fields)}"
                result['elapsed_seconds'] = time.time() - start_time
                return result

            # Step 4: Document-specific validations
            doc_valid, doc_error, doc_validation_info = self.perform_document_specific_validations(
                extracted_data, {}
            )

            result['validation_results']['document_specific'] = doc_validation_info.get('validation_results', {})

            if not doc_valid:
                result['error_message'] = doc_error
                result['elapsed_seconds'] = time.time() - start_time
                return result

            # All validations passed
            result['success'] = True
            result['elapsed_seconds'] = time.time() - start_time
            return result

        except Exception as e:
            self.logger.error(f"ID card validation failed: {e}")
            result['error_message'] = str(e)
            result['elapsed_seconds'] = time.time() - start_time
            return result

    # ============================================================
    # ID CARD SPECIFIC METHODS
    # ============================================================

    # ============================================================
    # PUBLIC ENTRY POINT
    # ============================================================

    async def process_id_card(
        self, client_public_key: str, file_data: str, filename: str,
        callback_url: Optional[str] = None
    ):
        """
        Process an ID card document (PAN, national ID, driver's license).

        ID cards are independent documents - they don't affect verification_state.

        Args:
            client_public_key: Client's public key
            file_data: Base64 encoded file data
            filename: File name
            callback_url: Optional callback URL

        Returns:
            SequentialJobResponse with extraction results
        """
        return await self.process_document(
            client_public_key=client_public_key,
            file_data=file_data,
            filename=filename,
            callback_url=callback_url
        )

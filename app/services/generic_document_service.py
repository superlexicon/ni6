"""
Generic Document Service for processing documents with auto-detection.

This service handles the 'auto' document type by:
1. Detecting document type, country, and entity (three-tier detection)
2. Selecting the appropriate schema with fallback
3. Extracting fields using GLiNER2 zero-shot NER
4. Validating name matching against passport (if available)
5. Returning results in the standard SequentialJobResponse format

The service integrates with the existing document processing pipeline
but uses the new generic schema system for flexibility.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from app.core import get_logger
from app.core.gliner_ner_model import GLiNERNERModel
from app.dto import DocumentErrorCode
from app.schemas.generic import (
    SchemaRegistry,
    DocumentDetectionResult,
    ExtractionResult,
)
from app.helper.doctr.document_text_extractor import DocumentTextExtractor
from app.helper.extractors.generic import (
    DocumentTypeDetector,
    SchemaSelector,
    detect_document_type,
    select_schema,
)
from app.dto.verification_session import SequentialJobResponse
from app.schemas.bank_statement_schema import BankStatementData
from app.schemas.id_card_schema import IDCardData
from app.schemas.tax_statement_schema import TaxStatementData
from app.schemas.passport_schema import PassportData
from app.utils.string_matching import fuzzy_match_names, get_match_details
from app.config.verification_config import verification_settings


logger = get_logger()


# Mapping of document types to their name field names
NAME_FIELD_MAPPING = {
    "tax_return": "taxpayer_name",
    "tax_statement": "taxpayer_name",
    "id_card": "full_name",
    "driving_license": "full_name",
    "utility_bill": "customer_name",
    "bank_statement": "account_holder_name",
    "resume": "full_name",
    "wealth_declaration": "declarant_name",
}

# Date field names that should be normalized to ISO format
DATE_FIELD_NAMES = {
    "date_of_birth", "dob", "issue_date", "expiry_date", "date_of_issue", "date_of_expiry",
    "valid_from", "valid_until", "notice_date", "bill_date", "due_date", "filing_date",
    "enrollment_date", "download_date", "statement_date", "assessment_date", "assessment_year"
}


class GenericDocumentService:
    """
    Service for processing documents with auto-detection.

    This service is triggered when document_type="auto" and:
    1. Detects the document type (tax_return, id_card, driving_license, etc.)
    2. Detects the country (SG, IN, US, etc.)
    3. Detects the entity (DBS, SBI, IRAS, etc.)
    4. Selects the appropriate schema
    5. Extracts fields using GLiNER2
    6. Returns results in standard format

    The service does NOT handle:
    - passport (handled by SequentialPassportService)
    - bank_statement (handled by SequentialBankStatementService)
    - selfie (handled by separate logic)
    """

    def __init__(
        self,
        gliner_model: Optional[GLiNERNERModel] = None,
        detector: Optional[DocumentTypeDetector] = None,
        selector: Optional[SchemaSelector] = None,
        user_identity_repo=None,
        photoholmes_service=None,
    ):
        """
        Initialize the generic document service.

        Args:
            gliner_model: Optional GLiNER model (singleton if None)
            detector: Optional document type detector (created if None)
            selector: Optional schema selector (created if None)
            user_identity_repo: Repository for fetching user identity (passport name)
            photoholmes_service: Optional PhotoHolmes service for forgery detection
        """
        self.logger = get_logger()
        self.gliner_model = gliner_model or GLiNERNERModel()
        self.detector = detector or DocumentTypeDetector(gliner_model=self.gliner_model)
        self.selector = selector or SchemaSelector()
        self.text_extractor = DocumentTextExtractor()
        self.user_identity_repo = user_identity_repo
        self.photoholmes_service = photoholmes_service  # Optional forgery detection

    async def process_auto_document(
        self,
        file_data: Dict[str, Any],
        client_public_key: str,
        user_identity_id: str,
        hints: Optional[Dict[str, str]] = None,
    ) -> SequentialJobResponse:
        """
        Process a document with auto-detection.

        Args:
            file_data: Dictionary containing file information
                - file_data: Base64 encoded file content
                - file_type: File type (pdf, jpg, png, etc.)
            client_public_key: Client's public key for encryption
            user_identity_id: User identity identifier
            hints: Optional hints to guide detection
                - document_type: Hint for document type
                - country: Hint for country code (ISO 2-letter)
                - entity: Hint for entity identifier

        Returns:
            SequentialJobResponse with extraction results
        """
        self.logger.info(
            f"Processing auto document for user {user_identity_id} "
            f"with hints: {hints}"
        )

        start_time = datetime.now()

        try:
            # Extract hints
            hint_document_type = hints.get("document_type") if hints else None
            hint_country = hints.get("country") if hints else None
            hint_entity = hints.get("entity") if hints else None

            # Step 1: Perform OCR
            ocr_text, text_blocks = await self._perform_ocr(file_data)

            if not ocr_text or len(ocr_text.strip()) < 10:
                self.logger.warning("OCR extracted too little text, may be image-only document")
                return self._create_error_response(
                    error_code=DocumentErrorCode.OCR_FAILED,
                    error_message="Unable to extract text from document",
                    user_identity_id=user_identity_id,
                )

            self.logger.info(f"OCR extracted {len(ocr_text)} characters of text")

            # Step 2: Detect document type, country, entity
            detection_result = await self.detector.detect(
                text=ocr_text,
                hint_document_type=hint_document_type,
                hint_country=hint_country,
                hint_entity=hint_entity,
            )

            self.logger.info(
                f"Detection result: type={detection_result.document_type}, "
                f"country={detection_result.country_code}, "
                f"entity={detection_result.entity}, "
                f"confidence={detection_result.confidence:.2f}"
            )

            # Step 3: Select schema
            schema_selection = self.selector.select(
                detection_result=detection_result,
                confidence_threshold=0.4,
            )

            if not schema_selection.selected_schema:
                self.logger.warning("No schema found for detection result")
                return self._create_error_response(
                    error_code=DocumentErrorCode.NO_SCHEMA,
                    error_message=f"No schema found for document type: {detection_result.document_type}",
                    user_identity_id=user_identity_id,
                    detection_result=detection_result,
                )

            selected_schema = schema_selection.selected_schema
            self.logger.info(
                f"Selected schema: {selected_schema.schema_id} "
                f"(method={schema_selection.selection_method})"
            )

            # Step 4: Extract fields using GLiNER2
            extraction_result = await self._extract_fields(
                ocr_text=ocr_text,
                text_blocks=text_blocks,
                schema=selected_schema,
            )

            # Step 4a: Validate extraction results
            validation_result = self._validate_extraction_result(
                extraction_result=extraction_result,
                selected_schema=selected_schema,
                detection_result=detection_result,
                user_identity_id=user_identity_id,
            )
            if validation_result:
                # Validation failed - return error response
                return validation_result

            # Step 4b: Run PhotoHolmes forgery detection
            forgery_checks = None
            if self.photoholmes_service:
                forgery_checks = await self._run_forgery_detection(
                    file_data=file_data,
                    detection_result=detection_result,
                )

                # Step 4c: Validate forgery results
                forgery_validation = self._validate_forgery_result(
                    forgery_checks=forgery_checks,
                    detection_result=detection_result,
                    user_identity_id=user_identity_id,
                )
                if forgery_validation:
                    # Forgery detected - return error response
                    return forgery_validation

            # Step 5: Validate name matching against passport
            name_match_result = await self._validate_name_matching(
                extracted_data=extraction_result.extracted_data,
                document_type=detection_result.document_type,
                user_identity_id=user_identity_id,
            )

            # Step 6: Build response
            processing_time = (datetime.now() - start_time).total_seconds()

            response = self._create_success_response(
                detection_result=detection_result,
                extraction_result=extraction_result,
                selected_schema=selected_schema,
                user_identity_id=user_identity_id,
                processing_time=processing_time,
                name_match_result=name_match_result,
                forgery_checks=forgery_checks,
            )

            self.logger.info(
                f"Auto document processing complete in {processing_time:.2f}s: "
                f"type={detection_result.document_type}, "
                f"fields_extracted={len(extraction_result.extracted_data)}, "
                f"name_match={name_match_result.get('score', 'N/A') if name_match_result else 'skipped'}"
            )

            return response

        except Exception as e:
            self.logger.error(f"Error processing auto document: {e}")
            return self._create_error_response(
                error_code=DocumentErrorCode.PROCESSING_ERROR,
                error_message=str(e),
                user_identity_id=user_identity_id,
            )

    async def _perform_ocr(
        self,
        file_data: Dict[str, Any]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Perform OCR on the document.

        Returns:
            Tuple of (raw_text, text_blocks)
        """
        import base64

        # Decode file data
        file_content = file_data.get("file_data", "")
        if not file_content:
            raise ValueError("No file_data provided")

        # Decode base64
        file_bytes = base64.b64decode(file_content)

        # Determine if PDF
        file_type = file_data.get("file_type", "").lower()
        is_pdf = file_type == "pdf"

        # Extract text with geometry using DocumentTextExtractor
        text_blocks = await self.text_extractor.extract_text_with_geometry(file_bytes, is_pdf=is_pdf)

        # Also get raw text for GLiNER processing
        ocr_text = await self.text_extractor.extract_text(file_bytes, is_pdf=is_pdf)

        return ocr_text, text_blocks or []

    async def _extract_fields(
        self,
        ocr_text: str,
        text_blocks: List[Dict[str, Any]],
        schema: Any,
    ) -> ExtractionResult:
        """
        Extract fields using GLiNER2 schema-based extraction.

        Args:
            ocr_text: Raw OCR text
            text_blocks: Text blocks with spatial information
            schema: DocumentTypeSchema to use for extraction

        Returns:
            ExtractionResult with extracted data and confidence scores
        """
        try:
            model = await self.gliner_model.get_model_with_gpu()

            # Build GLiNER2 schema from extraction_schema
            field_descriptions = schema.extraction_schema.fields

            if not field_descriptions:
                self.logger.warning(f"No field descriptions in schema {schema.schema_id}")
                return ExtractionResult(
                    schema_used=schema,
                    extracted_data={},
                    confidence_scores={},
                    overall_confidence=0.0,
                    missing_required_fields=schema.required_fields[:],
                    extracted_fields=[],
                )

            # Create GLiNER2 schema
            gliner_schema = model.create_schema().entities(field_descriptions)

            # Run extraction
            entities_dict = model.extract(
                ocr_text,
                schema=gliner_schema,
                threshold=schema.extraction_schema.threshold,
                include_confidence=True,
                include_spans=True,
            )

            self.logger.debug(f"GLiNER2 raw response: {entities_dict}")

            # Handle GLiNER2 response format: {'entities': {field_name: [values]}}
            if 'entities' in entities_dict and isinstance(entities_dict['entities'], dict):
                entities_dict = entities_dict['entities']

            self.logger.debug(f"GLiNER2 extracted entities: {list(entities_dict.keys())}")

            # Process extraction results
            extracted_data = {}
            confidence_scores = {}

            for field_name, entities in entities_dict.items():
                if entities is None:
                    continue

                # Skip empty lists
                if isinstance(entities, list) and len(entities) == 0:
                    continue

                # Handle different entity formats
                if isinstance(entities, list) and len(entities) > 0:
                    # Get the highest confidence entity
                    best = max(entities, key=lambda e: e.get("confidence", 0))
                    # Use 'text' or 'value' for the entity text
                    value = best.get("text", best.get("value", "")).strip()
                    confidence = best.get("confidence", 0.0)
                elif isinstance(entities, dict):
                    value = entities.get("text", entities.get("value", "")).strip()
                    confidence = entities.get("confidence", 0.0)
                else:
                    value = str(entities).strip()
                    confidence = 0.5

                # Clean internal whitespace/newlines
                value = " ".join(value.split())

                if value:
                    extracted_data[field_name] = value
                    confidence_scores[field_name] = confidence

            # Calculate overall confidence
            overall_confidence = 0.0
            if confidence_scores:
                overall_confidence = sum(confidence_scores.values()) / len(confidence_scores)

            # Determine extracted and missing fields
            extracted_fields = list(extracted_data.keys())
            missing_required_fields = [
                f for f in schema.required_fields
                if f not in extracted_data
            ]

            return ExtractionResult(
                schema_used=schema,
                extracted_data=extracted_data,
                confidence_scores=confidence_scores,
                overall_confidence=overall_confidence,
                missing_required_fields=missing_required_fields,
                extracted_fields=extracted_fields,
            )

        except Exception as e:
            self.logger.error(f"GLiNER2 extraction failed: {e}")
            # Return empty result on error
            return ExtractionResult(
                schema_used=schema,
                extracted_data={},
                confidence_scores={},
                overall_confidence=0.0,
                missing_required_fields=schema.required_fields[:],
                extracted_fields=[],
            )

    async def _validate_name_matching(
        self,
        extracted_data: Dict[str, Any],
        document_type: str,
        user_identity_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Validate name matching against passport name.

        Args:
            extracted_data: Extracted data from the document
            document_type: Type of document being processed
            user_identity_id: User identity ID to fetch passport name

        Returns:
            Dictionary with name match result or None if validation skipped
        """
        try:
            # Check if user_identity_repo is available
            if not self.user_identity_repo:
                self.logger.debug("Name matching skipped: user_identity_repo not available")
                return None

            # Get the name field for this document type
            name_field = NAME_FIELD_MAPPING.get(document_type)
            if not name_field:
                self.logger.debug(f"Name matching skipped: no name field mapping for document type '{document_type}'")
                return None

            # Extract name from the document
            extracted_name = extracted_data.get(name_field)
            if not extracted_name:
                self.logger.debug(f"Name matching skipped: no '{name_field}' field in extracted data")
                return None

            # Fetch passport name from user identity
            user_identity = self.user_identity_repo.get_user_by_id(user_identity_id)
            if not user_identity:
                self.logger.warning(f"Name matching skipped: user identity not found for {user_identity_id}")
                return None

            stored_full_name = user_identity.get('full_name')
            if not stored_full_name:
                self.logger.debug("Name matching skipped: no passport name stored for user")
                return None

            # Perform fuzzy name matching
            score = round(fuzzy_match_names(stored_full_name, extracted_name) * 100, 1)
            details = get_match_details(stored_full_name, extracted_name)

            threshold = verification_settings.name_match_threshold  # Default: 70%
            is_valid = score >= threshold

            self.logger.info(
                f"Name matching for {document_type}: "
                f"passport='{stored_full_name}' vs extracted='{extracted_name}' "
                f"-> score={score}% (threshold={threshold}%) - "
                f"{'PASS' if is_valid else 'FAIL'}"
            )

            return {
                'is_valid': is_valid,
                'score': score,
                'threshold': threshold,
                'passport_name': stored_full_name,
                'extracted_name': extracted_name,
                'extracted_name_field': name_field,
                'normalized_passport_name': details.get('normalized_name1', ''),
                'normalized_extracted_name': details.get('normalized_name2', ''),
            }

        except Exception as e:
            self.logger.error(f"Name matching validation failed: {e}")
            return {
                'is_valid': False,
                'score': 0.0,
                'error': str(e),
            }

    def _normalize_date_to_iso(self, date_str: Optional[str]) -> Optional[str]:
        """
        Normalize a date string to ISO format (YYYY-MM-DD).

        Args:
            date_str: Date string in any supported format

        Returns:
            Date in ISO format (YYYY-MM-DD) or None if parsing fails
        """
        if not date_str:
            return None

        try:
            from datetime import datetime

            date_clean = str(date_str).strip()

            # Supported date formats
            date_formats = [
                "%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d", "%d/%m/%Y", "%Y%m%d",
                "%d %b %Y",   # 10 NOV 1992
                "%d %B %Y",   # 10 November 1992
                "%d-%b-%Y",   # 10-Nov-1992
                "%d/%b/%Y",   # 10/Nov/1992
                "%b %d, %Y",  # Nov 10, 1992
                "%B %d, %Y",  # November 10, 1992
            ]

            for date_variant in [date_clean, date_clean.upper(), date_clean.title()]:
                for fmt in date_formats:
                    try:
                        parsed = datetime.strptime(date_variant, fmt)
                        return parsed.strftime('%Y-%m-%d')
                    except ValueError:
                        continue

            return None
        except Exception:
            return None

    def _normalize_dates_in_extracted_data(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize all date fields in extracted data to ISO format.

        Args:
            extracted_data: Dictionary of extracted field values

        Returns:
            Dictionary with date fields normalized to ISO format
        """
        normalized_data = {}
        for field_name, value in extracted_data.items():
            if field_name in DATE_FIELD_NAMES and value:
                # Normalize date field to ISO format
                normalized_value = self._normalize_date_to_iso(str(value))
                normalized_data[field_name] = normalized_value if normalized_value else value
            else:
                normalized_data[field_name] = value

        return normalized_data

    def _validate_extraction_result(
        self,
        extraction_result: ExtractionResult,
        selected_schema: Any,
        detection_result: DocumentDetectionResult,
        user_identity_id: str,
    ) -> Optional[SequentialJobResponse]:
        """
        Validate extraction results and return error response if validation fails.

        Args:
            extraction_result: Result from field extraction
            selected_schema: The schema used for extraction
            detection_result: Document detection result
            user_identity_id: User identity ID

        Returns:
            SequentialJobResponse if validation fails, None otherwise
        """
        required_fields = selected_schema.required_fields or []
        if not required_fields:
            # No required fields defined - skip validation
            return None

        extracted_fields = set(extraction_result.extracted_data.keys())
        missing_required = set(required_fields) - extracted_fields

        # Calculate extraction percentage
        extraction_pct = len(extracted_fields & set(required_fields)) / len(required_fields) if required_fields else 1.0

        # Validation: Less than 50% of required fields - wrong document type
        if extraction_pct < 0.5:
            self.logger.warning(
                f"Wrong document type: only {extraction_pct*100:.0f}% of required fields extracted. "
                f"Required: {required_fields}, Extracted: {extracted_fields}"
            )
            return self._create_error_response(
                error_code=DocumentErrorCode.LOGICAL_WRONG_DOCUMENT_TYPE,
                error_message=(
                    f"Document appears to be incorrect type. Only extracted {extraction_pct*100:.0f}% "
                    f"of required fields. Extracted: {list(extracted_fields)}, Required: {required_fields}"
                ),
                user_identity_id=user_identity_id,
                detection_result=detection_result,
            )

        # Validation: 100% required not met - poor quality or low confidence
        if missing_required:
            self.logger.warning(
                f"Incomplete extraction: missing required fields {missing_required}"
            )
            return self._create_error_response(
                error_code=DocumentErrorCode.LOGICAL_EXTRACTION_INCOMPLETE,
                error_message=(
                    f"Could not extract all required fields. Missing: {list(missing_required)}. "
                    f"Please upload a clearer image or try again."
                ),
                user_identity_id=user_identity_id,
                detection_result=detection_result,
            )

        # Check confidence scores for required fields
        threshold = selected_schema.extraction_schema.threshold if hasattr(selected_schema, 'extraction_schema') else 0.5
        low_confidence_fields = []
        for field in required_fields:
            confidence = extraction_result.confidence_scores.get(field, 0.0)
            if confidence < threshold:
                low_confidence_fields.append(field)

        if low_confidence_fields:
            self.logger.warning(
                f"Low confidence extraction for fields: {low_confidence_fields}"
            )
            return self._create_error_response(
                error_code=DocumentErrorCode.LOGICAL_EXTRACTION_LOW_CONFIDENCE,
                error_message=(
                    f"Low confidence extraction for fields: {low_confidence_fields}. "
                    f"Please upload a clearer image or try again."
                ),
                user_identity_id=user_identity_id,
                detection_result=detection_result,
            )

        # All validations passed
        return None

    async def _run_forgery_detection(
        self,
        file_data: Dict[str, Any],
        detection_result: DocumentDetectionResult,
    ) -> Optional[Dict[str, Any]]:
        """
        Run PhotoHolmes forgery detection on the document.

        Args:
            file_data: Dictionary containing file information
            detection_result: Document detection result

        Returns:
            Forgery detection results as dict, or None if service not available
        """
        if not self.photoholmes_service:
            self.logger.debug("PhotoHolmes service not available - skipping forgery detection")
            return None

        try:
            import base64
            file_bytes = base64.b64decode(file_data.get("file_data", ""))

            self.logger.info(
                f"Running PhotoHolmes forgery detection for {detection_result.document_type}"
            )
            forgery_result = await self.photoholmes_service.run_all_methods(
                image_bytes=file_bytes,
                document_type=detection_result.document_type
            )

            if forgery_result:
                self.logger.info(
                    f"PhotoHolmes completed: {forgery_result.methods_with_detections}/{forgery_result.total_methods_run} "
                    f"methods detected forgery, overall probability: {forgery_result.overall_forgery_probability:.4f}"
                )
                return forgery_result.model_dump() if hasattr(forgery_result, 'model_dump') else None

        except Exception as e:
            self.logger.warning(f"PhotoHolmes check failed: {e}")

        return None

    def _validate_forgery_result(
        self,
        forgery_checks: Optional[Dict[str, Any]],
        detection_result: DocumentDetectionResult,
        user_identity_id: str,
    ) -> Optional[SequentialJobResponse]:
        """
        Validate PhotoHolmes forgery detection results.

        Args:
            forgery_checks: PhotoHolmes forgery detection results
            detection_result: Document detection result
            user_identity_id: User identity ID

        Returns:
            SequentialJobResponse if forgery detected, None otherwise
        """
        if not forgery_checks:
            return None

        # Get the threshold from verification settings
        threshold = verification_settings.forgery_detection_threshold  # Default: 3

        # Get methods_with_detections from forgery_checks
        detections = forgery_checks.get('methods_with_detections', 0)

        if detections >= threshold:
            # Get names of methods that detected forgery
            detected_methods = []
            for method_key in ['dq', 'adaptive', 'noisesniffer', 'psccnet', 'focal', 'splicebuster', 'trufor', 'zero']:
                method_data = forgery_checks.get(method_key)
                if method_data:
                    # Check if this method detected forgery (score > 0.1)
                    score = 0.0
                    if hasattr(method_data, 'max_probability'):
                        score = method_data.max_probability
                    elif hasattr(method_data, 'tampered_ratio'):
                        score = method_data.tampered_ratio
                    elif isinstance(method_data, dict):
                        score = method_data.get('max_probability', method_data.get('tampered_ratio', 0.0))

                    if score > 0.1:
                        method_names = {
                            'dq': 'DQ',
                            'adaptive': 'Adaptive',
                            'noisesniffer': 'NoiseSniffer',
                            'psccnet': 'PSCCNet',
                            'focal': 'FOCAL',
                            'splicebuster': 'Splicebuster',
                            'trufor': 'TruFor',
                            'zero': 'ZERO',
                        }
                        detected_methods.append(method_names.get(method_key, method_key))

            self.logger.error(
                f"Forgery detected: {detections} methods flagged manipulation: {detected_methods}"
            )

            return self._create_error_response(
                error_code=DocumentErrorCode.LOGICAL_FORGERY_DETECTED,
                error_message=(
                    f"Document validation failed - {detections} forgery detection methods "
                    f"flagged potential manipulation: {', '.join(detected_methods)}. "
                    f"This document cannot be accepted for verification."
                ),
                user_identity_id=user_identity_id,
                detection_result=detection_result,
                forgery_checks=forgery_checks,
            )

        # Forgery validation passed
        self.logger.info(f"✅ PhotoHolmes validation passed - {detections} detections (threshold: {threshold})")
        return None

    def _create_success_response(
        self,
        detection_result: DocumentDetectionResult,
        extraction_result: ExtractionResult,
        selected_schema: Any,
        user_identity_id: str,
        processing_time: float,
        name_match_result: Optional[Dict[str, Any]] = None,
        forgery_checks: Optional[Dict[str, Any]] = None,
    ) -> SequentialJobResponse:
        """
        Create a success response in standard format.
        """
        # Normalize date fields in extracted data to ISO format
        normalized_extracted_data = self._normalize_dates_in_extracted_data(
            extraction_result.extracted_data
        )

        # Build extracted_data dict
        extracted_data = {
            **normalized_extracted_data,
            # Add detection metadata
            "_detection": {
                "document_type": detection_result.document_type,
                "country": detection_result.country_code,
                "entity": detection_result.entity,
                "schema_id": selected_schema.schema_id,
                "detection_confidence": detection_result.confidence,
            }
        }

        # Add name match result to extracted_data if available
        if name_match_result:
            extracted_data["_name_match"] = name_match_result

        # Build confidence_data dict
        confidence_data = {}
        for field, confidence in extraction_result.confidence_scores.items():
            confidence_data[field] = {
                'overall_confidence': confidence,
                'sources': ['gliner2_ner']
            }

        # Map to appropriate schema model based on document type
        document_type = detection_result.document_type
        if document_type == "id_card":
            data_schema = IDCardData(
                field_values=extraction_result.extracted_data,
                confidence_scores=confidence_data,
                overall_confidence=extraction_result.overall_confidence * 100,
            )
        elif document_type == "tax_return":
            data_schema = TaxStatementData(
                taxpayer_name=extraction_result.extracted_data.get("taxpayer_name"),
                tax_id=extraction_result.extracted_data.get("tax_id_number"),
                tax_year=extraction_result.extracted_data.get("assessment_year"),
                gross_income=extraction_result.extracted_data.get("total_income"),
                tax_paid=extraction_result.extracted_data.get("tax_paid"),
                confidence_scores=confidence_data,
                overall_confidence=extraction_result.overall_confidence * 100,
            )
        elif document_type == "bank_statement":
            data_schema = BankStatementData(
                account_holder_name=extraction_result.extracted_data.get("account_holder_name"),
                account_number=extraction_result.extracted_data.get("account_number"),
                bank_name=extraction_result.extracted_data.get("bank_name"),
                currency=extraction_result.extracted_data.get("currency"),
                statement_date=extraction_result.extracted_data.get("statement_date"),
                address=extraction_result.extracted_data.get("address"),
                confidence_scores=confidence_data,
                overall_confidence=extraction_result.overall_confidence * 100,
            )
        else:
            # Generic schema - use dict-based response
            data_schema = extraction_result.extracted_data

        # Create response
        response = SequentialJobResponse(
            result=True,
            job_id="",  # Not used in test mode
            verification_state=0,  # Not used in test mode
            message=f"Successfully processed {detection_result.document_type_name or detection_result.document_type}",
            extracted_data=extracted_data,
            confidence_data=confidence_data,
            overall_confidence=extraction_result.overall_confidence * 100,
            document_type=detection_result.document_type,
            processing_time_seconds=processing_time,
            user_identity_id=user_identity_id,
            forgery_checks=forgery_checks,
            # Additional fields for auto-detection
            detected_document_type=detection_result.document_type,
            detected_country=detection_result.country_code,
            detected_entity=detection_result.entity,
            detection_confidence=detection_result.confidence,
            selected_schema=selected_schema.schema_id,
        )

        return response

    def _create_error_response(
        self,
        error_code: str,
        error_message: str,
        user_identity_id: str,
        detection_result: Optional[DocumentDetectionResult] = None,
        forgery_checks: Optional[Dict[str, Any]] = None,
    ) -> SequentialJobResponse:
        """Create an error response."""
        return SequentialJobResponse(
            result=False,
            job_id="",  # Not used in test mode
            verification_state=0,  # Not used in test mode
            message=f"Error processing document: {error_message}",
            error_code=error_code,
            extracted_data={},
            confidence_data={},
            overall_confidence=0.0,
            document_type=detection_result.document_type if detection_result else "unknown",
            user_identity_id=user_identity_id,
            forgery_checks=forgery_checks,
            detected_document_type=detection_result.document_type if detection_result else None,
            detected_country=detection_result.country_code if detection_result else None,
            detected_entity=detection_result.entity if detection_result else None,
            detection_confidence=detection_result.confidence if detection_result else 0.0,
        )


__all__ = [
    "GenericDocumentService",
]

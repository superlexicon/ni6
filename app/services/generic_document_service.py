"""
Generic Document Service for processing documents with auto-detection.

This service handles the 'auto' document type by:
1. Routing to the appropriate Qwen extractor based on document type and country
2. Extracting fields using Qwen3-VL (NRIC, PAN, UAE TRC, Generic)
3. Validating name matching against passport (if available)
4. Returning results in the standard SequentialJobResponse format

The service does NOT handle:
- passport (handled by SequentialPassportService)
- bank_statement (handled by SequentialBankStatementService)
- selfie (handled by separate logic)
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from app.core import get_logger
from app.dto import DocumentErrorCode
from app.dto.verification_session import SequentialJobResponse
from app.schemas.bank_statement_schema import BankStatementData
from app.schemas.id_card_schema import IDCardData
from app.schemas.tax_statement_schema import TaxStatementData
from app.schemas.passport_schema import PassportData
from app.utils.string_matching import fuzzy_match_names, get_match_details
from app.config.verification_config import verification_settings
from app.services.extractors.qwen_pan_extractor import QwenPANExtractor
from app.services.extractors.qwen_singapore_nric_extractor import QwenSingaporeNricExtractor
from app.services.extractors.qwen_uae_trc_extractor import QwenUAETrcExtractor
from app.services.extractors.qwen_generic_document_extractor import QwenGenericDocumentExtractor


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
    1. Routes to the appropriate Qwen extractor based on document type and country
    2. Extracts fields using Qwen3-VL (NRIC, PAN, UAE TRC, Generic)
    3. Validates name matching against passport (if available)
    4. Returns results in standard format

    Supported document types:
    - Singapore NRIC (id_card + SG)
    - Indian PAN (id_card + IN)
    - UAE TRC (id_card + AE)
    - Generic documents (misc, other, unknown)
    """

    def __init__(
        self,
        user_identity_repo=None,
        photoholmes_service=None,
    ):
        """
        Initialize the generic document service.

        Args:
            user_identity_repo: Repository for fetching user identity (passport name)
            photoholmes_service: Optional PhotoHolmes service for forgery detection
        """
        self.logger = get_logger()
        self.user_identity_repo = user_identity_repo
        self.photoholmes_service = photoholmes_service  # Optional forgery detection

    async def process_auto_document(
        self,
        file_data: Dict[str, Any],
        client_public_key: str,
        user_identity_id: str,
        document_type: str,
        country_code: Optional[str] = None,
        entity: Optional[str] = None,
    ) -> SequentialJobResponse:
        """
        Process a document with the specified document type and country.

        Args:
            file_data: Dictionary containing file information
                - file_data: Base64 encoded file content
                - file_type: File type (pdf, jpg, png, etc.)
            client_public_key: Client's public key for encryption
            user_identity_id: User identity identifier
            document_type: Document type (e.g., 'id_card', 'nric', 'pan', 'uae_trc', 'misc')
            country_code: Optional ISO 3166-1 alpha-2 country code (e.g., 'SG', 'IN', 'AE')
            entity: Optional entity identifier (e.g., bank name)

        Returns:
            SequentialJobResponse with extraction results
        """
        self.logger.info(
            f"Processing auto document for user {user_identity_id} "
            f"with document_type={document_type}, country_code={country_code}, entity={entity}"
        )

        start_time = datetime.now()

        try:
            # Step 1: Decode file bytes and normalize document type
            import base64

            file_content = file_data.get("file_data", "")
            if not file_content:
                raise ValueError("No file_data provided")

            file_bytes = base64.b64decode(file_content)

            # Step 2: Normalize document type and country
            normalized_doc_type = self._normalize_document_type(document_type)

            # Infer country from document type if not provided
            if not country_code:
                country_code = self._infer_country_from_document_type(document_type)
                self.logger.info(f"Inferred country code '{country_code}' from document type '{document_type}'")

            # Infer entity from document type if not provided
            if not entity:
                entity = self._infer_entity_from_document_type(document_type)
                if entity:
                    self.logger.info(f"Inferred entity '{entity}' from document type '{document_type}'")

            normalized_country = self._normalize_country_code(country_code, document_type) if country_code else None

            # Create DocumentDetectionResult from passed parameters
            from app.schemas.generic import DocumentDetectionResult
            detection_result = DocumentDetectionResult(
                document_type=normalized_doc_type,
                document_type_name=document_type,
                country_code=normalized_country,
                country_name=country_code,
                entity=entity,
                entity_name=entity,
                confidence=1.0,  # High confidence since it's provided by caller
                type_confidence=1.0,
                country_confidence=1.0 if country_code else 0.0,
                entity_confidence=1.0 if entity else 0.0,
                detected_keywords=[],
                detected_patterns=[],
                detection_method="direct",
            )

            self.logger.info(
                f"Using document type: {detection_result.document_type}, "
                f"country: {detection_result.country_code}"
            )

            # Step 3: Route to appropriate Qwen extractor
            extraction_result = await self._route_to_extractor(file_bytes, detection_result)

            if not extraction_result or not extraction_result.extracted_data:
                self.logger.warning("Extraction returned no data")
                return self._create_error_response(
                    error_code=DocumentErrorCode.PROCESSING_ERROR,
                    error_message="No data could be extracted from the document",
                    user_identity_id=user_identity_id,
                    detection_result=detection_result,
                )

            # Step 4: Run PhotoHolmes forgery detection
            forgery_checks = None
            if self.photoholmes_service:
                forgery_checks = await self._run_forgery_detection(
                    file_data=file_data,
                    detection_result=detection_result,
                )

                # Step 5: Validate forgery results
                forgery_validation = self._validate_forgery_result(
                    forgery_checks=forgery_checks,
                    detection_result=detection_result,
                    user_identity_id=user_identity_id,
                )
                if forgery_validation:
                    # Forgery detected - return error response
                    return forgery_validation

            # Step 6: Validate name matching against passport
            name_match_result = await self._validate_name_matching(
                extracted_data=extraction_result.extracted_data,
                document_type=detection_result.document_type,
                user_identity_id=user_identity_id,
            )

            # Step 7: Build response
            processing_time = (datetime.now() - start_time).total_seconds()

            response = self._create_success_response(
                detection_result=detection_result,
                extraction_result=extraction_result,
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

    async def _route_to_extractor(
        self,
        file_bytes: bytes,
        detection_result,
    ) -> Any:
        """
        Route to the appropriate Qwen extractor based on document type and country.

        Args:
            file_bytes: Image bytes
            detection_result: Document detection result

        Returns:
            ExtractionResult with extracted data
        """
        from app.schemas.generic import ExtractionResult

        doc_type = detection_result.document_type
        country_code = detection_result.country_code

        self.logger.info(f"Routing document: type={doc_type}, country={country_code}")

        # Handle generic/unknown document types
        generic_doc_types = ["other", "misc", "unknown", "generic", "document"]
        if doc_type in generic_doc_types:
            self.logger.info(f"Using Qwen3-VL generic extractor for document type '{doc_type}'")
            return await self._extract_with_qwen_generic(file_bytes)

        # Singapore NRIC/FIN: id_card + SG country
        if doc_type == "id_card" and country_code == "SG":
            self.logger.info("Using Qwen3-VL for Singapore NRIC/FIN extraction")
            return await self._extract_with_qwen_nric(file_bytes)

        # PAN Card: id_card + IN country
        if doc_type == "id_card" and country_code == "IN":
            self.logger.info("Using Qwen3-VL for PAN Card extraction")
            return await self._extract_with_qwen_pan(file_bytes)

        # UAE TRC: id_card + AE country (UAE)
        if doc_type == "id_card" and country_code == "AE":
            self.logger.info("Using Qwen3-VL for UAE TRC extraction")
            return await self._extract_with_qwen_uae_trc(file_bytes)

        # For unsupported document types, use generic extractor
        self.logger.warning(
            f"Unsupported document type '{doc_type}' with country '{country_code}' - "
            f"falling back to generic extractor"
        )
        return await self._extract_with_qwen_generic(file_bytes)

    async def _extract_with_qwen_nric(self, file_bytes: bytes) -> Any:
        """
        Extract Singapore NRIC/FIN fields using Qwen3-VL.

        Args:
            file_bytes: Image bytes

        Returns:
            ExtractionResult with extracted data
        """
        from app.schemas.generic import ExtractionResult

        try:
            extractor = QwenSingaporeNricExtractor()
            qwen_extracted_data, qwen_confidence_data = await extractor.extract_fields(file_bytes)

            # Convert Qwen format to ExtractionResult format
            extracted_data = {}
            confidence_scores = {}

            for field_name, field_data in qwen_extracted_data.items():
                if isinstance(field_data, dict) and "value" in field_data:
                    value = field_data["value"]
                    confidence = field_data.get("confidence", 1.0)

                    # Skip None/null values
                    if value is None:
                        continue

                    # Map Qwen field names to standard field names
                    if field_name == "nric_fin_number":
                        extracted_data["nric_number"] = value
                        confidence_scores["nric_number"] = confidence
                        # Also map to universal tax_id_number field
                        if "tax_id_number" not in extracted_data:
                            extracted_data["tax_id_number"] = value
                            confidence_scores["tax_id_number"] = confidence
                    elif field_name == "full_name":
                        extracted_data["full_name"] = value
                        confidence_scores["full_name"] = confidence
                    elif field_name == "date_of_birth":
                        extracted_data["date_of_birth"] = value
                        confidence_scores["date_of_birth"] = confidence
                    elif field_name == "sex":
                        extracted_data["sex"] = value
                        confidence_scores["sex"] = confidence
                    elif field_name == "card_type":
                        extracted_data["card_type"] = value
                        confidence_scores["card_type"] = confidence
                    else:
                        extracted_data[field_name] = value
                        confidence_scores[field_name] = confidence

            # Calculate overall confidence
            overall_confidence = 0.0
            if confidence_scores:
                overall_confidence = sum(confidence_scores.values()) / len(confidence_scores)

            # Determine extracted fields
            extracted_fields = list(extracted_data.keys())

            # No required fields for NRIC extraction
            missing_required_fields = []

            self.logger.info(f"Qwen3-VL NRIC extraction completed: {len(extracted_data)} fields extracted")

            return ExtractionResult(
                schema_used=None,  # No schema needed
                extracted_data=extracted_data,
                confidence_scores=confidence_scores,
                overall_confidence=overall_confidence,
                missing_required_fields=missing_required_fields,
                extracted_fields=extracted_fields,
            )

        except Exception as e:
            self.logger.error(f"Qwen3-VL NRIC extraction failed: {e}")
            # Return empty result on error
            return ExtractionResult(
                schema_used=None,
                extracted_data={},
                confidence_scores={},
                overall_confidence=0.0,
                missing_required_fields=[],
                extracted_fields=[],
            )

    async def _extract_with_qwen_pan(self, file_bytes: bytes) -> Any:
        """
        Extract PAN card fields using Qwen3-VL.

        Args:
            file_bytes: Image bytes

        Returns:
            ExtractionResult with extracted data
        """
        from app.schemas.generic import ExtractionResult

        try:
            extractor = QwenPANExtractor()
            qwen_extracted_data, qwen_confidence_data = await extractor.extract_fields(file_bytes)

            # Convert Qwen format to ExtractionResult format
            extracted_data = {}
            confidence_scores = {}

            for field_name, field_data in qwen_extracted_data.items():
                if isinstance(field_data, dict) and "value" in field_data:
                    value = field_data["value"]
                    confidence = field_data.get("confidence", 1.0)

                    # Skip None/null values
                    if value is None:
                        continue

                    # Store all fields
                    extracted_data[field_name] = value
                    confidence_scores[field_name] = confidence

            # Calculate overall confidence
            overall_confidence = 0.0
            if confidence_scores:
                overall_confidence = sum(confidence_scores.values()) / len(confidence_scores)

            # Determine extracted fields
            extracted_fields = list(extracted_data.keys())

            # No required fields for PAN extraction
            missing_required_fields = []

            self.logger.info(f"Qwen3-VL PAN extraction completed: {len(extracted_data)} fields extracted")

            return ExtractionResult(
                schema_used=None,  # No schema needed
                extracted_data=extracted_data,
                confidence_scores=confidence_scores,
                overall_confidence=overall_confidence,
                missing_required_fields=missing_required_fields,
                extracted_fields=extracted_fields,
            )

        except Exception as e:
            self.logger.error(f"Qwen3-VL PAN extraction failed: {e}")
            # Return empty result on error
            return ExtractionResult(
                schema_used=None,
                extracted_data={},
                confidence_scores={},
                overall_confidence=0.0,
                missing_required_fields=[],
                extracted_fields=[],
            )

    async def _extract_with_qwen_uae_trc(self, file_bytes: bytes) -> Any:
        """
        Extract UAE TRC fields using Qwen3-VL.

        Args:
            file_bytes: Image bytes

        Returns:
            ExtractionResult with extracted data
        """
        from app.schemas.generic import ExtractionResult

        try:
            extractor = QwenUAETrcExtractor()
            qwen_extracted_data, qwen_confidence_data = await extractor.extract_fields(file_bytes)

            # Convert Qwen format to ExtractionResult format
            extracted_data = {}
            confidence_scores = {}

            for field_name, field_data in qwen_extracted_data.items():
                if isinstance(field_data, dict) and "value" in field_data:
                    value = field_data["value"]
                    # Skip None values
                    if value is None:
                        continue
                    confidence = field_data.get("confidence", 1.0)

                    # Map Qwen field names to standard field names
                    if field_name == "certificate_number":
                        extracted_data["certificate_number"] = value
                        confidence_scores["certificate_number"] = confidence
                    elif field_name == "full_name":
                        extracted_data["full_name"] = value
                        confidence_scores["full_name"] = confidence
                    elif field_name == "valid_until":
                        extracted_data["expiry_date"] = value
                        confidence_scores["expiry_date"] = confidence
                    elif field_name == "valid_from":
                        extracted_data["valid_from"] = value
                        confidence_scores["valid_from"] = confidence
                    elif field_name == "application_number":
                        extracted_data["application_number"] = value
                        confidence_scores["application_number"] = confidence
                    elif field_name == "passport_number":
                        extracted_data["passport_number"] = value
                        confidence_scores["passport_number"] = confidence
                    elif field_name == "nationality":
                        extracted_data["nationality"] = value
                        confidence_scores["nationality"] = confidence
                    else:
                        extracted_data[field_name] = value
                        confidence_scores[field_name] = confidence

            # Calculate overall confidence
            overall_confidence = 0.0
            if confidence_scores:
                overall_confidence = sum(confidence_scores.values()) / len(confidence_scores)

            # Determine extracted fields
            extracted_fields = list(extracted_data.keys())

            # No required fields for UAE TRC extraction
            missing_required_fields = []

            self.logger.info(f"Qwen3-VL UAE TRC extraction completed: {len(extracted_data)} fields extracted")

            return ExtractionResult(
                schema_used=None,  # No schema needed
                extracted_data=extracted_data,
                confidence_scores=confidence_scores,
                overall_confidence=overall_confidence,
                missing_required_fields=missing_required_fields,
                extracted_fields=extracted_fields,
            )

        except Exception as e:
            self.logger.error(f"Qwen3-VL UAE TRC extraction failed: {e}")
            # Return empty result on error
            return ExtractionResult(
                schema_used=None,
                extracted_data={},
                confidence_scores={},
                overall_confidence=0.0,
                missing_required_fields=[],
                extracted_fields=[],
            )

    async def _extract_with_qwen_generic(self, file_bytes: bytes) -> Any:
        """
        Extract document type and PII fields using Qwen3-VL generic extractor.

        This is used for unknown or unspecified document types.
        It both classifies the document type and extracts all PII fields.

        Args:
            file_bytes: Image bytes

        Returns:
            ExtractionResult with extracted data
        """
        from app.schemas.generic import ExtractionResult

        try:
            extractor = QwenGenericDocumentExtractor()
            qwen_extracted_data, qwen_confidence_data = await extractor.extract_fields(file_bytes)

            # Convert Qwen format to ExtractionResult format
            extracted_data = {}
            confidence_scores = {}

            for field_name, field_data in qwen_extracted_data.items():
                if isinstance(field_data, dict) and "value" in field_data:
                    value = field_data["value"]
                    # Skip None values
                    if value is None:
                        continue
                    confidence = field_data.get("confidence", 1.0)

                    # Store all extracted fields
                    extracted_data[field_name] = value
                    confidence_scores[field_name] = confidence

            # Calculate overall confidence
            overall_confidence = 0.0
            if confidence_scores:
                overall_confidence = sum(confidence_scores.values()) / len(confidence_scores)

            # Determine extracted fields
            extracted_fields = list(extracted_data.keys())

            # No required fields for generic extraction
            missing_required_fields = []

            self.logger.info(
                f"Qwen3-VL generic document extraction completed: "
                f"{len(extracted_data)} fields extracted, "
                f"document_type={extracted_data.get('document_type', 'unknown')}"
            )

            return ExtractionResult(
                schema_used=None,  # No schema needed
                extracted_data=extracted_data,
                confidence_scores=confidence_scores,
                overall_confidence=overall_confidence,
                missing_required_fields=missing_required_fields,
                extracted_fields=extracted_fields,
            )

        except Exception as e:
            self.logger.error(f"Qwen3-VL generic extraction failed: {e}")
            # Return empty result on error
            return ExtractionResult(
                schema_used=None,
                extracted_data={},
                confidence_scores={},
                overall_confidence=0.0,
                missing_required_fields=[],
                extracted_fields=[],
            )

    def _normalize_document_type(self, document_type: str) -> str:
        """
        Normalize document type to standard format.

        Maps user-friendly names to internal document types:
        - 'nric', 'nric_card' -> 'id_card'
        - 'pan', 'pan_card' -> 'id_card'
        - 'uae_trc', 'trc' -> 'id_card'
        - 'id_card' -> 'id_card'

        Args:
            document_type: User-provided document type

        Returns:
            Normalized document type for internal use
        """
        doc_type_lower = document_type.lower().strip()

        # Singapore NRIC variants
        if doc_type_lower in ['nric', 'nric_card', 'singapore_nric', 'singapore_nric_card']:
            return 'id_card'

        # PAN Card variants
        if doc_type_lower in ['pan', 'pan_card', 'indian_pan']:
            return 'id_card'

        # UAE TRC variants
        if doc_type_lower in ['uae_trc', 'trc', 'tax_residency_certificate']:
            return 'id_card'

        # Already normalized
        if doc_type_lower == 'id_card':
            return 'id_card'

        # Default: return as-is for other document types
        return doc_type_lower

    def _normalize_country_code(self, country_code: str, document_type: str) -> str:
        """
        Normalize country code to ISO 3166-1 alpha-2 format.

        Maps user-friendly country names/codes to standard ISO codes:
        - 'singapore', 'sg' -> 'SG'
        - 'india', 'in' -> 'IN'
        - 'uae', 'ae' -> 'AE'
        - etc.

        Args:
            country_code: User-provided country code or name
            document_type: Document type for context-aware normalization

        Returns:
            Normalized ISO 3166-1 alpha-2 country code
        """
        country_upper = country_code.upper().strip()

        # If already in ISO format (2 letters), return as-is
        if len(country_upper) == 2 and country_upper.isalpha():
            return country_upper

        # Map common country names to ISO codes
        country_map = {
            'SINGAPORE': 'SG',
            'INDIA': 'IN',
            'UNITED ARAB EMIRATES': 'AE',
            'UAE': 'AE',
            'MALAYSIA': 'MY',
            'THAILAND': 'TH',
            'USA': 'US',
            'UNITED STATES': 'US',
            'UK': 'GB',
            'UNITED KINGDOM': 'GB',
        }

        # Also check lowercase variants
        country_lower = country_code.lower().strip()
        country_map_lower = {k.lower(): v for k, v in country_map.items()}

        if country_lower in country_map_lower:
            return country_map_lower[country_lower]

        # Context-aware defaults based on document type
        doc_type_lower = document_type.lower().strip()
        if doc_type_lower in ['nric', 'nric_card', 'singapore_nric']:
            return 'SG'
        elif doc_type_lower in ['pan', 'pan_card', 'indian_pan']:
            return 'IN'
        elif doc_type_lower in ['uae_trc', 'trc', 'tax_residency_certificate']:
            return 'AE'

        # Default: return original (will likely fail schema selection)
        return country_upper

    def _infer_country_from_document_type(self, document_type: str) -> Optional[str]:
        """
        Infer country code from document type when not explicitly provided.

        This is used when the caller only provides document_type but not country_code.
        For Qwen-supported document types, we can infer the country:
        - 'nric' -> 'SG' (Singapore NRIC)
        - 'pan' -> 'IN' (India PAN)
        - 'uae_trc' -> 'AE' (UAE Tax Residency Certificate)

        Args:
            document_type: User-provided document type

        Returns:
            Inferred ISO 3166-1 alpha-2 country code, or None if cannot infer
        """
        doc_type_lower = document_type.lower().strip()

        # Singapore NRIC variants
        if doc_type_lower in ['nric', 'nric_card', 'singapore_nric', 'singapore_nric_card']:
            return 'SG'

        # PAN Card variants
        if doc_type_lower in ['pan', 'pan_card', 'indian_pan']:
            return 'IN'

        # UAE TRC variants
        if doc_type_lower in ['uae_trc', 'trc', 'tax_residency_certificate']:
            return 'AE'

        # Cannot infer country from this document type
        return None

    def _infer_entity_from_document_type(self, document_type: str) -> Optional[str]:
        """
        Infer entity from document type when not explicitly provided.

        This is used when the caller only provides document_type but not entity.
        For Qwen-supported document types with specific entities:
        - 'uae_trc', 'trc', 'tax_residency_certificate' -> 'trc'

        Args:
            document_type: User-provided document type

        Returns:
            Inferred entity identifier, or None if cannot infer
        """
        doc_type_lower = document_type.lower().strip()

        # UAE TRC variants
        if doc_type_lower in ['uae_trc', 'trc', 'tax_residency_certificate']:
            return 'trc'

        # Cannot infer entity from this document type
        return None

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

    async def _run_forgery_detection(
        self,
        file_data: Dict[str, Any],
        detection_result,
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
        detection_result,
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
        self.logger.info(f"PhotoHolmes validation passed - {detections} detections (threshold: {threshold})")
        return None

    def _create_success_response(
        self,
        detection_result,
        extraction_result,
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
                "extraction_method": "qwen3_vl",
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
                'sources': ['qwen3_vl']
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
            selected_schema="qwen3_vl",
        )

        return response

    def _create_error_response(
        self,
        error_code: str,
        error_message: str,
        user_identity_id: str,
        detection_result = None,
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

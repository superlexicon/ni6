"""
Sequential Bank Statement Service - Handles bank statement processing in sequential mode.

REFACTORED: Now uses DocumentProcessorBase for unified validation pipeline.
Uses HYBRID GLiNER2 + Spatial extraction for bank statement field extraction.

HYBRID STRATEGY:
1. Always run GLiNER2 extraction (primary, semantic understanding)
2. Always run Spatial extraction IN FULL (backup, geometry-based)
3. Merge results at field level based on confidence:
   - Use GLiNER result if confidence >= threshold (50%)
   - Otherwise use spatial result (if available)
4. Account numbers always come from spatial (requires label proximity)
5. Statement date uses post-processing to select the largest/latest date

GLiNER2 provides:
- Semantic understanding of bank statement fields
- Works on raw text (OCR or PDF-extracted)
- Validated against SWIFT bank database

Spatial extractor provides:
- Geometry-based extraction (60-85% confidence)
- Backup for missing or low-confidence GLiNER fields

State is tracked via verification_state column in user_identity_index.
"""

import asyncio
import re
from typing import Dict, Any, Optional, Tuple, List
from datetime import date
from app.services.sequential_document_processor_base import DocumentProcessorBase
from app.core.key_injection.simple_bank_analyzer import simple_bank_analyzer
from app.core.key_injection.key_injection_manager import key_injection_manager
from app.core.key_injection import DocumentType
from app.helper.extractors.spatial_bank_statement_extractor import SpatialBankStatementExtractor
from app.helper.extractors import get_gliner_bank_statement_extractor
from app.helper.utils.address_cleaner import clean_gliner_address
from app.dto import DocumentErrorCode


class SequentialBankStatementService(DocumentProcessorBase):
    """Service for handling bank statement processing using unified framework."""

    # ============================================================
    # ABSTRACT METHOD IMPLEMENTATIONS
    # ============================================================

    def get_document_type(self) -> str:
        return "bank_statement"

    def get_required_fields(self) -> List[str]:
        return [
            'account_holder_name', 'address', 'address_country',
            'account_number', 'bank_name', 'currency'
        ]

    def get_name_field(self) -> str:
        """Bank statement uses 'account_holder_name' as the name field."""
        return 'account_holder_name'

    def should_validate_photoholmes(self) -> bool:
        """Enable PhotoHolmes for bank statements.

        Even though spatial extractor uses PyMuPDF for direct PDF parsing,
        PhotoHolmes forgery detection is important for bank statements to detect tampering.
        """
        return True

    async def extract_fields_from_ocr(
        self, text_blocks: list, raw_text: str, image_bytes: bytes, is_pdf: bool
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Extract bank statement fields using hybrid GLiNER2 + Spatial approach.

        HYBRID STRATEGY:
        1. Always run GLiNER2 extraction (primary, semantic understanding)
        2. Always run Spatial extraction IN FULL (backup, geometry-based)
        3. Merge results at field level:
           - Use GLiNER result if confidence >= threshold (50%)
           - Otherwise use spatial result (if available)
        4. Account numbers always come from spatial (requires label proximity)
        5. Statement date uses post-processing to select the largest/latest date

        IMPORTANT:
        - Spatial extraction runs completely (all fields) because its logic depends
          on analyzing the entire document structure
        - The merge happens as a post-processing step, not by selectively running
          spatial for missing fields

        Args:
            text_blocks: OCR text blocks with geometry (not used by GLINER2, kept for compatibility)
            raw_text: Raw OCR text or PDF-extracted text (used by GLINER2)
            image_bytes: Raw file bytes (used for spatial extraction)
            is_pdf: True if input is a PDF file

        Returns:
            (extracted_data, confidence_data)
        """
        try:
            # Step 1: Always run GLiNER extraction
            self.logger.info("Step 1: Running GLiNER2 extraction...")
            gliner_extractor = get_gliner_bank_statement_extractor()
            gliner_data = await gliner_extractor.extract(
                ocr_text=raw_text,
                text_blocks=text_blocks if text_blocks else None
            )

            self.logger.info(
                f"GLiNER2 extracted: confidence={gliner_data.overall_confidence:.1f}%, "
                f"source={gliner_data.extraction_source}"
            )

            # Step 2: Always run spatial extraction IN FULL (extracts all fields)
            # Spatial must run completely - its logic depends on full document analysis
            self.logger.info("Step 2: Running Spatial extraction (full mode)...")
            spatial_data = await self._extract_spatial_result(image_bytes, is_pdf)

            self.logger.info(
                f"Spatial extracted: confidence={spatial_data.overall_confidence:.1f}%, "
                f"source={spatial_data.extraction_source}"
            )

            # Step 3: Merge results (GLiNER优先，spatial填充缺失字段)
            self.logger.info("Step 3: Merging GLiNER and Spatial results...")
            merged_data = self._merge_extraction_results(
                gliner_data=gliner_data,
                spatial_data=spatial_data,
                confidence_threshold=50.0
            )

            # Log merge summary
            for field in ['account_holder_name', 'bank_name', 'address', 'account_number']:
                field_conf = merged_data.confidence_scores.get(field, {})
                if isinstance(field_conf, dict):
                    sources = field_conf.get('sources', [])
                    conf = field_conf.get('overall_confidence', 0)
                else:
                    sources = ['unknown']
                    conf = float(field_conf) if field_conf else 0
                self.logger.info(f"  {field}: {conf:.1f}% confidence, sources={sources}")

            # Step 4: Apply statement date fix
            largest_date = self._get_largest_date_from_text(raw_text)
            if largest_date:
                old_date = merged_data.statement_date
                merged_data.statement_date = largest_date
                if old_date != largest_date:
                    self.logger.info(f"Statement date updated: {old_date} -> {largest_date} (largest date)")

            return self._build_gliner_response(merged_data, is_pdf)

        except Exception as e:
            self.logger.error(f"Extraction error: {str(e)}, falling back to spatial only")
            return await self._extract_with_spatial(image_bytes, is_pdf, text_blocks)

    async def _extract_spatial_result(self, file_bytes: bytes, is_pdf: bool):
        """Extract using Spatial extractor and return BankStatementData"""
        extractor = SpatialBankStatementExtractor()
        result = await extractor.extract_from_bytes(file_bytes, is_pdf=is_pdf)

        # Set extraction source if not already set
        if not result.extraction_source:
            result.extraction_source = 'spatial_ocr' if not is_pdf else 'spatial_geometry'
        if not result.account_number_extraction_method:
            result.account_number_extraction_method = 'spatial_geometry'

        return result

    async def _extract_with_spatial(
        self, image_bytes: bytes, is_pdf: bool, text_blocks: list
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Extract using Spatial extractor (error fallback only).

        NOTE: This is only used when the primary hybrid extraction flow fails.
        Normal processing uses the hybrid approach in extract_fields_from_ocr.
        """
        extractor = SpatialBankStatementExtractor()
        result = await extractor.extract_from_bytes(image_bytes, is_pdf=is_pdf)

        # Build extracted_data from BankStatementData
        extracted_data = {
            "account_holder_name": result.account_holder_name,
            "account_number": result.account_number,
            "bank_name": result.bank_name,
            "bank_code": result.bank_code,  # IFSC/SWIFT code
            "bank_branch": result.bank_branch,
            "bank_country": result.bank_country,  # Bank's registered country
            "address": result.address,
            # Address components (decomposed from full address)
            "address_city": result.address_city,
            "address_state": result.address_state,
            "address_postal": result.address_postal,
            "address_country": result.address_country,
            "currency": result.currency,
            "statement_date": self._normalize_date_to_iso(result.statement_date) or "",  # ISO format (YYYY-MM-DD)
            "account_number_extraction_method": result.account_number_extraction_method,
            "raw_data": result.raw_data,
        }

        # Fallback: Infer currency from bank_country if not extracted
        if not extracted_data.get("currency") and extracted_data.get("bank_country"):
            from app.helper.validators.bank_statement_validator import get_bank_statement_validator
            validator = get_bank_statement_validator()
            inferred_currency = validator.get_currency_for_country(extracted_data["bank_country"])
            if inferred_currency:
                extracted_data["currency"] = inferred_currency
                self.logger.info(
                    f"Inferred currency '{inferred_currency}' from bank_country '{extracted_data['bank_country']}'"
                )

        # Build confidence_data from BankStatementData
        # Determine extraction source: 'spatial_geometry' for PDF, 'spatial_ocr' for images
        extraction_source = 'spatial_ocr' if not is_pdf else 'spatial_geometry'

        confidence_data = {}
        for field, confidence in result.confidence_scores.items():
            # Map normalized field names to required field names
            normalized_field = self._normalize_field_name(field)
            if normalized_field:
                conf_value = confidence / 100 if confidence > 1 else confidence
                confidence_data[normalized_field] = {
                    'overall_confidence': conf_value,
                    'sources': [extraction_source]
                }

        # Add overall confidence if available
        if result.overall_confidence:
            confidence_data['overall'] = {
                'overall_confidence': result.overall_confidence / 100,
                'sources': [extraction_source]
            }

        # Fallback: Populate missing field confidences using overall_confidence
        # This ensures validation doesn't fail for extracted fields with missing per-field confidence
        # This matches the test script behavior (test_documents.py:220-224)
        if result.overall_confidence:
            required_fields = ['bank_name', 'account_holder_name', 'account_number', 'address', 'currency',
                               'account_holder_country']
            for field in required_fields:
                if field not in confidence_data and extracted_data.get(field):
                    # Field was extracted but has no confidence score - use overall as fallback
                    confidence_data[field] = {
                        'overall_confidence': result.overall_confidence / 100,
                        'sources': [extraction_source]
                    }

        self.logger.info(
            f"Spatial bank statement extraction ({'OCR' if not is_pdf else 'direct'}): "
            f"bank={result.bank_name}, "
            f"account_holder={result.account_holder_name}, "
            f"overall_confidence={result.overall_confidence:.2f}%"
        )

        # Set extraction_method based on actual source
        # For images: 'ocr', for text-based PDFs: 'direct'
        extraction_method = 'ocr' if not is_pdf else 'direct'

        # Check text_blocks metadata for PDFs (may have come from image-based PDF)
        if is_pdf and text_blocks and len(text_blocks) > 0:
            first_block = text_blocks[0]
            if first_block.get('text') == '__EXTRACTION_METADATA__':
                detected_method = first_block.get('extraction_method', 'unknown')
                if detected_method == 'ocr':
                    # This was an image-based PDF processed via OCR
                    extraction_method = 'ocr'
                    self.logger.info("PDF was image-based, processed via OCR")
                else:
                    self.logger.info(f"PDF extraction method: {detected_method}")

        # Add extraction_method to extracted_data for validation
        extracted_data['extraction_method'] = extraction_method

        return extracted_data, confidence_data

    def _has_required_fields(self, bank_data) -> bool:
        """Check if bank data has all required fields"""
        required = ['account_holder_name', 'account_number', 'bank_name', 'address']
        for field in required:
            value = getattr(bank_data, field, None)
            if not value or not str(value).strip():
                return False
        return True

    def _has_required_fields_excluding_account(self, bank_data) -> bool:
        """Check if bank data has all required fields EXCEPT account number.

        NOTE: This method is kept for informational purposes and potential
        fallback scenarios. The primary extraction flow now uses the hybrid
        merge strategy in _merge_extraction_results.

        Account number is always handled via spatial extraction due to
        the need for label proximity detection.
        """
        required = ['account_holder_name', 'bank_name', 'address']  # No 'account_number'
        for field in required:
            value = getattr(bank_data, field, None)
            if not value or not str(value).strip():
                return False
        return True

    def _get_largest_date_from_text(self, raw_text: str) -> Optional[str]:
        """
        Extract all dates from text and return the largest (latest) one.

        This ensures statement_date is always the closing/end date of the statement period,
        not a random date from transaction history or other fields.

        Args:
            raw_text: Raw OCR or PDF text

        Returns:
            Latest date as ISO string (YYYY-MM-DD) or None
        """
        import re
        from datetime import datetime

        # Date patterns to match
        date_patterns = [
            # DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY
            r'\b(\d{2})[\/\-\.](\d{2})[\/\-\.](\d{4})\b',
            # YYYY-MM-DD, YYYY/MM/DD, YYYY.MM.DDD
            r'\b(\d{4})[\/\-\.](\d{2})[\/\-\.](\d{2})\b',
            # DD MMM YYYY, DD-MMM-YYYY (e.g., 31 Dec 2025)
            r'\b(\d{2})\s+([A-Za-z]{3})\s+(\d{4})\b',
            # MMM DD, YYYY (e.g., Dec 31, 2025)
            r'\b([A-Za-z]{3})\s+(\d{2}),?\s+(\d{4})\b',
        ]

        dates_found = []

        for pattern in date_patterns:
            matches = re.finditer(pattern, raw_text)
            for match in matches:
                try:
                    groups = match.groups()
                    parsed_date = None

                    # Handle different formats
                    if len(groups[0]) == 4:  # Year first
                        year, month, day = groups
                        parsed_date = datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d")
                    elif len(groups[2]) == 4:  # Year last
                        if groups[1].isalpha():  # Month is text
                            day, month_str, year = groups
                            parsed_date = datetime.strptime(f"{day} {month_str} {year}", "%d %b %Y")
                        else:  # Day/Month/Year
                            day, month, year = groups
                            parsed_date = datetime.strptime(f"{day}-{month}-{year}", "%d-%m-%Y")

                    if parsed_date:
                        dates_found.append(parsed_date)
                except (ValueError, IndexError):
                    continue

        if dates_found:
            # Return the largest (latest) date
            latest_date = max(dates_found)
            return latest_date.strftime('%Y-%m-%d')

        return None

    def _has_account_number_label(self, text_blocks: List[Dict]) -> bool:
        """Check if document contains an account number label.

        This early detection helps reject documents like credit card statements
        that don't have proper account number labels.

        Args:
            text_blocks: List of text blocks from OCR extraction

        Returns:
            True if an account number label is found, False otherwise
        """
        from app.helper.validators.bank_statement_validator import get_bank_statement_validator

        validator = get_bank_statement_validator()
        account_number_labels = validator.get_account_number_labels()

        for block in text_blocks:
            text = block.get('text', '').strip().upper()
            for label in account_number_labels:
                if re.search(r'\b' + re.escape(label) + r'\b', text):
                    return True
        return False

    def _build_gliner_response(self, bank_data, is_pdf: bool) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Build response from GLiNER extractor result (or hybrid merged result)"""
        # Determine extraction method based on source
        if bank_data.extraction_source == 'hybrid_gliner_spatial':
            extraction_method = 'hybrid_gliner_spatial'
        elif is_pdf:
            extraction_method = 'gliner'
        else:
            extraction_method = 'gliner_ocr'

        extracted_data = {
            "account_holder_name": bank_data.account_holder_name,
            "account_number": bank_data.account_number,
            "bank_name": bank_data.bank_name,
            "bank_code": bank_data.bank_code,
            "bank_branch": bank_data.bank_branch,
            "bank_country": bank_data.bank_country,
            "address": bank_data.address,
            "address_city": bank_data.address_city,
            "address_state": bank_data.address_state,
            "address_postal": bank_data.address_postal,
            "address_country": bank_data.address_country,
            "currency": bank_data.currency,
            "statement_date": self._normalize_date_to_iso(bank_data.statement_date) or "",
            "account_number_extraction_method": bank_data.account_number_extraction_method or "gliner_ner",
            "extraction_method": extraction_method,
            "raw_data": bank_data.raw_data,
        }

        confidence_data = {}
        for field, confidence_info in bank_data.confidence_scores.items():
            # Handle both legacy (float) and new (dict) formats
            if isinstance(confidence_info, dict):
                # Already in new format - just copy
                confidence_data[field] = confidence_info
            elif isinstance(confidence_info, (int, float)):
                # Legacy format - convert to new format
                confidence_data[field] = {
                    'overall_confidence': float(confidence_info) / 100 if confidence_info > 1 else float(confidence_info),
                    'sources': ['gliner']
                }
            else:
                # Unknown format - copy as-is
                confidence_data[field] = confidence_info

        if bank_data.overall_confidence:
            # Determine sources based on extraction_source
            if bank_data.extraction_source == 'hybrid_gliner_spatial':
                sources = ['gliner', 'spatial']
            else:
                sources = ['gliner']

            confidence_data['overall'] = {
                'overall_confidence': bank_data.overall_confidence / 100,
                'sources': sources
            }

        return extracted_data, confidence_data

    def _update_confidence_source(
        self,
        merged_data: 'BankStatementData',
        field: str,
        source_data: 'BankStatementData',
        source_name: str
    ) -> None:
        """Update confidence_scores to track which extraction method provided the field."""
        if field not in merged_data.confidence_scores:
            merged_data.confidence_scores[field] = {}

        # Handle both legacy (float) and new (dict) formats
        current = merged_data.confidence_scores[field]
        if isinstance(current, dict):
            current['overall_confidence'] = current.get('overall_confidence', source_data.overall_confidence or 85.0)
            if 'sources' not in current:
                current['sources'] = []
            if source_name not in current['sources']:
                current['sources'].append(source_name)
        else:
            # Convert legacy float to dict format
            merged_data.confidence_scores[field] = {
                'overall_confidence': float(current) if current else (source_data.overall_confidence or 85.0),
                'sources': [source_name]
            }

    def _merge_extraction_results(
        self,
        gliner_data: 'BankStatementData',
        spatial_data: 'BankStatementData',
        confidence_threshold: float = 50.0
    ) -> 'BankStatementData':
        """
        Merge GLiNER and spatial extraction results.

        For each field:
        - Use GLiNER result if confidence >= threshold
        - Otherwise use spatial result (if available)

        Tracks sources in confidence_scores for debugging.
        """
        merged = gliner_data.model_copy()

        # Fields to check (all extracted fields)
        fields_to_merge = [
            'account_holder_name', 'bank_name', 'address', 'currency',
            'account_number', 'statement_date', 'bank_branch', 'bank_code',
            'bank_country', 'address_city', 'address_state', 'address_country',
            'account_holder_country'
        ]

        for field in fields_to_merge:
            gliner_value = getattr(gliner_data, field, None)
            gliner_conf = gliner_data.confidence_scores.get(field, 0)

            # Handle both dict format (new) and float format (from GLiNER)
            if isinstance(gliner_conf, dict):
                gliner_conf = gliner_conf.get('overall_confidence', 0)
            elif isinstance(gliner_conf, (int, float)):
                # GLiNER returns raw float - use directly
                gliner_conf = float(gliner_conf)

            # Special case: account_number - always use spatial if available
            if field == 'account_number' and spatial_data.account_number:
                setattr(merged, field, spatial_data.account_number)
                merged.account_number_extraction_method = spatial_data.account_number_extraction_method
                self._update_confidence_source(merged, field, spatial_data, 'spatial')
                continue

            # For other fields: use GLiNER if confident, else spatial
            if not gliner_value or gliner_conf < confidence_threshold:
                spatial_value = getattr(spatial_data, field, None)
                if spatial_value:
                    setattr(merged, field, spatial_value)
                    # Also copy related address components when spatial address is used
                    # This ensures consistency: all address components come from the same source
                    if field == 'address':
                        if hasattr(spatial_data, 'address_city') and spatial_data.address_city:
                            merged.address_city = spatial_data.address_city
                        if hasattr(spatial_data, 'address_state') and spatial_data.address_state:
                            merged.address_state = spatial_data.address_state
                        if hasattr(spatial_data, 'address_postal') and spatial_data.address_postal:
                            merged.address_postal = spatial_data.address_postal
                        if hasattr(spatial_data, 'address_country') and spatial_data.address_country:
                            merged.address_country = spatial_data.address_country
                        self.logger.info(
                            f"Copied spatial address components: "
                            f"city={spatial_data.address_city}, "
                            f"state={spatial_data.address_state}, "
                            f"postal={spatial_data.address_postal}, "
                            f"country={spatial_data.address_country}"
                        )
                    self._update_confidence_source(merged, field, spatial_data, 'spatial')
            else:
                # GLiNER value is being used - apply cleaning for address field
                if field == 'address' and gliner_value:
                    # Get address components from GLiNER data for cleaning
                    city = gliner_data.address_city
                    state = gliner_data.address_state
                    postal_code = gliner_data.address_postal
                    country_code = gliner_data.address_country

                    # Clean the address using spatial extractor's cleaning logic
                    cleaned_address = clean_gliner_address(
                        gliner_value,
                        city=city,
                        state=state,
                        postal_code=postal_code,
                        country_code=country_code
                    )
                    setattr(merged, field, cleaned_address)
                    self.logger.info(
                        f"Cleaned GLiNER address: original='{gliner_value[:50]}...', "
                        f"cleaned='{cleaned_address[:50]}...', "
                        f"city={city}, state={state}, postal={postal_code}"
                    )

        # Recalculate overall confidence based on merged results
        merged_confidence = 0.0
        field_count = 0
        for field in ['account_holder_name', 'bank_name', 'address', 'currency']:
            field_conf = merged.confidence_scores.get(field, {})
            if isinstance(field_conf, dict):
                conf = field_conf.get('overall_confidence', 0)
            elif isinstance(field_conf, (int, float)):
                conf = float(field_conf)
            else:
                conf = 0
            if conf > 0:
                merged_confidence += conf
                field_count += 1

        if field_count > 0:
            merged.overall_confidence = merged_confidence / field_count
        else:
            # Fallback to spatial overall confidence if no fields have confidence
            merged.overall_confidence = spatial_data.overall_confidence or 85.0

        # Set extraction source to reflect hybrid approach
        merged.extraction_source = 'hybrid_gliner_spatial'

        return merged

    def _normalize_field_name(self, field: str) -> Optional[str]:
        """Normalize extractor field names to service field names.

        Handles both underscore-cased names (from confidence_scores) and
        space-separated names (from GLiNER2 schema extraction).
        """
        field_lower = field.lower()

        # Direct mappings - support both formats
        mappings = {
            # Underscore format (from confidence_scores)
            'bank_name': 'bank_name',
            'account_holder_name': 'account_holder_name',
            'account_number': 'account_number',
            'customer_address': 'address',
            'address': 'address',
            'ifsc_code': 'bank_code',
            'swift_code': 'bank_code',
            'currency': 'currency',
            'bank_branch': 'bank_branch',
            'branch': 'bank_branch',

            # Space-separated format (from GLiNER2 schema extraction)
            'bank name': 'bank_name',
            'bank or financial institution name': 'bank_name',
            'account holder name': 'account_holder_name',
            'person name or customer name with title or honorific like mr mrs ms dr': 'account_holder_name',
            'account number': 'account_number',
            'customer address': 'address',
            'ifsc code': 'bank_code',
            'swift code': 'bank_code',
        }

        return mappings.get(field_lower)

    def perform_document_specific_validations(
        self, extracted_data: Dict[str, Any], user_identity: Dict[str, Any]
    ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Perform bank statement specific validations.

        Validations:
        - Extraction method tracking (supports both 'direct' for text-based PDFs and 'ocr' for images)
        - Statement age check (max 90 days old)
        - Address components validation
        - Bank SWIFT lookup validation
        - Account number extraction method validation
        - Account number format validation (per currency)
        - Credit card rejection (Luhn check)

        NOTE: Collects ALL validation errors before returning (no fail-fast).
        This provides complete feedback on all issues with the document.

        NOTE: Both text-based PDFs and images (including image-based PDFs) are now supported.
        The extraction method will be 'direct' for text-based PDFs and 'ocr' for images/image-based PDFs.
        """
        from app.helper.validators.bank_statement_validator import get_bank_statement_validator

        validator = get_bank_statement_validator()
        validation_results = {}
        validation_errors = []  # Collect all errors

        # 1. Extraction method tracking - for informational purposes, no validation rejection
        # Both 'direct' (text-based PDF) and 'ocr' (images/image-based PDFs) are now supported
        extraction_method = extracted_data.get('extraction_method', 'unknown')
        extraction_method_valid = True  # Always valid now

        validation_results['extraction_method'] = {
            'method': extraction_method,
            'valid': extraction_method_valid
        }

        # 2. Statement age check
        statement_date = self._parse_statement_date(extracted_data.get('statement_date'))
        statement_age_days = None
        statement_age_valid = False
        max_age_days = getattr(self._get_verification_settings(), 'bank_statement_max_age_days', 90)

        if statement_date:
            statement_age_days = (date.today() - statement_date).days
            statement_age_valid = statement_age_days <= max_age_days

        validation_results['statement_age'] = {
            'days': statement_age_days,
            'valid': statement_age_valid
        }

        if not statement_age_valid and statement_date:
            validation_errors.append(
                f"Statement too old: {statement_age_days} days (max: {max_age_days} days)"
            )

        # 3. Address components validation
        # Use the validator's proper address validation that respects country-specific rules
        # (e.g., UAE and Gulf countries don't require postal codes)
        street = extracted_data.get('address', '')
        city = extracted_data.get('address_city', '')
        postal_code = extracted_data.get('address_postal', '')
        country = extracted_data.get('address_country', '')
        country_code = country  # Pass country code for optional postal code check

        # Build components dict for the validator
        address_components = {
            'street': street,
            'street_name': street,  # Used by validator as fallback
            'city': city,
            'postal_code': postal_code,
            'state': extracted_data.get('address_state', ''),
            'country': country_code,
        }

        # Use validator's address validation which respects country-specific config
        address_valid, missing_components = validator.validate_address_components(address_components)

        validation_results['address_components'] = {
            'valid': address_valid,
            'components': address_components,
            'missing': missing_components
        }

        if not address_valid:
            validation_errors.append(
                f"Address: Missing address components: {', '.join(missing_components)}"
            )

        # 4. Bank SWIFT lookup validation
        # Get country from multiple sources in priority order:
        # 1. bank_country from document extraction (most reliable - bank's registered country)
        # 2. Bank name lookup from JSON (fallback - default country)
        # 3. Currency config (last resort - may be wrong for multi-currency accounts)
        bank_name = extracted_data.get('bank_name', '')
        currency = extracted_data.get('currency', '')
        bank_country = extracted_data.get('bank_country')  # From document extraction

        # If bank_country not extracted from document, try JSON lookup
        if not bank_country and bank_name:
            from app.core.key_injection.bank_lookup import get_country_for_bank
            bank_country = get_country_for_bank(bank_name)

        # Last resort: infer from currency (may be wrong for multi-currency accounts)
        if not bank_country and currency:
            currency_info = validator.get_currency_info(currency)
            if currency_info:
                bank_country = currency_info.get('country', '')
                self.logger.warning(
                    f"Using currency-derived country '{bank_country}' for bank lookup. "
                    f"This may be incorrect for multi-currency accounts."
                )

        # Validate bank with determined country
        bank_found, swift_or_error = validator.validate_bank_in_swift_codes(bank_name, bank_country)

        validation_results['bank_lookup'] = {
            'valid': bank_found,
            'bank_name': bank_name,
            'country': bank_country,
            'swift_code': swift_or_error if bank_found else None,
            'error': swift_or_error if not bank_found else None
        }

        if not bank_found:
            validation_errors.append(swift_or_error)

        # 5. Account number extraction method validation
        account_number = extracted_data.get('account_number', '')
        extraction_method = extracted_data.get('account_number_extraction_method', '')

        # Account number MUST be extracted via validated methods
        # Accept: spatial_label (legacy), spatial_geometry (spatial extractor), gliner_ner (GLINER2)
        # This prevents random numbers from being accepted as account numbers
        valid_methods = {'spatial_label', 'spatial_geometry', 'gliner_ner'}
        extraction_valid = extraction_method in valid_methods
        validation_results['account_extraction'] = {
            'method': extraction_method,
            'valid': extraction_valid
        }

        if not extraction_valid:
            if not account_number:
                validation_errors.append(
                    "Account number: No account number found with nearby label (e.g., 'Account Number:', 'A/C No:')"
                )
            else:
                validation_errors.append(
                    f"Account number: Extracted via '{extraction_method}' method, but requires spatial extraction method (with nearby label)"
                )

        # 6. Account number format validation (including credit card check)
        if account_number:
            currency = extracted_data.get('currency', '')
            account_valid, account_error = validator.validate_account_number(account_number, currency)

            validation_results['account_number'] = {
                'value': account_number[:4] + '****' if account_number else None,  # Masked for privacy
                'valid': account_valid,
                'error': account_error
            }

            if not account_valid:
                validation_errors.append(account_error)
        else:
            validation_results['account_number'] = {
                'value': None,
                'valid': False,
                'error': 'Account number is empty'
            }
            # Only add error if not already added by extraction validation
            if not any('Account number' in err for err in validation_errors):
                validation_errors.append('Account number is empty')

        # Return combined results
        if validation_errors:
            combined_error = '; '.join(validation_errors)

            # Determine most appropriate error code based on validation errors
            error_code = DocumentErrorCode.PROCESSING_ERROR
            for error in validation_errors:
                if 'too old' in error.lower():
                    error_code = DocumentErrorCode.BANK_STATEMENT_TOO_OLD
                    break
                elif 'missing address components' in error.lower():
                    error_code = DocumentErrorCode.BANK_STATEMENT_ADDRESS_INCOMPLETE
                    break
                elif 'not recognized' in error.lower() or 'not found in swift' in error.lower():
                    error_code = DocumentErrorCode.BANK_STATEMENT_BANK_NOT_RECOGNIZED
                    break
                elif 'invalid length' in error.lower():
                    error_code = DocumentErrorCode.BANK_STATEMENT_ACCOUNT_FORMAT_INVALID
                    break
                elif 'masked' in error.lower():
                    error_code = DocumentErrorCode.BANK_STATEMENT_ACCOUNT_MASKED
                    break
                elif 'credit card' in error.lower():
                    error_code = DocumentErrorCode.BANK_STATEMENT_CREDIT_CARD_DETECTED
                    break
                elif 'account number' in error.lower():
                    error_code = DocumentErrorCode.BANK_STATEMENT_ACCOUNT_FORMAT_INVALID
                    break

            return False, combined_error, {
                'validation_results': validation_results,
                'error_code': error_code
            }

        # All validations passed
        return True, None, {'validation_results': validation_results}

    def _get_failed_field_names(self, validation_results: dict) -> set:
        """Get the set of field names that failed validation.

        Maps validation result keys to actual extracted_data field names.
        Only filters fields with data quality issues, not business rule failures.

        Business rule failures (e.g., statement too old) do NOT remove the field.
        Data quality issues (e.g., invalid format, not found) DO remove the field.

        Args:
            validation_results: Validation results from document_specific validations

        Returns:
            Set of field names that should be removed from extracted_data
        """
        failed_fields = set()

        doc_specific = validation_results.get('document_specific', {})

        # Account number validation failed (data quality issue - format/length)
        if 'account_number' in doc_specific and not doc_specific['account_number'].get('valid', True):
            failed_fields.add('account_number')

        # Bank lookup validation failed (data quality issue - bank not recognized)
        if 'bank_lookup' in doc_specific and not doc_specific['bank_lookup'].get('valid', True):
            failed_fields.add('bank_name')
            failed_fields.add('bank_code')
            failed_fields.add('bank_branch')

        # Address components validation failed (data quality issue - missing components)
        if 'address_components' in doc_specific and not doc_specific['address_components'].get('valid', True):
            failed_fields.add('address')
            failed_fields.add('address_city')
            failed_fields.add('address_state')
            failed_fields.add('address_country')
            failed_fields.add('address_postal_code')

        # NOTE: statement_age is a business rule (max 90 days), NOT a data quality issue.
        # We keep the statement_date even if it's too old - the rejection is for age, not extraction.

        return failed_fields

    def _filter_failed_validation_fields(self, extracted_data: dict, validation_results: dict) -> dict:
        """Remove fields that failed validation from extracted_data.

        Args:
            extracted_data: The raw extracted data
            validation_results: Validation results containing which fields failed

        Returns:
            Filtered extracted_data with failed fields removed
        """
        failed_fields = self._get_failed_field_names(validation_results)

        filtered_data = {}
        for key, value in extracted_data.items():
            if key not in failed_fields:
                filtered_data[key] = value

        return filtered_data

    def should_increment_state(self) -> bool:
        """Bank statement processing increments state from 2 to 3."""
        return True

    # ============================================================
    # BANK STATEMENT SPECIFIC METHODS
    # ============================================================

    def _build_extracted_data_from_result(
        self, line_result, raw_text: str = "", text_blocks: list = None
    ) -> Dict[str, Any]:
        """
        Build extracted_data dict from LineExtractionResult (from simple_bank_analyzer).

        Args:
            line_result: LineExtractionResult from simple_bank_analyzer.analyze_bank_statement()
            raw_text: Raw OCR text for bank/currency detection
            text_blocks: OCR blocks with geometry for bank address detection

        Returns:
            Dict with extracted fields
        """
        from app.core.key_injection.bank_lookup import detect_bank_in_text, BankInfo
        from app.core.key_injection.global_banks import (
            detect_currency_in_text, detect_country_in_text
        )

        # Get base fields from LineExtractionResult
        account_holder_name = line_result.account_holder_name if line_result else None
        account_number = line_result.account_number if line_result else None
        statement_date = line_result.statement_date if line_result else None
        address = line_result.account_holder_address if line_result else None
        bank_name = line_result.bank_name if line_result else None
        bank_swift_code = None  # Not available in LineExtractionResult
        bank_country = None

        # Try to detect bank from text if not from LineExtractionResult
        # Use new BankLookup with comprehensive bank configuration
        bank_info = detect_bank_in_text(raw_text) if raw_text else None

        if bank_info:
            self.logger.info(f"Bank detected via BankLookup: {bank_info.abbreviation} ({bank_info.full_name})")

            # Try to extract bank address from blocks below bank name for country detection
            if text_blocks:
                bank_address_text = simple_bank_analyzer.extract_bank_address(text_blocks, bank_info.abbreviation)
                if bank_address_text:
                    detected_country = detect_country_in_text(bank_address_text)
                    if detected_country:
                        bank_country = detected_country
                        self.logger.info(f"Bank country detected from bank address: {bank_country}")

            # Use detected country or bank_info's default country
            if not bank_country:
                bank_country = bank_info.country

            # Get BankInfo with specific country if we have it
            if bank_country:
                from app.core.key_injection.bank_lookup import lookup_bank_by_name
                bank_info_with_country = lookup_bank_by_name(bank_info.abbreviation, bank_country)
                if bank_info_with_country:
                    bank_info = bank_info_with_country

            if not bank_name:
                bank_name = bank_info.full_name
            if not bank_swift_code and bank_info.swift_codes:
                bank_swift_code = bank_info.swift_codes[0]
            self.logger.info(f"Bank resolved: {bank_name} (SWIFT: {bank_swift_code})")

        # Detect currency from text
        currency = line_result.account_currency if line_result else None
        if not currency and raw_text:
            currency = detect_currency_in_text(raw_text)
        if currency:
            self.logger.info(f"Currency detected: {currency}")

        # Infer client country from client address
        client_country = None
        if line_result and line_result.account_holder_country:
            client_country = line_result.account_holder_country
            self.logger.info(f"Client country from extraction result: {client_country}")
        elif address:
            client_country = detect_country_in_text(address)
            if not client_country:
                client_country = self._infer_country_from_address(address)

        return {
            "account_holder_name": account_holder_name,
            "account_number": account_number,
            "bank_name": bank_name,
            "bank_swift_code": bank_swift_code,
            "bank_country": bank_country,
            "address": address,
            "statement_date": self._normalize_date_to_iso(statement_date) or '',  # ISO format (YYYY-MM-DD)
            "opening_balance": None,  # Not extracted by simple_bank_analyzer
            "closing_balance": None,  # Not extracted by simple_bank_analyzer
            "currency": currency,
            "country": client_country
        }

    def _parse_statement_date(self, date_str: Optional[str]) -> Optional[date]:
        """Parse statement date string to date object."""
        if not date_str:
            return None

        try:
            date_clean = str(date_str).strip()

            date_formats = [
                "%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d", "%d/%m/%Y", "%Y%m%d",
                "%d %b %Y",   # 30 NOV 2025
                "%d %B %Y",   # 30 November 2025
            ]

            for date_variant in [date_clean, date_clean.title()]:
                for fmt in date_formats:
                    try:
                        from datetime import datetime
                        return datetime.strptime(date_variant, fmt).date()
                    except ValueError:
                        continue

            return None
        except Exception:
            return None

    def _normalize_date_to_iso(self, date_str: Optional[str]) -> Optional[str]:
        """
        Normalize a date string to ISO format (YYYY-MM-DD).

        Args:
            date_str: Date string in any supported format

        Returns:
            Date in ISO format (YYYY-MM-DD) or None if parsing fails
        """
        parsed_date = self._parse_statement_date(date_str)
        if parsed_date:
            return parsed_date.strftime('%Y-%m-%d')
        return None

    def _infer_country_from_address(self, address: str) -> Optional[str]:
        """Infer ISO 3166-1 alpha-2 country code from address text."""
        if not address:
            return None

        addr_lower = address.lower()

        # Country patterns (order matters - more specific first)
        country_patterns = {
            "SG": ["singapore"],
            "US": ["united states", "usa", "u.s.a", "u.s.a."],
            "GB": ["united kingdom", "england", "scotland", "wales", "northern ireland", "britain", ", uk"],
            "DE": ["germany", "deutschland"],
            "HK": ["hong kong"],
            "AU": ["australia"],
            "MY": ["malaysia"],
            "TH": ["thailand"],
            "IN": ["india"],
            "JP": ["japan"],
            "CN": ["china", "中国"],
            "KR": ["korea", "south korea"],
            "ID": ["indonesia"],
            "PH": ["philippines"],
            "VN": ["vietnam", "viet nam"],
            "TW": ["taiwan"],
            "NZ": ["new zealand"],
            "CA": ["canada"],
            "FR": ["france"],
            "IT": ["italy", "italia"],
            "ES": ["spain", "españa"],
            "NL": ["netherlands", "holland"],
            "CH": ["switzerland", "suisse", "schweiz"],
            "AE": ["united arab emirates", "uae", "dubai", "abu dhabi"],
            "SA": ["saudi arabia"],
            "BR": ["brazil", "brasil"],
            "MX": ["mexico", "méxico"],
        }

        for code, patterns in country_patterns.items():
            if any(p in addr_lower for p in patterns):
                return code

        return None

    def _get_verification_settings(self):
        """Get verification settings - lazy import to avoid circular dependency."""
        from app.config.verification_config import verification_settings
        return verification_settings

    # ============================================================
    # SHARED VALIDATION (used by both endpoint and test script)
    # ============================================================

    @staticmethod
    def validate_bank_statement(
        account_holder_name: str,
        bank_name: str,
        currency: str,
        address: str,
        overall_confidence: float
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Validate bank statement extraction results.

        Shared validation logic used by both:
        - The endpoint (SequentialBankStatementService)
        - The test script (scripts/test_documents.py)

        Args:
            account_holder_name: Extracted account holder name
            bank_name: Extracted bank name
            currency: Extracted currency code
            address: Extracted address
            overall_confidence: Overall confidence score (0-100)

        Returns:
            (is_valid, missing_fields, field_validations)
        """
        from app.config.verification_config import verification_settings

        # Build extracted_data and confidence_data in the format expected by validation
        extracted_data = {
            'account_holder_name': account_holder_name,
            'bank_name': bank_name,
            'currency': currency,
            'address': address,
        }

        confidence_data = {
            'overall': {'overall_confidence': overall_confidence / 100},  # Convert to 0-1 range
        }

        # Use the base class validation method
        service = SequentialBankStatementService()
        return service.validate_required_fields_with_confidence(extracted_data, confidence_data)

    # ============================================================
    # PUBLIC ENTRY POINT
    # ============================================================

    async def process_bank_statement(
        self, client_public_key: str, file_data: str, filename: str,
        callback_url: Optional[str] = None
    ):
        """
        Process a bank statement document.

        Uses validate_from_file() for consistent behavior with test script.
        This avoids the base class pipeline that was causing postal code extraction issues.
        """
        import base64
        import time
        import uuid
        from app.repositories.user_key_repository import UserKeyRepository
        from app.repositories.user_identity_repository import UserIdentityRepository
        from app.dto.verification_session import SequentialJobResponse

        job_id = f"bank_statement_{uuid.uuid4().hex[:12]}"
        start_time = time.time()

        # ============================================================
        # PHASE 1: State validation
        # ============================================================
        is_valid, error_msg, is_resubmission = self.state_service.validate_document_submission(
            client_public_key, "bank_statement"
        )
        if not is_valid:
            self.logger.error(f"State validation failed: {error_msg}")
            current_state = self.state_service.get_verification_state(client_public_key)
            current_seq = self.state_service.get_sequence_no(client_public_key)
            return SequentialJobResponse(
                result=False,
                job_id=job_id,
                verification_state=current_state,
                sequence_no=current_seq,
                processing_time_seconds=0.0,
                error=error_msg,
                error_code=DocumentErrorCode.INVALID_STATE
            )

        # ============================================================
        # PHASE 2: Get user_identity_id (required for state management)
        # ============================================================
        user_key_repo = UserKeyRepository()
        user_key = user_key_repo.get_key_by_public_key(client_public_key)
        if not user_key or not user_key.get('user_identity_id'):
            current_state = self.state_service.get_verification_state(client_public_key)
            current_seq = self.state_service.get_sequence_no(client_public_key)
            return SequentialJobResponse(
                result=False,
                job_id=job_id,
                verification_state=current_state,
                sequence_no=current_seq,
                processing_time_seconds=0.0,
                error="User identity not found. Complete previous steps first.",
                error_code=DocumentErrorCode.USER_NOT_FOUND
            )

        user_identity_id = user_key['user_identity_id']
        user_identity_repo = UserIdentityRepository()
        user_identity = user_identity_repo.get_user_by_id(user_identity_id)

        if not user_identity:
            current_state = self.state_service.get_verification_state(client_public_key)
            current_seq = self.state_service.get_sequence_no(client_public_key)
            return SequentialJobResponse(
                result=False,
                job_id=job_id,
                verification_state=current_state,
                sequence_no=current_seq,
                processing_time_seconds=0.0,
                error="User identity record not found",
                error_code=DocumentErrorCode.USER_NOT_FOUND
            )

        # ============================================================
        # PHASE 3: Decode file data and validate using SINGLE ENTRY POINT
        # ============================================================
        image_bytes = base64.b64decode(file_data)

        # Use the SINGLE ENTRY POINT - same as test script
        # This ensures exact same behavior between API and tests
        validation_result = await SequentialBankStatementService.extract_from_file(
            file_bytes=image_bytes,
            filename=filename
        )

        # ============================================================
        # PHASE 4: Build response from validation result
        # ============================================================
        processing_time = time.time() - start_time

        if not validation_result['success']:
            # Validation failed - return error response
            current_state = self.state_service.get_verification_state(client_public_key)
            current_seq = self.state_service.get_sequence_no(client_public_key)

            return SequentialJobResponse(
                result=False,
                job_id=job_id,
                verification_state=current_state,
                sequence_no=current_seq,
                processing_time_seconds=round(processing_time, 2),
                extracted_data=validation_result.get('extracted_data'),
                other_checks=validation_result.get('validation_results'),
                error=validation_result.get('error_message', 'Validation failed'),
                error_code=validation_result.get('error_code', DocumentErrorCode.VALIDATION_FAILED)
            )

        # ============================================================
        # PHASE 5: Validation passed - handle name matching and state increment
        # ============================================================
        extracted_data = validation_result['extracted_data']
        stored_full_name = user_identity.get('full_name')

        # Name matching validation (gatekeeper)
        name_match_valid, name_match_score, name_match_details = \
            self.validate_name_matching(stored_full_name, extracted_data)

        if not name_match_valid:
            error_msg = (
                f"Name matching failed: score={name_match_score}%, "
                f"threshold={getattr(self._get_verification_settings(), 'name_match_threshold', 70)}%"
            )
            self.logger.error(error_msg)

            current_state = self.state_service.get_verification_state(client_public_key)
            current_seq = self.state_service.get_sequence_no(client_public_key)
            return SequentialJobResponse(
                result=False,
                job_id=job_id,
                verification_state=current_state,
                sequence_no=current_seq,
                processing_time_seconds=round(processing_time, 2),
                extracted_data=extracted_data,
                other_checks={
                    **validation_result.get('validation_results', {}),
                    **name_match_details
                },
                error=error_msg,
                error_code=DocumentErrorCode.NAME_MATCH_FAILED
            )

        # ============================================================
        # PHASE 6: State increment (if applicable)
        # ============================================================
        current_state = self.state_service.get_verification_state(client_public_key)
        current_seq = self.state_service.get_sequence_no(client_public_key)
        new_state = current_state
        new_seq = current_seq

        # Only increment if at the expected state (not resubmission)
        expected_state = self.state_service.EXPECTED_STATE.get("bank_statement")
        expected_states = expected_state if isinstance(expected_state, list) else [expected_state]
        if current_state in expected_states:
            new_state = 3
            new_seq = 3
            # Update state in BOTH user_keys (per-device) AND user_identity_index (overall)
            self.user_key_repo.update_state_and_sequence(
                user_public_key=client_public_key,
                verification_state=new_state,
                sequence_no=new_seq
            )
            # Update user_identity_index to match (use SET, not INCREMENT)
            self.user_identity_repo.set_verification_state(user_identity_id, new_state)
            self.user_identity_repo.set_sequence_no(user_identity_id, new_seq)
            self.logger.info(
                f"Bank statement processing completed. "
                f"New state: {new_state}, sequence_no: {new_seq}"
            )
        else:
            self.logger.info(
                f"Bank statement resubmission (state={current_state}). "
                f"State unchanged."
            )

        # ============================================================
        # PHASE 7: Build successful response
        # ============================================================
        return SequentialJobResponse(
            result=True,
            job_id=job_id,
            verification_state=new_state,
            sequence_no=new_seq,
            processing_time_seconds=round(processing_time, 2),
            extracted_data=extracted_data,
            other_checks={
                **validation_result.get('validation_results', {}),
                **name_match_details,
                'is_resubmission': is_resubmission
            },
            user_identity_id=user_identity_id
        )

    # ============================================================
    # UNIFIED VALIDATION ENTRY POINT (for both API and test script)
    # ============================================================

    @staticmethod
    async def extract_from_file(
        file_bytes: bytes,
        filename: str
    ) -> Dict[str, Any]:
        """
        SINGLE ENTRY POINT for bank statement extraction.

        This is the ONLY method that should be called for bank statement processing.
        Both the API endpoint and test script MUST use this method to ensure
        consistent behavior.

        This method:
        1. Extracts all fields using GLINER2 (primary) with Spatial fallback
           - GLINER2: Semantic understanding, 50%+ confidence threshold
           - Spatial: PyMuPDF geometry-based, 60-85% accuracy for structured documents
        2. Validates required fields
        3. Performs bank statement specific validations

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
        service = SequentialBankStatementService()
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
        Unified validation entry point for bank statements.

        This method does everything from file bytes to full validation:
        1. Text extraction (direct PDF or OCR)
        2. Field extraction using GLINER2 (primary) with Spatial fallback
           - GLINER2: Semantic understanding, 50%+ confidence threshold
           - Spatial: PyMuPDF geometry-based, 60-85% accuracy for structured documents
        3. Required fields validation
        4. Document-specific validations

        Supports:
        - Text-based PDFs (direct extraction)
        - Image-based PDFs (OCR extraction)
        - Regular images (JPG, PNG, etc. via OCR)

        Both the test script and API should use this method to ensure
        consistent validation behavior.

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
        from dataclasses import dataclass
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
            # Uses intelligent routing: direct PDF text extraction for text-based PDFs, OCR for images
            # This is the EXACT SAME CODE PATH used by both the API and test script
            self.logger.info("=" * 80)
            self.logger.info("UNIFIED BANK STATEMENT EXTRACTION PATH")
            self.logger.info("Both API and test script use this exact code path")
            self.logger.info(f"Processing {'PDF' if is_pdf else 'image'}: "
                           f"{'direct extraction' if is_pdf else 'OCR extraction'}")
            self.logger.info("=" * 80)

            ocr = DocumentTextExtractor()
            text_blocks = await ocr.extract_text_with_geometry_enhanced(
                file_bytes, is_pdf=is_pdf, max_pages=1, document_type=DocumentType.BANK_STATEMENT
            )

            self.logger.info(f"Text extraction complete: {len(text_blocks)} text blocks extracted")

            if not text_blocks:
                result['error_message'] = "Text extraction failed: no text found"
                result['elapsed_seconds'] = time.time() - start_time
                return result

            raw_text = "\n".join([block.get('text', '') for block in text_blocks])

            # Early check: Verify document contains an account number label
            # This rejects credit card statements and other unsupported documents early
            if not self._has_account_number_label(text_blocks):
                self.logger.warning("No account number label found in document - rejecting as unsupported document type")
                result['error_message'] = "Document rejected: No account number label found. This may be a credit card statement or unsupported document type."
                result['elapsed_seconds'] = time.time() - start_time
                return result

            # Step 2: Field extraction using GLINER2 with Spatial fallback
            # GLINER2 provides semantic understanding of bank statement fields
            # Falls back to Spatial extractor (PyMuPDF geometry-based) if confidence < 50%
            extracted_data, confidence_data = await self.extract_fields_from_ocr(
                text_blocks=text_blocks,
                raw_text=raw_text,
                image_bytes=file_bytes,
                is_pdf=is_pdf
            )

            # Log extraction method used
            extraction_method = extracted_data.get('extraction_method', 'unknown')
            self.logger.info(f"Extraction method: {extraction_method}")

            # Fallback: Infer currency from bank_country if not extracted
            if not extracted_data.get("currency") and extracted_data.get("bank_country"):
                from app.helper.validators.bank_statement_validator import get_bank_statement_validator
                validator = get_bank_statement_validator()
                inferred_currency = validator.get_currency_for_country(extracted_data["bank_country"])
                if inferred_currency:
                    extracted_data["currency"] = inferred_currency
                    self.logger.info(
                        f"Inferred currency '{inferred_currency}' from bank_country '{extracted_data['bank_country']}'"
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
                # Filter out fields that failed validation from extracted_data
                result['extracted_data'] = self._filter_failed_validation_fields(
                    result['extracted_data'],
                    result['validation_results']
                )
                result['elapsed_seconds'] = time.time() - start_time
                return result

            # All validations passed
            result['success'] = True
            result['elapsed_seconds'] = time.time() - start_time
            return result

        except Exception as e:
            self.logger.error(f"Bank statement validation failed: {e}")
            result['error_message'] = f"Processing error: {str(e)}"
            result['elapsed_seconds'] = time.time() - start_time
            return result


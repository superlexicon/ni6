"""
Sequential Bank Statement Service - Handles bank statement processing in sequential mode.

REFACTORED: Now uses DocumentProcessorBase for unified validation pipeline.
Uses QWEN3-VL DIRECT EXTRACTION for bank statement field extraction.

NEW EXTRACTION STRATEGY:
1. Qwen3-VL extracts bank metadata (bank_name, bank_country, bank_address) + all fields + confidence directly from image
2. Post-processing: Look up SWIFT code from database based on extracted bank metadata
3. Address component parsing for structured address data

This approach replaces the complex multi-stage pipeline:
- REMOVED: DocTR OCR + GLiNER2 + Layout Cache + Spatial Coordinate Extraction
- ADDED: Direct Qwen3-VL extraction with confidence scores

Benefits:
- Simpler architecture with single extraction call
- No layout database maintenance
- No coordinate transformations
- Direct field extraction from image with confidence scores
- Bank metadata explicitly extracted for database lookup
- SWIFT code enrichment via database lookup

State is tracked via verification_state column in user_identity_index.
"""

import asyncio
import hashlib
import io
import json
import re
from typing import Dict, Any, Optional, Tuple, List
from datetime import date
import numpy as np
from PIL import Image
from app.services.sequential_document_processor_base import DocumentProcessorBase
# Legacy: simple_bank_analyzer removed - GLiNER functionality replaced with Qwen3-VL
# from app.core.key_injection.simple_bank_analyzer import simple_bank_analyzer
from app.core.key_injection.key_injection_manager import key_injection_manager
from app.core.key_injection import DocumentType
from app.core.key_injection.global_banks import COUNTRY_NAMES, CURRENCY_COUNTRIES, detect_country_in_text
from app.dto import DocumentErrorCode
from app.core.logger import get_logger


def decompose_address_simple(address_text: str, country_hint: str = None) -> dict:
    """
    Decompose address into components using simple list matching.

    Approach:
    1. Extract country by matching against country name mapping
    2. Extract state by matching against states (using _load_country_locations)
    3. Extract city by matching against cities (using _load_country_locations)
    4. Remaining text is the street address

    Args:
        address_text: Full address string
        country_hint: Optional country code to filter states/cities

    Returns:
        dict with keys: country, state, city, postal_code, street_address
    """
    from app.core.key_injection.bank_database_lookup import _COUNTRY_NAME_TO_ISO
    import re

    if not address_text:
        return {}

    # Clean up the address
    address = address_text.strip()
    address_upper = address.upper()

    result = {
        'country': None,
        'state': None,
        'city': None,
        'postal_code': None,
        'street_address': address  # Default to full address
    }

    # Step 1: Extract postal code first (to remove it from text)
    postal_patterns = [
        r'\b(\d{6})\b',  # 6-digit (SG, IN)
        r'\b(\d{5})\b',  # 5-digit (MY, TH, US)
        r'\b(\d{4})\b',  # 4-digit (AU, NZ)
    ]
    for pattern in postal_patterns:
        match = re.search(pattern, address)
        if match:
            result['postal_code'] = match.group(1)
            # Remove postal code from address for further processing
            address = re.sub(pattern, '', address).strip()
            address_upper = address.upper()
            break

    # Step 2: Extract country by matching against country name mapping
    # Sort by length (longest first) to match multi-word countries first
    sorted_countries = sorted(_COUNTRY_NAME_TO_ISO.keys(), key=len, reverse=True)
    for country_name in sorted_countries:
        country_name_upper = country_name.upper()
        if country_name_upper in address_upper:
            result['country'] = _COUNTRY_NAME_TO_ISO[country_name]
            # Remove country from address for further processing
            address = re.sub(r'\b' + re.escape(country_name) + r'\b', '', address, flags=re.IGNORECASE).strip()
            address_upper = address.upper()
            break

    # Use extracted country or hint
    country_code = result.get('country') or country_hint

    # Remaining text is the street address
    result['street_address'] = address.strip()

    return result


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

        PhotoHolmes forgery detection is important for bank statements to detect tampering.
        This is used regardless of the extraction method (Qwen3-VL).
        """
        return True

    async def extract_fields_from_ocr(
        self, text_blocks: list, raw_text: str, image_bytes: bytes, is_pdf: bool
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        QWEN3-VL DIRECT EXTRACTION: Extract bank statement fields using Qwen3-VL vision model.

        This method replaces the complex multi-stage pipeline (DocTR OCR + GLiNER2 + Layout Cache + Spatial Coordinate Extraction - all removed)
        with a single vision LLM call that extracts field values directly from the image.

        NEW ARCHITECTURE:
        1. Qwen3-VL extracts bank metadata + all fields + confidence directly from image
        2. Post-processing: Look up SWIFT code from database based on extracted bank metadata
        3. Address component parsing for structured address data

        Args:
            text_blocks: OCR text blocks with geometry (kept for interface compatibility, not used)
            raw_text: Raw OCR text or PDF-extracted text (kept for interface compatibility, not used)
            image_bytes: JPEG image data (already preprocessed to meet Qwen3-VL requirements)
            is_pdf: True if input is a PDF file

        Returns:
            (extracted_data, confidence_data) tuple with:
            - extracted_data: Dict with field values including bank metadata
            - confidence_data: Dict with per-field confidence scores
        """
        from app.services.extractors.qwen_bank_statement_extractor import get_qwen_bank_statement_extractor
        from app.services.helpers.address_component_parser import get_address_component_parser

        try:
            if not image_bytes:
                self.logger.error("No image bytes provided for Qwen3-VL extraction")
                return self._build_error_response("No image data provided")

            self.logger.info("QWEN3-VL DIRECT EXTRACTION: Starting direct field extraction...")

            # Step 1: Extract with Qwen3-VL
            qwen_extractor = get_qwen_bank_statement_extractor()
            extracted_data, confidence_data = await qwen_extractor.extract_fields(
                image_bytes=image_bytes,
                bank_name_hint=None,  # Let Qwen3-VL detect the bank
                country_hint=None
            )

            if not extracted_data or "error" in confidence_data:
                error_msg = confidence_data.get("error", "Unknown extraction error")
                self.logger.error(f"Qwen3-VL extraction failed: {error_msg}")
                return self._build_error_response(f"Qwen3-VL extraction failed: {error_msg}")

            self.logger.info(
                f"Qwen3-VL extracted {len(extracted_data)} fields: "
                f"{', '.join([k for k in extracted_data.keys() if extracted_data[k].get('value')])}"
            )

            # Step 2: Parse address components if customer_address was extracted
            customer_address = extracted_data.get("customer_address", {}).get("value")
            bank_country = extracted_data.get("bank_country", {}).get("value")

            if customer_address and bank_country:
                self.logger.info(f"Parsing address components for country: {bank_country}")
                parser = get_address_component_parser()
                address_components = parser.parse_address(
                    address=customer_address,
                    country_hint=bank_country
                )

                # Merge address components into extracted_data
                # Skip 'street_address' since Qwen's customer_address is more reliable
                # Parser returns address_* field names directly (address_city, address_state, etc.)
                for component_name, component_data in address_components.items():
                    if component_data and component_name != 'street_address':  # Skip street_address
                        extracted_data[component_name] = component_data

                self.logger.info(
                    f"Parsed address components: "
                    f"{', '.join([k for k in address_components.keys() if address_components[k]])}"
                )

            # Step 3: Build standardized response format
            # Convert from new format to the format expected by the pipeline
            standardized_data = self._standardize_qwen_response(extracted_data, confidence_data, is_pdf)

            self.logger.info(
                f"Qwen3-VL direct extraction completed. "
                f"Method: qwen3_vl_direct, "
                f"Fields extracted: {len([k for k in standardized_data.keys() if k not in ['extraction_method', 'is_pdf']])}"
            )

            # Add bank code/SWIFT code lookup (enrichment via database)
            if 'bank_name' in standardized_data and 'bank_country' in standardized_data:
                from app.core.key_injection.bank_database_lookup import get_swift_code_for_bank
                bank_name = standardized_data.get('bank_name')
                bank_country = standardized_data.get('bank_country')

                # Look up SWIFT code
                swift_code = get_swift_code_for_bank(bank_name, bank_country)
                if swift_code:
                    standardized_data['bank_code'] = swift_code
                    # Don't override swift_code if Qwen already extracted it
                    if 'swift_code' not in standardized_data:
                        standardized_data['swift_code'] = swift_code
                    confidence_data['bank_code'] = {
                        'overall_confidence': 0.8,  # High confidence for database lookup
                        'sources': ['qwen3_vl_direct']
                    }
                    if 'swift_code' not in confidence_data:
                        confidence_data['swift_code'] = {
                            'overall_confidence': 0.8,
                            'sources': ['qwen3_vl_direct']
                        }

            return standardized_data, confidence_data

        except Exception as e:
            self.logger.error(f"Qwen3-VL extraction error: {str(e)}", exc_info=True)
            return self._build_error_response(f"Extraction error: {str(e)}")

    def _standardize_qwen_response(
        self,
        extracted_data: Dict[str, Any],
        confidence_data: Dict[str, float],
        is_pdf: bool
    ) -> Dict[str, Any]:
        """
        Standardize Qwen3-VL response format to match pipeline expectations.

        Converts from the new format:
        {"field_name": {"value": "...", "confidence": 0.9, "source": "..."}}
        To the format expected by downstream validation.

        Args:
            extracted_data: Raw extracted data from Qwen3-VL
            confidence_data: Per-field confidence scores
            is_pdf: Whether input was PDF

        Returns:
            Standardized extracted data dictionary
        """
        standardized = {
            'extraction_method': 'qwen3_vl_direct',
            'account_number_extraction_method': 'qwen3_vl_direct',
            'is_pdf': is_pdf
        }

        # Map extracted fields to standard names
        # Include fields from Qwen3-VL extraction and address component parsing
        field_mapping = {
            'bank_name': 'bank_name',
            'bank_country': 'bank_country',
            'bank_address': 'bank_address',  # Renamed from bank_branch
            'account_holder_name': 'account_holder_name',
            'customer_address': 'address',
            'account_number': 'account_number',
            'currency': 'currency',
            'cif_number': 'cif_number',
            'statement_date': 'statement_date',
            'swift_code': 'swift_code',  # Added by database lookup after extraction
            'bank_code': 'bank_code',  # Added by database lookup after extraction
            # Address components from address component parser (already uses address_* names)
            'address_city': 'address_city',
            'address_state': 'address_state',
            'address_postal': 'address_postal',
            'address_country': 'address_country',
        }

        for source_field, target_field in field_mapping.items():
            if source_field in extracted_data:
                field_data = extracted_data[source_field]
                if isinstance(field_data, dict):
                    value = field_data.get('value')
                    confidence = field_data.get('confidence', 0.0)

                    if value is not None:  # Only include non-null values
                        # Normalize statement_date to ISO format
                        if target_field == 'statement_date':
                            normalized_value = self._normalize_date_to_iso(value)
                            if normalized_value:
                                standardized[target_field] = normalized_value
                            else:
                                standardized[target_field] = value  # Fallback to original if parsing fails
                        else:
                            standardized[target_field] = value
                        # Add confidence to confidence_data (confidence_data is field->float mapping)
                        confidence_data[target_field] = confidence

        return standardized

    def _build_error_response(self, error_message: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Build standardized error response.

        Args:
            error_message: Error message

        Returns:
            Tuple of (extracted_data, confidence_data) with error information
        """
        extracted_data = {
            'extraction_method': 'qwen3_vl_direct',
            'account_number_extraction_method': 'qwen3_vl_direct',
            'error': error_message
        }
        confidence_data = {
            'overall': {
                'overall_confidence': 0.0,
                'sources': ['qwen3_vl_direct_error']
            }
        }
        return extracted_data, confidence_data



    def _has_required_fields(self, bank_data) -> bool:
        """Check if bank data has all required fields"""
        required = ['account_holder_name', 'account_number', 'bank_name', 'address']
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


    def _normalize_field_name(self, field: str) -> Optional[str]:
        """Normalize extractor field names to service field names.

        Handles underscore-cased names from Qwen3-VL extraction results.
        """
        field_lower = field.lower()

        # Direct mappings from Qwen3-VL extraction format
        mappings = {
            'bank_name': 'bank_name',
            'account_holder_name': 'account_holder_name',
            'account_number': 'account_number',
            'customer_address': 'address',
            'address': 'address',
            'ifsc_code': 'bank_code',
            'swift_code': 'bank_code',
            'currency': 'currency',
            'bank_branch': 'bank_branch',
            'bank_address': 'bank_address',
            'branch': 'bank_branch',
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
        from app.core.key_injection.bank_database_lookup import get_bank_database_lookup

        bank_lookup = get_bank_database_lookup()
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

        # 3. Bank SWIFT lookup validation
        # Get country from multiple sources in priority order:
        # 1. bank_country from document extraction (most reliable - bank's registered country)
        # 2. Bank name lookup from database (fallback - default country)
        bank_name = extracted_data.get('bank_name', '')
        bank_country = extracted_data.get('bank_country')  # From document extraction

        # If bank_country not extracted from document, try database lookup
        if not bank_country and bank_name:
            bank_info = bank_lookup.lookup_by_name(bank_name)
            if bank_info:
                bank_country = bank_info.country

        # Validate bank with determined country
        bank_found = False
        swift_code = None
        error_msg = None

        if bank_name and bank_country:
            bank_info = bank_lookup.lookup_by_name(bank_name, bank_country)
            if bank_info:
                bank_found = True
                swift_code = bank_info.swift_code
            else:
                error_msg = f"Bank '{bank_name}' not found in SWIFT codes for country '{bank_country}'"
        else:
            error_msg = "Bank name or country missing for validation"

        validation_results['bank_lookup'] = {
            'valid': bank_found,
            'bank_name': bank_name,
            'country': bank_country,
            'swift_code': swift_code,
            'error': error_msg
        }

        if not bank_found:
            validation_errors.append(error_msg)

        # 4. Account number extraction method validation
        account_number = extracted_data.get('account_number', '')
        extraction_method = extracted_data.get('account_number_extraction_method', '')

        # Account number MUST be extracted via validated methods
        # Accept: qwen3_vl_direct (vision LLM)
        # This prevents random numbers from being accepted as account numbers
        valid_methods = {'qwen3_vl_direct'}
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

        # Return combined results
        if validation_errors:
            combined_error = '; '.join(validation_errors)

            # Determine most appropriate error code based on validation errors
            error_code = DocumentErrorCode.PROCESSING_ERROR
            for error in validation_errors:
                if 'too old' in error.lower():
                    error_code = DocumentErrorCode.BANK_STATEMENT_TOO_OLD
                    break
                elif 'not found' in error.lower() or 'not recognized' in error.lower():
                    error_code = DocumentErrorCode.BANK_STATEMENT_BANK_NOT_RECOGNIZED
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
            failed_fields.add('bank_address')

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
        Build extracted_data dict from LineExtractionResult (from key injection).

        Args:
            line_result: LineExtractionResult from key injection analysis
            raw_text: Raw OCR text for bank/currency detection
            text_blocks: OCR blocks with geometry for bank address detection

        Returns:
            Dict with extracted fields
        """
        from app.core.key_injection.bank_database_lookup import detect_bank_in_text, BankInfo
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
            # Legacy: simple_bank_analyzer.extract_bank_address removed with GLiNER
            # Bank address extraction for country detection disabled
            # if text_blocks:
            #     bank_address_text = simple_bank_analyzer.extract_bank_address(text_blocks, bank_info.abbreviation)
            #     if bank_address_text:
            #         detected_country = detect_country_in_text(bank_address_text)
            #         if detected_country:
            #             bank_country = detected_country
            #             self.logger.info(f"Bank country detected from bank address: {bank_country}")

            # Use detected country or bank_info's default country
            if not bank_country:
                bank_country = bank_info.country

            # Get BankInfo with specific country if we have it
            if bank_country:
                from app.core.key_injection.bank_database_lookup import lookup_bank_by_name
                bank_info_with_country = lookup_bank_by_name(bank_info.abbreviation, bank_country)
                if bank_info_with_country:
                    bank_info = bank_info_with_country

            if not bank_name:
                bank_name = bank_info.full_name
            if not bank_swift_code and bank_info.swift_code:
                bank_swift_code = bank_info.swift_code
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
            "opening_balance": None,  # Not extracted by current implementation
            "closing_balance": None,  # Not extracted by current implementation
            "currency": currency,
            "country": client_country
        }

    def _parse_statement_date(self, date_str: Optional[str]) -> Optional[date]:
        """Parse statement date string to date object."""
        if not date_str:
            return None

        try:
            date_clean = str(date_str).strip()

            # Handle date ranges (e.g., "01/09/25 to 31/12/25")
            # Take the end date (the second date)
            range_pattern = r'.*?\s+to\s+(.+)'
            range_match = re.match(range_pattern, date_clean, re.IGNORECASE)
            if range_match:
                # Extract the end date
                date_clean = range_match.group(1).strip()
                # Handle 2-digit year (25 -> 2025)
                # If format is DD/MM/YY, convert to DD/MM/YYYY
                if re.match(r'^\d{1,2}/\d{1,2}/\d{2}$', date_clean):
                    parts = date_clean.split('/')
                    year = int(parts[2])
                    # Assume 20xx for years 00-49, 19xx for years 50-99
                    parts[2] = f"{20 if year < 50 else 19}{year:02d}"
                    date_clean = '/'.join(parts)

            # Special handling for DDMMMYYYY format (no spaces, e.g., "05JAN2026")
            # Check if it matches pattern: digits + uppercase month + digits
            if re.match(r'^\d{1,2}[A-Z]{3}\d{4}$', date_clean.upper()):
                # Insert spaces between components: "05JAN2026" -> "05 JAN 2026"
                date_clean = re.sub(r'(\d{1,2})([A-Z]{3})(\d{4})', r'\1 \2 \3', date_clean.upper())

            date_formats = [
                "%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d", "%d/%m/%Y", "%Y%m%d",
                "%d %b %Y",   # 30 NOV 2025
                "%d %b, %Y",  # 01 JAN, 2026 (Qwen3-VL format with comma)
                "%d %B %Y",   # 30 November 2025
                "%B %d, %Y",  # January 18, 2026 (LLM format)
            ]

            # Handle Unix timestamp format: "Tue Feb 03 11:14:35 GMT+05:30 2026"
            # Extract just the date part (Day Month Year) and parse it
            timestamp_pattern = r'^[A-Z][a-z]+\s+([A-Z][a-z]+)\s+(\d+)\s+[\d:]+\s+[A-Z]+\s+(\d{4})'
            timestamp_match = re.match(timestamp_pattern, date_clean)
            if timestamp_match:
                month_str, day_str, year_str = timestamp_match.groups()
                date_clean = f"{day_str} {month_str} {year_str}"

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
        1. Extracts all fields using Qwen3-VL direct extraction
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
        2. Field extraction using Qwen3-VL direct extraction
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
            # Step 1: Prepare image bytes for Qwen3.5-VL
            # PDFs require conversion to JPEG (vision models need image format)
            # Images pass through unchanged (Qwen3.5-VL handles all formats)
            self.logger.info("=" * 80)
            self.logger.info("QWEN3.5-VL DIRECT EXTRACTION PATH")
            self.logger.info("Both API and test script use this exact code path")
            self.logger.info(f"Processing {'PDF' if is_pdf else 'image'} (page 1 only for PDFs)")
            self.logger.info("=" * 80)

            if is_pdf:
                # Convert PDF first page to JPEG (vision models require image format)
                self.logger.info("Converting PDF page 1 to JPEG for vision model")
                import fitz  # PyMuPDF
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                page = doc[0]  # First page only

                # Render page to image at high resolution
                mat = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=mat)
                image_bytes = pix.tobytes("jpeg")
                doc.close()

                self.logger.info(f"Converted PDF to JPEG: {len(image_bytes)} bytes")
            else:
                # Use original image bytes directly (no preprocessing needed)
                image_bytes = file_bytes
                self.logger.info(f"Using original image bytes: {len(image_bytes)} bytes")

            # Step 2: Qwen3.5-VL Direct Extraction
            # Extract all fields directly from image using vision LLM
            # For PDFs, only page 1 was converted and processed
            self.logger.info("Step 2: Running Qwen3.5-VL direct extraction...")
            extracted_data, confidence_data = await self.extract_fields_from_ocr(
                text_blocks=[],  # Not used by Qwen3.5-VL
                raw_text="",     # Not used by Qwen3.5-VL
                image_bytes=image_bytes,  # JPEG (from PDF) or original image
                is_pdf=is_pdf
            )

            # Log extraction method used
            extraction_method = extracted_data.get('extraction_method', 'unknown')
            self.logger.info(f"Extraction method: {extraction_method}")

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

            # Step 4: Document-specific validations
            doc_valid, doc_error, doc_validation_info = self.perform_document_specific_validations(
                extracted_data, {}
            )

            result['validation_results']['document_specific'] = doc_validation_info.get('validation_results', {})

            # Retry loop for Qwen3-VL direct extraction that failed validation
            # Note: Qwen3-VL doesn't use prompt refinement, so retries are limited
            from app.config.llm_config import llm_settings
            max_retries = 1  # Limited retries for direct extraction (no prompt refinement)
            retry_count = 0
            extraction_context = {
                'image_bytes': image_bytes,  # JPEG bytes (from PDF or original image)
                'is_pdf': is_pdf,
                'extraction_method': extraction_method,
                'bank_name': extracted_data.get('bank_name'),
                'bank_country': extracted_data.get('bank_country')
            }

            # Store bank_abbrev for logging
            if 'bank_abbrev' in extracted_data:
                extraction_context['bank_abbrev'] = extracted_data['bank_abbrev']

            while retry_count <= max_retries:
                # Check if validation passed
                if fields_valid and doc_valid:
                    # Validation succeeded - no need to retry
                    break

                # Qwen3-VL doesn't support prompt refinement
                # Just log the validation failure and break after max retries
                self.logger.info(
                    f"Validation failed (attempt {retry_count + 1}/{max_retries + 1}). "
                    f"Missing required fields: {missing_fields if not fields_valid else 'none'}. "
                    f"Document validation error: {doc_error if not doc_valid else 'none'}."
                )

                # Check if we've exceeded max retries
                if retry_count >= max_retries:
                    self.logger.warning(
                        f"Validation failed after {max_retries} retry attempts. "
                        f"Qwen3-VL direct extraction doesn't support prompt refinement."
                    )
                    break

                retry_count += 1
                self.logger.info(f"Attempt {retry_count}/{max_retries}: Logging validation failure...")

                # No retry mechanism for Qwen3-VL (would just repeat the same extraction)
                # In future, could implement temperature adjustment or different prompts
                break

            # After retry loop, handle final result
            if not (fields_valid and doc_valid):
                result['error_message'] = f"Missing required fields: {', '.join(missing_fields)}" if not fields_valid else doc_error
                result['extracted_data'] = self._filter_failed_validation_fields(
                    extracted_data,
                    result['validation_results']
                )
                result['elapsed_seconds'] = time.time() - start_time
                return result

            # All validations passed - update result with final extracted_data
            result['extracted_data'] = extracted_data
            result['confidence_data'] = confidence_data
            result['success'] = True
            result['elapsed_seconds'] = time.time() - start_time
            return result

        except Exception as e:
            self.logger.error(f"Bank statement validation failed: {e}")
            result['error_message'] = f"Processing error: {str(e)}"
            result['elapsed_seconds'] = time.time() - start_time
            return result


    async def extract_with_spatial_coordinates(
        self,
        ocr_text: str,
        text_blocks: List[Dict[str, Any]],
        image_bytes: bytes,
        bank_id: int,
        bank_name: str,
        bank_abbrev: str,
        country_code: str,
        is_pdf: bool = False
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Spatial coordinate extraction: Uses LLM-identified coordinates to extract field values.

        This method implements the spatial extraction workflow:
        1. Generate layout signature from text blocks
        2. Check cache for existing coordinates
        3. If not cached: Call LLM Layout Analyzer to generate coordinates
        4. Extract with spatial coordinates
        5. Validate results
        6. Return extracted values

        Note: If spatial extraction fails for a field, the LLM already had full context
        and could not identify it. Qwen3-VL direct extraction is used as fallback.

        Args:
            ocr_text: OCR text from the statement
            text_blocks: OCR text blocks with geometry
            image_bytes: First page image bytes
            bank_id: Bank ID from database
            bank_name: Full bank name
            bank_abbrev: Bank abbreviation
            country_code: ISO country code
            is_pdf: True if input is a PDF file

        Returns:
            (extracted_data, confidence_data) tuple
        """
        from app.services.spatial_coordinate_extractor import spatial_coordinate_extractor
        from app.services.bank_layout_database_service import bank_layout_database_service
        from app.config.llm_config import llm_settings
        from app.helper.doctr.document_text_extractor import DocumentTextExtractor

        self.logger.info("=" * 80)
        self.logger.info("SPATIAL COORDINATE EXTRACTION FLOW")
        self.logger.info("=" * 80)

        # Step 0: Preprocess image ONCE for the entire pipeline
        # This ensures both DocTR OCR and Vision LLM process the SAME image
        # with identical dimensions, preventing coordinate misalignment.
        self.logger.info("Preprocessing image for pipeline...")
        preprocessed_image_bytes, img_width, img_height, padding = self._preprocess_image_for_pipeline(image_bytes)
        self.logger.info(f"Image preprocessed: {img_width}×{img_height}, padding={padding}")

        # Extract OCR from the preprocessed image
        self.logger.info("Extracting OCR from preprocessed image...")
        ocr_extractor = DocumentTextExtractor()
        text_blocks = await ocr_extractor.extract_text_with_geometry_enhanced(
            preprocessed_image_bytes, is_pdf=False, max_pages=1, document_type=DocumentType.BANK_STATEMENT,
            skip_resize=True  # Skip resizing - image is already preprocessed
        )
        self.logger.info(f"OCR extraction complete: {len(text_blocks)} text blocks")

        # DocTR processes the preprocessed image (1008×1008 with padding)
        # Coordinates are already in canvas space - no conversion needed

        # Step 1: Generate layout signature from preprocessed image text blocks
        layout_signature = self._generate_layout_signature(text_blocks)
        layout_hash = hashlib.sha256(layout_signature.encode()).hexdigest()

        self.logger.info(
            f"Generated layout signature: hash={layout_hash[:16]}, "
            f"signature={layout_signature[:100]}..."
        )

        # Step 2: Check cache for existing coordinates
        cached_layout = bank_layout_database_service.get_layout_cache(
            bank_id=bank_id,
            country_code=country_code,
            layout_hash=layout_hash
        )

        if cached_layout:
            self.logger.info(
                f"Using cached layout for {bank_abbrev}/{country_code}: "
                f"{len(cached_layout.get('coordinates', {}))} fields, "
                f"usage_count={cached_layout['usage_count']}"
            )

            # Vision layout already provides coordinates directly
            coordinates = cached_layout.get('coordinates', {})
            layout_analysis = cached_layout

            # Update usage stats
            bank_layout_database_service.update_usage_stats(
                bank_id=bank_id,
                country_code=country_code,
                layout_hash=layout_hash
            )
        else:
            # Step 3: Layout Analysis - Priority: Text → Vision → Fallback
            # Text layout analyzer uses OCR coordinates directly (no padding issues)
            # Vision layout analyzer uses image input (may have padding misalignment)
            if llm_settings.enable_text_layout_analysis:
                self.logger.info(
                    f"Generating new layout for {bank_abbrev}/{country_code} "
                    f"with TEXT-ONLY LLM analysis (OCR coordinates)..."
                )

                from app.services.text_layout_analyzer import text_layout_analyzer

                # Get OCR text for text layout analyzer
                ocr_text = "\n".join([block.get('text', '') for block in text_blocks])

                layout_analysis = await text_layout_analyzer.analyze_layout(
                    ocr_text=ocr_text,
                    text_blocks=text_blocks,
                    bank_name=bank_name,
                    bank_abbrev=bank_abbrev,
                    country_code=country_code,
                    page_width=img_width,
                    page_height=img_height
                )

                coordinates = layout_analysis.get('coordinates', {})

            elif llm_settings.enable_vision_layout_analysis:
                self.logger.info(
                    f"Generating new layout for {bank_abbrev}/{country_code} "
                    f"with VISION LLM analysis (direct image input)..."
                )

                from app.services.vision_layout_analyzer import vision_layout_analyzer

                layout_analysis = await vision_layout_analyzer.analyze_layout(
                    image_bytes=preprocessed_image_bytes,  # Preprocessed image for coordinate alignment
                    bank_name=bank_name,
                    bank_abbrev=bank_abbrev,
                    country_code=country_code,
                    padding=padding  # Pass padding info for coordinate adjustment
                )

                coordinates = layout_analysis.get('coordinates', {})

            else:
                self.logger.info(
                    f"Both text and vision layout analysis disabled. "
                    f"Skipping to Qwen3-VL direct extraction for {bank_abbrev}/{country_code}."
                )
                coordinates = {}
                layout_analysis = None

            # Step 4: Save to cache (only if we have spatial coordinates)
            if coordinates and layout_analysis:
                save_success = bank_layout_database_service.save_layout_cache(
                    bank_id=bank_id,
                    country_code=country_code,
                    layout_hash=layout_hash,
                    layout_signature=layout_analysis.get('layout_signature', layout_signature),
                    coordinates=layout_analysis.get('coordinates', {}),  # Cache coordinates directly for vision mode
                    gliner_prompts={},  # Empty prompts - GLiNER removed, Qwen3-VL used instead
                    metadata=layout_analysis.get('metadata', {})
                )

                if save_success:
                    self.logger.info(
                        f"Saved layout cache for {bank_abbrev}/{country_code}: "
                        f"{len(layout_analysis.get('coordinates', {}))} fields"
                    )
                else:
                    self.logger.warning("Failed to save layout cache")
            else:
                self.logger.warning("LLM layout analysis returned no coordinates, falling back to Qwen3-VL direct extraction")
                # Fall back to Qwen3-VL direct extraction
                return await self.extract_fields_from_ocr(
                    text_blocks=text_blocks,
                    raw_text=ocr_text,
                    image_bytes=preprocessed_image_bytes,
                    is_pdf=is_pdf
                )

        # Step 5: Spatial extraction with preprocessed text blocks
        self.logger.info(f"Extracting with spatial coordinates: {len(coordinates)} fields")
        spatial_results = spatial_coordinate_extractor.extract_from_coordinates(
            text_blocks=text_blocks,  # Use preprocessed text blocks for alignment
            coordinates=coordinates
        )

        # Step 6: Validate spatial results
        validation_result = self._validate_spatial_results(spatial_results)

        # Note: No GLiNER fallback - Qwen3-VL direct extraction is used instead
        # If spatial extraction failed for a field, the LLM (with full context)
        # already could not identify it. Re-running extraction on the same text
        # would not succeed where the LLM failed.

        # Convert to response format
        return self._convert_spatial_results_to_response(
            spatial_results=spatial_results,
            is_pdf=is_pdf,
            extraction_method='spatial_coordinates',
            bank_name=bank_name,
            bank_country=country_code
        )

    def _crop_whitespace(self, img, margin_threshold: int = 30) -> Image.Image:
        """
        Crop whitespace margins from image to improve canvas utilization.

        Detects and removes blank margins from all four sides, keeping only the
        content area. This maximizes the effective resolution for Vision LLM processing.

        Args:
            img: PIL Image to crop
            margin_threshold: Pixels of whitespace to preserve at edges (safety margin)

        Returns:
            Cropped PIL Image
        """
        self.logger.info(f"Cropping whitespace from {img.size[0]}×{img.size[1]} image")

        # Convert to numpy array for efficient processing
        img_array = np.array(img)

        # For RGB images, check if all channels are near black/white
        if len(img_array.shape) == 3:
            # Convert to grayscale for simpler thresholding
            if img_array.shape[2] == 3:  # RGB
                gray = np.mean(img_array, axis=2).astype(np.uint8)
            else:  # RGBA
                gray = np.mean(img_array[:, :, :3], axis=2).astype(np.uint8)
        else:
            gray = img_array

        # Define whitespace (either near black OR near white)
        # Documents can have black or white backgrounds
        is_dark = gray < 30  # Near black
        is_light = gray > 240  # Near white
        has_content = ~(is_dark | is_light)  # Has actual content

        # Find content bounds
        rows_with_content = np.any(has_content, axis=1)
        cols_with_content = np.any(has_content, axis=0)

        if not np.any(rows_with_content) or not np.any(cols_with_content):
            # No content found, return original
            self.logger.warning("No content detected in image, skipping crop")
            return img

        # Find bounding box of content
        top = np.argmax(rows_with_content)
        bottom = len(rows_with_content) - np.argmax(rows_with_content[::-1])
        left = np.argmax(cols_with_content)
        right = len(cols_with_content) - np.argmax(cols_with_content[::-1])

        # Add safety margin
        top = max(0, top - margin_threshold)
        left = max(0, left - margin_threshold)
        bottom = min(img.size[1], bottom + margin_threshold)
        right = min(img.size[0], right + margin_threshold)

        # Crop image
        cropped = img.crop((left, top, right, bottom))

        self.logger.info(f"Cropped from {img.size[0]}×{img.size[1]} to {cropped.size[0]}×{cropped.size[1]} "
                         f"(removed {left}px left, {top}px top, {img.size[0]-right}px right, {img.size[1]-bottom}px bottom)")

        return cropped

    def _preprocess_image_for_pipeline(self, image_bytes: bytes) -> tuple[bytes, int, int, dict]:
        """
        Preprocess image for the pipeline - simplified to eliminate coordinate bleeding.

        This is the ONLY place where image resizing should happen.
        The resulting image is then passed consistently to both DocTR OCR and Vision LLM,
        ensuring coordinate alignment between text blocks and bbox coordinates.

        SIMPLIFIED APPROACH:
        - Original image is pasted onto 1008×1008 canvas with minimal resizing
        - Only scales if dimension > 1008 (to fit on canvas)
        - Maintains single coordinate system to avoid transformation errors
        - This fixes coordinate bleeding caused by multiple resize operations

        Target dimensions satisfy:
        - Exactly 1008×1008 (Qwen3-VL optimal reference resolution)
        - Multiples of 28 (qwen3-vl requirement)

        The 1008×1008 requirement is critical for Qwen3-VL coordinate accuracy.
        Non-square images cause coordinate misalignment due to aspect ratio issues.
        See: https://github.com/ggerganov/llama.cpp/issues/16880

        Args:
            image_bytes: Original image or PDF bytes

        Returns:
            Tuple of (preprocessed JPEG bytes, width, height, padding_info)
            - width and height are always 1008
            - padding_info contains {'left': x, 'top': y, 'right': x, 'bottom': y}
        """
        import fitz  # PyMuPDF

        self.logger.info("=" * 80)
        self.logger.info("UNIFIED IMAGE PREPROCESSING")
        self.logger.info("=" * 80)

        # Step 1: Convert PDF to image if needed
        is_pdf = image_bytes.startswith(b'%PDF')
        if is_pdf:
            self.logger.info("Input is PDF - converting to image...")

            pdf_doc = fitz.open(stream=image_bytes, filetype="pdf")
            if len(pdf_doc) == 0:
                raise ValueError("PDF has no pages")

            page = pdf_doc[0]

            # Render at appropriate DPI for quality
            dpi = 150
            pix = page.get_pixmap(dpi=dpi)

            # Convert to PIL Image
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))

            pdf_doc.close()
            self.logger.info(f"PDF rendered to image: {img.size[0]}×{img.size[1]}")
        else:
            # Load image
            img = Image.open(io.BytesIO(image_bytes))
            self.logger.info(f"Input image dimensions: {img.size[0]}×{img.size[1]}")

        # Convert to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Step 1.5: Crop whitespace margins to improve canvas utilization
        img = self._crop_whitespace(img, margin_threshold=30)

        width, height = img.size

        # Step 2: Scale only if too large for 1008×1008 canvas
        max_dimension = max(width, height)
        if max_dimension > 1008:
            scale = 1008 / max_dimension
            width = int(width * scale)
            height = int(height * scale)
            self.logger.info(f"Scaled to fit 1008×1008 canvas: {width}×{height}")

        # Step 3: Align to multiples of 28 (Qwen3-VL requirement)
        new_width = (width // 28) * 28
        new_height = (height // 28) * 28

        # Ensure minimum dimensions (at least 28)
        new_width = max(28, new_width)
        new_height = max(28, new_height)

        self.logger.info(f"Aligned to multiples of 28: {new_width}×{new_height}")

        # Step 4: Resize if needed (single transformation)
        if (new_width, new_height) != img.size:
            self.logger.info(f"Resizing from {img.size[0]}×{img.size[1]} to {new_width}×{new_height}")
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        else:
            self.logger.info("No resizing needed - dimensions already compatible")

        # Step 5: Pad to exactly 1008×1008 (Qwen3-VL optimal reference resolution)
        # Qwen3-VL has patch size 14×14, merging 2×2 patches = 28 pixels
        # 1008 is divisible by 28 (1008 ÷ 28 = 36), preventing internal resizing
        # See: https://github.com/ggerganov/llama.cpp/issues/16880
        if (new_width, new_height) != (1008, 1008):
            self.logger.info(f"Padding to 1008×1008 for Qwen3-VL coordinate accuracy (top-left alignment)")

            # Create 1008×1008 canvas with black padding
            canvas = Image.new('RGB', (1008, 1008), 'black')

            # Align to top-left (no centering) - simplifies coordinate matching
            paste_x = 0
            paste_y = 0

            canvas.paste(img, (paste_x, paste_y))

            # Calculate padding info (padding is only on right/bottom)
            padding = {
                'left': 0,
                'top': 0,
                'right': 1008 - new_width,
                'bottom': 1008 - new_height
            }

            self.logger.info(f"Padding applied: left={padding['left']}, top={padding['top']}, "
                           f"right={padding['right']}, bottom={padding['bottom']} (top-left aligned)")

            img = canvas
        else:
            padding = {'left': 0, 'top': 0, 'right': 0, 'bottom': 0}
            self.logger.info("Image already 1008×1008 - no padding needed")

        # Step 6: Save as JPEG
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=95)
        result = output.getvalue()

        self.logger.info(f"Preprocessed image: {len(result)} bytes, 1008×1008")
        self.logger.info(f"Simplified preprocessing: only scale if > 1008, align to 28, pad to 1008×1008")
        self.logger.info(f"This eliminates coordinate bleeding from multiple resize operations")
        self.logger.info("=" * 80)

        return result, 1008, 1008, padding

    def _generate_layout_signature(self, text_blocks: List[Dict[str, Any]]) -> str:
        """
        Generate structural layout signature for cache matching.

        Uses relative positions of text elements, not exact content.
        This allows matching same format with different content.

        Format: "X_REGION_Y_REGION_COUNT..."
        Example: "header_top_5_contact_middle_3_table_bottom_20"
        """
        if not text_blocks:
            return "empty_layout"

        # Group blocks by Y regions (top, middle, bottom)
        y_regions = {"top": [], "middle": [], "bottom": []}

        for block in text_blocks:
            # Get normalized Y coordinates
            if isinstance(block, dict):
                geometry = block.get('geometry', {})
                if geometry:
                    y1 = geometry.get('y_min', 0)
                    y2 = geometry.get('y_max', 1)
                else:
                    y1 = block.get('y1', 0)
                    y2 = block.get('y2', 1)
            else:
                y1 = getattr(block, 'y1', 0)
                y2 = getattr(block, 'y2', 1)

            # Handle both normalized (0-1) and absolute coordinates
            if y1 > 1:
                page_height = block.get('page_height', 1000) if isinstance(block, dict) else 1000
                y1 = y1 / page_height
            if y2 > 1:
                page_height = block.get('page_height', 1000) if isinstance(block, dict) else 1000
                y2 = y2 / page_height

            y_center = (y1 + y2) / 2

            if y_center < 0.33:
                y_regions["top"].append(block)
            elif y_center < 0.66:
                y_regions["middle"].append(block)
            else:
                y_regions["bottom"].append(block)

        # Count blocks in each X region
        signature_parts = []
        for region_name, blocks in y_regions.items():
            x_regions = {"left": 0, "center": 0, "right": 0}

            for block in blocks:
                # Get normalized X coordinates
                if isinstance(block, dict):
                    geometry = block.get('geometry', {})
                    if geometry:
                        x1 = geometry.get('x_min', 0)
                        x2 = geometry.get('x_max', 1)
                    else:
                        x1 = block.get('x1', 0)
                        x2 = block.get('x2', 1)
                else:
                    x1 = getattr(block, 'x1', 0)
                    x2 = getattr(block, 'x2', 1)

                # Handle both normalized (0-1) and absolute coordinates
                if x1 > 1:
                    page_width = block.get('page_width', 1000) if isinstance(block, dict) else 1000
                    x1 = x1 / page_width
                if x2 > 1:
                    page_width = block.get('page_width', 1000) if isinstance(block, dict) else 1000
                    x2 = x2 / page_width

                x_center = (x1 + x2) / 2

                if x_center < 0.33:
                    x_regions["left"] += 1
                elif x_center < 0.66:
                    x_regions["center"] += 1
                else:
                    x_regions["right"] += 1

            signature_parts.append(
                f"{region_name}_L{x_regions['left']}_C{x_regions['center']}_R{x_regions['right']}"
            )

        return "_".join(signature_parts)

    def _validate_spatial_results(self, spatial_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate spatial extraction results.

        Returns:
            {
                "is_valid": bool,
                "failed_fields": list,
                "warnings": list
            }
        """
        failed_fields = []
        warnings = []

        # Check required fields
        required_fields = ['account_holder_name', 'customer_address', 'account_number', 'bank_name']

        for field in required_fields:
            if field not in spatial_results or not spatial_results[field].get('value'):
                failed_fields.append(field)

        # Check for low confidence results
        for field_name, field_data in spatial_results.items():
            confidence = field_data.get('confidence', 1.0)
            if confidence < 0.5:
                warnings.append(f"Low confidence for {field_name}: {confidence:.2f}")

        return {
            'is_valid': len(failed_fields) == 0,
            'failed_fields': failed_fields,
            'warnings': warnings
        }

    def _convert_spatial_results_to_response(
        self,
        spatial_results: Dict[str, Any],
        is_pdf: bool,
        extraction_method: str,
        bank_name: Optional[str] = None,
        bank_country: Optional[str] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Convert spatial extraction results to response format.

        Args:
            spatial_results: Spatial extraction results
            is_pdf: True if input is PDF
            extraction_method: Extraction method identifier
            bank_name: Bank name
            bank_country: Bank country code

        Returns:
            (extracted_data, confidence_data) tuple
        """
        extracted_data = {}
        confidence_data = {}

        # Field mappings from spatial schema to our schema
        field_mappings = {
            'bank_name': 'bank_name',
            'account_holder_name': 'account_holder_name',
            'account_number': 'account_number',
            'cif_number': 'cif_number',
            'customer_address': 'address',
            'address_country': 'address_country',
            'address_city': 'address_city',
            'address_state': 'address_state',
            'address_postal_code': 'address_postal',
            'branch_address': 'branch_address',
            'branch_name': 'bank_branch',
            'currency': 'currency',
            'statement_date': 'statement_date',
        }

        for spatial_field, our_field in field_mappings.items():
            if spatial_field in spatial_results:
                result = spatial_results[spatial_field]
                if isinstance(result, dict):
                    extracted_data[our_field] = result.get('value', '')
                    confidence = result.get('confidence', 0.0)
                    source = result.get('source', 'spatial')
                    confidence_data[our_field] = {
                        'overall_confidence': float(confidence),
                        'sources': [extraction_method, source]
                    }
                else:
                    extracted_data[our_field] = str(result)

        # Add bank_name and bank_country from context if not in results
        if bank_name and 'bank_name' not in extracted_data:
            extracted_data['bank_name'] = bank_name
        if bank_country and 'bank_country' not in extracted_data:
            extracted_data['bank_country'] = bank_country

        # Add metadata
        extracted_data['extraction_method'] = extraction_method
        extracted_data['account_number_extraction_method'] = 'spatial_geometry'
        extracted_data['is_pdf'] = is_pdf

        # Calculate overall confidence
        if confidence_data:
            confidences = [c['overall_confidence'] for c in confidence_data.values()]
            overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            confidence_data['overall'] = {
                'overall_confidence': overall_confidence,
                'sources': [extraction_method]
            }

        self.logger.info(
            f"Converted spatial results: {len(extracted_data)} fields, "
            f"overall_confidence={confidence_data.get('overall', {}).get('overall_confidence', 0):.1f}"
        )

        return extracted_data, confidence_data

    def _convert_country_name_to_iso_code(self, country_name: str) -> str:
        """Convert country name to ISO 3166-1 alpha-2 code.

        Args:
            country_name: Country name (e.g., "SINGAPORE"), ISO code (e.g., "SG"),
                         or currency code (e.g., "THB", "SGD")

        Returns:
            ISO 3166-1 alpha-2 code (e.g., "SG")

        Raises:
            ValueError: If country_name cannot be converted to a valid ISO code
        """
        if not country_name:
            raise ValueError("Country name is empty, cannot convert to ISO code")

        # If already a 2-letter ISO code, return as-is
        if len(country_name) == 2 and country_name.isalpha():
            return country_name.upper()

        # Try to convert country name to ISO code
        iso_code = COUNTRY_NAMES.get(country_name.lower())
        if iso_code:
            return iso_code.upper()

        # If not found in mapping, try to detect from text
        detected = detect_country_in_text(country_name)
        if detected:
            return detected

        # Fallback: Try currency-to-country conversion (e.g., "THB" -> "TH")
        # This handles cases where the extraction returns currency code instead of country code
        if len(country_name) == 3 and country_name.isalpha():
            currency_code = country_name.upper()
            if currency_code in CURRENCY_COUNTRIES:
                country_code = CURRENCY_COUNTRIES[currency_code]
                self.logger.info(f"Converted currency code '{currency_code}' to country code '{country_code}'")
                return country_code

        # Fail fast - do not silently default to a wrong country
        raise ValueError(f"Could not convert '{country_name}' to a valid ISO country code")

    # Bank validation removed - database lookup happens in main extraction flow
    # This method was redundant - let main flow handle unknown banks
    # Removed validate_bank_recognition method to allow unknown banks to continue processing



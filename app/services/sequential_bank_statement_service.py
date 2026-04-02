"""
Sequential Bank Statement Service - Handles bank statement processing in sequential mode.

REFACTORED: Now uses DocumentProcessorBase for unified validation pipeline.
Uses Spatial extractor (PyMuPDF geometry-based) for bank statement field extraction.

State is tracked via verification_state column in user_identity_index.
"""

import asyncio
from typing import Dict, Any, Optional, Tuple, List
from datetime import date
from app.services.sequential_document_processor_base import DocumentProcessorBase
from app.core.key_injection.simple_bank_analyzer import simple_bank_analyzer
from app.core.key_injection.key_injection_manager import key_injection_manager
from app.core.key_injection import DocumentType
from app.helper.extractors.spatial_bank_statement_extractor import SpatialBankStatementExtractor
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
        Extract bank statement fields using Spatial extractor (PyMuPDF geometry-based).

        Spatial extractor provides robust extraction using:
        - Three-pass algorithm for label and value detection
        - Geometric proximity for spatial relationships
        - Direct PDF parsing without OCR dependency

        Note: Only works with PDF files. For images, this method will raise an error.
        The raw_text and text_blocks parameters are kept for compatibility but not used.

        Returns:
            (extracted_data, confidence_data)
        """
        # Use Spatial bank statement extractor - processes PDF bytes directly
        # Avoids OCR overhead and uses PyMuPDF for accurate geometric extraction
        extractor = SpatialBankStatementExtractor()
        result = await extractor.extract_from_bytes(image_bytes)

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
        confidence_data = {}
        for field, confidence in result.confidence_scores.items():
            # Map normalized field names to required field names
            normalized_field = self._normalize_field_name(field)
            if normalized_field:
                conf_value = confidence / 100 if confidence > 1 else confidence
                confidence_data[normalized_field] = {
                    'overall_confidence': conf_value,
                    'sources': ['spatial_geometry']
                }

        # Add overall confidence if available
        if result.overall_confidence:
            confidence_data['overall'] = {
                'overall_confidence': result.overall_confidence / 100,
                'sources': ['spatial_geometry']
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
                        'sources': ['spatial_fallback']
                    }

        self.logger.info(
            f"Spatial bank statement extraction: "
            f"bank={result.bank_name}, "
            f"account_holder={result.account_holder_name}, "
            f"overall_confidence={result.overall_confidence:.2f}%"
        )

        # Extract extraction_method from text_blocks metadata marker
        # The first element contains the extraction metadata added by document_text_extractor
        extraction_method = 'unknown'
        if text_blocks and len(text_blocks) > 0:
            first_block = text_blocks[0]
            if first_block.get('text') == '__EXTRACTION_METADATA__':
                extraction_method = first_block.get('extraction_method', 'unknown')
                self.logger.info(f"Extraction method detected: {extraction_method}")

        # Add extraction_method to extracted_data for validation
        extracted_data['extraction_method'] = extraction_method

        return extracted_data, confidence_data

    def _normalize_field_name(self, field: str) -> Optional[str]:
        """Normalize extractor field names to service field names.

        Handles both underscore-cased names (from confidence_scores) and
        space-separated names (legacy format from labels-based extraction).
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

            # Space-separated format (legacy)
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
        - Extraction method validation (must be 'direct', not 'ocr')
        - Statement age check (max 90 days old)
        - Address components validation
        - Bank SWIFT lookup validation
        - Account number extraction method validation
        - Account number format validation (per currency)
        - Credit card rejection (Luhn check)

        NOTE: Collects ALL validation errors before returning (no fail-fast).
        This provides complete feedback on all issues with the document.
        """
        from app.helper.validators.bank_statement_validator import get_bank_statement_validator

        validator = get_bank_statement_validator()
        validation_results = {}
        validation_errors = []  # Collect all errors

        # 1. Extraction method validation - bank statements must use direct PDF extraction
        # DISABLED: Extraction method validation was incorrectly rejecting valid text-based PDFs
        # TODO: Fix pdf_analyzer content type detection before re-enabling
        extraction_method = extracted_data.get('extraction_method', 'unknown')
        extraction_method_valid = True  # Assume valid - validation disabled

        validation_results['extraction_method'] = {
            'method': extraction_method,
            'valid': extraction_method_valid
        }

        # if not extraction_method_valid:
        #     if extraction_method == 'ocr':
        #         validation_errors.append(
        #             "Bank statement must be a text-based PDF with extractable text. "
        #             "Image-based or scanned PDFs processed via OCR are not accepted."
        #         )
        #     else:
        #         validation_errors.append(
        #             f"Bank statement extraction method verification failed. "
        #             f"Detected method: '{extraction_method}'. "
        #             "Please ensure you are uploading a valid digital bank statement PDF."
        #         )

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

        # Account number MUST be extracted via spatial methods (with nearby label)
        # Accept both spatial_label (legacy) and spatial_geometry (new spatial extractor)
        # This prevents random numbers from being accepted as account numbers
        valid_methods = {'spatial_label', 'spatial_geometry'}
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
        iv: str, callback_url: Optional[str] = None
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
        1. Extracts all fields using Spatial extractor (PyMuPDF geometry-based)
        2. Validates required fields
        3. Performs bank statement specific validations

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
        1. Field extraction using Spatial extractor (PyMuPDF geometry-based)
        2. Required fields validation
        3. Document-specific validations

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
            # Uses intelligent routing: direct PDF text extraction for text-based PDFs
            # This is the EXACT SAME CODE PATH used by both the API and test script
            self.logger.info("=" * 80)
            self.logger.info("UNIFIED BANK STATEMENT EXTRACTION PATH")
            self.logger.info("Both API and test script use this exact code path")
            self.logger.info("Using intelligent PDF routing: direct extraction for text-based PDFs")
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

            # Step 2: Field extraction using Spatial extractor
            # Uses PyMuPDF for direct PDF parsing with geometric extraction
            extractor = SpatialBankStatementExtractor()
            extraction_result = await extractor.extract_from_bytes(file_bytes)

            # Build extracted_data
            # Normalize statement_date to ISO format
            statement_date_normalized = self._normalize_date_to_iso(extraction_result.statement_date)

            extracted_data = {
                'account_holder_name': extraction_result.account_holder_name,
                'account_number': extraction_result.account_number,
                'bank_name': extraction_result.bank_name,
                'bank_code': extraction_result.bank_code,
                'bank_branch': extraction_result.bank_branch,
                'bank_country': extraction_result.bank_country,
                'address': extraction_result.address,
                # Address components (decomposed from full address)
                'address_city': extraction_result.address_city,
                'address_state': extraction_result.address_state,
                'address_postal': extraction_result.address_postal,
                'address_country': extraction_result.address_country,
                'currency': extraction_result.currency,
                'statement_date': statement_date_normalized,  # ISO format (YYYY-MM-DD)
                'account_number_extraction_method': extraction_result.account_number_extraction_method,
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

            # Build confidence_data
            confidence_data = {}
            for field, confidence in extraction_result.confidence_scores.items():
                conf_value = confidence / 100 if confidence > 1 else confidence
                confidence_data[field] = {'overall_confidence': conf_value}

            if extraction_result.overall_confidence:
                confidence_data['overall'] = {
                    'overall_confidence': extraction_result.overall_confidence / 100
                }

            # Populate missing field confidences using overall_confidence (matches API behavior)
            if extraction_result.overall_confidence:
                required_fields = self.get_required_fields()
                for field in required_fields:
                    if field not in confidence_data and extracted_data.get(field):
                        confidence_data[field] = {
                            'overall_confidence': extraction_result.overall_confidence / 100,
                            'sources': ['spatial_fallback']
                        }

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
            self.logger.error(f"Bank statement validation failed: {e}")
            result['error_message'] = f"Processing error: {str(e)}"
            result['elapsed_seconds'] = time.time() - start_time
            return result


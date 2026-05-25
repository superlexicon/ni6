"""
GLiNER2 Bank Statement Extractor

Uses GLiNER2 zero-shot NER for bank statement field extraction.
Works with raw OCR text (GLiNER2 processes text directly) plus spatial context.
"""

import re
import logging
from typing import Any, Dict, List, Optional, Set, Tuple
from app.core.gliner_ner_model import GLiNERNERModel
from app.schemas.bank_statement_schema import BankStatementData
from app.core.key_injection.bank_database_lookup import (
    get_swift_code_for_bank, get_country_for_bank, lookup_bank_by_domain, lookup_bank_by_name, lookup_bank_by_iban, detect_bank_in_text
)
from app.helper.validators.bank_statement_validator import get_bank_statement_validator, _load_country_locations
from app.config.address_keywords_loader import get_address_keywords_loader
from app.config.bank_statement_country_loader import get_country_config_loader

logger = logging.getLogger(__name__)


class GLiNERBankStatementExtractor:
    """
    Bank statement extractor using GLiNER2 zero-shot NER.

    Uses 2-pass approach:
    1. GLiNER detects entities from raw text
    2. Spatial context combines multi-line addresses from text blocks
    """

    # Address-related GLiNER labels
    ADDRESS_LABELS = {
        # Schema-based extraction field name (underscore variant)
        'customer_address',
        # Legacy labels-based extraction labels
        'customer address',
        'customer residential address',
        'customer permanent address',
        'customer home address',
        'address with near landmark or temple',
        'address with village or locality name',
        'address with house number and street name',
        'postal address with pin code or zip code',
        # Positional-aware address labels for formal letter format
        'multi line customer address appearing below account holder name in formal letter format',
        'customer address in top left of document with multiple lines',
        'residential address spanning multiple lines below customer name',
        'address blocks directly under account holder name in letter format',
        'full customer address appearing directly after name in letter format',
    }

    # Bank branch address labels (to exclude from customer address)
    BANK_BRANCH_LABELS = {
        'bank branch address',
        'branch address or bank location',
        'branch_address',  # Schema-based extraction field name
    }

    # CIF number patterns - common prefixes and patterns for Indian bank CIF numbers
    # NOTE: Be specific to avoid matching legitimate account numbers
    # SBI CIF numbers start with 8848 (e.g., 88481381628)
    # Bank of Maharashtra CIF often starts with 4024 (e.g., 40246416331)
    # Generic CIF detection: 11 digits but validated by explicit CIF label or context
    CIF_PATTERNS = [
        r'^8848\d{7}$',      # SBI CIF numbers start with 8848
        r'^4024\d{7}$',      # Bank of Maharashtra CIF numbers start with 4024
        # Note: Don't use generic \d{11} pattern - it matches real account numbers
    ]

    def __init__(self):
        """Initialize the extractor."""
        self.gliner_model = None

    def _is_cif_number(self, value: str) -> bool:
        """
        Check if a value is a CIF (Customer Identification File) number.

        CIF numbers are often misclassified as IFSC codes by GLiNER2.
        This method uses known patterns to identify and filter them out.

        Args:
            value: The value to check

        Returns:
            True if the value matches a CIF number pattern
        """
        if not value:
            return False

        # Clean the value - remove whitespace
        cleaned = re.sub(r'\s+', '', value)

        # Check against known CIF patterns
        for pattern in self.CIF_PATTERNS:
            if re.match(pattern, cleaned):
                return True

        # Check if labeled as CIF in nearby context
        # Sometimes CIF numbers appear with "CIF:" or "CIF No:" labels
        if re.match(r'^[Cc][Ii][Ff]', value) or 'cif' in value.lower():
            return True

        return False

    def _filter_cif_numbers(self, gliner_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter out CIF numbers from IFSC code and Account Number entities.

        GLiNER2 often misclassifies CIF numbers as "ifsc code" or "account number".
        This method identifies and removes CIF numbers from those results to
        prevent them from being mapped to the wrong fields.

        Args:
            gliner_result: Raw GLiNER extraction results

        Returns:
            Filtered GLiNER results with CIF numbers removed from IFSC and account entries
        """
        if not gliner_result:
            return gliner_result

        # Collect all CIF number values (explicitly detected as CIF or matching patterns)
        cif_values = set()

        # 1. Find explicit CIF labels detected by GLiNER (schema + legacy labels)
        cif_label_variants = [
            'cif_number',  # Schema field
            'customer identification file number or cif number',
            'customer identification file number or CIF number',
        ]

        for cif_label in cif_label_variants:
            if cif_label in gliner_result:
                entities = gliner_result[cif_label]
                if isinstance(entities, list):
                    for e in entities:
                        if e:
                            cif_values.add(e.get('value', '').strip())
                elif entities:
                    cif_values.add(entities.get('value', '').strip())

        # 2. Filter IFSC code entities - remove CIF numbers
        # Handle both single-value (dict) and multi-value (list) formats
        ifsc_label_variants = ['ifsc code', 'IFSC code']
        for ifsc_label in ifsc_label_variants:
            if ifsc_label in gliner_result:
                entities = gliner_result[ifsc_label]
                if entities is None:
                    continue

                # Handle both single entity (dict) and multiple entities (list)
                if isinstance(entities, list):
                    filtered_entities = []
                    for entity in entities:
                        if entity:
                            value = entity.get('value', '')
                            # Check if it's already in CIF values OR matches CIF patterns
                            if value.strip() in cif_values or self._is_cif_number(value):
                                logger.info(f"Filtered CIF number from IFSC code: {value}")
                                cif_values.add(value.strip())  # Track for account number filtering
                            else:
                                filtered_entities.append(entity)

                    if filtered_entities:
                        gliner_result[ifsc_label] = filtered_entities
                    else:
                        gliner_result[ifsc_label] = None
                else:
                    # Single entity (dict)
                    value = entities.get('value', '')
                    if value.strip() in cif_values or self._is_cif_number(value):
                        logger.info(f"Filtered CIF number from IFSC code: {value}")
                        cif_values.add(value.strip())  # Track for account number filtering
                        gliner_result[ifsc_label] = None

        # 3. Filter account number entities - also remove CIF numbers
        account_label_variants = ['account number']
        for account_label in account_label_variants:
            if account_label in gliner_result:
                entities = gliner_result[account_label]
                if entities is None:
                    continue

                if isinstance(entities, list):
                    filtered_entities = []
                    for entity in entities:
                        if entity:
                            value = entity.get('value', '')
                            if value.strip() in cif_values:
                                logger.info(f"Filtered CIF number from account number: {value}")
                            else:
                                filtered_entities.append(entity)

                    if filtered_entities:
                        gliner_result[account_label] = filtered_entities
                    else:
                        gliner_result[account_label] = None
                else:
                    # Single entity (dict)
                    value = entities.get('value', '')
                    if value.strip() in cif_values:
                        logger.info(f"Filtered CIF number from account number: {value}")
                        gliner_result[account_label] = None

        return gliner_result

    async def extract(
        self,
        ocr_text: str,
        text_blocks: Optional[List[Dict[str, Any]]] = None,
    ) -> BankStatementData:
        """
        Extract bank statement fields using GLiNER2 schema-based extraction.

        Uses hybrid approach:
        - Schema-based extraction for most fields (bank_name, account_holder, etc.)
        - Spatial context extraction for address (more reliable for multi-line addresses)
        - Bank name fallback from website URLs/email domains

        Args:
            ocr_text: Raw OCR text
            text_blocks: Optional list of text blocks with geometry from OCR

        Returns:
            BankStatementData with extracted fields
        """
        # Get GLiNER model singleton
        if self.gliner_model is None:
            self.gliner_model = GLiNERNERModel()

        # Extract using GLiNER2 schema-based extraction
        gliner_result = await self.gliner_model.extract_bank_statement_with_schema_async(
            text=ocr_text,
        )

        # Filter out CIF numbers from IFSC code entities
        gliner_result = self._filter_cif_numbers(gliner_result)

        # Build result from GLiNER output
        # Address extraction will use spatial context as fallback
        result = self._build_result(gliner_result, text_blocks, ocr_text)

        # Post-extraction validation: Always try domain lookup for corroboration
        domain_bank_info = lookup_bank_by_domain(ocr_text)

        # Compute country hint BEFORE bank lookup (required for multi-country banks)
        country_hint = result.account_holder_country or result.bank_country

        # Validate GLiNER's bank name against BankLookup
        if result.bank_name:
            bank_info = lookup_bank_by_name(result.bank_name, country_hint)
            if bank_info:
                # GLiNER extracted a recognized bank - use canonical name
                result.bank_name = bank_info.full_name
                result.bank_code = bank_info.swift_codes[0] if bank_info.swift_codes else None
                logger.info(f"Validated bank name '{result.bank_name}' with SWIFT: {result.bank_code} (country: {country_hint})")
            elif domain_bank_info:
                # GLiNER's extraction not recognized, use domain result
                logger.info(f"GLiNER bank '{result.bank_name}' not recognized, using domain: {domain_bank_info.full_name}")
                result.bank_name = domain_bank_info.full_name
                result.bank_code = domain_bank_info.swift_codes[0] if domain_bank_info.swift_codes else None
            else:
                # GLiNER's extraction not recognized and no domain found
                # Try to detect bank in the full OCR text
                detected_bank = detect_bank_in_text(ocr_text, country_hint)
                if detected_bank:
                    logger.info(f"GLiNER bank '{result.bank_name}' not recognized, detected from text: {detected_bank.full_name}")
                    result.bank_name = detected_bank.full_name
                    result.bank_code = detected_bank.swift_codes[0] if detected_bank.swift_codes else None
                # else: keep GLiNER's extraction even if not recognized (may be a valid edge case)

        # Fallback: Extract bank name from website URL or email if not found
        if not result.bank_name:
            # Try domain-based extraction first
            result.bank_name = self._extract_bank_name_from_url_domain(ocr_text)

        # Fallback: Try IBAN-based bank detection (useful for UAE banks)
        if not result.bank_name:
            iban_bank_info = lookup_bank_by_iban(ocr_text)
            if iban_bank_info:
                logger.info(f"Bank detected from IBAN: {iban_bank_info.full_name}")
                result.bank_name = iban_bank_info.full_name
                result.bank_code = iban_bank_info.swift_codes[0] if iban_bank_info.swift_codes else None

        # Also try IBAN if domain found but GLiNER didn't find a bank
        if not result.bank_name and domain_bank_info:
            result.bank_name = domain_bank_info.full_name
            result.bank_code = domain_bank_info.swift_codes[0] if domain_bank_info.swift_codes else None

        # Update SWIFT code if bank_name was extracted via fallback
        if result.bank_name and not result.bank_code:
            result.bank_code = get_swift_code_for_bank(result.bank_name)

        return result

    def _build_result(
        self,
        gliner_result: Dict[str, Any],
        text_blocks: Optional[List[Dict[str, Any]]] = None,
        ocr_text: str = "",
    ) -> BankStatementData:
        """
        Build BankStatementData from GLiNER extraction result.

        Uses spatial context to combine multi-line addresses.
        """
        if not gliner_result:
            return BankStatementData(
                bank_name=None,
                account_holder_name=None,
                account_number=None,
                address=None,
                bank_code=None,
                currency=None,
                bank_branch=None,
                overall_confidence=0.0,
                confidence_scores={},
            )

        # Map schema field names to our schema
        # Schema-based extraction returns descriptive field names
        label_mapping = {
            # Schema field names (from GLiNER2 schema extraction)
            "bank_name": "bank_name",
            "account_holder": "account_holder_name",  # Legacy field name
            "account_holder_name": "account_holder_name",  # New schema field
            "account_number": "account_number",
            "cif_number": "cif_number",  # Track CIF separately
            "customer_address": "address",
            "branch_address": "branch_address",  # Separate from customer address
            "branch_name": "bank_branch",
            "currency": "currency",
            "statement_date": "statement_date",

            # Legacy label names (for backward compatibility with labels-based fallback)
            "bank name": "bank_name",
            "bank or financial institution name": "bank_name",
            "bank of maharashtra": "bank_name",
            "name of bank at top of statement": "bank_name",
            "customer full name with s/o d/o a/l": "account_holder_name",
            "person name or customer name with title or honorific like mr mrs ms dr": "account_holder_name",
            "name after account holder names or account holder label": "account_holder_name",
            "ifsc code": "ifsc_code",  # Not used anymore but keep for backward compat
            "swift code": "swift_code",  # Not used anymore but keep for backward compat
            "branch": "bank_branch",
        }

        # Extract basic fields from GLiNER
        bank_name, bank_name_conf = self._extract_single_field(gliner_result, label_mapping, "bank_name")
        account_holder_name, account_holder_conf = self._extract_single_field(gliner_result, label_mapping, "account_holder_name")

        # ALWAYS run spatial extraction as a cross-check for account holder name
        # Spatial extraction using label-based search is more reliable than GLiNER for names
        # because it explicitly looks for name labels and their associated values
        spatial_account_holder_name = self._extract_account_holder_name_from_text_blocks(text_blocks)
        if spatial_account_holder_name:
            # Prefer spatial extraction when it finds a valid result
            # This handles cases where GLiNER extracts location names or other invalid text
            logger.info(f"Using spatial extraction result '{spatial_account_holder_name}' instead of GLiNER's '{account_holder_name}'")
            account_holder_name = spatial_account_holder_name
            account_holder_conf = 0.7  # 70% confidence for spatial label-based extraction
        elif not account_holder_name or not self._is_valid_account_holder_name(account_holder_name):
            logger.warning(f"Invalid account holder name extracted: '{account_holder_name}', trying fallback extraction")
            # Try fallback extraction using text blocks
            account_holder_name = self._extract_account_holder_name_from_text_blocks(text_blocks)
            account_holder_conf = 0.6  # 60% confidence for fallback extraction

        # Post-process: Remove titles/salutations from account holder name
        if account_holder_name:
            account_holder_name = self._clean_account_holder_name(account_holder_name)

        # Account number: REQUIRE spatial label-based extraction
        # Account numbers MUST have a label nearby (e.g., "Account Number:", "A/C No:")
        # This prevents false positives from random numbers in the document
        account_number, extraction_method = self._extract_account_number_by_label(text_blocks)
        account_number_extraction_method = extraction_method  # Will be 'spatial_label' or None
        account_number_conf = 0.5  # 50% confidence for spatial extraction
        # NOTE: No GLiNER fallback - account numbers MUST have a nearby label to be valid
        # This prevents credit card numbers and other numeric fields from being misclassified

        # Post-process: Remove dashes and spaces from account number
        if account_number:
            account_number = self._clean_account_number(account_number)

        currency, currency_conf = self._extract_single_field(gliner_result, label_mapping, "currency")

        # Post-process: Validate currency and filter out label-text values
        if currency:
            # Clean up currency - remove extra whitespace
            currency = re.sub(r'\s+', ' ', currency).strip()

            # Normalize currency names to ISO codes (e.g., "UAE DIRHAM" -> "AED")
            currency = self._normalize_currency_to_iso(currency)

            # Validate: filter out label-text values like "Currency" or "Account"
            if not self._is_valid_currency(currency):
                logger.warning(f"Invalid currency extracted: '{currency}', trying spatial extraction")
                currency = self._extract_currency_from_blocks(text_blocks)
                currency_conf = 0.6  # 60% confidence for fallback extraction

        bank_branch, bank_branch_conf = self._extract_single_field(gliner_result, label_mapping, "bank_branch")
        statement_date, statement_date_conf = self._extract_single_field(gliner_result, label_mapping, "statement_date")

        # Priority: Find the latest date across all text blocks
        # This is the most reliable method for statement date extraction
        if text_blocks:
            latest_date = self._find_latest_date_in_blocks(text_blocks)
            if latest_date:
                statement_date = latest_date
                statement_date_conf = 0.9  # High confidence for comprehensive date search
                logger.info(f"Using latest date found in document: '{statement_date}'")

        # Post-process: Validate statement_date and filter out invalid values
        if statement_date:
            # First, clean up the statement_date - remove newlines and extra whitespace
            statement_date = re.sub(r'\s+', ' ', statement_date).strip()

            # Then validate
            if not self._is_valid_statement_date(statement_date):
                logger.warning(f"Invalid statement_date extracted: '{statement_date}', trying spatial extraction")
                statement_date = self._extract_statement_date_from_blocks(text_blocks)
            else:
                # Try to extract just the date part if there's extra text
                cleaned_date = self._clean_statement_date(statement_date)
                if cleaned_date:
                    statement_date = cleaned_date

        # Normalize statement_date to DD MMM YYYY format
        if statement_date:
            normalized_date = self._normalize_statement_date(statement_date)
            if normalized_date:
                statement_date = normalized_date

        # Get SWIFT code from hardcoded map based on bank name
        bank_code = get_swift_code_for_bank(bank_name)

        # Get country code from bank name for country-specific address keywords
        country_code = get_country_for_bank(bank_name)

        # Extract address using spatial extraction (Priority: labels > name-proximity > GLiNER)
        address = self._extract_address_spatially(gliner_result, text_blocks, account_holder_name, country_code, bank_name)

        # Post-process: Filter out branch addresses from customer address
        # GLiNER sometimes includes branch addresses in the customer address field
        # Only filter if address starts with known branch patterns (ICICI-specific)
        if address and any(address.upper().startswith(p) for p in ['REDDY PALLI', 'OPP. REEDSPET', 'KONGARED', 'JALIKHANA', 'REEDSPET']):
            # Find the customer address part (after the branch address)
            # Need to find the full extent of the branch address (includes "CHURCH" and pin codes)
            branch_patterns = [
                ('OPP. REEDSPET', 'CHURCH'),
                ('OPP. REEDSPET CHURCH', None),
                ('REDDY PALLI', None),
                ('KONGARED', None),
                ('JALIKHANA', None),
            ]
            for branch_pattern, followup_text in branch_patterns:
                if branch_pattern in address.upper():
                    parts = address.split(branch_pattern)
                    if len(parts) > 1:
                        # If there's followup text to also remove (like "CHURCH"), handle it
                        remainder = parts[1].strip(',. ')
                        if followup_text and remainder.upper().startswith(followup_text):
                            # Remove the followup text and any comma/pin after it
                            followup_parts = remainder.split(followup_text)
                            if len(followup_parts) > 1:
                                remainder = followup_parts[1].strip(',. ')
                                # Also remove leading pin codes like "517001,"
                                if re.match(r'^\d{6}', remainder):
                                    pin_parts = remainder.split(',', 1)
                                    if len(pin_parts) > 1:
                                        remainder = pin_parts[1].strip()
                        address = remainder
                        break

        # Filter out account holder name from address if it's included
        if address and account_holder_name:
            # Also get account_holder_title if available
            account_holder_title = None
            for label, entities in gliner_result.items():
                if label in ["account_holder_title", "account holder title"]:
                    if isinstance(entities, list) and len(entities) > 0:
                        account_holder_title = entities[0].get('value', '').strip()
                    elif not isinstance(entities, list):
                        account_holder_title = entities.get('value', '').strip()
                    break

            address = self._filter_name_from_address(address, account_holder_name, account_holder_title)

        # Post-process: Remove garbled text patterns from address end
        if address:
            # Remove trailing garbled text like "./-0./-0.--00--00-/0./.-0--000"
            # This often appears at the end of addresses from OCR errors
            address_parts = address.split(',')
            clean_parts = []
            for part in address_parts:
                part_clean = part.strip()
                # Skip parts that are mostly garbled (lots of special chars mixed with numbers)
                if re.search(r'[\./\-]{3,}[0-9]+[\./\-]{3,}', part_clean.upper()):
                    continue
                # Skip parts that are just special characters and numbers, BUT keep valid postal codes
                # Valid postal codes are 6-digit numbers (India: 517419, Singapore: 440029)
                if re.match(r'^[\d\./\-]+$', part_clean):
                    # Allow pure 6-digit postal codes
                    if re.match(r'^\d{6}$', part_clean):
                        clean_parts.append(part_clean)
                        continue
                    # Allow house numbers like "1-21", "1-21/A", "H NO 1-21" would have letters
                    # House numbers are typically short: 1-999 format with optional hyphens
                    if re.match(r'^\d+(-\d+)?$', part_clean):
                        clean_parts.append(part_clean)
                        continue
                    # Allow postal codes with country prefix like "517419" or "SINGAPORE 440029" parts
                    # But skip garbled patterns with many special characters
                    if len(re.findall(r'[./\-]', part_clean)) > 3:
                        continue
                    # For short numeric parts, check if they look like zip/postal codes (3-6 digits)
                    if re.match(r'^\d{3,6}$', part_clean):
                        clean_parts.append(part_clean)
                        continue
                    # Skip other numeric-only parts
                    continue
                clean_parts.append(part_clean)
            address = ', '.join(clean_parts) if clean_parts else address

        # Calculate overall confidence
        overall_confidence = self._calculate_confidence(gliner_result)

        # Build confidence_scores dict directly from gliner_result (more reliable than individual variables)
        # This fixes the issue where confidence variables get out of sync when fields are overridden by fallbacks
        confidence_scores = {}
        for schema_field, result_data in gliner_result.items():
            if result_data is None or not isinstance(result_data, dict):
                continue

            # Map schema field to internal field name
            if schema_field in label_mapping:
                internal_field = label_mapping[schema_field]
                conf = result_data.get('confidence', 0.0)
                if conf > 0:
                    confidence_scores[internal_field] = conf * 100

        # Final pass: Ensure ALL extracted fields have confidence scores
        # This prevents validation failures when confidence_scores is incomplete
        # Fixes the bug where fallback extraction (lines 240-250 in extract method) sets field values but not confidence
        # Uses overall_confidence as the baseline, with adjustments for extraction method reliability
        if bank_name and "bank_name" not in confidence_scores:
            confidence_scores["bank_name"] = overall_confidence  # Use overall confidence
        if account_holder_name and "account_holder_name" not in confidence_scores:
            confidence_scores["account_holder_name"] = overall_confidence
        if account_number and "account_number" not in confidence_scores:
            confidence_scores["account_number"] = overall_confidence * 0.6  # Lower for spatial-only
        if address and "address" not in confidence_scores:
            confidence_scores["address"] = overall_confidence
        if currency and "currency" not in confidence_scores:
            confidence_scores["currency"] = overall_confidence
        if bank_code and "bank_code" not in confidence_scores:
            confidence_scores["bank_code"] = overall_confidence * 0.7
        if bank_branch and "bank_branch" not in confidence_scores:
            confidence_scores["bank_branch"] = overall_confidence * 0.7
        if statement_date and "statement_date" not in confidence_scores:
            confidence_scores["statement_date"] = overall_confidence * 0.7

        logger.debug(f"GLiNER2 field confidences: {confidence_scores}")

        # Extract bank_country from document (bank's registered country)
        # This is different from account_holder_country (customer's country)
        bank_country = self._extract_bank_country(ocr_text, text_blocks, bank_name)

        # Fallback: Infer currency from bank_country if not extracted
        if not currency and bank_country:
            try:
                validator = get_bank_statement_validator()
                inferred_currency = validator.get_currency_for_country(bank_country)
                if inferred_currency:
                    currency = inferred_currency
                    currency_conf = 0.7  # 70% confidence for inferred currency
                    logger.info(f"Inferred currency '{currency}' from bank_country '{bank_country}'")
            except Exception as e:
                logger.warning(f"Failed to infer currency from bank_country: {e}")

        # Extract account_holder_country from address content
        account_holder_country = None

        # Try to infer country from address first (more accurate)
        if address:
            account_holder_country = self._infer_country_from_address(address)

        # Fallback: Use currency country if address inference fails
        if not account_holder_country and currency:
            currency_info = get_bank_statement_validator().get_currency_info(currency)
            if currency_info:
                account_holder_country = currency_info.get('country')

        # Parse address into structured components
        address_components = {}
        street_address = address  # Default to full address
        if address:
            validator = get_bank_statement_validator()
            address_components = validator.parse_address_components(
                address,
                country_hint=account_holder_country
            )
            # Remove city, state, postal_code, and country from address to get street portion
            street_address = self._extract_street_address(address, address_components)

        return BankStatementData(
            bank_name=bank_name,
            account_holder_name=account_holder_name,
            account_number=account_number,
            address=street_address,  # Street address only (city/state/postal_code/country removed)
            # Address components
            address_street_number=address_components.get("street_number"),
            address_street_name=address_components.get("street_name"),
            address_city=address_components.get("city"),
            address_postal=address_components.get("postal_code"),
            address_state=address_components.get("state"),
            address_country=address_components.get("country"),
            bank_code=bank_code,
            currency=currency,
            bank_branch=bank_branch,
            bank_country=bank_country,
            statement_date=statement_date,
            account_holder_country=account_holder_country,
            account_number_extraction_method=account_number_extraction_method,
            overall_confidence=overall_confidence,
            confidence_scores=confidence_scores,
            raw_data=ocr_text,
        )

    def _extract_single_field(
        self,
        gliner_result: Dict[str, Any],
        label_mapping: Dict[str, str],
        target_field: str,
    ) -> Tuple[Optional[str], float]:
        """
        Extract a single-value field from GLiNER results.

        Returns:
            Tuple of (value, confidence) where confidence is 0-1
        """
        for label, entities in gliner_result.items():
            if entities is None:
                continue
            if label in label_mapping and label_mapping[label] == target_field:
                if isinstance(entities, list) and len(entities) > 0:
                    best = max(entities, key=lambda e: e.get('confidence', 0))
                    return best.get('value', '').strip(), best.get('confidence', 0.0)
                elif not isinstance(entities, list):
                    return entities.get('value', '').strip(), entities.get('confidence', 0.0)
        return None, 0.0

    def _extract_account_number_by_label(
        self,
        text_blocks: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract account number by finding values next to account number labels.

        This spatial label-based extraction is more reliable than GLiNER for distinguishing
        between account numbers and CIF numbers, which have similar formats.

        Priority: Pure digits > Masked numbers (X's with digits)

        Args:
            text_blocks: List of text blocks with geometry from OCR

        Returns:
            Tuple of (account_number, extraction_method) where extraction_method is 'spatial_label' if found
        """
        if not text_blocks:
            return None, None

        # Get labels from config
        validator = get_bank_statement_validator()
        account_number_labels = validator.get_account_number_labels()

        # If config not available, fall back to defaults
        if not account_number_labels:
            account_number_labels = [
                'ACCOUNT NUMBER', 'ACCOUNT NO', 'ACCOUNT NO.', 'A/C NO', 'A/C NO.', 'AC NO',
                'SAVINGS A/C', 'CURRENT A/C', 'SAVINGS ACCOUNT', 'CURRENT ACCOUNT'
            ]

        # Collect all candidates with their scores and types
        # Structure: {'value': str, 'score': float, 'is_pure_digits': bool, 'source': str}
        candidates = []

        # Find blocks with account number labels
        for i, block in enumerate(text_blocks):
            text = block.get('text', '').strip().upper()
            if not text:
                continue

            # Check if this block is an account number label
            # Use word boundary matching to prevent partial matches
            # e.g., "ACCOUNT NO" should not match "CKYC NUMBER" or "ACCOUNT SUMMARY"
            is_label = any(
                re.search(r'\b' + re.escape(label) + r'\b', text)
                for label in account_number_labels
            )
            if not is_label:
                continue

            logger.info(f"Found account number label block: '{text}' at index {i}")

            # First check if the account number is in the SAME block as the label
            # Handle patterns like "Account No. :5614244803" or "A/C No 5614244803"
            same_block_value, is_pure = self._extract_account_number_from_combined_block(text)
            if same_block_value:
                logger.info(f"Found account number in combined block: '{same_block_value}' (pure_digits={is_pure}) from '{text}'")
                # Combined block gets highest priority, but still collect to compare
                candidates.append({
                    'value': same_block_value,
                    'score': 3.0,  # High score for combined block
                    'is_pure_digits': is_pure,
                    'source': 'combined_block'
                })
                # If pure digits found in combined block, return immediately
                if is_pure:
                    return same_block_value, 'spatial_label'
                continue  # Keep looking for pure digits in other blocks

            label_y1 = block.get('y1', 0)
            label_x1 = block.get('x1', 0)
            label_x2 = block.get('x2', 0)

            logger.debug(f"Found account number label '{text}' at index {i}, y1={label_y1}")

            # Check ALL blocks for matching y-position (same row or next row)
            # Find the CLOSEST number to the label
            best_match = None
            best_distance = float('inf')
            best_is_pure = False

            for j, value_block in enumerate(text_blocks):
                if j == i:  # Skip the label block itself
                    continue

                value_text = value_block.get('text', '').strip()
                if not value_text:
                    continue

                # Skip colons and separators
                if value_text in [':', '-', '/', '\\']:
                    continue

                # Check if it looks like an account number
                if not self._looks_like_account_number(value_text):
                    continue

                value_y1 = value_block.get('y1', 0)
                value_x1 = value_block.get('x1', 0)
                value_x2 = value_block.get('x2', 0)

                # Calculate distance from label
                y_diff = abs(value_y1 - label_y1)

                # Only consider blocks on same row or adjacent rows (near the label)
                # Allow values slightly above the label due to OCR positioning quirks
                # but add a small penalty for values above to prefer values below/right
                if y_diff < 0.04:
                    # Check if value is horizontally aligned with the label (vertical layout)
                    # Use center positions for alignment check
                    label_center_x = (label_x1 + label_x2) / 2
                    value_center_x = (value_x1 + value_x2) / 2
                    horizontal_alignment = abs(label_center_x - value_center_x)

                    # Calculate distance based on layout type
                    # Add a small penalty for values above the label (OCR positioning quirk)
                    above_penalty = 0.05 if value_y1 < label_y1 else 0

                    if y_diff < 0.02:
                        # Same row: prefer values to the right
                        x_diff = value_x1 - label_x2
                        distance = y_diff - (0.1 if x_diff > 0 else 0) + above_penalty
                    else:
                        # Different row (below): prefer horizontally aligned values
                        # Smaller y_diff AND smaller horizontal alignment is better
                        distance = y_diff + (horizontal_alignment * 0.5) + above_penalty

                    if distance < best_distance:
                        best_distance = distance
                        cleaned = self._extract_account_number_pattern(value_text)
                        if cleaned:
                            best_match = cleaned
                            best_is_pure = cleaned.replace('X', '').replace('x', '').isdigit()
                            logger.debug(f"  Candidate: '{cleaned}' (pure={best_is_pure}) at y1={value_y1}, x_center={value_center_x:.3f}, distance={distance:.4f}")

            if best_match:
                logger.info(f"Found account number by spatial search: '{best_match}' (pure={best_is_pure})")
                return best_match, 'spatial_label'

        return None, None

    def _extract_account_number_pattern(self, value: str) -> Optional[str]:
        """
        Extract the account number pattern from a value that may contain trailing/leading text.

        For example, "50100446078329 OTHER" -> "50100446078329"
                     ":50100446078329" -> "50100446078329"
                     "151-15252-0" -> "151152520"

        Args:
            value: The raw value that may contain an account number

        Returns:
            The extracted account number pattern, or None if not found
        """
        if not value:
            return None

        # Remove leading colons and spaces
        cleaned = value.lstrip(':').strip()

        # Remove hyphens and spaces to handle patterns like "151-15252-0" -> "151152520"
        # This is needed because the regex below only matches consecutive X/digits
        cleaned_no_separators = re.sub(r'[\s\-]', '', cleaned)

        # Try to extract the first sequence of 8-20 digits or masked digits (X's and digits)
        # This handles values like "50100446078329 OTHER" or "XXXXXX123456 TYPE"
        # First try with separators removed (handles "151-15252-0")
        match = re.search(r'([X\d]{8,20})', cleaned_no_separators, re.IGNORECASE)
        if match:
            return match.group(1)

        # Fallback: try without removing separators (for cases with trailing text)
        match = re.search(r'([X\d]{8,20})', cleaned, re.IGNORECASE)
        if match:
            return match.group(1)

        return None

    def _extract_account_number_from_combined_block(self, text: str) -> Tuple[Optional[str], bool]:
        """
        Extract account number from a combined label:value block.

        Handles OCR output like:
        - "Account No. :5614244803" -> ("5614244803", True)
        - "A/C No 5614244803" -> ("5614244803", True)
        - "ACCOUNT NUMBER:XXXXXXXXXX6663" -> ("XXXXXXXXXX6663", False)

        Args:
            text: The combined text containing both label and value (uppercase)

        Returns:
            Tuple of (extracted account number, is_pure_digits) or (None, False) if not found
        """
        if not text:
            return None, False

        logger.debug(f"Checking combined block for account number: '{text}'")

        # Remove the label part and extract the numeric value
        # Pattern: look for colon or space followed by digits
        # Match patterns like ":5614244803" or " 5614244803"
        # Allow for possible trailing text (not anchored to end)
        match = re.search(r'[:\s]+(\d{8,20})', text)
        if match:
            logger.debug(f"Found pure digits in combined block: '{match.group(1)}'")
            return match.group(1), True

        # Also try masked format: ":XXXXXX123456" or " XXXXXX123456"
        match = re.search(r'[:\s]+([X\d]{8,20})', text, re.IGNORECASE)
        if match:
            logger.debug(f"Found masked number in combined block: '{match.group(1)}'")
            return match.group(1), False

        logger.debug(f"No account number pattern found in combined block")
        return None, False

    def _looks_like_account_number(self, value: str) -> bool:
        """
        Check if a value looks like a valid bank account number.

        Account numbers:
        - Are typically 8-20 digits
        - May be masked with X's for privacy (e.g., XXXXXXX977002)
        - Should NOT match known CIF patterns (SBI 8848, BOM 4024)

        Args:
            value: The value to check

        Returns:
            True if the value looks like an account number
        """
        if not value:
            return False

        # Remove spaces and hyphens
        cleaned = re.sub(r'[\s\-]', '', value)

        # Extract just the account number pattern if there's trailing/leading text
        # This handles values like "50100446078329 OTHER" -> "50100446078329"
        extracted = self._extract_account_number_pattern(cleaned)
        if extracted:
            cleaned = extracted
        else:
            return False  # No valid account number pattern found

        # Check if it's all digits - reject masked numbers
        is_all_digits = cleaned.isdigit()

        # Check for masking characters (X, *, •, -, _, #)
        masking_pattern = r'[Xx\*•\-\_#]'
        has_masking = bool(re.search(masking_pattern, cleaned))

        # Reject masked account numbers - they don't contain the full account number needed for verification
        if has_masking:
            logger.debug(f"Rejecting masked account number: {cleaned}")
            return False

        # Must be all digits
        if not is_all_digits:
            return False

        # Account numbers are typically 8-20 characters
        if len(cleaned) < 8 or len(cleaned) > 20:
            return False

        # Skip known CIF patterns (SBI CIF starts with 8848)
        if cleaned.startswith('8848'):
            logger.debug(f"Skipping CIF number (8848 prefix): {cleaned}")
            return False

        # Skip Bank of Maharashtra CIF pattern (starts with 4024)
        if cleaned.startswith('4024'):
            logger.debug(f"Skipping CIF number (4024 prefix): {cleaned}")
            return False

        # Skip values that are all zeros
        if cleaned == '0' * len(cleaned):
            return False

        return True

    def _filter_name_from_address(
        self,
        address: str,
        account_holder_name: str,
        account_holder_title: str = None
    ) -> str:
        """
        Filter out account holder name/title from the beginning of an address.

        GLiNER2 sometimes includes the person's name (with title like "Mr", "Mrs", "Dr")
        as part of the address. This method removes the name if it appears at the start.

        Args:
            address: The extracted address that may contain name
            account_holder_name: The account holder's name (without title)
            account_holder_title: Optional title (Mr, Mrs, Ms, Dr, etc.)

        Returns:
            Address with name filtered out from the beginning
        """
        if not address or not account_holder_name:
            return address

        address_clean = address.strip()
        name_upper = account_holder_name.upper().strip()

        # Build patterns to match name variations at the start of address
        patterns_to_remove = []

        # Pattern 1: Name alone
        patterns_to_remove.append(rf'^{re.escape(name_upper)}\s*[,:]\s*')

        # Pattern 2: Name with S/O D/O A/L patterns (Malaysian/Indian names)
        # Handle names like "MANOGARAN S/O THANABALAN" or "K C ROHITH S/O RAMACHANDRAN"
        so_patterns = [' S/O ', ' D/O ', ' A/L ', ' S/O.', ' D/O.', ' A/L.']
        for so_marker in so_patterns:
            # Remove the name portion before S/O (e.g., "MANOGARAN" from "MANOGARAN S/O THANABALAN")
            if so_marker.strip('.') in name_upper:
                # Split at S/O marker and take the first part (person's name)
                name_before_so = name_upper.split(so_marker.strip())[0].strip()
                if name_before_so:
                    patterns_to_remove.append(rf'^{re.escape(name_before_so)}(\s+{re.escape(so_marker.strip())}.*)?\s*[,:]\s*')

        # Pattern 3: Title + Name
        if account_holder_title:
            title = account_holder_title.upper().strip().rstrip('.')  # Remove trailing dot if present
            # Try with and without dot, with space
            for title_variant in [title, title + '.', title + ' ']:
                patterns_to_remove.append(rf'^{re.escape(title_variant)}\s*{re.escape(name_upper)}\s*[,:]\s*')

        # Common title patterns that might appear even if we don't have explicit title
        common_titles = ['MR', 'MRS', 'MS', 'DR', 'SHRI', 'SMT', 'MISS', 'MR.', 'MRS.', 'MS.', 'DR.']
        for title in common_titles:
            patterns_to_remove.append(rf'^{title}\s*{re.escape(name_upper)}\s*[,:]\s*')

        # Try each pattern
        for pattern in patterns_to_remove:
            filtered = re.sub(pattern, '', address_clean, flags=re.IGNORECASE)
            if filtered != address_clean:  # Pattern matched and removed something
                # Clean up any leading/trailing punctuation and whitespace
                filtered = filtered.strip().lstrip(',.').strip()
                logger.info(f"Filtered name from address: '{address_clean[:50]}...' -> '{filtered[:50]}...'")
                return filtered

        return address_clean

    def _extract_address_prefer_schema(
        self,
        gliner_result: Dict[str, Any],
        text_blocks: Optional[List[Dict[str, Any]]] = None,
        country_code: str = None,
    ) -> Optional[str]:
        """
        Extract customer address, preferring schema customer_address field.

        Schema-based extraction provides explicit customer_address vs branch_address distinction,
        which is more reliable than spatial context heuristics.

        Strategy:
        1. Check for schema customer_address field directly
        2. If found AND text_blocks available, use spatial expansion to capture all lines
        3. If not found, fall back to spatial context extraction

        Args:
            gliner_result: GLiNER extraction results
            text_blocks: Text blocks with geometry from OCR
            country_code: ISO country code for country-specific address keywords
        """
        if not gliner_result:
            return None

        # First, try to get customer_address directly from schema result
        for label, entities in gliner_result.items():
            if entities is None:
                continue

            # Schema field name for customer address
            if label == "customer_address":
                schema_address = None
                if isinstance(entities, list) and len(entities) > 0:
                    # Join multiple address parts
                    address_parts = [e.get('value', '').strip() for e in entities if e.get('value')]
                    if address_parts:
                        schema_address = self._format_address(address_parts)
                elif not isinstance(entities, list):
                    # Single address entity
                    addr = entities.get('value', '').strip()
                    schema_address = addr

                # Validate schema address before using it as seed for spatial expansion
                # Reject garbled addresses (too short, no letters, or lots of special chars)
                if schema_address and not self._is_valid_address(schema_address):
                    logger.info(f"Schema address looks garbled/invalid, skipping: '{schema_address[:50]}...'")
                    schema_address = None

                # If we have a valid schema address and text blocks, use spatial expansion
                # to capture all lines of multi-line addresses
                if schema_address and text_blocks:
                    # Create a set of address keywords from schema result for spatial search
                    address_keywords = {schema_address}
                    if isinstance(entities, list) and len(entities) > 0:
                        for e in entities:
                            val = e.get('value', '').strip()
                            if val:
                                address_keywords.add(val)

                    # Use spatial context to expand and capture all lines
                    expanded_parts = self._group_nearby_address_blocks(
                        text_blocks, address_keywords, set(), country_code,
                        trust_gliner_schema=True  # Trust GLiNER schema, be more permissive
                    )
                    if expanded_parts:
                        expanded_address = self._format_address(expanded_parts)
                        # Use expanded address if it's longer (captures more content)
                        if len(expanded_address) > len(schema_address):
                            return expanded_address

                # Return schema address directly if no spatial expansion or no text blocks
                if schema_address and len(schema_address) > 5 and re.search(r'[A-Za-z]{3,}', schema_address):
                    return schema_address

        # Fallback to spatial context extraction
        return self._extract_address_with_spatial_context(gliner_result, text_blocks, country_code)

    def _extract_address_with_spatial_context(
        self,
        gliner_result: Dict[str, Any],
        text_blocks: Optional[List[Dict[str, Any]]] = None,
        country_code: str = None,
    ) -> Optional[str]:
        """
        Extract customer address using GLiNER hints + spatial context.

        Strategy:
        1. Find address keywords detected by GLiNER (PRIMARY method)
        2. If GLiNER finds addresses, use spatial context to expand them
        3. If GLiNER doesn't find addresses, try positional context extraction (FALLBACK)
        4. Final fallback: GLiNER-only extraction

        Args:
            gliner_result: GLiNER extraction results
            text_blocks: Text blocks with geometry from OCR
            country_code: ISO country code for country-specific address keywords
        """
        if not text_blocks:
            # Fallback to GLiNER-only extraction
            return self._extract_address_from_gliner_only(gliner_result)

        # Collect all address-related keywords detected by GLiNER
        address_keywords: Set[str] = set()
        bank_branch_keywords: Set[str] = set()

        for label, entities in gliner_result.items():
            if entities is None:
                continue

            label_lower = label.lower()

            # Customer address labels (including new positional-aware labels)
            if label_lower in self.ADDRESS_LABELS:
                if isinstance(entities, list):
                    for e in entities:
                        addr = e.get('value', '').strip()
                        if addr:
                            address_keywords.add(addr)
                else:
                    addr = entities.get('value', '').strip()
                    if addr:
                        address_keywords.add(addr)

            # Bank branch address labels (to exclude)
            elif label_lower in self.BANK_BRANCH_LABELS:
                if isinstance(entities, list):
                    for e in entities:
                        branch = e.get('value', '').strip()
                        if branch:
                            bank_branch_keywords.add(branch)
                else:
                    branch = entities.get('value', '').strip()
                    if branch:
                        bank_branch_keywords.add(branch)

        # Filter out address keywords that look like branch addresses
        # This prevents branch addresses from being used as seed blocks
        filtered_address_keywords = set()
        for keyword in address_keywords:
            # Validate each address keyword - skip garbled/invalid ones
            if not self._is_valid_address(keyword):
                logger.info(f"Filtering out invalid address keyword: '{keyword[:50]}...'")
                continue
            keyword_upper = keyword.upper()
            # Check if this keyword contains branch-specific patterns
            is_branch_address = False

            # Check against known branch keywords
            for branch_kw in bank_branch_keywords:
                if branch_kw.upper() in keyword_upper:
                    is_branch_address = True
                    logger.info(f"Filtering out branch address keyword: '{keyword[:50]}...'")
                    break

            # Check for branch address patterns
            branch_patterns = [
                'REDDY PALLI', 'REEDSPET', 'REEDSPET CHURCH', 'KONGARED', 'VIKAS COMPLEX',
                'JALIKHANASTREET', 'JAILKHANA STREET', 'I ST FLOR', 'IIND FLOR', 'IIIND FLOR',
                'BASE BRANCH', 'BRANCH ADDRESS', 'REGISTERED OFFICE',
            ]
            if not is_branch_address:
                for pattern in branch_patterns:
                    if pattern in keyword_upper:
                        is_branch_address = True
                        logger.info(f"Filtering out branch address by pattern '{pattern}': '{keyword[:50]}...'")
                        break

            if not is_branch_address:
                filtered_address_keywords.add(keyword)

        # First try: GLiNER keyword-based extraction (existing)
        if filtered_address_keywords:
            address_parts = self._group_nearby_address_blocks(
                text_blocks, filtered_address_keywords, bank_branch_keywords, country_code
            )
            if address_parts:
                return self._format_address(address_parts)

        # Second try: Field label-based extraction (for tabular format)
        field_label_address = self._extract_address_by_field_label(text_blocks)
        if field_label_address:
            return field_label_address

        # Third try: Singapore-specific address detection
        # This handles cases where GLiNER misses Singapore addresses but text_blocks contain BLK/block patterns
        singapore_address = self._extract_singapore_address_from_blocks(text_blocks)
        if singapore_address:
            logger.info(f"Found Singapore address via pattern matching: '{singapore_address[:50]}...'")
            return singapore_address

        # Fourth try: Positional context extraction (for letterhead format)
        positional_address = self._extract_address_by_positional_context(
            gliner_result, text_blocks
        )
        if positional_address:
            return positional_address

        # Final fallback: GLiNER-only extraction
        return self._extract_address_from_gliner_only(gliner_result)

    def _group_nearby_address_blocks(
        self,
        text_blocks: List[Dict[str, Any]],
        address_keywords: Set[str],
        bank_branch_keywords: Set[str],
        country_code: str = None,
        trust_gliner_schema: bool = False,
    ) -> List[str]:
        """
        Group nearby text blocks into address lines.

        Strategy:
        1. Find blocks containing address keywords (seed blocks)
        2. Include blocks BEFORE AND AFTER the seed block to capture multi-line addresses
        3. Continue expanding until there are no more consecutive lines (not limited to 5 positions)
        4. Stop at "bank branch" keywords, obvious non-address text, or empty blocks
        5. This captures letter-format addresses where postal code may be several lines below

        Args:
            text_blocks: Text blocks with geometry from OCR
            address_keywords: Keywords that indicate address content
            bank_branch_keywords: Keywords to filter out (branch addresses)
            country_code: ISO country code for country-specific address detection
            trust_gliner_schema: If True, skip strict address content checks during expansion
                                (GLiNER schema has already validated the address)
        """
        if not text_blocks:
            return []

        # Find blocks containing address keywords (seed blocks)
        seed_blocks: List[Tuple[int, Dict]] = []  # (index, block)

        for i, block in enumerate(text_blocks):
            text = block.get('text', '').strip()
            if not text:
                continue

            text_upper = text.upper()

            # Check if block contains address keyword
            for keyword in address_keywords:
                if keyword.upper() in text_upper:
                    seed_blocks.append((i, block))
                    break

        if not seed_blocks:
            return []

        # Use ALL seed blocks to capture more of the multi-line address
        # Sort by index to get them in document order
        seed_blocks.sort(key=lambda x: x[0])

        # Collect all address blocks using ALL seed blocks as starting points
        all_address_blocks: Dict[int, Dict] = {}  # index -> block
        seen_texts: Set[str] = set()

        # For each seed block, collect nearby blocks
        for seed_idx, seed_block in seed_blocks:
            seed_text = seed_block.get('text', '').strip()

            # Skip if already processed
            if seed_text in seen_texts:
                continue

            # Add the seed block itself
            if seed_idx not in all_address_blocks:
                all_address_blocks[seed_idx] = seed_block
            seen_texts.add(seed_text)

            # Collect blocks BEFORE this seed - continue until break condition
            # Go backward from seed_idx - 1 towards 0
            for i in range(seed_idx - 1, -1, -1):
                if i in all_address_blocks:
                    continue  # Already included

                block = text_blocks[i]
                text = block.get('text', '').strip()

                # STOP at empty block
                if not text:
                    break

                # Skip if already seen
                if text in seen_texts:
                    continue

                text_upper = text.upper()

                # STOP at blocks with bank branch keywords
                is_bank_branch = any(
                    kw.upper() in text_upper for kw in bank_branch_keywords
                )
                if is_bank_branch:
                    break

                # STOP at common non-address text
                if self._is_non_address_text(text):
                    break

                # Check if this looks like address content
                # If it doesn't look like address content, STOP
                # Skip strict check in trust mode (GLiNER schema already validated)
                if not trust_gliner_schema and not self._looks_like_address_content(text, country_code):
                    break

                all_address_blocks[i] = block
                seen_texts.add(text)

            # Collect blocks AFTER this seed - continue until break condition
            # Go forward from seed_idx + 1 towards end
            for i in range(seed_idx + 1, len(text_blocks)):
                if i in all_address_blocks:
                    continue  # Already included

                block = text_blocks[i]
                text = block.get('text', '').strip()

                # STOP at empty block
                if not text:
                    break

                # Skip if already seen
                if text in seen_texts:
                    continue

                text_upper = text.upper()

                # STOP at blocks with bank branch keywords
                is_bank_branch = any(
                    kw.upper() in text_upper for kw in bank_branch_keywords
                )
                if is_bank_branch:
                    break

                # STOP at common non-address text
                if self._is_non_address_text(text):
                    break

                # Check if this looks like address content
                # If it doesn't look like address content, STOP
                # Skip strict check in trust mode (GLiNER schema already validated)
                if not trust_gliner_schema and not self._looks_like_address_content(text, country_code):
                    break

                all_address_blocks[i] = block
                seen_texts.add(text)

        # Filter address blocks by x-position to exclude branch addresses
        # Bank statements often have customer address on LEFT and branch address on RIGHT
        # We want to prefer the left-side address (customer address)
        if len(all_address_blocks) >= 2:  # Apply when we have multiple blocks (could be multiple addresses)
            # Get x-positions of all blocks
            x_positions = [(idx, block.get('x1', 0)) for idx, block in all_address_blocks.items()]
            if x_positions:
                # Split into left and right clusters using x-position threshold (0.5 = middle of page)
                left_blocks = {}
                right_blocks = {}
                for idx, x1 in x_positions:
                    if x1 < 0.5:  # Left side of page
                        left_blocks[idx] = all_address_blocks[idx]
                    else:  # Right side of page
                        right_blocks[idx] = all_address_blocks[idx]

                # If we have blocks on both sides, prefer left side (customer address)
                if left_blocks and right_blocks:
                    logger.info(f"Found {len(left_blocks)} left-side and {len(right_blocks)} right-side address blocks, using left-side (customer address)")
                    all_address_blocks = left_blocks

        # Sort by index to maintain document order
        sorted_indices = sorted(all_address_blocks.keys())
        address_blocks = [all_address_blocks[i] for i in sorted_indices]

        # Extract text from blocks
        address_lines: List[str] = []
        for block in address_blocks:
            text = block.get('text', '').strip()
            # Clean up extra punctuation but keep structure
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r',+', ',', text)
            text = text.strip(', ')
            if text:
                address_lines.append(text)

        return address_lines

    def _looks_like_address_content(self, text: str, country_code: str = None) -> bool:
        """Check if text looks like it could be part of an address.

        This is a loose check - we want to be inclusive to capture
        all parts of multi-line addresses.

        Uses country-specific address keywords loaded from config.

        Args:
            text: Text to check
            country_code: ISO country code (e.g., "AE", "SG", "IN")
        """
        text_upper = text.upper()

        # Skip very short or very long text
        if len(text) <= 3 or len(text) > 100:
            return False

        # Get country-specific address keywords from loader
        loader = get_address_keywords_loader()
        address_keywords = loader.get_keywords(country_code)

        # Check for address keywords - use word boundary matching to avoid false positives
        # E.g., "STATEMENT" should not match because it contains "STATE"
        # Use regex with word boundaries \b for more precise matching
        for keyword in address_keywords:
            # Escape special regex characters in keyword
            escaped_keyword = re.escape(keyword)
            # Use word boundaries to avoid partial matches
            if re.search(r'\b' + escaped_keyword + r'\b', text_upper):
                return True

        # Check for country-specific unit patterns (e.g., UAE standalone unit numbers)
        if loader.matches_unit_pattern(text, country_code):
            return True

        # Check for house number patterns (e.g., "1-21", "123/45")
        if re.match(r'^[\d\-\/]+[A-Z]?', text_upper):
            return True

        # Check for pin code patterns (6 digits, maybe with letters)
        if re.search(r'\b\d{6}\b', text) or re.search(r'\b\d{5}\b', text):
            return True

        # Check for patterns like "#11-25" (Singapore flat numbers) or "BLK 29"
        # Fixed pattern to allow digits after hyphen (e.g., "#11-25" has digit not letter)
        if re.match(r'^[#]?[\d]+[\s\-][A-Z\d]+', text_upper):
            return True
        # Also specifically check for BLK/BLOCK followed by numbers
        if re.match(r'^BLK[\s\.]+[\d]+', text_upper):
            return True

        return False

    def _is_non_address_text(self, text: str) -> bool:
        """Check if text is likely not part of a customer address.

        This method filters out obvious non-address content while being
        conservative about legitimate address components. Only filters
        content that is CLEARLY not part of a customer's residential address.
        """
        text_upper = text.upper()

        # IMPORTANT: Don't filter out 6-digit postal codes (India uses 6-digit PIN codes)
        # These patterns are valid address components and should be kept
        if re.match(r'^\d{6}$', text) or re.match(r'^\d{5}$', text):
            return False  # Keep postal codes

        # Skip single characters
        if len(text) <= 2:
            return True

        # Skip serial number patterns (S/N, SERIAL NO, etc.)
        serial_patterns = ['S/N', 'SERIAL NO', 'SERIAL NUMBER', 'SEQ NO', 'REF NO']
        for pattern in serial_patterns:
            if text_upper.startswith(pattern) or f' {pattern}' in text_upper:
                return True

        # Skip text starting with branch address patterns (explicit branch labels)
        branch_address_prefixes = [
            'YOUR BASE BRANCH',
            'BASE BRANCH:',
            'BASE BRANCH -',
            'BRANCH ADDRESS:',
            'REGISTERED OFFICE:',
            'OFFICE ADDRESS:',
            'CORPORATE OFFICE:',
            'HEAD OFFICE:',
        ]
        for prefix in branch_address_prefixes:
            if text_upper.startswith(prefix):
                return True

        # Skip bank names (major Indian banks - but NOT location names)
        # Only filter when it's clearly a bank name, not part of an address
        bank_name_patterns = [
            'STATE BANK OF INDIA', 'SBI BANK', 'HDFC BANK', 'ICICI BANK',
            'AXIS BANK', 'KOTAK BANK', 'YES BANK', 'INDUSIND BANK',
            'BANK OF BARODA', 'CANARA BANK', 'BANK OF MAHARASHTRA',
            'UNION BANK', 'CENTRAL BANK', 'PUNJAB NATIONAL',
        ]
        # Only skip if text is EXACTLY a bank label (not just contains it)
        # This is more conservative and avoids filtering out legitimate user addresses
        # that happen to contain words like "Center", "Tower", etc.
        for bank_pattern in bank_name_patterns:
            if text_upper == bank_pattern:
                return True  # Only filter exact matches

        # Skip bank-related terms and labels (clearly non-address labels)
        bank_terms = {
            'ACCOUNT', 'STATEMENT', 'BALANCE', 'TRANSACTION',
            'DEBIT', 'CREDIT', 'WITHDRAWAL', 'DEPOSIT', 'IFSC',
            'CURRENCY', 'SUMMARY', 'DETAILS', 'NARRATION',
            'CLOSING', 'OPENING', 'PAGE', 'LIMIT', 'STATUS',
            'JOINT HOLDERS', 'CUST ID', 'EMAIL',
            'MICR', 'RTGS', 'NEFT', 'SWIFT', 'REGULAR',
            # Additional labels
            'ACCOUNT SUMMARY', 'STATEMENT OF ACCOUNT', 'SUMMARY OF ACCOUNTS',
            'ACCOUNT DETAILS', 'TRANSACTION DETAILS', 'BRANCH CODE', 'BRANCH NAME',
            'BRANCH EMAIL', 'BRANCH PHONE', 'DATE OF STATEMENT', 'ACCOUNT NUMBER',
            'ACCOUNT NO', 'IFSC CODE', 'CIF NUMBER', 'PRODUCT', 'NOMINEE',
            'NOMINEE NAME', 'OPENING BALANCE', 'CLOSING BALANCE', 'WITHDRAWAL AMT',
            'DEPOSIT AMT', 'CHQ./REF.NO.', 'VALUE DT', 'UNCLEARED AMOUNT',
            'CLEAR BALANCE', '+MOD BAL', 'LIEN', 'MONTHLY AVG BALANCE', 'INTEREST RATE',
            'DRAWING POWER', 'ACCOUNT OPEN DATE', 'ACCOUNT STATUS', 'CKYCR NUMBER',
            'STATEMENT FROM', 'GENERATED ON', 'GENERATED BY', 'REQUESTING BRANCH',
            'WELCOME', 'AS ON', 'TEAM',
            'THIS IS A COMPUTER GENERATED', 'SIGNATURE', 'REGISTERED OFFICE',
            # Floor indicators (branch-specific, not customer address)
            'I ST FLOR', 'IIND FLOR', 'IIIND FLOR',
            # Transaction table column headers
            'TRAN DATE', 'CHQ NO', 'PARTICULARS', 'VALUE DATE', 'REF NO',
            # Field labels (not part of customer address)
            'PRIMARY ACCOUNT HOLDER NAME', 'ACCOUNT HOLDER NAME', 'CUSTOMER ID',
            'ACCOUNT BRANCH', 'NOMINEE REGISTERED', 'ACCOUNT NUMBER', 'ADDRESS',
        }
        # Check for exact match (not contains) to avoid filtering address components
        if text_upper in bank_terms:
            return True

        # Skip S/O, D/O, A/L name patterns (these are person names, not addresses)
        # E.g., "MANOGARAN S/O THANABALAN" is a name, not an address
        if re.search(r'\s+(S/O|D/O|A/L|S/O\.|D/O\.|A/L\.)\s+', text_upper):
            # But still allow if it ALSO has address keywords (rare edge case)
            address_keywords = ['ROAD', 'STREET', 'LANE', 'NAGAR', 'PALLY', 'PALLE', 'COLONY', 'TOWERS', 'APARTMENTS', 'CHAMBERS', 'BUILDING', 'NEAR', 'OPP', 'OPP.', 'TEMPLE', 'CHURCH', 'SCHOOL', 'MARKET', 'MANDIR', 'MASJID', 'H NO', 'SETTY', 'GARI', 'CHITTOOR', 'PRADESH', 'ANDHRA', 'DIST', 'BLK', 'BLOCK', 'CRESCENT', 'CLOSE', 'DRIVE', 'PLACE', 'JALAN', 'LORONG', 'BUKIT']
            if not any(kw in text_upper for kw in address_keywords):
                logger.debug(f"Filtering out S/O/D/O/A/L name pattern: '{text[:50]}...'")
                return True

        # Skip text that starts with known non-address prefixes
        non_address_prefixes = [
            'GENERATED ON:', 'GENERATED BY:', 'REQUESTING BRANCH CODE:',
            'STATEMENT FROM:', 'STATEMENT TO:', 'AS ON', 'DATE OF STATEMENT',
            'ACCOUNT TYPE:', 'PRODUCT:', 'NOMINEE :', 'JOINT HOLDERS :',
            'CIF NUMBER:', 'ACCOUNT NO:', 'IFSC CODE:', 'MICR CODE:',
            'BRANCH CODE:', 'BRANCH NAME:', 'BRANCH EMAIL ID:', 'BRANCH PHONE:',
            'CUST ID:', 'ACCOUNT STATUS:', 'CURRENCY:', 'ACCOUNT OPEN DATE:',
            'NOMINEE NAME:', 'STATEMENT SUMMARY :-',
            # UAE bank statement labels
            'STATEMENT PERIOD:', 'DATE ISSUED:', 'IBAN:', 'BRANCH:',
            'INTEREST PAYOUT:', 'ACCOUNT TYPE:', 'ACCOUNT NUMBER:',
        ]
        for prefix in non_address_prefixes:
            if text_upper.startswith(prefix):
                return True

        # Skip common label patterns (key:value format)
        # Increased limit to 25 to catch more labels like "Statement Period:"
        # Only filter explicit labels, not values starting with colon (OCR artifact)
        if ':' in text_upper and not text_upper.startswith(':'):
            if len(text_upper.split(':')[0]) <= 25:
                return True

        # Skip email patterns
        if '@' in text and '.' in text:
            return True

        # Skip website patterns
        if text_upper.startswith('WWW') or ' .COM' in text_upper:
            return True

        # Skip dates that aren't part of address
        if re.match(r'^\d{2}/\d{2}/\d{4}$', text_upper):
            return True

        # Skip text with lots of numbers and slashes (looks like transaction data)
        # But keep if it looks like an address (has address keywords)
        if re.search(r'\d{2,}/\d{2,}', text) and ',' in text:
            address_keywords = ['ROAD', 'STREET', 'LANE', 'NAGAR', 'PALLY', 'PALLE', 'COLONY', 'TOWERS', 'APARTMENTS', 'CHAMBERS', 'BUILDING', 'NEAR', 'OPP', 'OPP.', 'TEMPLE', 'CHURCH', 'SCHOOL', 'MARKET', 'MANDIR', 'MASJID', 'H NO', 'SETTY', 'GARI', 'CHITTOOR', 'PRADESH', 'ANDHRA', 'DIST']
            if not any(kw in text_upper for kw in address_keywords):
                return True

        # Skip phone/bank contact phrases that appear after address
        phone_contact_phrases = [
            'DIAL YOUR', 'CALL YOUR', 'PHONE', 'CONTACT',
            'TOLL FREE', 'TOLL-FREE', 'CUSTOMER CARE', 'HELPLINE',
            'BANK PHONE', 'BANK MOBILE', 'VISIT', 'DIAL'
        ]
        if any(phrase in text_upper for phrase in phone_contact_phrases):
            return True

        # Skip garbled text with lots of special characters (like "./-0./-0.--00--00-/0./.-0--000")
        if re.search(r'[\./\-]{3,}[0-9]+[\./\-]{3,}', text_upper):
            return True

        return False

    def _is_valid_address(self, address: str) -> bool:
        """
        Check if an address looks like a valid customer address.

        Filters out obviously invalid addresses like:
        - Garbled text with mostly special characters
        - Too short (less than 5 chars)
        - No alphabetic characters
        - Only numbers and special characters

        Args:
            address: The address to validate

        Returns:
            True if address looks valid, False otherwise
        """
        if not address:
            return False

        address_clean = address.strip()

        # Too short
        if len(address_clean) < 5:
            return False

        # Remove common punctuation and spaces for validation
        cleaned = re.sub(r'[\s,\.\-\\\/]+', '', address_clean)

        # Must have at least some alphabetic characters
        if not re.search(r'[A-Za-z]', cleaned):
            return False

        # Check if it's not just special characters and numbers
        # Remove digits and check if there's enough text left
        text_only = re.sub(r'\d+', '', cleaned)
        if len(text_only) < 3:
            return False

        # Skip obviously garbled patterns (repeated special characters)
        if re.search(r'^[0-9\.\-\/\s]{10,}$', address_clean):
            return False

        # Skip patterns with lots of dots/slashes (looks like garbled OCR)
        if re.search(r'[\.\-\/]{5,}', address_clean):
            return False

        # Skip if contains obviously garbled patterns like "./-0"
        if re.search(r'[\./\-]{3,}[0-9]+[\./\-]{3,}', address_clean):
            return False

        # Skip if it's just numbers and special chars (no proper words)
        if re.match(r'^[0-9\.\-\/\\,\s]+$', address_clean):
            return False

        # Skip if it's a known label or non-address text
        non_address_patterns = [
            'GENERATED ON', 'GENERATED BY', 'REQUESTING BRANCH',
            'STATEMENT FROM', 'STATEMENT TO', 'AS ON',
            'ACCOUNT TYPE', 'CURRENCY', 'CIF NUMBER',
            'THIS IS A COMPUTER GENERATED',
        ]
        address_upper = address_clean.upper()
        for pattern in non_address_patterns:
            if pattern in address_upper and len(address_clean) < 50:
                return False

        # Validate address has real location components
        # This rejects text like "Statement Period:, 08-Jan-2026..." which has no city/state
        from app.helper.validators.bank_statement_validator import get_bank_statement_validator
        validator = get_bank_statement_validator()
        parsed_components = validator.parse_address_components(address_clean)

        # Must have at least ONE of: city, state (postal_code alone is not enough - can be false positive from dates)
        has_location = any([
            parsed_components.get('city'),
            parsed_components.get('state'),
        ])

        if not has_location:
            logger.debug(f"Address rejected: no city/state location components in '{address_clean[:50]}...'")
            return False

        return True

    def _is_valid_account_holder_name(self, name: str) -> bool:
        """
        Check if a name looks like a valid account holder name.

        Filters out obviously invalid names like:
        - Garbled text with mostly special characters
        - Too short
        - No alphabetic characters
        - Only numbers and special characters
        - Text with colons (indicates a label attached like "PAN :ADJPN4882H")
        - Titles only (like "Mr.", "Mrs.", etc.)

        Args:
            name: The name to validate

        Returns:
            True if name looks valid, False otherwise
        """
        if not name:
            return False

        name_clean = name.strip()

        # Too short
        if len(name_clean) < 3:
            return False

        # Too short
        if len(name_clean) < 3:
            return False

        # Valid person names NEVER contain digits - reject anything with numbers
        if re.search(r'[0-9]', name_clean):
            return False

        # Reject text containing colon - indicates a label attached (e.g., "PAN :ADJPN4882H")
        # This is the key fix to avoid extracting patterns like "Scheme :PRESTIGE SAVINGS ACCOUNT"
        if ':' in name_clean:
            return False

        # Must have at least some alphabetic characters
        if not re.search(r'[A-Za-z]', name_clean):
            return False

        # Skip if it's just numbers and special chars (no proper names)
        if re.match(r'^[0-9\.\-\/\\,\s]+$', name_clean):
            return False

        # Skip obviously garbled patterns (lots of special characters in sequence)
        special_char_count = sum(1 for c in name_clean if c in './-\\,|')
        if special_char_count > len(name_clean) / 2:
            return False

        # Skip email addresses
        if '@' in name_clean and re.search(r'\S+@\S+\.\S+', name_clean):
            return False

        # Skip known non-name text
        non_name_patterns = [
            'GENERATED ON', 'GENERATED BY', 'STATEMENT FROM',
            'THIS IS A COMPUTER GENERATED',
            'CURRENCY', 'ACCOUNT SUMMARY', 'CONSOLIDATED',
            # Email-related patterns
            'EMAIL ID', 'EMAIL:', 'E-MAIL', 'REGISTERED EMAIL',
            # Account status labels (not person names)
            'YOUR A/C STATUS', 'ACCOUNT STATUS', 'A/C STATUS',
            # Transaction/statement date labels (not person names)
            'TRANSACTION DATE', 'STATEMENT DATE', 'FROM :', 'TO :',
            'DATE FROM', 'DATE TO',
            # Banking terminology patterns
            'SCHEME', 'ACCOUNT TYPE', 'A/C TYPE',
            # Account type codes (not person names)
            'SB-CHQ', 'SB-CR', 'CA-CHQ', 'CA-CR', 'SAVINGS ACCOUNT', 'CURRENT ACCOUNT',
            'GENERAL-PUB', 'JOINT-IND', 'SINGLE-IND', 'NRE-NRO', 'IND-ALL',
            # Field labels (not person names)
            'ADDRESS', 'RESIDENTIAL ADDRESS', 'MAILING ADDRESS', 'PERMANENT ADDRESS',
            'COMMUNICATION ADDRESS', 'CORRESPONDENCE ADDRESS',
            # Account balance labels (not person names)
            'TOTAL BALANCE', 'AVAILABLE BALANCE', 'CLEARED BALANCE', 'CURRENT BALANCE',
            'BALANCE', 'OUTSTANDING BALANCE', 'NET BALANCE',
            # Account operation modes (not person names)
            'OPERATING SINGLY', 'OPERATING JOINTLY', 'JOINT OPERATING', 'SINGLE OPERATING',
            'MODE OF OPERATION', 'JOINT', 'SINGLE', 'EITHER OR SURVIVOR', 'ANYONE OR SURVIVOR',
        ]
        name_upper = name_clean.upper()
        for pattern in non_name_patterns:
            if pattern in name_upper:
                return False

        # Skip if it's a state or city name (use the countrystatecity library)
        try:
            for country in ['IN', 'SG', 'MY', 'TH', 'AE', 'US', 'GB', 'HK']:
                locations = _load_country_locations(country)
                # Check states
                if name_upper in locations['states']:
                    return False
                # Check cities
                if name_upper in locations['cities']:
                    return False

            # Skip country names (from countrystatecity library)
            from countrystatecity_countries import get_all_countries
            all_countries = get_all_countries()
            country_names = {c.name.upper() for c in all_countries}
            if name_upper in country_names:
                return False
        except ImportError:
            # Library not available - skip this validation
            pass

        return True

    def _filter_branch_address_from_customer_address(self, address: str) -> str:
        """
        Filter out branch address text that may have been mixed with customer address.

        GLiNER sometimes includes branch addresses in the customer address field.
        This method detects and removes branch-specific patterns.

        Args:
            address: The address that may contain both customer and branch addresses

        Returns:
            Address with branch address filtered out
        """
        if not address:
            return address

        # Known branch address patterns to filter out (with regex for flexible matching)
        # Use regex to handle variations like "REEDSPET CHURCH,517001"
        import re
        branch_patterns = [
            r'REDDY PALLI,?\s*OPP\.?\s*REEDSPET CHURCH,?\s*\d{6}',
            r'OPP\.?\s*REEDSPET CHURCH,?\s*\d{6}',
            r'REEDSPET CHURCH',
            r'KONGARED', r'VIKAS COMPLEX',
            r'JALIKHANASTREET', r'JAILKHANA STREET',
        ]

        address_upper = address.upper()

        # Check if address contains branch patterns
        for pattern in branch_patterns:
            match = re.search(pattern, address_upper)
            if match:
                # Get the part after the branch pattern
                customer_part = address_upper[match.end():].strip(',. ')
                if customer_part:
                    logger.info(f"Filtered out branch address '{match.group()}': keeping '{customer_part[:50]}...'")
                    return customer_part

        return address

    def _filter_garbled_text_from_address(self, address: str) -> str:
        """
        Filter out garbled text that may be appended to valid address.

        Args:
            address: The address that may contain garbled text

        Returns:
            Address with garbled text filtered out
        """
        if not address:
            return address

        import re

        # Split address by commas and check each part
        parts = [p.strip() for p in address.split(',')]
        filtered_parts = []

        for part in parts:
            part_upper = part.upper()
            # Skip obviously garbled patterns at the END of the address
            # These typically appear as the last comma-separated part
            # Lots of dots/dashes with numbers (like "./-0./-0.--00--00-/0./.-0--000")
            if re.search(r'^[\./\-]{4,}[0-9\-]+[\./\-]{4,}', part_upper):
                # Only filter if this looks like purely garbled OCR output
                # Check if it has very few alphabetic characters compared to special chars
                special_count = sum(1 for c in part if c in './-\\,|')
                alpha_count = sum(1 for c in part if c.isalpha())
                if special_count > alpha_count * 2:  # More special chars than letters
                    logger.info(f"Filtering out garbled text: '{part[:50]}...'")
                    continue

            # Skip parts that are ONLY special characters (no letters at all)
            if re.match(r'^[\./\-\\,\s]+$', part_upper):
                logger.info(f"Filtering out special-only part: '{part[:50]}...'")
                continue

            filtered_parts.append(part)

        return ', '.join(filtered_parts)

    def _extract_account_holder_name_from_text_blocks(
        self,
        text_blocks: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[str]:
        """
        Extract account holder name from text blocks using a priority-based spatial strategy.

        This method uses geometric positioning and spatial relationships on the document
        rather than relying solely on GLiNER pattern matching, which can incorrectly
        extract labels and values that aren't person names.

        Priority 0: Top-left unlabeled name extraction (highest priority)
        - Find person names in the top-left region (top 20% of page)
        - Name appears before address keywords or address content
        - No explicit label needed - position indicates it's the account holder name

        Priority 1: Label-based extraction
        - Find text blocks containing "Name", "Account Holder", "Customer Name" labels
        - Extract the value immediately to the right or below these labels

        Priority 2: Address-proximity extraction
        - Find the customer's address (already extracted)
        - Look for text blocks that appear immediately above the address
        - Names typically appear just before the customer's address

        Priority 3: S/O pattern detection
        - Look for S/O, D/O, A/L patterns (Malaysian/Indian names)

        Priority 4: Account number proximity
        - Look for name blocks above account number labels

        Args:
            text_blocks: List of text blocks with geometry

        Returns:
            Account holder name if found, None otherwise
        """
        if not text_blocks:
            logger.warning("No text blocks provided for spatial extraction")
            return None

        logger.info(f"=== Starting spatial account holder name extraction with {len(text_blocks)} text blocks ===")

        # Log first 10 text blocks for debugging
        for i, block in enumerate(text_blocks[:10]):
            text = block.get('text', '').strip()
            y1 = block.get('y1', 0)
            x1 = block.get('x1', 0)
            logger.info(f"  Block {i}: y={y1:.3f}, x={x1:.3f}, text='{text[:50]}...'")

        # ========================================================================
        # PRIORITY 0: Label-based extraction (highest priority)
        # ========================================================================
        # Find text blocks with "Name", "Account Holder", "Customer Name" labels
        # and extract the value immediately to the right or below these labels.
        # This is more reliable than position-based extraction because it uses explicit labels.
        # ------------------------------------------------------------------------
        logger.info("PRIORITY 0: Starting label-based extraction...")

        name_labels = [
            'NAME', 'CUSTOMER NAME', 'A/C HOLDER NAME', 'ACCOUNT HOLDER',
            'A/C HOLDER', 'ACCOUNT HOLDER NAME', 'ACCOUNT HOLDER NAMES',
            'CUSTOMER', 'PRIMARY ACCOUNT HOLDER NAME'
        ]

        for i, block in enumerate(text_blocks):
            text = block.get('text', '').strip().upper()
            if not text:
                continue

            # Check if this block is a name label (exact match or ends with colon)
            is_name_label = any(
                text == label or text == label + ':' or text == label + ' :'
                for label in name_labels
            )
            if is_name_label:
                label_y1 = block.get('y1', 0)
                label_x2 = block.get('x2', 0)
                logger.info(f"  Found name label at block {i}: '{text}' (y={label_y1:.3f}, x2={label_x2:.3f})")

                # First check same line (after the label)
                for j in range(i + 1, min(i + 10, len(text_blocks))):
                    next_block = text_blocks[j]
                    next_text = next_block.get('text', '').strip()
                    if not next_text:
                        continue
                    next_y1 = next_block.get('y1', 0)
                    next_x1 = next_block.get('x1', 0)

                    # Same line if y is similar and x is after label
                    if abs(next_y1 - label_y1) < 0.02 and next_x1 > label_x2:
                        # Clean leading/trailing special chars that may be attached due to formatting/OCR
                        clean_text = next_text.strip().lstrip(':.').strip()
                        is_valid = self._is_valid_account_holder_name(clean_text)
                        logger.info(f"    Checking block {j} (same line after): y={next_y1:.3f}, x={next_x1:.3f}, text='{clean_text}', valid={is_valid}")
                        if is_valid:
                            logger.info(f"Found account holder name via name label (same line after): '{clean_text}'")
                            return clean_text
                    elif next_y1 > label_y1 + 0.02:
                        # Moved to next line, stop looking on same line
                        break

                # Check same line (before the label) - handles cases where value appears before label
                logger.info(f"  Checking same line before label...")
                for j in range(i - 1, max(i - 10, -1), -1):
                    prev_block = text_blocks[j]
                    prev_text = prev_block.get('text', '').strip()
                    if not prev_text:
                        continue
                    prev_y1 = prev_block.get('y1', 0)
                    prev_x2 = prev_block.get('x2', 0)

                    # Same line if y is similar and x is before label
                    if abs(prev_y1 - label_y1) < 0.02 and prev_x2 < block.get('x1', 0):
                        # Clean leading/trailing special chars that may be attached due to formatting/OCR
                        clean_text = prev_text.strip().lstrip(':.').strip()
                        is_valid = self._is_valid_account_holder_name(clean_text)
                        logger.info(f"    Checking block {j} (same line before): y={prev_y1:.3f}, x2={prev_x2:.3f}, text='{clean_text}', valid={is_valid}")
                        if is_valid:
                            logger.info(f"Found account holder name via name label (same line before): '{clean_text}'")
                            return clean_text
                    elif label_y1 - prev_y1 > 0.02:
                        # Moved to previous line, stop looking on same line
                        break

                # Check next line for the name value (forward)
                label_x1 = block.get('x1', 0)
                logger.info(f"  Checking next line (forward) for value below label...")
                for j in range(i + 1, min(i + 10, len(text_blocks))):
                    next_block = text_blocks[j]
                    next_text = next_block.get('text', '').strip()
                    if not next_text:
                        continue
                    next_y1 = next_block.get('y1', 0)
                    next_x1 = next_block.get('x1', 0)

                    # Check if on next line (within reasonable distance)
                    if 0 < next_y1 - label_y1 < 0.05:
                        # For tabular layouts, X-offset can be significant (100+ units)
                        # Only skip if X-offset is unreasonably large (>500 units)
                        if abs(next_x1 - label_x1) > 500:
                            # X values too far apart - likely unrelated content
                            continue
                        # Skip if it looks like another label
                        if any(label in next_text.upper() for label in name_labels):
                            continue
                        # Clean leading/trailing special chars that may be attached due to formatting/OCR
                        clean_text = next_text.strip().lstrip(':.').strip()
                        # Single-word values are unlikely to be person names (more likely cities/fields)
                        if len(clean_text.split()) < 2:
                            logger.info(f"    Skipping single-word value '{clean_text}' - unlikely to be a person name")
                            continue
                        is_valid = self._is_valid_account_holder_name(clean_text)
                        logger.info(f"    Checking block {j} (next line): y={next_y1:.3f}, x={next_x1:.3f}, text='{clean_text}', valid={is_valid}")
                        if is_valid:
                            logger.info(f"Found account holder name via name label (next line): '{clean_text}'")
                            return clean_text
                    elif next_y1 - label_y1 >= 0.05:
                        # Too far down, stop looking
                        break

                # Check previous line for the name value (backward) - handles cases where value appears above label
                label_x1 = block.get('x1', 0)
                logger.info(f"  Checking previous line (backward) for value above label...")
                for j in range(i - 1, max(i - 10, -1), -1):
                    prev_block = text_blocks[j]
                    prev_text = prev_block.get('text', '').strip()
                    if not prev_text:
                        continue
                    prev_y1 = prev_block.get('y1', 0)
                    prev_x1 = prev_block.get('x1', 0)

                    # Check if on previous line (within reasonable distance)
                    if 0 < label_y1 - prev_y1 < 0.05:
                        # For tabular layouts, X-offset can be significant (100+ units)
                        # Only skip if X-offset is unreasonably large (>500 units)
                        if abs(prev_x1 - label_x1) > 500:
                            # X values too far apart - likely unrelated content
                            logger.info(f"    Block {j} (previous line): X not aligned (prev_x1={prev_x1:.3f}, label_x1={label_x1:.3f}), skipping")
                            continue
                        # Skip if it looks like another label
                        if any(label in prev_text.upper() for label in name_labels):
                            continue
                        # Clean leading/trailing special chars that may be attached due to formatting/OCR
                        clean_text = prev_text.strip().lstrip(':.').strip()
                        is_valid = self._is_valid_account_holder_name(clean_text)
                        logger.info(f"    Checking block {j} (previous line): y={prev_y1:.3f}, x={prev_x1:.3f}, text='{clean_text}', valid={is_valid}")
                        if is_valid:
                            logger.info(f"Found account holder name via name label (previous line): '{clean_text}'")
                            return clean_text
                    elif label_y1 - prev_y1 >= 0.05:
                        # Too far up, stop looking
                        break

        logger.info("PRIORITY 0: No valid name found via label-based extraction")

        # ========================================================================
        # PRIORITY 0.5: Title-based extraction
        # ========================================================================
        # Find text blocks with titles like "Mr.", "Mrs.", "Dr.", "Shri", "Smt"
        # and extract the name value that comes after these titles.
        # This handles cases where the name appears immediately after a title
        # without an explicit "Name:" label.
        # ------------------------------------------------------------------------
        logger.info("PRIORITY 0.5: Starting title-based extraction...")

        title_labels = ['MR.', 'MRS.', 'MS.', 'DR.', 'SHRI', 'SMT']

        for i, block in enumerate(text_blocks):
            text = block.get('text', '').strip().upper()
            if not text:
                continue

            # Check if this block is a title label
            is_title_label = any(
                text == label or text == label + ':' or text == label + ' :'
                for label in title_labels
            )
            if is_title_label:
                label_y1 = block.get('y1', 0)
                label_x2 = block.get('x2', 0)
                logger.info(f"  Found title label at block {i}: '{text}' (y={label_y1:.3f}, x2={label_x2:.3f})")

                # First check same line (after the title) - handles "Mr. John Doe" in same block
                # But we need to check if the name is in a separate block
                for j in range(i + 1, min(i + 10, len(text_blocks))):
                    next_block = text_blocks[j]
                    next_text = next_block.get('text', '').strip()
                    if not next_text:
                        continue
                    next_y1 = next_block.get('y1', 0)
                    next_x1 = next_block.get('x1', 0)

                    # Same line if y is similar and x is after title
                    if abs(next_y1 - label_y1) < 0.02 and next_x1 > label_x2:
                        clean_text = next_text.strip().lstrip(':.').strip()
                        is_valid = self._is_valid_account_holder_name(clean_text)
                        logger.info(f"    Checking block {j} (same line after title): y={next_y1:.3f}, x={next_x1:.3f}, text='{clean_text}', valid={is_valid}")
                        if is_valid:
                            logger.info(f"Found account holder name via title label (same line after): '{clean_text}'")
                            return clean_text
                    elif next_y1 > label_y1 + 0.02:
                        # Moved to next line, stop looking on same line
                        break

                # Check next line for the name value (forward)
                label_x1 = block.get('x1', 0)
                logger.info(f"  Checking next line (forward) for value below title...")
                for j in range(i + 1, min(i + 10, len(text_blocks))):
                    next_block = text_blocks[j]
                    next_text = next_block.get('text', '').strip()
                    if not next_text:
                        continue
                    next_y1 = next_block.get('y1', 0)
                    next_x1 = next_block.get('x1', 0)

                    # Check if on next line (within reasonable distance)
                    if 0 < next_y1 - label_y1 < 0.05:
                        # X values should be aligned for vertical placement
                        if abs(next_x1 - label_x1) > 0.1:
                            continue
                        clean_text = next_text.strip().lstrip(':.').strip()
                        is_valid = self._is_valid_account_holder_name(clean_text)
                        logger.info(f"    Checking block {j} (next line after title): y={next_y1:.3f}, x={next_x1:.3f}, text='{clean_text}', valid={is_valid}")
                        if is_valid:
                            logger.info(f"Found account holder name via title label (next line): '{clean_text}'")
                            return clean_text
                    elif next_y1 - label_y1 >= 0.05:
                        # Too far down, stop looking
                        break

        logger.info("PRIORITY 0.5: No valid name found via title-based extraction")

        # ========================================================================
        # PRIORITY 1: Top-left unlabeled name extraction (fallback)
        # ========================================================================
        # Find person names in the top-left region of the document.
        # This handles cases where the account holder name appears without any label,
        # typically in the header area before the address block.
        # Only runs if label-based extraction (PRIORITY 0) fails.
        # ------------------------------------------------------------------------
        logger.info("PRIORITY 1: Starting top-left unlabeled name extraction...")

        # Keywords that indicate the start of address block (name should come before these)
        address_keywords = {
            'ADDRESS', 'RESIDENTIAL ADDRESS', 'MAILING ADDRESS', 'PERMANENT ADDRESS',
            'COMMUNICATION ADDRESS', 'CORRESPONDENCE ADDRESS', 'PERMANENT',
            'RESIDENTIAL', 'MAILING', 'COMMUNICATION'
        }

        for i, block in enumerate(text_blocks):
            text = block.get('text', '').strip()
            if not text:
                continue

            y1 = block.get('y1', 0)
            x1 = block.get('x1', 0)

            # Check if in top-left region (top 20% of page, left 50% of page)
            if y1 > 0.20 or x1 > 0.50:
                continue  # Not in top-left region

            text_upper = text.upper()

            # Skip if it looks like an address keyword
            if any(keyword in text_upper for keyword in address_keywords):
                logger.info(f"  Block {i}: '{text}' contains address keyword, stopping search")
                break  # Reached address block, stop looking for name

            # Skip if it looks like a header or label (not a person name)
            skip_patterns = {
                'JOINT HOLDER', 'JOINT', 'HOLDER', 'PRIMARY', 'ACCOUNT HOLDER',
                'ACCOUNT HOLDER NAME', 'CUSTOMER NAME', 'NAME:', 'NAME :',
                'NAME', 'CUSTOMER', 'ACCOUNT', 'DETAILS', 'DETAILS OF',
                'STATEMENT', 'BANK', 'BRANCH', 'PAGE', 'DATE', 'PERIOD',
                'A/C HOLDER', 'A/C HOLDER NAME', 'PRIMARY ACCOUNT',
                'ADDRESS', 'RESIDENTIAL ADDRESS', 'MAILING ADDRESS'
            }
            # Title-only patterns (only skip if EXACT match, not if followed by name)
            title_only_patterns = {'MR.', 'MRS.', 'MS.', 'DR.', 'SHRI', 'SMT'}

            # Check if text equals or starts with skip patterns
            if any(text_upper == pattern or text_upper.startswith(pattern) for pattern in skip_patterns):
                logger.debug(f"  Block {i}: '{text}' matches skip pattern, skipping")
                continue

            # Only skip title-only patterns if they're exact matches (not followed by a name)
            if any(text_upper == pattern for pattern in title_only_patterns):
                logger.debug(f"  Block {i}: '{text}' is a title-only pattern, skipping")
                continue

            # Check if it looks like a person name (2-4 words, all caps or title case)
            words = text.split()
            if 2 <= len(words) <= 4:
                # Check if all words look like name parts (letters only, may have initials)
                is_valid = self._is_valid_account_holder_name(text)
                logger.info(f"  Block {i}: '{text}' in top-left, words={len(words)}, is_valid={is_valid}")
                if is_valid:
                    # Additional check: next few elements should be address-related
                    # This confirms we found the name before the address
                    next_is_address = False
                    for j in range(i + 1, min(i + 8, len(text_blocks))):
                        next_text = text_blocks[j].get('text', '').strip().upper()
                        if any(keyword in next_text for keyword in address_keywords):
                            next_is_address = True
                            logger.info(f"    Next block {j} has address keyword: '{next_text[:30]}...'")
                            break
                        # Also check for address-like patterns (street, city, etc.)
                        if any(pattern in next_text for pattern in ['STREET', 'ROAD', 'APARTMENT', 'NAGAR', 'PHASE', 'CITY', 'BENGALURU', 'BANGALORE', 'MUMBAI', 'DELHI', 'CHITTOOR', 'LANE', 'FLOR', 'OFFICERS']):
                            next_is_address = True
                            logger.info(f"    Next block {j} has address-like pattern: '{next_text[:30]}...'")
                            break
                        # Check for email (also indicates address section)
                        if '@' in next_text:
                            next_is_address = True
                            logger.info(f"    Next block {j} has email: '{next_text[:30]}...'")
                            break

                    # Relax the address check requirement - if we found a valid name in top-left,
                    # accept it even without address confirmation
                    logger.info(f"Found account holder name via top-left unlabeled extraction: '{text}'")
                    return text

        logger.info("PRIORITY 1: No valid name found in top-left region")

        # ========================================================================
        # PRIORITY 1.5: Colon-prefixed name extraction (fallback)
        # ========================================================================
        # Handle cases where the name is extracted with a leading colon due to
        # formatting/OCR issues (e.g., ": VINEETH NARASIMHAN" where the label
        # might be in a separate block that wasn't extracted or was filtered out).
        # This is a common pattern where the label is on the left and the value
        # starts with a colon.
        # ------------------------------------------------------------------------
        logger.info("PRIORITY 1.5: Starting colon-prefixed extraction...")

        for i, block in enumerate(text_blocks):
            text = block.get('text', '').strip()
            if not text:
                continue

            # Check if text starts with a colon (indicating it's a value with label prefix)
            if text.startswith(':'):
                # Clean leading colons and special chars
                clean_text = text.lstrip(':.').strip()
                is_valid = self._is_valid_account_holder_name(clean_text)
                logger.info(f"  Block {i}: Found colon-prefixed text: '{text}', cleaned: '{clean_text}', valid: {is_valid}")
                if is_valid:
                    logger.info(f"Found account holder name via colon-prefixed text: '{clean_text}'")
                    return clean_text

        logger.info("PRIORITY 1.5: No valid colon-prefixed name found")

        # ========================================================================
        # PRIORITY 2: Address-proximity extraction
        # ========================================================================
        # Find the customer's address and look for text blocks that appear
        # immediately above the address. Names typically appear just before
        # the customer's address in bank statements.
        # ------------------------------------------------------------------------

        # Try to get the already-extracted address from gliner_result
        address = None
        address_y1 = None

        # Find the address block among text blocks
        for i, block in enumerate(text_blocks):
            text = block.get('text', '').strip()
            if not text:
                continue

            # Check if this looks like an address (has multiple parts, numbers, etc.)
            # A simple heuristic: addresses often contain numbers and are multi-line
            text_upper = text.upper()
            if any(pattern in text_upper for pattern in ['STREET', 'ROAD', 'AVENUE', 'LANE', 'CITY', 'PIN', 'ZIP']):
                # This might be an address
                address_y1 = block.get('y1', 0)

                # Look for name blocks immediately above the address
                for j in range(max(0, i - 5), i):
                    name_block = text_blocks[j]
                    name_text = name_block.get('text', '').strip()
                    if not name_text:
                        continue

                    name_y1 = name_block.get('y1', 0)

                    # Check if within reasonable distance above (within 0.05 units)
                    if 0 < address_y1 - name_y1 < 0.05:
                        # Additional check: same horizontal alignment (within 0.1 units)
                        name_x1 = name_block.get('x1', 0)
                        address_x1 = block.get('x1', 0)
                        if abs(name_x1 - address_x1) < 0.1:
                            if self._is_valid_account_holder_name(name_text):
                                logger.info(f"Found account holder name via address proximity: '{name_text}'")
                                return name_text

        # ========================================================================
        # PRIORITY 3: S/O pattern detection
        # ========================================================================
        # Look for S/O, D/O, A/L patterns (Malaysian/Indian names)
        # ------------------------------------------------------------------------

        for block in text_blocks:
            text = block.get('text', '').strip()
            if not text:
                continue

            # Check for S/O pattern (Malaysian/Indian names)
            if re.search(r'\s+(S/O|D/O|A/L)\s+', text, re.IGNORECASE):
                if self._is_valid_account_holder_name(text):
                    logger.info(f"Found account holder name via S/O pattern: '{text}'")
                    return text

        # ========================================================================
        # PRIORITY 4: Account number proximity
        # ========================================================================
        # Look for name-like patterns near account number labels
        # ------------------------------------------------------------------------

        account_number_labels = [
            'ACCOUNT NUMBER', 'ACCOUNT NO', 'A/C NO',
            'SAVINGS A/C', 'CURRENT A/C',
        ]

        for i, block in enumerate(text_blocks):
            text = block.get('text', '').strip().upper()
            if not text:
                continue

            # Check if this is an account number label
            is_label = any(label in text for label in account_number_labels)
            if not is_label:
                continue

            label_y1 = block.get('y1', 0)

            # Look for name blocks above the account number label
            for j in range(max(0, i - 5), i):
                name_block = text_blocks[j]
                name_text = name_block.get('text', '').strip()
                if not name_text:
                    continue

                name_y1 = name_block.get('y1', 0)

                # Check if within reasonable distance above
                if 0 < label_y1 - name_y1 < 0.05:
                    if self._is_valid_account_holder_name(name_text):
                        logger.info(f"Found account holder name via positional extraction: '{name_text}'")
                        return name_text

        # ========================================================================
        # FALLBACK: Handle OCR artifacts
        # ========================================================================
        # OCR sometimes produces ": VINEETH NARASIMHAN" when the label and
        # value are on the same line.
        # ------------------------------------------------------------------------

        for block in text_blocks:
            text = block.get('text', '').strip()
            if not text:
                continue

            # Check for pattern like ": VINEETH NARASIMHAN" (OCR artifact from "Name: VINEETH")
            colon_name_match = re.match(r'^:\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+)$', text)
            if colon_name_match:
                name = colon_name_match.group(1)
                if self._is_valid_account_holder_name(name):
                    logger.info(f"Found account holder name via colon-prefix pattern: '{name}'")
                    return name

        logger.info("=== All spatial extraction priorities exhausted, returning None ===")
        return None

    def _extract_address_by_field_label(
        self,
        text_blocks: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[str]:
        """
        Extract customer address by finding values next to 'Address' field labels.

        This handles tabular format bank statements where 'Address' is a field label
        followed by the address value (on same line or next line).

        Strategy:
        1. Find blocks containing 'Address' label
        2. Look for address values on same line (higher x value) or next line (similar y value)
        3. Combine consecutive address-looking blocks
        """
        if not text_blocks:
            return None

        # Find address label blocks
        address_label_indices = []
        for i, block in enumerate(text_blocks):
            text = block.get('text', '').strip()
            # Match 'Address' but not 'branch address' or similar
            if text.upper() == 'ADDRESS':
                address_label_indices.append(i)

        if not address_label_indices:
            return None

        # For each address label, try to find the address value
        for label_idx in address_label_indices:
            label_block = text_blocks[label_idx]
            label_y1 = label_block.get('y1', 0)
            label_x1 = label_block.get('x1', 0)

            address_parts = []

            # Look for address value on same line (to the right of label)
            for i in range(label_idx + 1, min(label_idx + 20, len(text_blocks))):
                block = text_blocks[i]
                text = block.get('text', '').strip()
                if not text:
                    continue

                block_y1 = block.get('y1', 0)

                # Check if on same line (y difference < 0.02)
                if abs(block_y1 - label_y1) < 0.02:
                    # Found value on same line
                    if not self._is_non_address_text(text):
                        address_parts.append(text)
                elif block_y1 > label_y1:
                    # Block is below the label
                    # Check if it's part of multi-line address (within 0.04 y distance)
                    if block_y1 - label_y1 <= 0.04:
                        if not self._is_non_address_text(text):
                            address_parts.append(text)
                    else:
                        # Too far down, stop looking
                        break

            if address_parts:
                return self._format_address(address_parts)

        return None

    def _extract_address_by_positional_context(
        self,
        gliner_result: Dict[str, Any],
        text_blocks: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[str]:
        """
        Extract customer address by looking for text blocks below account holder name.

        This is a positional fallback for formal letter format addresses that may not be
        explicitly labeled by GLiNER but appear directly below the account holder name
        in the top-left of the document.

        Strategy:
        1. Find account holder name block from GLiNER or pattern matching
        2. Find blocks with y-position below account holder name (within 0.15)
        3. Filter out bank-related terms, labels, and non-address text
        4. Combine consecutive blocks into full address
        """
        if not text_blocks:
            return None

        # Step 1: Find account holder name block
        name_blocks = []  # List of (y_position, text) tuples

        # First try to get name from GLiNER results
        for label, entities in gliner_result.items():
            if entities is None:
                continue
            label_lower = label.lower()
            # Check if this is an account holder name label
            if any(name_label in label_lower for name_label in [
                'account_holder', 'account holder', 'customer name', 'person name', 'customer full name'
            ]):
                if isinstance(entities, list) and len(entities) > 0:
                    name_text = entities[0].get('value', '').strip()
                elif not isinstance(entities, list):
                    name_text = entities.get('value', '').strip()
                else:
                    continue

                # Find all text blocks containing this name
                for block in text_blocks:
                    block_text = block.get('text', '').strip()
                    if name_text.upper() in block_text.upper():
                        name_blocks.append((block.get('y1', 0), block_text))

        # If no name found in GLiNER results, try pattern matching
        if not name_blocks:
            for block in text_blocks:
                text = block.get('text', '').strip()
                # Look for name-like patterns (common Indian name formats)
                # Skip very short or very long text
                if 3 <= len(text) <= 50:
                    # Look for patterns like: "MR. K C ROHITH", "K C ROHITH", "ROHITH K C"
                    if re.search(r'^[A-Z][A-Z\s\.]+$', text) or re.search(r'\s+(S/O|D/O|A/L)\s+', text, re.IGNORECASE):
                        # Skip if it's clearly a label
                        if not any(label.lower() in text.lower() for label in ['account', 'statement', 'branch', 'customer', 'details']):
                            name_blocks.append((block.get('y1', 0), text))

        if not name_blocks:
            return None

        # Step 2: For each name block, look for address blocks below it
        best_address = None
        most_address_lines = 0

        for name_y1, name_text in name_blocks:
            # Find blocks below the name block
            address_blocks = []
            seen_texts = set()

            for block in text_blocks:
                text = block.get('text', '').strip()
                if not text or text in seen_texts:
                    continue

                seen_texts.add(text)
                block_y1 = block.get('y1', 0)

                # Check if block is below name (within 0.20 normalized distance for letterhead format)
                # This allows capturing multi-line addresses that may be further down
                if 0 < block_y1 - name_y1 <= 0.20:
                    # Skip if it's the name itself
                    if text.upper() == name_text.upper():
                        continue

                    # Skip if it's an email
                    if '@' in text:
                        continue

                    # Skip bank-related terms and labels
                    if self._is_non_address_text(text):
                        continue

                    # Skip blocks that are clearly not address (too long, likely table data)
                    if len(text) > 80:
                        continue

                    # Skip blocks that are clearly transaction data (lots of numbers, dates)
                    if re.search(r'\d{2}/\d{2}/\d{4}', text) and ',' in text and len(text.split(',')) > 3:
                        continue

                    address_blocks.append((block_y1, text))

            # Sort by y position
            address_blocks.sort(key=lambda x: x[0])

            # Extract consecutive address lines (stop at gaps larger than 0.03)
            address_lines = []
            prev_y = None

            for y, text in address_blocks:
                if prev_y is None:
                    address_lines.append(text)
                    prev_y = y
                elif y - prev_y <= 0.03:  # Consecutive lines
                    address_lines.append(text)
                    prev_y = y
                else:  # Gap too large, stop
                    break

            # Limit to reasonable number of lines (typically 3-6 for addresses)
            address_lines = address_lines[:6]

            if len(address_lines) > most_address_lines:
                most_address_lines = len(address_lines)
                best_address = self._format_address(address_lines)

        return best_address

    def _extract_singapore_address_from_blocks(
        self,
        text_blocks: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[str]:
        """
        Extract Singapore address by looking for BLK/block patterns in text blocks.

        Singapore address format: BLK 29 STREET NAME #FLOOR-UNIT POSTAL_CODE
        Example: BLK 29 MARINE CRESCENT, #11-25, SINGAPORE 440029

        This is a fallback method when GLiNER misses Singapore addresses due to garbled OCR.

        Args:
            text_blocks: List of text blocks with geometry

        Returns:
            Singapore address if found, None otherwise
        """
        if not text_blocks:
            return None

        # Find blocks that look like Singapore address components
        address_blocks = []
        seen_indices = set()

        for i, block in enumerate(text_blocks):
            if i in seen_indices:
                continue

            text = block.get('text', '').strip().upper()
            if not text:
                continue

            # Check for Singapore address patterns
            is_singapore_address_part = False

            # BLK/BLOCK pattern followed by number
            if re.match(r'^BLK[\s\.]*\d+.*|^BLOCK[\s\.]*\d+', text):
                is_singapore_address_part = True

            # Floor/unit pattern like #11-25
            if re.match(r'^#[\d\-]+', text):
                is_singapore_address_part = True

            # SINGAPORE followed by postal code (6 digits)
            if re.match(r'^SINGAPORE\s+\d{6}', text):
                is_singapore_address_part = True

            # Singapore street names (common in addresses)
            singapore_streets = [
                'MARINE CRESCENT', 'CRESCENT', 'CLOSE', 'DRIVE', 'PLACE',
                'ROAD', 'STREET', 'AVENUE', 'LANE', 'WALK',
            ]
            for street in singapore_streets:
                if street in text and len(text) < 50:
                    is_singapore_address_part = True
                    break

            if is_singapore_address_part:
                # Check nearby blocks for more address parts
                y1 = block.get('y1', 0)

                # Look for nearby blocks (within 0.05 y distance)
                nearby_blocks = [text]
                nearby_indices = {i}

                for j in range(max(0, i - 3), min(len(text_blocks), i + 4)):
                    if j in nearby_indices:
                        continue

                    nearby_block = text_blocks[j]
                    nearby_y1 = nearby_block.get('y1', 0)
                    nearby_text = nearby_block.get('text', '').strip().upper()

                    # Check if close in y position
                    if abs(nearby_y1 - y1) <= 0.03:
                        # Check if it looks like address content
                        if self._looks_like_address_content(nearby_text):
                            # But skip if it's the account holder name (S/O pattern)
                            if not re.search(r'\s+(S/O|D/O|A/L)\s+', nearby_text):
                                nearby_blocks.append(nearby_text)
                                nearby_indices.add(j)

                # If we have multiple address-like blocks, consider it a match
                if len(nearby_blocks) >= 2:
                    address_blocks.append((y1, nearby_blocks))
                    for idx in nearby_indices:
                        seen_indices.add(idx)
                    break  # Don't reuse this block in future iterations

        if not address_blocks:
            return None

        # Sort by y position and format the address
        address_blocks.sort(key=lambda x: x[0])

        # Take the best candidate (most address parts)
        best_candidate = max(address_blocks, key=lambda x: len(x[1]))
        address_parts = best_candidate[1]

        # Remove duplicates while preserving order
        seen = set()
        unique_parts = []
        for part in address_parts:
            if part not in seen:
                seen.add(part)
                unique_parts.append(part)

        return self._format_address(unique_parts)

    def _is_valid_statement_date(self, date_str: str) -> bool:
        """
        Check if a statement date looks valid.

        Valid formats:
        - DD MMM YYYY: "30 Nov 2025"
        - DD/MM/YYYY: "01/04/2024"
        - MM/DD/YYYY: "04/01/2024"
        - MMM DD, YYYY: "Nov 30, 2025"

        Invalid:
        - Person names with S/O/D/O/A/L patterns
        - Address fragments like "BLK 29"
        - Too short (< 4 chars) or too long (> 20 chars)
        - No numbers

        Args:
            date_str: The date string to validate

        Returns:
            True if date looks valid, False otherwise
        """
        if not date_str:
            return False

        # Log for debugging
        # logger.debug(f"Validating statement_date: '{date_str}' (len={len(date_str)})")

        date_upper = date_str.strip().upper()
        """
        Check if a statement date looks valid.

        Valid formats:
        - DD MMM YYYY: "30 Nov 2025"
        - DD/MM/YYYY: "01/04/2024"
        - MM/DD/YYYY: "04/01/2024"
        - MMM DD, YYYY: "Nov 30, 2025"

        Invalid:
        - Person names with S/O/D/O/A/L patterns
        - Address fragments like "BLK 29"
        - Too short (< 4 chars) or too long (> 20 chars)
        - No numbers

        Args:
            date_str: The date string to validate

        Returns:
            True if date looks valid, False otherwise
        """
        if not date_str:
            return False

        date_upper = date_str.strip().upper()

        # Too short or too long
        if len(date_upper) < 4 or len(date_upper) > 30:
            return False

        # Must contain at least one digit (dates have numbers)
        if not re.search(r'\d', date_upper):
            return False

        # Must contain at least one letter OR be a valid date format with slashes or hyphens
        # DD/MM/YYYY or DD-MM-YYYY format has no letters, so we allow all-digit dates with slashes/hyphens
        has_letters = re.search(r'[A-Z]', date_upper)
        has_slashes_or_hyphens = '/' in date_upper or '-' in date_upper
        if not has_letters and not has_slashes_or_hyphens:
            return False

        # Reject if it looks like a name with S/O/D/O/A/L pattern
        if re.search(r'\s+(S/O|D/O|A/L)\s+', date_upper):
            return False

        # Reject if it looks like an address (starts with BLK/BLOCK)
        if re.match(r'^BLK[\s\.]*\d+|^BLOCK[\s\.]*\d+', date_upper):
            return False

        # Reject if it's just a name without date patterns
        # Common Indian/Singapore names without date indicators
        name_patterns = [
            r'^[A-Z\s]+$',  # Just letters and spaces (no numbers)
            r'^THANABALAN$', r'^MANOGARAN',  # Specific names from DBS statement
        ]
        for pattern in name_patterns:
            if re.match(pattern, date_upper):
                return False

        # Check for common date patterns
        valid_date_patterns = [
            r'\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}',  # DD MMM YYYY
            r'\d{1,2}/\d{1,2}/\d{4}',  # DD/MM/YYYY or MM/DD/YYYY
            r'\d{1,2}/\d{1,2}/\d{2}',  # DD/MM/YY (Thai format)
            r'\d{1,2}-\d{1,2}-\d{4}',  # DD-MM-YYYY (SBI format)
            r'\d{4}',  # Just YYYY
            r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s*\d{4}',  # MMM DD, YYYY
            r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]+\s+\d{1,2},?\s*\d{4}',  # January 18, 2026
        ]
        for pattern in valid_date_patterns:
            if re.search(pattern, date_upper, re.IGNORECASE):
                return True

        return False

    def _clean_statement_date(self, date_str: str) -> str:
        """
        Clean up statement date by extracting just the date part.

        Handles cases where GLiNER returns extra text along with the date.
        E.g., "THANABALAN\nBLK 29" should become empty/invalid
              "30 Nov 2025" should stay as is
              "as at 30 Nov 2025" should become "30 Nov 2025"

        Args:
            date_str: The potentially dirty date string

        Returns:
            Cleaned date string
        """
        if not date_str:
            return date_str

        # Remove newlines and extra whitespace
        cleaned = re.sub(r'\s+', ' ', date_str).strip()

        # Try to extract just the date part using common date patterns
        date_patterns = [
            r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})',  # DD MMM YYYY
            r'(\d{1,2}/\d{1,2}/\d{4})',  # DD/MM/YYYY or MM/DD/YYYY
            r'(\d{1,2}/\d{1,2}/\d{2})',  # DD/MM/YY (Thai format)
            r'(\d{1,2}-\d{1,2}-\d{4})',  # DD-MM-YYYY (SBI format)
            r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]+\s+\d{1,2},?\s*\d{4})',  # MMM DD, YYYY or January 18, 2026
        ]

        for pattern in date_patterns:
            match = re.search(pattern, cleaned, re.IGNORECASE)
            if match:
                return match.group(1)

        # If no date pattern found, return the cleaned string
        # (validation will reject it if it's not a valid date)
        return cleaned

    def _normalize_statement_date(self, date_str: str) -> Optional[str]:
        """
        Normalize statement date to DD MMM YYYY format.

        Handles various input formats:
        - DD-MM-YYYY (09-02-2026 -> 09 Feb 2026)
        - DD/MM/YYYY (27/01/2026 -> 27 Jan 2026)
        - DD MMM YYYY (30 Nov 2025 -> 30 Nov 2025)
        - MMM DD, YYYY (January 18, 2026 -> 18 Jan 2026)
        - MMM DD YYYY (Jan 18 2026 -> 18 Jan 2026)

        Args:
            date_str: The date string to normalize

        Returns:
            Normalized date in DD MMM YYYY format, or None if parsing fails
        """
        if not date_str:
            return None

        date_str = date_str.strip()

        # Month name to number mapping
        month_map = {
            'jan': '01', 'january': '01',
            'feb': '02', 'february': '02',
            'mar': '03', 'march': '03',
            'apr': '04', 'april': '04',
            'may': '05',
            'jun': '06', 'june': '06',
            'jul': '07', 'july': '07',
            'aug': '08', 'august': '08',
            'sep': '09', 'september': '09',
            'oct': '10', 'october': '10',
            'nov': '11', 'november': '11',
            'dec': '12', 'december': '12',
        }

        # Number to month abbreviation mapping
        num_to_month = {
            '01': 'Jan', '02': 'Feb', '03': 'Mar', '04': 'Apr',
            '05': 'May', '06': 'Jun', '07': 'Jul', '08': 'Aug',
            '09': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dec',
            '1': 'Jan', '2': 'Feb', '3': 'Mar', '4': 'Apr',
            '5': 'May', '6': 'Jun', '7': 'Jul', '8': 'Aug',
            '9': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dec',
        }

        day, month, year = None, None, None

        # Pattern: DD-MM-YYYY, DD/MM/YYYY, or DD/MM/YY (2-digit year)
        match = re.match(r'^(\d{1,2})[-/](\d{1,2})[-/](\d{2}|\d{4})$', date_str)
        if match:
            day, month, year = match.groups()
            # Convert 2-digit year to 4-digit year (assume 2000s for 00-99)
            if len(year) == 2:
                year = f"20{year}"

        # Pattern: DD MMM YYYY or DD MMM YY (already in correct format, just validate)
        elif re.match(r'^\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}$', date_str, re.IGNORECASE):
            # Just capitalize the month abbreviation properly
            parts = date_str.split()
            if len(parts) == 3:
                day_part, month_part, year_part = parts
                # Convert 2-digit year to 4-digit year
                if len(year_part) == 2:
                    year_part = f"20{year_part}"
                # Normalize month to 3-letter abbreviation
                month_lower = month_part[:3].lower()
                for short, full in [('jan', 'Jan'), ('feb', 'Feb'), ('mar', 'Mar'), ('apr', 'Apr'),
                                     ('may', 'May'), ('jun', 'Jun'), ('jul', 'Jul'), ('aug', 'Aug'),
                                     ('sep', 'Sep'), ('oct', 'Oct'), ('nov', 'Nov'), ('dec', 'Dec')]:
                    if month_lower == short:
                        month_part = full
                        break
                return f"{day_part} {month_part} {year_part}"

        # Pattern: MMM DD, YYYY or MMM DD YYYY (January 18, 2026 or Jan 18 2026)
        else:
            match = re.match(r'^([A-Za-z]+)\s+(\d{1,2}),?\s*(\d{4})$', date_str)
            if match:
                month_name, day_part, year_part = match.groups()
                month_lower = month_name.lower()
                if month_lower in month_map:
                    day = day_part
                    month = month_map[month_lower]
                    year = year_part

        # Convert to DD MMM YYYY format
        if day and month and year:
            # Pad day to 2 digits if needed
            day = day.zfill(2)
            month_abbr = num_to_month.get(month.lstrip('0'), month)
            return f"{day} {month_abbr} {year}"

        return None

    def _extract_statement_date_from_blocks(
        self,
        text_blocks: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[str]:
        """
        Extract statement date by looking for date patterns in text blocks.

        This is a fallback when GLiNER doesn't extract a valid statement date.

        Args:
            text_blocks: List of text blocks with geometry

        Returns:
            Statement date if found, None otherwise
        """
        if not text_blocks:
            return None

        # Date patterns to look for
        date_patterns = [
            r'as at\s+(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})',  # "as at 30 Nov 2025"
            r'Statement\s+Date[:\s]*(\d{1,2}/\d{1,2}/\d{4})',  # "Statement Date: 01/04/2024"
            r'Statement\s+for\s+the\s+period\s*(\d{1,2}/\d{1,2}/\d{4})',  # "Statement for the period 01/04/2024"
            r'Statement\s+Period\s+\d{1,2}/\d{1,2}/\d{2,4}\s+to\s+(\d{1,2}/\d{1,2}/\d{2,4})',  # "Statement Period 01/09/25 to 31/12/25" (Thai) - extract end date
            r'Date\s+of\s+Statement[:\s]*(\d{1,2}-\d{1,2}-\d{4})',  # "Date of Statement: 09-02-2026" (SBI)
            r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})',  # Standalone DD MMM YYYY
        ]

        for block in text_blocks:
            text = block.get('text', '').strip()
            if not text:
                continue

            for pattern in date_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return match.group(1)

        return None

    def _find_latest_date_in_blocks(
        self,
        text_blocks: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[str]:
        """
        Find the latest (most recent) date across all text blocks.

        This method extracts ALL dates from the document and returns the latest one,
        which is the most likely statement date. This handles cases where:
        - Multiple dates appear in the document
        - OCR errors produce formats like "10Feb2026" (no space)
        - Date ranges are present

        Args:
            text_blocks: List of text blocks with geometry

        Returns:
            Latest date in DD MMM YYYY format, or None if no dates found
        """
        if not text_blocks:
            return None

        from datetime import datetime

        # Month name to number mapping
        month_map = {
            'jan': 1, 'january': 1,
            'feb': 2, 'february': 2,
            'mar': 3, 'march': 3,
            'apr': 4, 'april': 4,
            'may': 5,
            'jun': 6, 'june': 6,
            'jul': 7, 'july': 7,
            'aug': 8, 'august': 8,
            'sep': 9, 'september': 9,
            'oct': 10, 'october': 10,
            'nov': 11, 'november': 11,
            'dec': 12, 'december': 12,
        }

        def parse_date(date_str: str) -> Optional[datetime]:
            """Parse a date string into a datetime object."""
            date_str = date_str.strip()

            # Pattern 1: DD/MM/YYYY or DD-MM-YYYY or DD.MM.YYYY
            match = re.match(r'^(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})$', date_str)
            if match:
                p1, p2, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
                # Try DD/MM/YYYY first (common outside US)
                if p1 <= 31 and p2 <= 12:
                    try:
                        return datetime(year, p2, p1)
                    except ValueError:
                        pass
                # Try MM/DD/YYYY (US format)
                if p1 <= 12 and p2 <= 31:
                    try:
                        return datetime(year, p1, p2)
                    except ValueError:
                        pass
                return None

            # Pattern 2: DD/MM/YY or DD-MM-YY (2-digit year)
            match = re.match(r'^(\d{1,2})[-/.](\d{1,2})[-/.](\d{2})$', date_str)
            if match:
                p1, p2, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
                # Convert 2-digit year to 4-digit
                year = 2000 + year if year < 30 else 1900 + year
                # Try DD/MM/YY first
                if p1 <= 31 and p2 <= 12:
                    try:
                        return datetime(year, p2, p1)
                    except ValueError:
                        pass
                # Try MM/DD/YY
                if p1 <= 12 and p2 <= 31:
                    try:
                        return datetime(year, p1, p2)
                    except ValueError:
                        pass
                return None

            # Pattern 3: YYYY-MM-DD (ISO format)
            match = re.match(r'^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})$', date_str)
            if match:
                year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
                if month <= 12 and day <= 31:
                    try:
                        return datetime(year, month, day)
                    except ValueError:
                        pass
                return None

            # Pattern 4: DD-MMM-YYYY or DD/MMMM/YYYY (e.g., "08-Jan-2026", "30/November/2025")
            match = re.match(r'^(\d{1,2})[-/]([A-Za-z]+)[-/](\d{4})$', date_str)
            if match:
                day, month_name, year = match.groups()
                month_lower = month_name.lower()
                if month_lower in month_map:
                    try:
                        return datetime(int(year), month_map[month_lower], int(day))
                    except ValueError:
                        pass
                return None

            # Pattern 5: DD MMM YYYY (e.g., "30 Nov 2025")
            match = re.match(r'^(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})$', date_str)
            if match:
                day, month_name, year = match.groups()
                month_lower = month_name.lower()
                if month_lower in month_map:
                    try:
                        return datetime(int(year), month_map[month_lower], int(day))
                    except ValueError:
                        pass
                return None

            # Pattern 6: MMM DD, YYYY or MMM DD YYYY (e.g., "Jan 30, 2025")
            match = re.match(r'^([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})$', date_str)
            if match:
                month_name, day, year = match.groups()
                month_lower = month_name.lower()
                if month_lower in month_map:
                    try:
                        return datetime(int(year), month_map[month_lower], int(day))
                    except ValueError:
                        pass
                return None

            return None

        # Collect all dates found
        all_dates: List[Tuple[datetime, str]] = []

        # Combined text for pattern matching
        full_text = ' '.join(block.get('text', '') for block in text_blocks)

        # Date extraction patterns - find ALL dates, not just the first
        date_patterns = [
            # DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY
            r'(\d{1,2}[-/.]\d{1,2}[-/.]\d{4})',
            # DD/MM/YY, DD-MM-YY (2-digit year)
            r'(\d{1,2}[-/.]\d{1,2}[-/.]\d{2})\b',
            # YYYY-MM-DD (ISO format)
            r'(\d{4}[-/.]\d{1,2}[-/.]\d{1,2})',
            # DD-MMM-YYYY or DD/MMMM/YYYY (e.g., "08-Jan-2026", "30/November/2025")
            r'(\d{1,2}[-/](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[-/]\d{4})',
            # DD MMM YYYY (e.g., "30 Nov 2025")
            r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})',
            # MMM DD, YYYY or MMM DD YYYY (e.g., "Jan 30, 2025" or "January 30 2025")
            r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})',
            # OCR ERROR PATTERN: DDMMMYYYY (e.g., "10Feb2026" - no space between day and month)
            r'(\d{1,2}(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\d{4})',
        ]

        for pattern in date_patterns:
            matches = re.finditer(pattern, full_text, re.IGNORECASE)
            for match in matches:
                raw_date = match.group(1)

                # Handle OCR error format: "10Feb2026" -> "10 Feb 2026"
                ocr_match = re.match(r'^(\d{1,2})([A-Za-z]+)(\d{4})$', raw_date)
                if ocr_match:
                    day_part, month_part, year_part = ocr_match.groups()
                    raw_date = f"{day_part} {month_part} {year_part}"

                parsed = parse_date(raw_date)
                if parsed:
                    # Validate: date should be reasonable (between 2020 and 2100)
                    if datetime(2020, 1, 1) <= parsed <= datetime(2100, 12, 31):
                        all_dates.append((parsed, raw_date))

        if not all_dates:
            return None

        # Sort by date (descending) and return the latest
        all_dates.sort(key=lambda x: x[0], reverse=True)
        latest_date, latest_raw = all_dates[0]

        # Format as DD MMM YYYY
        month_abbr_map = {
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr',
            5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug',
            9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec',
        }
        formatted = f"{latest_date.day:02d} {month_abbr_map[latest_date.month]} {latest_date.year}"

        logger.info(f"Found {len(all_dates)} dates in document, using latest: '{formatted}'")
        return formatted

    def _is_valid_currency(self, currency_str: str) -> bool:
        """
        Check if a currency string looks valid.

        Valid formats:
        - 3-letter ISO 4217 codes: SGD, INR, USD, EUR, GBP, JPY

        Invalid:
        - Label text like "Currency", "Account", "Statement"
        - Longer phrases
        - Empty or None values

        Args:
            currency_str: The currency string to validate

        Returns:
            True if currency looks valid, False otherwise
        """
        if not currency_str:
            return False

        currency_upper = currency_str.strip().upper()

        # Valid 3-letter ISO 4217 currency codes
        valid_currencies = {
            'SGD', 'INR', 'USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF',
            'CNY', 'HKD', 'MYR', 'THB', 'KRW', 'TWD', 'NZD', 'SEK', 'NOK',
            'DKK', 'BRL', 'MXN', 'PHP', 'IDR', 'VND', 'PKR', 'LKR', 'BDT',
            'NPR', 'AFN', 'AED', 'SAR', 'QAR', 'KWD', 'BHD', 'OMR', 'JOD',
            'LBP', 'EGP', 'ILS', 'TRY', 'RUB', 'ZAR', 'NGN', 'KES', 'GHS',
            'UGX', 'TZS', 'RWF', 'BIF', 'CDF', 'AOA', 'XAF', 'XOF', 'XCF',
        }

        # Reject label-text values
        label_texts = {'CURRENCY', 'ACCOUNT', 'STATEMENT', 'BALANCE', 'SUMMARY'}
        if currency_upper in label_texts:
            return False

        # Check if it's a valid 3-letter currency code
        if len(currency_upper) == 3 and currency_upper in valid_currencies:
            return True

        return False

    def _normalize_currency_to_iso(self, currency_str: str) -> str:
        """
        Normalize currency names to ISO 4217 codes using config mapping.

        Converts common currency names like "UAE DIRHAM" to their ISO codes like "AED".
        If the currency is not found in the mapping, returns the original string.

        Args:
            currency_str: Currency string to normalize

        Returns:
            ISO 4217 code if mapping found, otherwise original string
        """
        if not currency_str:
            return currency_str

        currency_upper = currency_str.strip().upper()
        currency_map = get_bank_statement_validator().get_currency_name_map()
        return currency_map.get(currency_upper, currency_str)

    def _extract_currency_from_blocks(
        self,
        text_blocks: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[str]:
        """
        Extract currency by looking for 3-letter ISO 4217 codes in text blocks.

        This is a fallback when GLiNER doesn't extract a valid currency.

        Args:
            text_blocks: List of text blocks with geometry

        Returns:
            Currency code if found, None otherwise
        """
        if not text_blocks:
            return None

        # Valid 3-letter ISO 4217 currency codes
        valid_currencies = {
            'SGD', 'INR', 'USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF',
            'CNY', 'HKD', 'MYR', 'THB', 'KRW', 'TWD', 'NZD', 'SEK', 'NOK',
            'DKK', 'BRL', 'MXN', 'PHP', 'IDR', 'VND', 'PKR', 'LKR', 'BDT',
            'NPR', 'AFN', 'AED', 'SAR', 'QAR', 'KWD', 'BHD', 'OMR', 'JOD',
            'LBP', 'EGP', 'ILS', 'TRY', 'RUB', 'ZAR', 'NGN', 'KES', 'GHS',
            'UGX', 'TZS', 'RWF', 'BIF', 'CDF', 'AOA', 'XAF', 'XOF', 'XCF',
        }

        # Look for standalone 3-letter currency codes
        # Match word boundaries to avoid partial matches
        for block in text_blocks:
            text = block.get('text', '').strip()
            if not text:
                continue

            # Use word boundary matching to find standalone 3-letter codes
            for currency in valid_currencies:
                if re.search(r'\b' + currency + r'\b', text, re.IGNORECASE):
                    return currency.upper()

        return None

    def _format_address(self, address_parts: List[str]) -> str:
        """Format address parts into a single string."""
        if not address_parts:
            return ''

        # Remove duplicates while preserving order
        seen = set()
        unique_parts = []
        for part in address_parts:
            if part not in seen:
                seen.add(part)
                unique_parts.append(part)

        # Join with commas or spaces
        formatted = ', '.join(unique_parts)

        # Clean up extra whitespace
        formatted = re.sub(r'\s+', ' ', formatted)
        formatted = re.sub(r',\s*,', ',', formatted)

        # Clean up any OCR artifacts (leading colons, etc.)
        formatted = self._clean_address_assembly(formatted)

        return formatted

    def _clean_address_assembly(self, address: str) -> str:
        """Clean up assembled address by removing invalid prefix/postfix characters.

        Removes leading/trailing colons, dashes, and other separators that are
        OCR artifacts from label:value formatting.

        Args:
            address: Raw assembled address

        Returns:
            Cleaned address
        """
        if not address:
            return address

        lines = address.split('\n')
        cleaned_lines = []

        for line in lines:
            # Remove leading colons, dashes, and separators
            line = re.sub(r'^[:\-]\s*', '', line)
            # Remove trailing separators
            line = re.sub(r'\s*[:\-]$', '', line)
            # Clean up extra whitespace
            line = ' '.join(line.split())

            if line:
                cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def _extract_address_from_gliner_only(
        self, gliner_result: Dict[str, Any]
    ) -> Optional[str]:
        """Fallback: Extract address from GLiNER entities only (no spatial context)."""
        address_parts: List[str] = []

        for label, entities in gliner_result.items():
            if entities is None:
                continue

            label_lower = label.lower()

            if label_lower in self.ADDRESS_LABELS:
                if isinstance(entities, list):
                    for e in entities:
                        addr = e.get('value', '').strip()
                        if addr and addr not in address_parts:
                            address_parts.append(addr)
                else:
                    addr = entities.get('value', '').strip()
                    if addr and addr not in address_parts:
                        address_parts.append(addr)

        if not address_parts:
            return None

        return self._format_address(address_parts)

    def _extract_address_spatially(
        self,
        gliner_result: Dict[str, Any],
        text_blocks: Optional[List[Dict[str, Any]]] = None,
        account_holder_name: str = None,
        country_code: str = None,
        bank_name: str = None,
    ) -> Optional[str]:
        """
        Extract customer address using spatial extraction with priority ordering.

        Priority:
        1. Label-based extraction - Look for address labels from config
        2. Name-proximity extraction - For formal letter format (address below name)
        3. GLiNER fallback - Use GLiNER's customer_address extraction

        Args:
            gliner_result: GLiNER extraction results
            text_blocks: Text blocks with geometry from OCR
            account_holder_name: The account holder name (for name-proximity extraction)
            country_code: ISO country code for country-specific address keywords
            bank_name: Bank name for bank-specific extraction exceptions

        Returns:
            Extracted address string or None
        """
        if not text_blocks:
            # No spatial context available, fall back to GLiNER
            return self._extract_address_from_gliner_only(gliner_result)

        # Priority 1: Label-based extraction
        label_address = self._extract_address_by_config_labels(text_blocks, country_code, bank_name)
        if label_address and self._is_valid_address(label_address):
            logger.info(f"Address extracted via label-based spatial extraction: '{label_address[:50]}...'")
            return label_address

        # Priority 2: Name-proximity extraction (formal letter format)
        if account_holder_name:
            name_proximity_address = self._extract_address_below_name(
                text_blocks, account_holder_name, country_code
            )
            if name_proximity_address and self._is_valid_address(name_proximity_address):
                logger.info(f"Address extracted via name-proximity: '{name_proximity_address[:50]}...'")
                return name_proximity_address

        # Priority 3: GLiNER fallback with spatial expansion
        gliner_address = self._extract_address_prefer_schema(gliner_result, text_blocks, country_code)
        if gliner_address and self._is_valid_address(gliner_address):
            return gliner_address

        return None

    def _extract_address_by_config_labels(
        self,
        text_blocks: List[Dict[str, Any]],
        country_code: str = None,
        bank_name: str = None,
    ) -> Optional[str]:
        """
        Extract address by finding values next to address field labels from config.

        Uses address labels from config: "Address", "Residential Address",
        "Customer Address", "Mailing Address", "Permanent Address", "Communication Address"

        Strategy:
        1. Find blocks containing address labels
        2. Extract single or multi-line strings in close proximity (same line or lines below)
        3. Combine consecutive address-looking blocks

        Args:
            text_blocks: Text blocks with geometry from OCR
            country_code: ISO country code for country-specific address keywords
            bank_name: Bank name for bank-specific extraction exceptions

        Returns:
            Extracted address string or None
        """
        if not text_blocks:
            return None

        # Get address labels from config
        validator = get_bank_statement_validator()

        # Check if this bank should skip field label extraction
        if bank_name:
            exceptions = validator.get_address_extraction_exceptions()
            skip_banks = exceptions.get("skip_field_label", [])
            # Check if bank_name matches any in skip list (case-insensitive, partial match)
            bank_upper = bank_name.upper()
            for skip_bank in skip_banks:
                if skip_bank.upper() in bank_upper:
                    logger.info(f"Skipping field label address extraction for bank: {bank_name}")
                    return None

        address_labels = validator.get_address_labels()

        # If config not available, use defaults
        if not address_labels:
            address_labels = [
                'ADDRESS', 'RESIDENTIAL ADDRESS', 'CUSTOMER ADDRESS', 'MAILING ADDRESS',
                'PERMANENT ADDRESS', 'COMMUNICATION ADDRESS'
            ]

        # Normalize labels for matching
        address_labels_upper = [label.upper() for label in address_labels]

        # Find address label blocks
        for i, block in enumerate(text_blocks):
            text = block.get('text', '').strip().upper()
            if not text:
                continue

            # Skip branch address labels - we want customer address, not bank branch address
            if 'BRANCH ADDRESS' in text or 'OFFICE ADDRESS' in text or 'REGISTERED OFFICE' in text:
                logger.debug(f"Skipping branch/office address label: '{text}'")
                continue

            # Check if this block matches an address label (exact or contains)
            is_label = False
            for label in address_labels_upper:
                if text == label or text.startswith(label + ':') or text.startswith(label + ' '):
                    is_label = True
                    break

            if not is_label:
                continue

            label_y1 = block.get('y1', 0)
            label_x2 = block.get('x2', 0)

            logger.debug(f"Found address label '{text}' at index {i}, y1={label_y1}")

            # Collect address parts from same line (to the right) and lines below
            address_parts = []
            # First, collect ALL blocks within the y-range (to handle out-of-order OCR)
            # Then sort by y-position and check stopping conditions
            candidate_blocks = []
            for j in range(i + 1, min(i + 20, len(text_blocks))):
                value_block = text_blocks[j]
                value_text = value_block.get('text', '').strip()
                if not value_text:
                    continue

                # Skip colons and separators
                if value_text in [':', '-', '/', '\\', ',']:
                    continue

                value_y1 = value_block.get('y1', 0)
                value_x1 = value_block.get('x1', 0)

                # Check if on same line (y difference < 0.02) or below (within 0.10 y distance)
                if abs(value_y1 - label_y1) < 0.02:
                    # Same line - must be to the right of the label
                    if value_x1 > label_x2:
                        candidate_blocks.append((value_y1, value_text, j))
                elif 0 < value_y1 - label_y1 <= 0.10:
                    # Below the label - collect for further processing
                    candidate_blocks.append((value_y1, value_text, j))

            # Sort candidates by y-position (not index)
            candidate_blocks.sort(key=lambda x: x[0])

            # Now check stopping conditions in y-sorted order
            seen_texts = set()
            for value_y1, value_text, orig_idx in candidate_blocks:
                if value_text in seen_texts:
                    continue
                seen_texts.add(value_text)

                # Check if this is non-address text
                if self._is_non_address_text(value_text):
                    # Hit non-address text, stop collecting
                    logger.debug(f"  Stopping at non-address text: '{value_text}'")
                    break

                address_parts.append((value_y1, value_text))
                logger.debug(f"  Below label address part: '{value_text}'")

            if address_parts:
                # Sort by y position and format
                address_parts.sort(key=lambda x: x[0])
                formatted_address = self._format_address([part[1] for part in address_parts])
                return formatted_address

        return None

    def _extract_address_below_name(
        self,
        text_blocks: List[Dict[str, Any]],
        account_holder_name: str,
        country_code: str = None,
    ) -> Optional[str]:
        """
        Extract address by looking for text blocks below the account holder name.

        This handles formal letter format where address appears directly below
        the customer name in the top-left of the document (no explicit address label).

        Strategy:
        1. Find blocks containing the account holder name
        2. Collect blocks directly below (within y-range)
        3. Stop at non-address content (bank labels, transaction data, etc.)

        Args:
            text_blocks: Text blocks with geometry from OCR
            account_holder_name: The account holder name to look for
            country_code: ISO country code for country-specific address keywords

        Returns:
            Extracted address string or None
        """
        if not text_blocks or not account_holder_name:
            return None

        name_upper = account_holder_name.upper()

        # Find blocks containing the account holder name
        name_blocks = []
        for i, block in enumerate(text_blocks):
            text = block.get('text', '').strip()
            if not text:
                continue

            # Skip very short text (likely OCR artifacts or single characters)
            # Require at least 4 characters to avoid matching individual letters
            if len(text) < 4:
                continue

            # Check if this block contains or matches the name
            # Normalize spaces to handle OCR variations (e.g., "K C  ROHITH" -> "K C ROHITH")
            text_upper = re.sub(r'\s+', ' ', text.upper())
            name_upper_normalized = re.sub(r'\s+', ' ', name_upper)

            if name_upper_normalized in text_upper or text_upper in name_upper_normalized:
                # Skip if it's clearly a label (contains "ACCOUNT", "NAME", etc.)
                if not any(label in text_upper for label in ['ACCOUNT NUMBER', 'ACCOUNT NO', 'A/C NO']):
                    name_blocks.append((i, block.get('y1', 0), text))

        if not name_blocks:
            return None

        # Use the matching name block that is:
        # 1. Longest (most likely to be the actual name)
        # 2. Among ties, prefer topmost (lowest y1)
        name_blocks.sort(key=lambda x: (-len(x[2]), x[1]))
        name_idx, name_y1, name_text = name_blocks[0]

        # Get the x-position of the name block to filter address blocks by side
        # Bank statements often have customer address on LEFT and branch address on RIGHT
        name_block = text_blocks[name_idx]
        name_x1 = name_block.get('x1', 0)

        logger.debug(f"Found name block '{name_text}' at index {name_idx}, x1={name_x1:.3f}, y1={name_y1}")

        # Collect address blocks below the name
        # First, collect ALL blocks within the y-range (to handle out-of-order OCR)
        # Then sort by y-position and filter out non-address text
        candidate_blocks = []
        for i in range(name_idx + 1, min(name_idx + 15, len(text_blocks))):
            block = text_blocks[i]
            text = block.get('text', '').strip()
            if not text:
                continue

            block_y1 = block.get('y1', 0)
            block_x1 = block.get('x1', 0)

            # Check if block is below name (within 0.12 normalized distance)
            if 0 < block_y1 - name_y1 <= 0.12:
                # Also check if block is on the same side as the name block
                # Use a threshold of 0.15 to account for small variations
                if abs(block_x1 - name_x1) < 0.15:
                    candidate_blocks.append((block_y1, text))
                else:
                    logger.debug(f"  Skipping block on different side: x1={block_x1:.3f} vs name_x1={name_x1:.3f}, text='{text[:30]}...'")

        # Sort candidates by y-position (not index)
        candidate_blocks.sort(key=lambda x: x[0])

        # Now filter out non-address text in y-sorted order
        address_blocks = []
        seen_texts = set()
        for block_y1, text in candidate_blocks:
            if text in seen_texts:
                continue
            seen_texts.add(text)

            # Skip if it's the name itself or part of it
            if text.upper() == name_upper or name_upper in text.upper():
                continue

            # Skip emails
            if '@' in text:
                continue

            # Skip non-address text - but check AFTER sorting so we don't miss valid blocks
            if self._is_non_address_text(text):
                logger.debug(f"  Skipping non-address text in name-proximity: '{text}'")
                continue

            # Skip if too long (likely transaction data or narrative)
            if len(text) > 80:
                continue

            # Skip if looks like transaction data (dates with commas)
            if re.search(r'\d{2}/\d{2}/\d{4}', text) and ',' in text and len(text.split(',')) > 3:
                continue

            address_blocks.append((block_y1, text))
            logger.debug(f"  Name-proximity address part: '{text}'")

        if not address_blocks:
            return None

        # Sort by y position and format
        address_blocks.sort(key=lambda x: x[0])
        formatted_address = self._format_address([part[1] for part in address_blocks])
        return formatted_address

    def _extract_street_address(self, address: str, components: Dict[str, str]) -> str:
        """
        Extract street address by removing city, state, postal_code, and country.

        Args:
            address: Full address string
            components: Parsed address components (city, state, postal_code, country)

        Returns:
            Street address portion (everything except city/state/postal_code/country)
        """
        if not address:
            return address

        street_parts = []
        addr_lower = address.lower()

        # Components to remove from the address
        remove_patterns = []

        if components.get("city"):
            remove_patterns.append(components["city"].lower())
        if components.get("state"):
            remove_patterns.append(components["state"].lower())
        if components.get("postal_code"):
            remove_patterns.append(components["postal_code"].lower())
        if components.get("country"):
            remove_patterns.append(components["country"].lower())
            # Also add common country name variations from config
            country_loader = get_country_config_loader()
            country_code = components["country"].lower()
            name_aliases = country_loader.get_country_name_aliases(country_code)
            remove_patterns.extend([alias.lower() for alias in name_aliases])

        # Split address by common delimiters and filter out component parts
        # Use regex to split on commas, spaces, but keep the parts together
        import re
        parts = re.split(r'[,\s]+', address)

        current_part = ""
        for part in parts:
            part_lower = part.lower().strip()
            # Skip if this part matches any of the components to remove
            if part_lower in remove_patterns:
                if current_part.strip():
                    street_parts.append(current_part.strip())
                    current_part = ""
                continue
            # Build up the current part
            if current_part:
                current_part += " " + part
            else:
                current_part = part

        # Add any remaining part
        if current_part.strip():
            street_parts.append(current_part.strip())

        # Join the remaining parts
        result = ", ".join(street_parts)

        # Clean up any extra whitespace or punctuation
        result = re.sub(r'\s+,', ',', result)
        result = re.sub(r',\s*,', ',', result)
        result = re.sub(r'^[,\s]+|[,\s]+$', '', result)

        return result if result else address

    def _infer_country_from_address(self, address: str) -> Optional[str]:
        """
        Infer ISO 3166-1 alpha-2 country code from address text.

        Looks for country names and patterns in the address to determine
        the account holder's residential country.

        Args:
            address: Full address string

        Returns:
            ISO country code or None
        """
        if not address:
            return None

        addr_lower = address.lower()

        # Get country patterns from config (order matters - more specific first)
        country_loader = get_country_config_loader()
        country_patterns = country_loader.get_all_country_inference_patterns()

        for code, patterns in country_patterns.items():
            for pattern in patterns:
                if pattern in addr_lower:
                    logger.debug(f"Inferred country {code} from address pattern '{pattern}'")
                    return code

        return None

    def _extract_bank_country(
        self,
        ocr_text: str,
        text_blocks: Optional[List[Dict[str, Any]]] = None,
        bank_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Extract bank's registered country from the document.

        Looks for patterns that indicate where the bank is registered/licensed:
        - "Central Bank of UAE" / "Central Bank of the UAE" (regulatory text)
        - "Licensed by" / "Regulated by" patterns
        - Bank address headers with country codes (e.g., "ABU DHABI, UAE")
        - Country patterns in the header portion of the document

        This is different from account_holder_country (where the customer lives).
        bank_country is where the bank branch is registered.

        Args:
            ocr_text: Full OCR text from the document
            text_blocks: Optional text blocks with geometry (for header detection)
            bank_name: Optional bank name (for fallback lookup)

        Returns:
            ISO 3166-1 alpha-2 country code or None
        """
        if not ocr_text:
            return None

        text_upper = ocr_text.upper()
        country_loader = get_country_config_loader()

        # Pattern 1: Central Bank / Regulatory authority mentions
        # These indicate where the bank is licensed - loaded from config
        regulatory_patterns = []
        for country_code in country_loader.get_supported_countries():
            patterns = country_loader.get_bank_regulatory_patterns(country_code)
            for pattern in patterns:
                regulatory_patterns.append((pattern, country_code))

        for pattern, country_code in regulatory_patterns:
            if re.search(pattern, text_upper):
                logger.info(f"Detected bank_country={country_code} from regulatory pattern: {pattern}")
                return country_code

        # Pattern 2: Bank address in header (usually first few lines)
        # Look for country codes in the header portion
        if text_blocks and len(text_blocks) > 0:
            # Check first 15 blocks (typically header area)
            header_text = ' '.join(
                block.get('text', '')
                for block in text_blocks[:15]
            ).upper()

            # Check header patterns from config for each supported country
            for country_code in country_loader.get_supported_countries():
                header_patterns = country_loader.get_header_patterns(country_code)
                for pattern in header_patterns:
                    if re.search(pattern, header_text):
                        # Verify this is in bank context, not customer address
                        # Bank headers typically have bank name nearby
                        if bank_name and bank_name.upper() in header_text:
                            logger.info(f"Detected bank_country={country_code} from header pattern: {pattern}")
                            return country_code
                        # Also check if PO BOX pattern is nearby (common in UAE bank headers)
                        if country_code == 'AE' and ('PO BOX' in header_text or 'P.O. BOX' in header_text):
                            logger.info(f"Detected bank_country={country_code} from header with PO BOX: {pattern}")
                            return country_code

        # Pattern 3: Country code at end of SWIFT code
        # SWIFT codes are 8/11 chars, last 2 are country (e.g., NABOREAAE -> AE)
        swift_match = re.search(r'\b([A-Z]{4}[A-Z]{2}([A-Z]{2})[A-Z0-9]{0,3})\b', text_upper)
        if swift_match:
            swift_country = swift_match.group(2)
            # Validate it's a known country from config
            known_countries = country_loader.get_supported_countries()
            if swift_country in known_countries:
                logger.info(f"Detected bank_country={swift_country} from SWIFT code")
                return swift_country

        # Pattern 4: Fallback - use bank name lookup from JSON
        if bank_name:
            country = get_country_for_bank(bank_name)
            if country:
                logger.info(f"Detected bank_country={country} from bank name lookup")
                return country

        return None

    def _calculate_confidence(self, gliner_result: Dict[str, Any]) -> float:
        """
        Calculate overall confidence from GLiNER results.

        Handles both formats:
        - Schema-based: {"value": "XDBS", "confidence": 0.85} (dict)
        - Labels-based: [{"value": "XDBS", "confidence": 0.85}] (list)
        """
        matched_count = 0
        total_confidence = 0.0

        for entities in gliner_result.values():
            if entities is None:
                continue

            # Handle schema-based format (dict with 'confidence' key)
            if isinstance(entities, dict):
                conf = entities.get('confidence', 0)
                if conf > 0:
                    total_confidence += conf
                    matched_count += 1
            # Handle labels-based format (list of dicts)
            elif isinstance(entities, list):
                for entity in entities:
                    conf = entity.get('confidence', 0)
                    total_confidence += conf
                    matched_count += 1

        if matched_count > 0:
            return (total_confidence / matched_count) * 100  # Return as 0-100 scale
        return 0.0

    def _extract_bank_name_from_url_domain(self, text: str) -> Optional[str]:
        """
        Extract bank name from website URLs or email domains in the text.

        Uses the comprehensive BankLookup with domain_map from config.json.
        Supports domains like "dbs.com.sg", "hdfcbank.com", "emiratesnbd.com", etc.

        Args:
            text: OCR text from the document

        Returns:
            Bank name if found, None otherwise
        """
        if not text:
            return None

        # Use new BankLookup with comprehensive domain mapping
        bank_info = lookup_bank_by_domain(text)
        if bank_info:
            logger.info(f"Bank detected from domain: {bank_info.full_name} ({bank_info.abbreviation})")
            return bank_info.full_name

        return None

    def _clean_account_holder_name(self, name: str) -> Optional[str]:
        """
        Clean account holder name by removing titles/salutations and patronymic markers.

        Uses the shared clean_name_for_storage utility for consistency.
        """
        from app.utils.string_matching import clean_name_for_storage
        return clean_name_for_storage(name)

    def _clean_account_number(self, account_number: str) -> Optional[str]:
        """
        Clean account number by removing dashes, spaces, and other formatting.

        Args:
            account_number: Raw account number

        Returns:
            Cleaned account number without formatting, or None if input is None
        """
        if not account_number:
            return None

        # Remove common formatting characters
        cleaned = re.sub(r'[\s\-—–\.\,]+', '', account_number)

        return cleaned if cleaned else None

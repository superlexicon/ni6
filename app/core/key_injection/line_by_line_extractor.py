"""
Layout-agnostic bank statement extraction using simple line-by-line algorithm.

This module extracts bank statement fields by iterating through each line in the
docTR Document and using specific patterns and spatial proximity to identify:
- Bank name and address
- Account holder name and address
- Account number
- Account currency
- Statement date (newest date found in the document)
"""

import re
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass

from .global_banks import get_bank_info, BANKS_BY_NAME

logger = logging.getLogger(__name__)


# ISO 4217 Currency Codes (comprehensive list)
CURRENCY_CODES = {
    # Major currencies
    'USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'CNY', 'HKD', 'SGD',
    # Asian currencies
    'MYR', 'THB', 'IDR', 'PHP', 'VND', 'INR', 'KRW', 'TWD', 'AED', 'SAR',
    'PKR', 'LKR', 'BDT', 'NPR', 'LAK', 'KHR', 'BND', 'MOP', 'CZK', 'MMK',
    # European currencies
    'SEK', 'NOK', 'DKK', 'PLN', 'RUB', 'TRY', 'RON', 'HUF', 'CZK', 'BGN',
    # Americas
    'MXN', 'BRL', 'ARS', 'CLP', 'COP', 'PEN', 'UYU', 'COP', 'BOB',
    # African and Middle Eastern
    'ZAR', 'EGP', 'NGN', 'KES', 'GHS', 'UGX', 'TZS', 'RWF', 'BWP', 'NAD',
    'OMR', 'QAR', 'KWD', 'BHD', 'JOD', 'ILS',
    # Pacific
    'NZD', 'FJD', 'PGK', 'VUV', 'SBD', 'TOP', 'WST',
}


@dataclass
class LineExtractionResult:
    """Result of bank statement extraction using line-by-line algorithm."""
    bank_name: Optional[str]
    bank_address: Optional[str]
    account_holder_name: Optional[str]
    account_holder_address: Optional[str]
    account_holder_country: Optional[str]
    account_number: Optional[str]
    account_currency: Optional[str]
    statement_date: Optional[str]
    confidence: float
    extraction_metadata: Dict[str, Any]


class LineByLineExtractor:
    """
    Extract bank statement fields by iterating through each line.

    Algorithm:
    1. Iterate through each line in docTR Document
    2. Use passport first name as anchor for account holder name
    3. Use global banks database for bank name
    4. Extract multi-line addresses using spatial proximity (similar x1, y1 within 1 line height)
    5. Extract account number (8-15 digits, no "+" prefix to avoid phone numbers)
    6. Extract currency (3-letter ISO code)
    7. Stop when all 6 fields found
    """

    def __init__(self):
        self.currency_codes = CURRENCY_CODES

    def _get_bank_info_fuzzy(self, text: str) -> Optional[Any]:
        """
        Get bank info using fuzzy matching to handle OCR typos.

        Handles both full names ("DBS Bank") and abbreviations ("DBS", "OCBC", "UOB").

        Args:
            text: Text to match against bank names

        Returns:
            BankInfo if found, None otherwise
        """
        try:
            from rapidfuzz import fuzz, process
        except ImportError:
            # Fall back to exact matching if RapidFuzz not available
            return get_bank_info(text)

        if not text:
            return None

        text_lower = text.strip().lower()

        # Try exact match first (fastest)
        exact_match = get_bank_info(text)
        if exact_match:
            from app.core.logger import get_logger
            logger = get_logger()
            logger.info(f"  Exact matched '{text}' to '{exact_match.name}'")
            return exact_match

        # For longer text, use sliding window fuzzy matching
        # This helps when the bank name is embedded in a longer string
        if len(text) >= 5:
            words = text_lower.split()
            best_match = None
            best_score = 0

            # Try different window sizes (2-7 words)
            for window_size in range(2, min(8, len(words) + 1)):
                for i in range(len(words) - window_size + 1):
                    window = ' '.join(words[i:i + window_size])

                    # Compare against all bank names
                    result = process.extractOne(
                        window,
                        BANKS_BY_NAME.keys(),
                        scorer=fuzz.ratio,
                        score_cutoff=60  # Lower threshold for short bank abbreviations like DBS, POSB
                    )

                    if result:
                        match, score, _ = result
                        if score > best_score:
                            best_score = score
                            best_match = match

            if best_match:
                bank_info = get_bank_info(best_match)
                if bank_info:
                    from app.core.logger import get_logger
                    logger = get_logger()
                    logger.info(f"  Fuzzy matched '{text[:50]}...' to '{bank_info.name}' (similarity: {best_score}%)")
                    return bank_info

        return None

    def extract(self, doc_lines: List[Dict]) -> LineExtractionResult:
        """
        Extract bank statement fields using GLiNER zero-shot NER.

        Pure GLiNER approach - uses ML model directly on docTR output
        for all entity extraction. No heuristic fallbacks or spatial clustering.

        Args:
            doc_lines: List of line dicts from docTR with 'text' and 'geometry' keys
                       Example: [{'text': 'Account Name: John Doe', 'geometry': [[x1, y1], [x2, y2]]}]

        Returns:
            LineExtractionResult with all extracted fields
        """
        from app.core.logger import get_logger
        logger = get_logger()

        logger.info("=== PURE GLINER BANK STATEMENT EXTRACTION ===")
        logger.info(f"Processing {len(doc_lines)} lines")

        # Initialize results
        found_fields = {
            'bank_name': None,
            'bank_address': None,
            'account_holder_name': None,
            'account_holder_address': None,
            'account_holder_country': None,
            'account_number': None,
            'account_currency': None,
            'statement_date': None,
        }

        # === GLiNER ML-based extraction (primary and only strategy) ===
        logger.info("=== GLiNER ML-based extraction ===")
        gliner_entities = self._extract_all_entities_with_gliner(doc_lines)

        # Collect all addresses for classification (handled after GLiNER extraction)
        all_addresses = []  # List of (text, confidence) tuples

        if gliner_entities:
            # Extract all fields from GLiNER results
            for field_name, result in gliner_entities.items():
                # Handle multi-value labels (like addresses and branches)
                if field_name == 'address' and result:
                    if isinstance(result, list):
                        # GLiNER returned multiple addresses
                        for addr in result:
                            if isinstance(addr, dict) and addr.get('value'):
                                all_addresses.append((addr['value'], addr.get('confidence', 0.0)))
                    elif isinstance(result, dict) and result.get('value'):
                        # Single address (backward compatibility)
                        all_addresses.append((result['value'], result.get('confidence', 0.0)))
                    continue  # Skip to next field, addresses are handled later

                # Handle branch as multi-value (can have multiple bank abbreviations)
                if field_name == 'branch' and result:
                    branches_to_process = []
                    if isinstance(result, list):
                        # GLiNER returned multiple branches
                        for branch in result:
                            if isinstance(branch, dict) and branch.get('value'):
                                branches_to_process.append((branch['value'], branch.get('confidence', 0.0)))
                    elif isinstance(result, dict) and result.get('value'):
                        # Single branch (backward compatibility)
                        branches_to_process.append((result['value'], result.get('confidence', 0.0)))

                    # Process all branches to find the best bank name match
                    for branch_value, branch_conf in branches_to_process:
                        bank_info = self._get_bank_info_fuzzy(branch_value)
                        if bank_info:
                            bank_name = bank_info.swift_code
                            # Prefer this over existing bank name if it's a better match
                            current_bank = found_fields.get('bank_name', '')
                            if not current_bank or len(bank_name) > len(current_bank):
                                found_fields['bank_name'] = bank_name
                                logger.info(f"✓ Bank SWIFT (from GLiNER branch, conf={branch_conf:.2f}): '{bank_name}'")
                    continue  # Skip to next iteration, we processed all branches

                # Handle single-value labels
                if result and result.get('value'):
                    value = result['value']
                    confidence = result.get('confidence', 0.0)
                    reclassified = result.get('reclassified', False)

                    # Map GLiNER fields to our schema
                    if field_name == 'account holder name':
                        name = self._remove_name_titles(value)
                        found_fields['account_holder_name'] = name
                        found_fields.setdefault('account_holder_name_gliner_confidence', confidence)
                        found_fields.setdefault('account_holder_name_gliner_reclassified', reclassified)
                        reclass_str = ' (reclassified)' if reclassified else ''
                        logger.info(f"✓ Account holder name (GLiNER{reclass_str}, conf={confidence:.2f}): '{name}'")

                    elif field_name == 'bank name':
                        # Skip if this looks like a person name (GLiNER misclassification)
                        if self._looks_like_valid_name(value, gliner_confidence=confidence):
                            logger.info(f"GLiNER 'bank name' '{value}' looks like person name - storing as candidate account holder")
                            # Store as candidate account holder name (use if no account holder found)
                            if not found_fields.get('account_holder_name'):
                                found_fields.setdefault('account_holder_name_candidate', value)
                                found_fields.setdefault('account_holder_name_candidate_confidence', confidence)
                            continue
                        # Verify with global banks database for consistency
                        bank_info = self._get_bank_info_fuzzy(value)
                        bank_name = bank_info.swift_code if bank_info else value
                        found_fields['bank_name'] = bank_name
                        logger.info(f"✓ Bank SWIFT (GLiNER, conf={confidence:.2f}): '{bank_name}'")

                    elif field_name == 'account number':
                        # Validate: reject labels and non-numeric values
                        if self._is_valid_account_number(value):
                            # Normalize account number (remove dashes, spaces)
                            normalized_acct = value.replace('-', '').replace(' ', '')
                            found_fields['account_number'] = normalized_acct
                            logger.info(f"✓ Account number (GLiNER, conf={confidence:.2f}): '{normalized_acct[:4]}****'")
                        else:
                            logger.info(f"Skipping GLiNER 'account number' '{value}' (invalid format)")
                            # Try to find the actual account number value nearby
                            nearby_value = self._find_label_value_nearby(doc_lines, 'account number')
                            if nearby_value:
                                # Normalize account number (remove dashes, spaces)
                                normalized_acct = nearby_value.replace('-', '').replace(' ', '')
                                found_fields['account_number'] = normalized_acct
                                logger.info(f"✓ Account number (label-value lookup): '{normalized_acct[:4]}****'")

                    elif field_name == 'currency':
                        found_fields['account_currency'] = value
                        logger.info(f"✓ Currency (GLiNER, conf={confidence:.2f}): '{value}'")

                    elif field_name == 'statement date':
                        found_fields['statement_date'] = value
                        logger.info(f"✓ Statement date (GLiNER, conf={confidence:.2f}): '{value}'")

        # Detect country early (needed for address validation and classification)
        detected_country = None
        full_text = '\n'.join([line.get('text', '').strip() for line in doc_lines])
        from .global_banks import detect_country_in_text
        detected_country = detect_country_in_text(full_text)
        if detected_country:
            logger.info(f"Detected country: {detected_country}")
            found_fields['account_holder_country'] = detected_country

        # === Address Validation using pypostal ===
        if all_addresses:
            logger.info(f"Validating {len(all_addresses)} address(es) with pypostal")

            # Import validation module
            from .address_validator import should_keep_address

            filtered_addresses = []
            filtered_count = 0

            for addr_text, addr_conf in all_addresses:
                keep, final_conf, reason = should_keep_address(
                    addr_conf,
                    addr_text,
                    detected_country
                )

                if keep:
                    filtered_addresses.append((addr_text, final_conf))
                    logger.debug(f"Kept address: '{addr_text[:40]}...' ({reason}, conf={final_conf:.2f})")
                else:
                    filtered_count += 1
                    logger.debug(f"Filtered address: '{addr_text[:40]}...' ({reason})")

            all_addresses = filtered_addresses
            logger.info(f"Post-pypostal: {len(all_addresses)} addresses remain (filtered {filtered_count})")

        # After GLiNER processing, check if we should use the candidate account holder name
        # Prefer candidate from 'bank name' (misclassified person name) over reclassified 'branch' name
        candidate_name = found_fields.get('account_holder_name_candidate')
        candidate_conf = found_fields.get('account_holder_name_candidate_confidence', 0.0)
        current_name = found_fields.get('account_holder_name')
        current_conf = found_fields.get('account_holder_name_gliner_confidence', 0.0)
        current_reclassified = found_fields.get('account_holder_name_gliner_reclassified', False)

        # Use candidate if:
        # 1. No account holder name found, OR
        # 2. Current name was reclassified (likely misclassified 'branch') and candidate has higher confidence
        should_use_candidate = (
            not current_name or
            (current_reclassified and candidate_conf > current_conf)
        )

        if should_use_candidate and candidate_name:
            if current_name and current_reclassified:
                logger.info(f"Replacing reclassified account holder name '{current_name}' (conf={current_conf:.2f}) "
                          f"with candidate from 'bank name': '{candidate_name}' (conf={candidate_conf:.2f})")
            else:
                logger.info(f"No account holder name found, using candidate from 'bank name': '{candidate_name}'")
            found_fields['account_holder_name'] = found_fields.pop('account_holder_name_candidate')
            # Clean up temporary fields
            found_fields.pop('account_holder_name_candidate_confidence', None)
            found_fields.pop('account_holder_name_gliner_confidence', None)
            found_fields.pop('account_holder_name_gliner_reclassified', None)

        # === Fallback: S/O Name Extraction ===
        # When GLiNER doesn't find account holder name, try S/O pattern detection
        should_fallback = (
            not found_fields['account_holder_name'] or
            self._looks_like_bank_abbreviation(found_fields['account_holder_name']) or
            # NEW: Override if the found name contains address terms (likely misclassification)
            self._looks_like_address_component(found_fields['account_holder_name'])
        )

        if should_fallback:
            for line in doc_lines:
                text = line.get('text', '').strip()
                # Look for S/O, D/O, A/L patterns
                if re.search(r'\s+(S/O|D/O|A/L|S/O\.|D/O\.|A/L\.)\s+', text, re.IGNORECASE):
                    # Validate this looks like a name (not just "S/O" alone)
                    if self._looks_like_valid_name(text, gliner_confidence=0.0):
                        found_fields['account_holder_name'] = self._remove_name_titles(text)
                        logger.info(f"✓ Account holder name (S/O fallback): '{found_fields['account_holder_name']}'")
                        break

        # === Address Classification using Spatial Proximity ===
        if all_addresses:
            logger.info(f"Classifying {len(all_addresses)} address(es) by spatial proximity")

            # Build address list with coordinates by matching back to docTR lines
            addresses_with_coords = []
            for addr_text, addr_conf in all_addresses:
                # Find matching docTR line for this address
                for line in doc_lines:
                    line_text = line.get('text', '').strip()
                    # Use substring matching for flexibility with OCR variations
                    # When GLiNER extracts partial address (e.g., "MARINE CRESCENT"),
                    # use the full docTR line (e.g., "BLK 29 MARINE CRESCENT") for complete address
                    if addr_text in line_text or line_text in addr_text:
                        coords = self._get_line_coordinates(line)
                        # If the docTR line contains more information than the GLiNER entity,
                        # use the full docTR line text to get the complete address
                        full_address_text = line_text if len(line_text) > len(addr_text) else addr_text
                        addresses_with_coords.append({
                            'text': full_address_text,
                            'coords': coords
                        })
                        logger.debug(f"  Matched address '{addr_text[:30]}...' to docTR line, using '{full_address_text[:30]}...'")
                        break
                else:
                    # No coordinate match found - add without coords for fallback
                    logger.debug(f"  No coordinate match for address '{addr_text[:30]}...'")
                    addresses_with_coords.append({
                        'text': addr_text,
                        'coords': None
                    })

            # Combine nearby addresses (multi-line addresses)
            if len(addresses_with_coords) > 1:
                addresses_with_coords = self._combine_nearby_addresses(addresses_with_coords)
                logger.info(f"Combined into {len(addresses_with_coords)} address group(s)")

            # Get coordinates for account holder name and bank name
            account_holder_coords = None
            bank_name_coords = None

            # Find coordinates by matching text back to docTR lines
            if found_fields['account_holder_name']:
                for line in doc_lines:
                    if found_fields['account_holder_name'] in line.get('text', ''):
                        account_holder_coords = self._get_line_coordinates(line)
                        logger.debug(f"  Found account holder name coords: {account_holder_coords}")
                        break

            if found_fields['bank_name']:
                bank_name_found = False
                # Collect all matches and pick the one in the header area (lowest Y position)
                bank_name_matches = []

                # Strategy 1: Try case-insensitive substring match with cleaned bank name
                bank_name_lower = found_fields['bank_name'].lower()
                for line in doc_lines:
                    if bank_name_lower in line.get('text', '').lower():
                        coords = self._get_line_coordinates(line)
                        bank_name_matches.append((coords, 'exact'))
                        logger.debug(f"  Found bank name coords (exact): {coords}")

                # Strategy 2: Try fuzzy matching (always run, collect all matches)
                from .global_banks import BANKS_BY_NAME
                # Try matching against bank database keys
                for bank_key in BANKS_BY_NAME.keys():
                    for line in doc_lines:
                        if bank_key.lower() in line.get('text', '').lower():
                            coords = self._get_line_coordinates(line)
                            bank_name_matches.append((coords, f'key_{bank_key}'))
                            logger.debug(f"  Found bank name coords (key '{bank_key}'): {coords}")

                # Strategy 3: Try abbreviations (FAB, ENBD, etc.)
                common_abbrs = ['FAB', 'ENBD', 'ADCB', 'DIB', 'MASHREQ', 'RAKBANK',
                               'DBS', 'POSB', 'OCBC', 'UOB', 'CIMB', 'HSBC',
                               'CITIBANK', 'STANDARD', 'CHARTERED', 'ABU DHABI']
                for abbr in common_abbrs:
                    for line in doc_lines:
                        if abbr.lower() in line.get('text', '').lower():
                            coords = self._get_line_coordinates(line)
                            bank_name_matches.append((coords, f'abbr_{abbr}'))
                            logger.debug(f"  Found bank name coords (abbr '{abbr}'): {coords}")

                # Select the match for address classification
                # For address classification, prefer footer position (highest Y) since bank addresses
                # are typically at the bottom of bank statements
                if bank_name_matches:
                    # Sort by Y position (y1) descending - pick footer area for address classification
                    bank_name_matches.sort(key=lambda x: x[0][1], reverse=True)
                    bank_name_coords = bank_name_matches[0][0]
                    bank_name_found = True
                    logger.debug(f"  Selected bank name coords (footer position, y={bank_name_coords[1]:.0f}): {bank_name_coords}")

                if not bank_name_found:
                    logger.debug(f"  Could not find bank name coords for '{found_fields['bank_name']}'")

            # Classify addresses by proximity
            if addresses_with_coords:
                classified = self._classify_addresses_by_proximity(
                    addresses_with_coords,
                    account_holder_coords,
                    bank_name_coords,
                    country_code=found_fields.get('account_holder_country')
                )
                logger.info(f"Address classification result: {classified}")

                if 'account_holder_address' in classified:
                    # Find the coordinates of the classified address
                    classified_address_coords = None
                    classified_address_text = classified['account_holder_address']

                    # Find coordinates by matching the address text back to addresses_with_coords
                    for addr_with_coords in addresses_with_coords:
                        if addr_with_coords.get('coords') and classified_address_text in addr_with_coords.get('text', ''):
                            classified_address_coords = addr_with_coords['coords']
                            break

                    # Try to complete the address by finding nearby address components
                    try:
                        completed_address = self._complete_address_from_nearby_lines(
                            classified_address_text,
                            doc_lines,
                            classified_address_coords  # Use address coords, not account holder coords
                        )
                        # Normalize address using pypostal
                        from .address_validator import normalize_address
                        normalized_address = normalize_address(completed_address)
                        found_fields['account_holder_address'] = normalized_address
                        logger.info(f"✓ Account holder address (classified, normalized): '{normalized_address[:50]}...'")
                    except Exception as e:
                        logger.warning(f"Address completion failed: {e}")
                        # Normalize address using pypostal
                        from .address_validator import normalize_address
                        normalized_address = normalize_address(classified_address_text)
                        found_fields['account_holder_address'] = normalized_address
                        logger.info(f"✓ Account holder address (classified, no completion, normalized): '{normalized_address[:50]}...'")

                if 'bank_address' in classified:
                    found_fields['bank_address'] = classified['bank_address']
                    logger.info(f"✓ Bank address (classified): '{classified['bank_address'][:50]}...'")

        # Fallback: If no bank name found, try direct extraction from full text
        if not found_fields['bank_name']:
            full_text = '\n'.join([
                line.get('text', '').strip()
                for line in doc_lines
                if line.get('text', '').strip()
            ])
            bank_info = self._get_bank_info_fuzzy(full_text)
            if bank_info:
                found_fields['bank_name'] = bank_info.swift_code
                logger.info(f"✓ Bank SWIFT (fallback from full text): '{bank_info.swift_code}'")

        # Infer country from address
        if found_fields['account_holder_address']:
            from .global_banks import detect_country_in_text
            found_fields['account_holder_country'] = detect_country_in_text(
                found_fields['account_holder_address']
            )
            if found_fields['account_holder_country']:
                logger.info(f"✓ Account holder country: {found_fields['account_holder_country']}")

        # Calculate confidence
        confidence = self._calculate_confidence(found_fields)

        # Remove salutations/titles from account holder name
        if found_fields.get('account_holder_name'):
            original_name = found_fields['account_holder_name']
            found_fields['account_holder_name'] = self._remove_name_titles(original_name)
            if original_name != found_fields['account_holder_name']:
                logger.info(f"Removed salutation from account holder name: '{original_name}' -> '{found_fields['account_holder_name']}'")

        logger.info("=== EXTRACTION COMPLETE ===")
        logger.info(f"Confidence: {confidence:.2f}")
        for field, value in found_fields.items():
            logger.info(f"  {field}: {value}")

        return LineExtractionResult(
            bank_name=found_fields['bank_name'],
            bank_address=found_fields['bank_address'],
            account_holder_name=found_fields['account_holder_name'],
            account_holder_address=found_fields['account_holder_address'],
            account_holder_country=found_fields['account_holder_country'],
            account_number=found_fields['account_number'],
            account_currency=found_fields['account_currency'],
            statement_date=found_fields['statement_date'],
            confidence=confidence,
            extraction_metadata={
                'method': 'pure_gliner',
            }
        )

    def _extract_all_entities_with_gliner(self, doc_lines: List[Dict]) -> Optional[Dict[str, Any]]:
        """
        Extract ALL bank statement entities using GLiNER zero-shot NER.

        Args:
            doc_lines: List of line dicts from docTR with 'text' and 'geometry' keys

        Returns:
            Dict mapping field names to GLiNER results with confidence scores.
        """
        try:
            from app.core.gliner_ner_model import get_gliner_ner_model

            # Build full text from document lines
            full_text = '\n'.join([
                line.get('text', '').strip()
                for line in doc_lines
                if line.get('text', '').strip()
            ])

            if not full_text or len(full_text) < 20:
                logger.debug("Document text too short for GLiNER extraction")
                return None

            # Get GLiNER model
            gliner_model = get_gliner_ner_model()

            # Extract ALL entities
            entities = gliner_model.extract_bank_statement_entities(full_text)

            logger.info(f"GLiNER extracted {len([e for e in entities.values() if e])} entities")

            return entities

        except ImportError:
            logger.warning("GLiNER not available")
        except Exception as e:
            logger.warning(f"GLiNER extraction failed: {e}")

        return None

    # ==============================================================================
    # TWO-PASS HEURISTIC EXTRACTION (COMMENTED OUT - using pure GLiNER instead)
    # ==============================================================================
    # The following methods are kept for potential fallback use but are not called
    # in the pure GLiNER extraction flow. They can be re-enabled if needed.
    #
    # # === PASS 1: Extract Core Fields ===
    # # === PASS 2: Extract Addresses ===
    # # === Spatial Clustering ===
    # # === Address Classification ===
    # ==============================================================================

    def _score_line_as_account_holder_name(
        self,
        line: Dict,
        all_lines: List[Dict]
    ) -> Tuple[int, List[str]]:
        """
        Score a line as a potential account holder name using heuristics.

        Scoring factors (updated with reduced positional bias):
        - Positional: Top half of document (reduced from heavy upper-left bias)
        - Pattern: Malaysian/Indian S/O, D/O, A/L patterns
        - Pattern: Western name formats (First Middle Last, title case)
        - Pattern: Entity/company suffixes (PTY LTD, LTD, INC, LLC, SDN BHD)
        - Text: 2+ words, no digits, clean characters
        - Format: All caps (common in statements)

        Returns:
            (score, reasons) - Score >= 30 indicates likely a name (lowered from 40)
        """
        # Common bank statement document labels that should NOT be matched as names
        negative_keywords = {
            'account statement', 'statement period', 'requested date',
            'account type', 'branch', 'account number', 'account name',
            'bank statement', 'statement', 'summary', 'details',
            'transaction', 'debit', 'credit', 'balance', 'amount',
            'date', 'description', 'reference', 'page', 'total',
            'opening balance', 'closing balance', 'current balance',
            'statement date', 'period', 'from', 'to', 'currency'
        }

        text = line.get('text', '').strip()
        text_lower = text.lower()

        # Negative filter: Skip common document labels
        if text_lower in negative_keywords:
            return (0, ['negative-keyword'])

        # Also check if line starts with common negative patterns
        for keyword in negative_keywords:
            if text_lower.startswith(keyword + ':') or text_lower.startswith(keyword + ' '):
                return (0, ['negative-keyword-prefix'])

        coords = self._get_line_coordinates(line)
        x1, y1, x2, y2 = coords
        y_center = (y1 + y2) / 2

        score = 0
        reasons = []

        # Must have at least 2 words
        words = text.split()
        if len(words) < 2:
            return (0, [])

        # Must not contain digits
        if any(c.isdigit() for c in text.replace('/', '').replace('-', '')):
            return (0, [])

        # Positional: Top half of document (reduced from upper-left bias)
        if y_center < 0.5:
            score += 10
            reasons.append("top-half")

        # Positional: Upper-left area (still gives a boost but not as heavy)
        if y_center < 0.2 and x1 < 0.5:
            score += 15
            reasons.append("upper-left")

        # Positional: Below bank logo/name (y: 0.1-0.2)
        if 0.1 < y_center < 0.2:
            score += 15
            reasons.append("below-header")

        # Pattern: Malaysian/Indian name patterns (S/O, D/O, A/L)
        if re.search(r'\s+(S/O|D/O|A/L|S/O\.|D/O\.|A/L\.)\s+', text, re.IGNORECASE):
            score += 40
            reasons.append("s/o-d/o-pattern")

        # Pattern: Western name format (2-3 words, title case)
        if 2 <= len(words) <= 3 and text == text.title() and not text.isupper():
            score += 25
            reasons.append("western-format")

        # Pattern: Common Asian name format (2-4 words, all caps)
        if len(words) <= 4 and text.isupper():
            score += 10
            reasons.append("caps-format")

        # Pattern: Company/Entity suffixes
        text_upper = text.upper()
        entity_suffixes = ['PTY LTD', 'PTY. LTD.', 'PTY. LTD', 'PT LTD', 'PTE LTD',
                           'LTD', 'LIMITED', 'INC', 'INCORPORATED', 'LLC', 'LLP',
                           'SDN BHD', 'SDN. BHD.', 'BERHAD', 'BHD', 'GMBH', 'AG',
                           'SA', 'SPA', 'SRL', 'BV', 'NV', 'PLC', 'PCL', 'CORP',
                           'CORPORATION', 'CO.', 'COMPANY']
        for suffix in entity_suffixes:
            if text_upper.endswith(suffix):
                score += 30
                reasons.append("entity-suffix")
                break

        # Text characteristics: No special chars except slash/dash/dot
        allowed_special = set(" /-.&'")
        if all(c.isalnum() or c in allowed_special or c.isspace() for c in text):
            score += 15
            reasons.append("clean-text")

        return (score, reasons)

    def _remove_name_titles(self, text: str) -> str:
        """
        Remove common titles from the beginning of a name.

        Handles titles like: Mr, Mrs, Ms, Miss, Dr, Prof, etc.
        """
        # Common titles to remove (with and without periods)
        titles = [
            'MR.', 'MRS.', 'MS.', 'MISS', 'DR.', 'PROF.', 'DR',
            'HON.', 'SIR.', 'MADAM', 'MA\'AM',
            # Asian titles
            'MDM', 'MR', 'MRS', 'MS', 'PROF', 'DATO', 'DATIN', 'DRS',
            # Indian titles
            'SHRI', 'SMT', 'KUM', 'KUMARI',
            # Other
            'HR.', 'HERR', 'FRAU', 'SIGNOR', 'SIGNORA', 'SR.', 'SRA.',
            'MR.', 'MRS.', 'MS.', 'DR.', 'PROF.'
        ]

        words = text.split()
        if not words:
            return text

        # Check if first word is a title (case-insensitive)
        first_word_upper = words[0].upper()
        if first_word_upper in titles:
            # Remove the title and return the rest
            return ' '.join(words[1:]).strip()

        return text

    def _looks_like_name(self, text: str) -> bool:
        """
        Quick check if text looks like a person's or entity's name.

        Returns True if text appears to be a name, False otherwise.
        """
        words = text.split()
        if len(words) < 2 or len(words) > 5:
            return False
        # No digits
        if any(c.isdigit() for c in text):
            return False
        # Not all labels or common words
        text_lower = text.lower()
        negative = {'statement', 'summary', 'total', 'balance', 'account', 'page',
                    'transaction', 'debit', 'credit', 'amount', 'date', 'description',
                    'bank', 'branch', 'currency', 'period', 'requested', 'type'}
        if any(word in negative for word in words):
            return False
        return True

    def _extract_name_by_label(self, doc_lines: List[Dict]) -> Optional[str]:
        """
        Extract account holder name by finding labels like 'Account Holder:'.

        Searches for name-related labels and returns the name text from the
        following line(s).

        Returns:
            Name text if found, None otherwise.
        """
        from app.core.logger import get_logger
        logger = get_logger()

        name_labels = {
            'account holder', 'account name', 'customer name',
            'name of account holder', 'account holder name',
            'customer', 'beneficiary name', 'remitter name',
            'acc holder', 'acc name', 'acct holder', 'acct name'
        }

        for i, line in enumerate(doc_lines):
            text = line.get('text', '').strip().lower()

            # Check if this line is a name label
            if any(label in text for label in name_labels):
                # Look at the next 1-2 lines for the actual name
                for j in range(i + 1, min(i + 3, len(doc_lines))):
                    next_text = doc_lines[j].get('text', '').strip()
                    if self._looks_like_name(next_text):
                        logger.info(f"Found name by label '{text}' at position {i}: '{next_text}'")
                        return next_text

        return None

    def _extract_name_position_independent(self, doc_lines: List[Dict]) -> Optional[str]:
        """
        Extract name by searching entire document for name-like text.

        Ignores position, focuses on text characteristics.
        Returns the highest-scoring candidate regardless of position.

        Returns:
            Best candidate name text if found, None otherwise.
        """
        from app.core.logger import get_logger
        logger = get_logger()

        candidates = []

        for line in doc_lines:
            text = line.get('text', '').strip()
            score, reasons = self._score_line_as_account_holder_name(line, doc_lines)

            # Skip if clearly not a name
            if score == 0:
                continue

            # Add to candidates regardless of position
            candidates.append((score, text, reasons))

        # Sort by score and return highest
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_score, best_text, best_reasons = candidates[0]
            logger.info(f"Position-independent name: '{best_text}' (score={best_score})")
            logger.info(f"  Reasons: {', '.join(best_reasons)}")
            return best_text

        return None

    def _extract_name_with_gliner(self, doc_lines: List[Dict]) -> Optional[Tuple[str, float]]:
        """
        Extract account holder name using GLiNER zero-shot NER.

        Uses ML-based entity recognition to identify the account holder name
        from the document text, which is more robust than heuristic scoring.

        Returns:
            Tuple of (name, confidence) if found, None otherwise.
        """
        try:
            from app.core.gliner_ner_model import get_gliner_ner_model

            # Build full text from document lines
            full_text = '\n'.join([
                line.get('text', '').strip()
                for line in doc_lines
                if line.get('text', '').strip()
            ])

            if not full_text or len(full_text) < 20:
                logger.debug("Document text too short for GLiNER extraction")
                return None

            # Get GLiNER model
            gliner_model = get_gliner_ner_model()

            # Extract entities
            entities = gliner_model.extract_bank_statement_entities(full_text)

            # Get account holder name result
            name_result = entities.get('account holder name')

            if name_result:
                name = name_result.get('value')
                confidence = name_result.get('confidence', 0.0)

                # Log if this was a reclassified entity
                if name_result.get('reclassified'):
                    logger.info(f"GLiNER reclassified entity as account holder name: '{name}' (conf={confidence:.2f})")

                # Validate the extracted name
                if name and self._looks_like_valid_name(name, confidence):
                    logger.info(f"GLiNER extracted name: '{name}' (confidence={confidence:.2f})")
                    return (name, confidence)
                elif name:
                    logger.warning(f"GLiNER extracted name but validation failed: '{name}' (confidence={confidence:.2f})")
            else:
                logger.info("GLiNER did not extract an account holder name")

        except ImportError:
            logger.warning("GLiNER not available, skipping ML-based extraction")
        except Exception as e:
            logger.warning(f"GLiNER extraction failed: {e}")

        return None

    def _looks_like_valid_name(self, text: str, gliner_confidence: float = 0.0) -> bool:
        """
        Validate that text looks like a valid account holder name.

        Combines ML confidence with rule-based validation to filter out
        banking terminology and invalid patterns.

        Args:
            text: Candidate name text
            gliner_confidence: GLiNER's confidence score (0-1)

        Returns:
            True if text appears to be a valid name, False otherwise.
        """
        # Trust GLiNER more - if confidence >= 0.7, accept it (changed from 0.9)
        if gliner_confidence >= 0.7:
            return True

        text_lower = text.lower()

        # Reject banking terminology and document metadata
        banking_terms = [
            'consolidated', 'statement', 'summary', 'balance',
            'account', 'deposit', 'withdrawal', 'transaction',
            'opening', 'closing', 'current', 'savings', 'debit',
            'credit', 'transfer', 'payment', 'charge', 'fee',
            'interest', 'tax', 'page', 'chapter', 'section',
            'period', 'requested', 'currency', 'branch', 'type',
            # Address components (should not be person names)
            'street', 'st', 'road', 'rd', 'lane', 'ln', 'drive', 'dr',
            'avenue', 'ave', 'place', 'pl', 'court', 'way', 'walk',
            'crescent', 'jalan', 'lorong', 'block', 'blk', 'flat', 'unit',
            'building', 'postal', 'code', 'zip'
        ]

        for term in banking_terms:
            if term in text_lower:
                logger.debug(f"Rejected name due to banking term '{term}': '{text}'")
                return False

        # Must have reasonable name characteristics
        words = text.split()
        if len(words) == 0:
            return False

        # At least 50% of words must contain letters
        words_with_letters = sum(1 for w in words if any(c.isalpha() for c in w))
        if words_with_letters / len(words) < 0.5:
            logger.debug(f"Rejected name due to low letter content: '{text}'")
            return False

        # Reject very short single words (likely OCR errors)
        if len(words) == 1 and len(text) < 3:
            return False

        # Reject all-caps single words longer than 10 chars (likely OCR garbage)
        if len(words) == 1 and text.isupper() and len(text) > 10:
            logger.debug(f"Rejected all-caps long text: '{text}'")
            return False

        return True

    def _looks_like_bank_abbreviation(self, text: str) -> bool:
        """
        Check if text looks like a bank abbreviation rather than a person name.
        """
        bank_abbrs = ['DBS', 'POSB', 'OCBC', 'UOB', 'CIMB', 'XDBS', 'HSBC', 'CITIBANK',
                      'STANDARD', 'CHARTERED', 'MAYBANK', 'RHB', 'LLOYDS', 'BARCLAYS']
        return text.upper() in bank_abbrs or len(text) <= 5

    def _looks_like_address_component(self, text: str) -> bool:
        """
        Check if text looks like an address component rather than a person name.
        Returns True if text contains street types, building indicators, etc.
        """
        if not text:
            return False

        text_lower = text.lower()
        address_terms = [
            'street', 'st', 'road', 'rd', 'lane', 'ln', 'drive', 'dr',
            'avenue', 'ave', 'place', 'pl', 'court', 'crescent',
            'jalan', 'lorong', 'block', 'blk', 'flat', 'unit',
            'building', 'postal', 'code', 'zip',
            # Indian address terms
            'marg', 'nagar', 'society', 'complex', 'apartment',
            'towers', 'tower', 'chambers', 'centre', 'center'
        ]
        return any(term in text_lower for term in address_terms)

    def _complete_address_from_nearby_lines(
        self,
        address: str,
        doc_lines: List[Dict],
        address_coords: Optional[Tuple]
    ) -> str:
        """
        Complete an address by finding and combining nearby address-related lines.

        Args:
            address: The initial address found by GLiNER
            doc_lines: All docTR OCR lines
            address_coords: Coordinates of the initial address (if available)

        Returns:
            Completed address with nearby components (unit, postal, country, etc.)
        """
        if not address:
            return address

        import re

        logger.info(f"Address completion: starting with '{address}'")
        # Get Y position of address if coordinates available
        address_y = None
        if address_coords:
            address_y = (address_coords[1] + address_coords[3]) / 2
            logger.info(f"Address completion: address_y={address_y:.3f}")

        # Patterns that suggest address components
        address_patterns = [
            r'^#\d+[-\s]\d+',  # Unit numbers like #11-25
            r'^\d{6,7}$',  # Postal codes like 440029
            r'[A-Z]{6,}\s\d{6}',  # Country + postal like SINGAPORE 440029
            r'^[A-Z]{2,}\s\d{5,}',  # Country/state + postal
        ]

        # Components to add (in order)
        additional_parts = []

        # Look for nearby lines that match address patterns
        for i, line in enumerate(doc_lines):
            text = line.get('text', '').strip()
            if not text or text in address:
                continue

            coords = self._get_line_coordinates(line)
            line_y = (coords[1] + coords[3]) / 2

            # Check if this line is close to the address (within 100 pixels)
            distance_threshold = 100  # pixels
            if address_y and abs(line_y - address_y) > distance_threshold:
                continue

            # Check if this line looks like an address component
            is_address_component = False

            # Check for unit numbers (#11-25)
            if re.match(r'^#\d+[-\s]\d+', text):
                is_address_component = True
                logger.debug(f"  Found unit number: '{text}'")
            # Check for postal codes
            elif re.match(r'^\d{6}$', text):
                is_address_component = True
                logger.debug(f"  Found postal code: '{text}'")
            # Check for country + postal patterns
            elif re.search(r'[A-Z]{5,}\s+\d{6}', text):
                is_address_component = True
                logger.debug(f"  Found country+postal: '{text}'")
            # Skip lines that are clearly not address components
            elif any(skip in text.lower() for skip in ['reduce', 'learn', 'account', 'balance', 'page', 'dbs', 'posb', 'consolidated', 'statement']):
                logger.debug(f"  Skipping non-address line: '{text[:20]}...'")
                continue

            if is_address_component:
                # Avoid duplicates
                if text not in address and text not in additional_parts:
                    additional_parts.append(text)

        # Combine all parts
        if additional_parts:
            logger.info(f"Address completion: adding {len(additional_parts)} parts: {additional_parts}")
            # Add parts that aren't already in the address
            for part in additional_parts:
                if part not in address:
                    # Add comma separator if needed
                    if address and not address.endswith(','):
                        address += ', '
                    address += part

        return address

    def _find_label_value_nearby(self, doc_lines: List[Dict], label_type: str) -> Optional[str]:
        """
        Find the actual value for a label by searching nearby lines.

        When GLiNER extracts a label like "Account No." instead of the value,
        this method searches the next few lines for the actual value.

        Args:
            doc_lines: List of line dicts from docTR
            label_type: Type of label ('account number', etc.)

        Returns:
            The found value or None
        """
        # Label patterns to search for
        label_patterns = {
            'account number': [
                'account no', 'account number', 'acct no', 'acct number',
                'account #', 'acct #', 'a/c no', 'a/c number',
                'acc no', 'acc number'
            ],
        }

        if label_type not in label_patterns:
            return None

        patterns = label_patterns[label_type]

        # Find the line containing the label
        for i, line in enumerate(doc_lines):
            text = line.get('text', '').strip().lower()

            # Check if this line contains a label pattern
            for pattern in patterns:
                if pattern in text:
                    # Search the next several lines for the actual value
                    # Increased range to handle table-like layouts where values may be further down
                    for j in range(i + 1, min(i + 10, len(doc_lines))):
                        next_text = doc_lines[j].get('text', '').strip()

                        # Skip common non-value patterns
                        if not next_text or len(next_text) < 3:
                            continue

                        # Skip header rows and common labels
                        next_text_lower = next_text.lower()
                        skip_patterns = ['balance', 'currency', 'summary', 'page', 'account no', 'account #']
                        if any(skip in next_text_lower for skip in skip_patterns):
                            continue

                        # For account numbers, look for numeric strings
                        if label_type == 'account number':
                            # Try to extract account number from this line
                            if self._is_valid_account_number(next_text):
                                return next_text

                            # Also try to find account number within the line
                            # (e.g., "151-15252-0" or account numbers with dashes)
                            import re
                            # Look for 6-20 character sequences with digits/dashes
                            match = re.search(r'\b[\d\-]{6,20}\b', next_text)
                            if match:
                                # Normalize the account number
                                acct_num = match.group(0).replace('-', '').replace(' ', '')
                                # Check if it has enough digits to be an account number
                                if 6 <= len(acct_num) <= 20 and acct_num.isdigit():
                                    return acct_num

                    # Also check the same line (e.g., "Account No: 123456")
                    import re
                    match = re.search(r'\b(\d[\d\s\-]{6,13}\d)\b', line.get('text', ''))
                    if match:
                        acct_num = match.group(1).replace('-', '').replace(' ', '')
                        if 8 <= len(acct_num) <= 15:
                            return acct_num

        return None

    def _get_line_coordinates(self, line: Dict) -> Tuple[float, float, float, float]:
        """
        Extract (x1, y1, x2, y2) coordinates from a line.

        Handles different geometry formats from docTR:
        - [[x1, y1], [x2, y2]] format
        - {'x1': x, 'y1': y, 'x2': x, 'y2': y} format
        """
        geometry = line.get('geometry')

        if isinstance(geometry, list) and len(geometry) >= 2:
            # Format: [[x1, y1], [x2, y2]]
            return (geometry[0][0], geometry[0][1], geometry[1][0], geometry[1][1])
        elif isinstance(geometry, dict):
            # Format: {'x1': x, 'y1': y, 'x2': x, 'y2': y}
            return (
                geometry.get('x1', 0),
                geometry.get('y1', 0),
                geometry.get('x2', 0),
                geometry.get('y2', 0)
            )

        return (0, 0, 0, 0)  # Fallback

    def _find_currency_iso_code(self, text: str) -> Optional[str]:
        """
        Find 3-letter ISO currency code in text.
        """
        # Pattern: 3 uppercase letters, word boundaries
        pattern = r'\b[A-Z]{3}\b'

        for match in re.finditer(pattern, text):
            currency = match.group(0)
            if currency in self.currency_codes:
                return currency

        return None

    def _find_date_in_text(self, text: str) -> Optional[Tuple[datetime, str]]:
        """
        Find and parse a date from text.

        Supports various date formats:
        - DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY
        - MM/DD/YYYY, MM-DD-YYYY, MM.DD.YYYY
        - YYYY/MM/DD, YYYY-MM-DD, YYYY.MM.DD
        - DD/MM/YY, DD-MM-YY (2-digit years)
        - Month DD, YYYY (e.g., "January 15, 2025")
        - DD Month YYYY (e.g., "15 January 2025")
        - Mon DD, YYYY (e.g., "Jan 15, 2025")

        Returns:
            Tuple of (datetime object, matched_date_string) if date found and valid, None otherwise
        """
        # Common date patterns (in order of priority)
        date_patterns = [
            # YYYY/MM/DD or YYYY-MM-DD or YYYY.MM.DD (ISO format) - try first as it's most specific
            (r'\b(\d{4})([/.-])(\d{1,2})\2(\d{1,2})\b', lambda m: self._parse_ymd(m.group(1), m.group(3), m.group(4))),
            # DD/MM/YYYY or DD-MM-YYYY or DD.MM.YYYY
            (r'\b(\d{1,2})([/.-])(\d{1,2})\2(\d{4})\b', lambda m: self._parse_dmy_or_mdy(m.group(1), m.group(3), m.group(4))),
            # DD/MM/YY or DD-MM-YY (2-digit year)
            (r'\b(\d{1,2})([/.-])(\d{1,2})\2(\d{2})\b', lambda m: self._parse_dmy_2digit_any(m.group(1), m.group(3), m.group(4))),
            # Month DD, YYYY (e.g., "January 15, 2025")
            (r'\b([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})\b', lambda m: self._parse_month_name(m.group(1), m.group(2), m.group(3))),
            # DD Month YYYY (e.g., "15 January 2025")
            (r'\b(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})\b', lambda m: self._parse_month_name_reverse(m.group(1), m.group(2), m.group(3))),
        ]

        for pattern, parser in date_patterns:
            matches = list(re.finditer(pattern, text))
            for match in matches:
                try:
                    date_obj = parser(match)
                    # Validate the date is reasonable (between 1900 and 2100)
                    if date_obj and datetime(1900, 1, 1) <= date_obj <= datetime(2100, 12, 31):
                        # Return both the datetime object and the matched date string
                        matched_date_str = match.group(0)
                        return (date_obj, matched_date_str)
                except (ValueError, AttributeError):
                    continue

        return None

    def _parse_dmy(self, day: str, month: str, year: str) -> Optional[datetime]:
        """Parse DD/MM/YYYY format."""
        try:
            d, m, y = int(day), int(month), int(year)
            # Validate: day should be > 31 for month, month should be > 12 for day
            # This helps distinguish DD/MM from MM/DD
            if d > 31 or m > 12:
                return None
            return datetime(y, m, d)
        except ValueError:
            return None

    def _parse_mdy(self, month: str, day: str, year: str) -> Optional[datetime]:
        """Parse MM/DD/YYYY format (US format)."""
        try:
            m, d, y = int(month), int(day), int(year)
            if m > 12 or d > 31:
                return None
            return datetime(y, m, d)
        except ValueError:
            return None

    def _parse_ymd(self, year: str, month: str, day: str) -> Optional[datetime]:
        """Parse YYYY/MM/DD format (ISO format)."""
        try:
            y, m, d = int(year), int(month), int(day)
            if m > 12 or d > 31:
                return None
            return datetime(y, m, d)
        except ValueError:
            return None

    def _parse_dmy_2digit(self, day: str, month: str, year: str) -> Optional[datetime]:
        """Parse DD/MM/YY format with 2-digit year."""
        try:
            d, m, y = int(day), int(month), int(year)
            if d > 31 or m > 12:
                return None
            # Convert 2-digit year to 4-digit (assume 2000s for 00-29, 1900s for 30-99)
            y = 2000 + y if y < 30 else 1900 + y
            return datetime(y, m, d)
        except ValueError:
            return None

    def _parse_month_name(self, month: str, day: str, year: str) -> Optional[datetime]:
        """Parse Month DD, YYYY format (e.g., 'January 15, 2025')."""
        month_names = {
            'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
            'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6,
            'july': 7, 'jul': 7, 'august': 8, 'aug': 8, 'september': 9, 'sep': 9, 'sept': 9,
            'october': 10, 'oct': 10, 'november': 11, 'nov': 11, 'december': 12, 'dec': 12
        }
        try:
            month_lower = month.lower()
            if month_lower not in month_names:
                return None
            m = month_names[month_lower]
            d, y = int(day), int(year)
            if d > 31:
                return None
            return datetime(y, m, d)
        except ValueError:
            return None

    def _parse_month_name_reverse(self, day: str, month: str, year: str) -> Optional[datetime]:
        """Parse DD Month YYYY format (e.g., '15 January 2025')."""
        # Reuse the month name parser
        return self._parse_month_name(month, day, year)

    def _parse_dmy_or_mdy(self, part1: str, part2: str, year: str) -> Optional[datetime]:
        """
        Parse date that could be either DD/MM/YYYY or MM/DD/YYYY format.

        Tries DD/MM/YYYY first (common outside US), then MM/DD/YYYY (US format).
        """
        p1, p2, y = int(part1), int(part2), int(year)

        # Try DD/MM/YYYY first (day first)
        if p1 <= 31 and p2 <= 12:
            try:
                return datetime(y, p2, p1)  # day, month, year -> year, month, day
            except ValueError:
                pass

        # Try MM/DD/YYYY (month first)
        if p1 <= 12 and p2 <= 31:
            try:
                return datetime(y, p1, p2)  # month, day, year -> year, month, day
            except ValueError:
                pass

        return None

    def _parse_dmy_2digit_any(self, part1: str, part2: str, year: str) -> Optional[datetime]:
        """
        Parse date with 2-digit year that could be either DD/MM/YY or MM/DD/YY format.
        """
        p1, p2, y = int(part1), int(part2), int(year)

        # Convert 2-digit year to 4-digit (assume 2000s for 00-29, 1900s for 30-99)
        y = 2000 + y if y < 30 else 1900 + y

        # Try DD/MM/YY first (day first)
        if p1 <= 31 and p2 <= 12:
            try:
                return datetime(y, p2, p1)
            except ValueError:
                pass

        # Try MM/DD/YY (month first)
        if p1 <= 12 and p2 <= 31:
            try:
                return datetime(y, p1, p2)
            except ValueError:
                pass

        return None

    def _calculate_confidence(self, found_fields: Dict) -> float:
        """
        Calculate confidence score based on how many fields were found.
        """
        found_count = sum(1 for v in found_fields.values() if v is not None)
        total_count = len(found_fields)

        # Base confidence: percentage of fields found
        base_confidence = found_count / total_count

        # Bonus for having critical fields
        critical_fields = ['account_holder_name', 'account_number', 'bank_name']
        critical_found = sum(1 for f in critical_fields if found_fields.get(f) is not None)
        bonus = (critical_found / len(critical_fields)) * 0.2

        # Total confidence (capped at 1.0)
        confidence = min(base_confidence + bonus, 1.0)

        return confidence

    def _detect_country_in_text(self, text: str) -> Optional[str]:
        """
        Detect country names in text.

        Returns country name if found, None otherwise.
        Checks more specific names first (e.g., "United Arab Emirates" before "UAE").
        """
        text_lower = text.lower()

        # Country patterns (most specific first)
        countries = [
            'united arab emirates',
            'saudi arabia',
            'south korea',
            'new zealand',
            'sri lanka',
            'south africa',
            'united kingdom',
            'philippines',
            'indonesia',
            'malaysia',
            'myanmar',
            'burma',
            'thailand',
            'singapore',
            'vietnam',
            'india',
            'taiwan',
            'china',
            'japan',
            'australia',
            'hong kong',
            'macau',
            # Abbreviations
            'uae',
            'uk',
            'usa',
        ]

        # Check for multi-word country names first (more specific)
        for country in countries:
            if ' ' in country:  # Multi-word country
                if country in text_lower:
                    return country.title()

        # Check for single-word country names or abbreviations
        words = text_lower.split()
        for word in words:
            # Remove punctuation
            word = word.strip('.,;:')
            if word in countries:
                return word.title()

        return None

    def _find_nearest_country(self, address_coords: Tuple, detected_countries: Dict[str, Tuple]) -> Optional[str]:
        """
        Find the nearest detected country to an address based on Y-coordinate proximity.

        Args:
            address_coords: Tuple of (x1, y1, x2, y2) for the address
            detected_countries: Dict of {country_name: (x1, y1, x2, y2)}

        Returns:
            Country name closest to the address, or None if no countries detected
        """
        if not detected_countries:
            return None

        address_y = (address_coords[1] + address_coords[3]) / 2  # Center Y

        nearest_country = None
        min_distance = float('inf')

        for country, country_coords in detected_countries.items():
            country_y = (country_coords[1] + country_coords[3]) / 2
            distance = abs(address_y - country_y)

            if distance < min_distance:
                min_distance = distance
                nearest_country = country

        return nearest_country

    def _extract_account_number(self, text: str) -> Optional[str]:
        """
        Extract purely numeric account number with optional dashes and spaces.
        Returns normalized account number (digits only) or None.
        """
        # Remove all spaces, dashes, and other separators
        cleaned = text.replace('-', '').replace(' ', '')

        # Must be 8-15 digits
        if cleaned.isdigit() and 8 <= len(cleaned) <= 15:
            return cleaned

        return None

    def _is_valid_account_number(self, text: str) -> bool:
        """
        Validate that text looks like an account number, not a label.

        Rejects:
        - Label text like "Account No.", "Account Number"
        - Text that's too short (< 6 chars)
        - Text with no digits

        Accepts:
        - Numeric strings with 8-15 digits
        - Account numbers with dashes/spaces (normalized)
        """
        # Common label patterns to reject
        label_patterns = ['account no', 'account number', 'acct no', 'acct number',
                         'account #', 'acct #', 'number']

        text_lower = text.lower().strip()
        if any(pattern in text_lower for pattern in label_patterns):
            return False

        # Must contain digits
        if not any(c.isdigit() for c in text):
            return False

        # Normalize (remove spaces, dashes)
        normalized = text.replace('-', '').replace(' ', '')

        # Check length (8-15 digits typical for account numbers)
        if normalized.isdigit() and 8 <= len(normalized) <= 15:
            return True

        # Allow some OCR noise - at least 6 digits
        digit_count = sum(c.isdigit() for c in normalized)
        if digit_count >= 6:
            return True

        return False

    def _extract_addresses_with_postal(self, lines: List[Dict], detected_countries: Dict[str, Tuple] = None) -> List[Dict]:
        """
        Extract addresses using regex-based detection.

        Args:
            lines: List of text line dictionaries
            detected_countries: Dict of {country_name: (x1, y1, x2, y2)} for country hints

        Returns:
            List of address dicts with keys:
            - 'text': concatenated address lines
            - 'coords': (x1, y1, x2, y2) of bounding box
        """
        from app.core.logger import get_logger
        logger = get_logger()

        if detected_countries is None:
            detected_countries = {}

        # Use regex-based address detection
        address_candidates = []
        for line in lines:
            text = line.get('text', '').strip()
            coords = self._get_line_coordinates(line)

            # Skip very short lines
            if len(text) < 5:
                continue

            # Skip lines that are just numbers or dates
            if text.replace('-', '').replace('/', '').replace('.', '').isdigit():
                continue

            # Find nearest country for this address
            nearest_country = self._find_nearest_country(coords, detected_countries)

            # Use regex-based address detection with country hint
            is_address = self._is_address_line_regex(text, country_hint=nearest_country)

            if is_address:
                address_candidates.append({
                    'text': text,
                    'coords': coords,
                    'line': line,
                    'country': nearest_country
                })

        # 2. Cluster address lines by spatial proximity
        if not address_candidates:
            return []

        # Calculate page dimensions from all input lines for threshold scaling
        # Find max x2 and y2 across all lines to get page dimensions
        max_x = 0
        max_y = 0
        for line in lines:
            coords = self._get_line_coordinates(line)
            max_x = max(max_x, coords[2])  # x2
            max_y = max(max_y, coords[3])  # y2

        logger.info(f"Page dimensions: {max_x}x{max_y} pixels")

        # Calculate pixel-based thresholds from page dimensions
        x_threshold = max_x * 0.15  # 15% of page width in pixels
        y_threshold = max_y * 0.20  # 20% of page height in pixels

        logger.info(f"Clustering thresholds: x<{x_threshold:.1f}px, y<{y_threshold:.1f}px (both required)")

        # Sort by y-position before clustering to ensure multi-line addresses are processed top-to-bottom
        address_candidates.sort(key=lambda c: c['coords'][1])  # Sort by y1
        logger.info(f"Sorted {len(address_candidates)} address candidates by y-position")

        # DEBUG: Log all candidates with pixel coordinates
        for idx, cand in enumerate(address_candidates):
            x1, y1, x2, y2 = cand['coords']
            logger.info(f"  Candidate {idx}: '{cand['text']}' at x1={x1:.0f}, y1={y1:.0f}, x2={x2:.0f}, y2={y2:.0f}")

        # Group by similar x1 (left alignment) and y proximity
        address_clusters = []
        used = [False] * len(address_candidates)

        for i, candidate in enumerate(address_candidates):
            if used[i]:
                continue

            cluster = [candidate]
            used[i] = True
            x1, y1, x2, y2 = candidate['coords']

            # Find nearby lines with similar x1 (relaxed clustering)
            for j in range(i + 1, len(address_candidates)):
                if used[j]:
                    continue

                other_coords = address_candidates[j]['coords']
                other_x1 = other_coords[0]
                other_y1 = other_coords[1]

                x1_diff = abs(x1 - other_x1)
                y_diff = abs(y2 - other_coords[1])

                # Clustering thresholds using pixel values:
                # - y proximity: y_threshold (20% of page height)
                # - x1 similarity: x_threshold (15% of page width)
                # Both conditions must be true for clustering
                y_close = y_diff < y_threshold
                x_aligned = x1_diff < x_threshold

                # DEBUG: Log clustering decisions
                if y_close and x_aligned:
                    logger.info(f"  Clustering candidate {j} with {i}: x1_diff={x1_diff:.1f}px, y_diff={y_diff:.1f}px ✓")
                    cluster.append(address_candidates[j])
                    used[j] = True
                    # Update y2 for next comparison
                    y2 = max(y2, other_coords[3])
                else:
                    logger.info(f"  Skipping candidate {j}: x1_diff={x1_diff:.1f}px (need <{x_threshold:.1f}), y_diff={y_diff:.1f}px (need <{y_threshold:.1f})")

            if cluster:
                # Concatenate cluster lines
                cluster_text = ' '.join(c['text'] for c in cluster)

                # Validate cluster before adding
                if not self._is_valid_address_cluster(cluster_text):
                    logger.info(f"  Skipping invalid cluster: '{cluster_text[:100]}...'")
                    continue

                # Calculate bounding box
                all_coords = [c['coords'] for c in cluster]
                min_x = min(c[0] for c in all_coords)
                min_y = min(c[1] for c in all_coords)
                max_x = max(c[2] for c in all_coords)
                max_y = max(c[3] for c in all_coords)

                address_clusters.append({
                    'text': cluster_text,
                    'coords': (min_x, min_y, max_x, max_y)
                })

        logger.info(f"Clustered into {len(address_clusters)} address blocks")
        return address_clusters

    def _is_valid_address_cluster(self, cluster_text: str) -> bool:
        """
        Check if a cluster looks like a valid address.

        Filters out clusters that contain non-address content.
        """
        from .address_patterns import meets_minimum_quality, is_negative_keyword

        # Minimum quality checks
        if not meets_minimum_quality(cluster_text):
            return False

        # Negative pattern checks
        if is_negative_keyword(cluster_text):
            return False

        return True

    def _combine_nearby_addresses(self, addresses: List[Dict], y_threshold: float = 0.04, x_threshold: float = 0.1) -> List[Dict]:
        """
        Combine address entities that are spatially close (multi-line addresses).

        GLiNER may extract "THANABALAN" and "BLK 29" as separate addresses,
        but they're part of the same multi-line address block.

        Args:
            addresses: List of address dicts with 'text' and 'coords' keys
            y_threshold: Maximum Y distance to consider addresses as part of same block
            x_threshold: Maximum X distance to consider addresses as part of same block
                        (addresses far apart horizontally are likely separate addresses)

        Returns:
            List of combined address dicts
        """
        if not addresses:
            return []

        # Sort by Y position
        sorted_addrs = sorted(addresses, key=lambda a: a['coords'][1] if a['coords'] else float('inf'))

        logger.debug(f"  Address combining: y_threshold={y_threshold}, x_threshold={x_threshold}")
        for i, addr in enumerate(sorted_addrs):
            coords = addr.get('coords')
            if coords:
                logger.debug(f"    Address {i}: '{addr['text'][:30]}...' at coords={coords}")
            else:
                logger.debug(f"    Address {i}: '{addr['text'][:30]}...' at coords=None")

        combined = []
        current_group = [sorted_addrs[0]]

        for i in range(1, len(sorted_addrs)):
            prev = current_group[-1]
            curr = sorted_addrs[i]

            # Skip if no coordinates
            if not curr['coords'] or not prev['coords']:
                # No coords - start new group
                logger.debug(f"    No coords for '{curr['text'][:20]}...' or '{prev['text'][:20]}...', starting new group")
                combined.append(self._merge_address_group(current_group))
                current_group = [curr]
                continue

            # Check Y distance (vertical gap)
            y_diff = abs(curr['coords'][1] - prev['coords'][3])  # Current y1 vs previous y2

            # Check X distance (horizontal alignment)
            # Multi-line addresses typically have similar X positions (left-aligned)
            prev_center_x = (prev['coords'][0] + prev['coords'][2]) / 2
            curr_center_x = (curr['coords'][0] + curr['coords'][2]) / 2
            x_diff = abs(curr_center_x - prev_center_x)

            logger.debug(f"    Comparing '{prev['text'][:20]}...' vs '{curr['text'][:20]}...': y_diff={y_diff:.4f} (thresh={y_threshold}), x_diff={x_diff:.4f} (thresh={x_threshold})")

            if y_diff < y_threshold and x_diff < x_threshold:
                # Close enough in both dimensions - same address block
                logger.debug(f"      → Merging (both y and x within threshold)")
                current_group.append(curr)
            else:
                # Too far in Y or X - start new group
                logger.debug(f"      → Starting new group (y_diff {'<' if y_diff < y_threshold else '>='} y_threshold, x_diff {'<' if x_diff < x_threshold else '>='} x_threshold)")
                combined.append(self._merge_address_group(current_group))
                current_group = [curr]

        # Don't forget the last group
        if current_group:
            combined.append(self._merge_address_group(current_group))

        return combined

    def _merge_address_group(self, group: List[Dict]) -> Dict:
        """Merge a group of address lines into a single address."""
        # Sort by Y position and concatenate
        sorted_group = sorted(group, key=lambda a: a['coords'][1] if a['coords'] else 0)
        combined_text = ' '.join(a['text'] for a in sorted_group)

        logger.debug(f"      Merging {len(group)} address(es): '{combined_text[:80]}'")

        # Calculate bounding box
        coords_list = [a['coords'] for a in sorted_group if a['coords']]
        if coords_list:
            min_x = min(c[0] for c in coords_list)
            min_y = min(c[1] for c in coords_list)
            max_x = max(c[2] for c in coords_list)
            max_y = max(c[3] for c in coords_list)
            combined_coords = (min_x, min_y, max_x, max_y)
        else:
            combined_coords = None

        return {'text': combined_text, 'coords': combined_coords}

    def _classify_addresses_by_proximity(
        self,
        addresses: List[Dict],
        account_holder_coords: Optional[Tuple],
        bank_name_coords: Optional[Tuple],
        country_code: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Classify addresses as account holder or bank based on proximity and content.

        Args:
            addresses: List of address dicts with 'text' and 'coords' keys
            account_holder_coords: Coordinates of account holder name
            bank_name_coords: Coordinates of bank name
            country_code: Detected country code (e.g., 'TH', 'SG')

        Returns:
            Dict with 'account_holder_address' and/or 'bank_address' keys
        """
        logger.info(f"    _classify_addresses_by_proximity called with {len(addresses)} addresses, country={country_code}")
        for i, addr in enumerate(addresses):
            logger.info(f"      Address {i}: text='{addr.get('text', '')[:50]}...', coords={addr.get('coords')}")

        from .address_validator import looks_like_bank_address, _contains_non_latin_script

        if len(addresses) == 1:
            # Only one address - check if it looks like a bank address
            addr_text = addresses[0]['text']
            if looks_like_bank_address(addr_text):
                logger.info(f"  Single address looks like bank address, marking as bank_address instead")
                return {'bank_address': addr_text}
            return {'account_holder_address': addr_text}

        if len(addresses) >= 2:
            # Calculate proximity to account holder name and bank name
            distances = {}

            for addr in addresses:
                addr_coords = addr['coords']
                # Skip addresses without coordinates
                if not addr_coords:
                    continue
                addr_center = (
                    (addr_coords[0] + addr_coords[2]) / 2,
                    (addr_coords[1] + addr_coords[3]) / 2
                )

                # Distance to account holder name
                if account_holder_coords:
                    account_center = (
                        (account_holder_coords[0] + account_holder_coords[2]) / 2,
                        (account_holder_coords[1] + account_holder_coords[3]) / 2
                    )
                    account_dist = (
                        (addr_center[0] - account_center[0])**2 +
                        (addr_center[1] - account_center[1])**2
                    )**0.5
                else:
                    account_dist = float('inf')

                # Distance to bank name
                if bank_name_coords:
                    bank_center = (
                        (bank_name_coords[0] + bank_name_coords[2]) / 2,
                        (bank_name_coords[1] + bank_name_coords[3]) / 2
                    )
                    bank_dist = (
                        (addr_center[0] - bank_center[0])**2 +
                        (addr_center[1] - bank_center[1])**2
                    )**0.5
                else:
                    bank_dist = float('inf')

                distances[addr['text']] = (account_dist, bank_dist)
                logger.debug(f"    Address '{addr['text'][:30]}...': account_dist={account_dist:.2f}, bank_dist={bank_dist:.2f}")

            # Classify each address by comparing distances AND content
            account_holder_addresses = []
            bank_addresses = []

            # Special handling for mixed addresses (non-Latin + English)
            # In SE Asian countries, non-Latin/garbled addresses are more likely to be user addresses
            from .address_validator import _is_garbled_ocr
            has_non_latin = any(_contains_non_latin_script(addr) or _is_garbled_ocr(addr) for addr in distances.keys())
            has_english = any(not (_contains_non_latin_script(addr) or _is_garbled_ocr(addr)) for addr in distances.keys())

            logger.info(f"  Address classification: {len(distances)} addresses with coords, has_non_latin={has_non_latin}, has_english={has_english}, country={country_code}")

            for addr_text, (account_dist, bank_dist) in distances.items():
                # First check if address looks like a bank address (strong signal)
                if looks_like_bank_address(addr_text):
                    bank_addresses.append(addr_text)
                    logger.debug(f"      '{addr_text[:30]}...' → bank (looks like bank address)")
                    continue

                is_non_latin = _contains_non_latin_script(addr_text) or _is_garbled_ocr(addr_text)

                # Special case: For mixed non-Latin + English addresses in SE Asian countries
                if has_non_latin and has_english and country_code in ['TH', 'MY', 'VN', 'ID', 'MM', 'KH', 'LA']:
                    if is_non_latin:
                        # Non-Latin/garbled addresses are more likely to be user addresses
                        account_holder_addresses.append(addr_text)
                        logger.debug(f"      '{addr_text[:30]}...' → account holder (non-Latin/garbled in {country_code})")
                        continue
                    # For English addresses, fall through to spatial distance classification below
                    # Don't assume English addresses with postal codes are bank addresses - use spatial proximity instead

                # Default: Assign to account holder if closer to account holder name
                # Assign to bank if closer to bank name
                if account_dist < bank_dist:
                    account_holder_addresses.append(addr_text)
                    logger.debug(f"      '{addr_text[:30]}...' → account holder (closer to account holder name: {account_dist:.2f} < {bank_dist:.2f})")
                else:
                    bank_addresses.append(addr_text)
                    logger.debug(f"      '{addr_text[:30]}...' → bank (closer to bank name: {bank_dist:.2f} <= {account_dist:.2f})")

            result = {}

            if account_holder_addresses:
                result['account_holder_address'] = ' '.join(account_holder_addresses)

            if bank_addresses:
                result['bank_address'] = ' | '.join(bank_addresses)

            logger.info(f"  Address classification: {len(account_holder_addresses)} account holder, {len(bank_addresses)} bank")
            return result

        return {}

    def _is_address_line_regex(self, text: str, country_hint: str = None) -> bool:
        """
        Fallback address detection using country-specific patterns.

        Uses the address_patterns module which provides:
        - Country-specific address patterns (UAE, Thailand, Singapore, UK, Malaysia, India, Myanmar)
        - Bank name filtering using exact matching
        - Minimum quality filters (length, word count)
        - Negative keyword filtering

        Args:
            text: Text to check
            country_hint: Country to prioritize patterns for (e.g., 'Singapore', 'Thailand')
        """
        from .address_patterns import is_valid_address_text
        from .global_banks import get_bank_info

        return is_valid_address_text(text, get_bank_info_func=get_bank_info, country_hint=country_hint)

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime object."""
        if not date_str:
            return None
        date_result = self._find_date_in_text(date_str)
        if date_result:
            return date_result[0]  # Return just the datetime object
        return None

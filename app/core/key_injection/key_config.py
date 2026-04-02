"""
Universal key definitions for document analysis.
This module defines the key sets required for each document type,
regardless of layout or format variations.
"""

from enum import Enum
from typing import Dict, List, Pattern
import re
from dataclasses import dataclass


class DocumentType(Enum):
    """Supported document types for key injection."""
    PASSPORT = "passport"
    ID_CARD = "id_card"
    BANK_STATEMENT = "bank_statement"
    TAX_STATEMENT = "tax_statement"


@dataclass
class KeyPattern:
    """Defines a pattern for recognizing a key field."""
    pattern: Pattern[str]
    confidence_weight: float = 1.0
    description: str = ""
    value_format: Pattern[str] = None


@dataclass
class DocumentKeySet:
    """Defines the set of keys required for a document type."""
    document_type: DocumentType
    required_keys: Dict[str, KeyPattern]
    optional_keys: Dict[str, KeyPattern] = None
    description: str = ""


class KeyConfig:
    """Central configuration for all document key sets."""

    def __init__(self):
        self._key_sets = self._initialize_key_sets()

    def _initialize_key_sets(self) -> Dict[DocumentType, DocumentKeySet]:
        """Initialize all document key sets with their patterns."""

        # Universal passport key patterns (multi-language support)
        passport_patterns = {
            "passport_number": KeyPattern(
                pattern=re.compile(r'^(?:passport\s*(?:no|number|#)?\s*[:\s]?\s*)?([a-z]{1,2}\d{6,9})$', re.IGNORECASE),
                confidence_weight=1.0,
                description="Passport number with country prefix"
            ),
            "surname": KeyPattern(
                pattern=re.compile(r'^(?:surname|last\s*name|family\s*name|apellidos?)\s*[:\s]?\s*([a-z\s\-\'\.]+)$', re.IGNORECASE),
                confidence_weight=0.9,
                description="Last name or surname"
            ),
            "given_names": KeyPattern(
                pattern=re.compile(r'^(?:given\s*names?|first\s*names?|forename|nombre|prénom)s?\s*[:\s]?\s*([a-z\s\-\'\.]+)$', re.IGNORECASE),
                confidence_weight=0.9,
                description="First and middle names"
            ),
            "date_of_birth": KeyPattern(
                pattern=re.compile(r'^(?:date\s*of\s*birth|birth\s*date|born|nacimiento|né\s*le)s?\s*[:\s]?\s*(\d{1,2}[\\/-]\d{1,2}[\\/-]\d{2,4})$', re.IGNORECASE),
                confidence_weight=0.95,
                value_format=re.compile(r'\d{1,2}[\\/-]\d{1,2}[\\/-]\d{2,4}'),
                description="Date of birth"
            ),
            "place_of_birth": KeyPattern(
                pattern=re.compile(r'^(?:place\s*of\s*birth|birth\s*place|lieu\s*de\s*naissance)s?\s*[:\s]?\s*([a-z\s\-\,\']+$)', re.IGNORECASE),
                confidence_weight=0.8,
                description="Place of birth"
            ),
            "nationality": KeyPattern(
                pattern=re.compile(r'^(?:nationality|nationalité|nacionalidad)s?\s*[:\s]?\s*([a-z\s]+)$', re.IGNORECASE),
                confidence_weight=0.8,
                description="Nationality"
            ),
            "sex": KeyPattern(
                pattern=re.compile(r'^(?:sex|gender|sexo|sexe)s?\s*[:\s]?\s*([mfmx])$', re.IGNORECASE),
                confidence_weight=0.7,
                value_format=re.compile(r'^[mfmxf]$'),
                description="Gender/Sex"
            ),
            "date_of_expiry": KeyPattern(
                pattern=re.compile(r'^(?:date\s*of\s*expiry|expir(?:y|ation)\s*date|valid\s*until|valable\s*jusqu\'au)s?\s*[:\s]?\s*(\d{1,2}[\\/-]\d{1,2}[\\/-]\d{2,4})$', re.IGNORECASE),
                confidence_weight=0.95,
                value_format=re.compile(r'\d{1,2}[\\/-]\d{1,2}[\\/-]\d{2,4}'),
                description="Passport expiry date"
            ),
            "authority": KeyPattern(
                pattern=re.compile(r'^(?:authority|issuing\s*authority|autorité|autoridad)s?\s*[:\s]?\s*([a-z\s\-\.\'0-9]+)$', re.IGNORECASE),
                confidence_weight=0.7,
                description="Issuing authority"
            )
        }

        # Universal bank statement key patterns
        bank_statement_patterns = {
            "account_number": KeyPattern(
                pattern=re.compile(r'^(?:account\s*(?:no|number|#)?\s*[:\s]?\s*)?(\d{8,20})$', re.IGNORECASE),
                confidence_weight=1.0,
                value_format=re.compile(r'\d{8,20}'),
                description="Bank account number"
            ),
            "account_holder": KeyPattern(
                # Traditional pattern with explicit labels for account holder names
                pattern=re.compile(r'^(?:account\s*holder|account\s*name|customer\s*name|name)\s*[:\s]?\s*([a-z\s\-\'\.]+)$', re.IGNORECASE),
                confidence_weight=0.9,
                description="Account holder name with explicit label"
            ),
            "account_name": KeyPattern(
                # Unified pattern for both personal and business account names
                # Handles personal names, cultural formats (S/O, D/O, B/O), and company names
                # Excludes banking terminology and invalid patterns
                pattern=re.compile(r'^(?!.*(?:STATEMENT|BALANCE|DEPOSIT|WITHDRAWAL|TRANSACTION|SUMMARY|TOTAL|BANK|CONSOLIDATED|CURRENT|SAVINGS|CURRENCY|EQUIVALENT|BREAKDOWN|CLOSING|OPENING|DEBIT|CREDIT|TRANSFER|PAYMENT|CHARGE|FEE|INTEREST|TAX|PAGE|CHAPTER|SECTION))(?!.*-{2,})(?!.*^\d+\s+)([A-Z][A-Za-z\s&\.\-\'\/]+(?:\s+(?:S/O|D/O|B/O)\s+[A-Z][a-zA-Z\'\-]+|[A-Z][a-zA-Z\'\-]+|(?:PTE\.?\s*LTD\.?|LTD\.?|INC\.?|CORP\.?|SDN\.?\s*BHD\.?|PRIVATE\s*LIMITED|LIMITED|LLC|LP|ENTERPRISE|TRADING|SOLUTIONS|TECHNOLOGIES|CONSULTANCY|SERVICES|HOLDINGS|GROUP|INTERNATIONAL|GLOBAL|ASIA|PACIFIC|SINGAPORE)){1,4})$', re.IGNORECASE),
                confidence_weight=1.0,
                description="Unified account name detection (personal names, cultural formats, company names)"
            ),
            "bank_name": KeyPattern(
                pattern=re.compile(r'^(?:bank\s*name|institution)\s*[:\s]?\s*([a-z\s\-\&\.\']+$)', re.IGNORECASE),
                confidence_weight=0.8,
                description="Bank or financial institution name"
            ),
            "bank_name_singapore": KeyPattern(
                # Singapore bank names (POSB, DBS, OCBC, UOB, etc.) - self-contained
                pattern=re.compile(r'^(POSB|DBS|OCBC|UOB|CITIBANK|HSBC|STANDARD\s*CHARTERED|MAYBANK|CIMB|RHBCANK|XDBS)$', re.IGNORECASE),
                confidence_weight=0.95,
                description="Singapore bank name"
            ),
            "statement_period": KeyPattern(
                pattern=re.compile(r'^(?:statement\s*period|period|for\s*period|as\s*at)\s*[:\s]?\s*([a-z0-9\s\-\,]+)$', re.IGNORECASE),
                confidence_weight=0.8,
                description="Statement coverage period"
            ),
            "statement_date": KeyPattern(
                # DD MMM YYYY format with Singapore banking context - single date extraction
                pattern=re.compile(r'^(?:statement\s*date|as\s*at|date|as\s*of)\s*[:\s]?\s*(\d{1,2}\s+(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+\d{4})$', re.IGNORECASE),
                confidence_weight=0.95,
                description="Statement date in DD MMM YYYY format (latest date from ranges)"
            ),
            "account_summary": KeyPattern(
                # Pattern for "Account Summary as at" statements
                pattern=re.compile(r'^(?:account\s*summary|statement|consolidated\s*statement)\s*(?:as\s*at|for|period)?\s*[:\s]?\s*([a-z0-9\s\-\,\.]+)$', re.IGNORECASE),
                confidence_weight=0.8,
                description="Account summary information"
            ),
            "opening_balance": KeyPattern(
                pattern=re.compile(r'^(?:opening\s*balance|beginning\s*balance|brought\s*forward)\s*[:\s]?\s*([\$\£\€]?\s*[\d,]+\.\d{2})$', re.IGNORECASE),
                confidence_weight=0.85,
                value_format=re.compile(r'[\$\£\€]?\s*[\d,]+\.\d{2}'),
                description="Account opening balance"
            ),
            "closing_balance": KeyPattern(
                pattern=re.compile(r'^(?:closing\s*balance|ending\s*balance|balance\s*carry\s*forward|final\s*balance)\s*[:\s]?\s*([\$\£\€]?\s*[\d,]+\.\d{2})$', re.IGNORECASE),
                confidence_weight=0.9,
                value_format=re.compile(r'[\$\£\€]?\s*[\d,]+\.\d{2}'),
                description="Account closing balance"
            ),
            "address": KeyPattern(
                # Pattern for Singapore addresses (enhanced for multi-line support)
                pattern=re.compile(r'^(?:blk|block)\s*\d+.*$', re.IGNORECASE),
                confidence_weight=0.7,
                description="Address information"
            ),
            "address_block": KeyPattern(
                # Pattern for complete address blocks
                pattern=re.compile(r'^(BLK\s*\d+.*\s*#\d+-\d+.*\s*SINGAPORE\s*\d{6})$', re.IGNORECASE),
                confidence_weight=0.9,
                description="Complete Singapore address block"
            ),
            "address_unit": KeyPattern(
                # Pattern for unit numbers
                pattern=re.compile(r'^#\d+-\d+$', re.IGNORECASE),
                confidence_weight=0.6,
                description="Address unit number"
            ),
            "address_postal": KeyPattern(
                # Pattern for Singapore postal codes
                pattern=re.compile(r'^SINGAPORE\s+\d{6}$', re.IGNORECASE),
                confidence_weight=0.8,
                description="Singapore postal code"
            ),
            "serial_number": KeyPattern(
                # Pattern for statement serial numbers (self-contained)
                pattern=re.compile(r'^(SN:\s*[A-Z0-9]+)$', re.IGNORECASE),
                confidence_weight=0.8,
                description="Statement serial number"
            )
        }

        # Universal tax statement key patterns
        tax_statement_patterns = {
            "taxpayer_id": KeyPattern(
                pattern=re.compile(r'^(?:taxpayer\s*(?:id|identification)|social\s*security\s*no|ssn|tax\s*id)\s*[:\s]?\s*(\d{6,15})$', re.IGNORECASE),
                confidence_weight=1.0,
                value_format=re.compile(r'\d{6,15}'),
                description="Taxpayer identification number"
            ),
            "tax_year": KeyPattern(
                pattern=re.compile(r'^(?:tax\s*year|year\s*of\s*assessment|fiscal\s*year)\s*[:\s]?\s*(\d{4})$', re.IGNORECASE),
                confidence_weight=0.9,
                value_format=re.compile(r'\d{4}'),
                description="Tax year"
            ),
            "gross_income": KeyPattern(
                pattern=re.compile(r'^(?:gross\s*income|total\s*income|income\s*before\s*tax)\s*[:\s]?\s*([\$\£\€]?\s*[\d,]+\.\d{2})$', re.IGNORECASE),
                confidence_weight=0.95,
                value_format=re.compile(r'[\$\£\€]?\s*[\d,]+\.\d{2}'),
                description="Total gross income"
            ),
            "taxable_income": KeyPattern(
                pattern=re.compile(r'^(?:taxable\s*income|adjusted\s*gross\s*income)\s*[:\s]?\s*([\$\£\€]?\s*[\d,]+\.\d{2})$', re.IGNORECASE),
                confidence_weight=0.9,
                value_format=re.compile(r'[\$\£\€]?\s*[\d,]+\.\d{2}'),
                description="Taxable income amount"
            ),
            "tax_due": KeyPattern(
                pattern=re.compile(r'^(?:tax\s*due|tax\s*liability|total\s*tax)\s*[:\s]?\s*([\$\£\€]?\s*[\d,]+\.\d{2})$', re.IGNORECASE),
                confidence_weight=0.9,
                value_format=re.compile(r'[\$\£\€]?\s*[\d,]+\.\d{2}'),
                description="Total tax amount due"
            ),
            "tax_paid": KeyPattern(
                pattern=re.compile(r'^(?:tax\s*paid|tax\s*withheld|taxes\s*paid)\s*[:\s]?\s*([\$\£\€]?\s*[\d,]+\.\d{2})$', re.IGNORECASE),
                confidence_weight=0.9,
                value_format=re.compile(r'[\$\£\€]?\s*[\d,]+\.\d{2}'),
                description="Tax amount already paid"
            )
        }

        return {
            DocumentType.PASSPORT: DocumentKeySet(
                document_type=DocumentType.PASSPORT,
                required_keys=passport_patterns,
                description="Universal passport key set"
            ),
            DocumentType.BANK_STATEMENT: DocumentKeySet(
                document_type=DocumentType.BANK_STATEMENT,
                required_keys=bank_statement_patterns,
                description="Universal bank statement key set"
            ),
            DocumentType.TAX_STATEMENT: DocumentKeySet(
                document_type=DocumentType.TAX_STATEMENT,
                required_keys=tax_statement_patterns,
                description="Universal tax statement key set"
            )
        }

    def get_key_set(self, document_type: DocumentType) -> DocumentKeySet:
        """Get the key set for a specific document type."""
        return self._key_sets.get(document_type)

    def get_all_keys(self, document_type: DocumentType) -> Dict[str, KeyPattern]:
        """Get all keys (required + optional) for a document type."""
        key_set = self._key_sets.get(document_type)
        if not key_set:
            return {}

        all_keys = key_set.required_keys.copy()
        if key_set.optional_keys:
            all_keys.update(key_set.optional_keys)

        return all_keys

    def get_key_pattern(self, document_type: DocumentType, key_name: str) -> KeyPattern:
        """Get pattern for a specific key in a document type."""
        all_keys = self.get_all_keys(document_type)
        return all_keys.get(key_name)

    def is_required_key(self, document_type: DocumentType, key_name: str) -> bool:
        """Check if a key is required for a document type."""
        key_set = self._key_sets.get(document_type)
        return key_set and key_name in key_set.required_keys


# Global instance
key_config = KeyConfig()
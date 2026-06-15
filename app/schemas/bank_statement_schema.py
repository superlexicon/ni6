from pydantic import BaseModel
from typing import Optional, Dict, List, Any, Union


class BankStatementData(BaseModel):
    """Standard fields extracted from bank statement documents"""

    # Account Holder Information
    account_holder_name: Optional[str] = None
    address: Optional[str] = None  # Customer's residential address
    branch_address: Optional[str] = None  # Bank's branch address

    # Account Details
    account_number: Optional[str] = None
    account_type: Optional[str] = None

    # Bank Information
    bank_name: Optional[str] = None
    bank_branch: Optional[str] = None  # Legacy field name, use bank_address instead
    bank_address: Optional[str] = None  # Bank branch address (full address)
    bank_code: Optional[str] = None
    swift_code: Optional[str] = None
    iban: Optional[str] = None
    ifsc_code: Optional[str] = None  # Indian Financial System Code

    # Country fields
    bank_country: Optional[str] = None
    account_holder_country: Optional[str] = None

    # Address components (structured)
    address_city: Optional[str] = None
    address_postal: Optional[str] = None
    address_state: Optional[str] = None
    address_country: Optional[str] = None

    # Statement Period
    statement_from_date: Optional[str] = None
    statement_to_date: Optional[str] = None
    statement_date: Optional[str] = None  # Single statement date in DD MMM YYYY format (latest date from ranges)

    # Balances
    opening_balance: Optional[float] = None
    closing_balance: Optional[float] = None
    currency: Optional[str] = None

    # Transactions
    transaction_count: Optional[int] = None
    transactions: Optional[List[Dict]] = None

    # Extraction metadata
    account_number_extraction_method: Optional[str] = None

    # Extraction source tracking
    extraction_source: Optional[str] = None  # 'gliner', 'spatial_ocr', 'spatial_geometry'
    llm_model_used: Optional[str] = None  # Reserved for future LLM-based extraction

    # Overall confidence score (0-100) for the extracted data
    overall_confidence: Optional[float] = None

    # Confidence scores for individual fields - supports both:
    # - Dict[str, float] (legacy format, 0-100)
    # - Dict[str, dict] (new format with 'overall_confidence' and 'sources')
    confidence_scores: Dict[str, Union[float, Dict[str, Any]]] = {}

    # Validation results
    validation_results: Dict[str, Any] = {}

    # Raw OCR text as concatenated string
    raw_data: Optional[str] = None  # Full OCR text for debugging/auditing

from pydantic import BaseModel
from typing import Optional, Dict, Any, Union


class TaxStatementData(BaseModel):
    """Standard fields extracted from tax statement documents"""

    # Taxpayer Information
    taxpayer_name: Optional[str] = None
    tax_id: Optional[str] = None
    social_security_number: Optional[str] = None
    address: Optional[str] = None

    # Tax Period
    tax_year: Optional[str] = None
    tax_period_start: Optional[str] = None
    tax_period_end: Optional[str] = None

    # Income Information
    gross_income: Optional[float] = None
    net_income: Optional[float] = None
    taxable_income: Optional[float] = None

    # Tax Information
    tax_paid: Optional[float] = None
    tax_withheld: Optional[float] = None
    tax_due: Optional[float] = None
    tax_refund: Optional[float] = None

    # Filing Information
    filing_date: Optional[str] = None
    filing_status: Optional[str] = None
    tax_authority: Optional[str] = None

    # Confidence Scores - supports both:
    # - Dict[str, float] (legacy format, 0-100)
    # - Dict[str, dict] (new format with 'overall_confidence' and 'sources')
    confidence_scores: Dict[str, Union[float, Dict[str, Any]]] = {}

    # Raw OCR text as concatenated string
    raw_data: Optional[str] = None  # Full OCR text for debugging/auditing

from pydantic import BaseModel
from typing import Optional, Dict, List, Any, Union


class WealthDeclarationData(BaseModel):
    """Standard fields extracted from wealth declaration documents"""

    # Declarant Information
    declarant_name: Optional[str] = None
    declaration_date: Optional[str] = None
    declaration_period_start: Optional[str] = None
    declaration_period_end: Optional[str] = None

    # Assets
    real_estate_value: Optional[float] = None
    bank_deposits_value: Optional[float] = None
    investments_value: Optional[float] = None
    vehicles_value: Optional[float] = None
    other_assets_value: Optional[float] = None
    total_assets: Optional[float] = None

    # Liabilities
    loans_value: Optional[float] = None
    mortgages_value: Optional[float] = None
    other_liabilities_value: Optional[float] = None
    total_liabilities: Optional[float] = None

    # Net Worth
    net_worth: Optional[float] = None
    currency: Optional[str] = None

    # Details
    asset_categories: Optional[List[str]] = None
    number_of_properties: Optional[int] = None

    # Signature and Verification
    signature_found: bool = False
    notarized: bool = False

    # Confidence Scores - supports both:
    # - Dict[str, float] (legacy format, 0-100)
    # - Dict[str, dict] (new format with 'overall_confidence' and 'sources')
    confidence_scores: Dict[str, Union[float, Dict[str, Any]]] = {}

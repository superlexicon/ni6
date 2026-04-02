from typing import Dict, List
from pydantic import BaseModel
from datetime import datetime


class VerificationResponse(BaseModel):
    id: str
    email: str
    check_item: str
    service_type: str
    verification_percentage: float
    check_requirement: str
    check_result: bool
    manual_check: bool
    created_at: datetime


class VerificationSuccessMessageResponse(BaseModel):
    verified: bool
    message: str


class VerificationResponseList(BaseModel):
    data: Dict[str, List[VerificationResponse]]

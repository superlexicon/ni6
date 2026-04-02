
from pydantic import BaseModel
from typing import Literal


class UpdateManualCheckRequest(BaseModel):
    manual_check: Literal["passed", "failed"]


class VerificationRequest(BaseModel):
    email: str
    username: str
    account_number: str
    phone_number: str
    passport: str
    bank_stmt: str
    resume: str
    selfie: str


class DocumentVerifyRequest(BaseModel):
    email: str
    username: str
    account_number: str
    phone_number: str
    passport: bytes
    bank_stmt: bytes
    resume: bytes

from pydantic import BaseModel

class OTPVerificationEvent(BaseModel):
    selfie: bytes
    email: str
    
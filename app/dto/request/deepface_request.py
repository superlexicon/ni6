from pydantic import BaseModel


class DeepFaceVerifyRequest(BaseModel):
    passport: bytes
    selfie: bytes
    email: str

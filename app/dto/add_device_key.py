from pydantic import BaseModel, Field


class SignatureData(BaseModel):
    """ECDSA signature components"""
    r: str = Field(..., description="Signature r component (hex)")
    s: str = Field(..., description="Signature s component (hex)")


class AddDeviceKeyRequest(BaseModel):
    """Request to add a new device key"""
    new_public_key: str = Field(..., description="New device's public key (secp256k1 hex)")
    secret_share: str = Field(..., description="Plaintext Shamir secret share for new device")
    signature: SignatureData = Field(..., description="ECDSA signature of new_public_key")


class AddDeviceKeyResponse(BaseModel):
    """Response after adding device key"""
    success: bool = Field(..., description="Whether operation succeeded")
    public_key: str = Field(..., description="Stored public key")
    created_at: str = Field(..., description="Timestamp of creation")
    message: str = Field(..., description="Success/error message")

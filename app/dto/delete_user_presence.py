"""
Models for user presence deletion endpoint.
"""
from pydantic import BaseModel, Field
from typing import Dict


class SignatureData(BaseModel):
    """ECDSA signature components for authentication."""
    r: str = Field(..., description="Signature r component (hex)")
    s: str = Field(..., description="Signature s component (hex)")


class DeletePresenceRequest(BaseModel):
    """Request to delete entire user presence from database."""
    message: str = Field(
        ...,
        description="Message that was signed (e.g., 'delete:{timestamp}')"
    )
    signature: SignatureData = Field(
        ...,
        description="ECDSA signature proving ownership of the private key"
    )
    public_key: str = Field(
        ...,
        description="Public key (hex) to verify signature against"
    )


class DeletePresenceResponse(BaseModel):
    """Response after deleting user presence."""
    success: bool = Field(..., description="Whether deletion was successful")
    message: str = Field(..., description="Human-readable status message")
    deleted_records: Dict[str, int] = Field(
        ...,
        description="Count of deleted records per table"
    )

from pydantic import BaseModel, Field, computed_field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class VerificationStep(str, Enum):
    SELFIE_PENDING = "selfie_pending"
    SELFIE_COMPLETED = "selfie_completed"
    PASSPORT_PENDING = "passport_pending"
    PASSPORT_COMPLETED = "passport_completed"
    ID_CARD_PENDING = "id_card_pending"
    ID_CARD_COMPLETED = "id_card_completed"
    BANK_PENDING = "bank_pending"
    COMPLETED = "completed"


class SequentialJobResponse(BaseModel):
    """
    Simplified response model for sequential job processing.

    result: Overall step result (True = all checks passed)
    verification_state: 0=initial, 1=after selfie, 2=after passport, 3=complete
    sequence_no: 0=initial, 1=selfie done, 2=passport data extracted, 3=complete
    """
    result: bool
    job_id: str
    verification_state: int  # 0-3, value AFTER this step (only incremented on result=True)
    sequence_no: int = 0  # 0-3, tracks attempted submissions independently from verification_state
    processing_time_seconds: float = 0.0
    user_identity_id: Optional[str] = None  # User identity ID for this submission
    error: Optional[str] = None  # Error message when result=False
    error_code: Optional[str] = None  # Error code for categorizing errors

    # Flat structure for results
    extracted_data: Optional[Dict[str, Any]] = None  # Key fields extracted
    forgery_checks: Optional[Dict[str, Any]] = None  # PhotoHolmes results
    other_checks: Optional[Dict[str, Any]] = None    # All validation checks

    # Additional detection fields (optional)
    message: Optional[str] = None  # Status message
    document_type: Optional[str] = None  # Document type
    detected_document_type: Optional[str] = None  # Auto-detected document type
    detected_country: Optional[str] = None  # Auto-detected country code
    detected_entity: Optional[str] = None  # Auto-detected entity
    detection_confidence: Optional[float] = None  # Detection confidence score
    selected_schema: Optional[str] = None  # Selected schema ID
    confidence_data: Optional[Dict[str, Any]] = None  # Field-level confidence data
    overall_confidence: Optional[float] = None  # Overall confidence score

    model_config = {"extra": "allow"}  # Allow extra fields for flexibility

    @computed_field
    @property
    def status(self) -> str:
        """Computed status based on result field."""
        return "completed" if self.result else "failed"
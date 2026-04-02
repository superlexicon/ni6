"""
ID Card Schema

Generic ID card schema that works with any ID card type (PAN, national ID, driver's license, etc.).
Returns key-value pairs rather than fixed schema for maximum flexibility.
"""

from pydantic import BaseModel
from typing import Optional, Dict, List, Any, Union


class FieldConfidence(BaseModel):
    """
    Confidence information for a single extracted field.
    Contains overall confidence and the sources that contributed to extraction.
    """
    overall_confidence: float
    sources: List[str] = []

    def get_confidence_value(self) -> float:
        """Get the confidence value (0-1 range, may need *100 for percentage)."""
        return self.overall_confidence


class IDCardData(BaseModel):
    """
    Generic ID card schema that works with any ID card type.
    Returns key-value pairs rather than fixed schema for maximum flexibility.
    """

    # Document Type (auto-detected or provided)
    document_type: Optional[str] = None  # e.g., "PAN", "driver_license", "national_id"
    issuing_country: Optional[str] = None

    # Core fields (most common across ID cards) - extracted for convenience
    full_name: Optional[str] = None
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None
    identification_number: Optional[str] = None  # Generic field for any ID number

    # Generic key-value pairs for ALL extracted fields
    # This allows storing any field regardless of ID card type
    field_values: Dict[str, str] = {}

    # Confidence scores per field - can be:
    # - Dict[str, float] (legacy format, 0-100)
    # - Dict[str, FieldConfidence] (new format with sources)
    # - Dict[str, dict] (raw dict with 'overall_confidence' and 'sources')
    confidence_scores: Dict[str, Union[float, FieldConfidence, Dict[str, Any]]] = {}

    # Raw GLiNER entities for debugging
    raw_entities: Optional[List[Dict[str, Any]]] = None

    # Overall confidence score (0-100)
    overall_confidence: Optional[float] = None

    def _get_confidence_float(self, value: Union[float, FieldConfidence, Dict[str, Any]]) -> float:
        """
        Extract a float confidence value from various formats.

        Args:
            value: Can be a float, FieldConfidence object, or dict

        Returns:
            Float confidence value (0-1 range)
        """
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, FieldConfidence):
            return value.overall_confidence
        elif isinstance(value, dict):
            return value.get('overall_confidence', 0.5)
        return 0.5

    def calculate_overall_confidence(self) -> float:
        """
        Calculate aggregate confidence score based on field extraction.

        Returns:
            float: Overall confidence score (0-100)
        """
        # If overall_confidence is already set, use it
        if self.overall_confidence is not None:
            return self.overall_confidence

        # Calculate based on average confidence from confidence_scores
        if self.confidence_scores:
            confidences = [self._get_confidence_float(v) for v in self.confidence_scores.values()]
            avg_confidence = sum(confidences) / len(confidences)
            # Convert from 0-1 range to 0-100 range if needed
            if avg_confidence <= 1.0:
                avg_confidence *= 100
            return round(avg_confidence, 2)

        # Fallback: Check number of extracted fields
        if self.field_values:
            # Base confidence on field count (more fields = higher confidence)
            base_confidence = min(50.0 + len(self.field_values) * 5.0, 85.0)
            return round(base_confidence, 2)

        return 0.0

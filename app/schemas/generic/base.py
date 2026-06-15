"""
Base data structures for document detection and extraction results.

This module defines the core data structures used by GenericDocumentService.
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from dataclasses import dataclass


@dataclass
class DocumentDetectionResult:
    """
    Result of document type detection.

    Attributes:
        document_type: Detected document type (e.g., "tax_return", "id_card")
        document_type_name: Human-readable document type name
        country_code: Detected ISO 3166-1 alpha-2 country code
        country_name: Human-readable country name
        entity: Detected entity (bank, institution, organization)
        entity_name: Human-readable entity name
        confidence: Overall detection confidence (0-1)
        type_confidence: Confidence score for document type
        country_confidence: Confidence score for country detection
        entity_confidence: Confidence score for entity detection
        detected_keywords: Keywords that contributed to detection
        detected_patterns: Patterns that contributed to detection
        detection_method: Method used for detection (qwen, pattern, hybrid)
    """

    document_type: Optional[str] = None
    document_type_name: Optional[str] = None
    country_code: Optional[str] = None
    country_name: Optional[str] = None
    entity: Optional[str] = None
    entity_name: Optional[str] = None
    confidence: float = 0.0
    type_confidence: float = 0.0
    country_confidence: float = 0.0
    entity_confidence: float = 0.0
    detected_keywords: List[str] = None
    detected_patterns: List[str] = None
    detection_method: str = "pattern"

    def __post_init__(self):
        if self.detected_keywords is None:
            self.detected_keywords = []
        if self.detected_patterns is None:
            self.detected_patterns = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "document_type": self.document_type,
            "document_type_name": self.document_type_name,
            "country_code": self.country_code,
            "country_name": self.country_name,
            "entity": self.entity,
            "entity_name": self.entity_name,
            "confidence": self.confidence,
            "type_confidence": self.type_confidence,
            "country_confidence": self.country_confidence,
            "entity_confidence": self.entity_confidence,
            "detected_keywords": self.detected_keywords,
            "detected_patterns": self.detected_patterns,
            "detection_method": self.detection_method,
        }

    @property
    def is_confident(self) -> bool:
        """Check if detection confidence is above threshold."""
        return self.confidence >= 0.6


class ExtractionResult(BaseModel):
    """
    Result of document extraction.

    Attributes:
        schema_id: The schema identifier that was used for extraction (if any)
        extracted_data: Dictionary of extracted field values
        confidence_scores: Dictionary of confidence scores per field (0-1 or dict with 'overall_confidence' and 'sources')
        overall_confidence: Overall extraction confidence (0-1)
        missing_required_fields: List of required fields that weren't extracted
        extracted_fields: List of fields that were successfully extracted
        detection_result: Document detection result (type, country, entity)
    """

    schema_id: Optional[str] = None
    extracted_data: Dict[str, Any] = Field(default_factory=dict)
    confidence_scores: Dict[str, Union[float, Dict[str, Any]]] = Field(default_factory=dict)
    overall_confidence: float = 0.0
    missing_required_fields: List[str] = Field(default_factory=list)
    extracted_fields: List[str] = Field(default_factory=list)
    detection_result: Optional[DocumentDetectionResult] = None

    def _get_confidence_float(self, value: Union[float, Dict[str, Any]]) -> float:
        """Extract a float confidence value from various formats."""
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, dict):
            return value.get('overall_confidence', 0.0)
        return 0.0

    class Config:
        json_schema_extra = {
            "example": {
                "schema_id": "id_card:SG",
                "extracted_data": {
                    "full_name": "John Doe",
                    "id_number": "S1234567D",
                },
                "confidence_scores": {
                    "full_name": 0.92,
                    "id_number": 0.88,
                },
                "overall_confidence": 0.90,
                "missing_required_fields": [],
                "extracted_fields": ["full_name", "id_number"],
            }
        }

    def to_response_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        confidence_response = {}
        for k, v in self.confidence_scores.items():
            conf_val = self._get_confidence_float(v)
            confidence_response[k] = conf_val * 100

        return {
            "extracted_data": self.extracted_data,
            "confidence_scores": confidence_response,
            "overall_confidence": self.overall_confidence * 100,
            "missing_required_fields": self.missing_required_fields,
            "extracted_fields": self.extracted_fields,
            "schema_id": self.schema_id,
            "detection": self.detection_result.to_dict() if self.detection_result else None,
        }

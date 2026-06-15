"""
Base schema classes for the generic document type detector and schema library.

This module defines the core data structures for:
- GLINER2Schema: Natural language field descriptions for zero-shot extraction
- DocumentTypeSchema: Complete schema definition with detection patterns
- DocumentDetectionResult: Result of three-tier document type detection
"""

from typing import Dict, List, Optional, Any, Set, Union
from pydantic import BaseModel, Field
from dataclasses import dataclass
from enum import Enum


class SchemaEntityType(str, Enum):
    """Entity types that can be extracted using GLiNER2."""

    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    NUMBER = "number"
    MONEY = "money"
    IDENTIFIER = "identifier"
    EMAIL = "email"
    PHONE = "phone"
    ADDRESS = "address"
    CUSTOM = "custom"


@dataclass
class GLINER2SchemaField:
    """
    A single field definition for GLiNER2 schema-based extraction.

    GLiNER2 uses natural language descriptions to extract entities zero-shot.
    The more descriptive and specific the description, the better the extraction.

    Attributes:
        field_name: The internal field name (e.g., "taxpayer_name")
        description: Natural language description for zero-shot extraction
        entity_type: Type of entity (helps GLiNER2 understand what to extract)
        required: Whether this field is required for validation
        examples: Few-shot examples to improve extraction accuracy
        pattern: Regex pattern for validation fallback
    """

    field_name: str
    description: str
    entity_type: SchemaEntityType = SchemaEntityType.CUSTOM
    required: bool = False
    examples: Optional[List[str]] = None
    pattern: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to GLiNER2 schema format."""
        return {
            "field_name": self.field_name,
            "description": self.description,
            "entity_type": self.entity_type.value,
            "required": self.required,
            "examples": self.examples or [],
            "pattern": self.pattern,
        }


class GLINER2Schema(BaseModel):
    """
    GLiNER2 schema for zero-shot entity extraction.

    This schema defines what fields to extract from a document using
    natural language descriptions. GLiNER2 can extract these entities
    without any training data.

    Example:
        ```python
        schema = GLINER2Schema(
            fields={
                "taxpayer_name": "The taxpayer's full legal name",
                "tax_id_number": "The tax identification number or NRIC/FIN",
                "assessment_year": "The year of assessment (e.g., 2024)",
            }
        )
        ```
    """

    fields: Dict[str, str] = Field(
        default_factory=dict,
        description="Field name to natural language description mapping"
    )

    threshold: float = Field(
        default=0.2,
        description="Confidence threshold for entity extraction (0-1)"
    )

    entity_types: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional entity type hints for each field"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "fields": {
                    "taxpayer_name": "The taxpayer's full legal name",
                    "tax_id_number": "The tax identification number",
                    "assessment_year": "The tax assessment year",
                },
                "threshold": 0.2,
            }
        }

    def get_field_descriptions(self) -> List[str]:
        """Get field descriptions as a list for GLiNER2."""
        return list(self.fields.values())

    def get_entity_types(self) -> Dict[str, str]:
        """Get entity types mapping for GLiNER2."""
        if self.entity_types:
            return self.entity_types
        # Default: all fields are custom entities
        return {field: "custom" for field in self.fields.keys()}


@dataclass
class DocumentDetectionResult:
    """
    Result of three-tier document type detection.

    The detection follows a hierarchical approach:
    1. Document Type (tax_return, id_card, driving_license, etc.)
    2. Country (SG, IN, US, MY, TH, etc.)
    3. Entity (DBS, SBI, Chase, IRAS, etc.)

    Attributes:
        document_type: Detected document type (e.g., "tax_return")
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
        schema_used: The schema that was selected for extraction
        detection_method: Method used for detection (gliner2, pattern, hybrid)
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
    schema_used: Optional['DocumentTypeSchema'] = None
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
            "schema_id": self.schema_used.schema_id if self.schema_used else None,
        }

    @property
    def is_confident(self) -> bool:
        """Check if detection confidence is above threshold."""
        return self.confidence >= 0.6

    @property
    def schema_key(self) -> str:
        """Get the schema key for lookup (type:country:entity)."""
        parts = [self.document_type or "unknown"]
        if self.country_code:
            parts.append(self.country_code.lower())
        if self.entity:
            parts.append(self.entity.lower())
        return ":".join(parts)


class DocumentTypeSchema(BaseModel):
    """
    Complete schema definition for a document type.

    This schema combines:
    1. Extraction schema (GLiNER2 field definitions)
    2. Detection patterns (keywords, labels, regex)
    3. Validation requirements (required/optional fields)

    Schemas are organized hierarchically:
    - Generic: type only (e.g., "tax_return")
    - Country-specific: type:country (e.g., "tax_return:SG")
    - Entity-specific: type:country:entity (e.g., "tax_return:SG:iras")

    Attributes:
        schema_id: Unique identifier (type:country:entity format)
        document_type: Document type identifier (e.g., "tax_return")
        document_type_name: Human-readable document type name
        country_code: ISO 3166-1 alpha-2 country code (optional)
        country_name: Human-readable country name (optional)
        entity: Entity identifier (e.g., "iras", "dbs") (optional)
        entity_name: Human-readable entity name (optional)
        extraction_schema: GLiNER2 schema for field extraction
        detection_patterns: Pattern definitions for detection
        required_fields: List of required field names
        optional_fields: List of optional field names
        priority: Schema priority for selection (higher = more specific)
        enabled: Whether this schema is active
    """

    schema_id: str = Field(
        ...,
        description="Unique schema identifier (type:country:entity or type:country or type)"
    )

    document_type: str = Field(
        ...,
        description="Document type identifier (e.g., 'tax_return', 'id_card')"
    )

    document_type_name: str = Field(
        ...,
        description="Human-readable document type name"
    )

    country_code: Optional[str] = Field(
        default=None,
        description="ISO 3166-1 alpha-2 country code"
    )

    country_name: Optional[str] = Field(
        default=None,
        description="Human-readable country name"
    )

    entity: Optional[str] = Field(
        default=None,
        description="Entity identifier (e.g., 'iras', 'dbs', 'sbi')"
    )

    entity_name: Optional[str] = Field(
        default=None,
        description="Human-readable entity name"
    )

    extraction_schema: GLINER2Schema = Field(
        default_factory=GLINER2Schema,
        description="GLiNER2 schema for field extraction"
    )

    detection_patterns: Dict[str, Any] = Field(
        default_factory=dict,
        description="Pattern definitions for document detection"
    )

    required_fields: List[str] = Field(
        default_factory=list,
        description="List of required field names for validation"
    )

    optional_fields: List[str] = Field(
        default_factory=list,
        description="List of optional field names"
    )

    priority: int = Field(
        default=0,
        description="Schema priority for selection (higher = more specific)"
    )

    enabled: bool = Field(
        default=True,
        description="Whether this schema is active"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "schema_id": "tax_return:SG:iras",
                "document_type": "tax_return",
                "document_type_name": "Tax Return",
                "country_code": "SG",
                "country_name": "Singapore",
                "entity": "iras",
                "entity_name": "Inland Revenue Authority of Singapore",
                "extraction_schema": {
                    "fields": {
                        "taxpayer_name": "The taxpayer's full legal name",
                        "tax_id_number": "The tax identification number",
                    },
                    "threshold": 0.2,
                },
                "detection_patterns": {
                    "keywords": ["iras", "income tax", "assessment year"],
                    "labels": ["inland revenue authority of singapore"],
                },
                "required_fields": ["taxpayer_name"],
                "optional_fields": ["tax_id_number", "assessment_year"],
                "priority": 100,
                "enabled": True,
            }
        }

    @property
    def specificity(self) -> int:
        """
        Calculate schema specificity based on how many dimensions are specified.

        Returns:
            Specificity score: 1=type only, 2=type+country, 3=type+country+entity
        """
        specificity = 1  # At least document_type is always specified
        if self.country_code:
            specificity += 1
        if self.entity:
            specificity += 1
        return specificity

    @property
    def level(self) -> str:
        """
        Get the schema level in the hierarchy.

        Returns:
            "generic", "country", or "entity"
        """
        if self.entity:
            return "entity"
        elif self.country_code:
            return "country"
        else:
            return "generic"

    def matches(
        self,
        document_type: Optional[str] = None,
        country_code: Optional[str] = None,
        entity: Optional[str] = None
    ) -> bool:
        """
        Check if this schema matches the given criteria.

        Args:
            document_type: Document type to match (None = any)
            country_code: Country code to match (None = any)
            entity: Entity to match (None = any)

        Returns:
            True if this schema matches all non-None criteria
        """
        if document_type and self.document_type != document_type:
            return False
        if country_code and self.country_code != country_code:
            return False
        if entity and self.entity != entity:
            return False
        return True

    def get_detection_keywords(self) -> Set[str]:
        """Get keywords for detection from detection_patterns."""
        keywords = set()
        if "keywords" in self.detection_patterns:
            keywords.update(self.detection_patterns["keywords"])
        if "labels" in self.detection_patterns:
            keywords.update(self.detection_patterns["labels"])
        return keywords

    def matches_text(self, text: str) -> float:
        """
        Calculate how well this schema matches the given text.

        Args:
            text: The document text to match against

        Returns:
            Match score (0-1) based on keyword/pattern presence
        """
        if not text:
            return 0.0

        text_lower = text.lower()
        keywords = self.get_detection_keywords()

        if not keywords:
            return 0.5  # Neutral score if no keywords defined

        matched_count = sum(1 for kw in keywords if kw.lower() in text_lower)
        return min(matched_count / len(keywords), 1.0)


class ExtractionResult(BaseModel):
    """
    Result of document extraction using a DocumentTypeSchema.

    Attributes:
        schema_used: The schema that was used for extraction
        extracted_data: Dictionary of extracted field values
        confidence_scores: Dictionary of confidence scores per field (0-1)
        overall_confidence: Overall extraction confidence (0-1)
        missing_required_fields: List of required fields that weren't extracted
        extracted_fields: List of fields that were successfully extracted
        detection_result: Document detection result (type, country, entity)
    """

    schema_used: Optional[DocumentTypeSchema] = None
    extracted_data: Dict[str, Any] = Field(default_factory=dict)
    # Confidence scores - supports both float (0-1) and dict with 'overall_confidence' and 'sources'
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
                "schema_used": "tax_return:SG:iras",
                "extracted_data": {
                    "taxpayer_name": "John Doe",
                    "tax_id_number": "S1234567D",
                },
                "confidence_scores": {
                    "taxpayer_name": 0.92,
                    "tax_id_number": 0.88,
                },
                "overall_confidence": 0.90,
                "missing_required_fields": [],
                "extracted_fields": ["taxpayer_name", "tax_id_number"],
            }
        }

    def to_response_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        # Handle both float and dict confidence formats
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
            "schema_id": self.schema_used.schema_id,
            "detection": self.detection_result.to_dict() if self.detection_result else None,
        }

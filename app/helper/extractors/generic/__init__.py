"""
Generic Document Extractors

This package provides GLiNER2-powered extraction for generic document types.
It includes:
- Document type detection (three-tier: type, country, entity)
- Schema selection with hierarchical fallback
- Field extraction using zero-shot NER

Main exports:
- DocumentTypeDetector: Detect document type, country, and entity
- SchemaSelector: Select the best schema for extraction
- GenericDocumentExtractor: Extract fields using selected schema
"""

from .document_type_detector import (
    DocumentTypeDetector,
    DocumentDetectionResult,
    detect_document_type,
    EXISTING_DOCUMENT_TYPES,
    ENTITY_KEYWORDS,
)

from .schema_selector import (
    SchemaSelector,
    SchemaSelectionResult,
    select_schema,
    get_best_schema_for_text,
)

__all__ = [
    # Document Type Detector
    "DocumentTypeDetector",
    "DocumentDetectionResult",
    "detect_document_type",
    "EXISTING_DOCUMENT_TYPES",
    "ENTITY_KEYWORDS",
    # Schema Selector
    "SchemaSelector",
    "SchemaSelectionResult",
    "select_schema",
    "get_best_schema_for_text",
]

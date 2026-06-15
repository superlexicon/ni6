"""
Generic Document Schema Library

This module provides data structures for document detection and extraction results.

Note: The SchemaRegistry has been removed as it was unused. Document processing
now uses Qwen extractors directly via GenericDocumentService.
"""

from .base import (
    DocumentDetectionResult,
    ExtractionResult,
)


__all__ = [
    "DocumentDetectionResult",
    "ExtractionResult",
]

"""
Key Injection Module for Enhanced OCR Processing.

This module provides key injection functionality that enhances doctr OCR
with real-time key detection and spatial relationship analysis.

Main Components:
- KeyConfig: Universal key definitions for all document types
- KeyFieldDetector: Real-time key detection during OCR
- SpatialAnalyzer: Geometry-based key-value relationship mapping
- ConfidenceCalculator: Enhanced confidence scoring
- KeyInjectionManager: Main coordinator for key injection functionality

Usage Example:
    from app.core.key_injection import key_injection_manager, DocumentType

    # Process OCR results with key injection
    result = key_injection_manager.process_document_with_key_injection(
        ocr_geometry_data=ocr_results,
        document_type=DocumentType.PASSPORT
    )

    # Get high-confidence extractions
    high_confidence = key_injection_manager.get_high_confidence_extractions(result)
"""

from .key_config import (
    DocumentType,
    DocumentKeySet,
    KeyPattern,
    key_config
)

from .key_field_detector import (
    KeyMatch,
    DetectedKey,
    KeyFieldDetector
)

from .spatial_analyzer import (
    TextElement,
    SpatialRelationship,
    SpatialAnalyzer
)

from .confidence_calculator import (
    ConfidenceFactors,
    EnhancedConfidence,
    ConfidenceCalculator
)

from .key_injection_manager import (
    KeyInjectionResult,
    ExtractionSummary,
    KeyInjectionManager,
    key_injection_manager
)

__version__ = "1.0.0"
__all__ = [
    # Main classes
    'KeyInjectionManager',
    'KeyFieldDetector',
    'SpatialAnalyzer',
    'ConfidenceCalculator',

    # Data structures
    'KeyInjectionResult',
    'DetectedKey',
    'EnhancedConfidence',
    'ExtractionSummary',

    # Configuration
    'DocumentType',
    'DocumentKeySet',
    'KeyPattern',
    'key_config',

    # Global instance
    'key_injection_manager'
]
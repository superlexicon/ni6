"""
Common data structures for key injection components.
This module contains shared dataclasses to avoid circular imports.
"""

from typing import Dict, Optional, List, Any
from dataclasses import dataclass


@dataclass
class TextElement:
    """Represents a text element with geometry information."""
    text: str
    confidence: float
    geometry: Dict[str, float]  # {'x1': 0.123, 'y1': 0.456, 'x2': 0.789, 'y2': 0.890}
    is_key: bool = False
    key_name: Optional[str] = None


@dataclass
class SpatialRelationship:
    """Represents spatial relationship between a key and potential value."""
    key_element: TextElement
    value_element: TextElement
    spatial_confidence: float
    relationship_type: str  # 'horizontal', 'vertical', 'below', 'right'


@dataclass
class OCRLinesContainer:
    """Container for original OCR lines with confidences for post-processing."""
    lines: List[Dict[str, Any]]  # [{'text': str, 'confidence': float}]
    document_hash: str

    def find_confidence_for_text(self, target_text: str) -> Optional[float]:
        """Simple exact/substring matching for confidence lookup."""
        if not target_text:
            return None

        target_clean = target_text.strip().lower()

        # Exact match first (highest confidence)
        for line in self.lines:
            if line['text'].strip().lower() == target_clean:
                return line['confidence']

        # Substring match (good coverage)
        for line in self.lines:
            line_clean = line['text'].strip().lower()
            if target_clean in line_clean:
                return line['confidence']

        return None  # No match found
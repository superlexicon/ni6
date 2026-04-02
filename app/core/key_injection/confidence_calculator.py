"""
Confidence calculator for enhanced confidence scoring of key-value pairs.
This module provides sophisticated confidence calculation that considers
key field importance, spatial relationships, and format validation.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import re

from .key_field_detector import DetectedKey
from .key_config import key_config, DocumentType


@dataclass
class ConfidenceFactors:
    """Represents different factors contributing to confidence score."""
    ocr_confidence: float = 0.0           # Original OCR confidence
    spatial_confidence: float = 0.0       # Spatial relationship confidence
    format_confidence: float = 0.0        # Format validation confidence
    key_importance_confidence: float = 0.0 # Key importance weighting
    cross_validation_confidence: float = 0.0 # Cross-validation confidence
    pattern_specificity_confidence: float = 0.0 # Pattern specificity confidence


@dataclass
class EnhancedConfidence:
    """Enhanced confidence calculation with breakdown by factors."""
    overall_confidence: float
    factors: ConfidenceFactors
    explanation: str
    is_reliable: bool
    needs_review: bool


class ConfidenceCalculator:
    """Enhanced confidence calculator for key injection system."""

    def __init__(self):
        # Confidence weights for different factors
        self.weights = {
            'ocr': 0.25,              # Original OCR confidence
            'spatial': 0.25,          # Spatial relationship confidence
            'format': 0.20,           # Format validation confidence
            'key_importance': 0.15,   # Key importance weighting
            'pattern_specificity': 0.10, # Pattern specificity confidence
            'cross_validation': 0.05   # Cross-validation confidence
        }

        # Minimum confidence thresholds
        self.thresholds = {
            'reliable': 0.85,         # Considered reliable
            'acceptable': 0.70,       # Acceptable but review recommended
            'minimum': 0.50           # Minimum for processing
        }

    def calculate_enhanced_confidence(self, detected_key: DetectedKey,
                                    ocr_confidence: float = 0.9,
                                    cross_document_values: Optional[List[str]] = None) -> EnhancedConfidence:
        """
        Calculate enhanced confidence for a detected key-value pair.

        Args:
            detected_key: The detected key with potential value
            ocr_confidence: Original OCR confidence for the text
            cross_document_values: Values from other documents for cross-validation

        Returns:
            EnhancedConfidence object with detailed breakdown
        """
        factors = ConfidenceFactors()

        # Calculate individual confidence factors
        factors.ocr_confidence = self._calculate_ocr_confidence(detected_key, ocr_confidence)
        factors.spatial_confidence = self._calculate_spatial_confidence(detected_key)
        factors.format_confidence = self._calculate_format_confidence(detected_key)
        factors.key_importance_confidence = self._calculate_key_importance_confidence(detected_key)
        factors.pattern_specificity_confidence = self._calculate_pattern_specificity_confidence(detected_key)
        factors.cross_validation_confidence = self._calculate_cross_validation_confidence(
            detected_key, cross_document_values
        )

        # Calculate overall confidence using weighted average
        overall_confidence = self._combine_confidence_factors(factors)

        # Determine reliability and review status
        is_reliable = overall_confidence >= self.thresholds['reliable']
        needs_review = overall_confidence < self.thresholds['acceptable']

        # Generate explanation
        explanation = self._generate_confidence_explanation(detected_key, factors, overall_confidence)

        return EnhancedConfidence(
            overall_confidence=overall_confidence,
            factors=factors,
            explanation=explanation,
            is_reliable=is_reliable,
            needs_review=needs_review
        )

    def _calculate_ocr_confidence(self, detected_key: DetectedKey, base_ocr_confidence: float) -> float:
        """Calculate OCR confidence with adjustments for key fields."""
        # Base OCR confidence
        ocr_conf = base_ocr_confidence

        # Boost confidence if the key detection was high confidence
        if detected_key.confidence > 0.9:
            ocr_conf *= 1.05
        elif detected_key.confidence < 0.7:
            ocr_conf *= 0.9

        # Check for common OCR errors in the value
        if detected_key.value_candidate:
            ocr_conf *= self._check_ocr_quality(detected_key.value_candidate)

        return min(ocr_conf, 1.0)

    def _calculate_spatial_confidence(self, detected_key: DetectedKey) -> float:
        """Calculate spatial relationship confidence."""
        if not detected_key.value_confidence:
            return 0.0

        # Use the value confidence from spatial analysis
        spatial_conf = detected_key.value_confidence

        # Boost confidence if geometry is available and reasonable
        if detected_key.geometry:
            # Check if geometry coordinates are reasonable
            if self._is_geometry_reasonable(detected_key.geometry):
                spatial_conf *= 1.1

        return min(spatial_conf, 1.0)

    def _calculate_format_confidence(self, detected_key: DetectedKey) -> float:
        """Calculate format validation confidence."""
        if not detected_key.value_candidate:
            return 0.0

        # Get the expected format for this key
        key_pattern = key_config.get_key_pattern(detected_key.document_type, detected_key.key_name)
        if not key_pattern or not key_pattern.value_format:
            return 0.8  # Default confidence if no format restriction

        value = detected_key.value_candidate.strip()

        try:
            if key_pattern.value_format.match(value):
                return 1.0
            else:
                return 0.3  # Low confidence if format doesn't match
        except re.error:
            return 0.5  # Medium confidence if regex is invalid

    def _calculate_key_importance_confidence(self, detected_key: DetectedKey) -> float:
        """Calculate confidence based on key importance."""
        # Required keys get higher confidence
        if key_config.is_required_key(detected_key.document_type, detected_key.key_name):
            return 1.0
        else:
            return 0.7  # Lower confidence for optional keys

    def _calculate_pattern_specificity_confidence(self, detected_key: DetectedKey) -> float:
        """Calculate confidence based on pattern specificity."""
        # Get the pattern used for key detection
        key_pattern = key_config.get_key_pattern(detected_key.document_type, detected_key.key_name)
        if not key_pattern:
            return 0.5

        # More specific patterns get higher confidence
        confidence_weight = key_pattern.confidence_weight
        pattern = key_pattern.pattern.pattern

        # Boost confidence for patterns with capture groups (more specific)
        if '(' in pattern and ')' in pattern:
            confidence_weight *= 1.1

        # Boost confidence for patterns with anchors (more specific)
        if '^' in pattern or '$' in pattern:
            confidence_weight *= 1.05

        return min(confidence_weight, 1.0)

    def _calculate_cross_validation_confidence(self, detected_key: DetectedKey,
                                             cross_document_values: Optional[List[str]]) -> float:
        """Calculate cross-validation confidence from other documents."""
        if not cross_document_values or not detected_key.value_candidate:
            return 0.0

        current_value = detected_key.value_candidate.strip()

        # Check for exact matches in cross-document values
        if current_value in cross_document_values:
            return 1.0

        # Check for partial matches (e.g., names, addresses)
        partial_matches = 0
        for value in cross_document_values:
            if self._is_partial_match(current_value, value):
                partial_matches += 1

        if partial_matches > 0:
            return 0.8 + (0.2 * partial_matches / len(cross_document_values))

        return 0.0

    def _combine_confidence_factors(self, factors: ConfidenceFactors) -> float:
        """Combine individual confidence factors using weighted average."""
        combined = (
            factors.ocr_confidence * self.weights['ocr'] +
            factors.spatial_confidence * self.weights['spatial'] +
            factors.format_confidence * self.weights['format'] +
            factors.key_importance_confidence * self.weights['key_importance'] +
            factors.pattern_specificity_confidence * self.weights['pattern_specificity'] +
            factors.cross_validation_confidence * self.weights['cross_validation']
        )

        return min(combined, 1.0)  # Cap at 1.0

    def _check_ocr_quality(self, text: str) -> float:
        """Check for common OCR quality issues and adjust confidence."""
        quality_multiplier = 1.0

        # Penalize common OCR error patterns
        if re.search(r'[0-9]l[0-9]', text):  # Could be "101" misread as "1l1"
            quality_multiplier *= 0.9

        if re.search(r'[0-9]o[0-9]', text):  # Could be "101" misread as "1o1"
            quality_multiplier *= 0.9

        if re.search(r'[a-z]{2,}[0-9]+', text):  # Letters followed by numbers (common in IDs)
            # Check for potential character confusion
            if '0' in text or '1' in text or 'o' in text or 'l' in text:
                quality_multiplier *= 0.95

        # Boost confidence for clean patterns
        if re.match(r'^[a-zA-Z0-9\s\-\.]+$', text):  # Only expected characters
            quality_multiplier *= 1.02

        return quality_multiplier

    def _is_geometry_reasonable(self, geometry: Dict[str, float]) -> bool:
        """Check if geometry coordinates are reasonable."""
        x1, y1 = geometry['x1'], geometry['y1']
        x2, y2 = geometry['x2'], geometry['y2']

        # Check if coordinates are within valid range (0-1)
        if not all(0.0 <= coord <= 1.0 for coord in [x1, y1, x2, y2]):
            return False

        # Check if x2 > x1 and y2 > y1
        if x2 <= x1 or y2 <= y1:
            return False

        # Check if the text area is reasonable (not too large or small)
        area = (x2 - x1) * (y2 - y1)
        if area < 0.0001 or area > 0.5:  # Too small or too large
            return False

        return True

    def _is_partial_match(self, text1: str, text2: str) -> bool:
        """Check if two texts have partial similarity."""
        # Normalize both texts
        norm1 = text1.lower().strip().replace(' ', '')
        norm2 = text2.lower().strip().replace(' ', '')

        # Exact match
        if norm1 == norm2:
            return True

        # Check if one is substring of the other
        if norm1 in norm2 or norm2 in norm1:
            return True

        # Check for high similarity (Levenshtein distance would be ideal, but using simple check)
        if len(norm1) > 5 and len(norm2) > 5:
            # Check if the first/last parts match
            if norm1[:5] == norm2[:5] or norm1[-5:] == norm2[-5:]:
                return True

        return False

    def _generate_confidence_explanation(self, detected_key: DetectedKey,
                                        factors: ConfidenceFactors,
                                        overall_confidence: float) -> str:
        """Generate human-readable explanation of confidence calculation."""
        explanations = []

        if factors.ocr_confidence > 0.8:
            explanations.append("High OCR quality")
        elif factors.ocr_confidence < 0.6:
            explanations.append("Potential OCR issues detected")

        if factors.spatial_confidence > 0.8:
            explanations.append("Strong spatial relationship")
        elif factors.spatial_confidence < 0.5:
            explanations.append("Weak spatial relationship")

        if factors.format_confidence > 0.9:
            explanations.append("Value matches expected format")
        elif factors.format_confidence < 0.5:
            explanations.append("Value doesn't match expected format")

        if factors.key_importance_confidence > 0.9:
            explanations.append("Required field detected")
        elif factors.key_importance_confidence < 0.8:
            explanations.append("Optional field detected")

        if factors.cross_validation_confidence > 0.8:
            explanations.append("Consistent with other documents")
        elif factors.cross_validation_confidence > 0:
            explanations.append("Cross-validation available")
        else:
            explanations.append("No cross-validation data")

        if overall_confidence > 0.85:
            reliability = "High confidence - reliable extraction"
        elif overall_confidence > 0.70:
            reliability = "Medium confidence - review recommended"
        else:
            reliability = "Low confidence - manual verification needed"

        return f"{reliability}. Factors: {', '.join(explanations)}"

    def get_confidence_summary(self, enhanced_confidence: EnhancedConfidence) -> Dict:
        """Get a summary of confidence calculation for logging/debugging."""
        return {
            'overall_confidence': enhanced_confidence.overall_confidence,
            'is_reliable': enhanced_confidence.is_reliable,
            'needs_review': enhanced_confidence.needs_review,
            'factors': {
                'ocr_confidence': enhanced_confidence.factors.ocr_confidence,
                'spatial_confidence': enhanced_confidence.factors.spatial_confidence,
                'format_confidence': enhanced_confidence.factors.format_confidence,
                'key_importance_confidence': enhanced_confidence.factors.key_importance_confidence,
                'pattern_specificity_confidence': enhanced_confidence.factors.pattern_specificity_confidence,
                'cross_validation_confidence': enhanced_confidence.factors.cross_validation_confidence
            },
            'explanation': enhanced_confidence.explanation
        }
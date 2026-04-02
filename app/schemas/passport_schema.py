from pydantic import BaseModel
from typing import Optional, Dict, Any


class PassportData(BaseModel):
    """Standard fields extracted from passport documents"""

    # Personal Information
    full_name: Optional[str] = None
    date_of_birth: Optional[str] = None
    sex: Optional[str] = None
    place_of_birth: Optional[str] = None

    # Passport Details
    passport_number: Optional[str] = None
    nationality: Optional[str] = None
    passport_country: Optional[str] = None
    issuing_authority: Optional[str] = None
    date_of_issue: Optional[str] = None
    date_of_expiry: Optional[str] = None

    # Overall confidence score (0-100) for the extracted data
    overall_confidence: Optional[float] = None

    # Individual field confidence scores from OCR
    field_confidences: Optional[Dict[str, float]] = None

    def calculate_overall_confidence(self) -> float:
        """
        Calculate aggregate confidence score based on OCR quality when available.

        If overall_confidence is already set (from OCR extraction), use it.
        Otherwise, calculate based on field existence as fallback.

        Returns:
            float: Overall confidence score (0-100)
            - >= 85: High confidence, proceed with extraction
            - 70-84: Medium confidence, manual review recommended
            - < 70: Low confidence, rescan required
        """
        # If overall_confidence is already set from OCR extraction, use it
        if self.overall_confidence is not None:
            return self.overall_confidence

        # Fallback: Calculate confidence based on field existence
        # Define field weights (must sum to 1.0)
        # Note: Using full_name (0.20) instead of surname (0.10) + given_names (0.10)
        field_weights = {
            # Critical fields - 50% total weight
            'passport_number': 0.15,
            'full_name': 0.20,  # Combined weight for surname + given_names
            'date_of_birth': 0.10,
            'nationality': 0.05,

            # Important fields - 20% total weight
            'sex': 0.05,
            'date_of_expiry': 0.15,

            # Supporting fields - 30% total weight
            'passport_country': 0.10,
            'place_of_birth': 0.10,
            'issuing_authority': 0.10,
        }

        # Check if we have data for each field
        field_exists = {
            'passport_number': bool(self.passport_number),
            'full_name': bool(self.full_name),
            'date_of_birth': bool(self.date_of_birth),
            'nationality': bool(self.nationality),
            'sex': bool(self.sex),
            'date_of_expiry': bool(self.date_of_expiry),
            'passport_country': bool(self.passport_country),
            'place_of_birth': bool(self.place_of_birth),
            'issuing_authority': bool(self.issuing_authority),
        }

        weighted_score = 0.0
        total_weight = 0.0

        for field, weight in field_weights.items():
            if field_exists[field]:
                # Field exists - assume base confidence of 75%
                weighted_score += 75.0 * weight
                total_weight += weight
            # If field doesn't exist, skip it (don't penalize)

        # Calculate final score
        if total_weight > 0:
            overall = weighted_score / total_weight
        else:
            overall = 0.0

        # Apply penalties for missing critical fields
        # Note: Using full_name instead of surname, since they're concatenated
        critical_fields = ['passport_number', 'full_name', 'date_of_birth']
        missing_critical = sum(1 for f in critical_fields if not field_exists[f])

        if missing_critical > 0:
            # Reduce confidence by 15% for each missing critical field
            penalty = missing_critical * 15.0
            overall = max(0, overall - penalty)

        return round(overall, 2)

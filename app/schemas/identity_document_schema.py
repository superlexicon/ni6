"""
Unified Identity Document Schema

Supports both passports and ID cards with a common data model.
Uses field names compatible with UnifiedIDExtractor output.
"""

from pydantic import BaseModel
from typing import Optional, Dict, Any


class IdentityDocumentData(BaseModel):
    """Standard fields extracted from identity documents (passports and ID cards)"""

    # Personal Information
    full_name: Optional[str] = None
    dob: Optional[str] = None  # Date of birth
    sex: Optional[str] = None
    place_of_birth: Optional[str] = None
    address: Optional[str] = None  # For ID cards

    # Document Details
    number: Optional[str] = None  # Document number (passport_number or id_number)
    nationality: Optional[str] = None
    document_country: Optional[str] = None  # Country code (SG, TH, etc.)
    document_type: Optional[str] = None  # "passport" or "id_card"
    issuing_authority: Optional[str] = None
    date_of_issue: Optional[str] = None
    date_of_expiry: Optional[str] = None  # May be None for ID cards (e.g., Singapore NRIC)

    # NRC-specific (Myanmar)
    nrc_number: Optional[str] = None

    # Religion (for some ID cards like Malaysia MyKad)
    religion: Optional[str] = None

    # Ethnicity (for some ID cards like China Resident ID)
    ethnicity: Optional[str] = None

    # Overall confidence score (0-100) for the extracted data
    overall_confidence: Optional[float] = None

    # Individual field confidence scores from OCR
    field_confidences: Optional[Dict[str, float]] = None

    def calculate_overall_confidence(self) -> float:
        """
        Calculate aggregate confidence score based on OCR quality when available.

        If overall_confidence is already set (from OCR extraction), use it.
        Otherwise, calculate based on field existence as fallback.

        Note: Missing optional fields (e.g., expiry for Singapore NRIC) do not
        reduce confidence score. Only missing required fields affect confidence.

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
        field_weights = {
            # Critical fields - 50% total weight
            'number': 0.15,
            'full_name': 0.20,
            'dob': 0.10,
            'nationality': 0.05,

            # Important fields - 20% total weight
            'sex': 0.05,

            # Supporting fields - 30% total weight
            'document_country': 0.10,
            'place_of_birth': 0.10,
            'issuing_authority': 0.10,
        }

        # Check if we have data for each field
        field_exists = {
            'number': bool(self.number),
            'full_name': bool(self.full_name),
            'dob': bool(self.dob),
            'nationality': bool(self.nationality),
            'sex': bool(self.sex),
            'document_country': bool(self.document_country),
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
        critical_fields = ['number', 'full_name', 'dob']
        missing_critical = sum(1 for f in critical_fields if not field_exists[f])

        if missing_critical > 0:
            # Reduce confidence by 15% for each missing critical field
            penalty = missing_critical * 15.0
            overall = max(0, overall - penalty)

        return round(overall, 2)


# Backward compatibility aliases - map old passport-specific field names to new generic names
class PassportData(BaseModel):
    """Legacy passport-specific schema for backward compatibility.

    Deprecated: Use IdentityDocumentData instead.
    This class is kept for backward compatibility with existing code.
    """

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
        """Calculate aggregate confidence score (legacy method)."""
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


def to_identity_document_data(passport_data: PassportData) -> IdentityDocumentData:
    """Convert legacy PassportData to IdentityDocumentData."""
    return IdentityDocumentData(
        full_name=passport_data.full_name,
        dob=passport_data.date_of_birth,
        sex=passport_data.sex,
        place_of_birth=passport_data.place_of_birth,
        number=passport_data.passport_number,
        nationality=passport_data.nationality,
        document_country=passport_data.passport_country,
        document_type="passport",
        issuing_authority=passport_data.issuing_authority,
        date_of_issue=passport_data.date_of_issue,
        date_of_expiry=passport_data.date_of_expiry,
        overall_confidence=passport_data.overall_confidence,
        field_confidences=passport_data.field_confidences,
    )

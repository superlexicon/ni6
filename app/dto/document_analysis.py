from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, List, Any, Union
import base64
from .detailed_forgery_results import DetailedPhotoHolmesResults


class FileObject(BaseModel):
    """Individual file object with type specification"""
    filename: str
    file_data: str  # Base64 encoded file content
    file_type: str  # "document" or "selfie"
    document_type: Optional[str] = None  # Optional: "passport", "bank_statement", etc.

    # Selfie-specific fields
    secret_share: Optional[str] = None  # Shamir secret share (format: "{index}:{base64}")

    @field_validator('file_data')
    @classmethod
    def validate_file_data(cls, v):
        # Ensure it's a valid Base64 string
        try:
            base64.b64decode(v, validate=True)
        except Exception:
            raise ValueError("file_data must be a valid Base64 encoded string")
        return v

    @field_validator('secret_share')
    @classmethod
    def validate_selfie_fields(cls, v, info):
        if info.data.get('file_type') == 'selfie' and not v:
            field_name = info.field_name
            raise ValueError(f"{field_name} is required when file_type is 'selfie'")
        return v


# Enhanced confidence factors from key injection system
class ConfidenceFactors(BaseModel):
    """Detailed confidence breakdown from key injection analysis."""
    ocr_confidence: float = 0.0              # Original OCR confidence
    spatial_confidence: float = 0.0          # Spatial relationship confidence
    format_confidence: float = 0.0           # Format validation confidence
    key_importance_confidence: float = 0.0   # Key importance weighting
    cross_validation_confidence: float = 0.0 # Cross-validation confidence
    pattern_specificity_confidence: float = 0.0 # Pattern specificity confidence


class ExtractedFieldData(BaseModel):
    """Simplified field data for clean API responses."""
    # Core value and confidence
    value: Optional[str] = None  # The extracted value
    confidence: Optional[float] = None  # Confidence score (0-100)

    # Internal processing fields (not exposed in final response)
    key_name: Optional[str] = Field(None, exclude=True)  # Name of the detected key (for internal processing)
    confidence_factors: Optional[ConfidenceFactors] = Field(None, exclude=True)  # Detailed breakdown (internal only)


class PDFMetadataAnalysis(BaseModel):
    suspicious_indicators: List[str]  # List of suspicious findings
    authenticity_score: float  # Overall PDF authenticity (0-100)
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None
    producer: Optional[str] = None
    creator: Optional[str] = None


class AuthenticityData(BaseModel):
    """
    Simplified forgery detection results with direct detector mappings.

    Returns raw detector scores directly with thresholds.
    Higher raw scores indicate MORE likelihood of forgery.
    """
    # Direct mapping of detector names to their results
    psccnet: Optional[Dict[str, float]] = None  # {raw_score: float, problem_threshold: float}
    dq: Optional[Dict[str, float]] = None       # {raw_score: float, problem_threshold: float}
    adaptive: Optional[Dict[str, float]] = None # {raw_score: float, problem_threshold: float}
    noisesniffer: Optional[Dict[str, float]] = None # {raw_score: float, problem_threshold: float}

    overall_decision: str = "pass"  # "pass", "warning", or "fail"
    failed_detectors: List[str] = []
    warning_detectors: List[str] = []


class BasicSecurityChecks(BaseModel):
    exif_authenticity_score: float
    editing_software_detected: bool
    suspicious_indicators: List[str]
    manipulation_score: Optional[float] = None  # From ELA, only for images


class BankStatementValidation(BaseModel):
    transactions_extracted: int
    balance_validation_passed: bool
    date_sequence_valid: bool
    font_consistency_score: float


class CrossValidationResult(BaseModel):
    """Result of cross-document validation"""
    name_consistency_score: float  # 0-100
    dob_consistency: bool
    address_consistency_score: float  # 0-100
    face_match_score: Optional[float] = None  # 0-100
    otp_verified: bool
    overall_match: bool
    detailed_findings: List[str] = []



class DocumentResult(BaseModel):
    """Analysis result for a single document"""
    filename: str
    document_type: str  # "passport", "bank_statement", "selfie", etc.
    document_type_confidence: float

    # IMDL analysis results (from PhotoHolmes forgery analysis with research context)
    imdl_results: DetailedPhotoHolmesResults

    # Extracted data (schema varies by document type)
    extracted_data: Any  # Union of all schema types

    # Face embedding for biometric matching (when available - excluded from API responses)
    face_embedding: Optional[List[float]] = Field(None, exclude=True)


class MultiDocumentAnalyzeResponse(BaseModel):
    """Response for multi-document analysis"""
    documents: List[DocumentResult]  # Individual document results
    other_validation: Optional[CrossValidationResult] = None  # Includes cross-document and OTP validation
    total_documents: int
    document_types_found: List[str]
    is_recovery_initiation: bool = False  # True if duplicate passport with different public key
    user_identity_id: Optional[str] = None  # User identity ID if user exists or created

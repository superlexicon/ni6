"""
Error codes for document processing.

This module defines comprehensive error codes for document processing,
covering technical errors, OCR failures, schema issues, extraction problems,
tampering detection, and name matching.
"""


class DocumentErrorCode(str):
    """
    Error codes for document processing.

    Error codes are organized by category:
    - TECHNICAL_*: System-level errors (decryption, payload structure)
    - OCR_*: Text extraction failures
    - Schema-related: Missing or unsupported schemas
    - LOGICAL_*: Business logic errors (wrong type, poor quality, tampering)
    - NAME_MISMATCH: Name validation failures
    - PROCESSING_ERROR: Generic processing failures
    """

    # ========== TECHNICAL ERRORS ==========
    TECHNICAL_DECRYPTION_FAILED = "TECHNICAL_DECRYPTION_FAILED"
    TECHNICAL_INVALID_ENVELOPE = "TECHNICAL_INVALID_ENVELOPE"
    TECHNICAL_INVALID_PAYLOAD = "TECHNICAL_INVALID_PAYLOAD"
    TECHNICAL_MISSING_FIELD = "TECHNICAL_MISSING_FIELD"

    # ========== OCR ERRORS ==========
    OCR_FAILED = "OCR_FAILED"
    OCR_INSUFFICIENT_TEXT = "OCR_INSUFFICIENT_TEXT"

    # ========== SCHEMA ERRORS ==========
    NO_SCHEMA = "NO_SCHEMA"
    UNSUPPORTED_DOCUMENT_TYPE = "UNSUPPORTED_DOCUMENT_TYPE"

    # ========== LOGICAL ERRORS - Extraction ==========
    # Wrong document type (less than 50% of required fields extracted)
    LOGICAL_WRONG_DOCUMENT_TYPE = "LOGICAL_WRONG_DOCUMENT_TYPE"

    # Poor quality (100% required not met - at least one required field missing/low confidence)
    LOGICAL_EXTRACTION_INCOMPLETE = "LOGICAL_EXTRACTION_INCOMPLETE"
    LOGICAL_EXTRACTION_LOW_CONFIDENCE = "LOGICAL_EXTRACTION_LOW_CONFIDENCE"

    # ========== LOGICAL ERRORS - Tampering ==========
    # PhotoHolmes forgery detected
    LOGICAL_FORGERY_DETECTED = "LOGICAL_FORGERY_DETECTED"
    LOGICAL_AI_GENERATED_DETECTED = "LOGICAL_AI_GENERATED_DETECTED"
    LOGICAL_MANIPULATION_DETECTED = "LOGICAL_MANIPULATION_DETECTED"

    # ========== NAME MATCHING ERRORS ==========
    LOGICAL_NAME_MISMATCH = "LOGICAL_NAME_MISMATCH"
    NAME_MATCH_FAILED = "NAME_MATCH_FAILED"

    # ========== STATE MANAGEMENT ERRORS ==========
    INVALID_STATE = "INVALID_STATE"
    USER_NOT_FOUND = "USER_NOT_FOUND"

    # ========== VALIDATION ERRORS ==========
    VALIDATION_FAILED = "VALIDATION_FAILED"

    # ========== GENERAL ==========
    PROCESSING_ERROR = "PROCESSING_ERROR"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"

    # ========== SELFIE SPECIFIC ==========
    SELFIE_OTP_NOT_FOUND = "SELFIE_OTP_NOT_FOUND"
    SELFIE_OTP_EXPIRED = "SELFIE_OTP_EXPIRED"
    SELFIE_OTP_ALREADY_VERIFIED = "SELFIE_OTP_ALREADY_VERIFIED"
    SELFIE_OTP_INCORRECT = "SELFIE_OTP_INCORRECT"
    SELFIE_OTP_EXTRACTION_FAILED = "SELFIE_OTP_EXTRACTION_FAILED"
    SELFIE_NO_FACE_DETECTED = "SELFIE_NO_FACE_DETECTED"
    SELFIE_LIVENESS_FAILED = "SELFIE_LIVENESS_FAILED"
    SELFIE_DUPLICATE_FACE = "SELFIE_DUPLICATE_FACE"
    SELFIE_INVALID_VIDEO_FORMAT = "SELFIE_INVALID_VIDEO_FORMAT"

    # ========== BANK STATEMENT SPECIFIC ==========
    BANK_STATEMENT_TOO_OLD = "BANK_STATEMENT_TOO_OLD"
    BANK_STATEMENT_ADDRESS_INCOMPLETE = "BANK_STATEMENT_ADDRESS_INCOMPLETE"
    BANK_STATEMENT_BANK_NOT_RECOGNIZED = "BANK_STATEMENT_BANK_NOT_RECOGNIZED"
    BANK_STATEMENT_ACCOUNT_FORMAT_INVALID = "BANK_STATEMENT_ACCOUNT_FORMAT_INVALID"
    BANK_STATEMENT_ACCOUNT_MASKED = "BANK_STATEMENT_ACCOUNT_MASKED"
    BANK_STATEMENT_CREDIT_CARD_DETECTED = "BANK_STATEMENT_CREDIT_CARD_DETECTED"

    # Legacy aliases for backward compatibility
    NO_SCHEMA_FOUND = "NO_SCHEMA"  # Alias for NO_SCHEMA


# For backward compatibility - can be used as enum-like
class _ErrorCodeContainer:
    """Container for error codes that provides enum-like access."""
    DECRYPTION_FAILED = DocumentErrorCode.TECHNICAL_DECRYPTION_FAILED
    INVALID_ENVELOPE = DocumentErrorCode.TECHNICAL_INVALID_ENVELOPE
    INVALID_PAYLOAD = DocumentErrorCode.TECHNICAL_INVALID_PAYLOAD
    MISSING_REQUIRED_FIELD = DocumentErrorCode.TECHNICAL_MISSING_FIELD


# Legacy string constants for backward compatibility
OCR_FAILED = "OCR_FAILED"
NO_SCHEMA = "NO_SCHEMA"
PROCESSING_ERROR = "PROCESSING_ERROR"


__all__ = [
    "DocumentErrorCode",
    "OCR_FAILED",
    "NO_SCHEMA",
    "PROCESSING_ERROR",
]

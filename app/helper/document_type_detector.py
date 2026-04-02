from typing import Tuple, Optional
from app.core import logger


class DocumentTypeDetector:
    """Validates document type from explicit request parameters.

    Auto-detection has been removed - explicit document types are now required in requests.
    """

    def __init__(self):
        self.logger = logger

    async def detect(self, content: bytes, extracted_text: str, explicit_file_type: Optional[str] = None, explicit_document_type: Optional[str] = None) -> Tuple[str, float]:
        """
        Validate document type from explicit request parameters.
        Auto-detection has been removed - explicit types are now required.

        Args:
            content: Document bytes (unused, kept for compatibility)
            extracted_text: OCR extracted text from document (unused, kept for compatibility)
            explicit_file_type: Required explicit file type ("selfie", "passport", or "bank_statement")
            explicit_document_type: Unused, kept for backward compatibility

        Returns:
            Tuple of (document_type, confidence) - always 100.0 confidence since type is explicit

        Raises:
            ValueError: If required explicit types are not provided
        """
        # REQUIRE explicit file type - no auto-detection
        if not explicit_file_type:
            raise ValueError("file_type is required in request. Must be 'selfie', 'passport', or 'bank_statement'")

        # Validate allowed file types
        allowed_types = ["selfie", "passport", "id_card", "bank_statement", "tax_statement", "add_public_key", "remove_public_key"]
        if explicit_file_type not in allowed_types:
            raise ValueError(f"Invalid file_type: '{explicit_file_type}'. "
                           f"Must be one of: {', '.join(allowed_types)}")

        self.logger.info(f"File explicitly marked as type: {explicit_file_type}")
        return (explicit_file_type, 100.0)

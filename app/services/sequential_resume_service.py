"""
Sequential Resume Service - Handles resume/CV processing.

NEW: Uses DocumentProcessorBase for unified validation pipeline with name-first validation.

Resumes are optional documents that can be submitted after bank statement completion.
They implement name matching against the stored full_name from passport step.

Required fields: full_name, email, phone (minimum contact info)
Optional fields: address, education, work_experience, skills
"""

from typing import Dict, Any, Optional, Tuple, List
import re
from app.services.sequential_document_processor_base import DocumentProcessorBase


class SequentialResumeService(DocumentProcessorBase):
    """Service for handling resume/CV processing using unified framework."""

    def __init__(self):
        super().__init__()
        # Lazy load the resume extractor to avoid import issues
        self._resume_extractor = None

    @property
    def resume_extractor(self):
        """Lazy load the BERT NER resume extractor."""
        if self._resume_extractor is None:
            from app.helper.extractors.bert_ner_resume_extractor import BertNerResumeExtractor
            self._resume_extractor = BertNerResumeExtractor()
        return self._resume_extractor

    # ============================================================
    # ABSTRACT METHOD IMPLEMENTATIONS
    # ============================================================

    def get_document_type(self) -> str:
        return "resume"

    def get_required_fields(self) -> List[str]:
        return [
            'full_name',
            'email',
            'phone'
        ]

    def get_optional_fields(self) -> List[str]:
        return [
            'address',
            'education',
            'work_experience',
            'skills'
        ]

    def get_name_field(self) -> str:
        """Resume uses 'full_name' as the name field."""
        return 'full_name'

    def get_photoholmes_document_type(self) -> str:
        return "resume"

    def should_validate_photoholmes(self) -> bool:
        """Resumes typically don't need PhotoHolmes forgery detection."""
        return False

    def extract_fields_from_ocr(
        self, text_blocks: list, raw_text: str, image_bytes: bytes, is_pdf: bool
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Extract resume fields using BERT NER resume extractor.

        Returns:
            (extracted_data, confidence_data)
        """
        import asyncio

        # Run the async extractor in sync context
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            resume_data = loop.run_until_complete(
                self.resume_extractor.extract(image_bytes, is_pdf)
            )
        except Exception as e:
            self.logger.warning(f"Resume extraction failed: {e}")
            # Fall back to regex-based extraction
            return self._extract_with_regex(raw_text)

        extracted_data = self._build_extracted_data(resume_data)

        # Build confidence_data based on extraction results
        confidence_data = {}
        for field in self.get_required_fields() + self.get_optional_fields():
            value = extracted_data.get(field)
            if value:
                confidence_data[field] = {
                    'overall_confidence': 0.75,  # Moderate confidence for NER extraction
                    'sources': ['bert_ner']
                }
            else:
                confidence_data[field] = {
                    'overall_confidence': 0.0,
                    'sources': []
                }

        return extracted_data, confidence_data

    def perform_document_specific_validations(
        self, extracted_data: Dict[str, Any], user_identity: Dict[str, Any]
    ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Perform resume specific validations.

        Validations:
        - Email format validation
        - Phone number format validation
        """
        # Validate email format
        email = extracted_data.get('email')
        email_valid = False
        if email:
            email_valid = self._validate_email(email)

        # Validate phone format
        phone = extracted_data.get('phone')
        phone_valid = False
        if phone:
            phone_valid = self._validate_phone(phone)

        additional_checks = {
            'email_valid': email_valid,
            'phone_valid': phone_valid,
            'document_type': 'resume'
        }

        if not email_valid:
            return False, "Invalid email format", additional_checks

        if not phone_valid:
            return False, "Invalid phone number format", additional_checks

        return True, None, additional_checks

    def should_increment_state(self) -> bool:
        """Resume is optional - no state increment."""
        return False

    # ============================================================
    # RESUME SPECIFIC METHODS
    # ============================================================

    def _build_extracted_data(self, resume_data) -> Dict[str, Any]:
        """Build extracted_data dict from BertNerResumeExtractor result."""
        if not resume_data:
            return {}

        return {
            "full_name": getattr(resume_data, 'full_name', None),
            "email": getattr(resume_data, 'email', None),
            "phone": getattr(resume_data, 'phone', None),
            "address": getattr(resume_data, 'address', None),
            "education": getattr(resume_data, 'education', None),
            "work_experience": getattr(resume_data, 'work_experience', None),
            "skills": getattr(resume_data, 'skills', None),
        }

    def _extract_with_regex(self, raw_text: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Fallback regex-based extraction for resumes."""
        extracted_data = {}

        # Extract email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, raw_text)
        if emails:
            extracted_data['email'] = emails[0]

        # Extract phone (basic patterns)
        phone_patterns = [
            r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',  # US format
            r'\+?\d{1,3}[-.\s]?\d{3,4}[-.\s]?\d{3,4}[-.\s]?\d{3,4}',  # International
        ]
        for pattern in phone_patterns:
            phones = re.findall(pattern, raw_text)
            if phones:
                extracted_data['phone'] = phones[0]
                break

        confidence_data = {}
        for field in self.get_required_fields():
            value = extracted_data.get(field)
            if value:
                confidence_data[field] = {
                    'overall_confidence': 0.60,  # Lower confidence for regex
                    'sources': ['regex']
                }
            else:
                confidence_data[field] = {
                    'overall_confidence': 0.0,
                    'sources': []
                }

        return extracted_data, confidence_data

    def _validate_email(self, email: str) -> bool:
        """Validate email format."""
        email_pattern = r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$'
        return bool(re.match(email_pattern, email))

    def _validate_phone(self, phone: str) -> bool:
        """Validate phone number format (basic check)."""
        # Remove common separators
        clean_phone = re.sub(r'[\s\-\.\(\)]', '', phone)

        # Check if it's at least 10 digits
        if len(clean_phone) < 10:
            return False

        # Check if it contains mostly digits
        digits = sum(c.isdigit() for c in clean_phone)
        return digits >= 10

    # ============================================================
    # PUBLIC ENTRY POINT
    # ============================================================

    async def process_resume(
        self, client_public_key: str, file_data: str, filename: str,
        iv: str, callback_url: Optional[str] = None
    ):
        """
        Process a resume/CV document.

        This is the public entry point that calls the base class pipeline.
        """
        return await self.process_document(
            client_public_key=client_public_key,
            file_data=file_data,
            filename=filename,
            iv=iv,
            callback_url=callback_url
        )

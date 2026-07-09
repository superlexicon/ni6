"""
Qwen3-VL Universal ID Document Extractor

Extracts ID document fields for supplementary information only.
Does NOT affect verification state or sequence.

Supports all ID document types across all countries:
- National ID cards (NRIC, Emirates ID, etc.)
- Driving licenses
- PAN cards
- Any other ID document

Universal extraction in a single pass:
- issuing_country
- id_type (determined naturally by Qwen from the document)
- id_number
- full_name
- expiry_date (optional)
"""

import json
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

from app.core.logger import get_logger
from app.services.llm_service import LLMService


logger = get_logger()


@dataclass
class ExtractionResult:
    """Result of universal ID extraction."""
    extracted_data: Dict[str, Any]
    confidence_data: Dict[str, float]
    extraction_method: str = "qwen3_vl_universal_id"
    raw_response: Optional[str] = None


class QwenUniversalIDExtractor:
    """
    Universal ID document extraction using Qwen3-VL vision model.

    Extracts universal ID fields from ANY ID document regardless of country:
    - issuing_country: ISO country code or full name
    - id_type: Document type as shown on the document (e.g., "National ID Card", "Driving License", "PAN Card", etc.)
    - id_number: The main identification number
    - full_name: Full name of the ID holder
    - expiry_date: Expiry date if available (null for permanent IDs)

    This is for SUPPLEMENTARY information only and does NOT affect verification.
    """

    # Required fields that must be extracted
    REQUIRED_FIELDS = [
        "issuing_country",
        "id_type",
        "id_number",
        "full_name"
    ]

    # Optional fields
    OPTIONAL_FIELDS = [
        "expiry_date"
    ]

    def __init__(self):
        """Initialize the universal ID extractor."""
        self.llm_service = LLMService()

    def _build_system_prompt(self) -> str:
        """Build system prompt for universal ID extraction."""
        return """You are an expert at extracting information from ID documents (national ID cards, driving licenses, etc.) from any country.

Extract the following fields from the ID document image:
1. issuing_country: The country that issued this document (ISO code like "SG", "US", "IN" or full name)
2. id_type: The type of ID document as shown on the document (e.g., "National ID Card", "Driving License", "PAN Card", "NRIC", etc.)
3. id_number: The main identification number on the document
4. full_name: Full name of the ID holder (as printed on the document)
5. expiry_date: Expiry date if present (YYYY-MM-DD format), null if document has no expiry

Rules:
- Extract ONLY the fields listed above
- For expiry_date, return null if the document doesn't have an expiry date
- Return valid JSON only, no explanations"""

    def _build_user_prompt(self) -> str:
        """Build user prompt for universal ID extraction."""
        return """Extract the ID document information from this image.

Return JSON with these exact keys:
{
    "issuing_country": "country code or name",
    "id_type": "document type as shown on the document",
    "id_number": "the ID number",
    "full_name": "full name as shown on document",
    "expiry_date": "YYYY-MM-DD or null"
}

IMPORTANT: Return ONLY valid JSON, no other text."""

    async def extract_from_image(
        self,
        image_bytes: bytes,
        max_retries: int = 3
    ) -> ExtractionResult:
        """Extract universal ID fields from image using Qwen3-VL."""
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt()

        for attempt in range(max_retries):
            try:
                response = await self.llm_service.call_vision_llm(
                    image_bytes=image_bytes,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt
                )

                # Get content from response
                content = response.get("content", "")

                if not content:
                    logger.warning(f"Attempt {attempt + 1}: Empty response content")
                    if attempt == max_retries - 1:
                        raise ValueError("Empty response from vision LLM")
                    continue

                # Parse JSON response
                try:
                    extracted_data = json.loads(content)
                except json.JSONDecodeError as e:
                    logger.warning(f"Attempt {attempt + 1}: Failed to parse JSON: {e}")
                    logger.debug(f"Response content: {content[:200]}")
                    if attempt == max_retries - 1:
                        raise ValueError(f"Failed to parse JSON response after {max_retries} attempts")
                    continue

                # Validate that we have a dictionary
                if not isinstance(extracted_data, dict):
                    logger.warning(f"Attempt {attempt + 1}: Response is not a dictionary: {type(extracted_data)}")
                    if attempt == max_retries - 1:
                        raise ValueError("Vision LLM returned non-dictionary response")
                    continue

                # Add confidence scores (1.0 for vision LLM)
                extracted_with_confidence = {
                    k: {"value": v, "confidence": 1.0, "source": "vision_llm"}
                    for k, v in extracted_data.items()
                }

                return ExtractionResult(
                    extracted_data=extracted_with_confidence,
                    confidence_data={k: 1.0 for k in extracted_data.keys()},
                    extraction_method="qwen3_vl_universal_id",
                    raw_response=content
                )

            except ValueError:
                # Re-raise ValueError with custom messages
                raise
            except Exception as e:
                logger.error(f"Extraction attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise ValueError(f"Failed to extract ID data after {max_retries} attempts: {str(e)}")

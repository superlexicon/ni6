"""
Qwen3-VL Direct Generic Document Classifier and PII Extractor

Classifies document type and extracts all PII (personally identifiable information)
directly from image using Qwen3-VL vision model.
No intermediate OCR, layout caching, or spatial coordinate extraction needed.

Qwen3-VL extracts document type and field values directly from the image.
Confidence scores are added post-extraction for backwards compatibility (1.0).
"""

import json
import re
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime

from app.core.logger import get_logger
from app.services.llm_service import LLMService


logger = get_logger()


@dataclass
class ExtractionResult:
    """Result of direct field extraction with confidence scores."""
    extracted_data: Dict[str, Any]  # Field values with metadata
    confidence_data: Dict[str, float]  # Per-field confidence scores
    extraction_method: str = "qwen3_vl_direct"
    raw_response: Optional[str] = None


class QwenGenericDocumentExtractor:
    """
    Direct generic document classification and PII extraction using Qwen3-VL vision model.

    This extractor handles documents of unknown or unspecified type.
    It performs two tasks:
    1. Classifies the document type from a comprehensive list
    2. Extracts all PII fields regardless of document type

    Flow:
    1. Qwen3-VL analyzes the document image
    2. Returns document_type + all applicable PII fields
    3. Post-processing: Add confidence scores (1.0) for backwards compatibility
    4. Normalize dates and clean names
    5. Return structured data with confidence scores

    JSON Output Format (from LLM):
    {
        "document_type": "passport",
        "document_subtype": "ordinary",
        "issuing_country": "United States",
        "full_name": "JOHN DOE",
        "date_of_birth": "1990-05-15",
        "document_number": "A12345678",
        "passport_number": "A12345678",
        "expiry_date": "2025-12-31",
        "issue_date": "2020-01-15",
        "address": "123 Main St, City, State",
        "nationality": "United States",
        "phone": "+1-234-567-8900",
        "email": "john.doe@example.com",
        "account_number": "123456789",
        "tax_id": "12-3456789",
        "employer": "Company Inc",
        "income": "75000",
        "other_identifiers": ["SSN: 123-45-6789"]
    }

    Post-processed Format (with confidence added):
    {
        "document_type": {"value": "passport", "confidence": 1.0, "source": "vision_llm"},
        "full_name": {"value": "JOHN DOE", "confidence": 1.0, "source": "vision_llm"},
        ...
    }
    """

    # Required fields that must be extracted
    REQUIRED_FIELDS = [
        "document_type"
    ]

    # All supported PII fields (optional - may not be present in all documents)
    PII_FIELDS = [
        "document_subtype",
        "issuing_country",
        "full_name",
        "date_of_birth",
        "document_number",
        "passport_number",
        "id_number",
        "nric_fin_number",
        "pan_number",
        "expiry_date",
        "issue_date",
        "valid_until",
        "valid_from",
        "address",
        "nationality",
        "sex",
        "phone",
        "email",
        "account_number",
        "tax_id",
        "employer",
        "income",
        "certificate_number",
        "application_number",
        "other_identifiers"
    ]

    # Supported document types for classification
    DOCUMENT_TYPES = [
        "passport",
        "id_card",
        "nric",
        "pan_card",
        "bank_statement",
        "tax_return",
        "tax_residency_certificate",
        "driving_license",
        "utility_bill",
        "employment_letter",
        "residence_permit",
        "visa",
        "social_security_card",
        "other"
    ]

    def __init__(self):
        """Initialize the Qwen3-VL generic extractor."""
        self.llm_service = LLMService()

    async def extract_fields(
        self,
        image_bytes: bytes,
        max_retries: int = 2
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Classify document type and extract all PII fields directly from image using Qwen3-VL.

        Args:
            image_bytes: JPEG image data (already preprocessed to meet Qwen3-VL requirements)
            max_retries: Maximum number of extraction attempts with different strategies

        Returns:
            Tuple of (extracted_data, confidence_data):
            - extracted_data: Dict with field values, structured as:
                {
                    "field_name": {"value": "...", "confidence": 0.9, "source": "..."}
                }
            - confidence_data: Dict with per-field confidence scores
        """
        logger.info("Starting Qwen3-VL direct generic document classification and PII extraction")

        # Try multiple extraction strategies
        for attempt in range(max_retries):
            try:
                # Build prompts for direct extraction
                system_prompt = self._build_system_prompt()
                user_prompt = self._build_user_prompt()

                # Call vision LLM
                response = await self.llm_service.call_vision_llm(
                    image_bytes=image_bytes,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=0.3,
                    max_tokens=8000
                )

                if "error" in response:
                    logger.error(f"Vision LLM extraction failed (attempt {attempt + 1}/{max_retries}): {response['error']}")
                    if attempt == max_retries - 1:
                        return {}, {"error": response['error']}
                    continue

                # Try to parse the response content
                content = response.get("content", "")

                # If content is empty, check if there's thinking that might contain useful info
                if not content and "thinking" in response:
                    thinking = response.get("thinking", "")
                    logger.warning(f"Content empty, thinking contains: {thinking[:200]}...")

                    # Try to extract JSON from thinking field
                    if "{" in thinking and "}" in thinking:
                        json_start = thinking.rfind("{")
                        json_end = thinking.rfind("}")
                        if json_start >= 0 and json_end > json_start:
                            potential_json = thinking[json_start:json_end + 1]
                            try:
                                json.loads(potential_json)
                                logger.info(f"Successfully extracted JSON from thinking field ({len(potential_json)} chars)")
                                content = potential_json
                            except json.JSONDecodeError:
                                logger.error(f"Extracted text from thinking is not valid JSON")
                                if attempt == max_retries - 1:
                                    return {}, {"error": "Model does not support valid JSON output. Try qwen2.5-vl:7b model."}
                                continue
                        else:
                            logger.error(f"Could not find complete JSON object in thinking")
                            if attempt == max_retries - 1:
                                return {}, {"error": "Model does not support JSON output. Try qwen2.5-vl:7b model."}
                            continue
                    else:
                        logger.error(f"Model returned reasoning without JSON output (attempt {attempt + 1}/{max_retries})")
                        if attempt == max_retries - 1:
                            return {}, {"error": "Model does not support JSON output. Try qwen2.5-vl:7b model."}
                        continue

                # Parse the LLM response
                extracted_data = self._parse_llm_response(content)

                # If we got here, extraction succeeded
                break

            except Exception as e:
                logger.error(f"Extraction attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                if attempt == max_retries - 1:
                    return {}, {"error": f"Parse error: {str(e)}"}
                continue
        else:
            # All retries exhausted
            return {}, {"error": "All extraction attempts failed"}

        # Post-processing: Normalize document_type to lowercase
        self._normalize_document_type(extracted_data)

        # Post-processing: Map document-specific fields to generic ones
        self._map_document_specific_fields(extracted_data)

        # Post-processing: Clean full_name
        self._clean_full_name(extracted_data)

        # Post-processing: Normalize dates to ISO format
        self._normalize_all_dates(extracted_data)

        # Post-processing: Normalize sex to single letter
        self._normalize_sex(extracted_data)

        # Post-processing: Process other_identifiers array
        self._process_other_identifiers(extracted_data)

        # Compute confidence scores from extracted data
        confidence_data = self._compute_confidence_scores(extracted_data)

        doc_type = extracted_data.get("document_type", {}).get("value", "unknown")
        logger.info(
            f"Qwen3-VL generic document extraction completed: "
            f"document_type={doc_type}, "
            f"{len(extracted_data)} fields extracted, "
            f"avg confidence: {sum(confidence_data.values()) / len(confidence_data) if confidence_data else 0:.2f}"
        )

        return extracted_data, confidence_data

    def _build_system_prompt(self) -> str:
        """
        Build system prompt for Qwen3-VL direct generic document extraction.

        CRITICAL: Must enforce JSON-only output to avoid reasoning mode.
        """
        return """You are a document analysis expert. Analyze the provided image and:

1. CLASSIFY the document type
2. EXTRACT all personally identifiable information (PII)

Return ONLY a JSON object. No explanations, no thinking, no markdown code blocks.

Document types to classify:
- passport: Travel passport document
- id_card: National ID card, identity card
- nric: Singapore NRIC/FIN card
- pan_card: Indian PAN card
- bank_statement: Bank account statement
- tax_return: Tax return, tax statement
- tax_residency_certificate: Tax residency certificate (TRC)
- driving_license: Driver's license
- utility_bill: Utility service bill (water, electricity, gas, internet)
- employment_letter: Employment verification letter, salary certificate
- residence_permit: Residence or work permit
- visa: Visa document
- social_security_card: Social security card, national insurance card
- other: Any other document type

PII Fields to Extract (use null if not present):
- document_type: The classified document type (REQUIRED)
- document_subtype: More specific type if applicable (e.g., "ordinary", "diplomatic" for passports)
- issuing_country: Country that issued the document (full name or ISO code)
- full_name: Person's full legal name
- date_of_birth: Date of birth (YYYY-MM-DD)
- document_number: Any document identifier number
- passport_number: Passport number if present
- id_number: National ID number if present
- nric_fin_number: Singapore NRIC/FIN number
- pan_number: Indian PAN number
- expiry_date: Document expiry date (YYYY-MM-DD)
- issue_date: Document issue date (YYYY-MM-DD)
- valid_until: Valid until date (YYYY-MM-DD)
- valid_from: Valid from date (YYYY-MM-DD)
- address: Residential or postal address
- nationality: Person's nationality
- sex: Gender (M/F)
- phone: Phone number
- email: Email address
- account_number: Bank account number
- tax_id: Tax identification number
- employer: Employer or company name
- income: Income or salary amount (with currency if available)
- certificate_number: Certificate number (for TRC, etc.)
- application_number: Application number
- other_identifiers: Array of any other identifying numbers/codes found

CRITICAL RULES:
1. Return ONLY JSON - no text, no markdown, no code blocks
2. document_type is REQUIRED - must always be present
3. All dates in YYYY-MM-DD format
4. Remove salutations from names (Mr., Mrs., Dr., Shri, etc.)
5. For unclear or missing fields, use null rather than guessing
6. Extract as many PII fields as are present in the document

Name Cleaning:
- Remove ALL salutations, titles, and prefixes
- Examples: "Mr. JOHN DOE" -> "JOHN DOE", "Shri RAJ KUMAR" -> "RAJ KUMAR"

Date Normalization:
- All dates must be in ISO format: YYYY-MM-DD
- Common date formats to handle:
  - DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY
  - YYYY/MM/DD, YYYY-MM-DD, YYYY.MM.DD
  - D MMM YYYY (e.g., "15 Jan 1990")
  - MMM D, YYYY (e.g., "Jan 15, 1990")"""

    def _build_user_prompt(self) -> str:
        """
        Build user prompt with extraction instructions.

        CRITICAL: Prepend /no_think to bypass Qwen3's CoT generation.

        Returns:
            User prompt string
        """
        prompt_parts = [
            "/no_think",  # Qwen3 trigger to skip thinking block generation
            "Analyze this document image.",
            "Classify the document type and extract all PII (personally identifiable information).",
            "Return JSON with document_type and all applicable PII fields.",
            "Use null for missing fields."
        ]

        return " ".join(prompt_parts)

    def _parse_llm_response(self, response_content: str) -> Dict[str, Any]:
        """
        Parse Qwen3-VL JSON response into structured extracted data.

        Handles multiple response formats:
        1. Direct JSON content
        2. JSON in markdown code blocks
        3. JSON embedded in reasoning text (fallback)

        Args:
            response_content: Raw JSON string from LLM

        Returns:
            Parsed extracted data dictionary

        Raises:
            ValueError: If JSON is invalid or doesn't match expected format
        """
        # Clean up response content
        content = response_content.strip()

        # Strategy 1: Extract JSON from markdown code blocks
        if "```json" in content:
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)
                logger.debug("Extracted JSON from ```json code block")
        elif "```" in content:
            json_match = re.search(r'```\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)
                logger.debug("Extracted JSON from ``` code block")

        # Strategy 2: Find JSON object boundaries
        if not content.startswith("{"):
            first_brace = content.find("{")
            last_brace = content.rfind("}")
            if first_brace >= 0 and last_brace > first_brace:
                content = content[first_brace:last_brace + 1]
                logger.debug("Extracted JSON using brace boundaries")
            else:
                raise ValueError(f"Response does not contain valid JSON: {content[:200]}...")

        # Parse JSON
        try:
            parsed = json.loads(content)
            logger.debug(f"Successfully parsed JSON with {len(parsed)} fields")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}\nContent: {content[:500]}")

            # Final fallback: Try to extract any JSON-like structure
            if '{' in content and '}' in content:
                try:
                    first_brace = content.find('{')
                    last_brace = content.rfind('}')
                    if first_brace >= 0 and last_brace > first_brace:
                        json_candidate = content[first_brace:last_brace + 1]
                        parsed = json.loads(json_candidate)
                        logger.debug("Successfully parsed JSON using fallback extraction")
                    else:
                        raise ValueError("No valid JSON object found")
                except:
                    raise ValueError(f"Invalid JSON: {str(e)}")
            else:
                raise ValueError(f"Invalid JSON: {str(e)}")

        # Validate structure - should have field entries
        if not isinstance(parsed, dict):
            raise ValueError(f"Response is not a JSON object: {type(parsed)}")

        # Validate required fields
        if "document_type" not in parsed:
            logger.warning("Response missing required field: document_type")
            # Add with default value
            parsed["document_type"] = "other"

        # Normalize structure - ensure each field has value/confidence/source
        extracted_data = {}
        for field_name, field_data in parsed.items():
            if isinstance(field_data, dict):
                # Handle nested format: {"value": ..., "confidence": ..., "source": ...}
                value = field_data.get("value")
                extracted_data[field_name] = {
                    "value": value,
                    "confidence": 1.0,
                    "source": field_data.get("source", "vision_llm")
                }
            elif field_data is None:
                extracted_data[field_name] = {
                    "value": None,
                    "confidence": 0.0,
                    "source": "vision_llm"
                }
            elif isinstance(field_data, list) and field_name == "other_identifiers":
                # Handle array field for other_identifiers
                extracted_data[field_name] = {
                    "value": field_data,
                    "confidence": 1.0,
                    "source": "vision_llm"
                }
            else:
                # Simple value - wrap in structure with confidence: 1.0
                extracted_data[field_name] = {
                    "value": field_data,
                    "confidence": 1.0,
                    "source": "vision_llm"
                }

        return extracted_data

    def _normalize_document_type(self, extracted_data: Dict[str, Any]) -> None:
        """
        Normalize document_type to lowercase and validate against supported types.

        Args:
            extracted_data: Extracted data dictionary (modified in place)
        """
        if "document_type" in extracted_data:
            value = extracted_data["document_type"]["value"]
            if value and isinstance(value, str):
                # Normalize to lowercase and replace spaces with underscores
                normalized = value.lower().strip().replace(" ", "_").replace("-", "_")

                # Check if it's a valid document type
                if normalized not in self.DOCUMENT_TYPES:
                    logger.warning(f"Unknown document type '{normalized}', mapping to 'other'")
                    normalized = "other"

                extracted_data["document_type"]["value"] = normalized

    def _map_document_specific_fields(self, extracted_data: Dict[str, Any]) -> None:
        """
        Map document-specific fields to generic ones.

        For example, if document_type is passport and we have passport_number,
        also map it to document_number for generic access.

        Args:
            extracted_data: Extracted data dictionary (modified in place)
        """
        doc_type = extracted_data.get("document_type", {}).get("value", "")

        # Map passport_number to document_number for passports
        if doc_type == "passport" and "passport_number" in extracted_data:
            passport_number = extracted_data["passport_number"]["value"]
            if passport_number and "document_number" not in extracted_data:
                extracted_data["document_number"] = {
                    "value": passport_number,
                    "confidence": 1.0,
                    "source": "vision_llm"
                }

        # Map id_number to document_number for ID cards
        if doc_type in ["id_card", "nric"] and "id_number" in extracted_data:
            id_number = extracted_data["id_number"]["value"]
            if id_number and "document_number" not in extracted_data:
                extracted_data["document_number"] = {
                    "value": id_number,
                    "confidence": 1.0,
                    "source": "vision_llm"
                }

        # Map pan_number to document_number for PAN cards
        if doc_type == "pan_card" and "pan_number" in extracted_data:
            pan_number = extracted_data["pan_number"]["value"]
            if pan_number and "document_number" not in extracted_data:
                extracted_data["document_number"] = {
                    "value": pan_number,
                    "confidence": 1.0,
                    "source": "vision_llm"
                }

        # Map nric_fin_number to id_number for NRIC
        if doc_type == "nric" and "nric_fin_number" in extracted_data:
            nric_number = extracted_data["nric_fin_number"]["value"]
            if nric_number and "id_number" not in extracted_data:
                extracted_data["id_number"] = {
                    "value": nric_number,
                    "confidence": 1.0,
                    "source": "vision_llm"
                }

        # Map certificate_number to document_number for TRC
        if doc_type == "tax_residency_certificate" and "certificate_number" in extracted_data:
            cert_number = extracted_data["certificate_number"]["value"]
            if cert_number and "document_number" not in extracted_data:
                extracted_data["document_number"] = {
                    "value": cert_number,
                    "confidence": 1.0,
                    "source": "vision_llm"
                }

        # Map valid_until to expiry_date
        if "valid_until" in extracted_data and "expiry_date" not in extracted_data:
            valid_until = extracted_data["valid_until"]["value"]
            if valid_until:
                extracted_data["expiry_date"] = {
                    "value": valid_until,
                    "confidence": extracted_data["valid_until"]["confidence"],
                    "source": "vision_llm"
                }

        # Map valid_from to issue_date
        if "valid_from" in extracted_data and "issue_date" not in extracted_data:
            valid_from = extracted_data["valid_from"]["value"]
            if valid_from:
                extracted_data["issue_date"] = {
                    "value": valid_from,
                    "confidence": extracted_data["valid_from"]["confidence"],
                    "source": "vision_llm"
                }

    def _clean_full_name(self, extracted_data: Dict[str, Any]) -> None:
        """
        Clean full name by removing salutations and titles.

        Args:
            extracted_data: Extracted data dictionary (modified in place)
        """
        if "full_name" in extracted_data:
            value = extracted_data["full_name"]["value"]
            if value and isinstance(value, str):
                cleaned = self._remove_salutations(value)
                extracted_data["full_name"]["value"] = cleaned

    def _normalize_sex(self, extracted_data: Dict[str, Any]) -> None:
        """
        Normalize sex/gender to single letter (M/F).

        Args:
            extracted_data: Extracted data dictionary (modified in place)
        """
        if "sex" in extracted_data:
            value = extracted_data["sex"]["value"]
            if value and isinstance(value, str):
                # Normalize to single letter
                value_lower = value.lower().strip()

                # Male variations
                if value_lower in ['m', 'male', 'm ale']:
                    extracted_data["sex"]["value"] = "M"
                # Female variations
                elif value_lower in ['f', 'female', 'f emale']:
                    extracted_data["sex"]["value"] = "F"
                else:
                    # Try to extract first letter if it's M or F
                    if value_lower and value_lower[0] in ['m', 'f']:
                        extracted_data["sex"]["value"] = value_lower[0].upper()
                    else:
                        logger.warning(f"Could not normalize sex value: {value}")

    def _normalize_all_dates(self, extracted_data: Dict[str, Any]) -> None:
        """
        Normalize all date fields to ISO format (YYYY-MM-DD).

        Args:
            extracted_data: Extracted data dictionary (modified in place)
        """
        date_fields = [
            "date_of_birth", "expiry_date", "issue_date",
            "valid_until", "valid_from"
        ]

        for date_field in date_fields:
            if date_field in extracted_data:
                value = extracted_data[date_field]["value"]
                if value and isinstance(value, str):
                    normalized = self._normalize_date(value)
                    if normalized:
                        extracted_data[date_field]["value"] = normalized
                    else:
                        logger.debug(f"Could not normalize {date_field}: {value}")

    def _process_other_identifiers(self, extracted_data: Dict[str, Any]) -> None:
        """
        Process other_identifiers array - ensure it's a valid list.

        Args:
            extracted_data: Extracted data dictionary (modified in place)
        """
        if "other_identifiers" in extracted_data:
            value = extracted_data["other_identifiers"]["value"]
            if value is None:
                # Ensure empty list instead of null
                extracted_data["other_identifiers"]["value"] = []
            elif isinstance(value, str):
                # If it's a string, convert to single-element list
                extracted_data["other_identifiers"]["value"] = [value]
            elif not isinstance(value, list):
                # Ensure it's a list
                logger.warning(f"other_identifiers is not a list: {type(value)}")
                extracted_data["other_identifiers"]["value"] = []

    def _remove_salutations(self, name: str) -> str:
        """
        Remove salutations and titles from name.

        Args:
            name: Raw name that may contain titles

        Returns:
            Name with titles removed
        """
        if not name:
            return name

        # Common salutations/titles to remove (with and without dots)
        titles_pattern = r'\b(?:MR|MRS|MS|MISS|DR|PROF|SIR|MADAM|SHRI|SMT|KUM|MM|SH|MME|HERR|FRAU|SIGNOR|SIGNORA)\.?\s*'

        # Remove titles from beginning of string
        cleaned = re.sub(f'^{titles_pattern}', '', name, flags=re.IGNORECASE)

        # Strip any remaining leading/trailing whitespace and dots
        cleaned = cleaned.strip().strip('.')

        return cleaned

    def _normalize_date(self, date_str: str) -> Optional[str]:
        """
        Normalize date string to ISO format (YYYY-MM-DD).

        Handles various date formats:
        - DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY
        - YYYY/MM/DD, YYYY-MM-DD, YYYY.MM.DD
        - D MMM YYYY (e.g., "15 Jan 1990")
        - MMM D, YYYY (e.g., "Jan 15, 1990")

        Args:
            date_str: Date string in any format

        Returns:
            Date in YYYY-MM-DD format, or None if parsing fails
        """
        if not date_str:
            return None

        date_str = str(date_str).strip()

        # Try to use the existing date parser utility
        try:
            from app.utils.date_parser import parse_date_to_mariadb
            parsed_date = parse_date_to_mariadb(date_str)
            if parsed_date:
                # Format as YYYY-MM-DD
                return parsed_date.strftime('%Y-%m-%d')
        except Exception as e:
            logger.debug(f"Date parser failed for {date_str}: {e}")

        # Fallback: Manual parsing for common formats
        # DD/MM/YYYY or DD-MM-YYYY
        match = re.match(r'^(\d{1,2})[/-](\d{1,2})[/-](\d{4})$', date_str)
        if match:
            d, m, y = match.groups()
            return f"{y}-{m.zfill(2)}-{d.zfill(2)}"

        # YYYY/MM/DD or YYYY-MM-DD
        match = re.match(r'^(\d{4})[/-](\d{1,2})[/-](\d{1,2})$', date_str)
        if match:
            y, m, d = match.groups()
            return f"{y}-{m.zfill(2)}-{d.zfill(2)}"

        return None

    def _compute_confidence_scores(self, extracted_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute per-field confidence scores from extracted data.

        Args:
            extracted_data: Extracted data with confidence metadata

        Returns:
            Dictionary mapping field names to confidence scores (0.0-1.0)
        """
        confidence_scores = {}

        for field_name, field_data in extracted_data.items():
            if isinstance(field_data, dict):
                conf = field_data.get("confidence", 0.0)
                value = field_data.get("value")

                # Adjust confidence based on value presence
                if value is None or value == "":
                    conf = 0.0
                elif isinstance(value, str) and len(value.strip()) == 0:
                    conf = 0.0
                elif isinstance(value, list) and len(value) == 0:
                    conf = 0.0

                confidence_scores[field_name] = float(conf)

        return confidence_scores


# Singleton instance
_instance = None


def get_qwen_generic_document_extractor() -> QwenGenericDocumentExtractor:
    """Get the singleton Qwen3-VL generic document extractor instance."""
    global _instance
    if _instance is None:
        _instance = QwenGenericDocumentExtractor()
    return _instance

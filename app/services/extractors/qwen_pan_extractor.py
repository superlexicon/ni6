"""
Qwen3-VL Direct Indian PAN Card Extractor

Extracts Indian PAN Card fields directly from image using Qwen3-VL vision model.
No intermediate OCR, layout caching, or spatial coordinate extraction needed.

Qwen3-VL extracts field values directly from the image.
Confidence scores are added post-extraction for backwards compatibility (1.0).
"""

import json
import re
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

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


class QwenPANExtractor:
    """
    Direct Indian PAN Card extraction using Qwen3-VL vision model.

    Replaces the SpatialPANExtractor (DocTR OCR + spatial coordinate extraction)
    with a single vision LLM call that extracts field values directly.

    Flow:
    1. Qwen3-VL extracts all PAN card fields from image
    2. Post-processing: Add confidence scores (1.0) for backwards compatibility
    3. Validate PAN number format (AAAAA9999A)
    4. Ensure name has at least 2 words
    5. Normalize DOB to ISO format
    6. Return structured data with confidence scores

    JSON Output Format (from LLM):
    {
        "pan_number": "ABCDE1234F",
        "full_name": "RAKESH KUMAR SHARMA",
        "date_of_birth": "1990-05-15"
    }

    Post-processed Format (with confidence added):
    {
        "pan_number": {"value": "ABCDE1234F", "confidence": 1.0, "source": "vision_llm"},
        "full_name": {"value": "RAKESH KUMAR SHARMA", "confidence": 1.0, "source": "vision_llm"},
        ...
    }
    """

    # Required fields that must be extracted
    REQUIRED_FIELDS = [
        "pan_number",
        "full_name"
    ]

    # Optional fields that may or may not be present
    OPTIONAL_FIELDS = [
        "date_of_birth"
    ]

    def __init__(self):
        """Initialize the Qwen3-VL extractor."""
        self.llm_service = LLMService()

    async def extract_fields(
        self,
        image_bytes: bytes,
        max_retries: int = 2
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Extract Indian PAN Card fields directly from image using Qwen3-VL.

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
        logger.info("Starting Qwen3-VL direct Indian PAN Card extraction")

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

        # Post-processing: Validate PAN number format
        self._validate_pan_number(extracted_data)

        # Post-processing: Clean and validate full_name
        self._clean_full_name(extracted_data)

        # Post-processing: Normalize DOB to ISO format
        self._normalize_dob(extracted_data)

        # Compute confidence scores from extracted data
        confidence_data = self._compute_confidence_scores(extracted_data)

        logger.info(
            f"Qwen3-VL PAN Card extraction completed: "
            f"{len(extracted_data)} fields extracted, "
            f"avg confidence: {sum(confidence_data.values()) / len(confidence_data) if confidence_data else 0:.2f}"
        )

        return extracted_data, confidence_data

    def _build_system_prompt(self) -> str:
        """
        Build system prompt for Qwen3-VL direct Indian PAN Card extraction.

        CRITICAL: Must enforce JSON-only output to avoid reasoning mode.
        """
        return """Extract Indian PAN Card information and return ONLY JSON.

Your response must be a single JSON object. No explanations, no thinking.

Format:
{
    "pan_number": "...",
    "full_name": "...",
    "date_of_birth": "YYYY-MM-DD"
}

Use null for missing fields.

CRITICAL RULE FOR pan_number:
- PAN number follows the format: AAAAA9999A (5 letters + 4 digits + 1 letter)
- Example: "ABCDE1234F"
- Extract the EXACT PAN number as shown on the card
- Pan numbers are ALWAYS 10 characters long
- Remove any spaces or special characters

CRITICAL RULE FOR full_name:
- Extract the cardholder's full name (the person whose PAN it is)
- This is NOT the father's name
- The cardholder's name is typically listed first or prominently
- Remove ALL salutations, titles, and prefixes
- Examples: "Mr. RAKESH KUMAR" -> "RAKESH KUMAR", "Shri RAJ KUMAR" -> "RAJ KUMAR"

CRITICAL RULE FOR date_of_birth:
- Extract date of birth if present
- Common formats: DD/MM/YYYY, DD-MM-YYYY
- Return in YYYY-MM-DD format

Indian PAN Card Layout Patterns:
1. Old PAN cards: Full name at top, father name below with S/O
2. New PAN cards: Full name prominent, father name separate
3. Always extract the CARDHOLDER'S name as full_name

Remove ALL salutations, titles, and prefixes from names.
Examples: "Mr. JOHN DOE" -> "JOHN DOE", "Shri RAJ KUMAR" -> "RAJ KUMAR"

Date Normalization:
- All dates must be in ISO format: YYYY-MM-DD
- Common date formats to handle:
  - DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY
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
            "Extract fields from this Indian PAN Card image.",
            "Return JSON with field values. Use null for missing fields.",
            "PAN number format: 5 letters + 4 digits + 1 letter (e.g., ABCDE1234F)"
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
            else:
                # Simple value - wrap in structure with confidence: 1.0
                extracted_data[field_name] = {
                    "value": field_data,
                    "confidence": 1.0,
                    "source": "vision_llm"
                }

        return extracted_data

    def _validate_pan_number(self, extracted_data: Dict[str, Any]) -> None:
        """
        Validate PAN number format.

        Expected format: AAAAA9999A (5 letters + 4 digits + 1 letter)
        Example: ABCDE1234F

        Args:
            extracted_data: Extracted data dictionary (modified in place)
        """
        if "pan_number" in extracted_data:
            value = extracted_data["pan_number"]["value"]
            if value and isinstance(value, str):
                # Clean up: remove spaces and make uppercase
                cleaned = re.sub(r'[\s\-]', '', value).upper()
                extracted_data["pan_number"]["value"] = cleaned

                # Validate pattern
                pattern = r'^[A-Z]{5}[0-9]{4}[A-Z]$'
                if not re.match(pattern, cleaned):
                    logger.warning(f"PAN number {cleaned} does not match expected format AAAAA9999A")

    def _clean_full_name(self, extracted_data: Dict[str, Any]) -> None:
        """
        Clean full name by removing salutations, titles, and ensuring minimum word count.

        Args:
            extracted_data: Extracted data dictionary (modified in place)
        """
        if "full_name" in extracted_data:
            value = extracted_data["full_name"]["value"]
            if value and isinstance(value, str):
                # Remove salutations and titles
                cleaned = self._remove_salutations(value)

                # Ensure name has at least 2 words (most Indian names have at least 2-3 words)
                words = cleaned.split()
                if len(words) < 2:
                    logger.warning(f"Full name '{cleaned}' has less than 2 words - may be incomplete")

                extracted_data["full_name"]["value"] = cleaned

    def _normalize_dob(self, extracted_data: Dict[str, Any]) -> None:
        """
        Normalize date of birth to ISO format (YYYY-MM-DD).

        Args:
            extracted_data: Extracted data dictionary (modified in place)
        """
        if "date_of_birth" in extracted_data:
            value = extracted_data["date_of_birth"]["value"]
            if value and isinstance(value, str):
                normalized = self._normalize_date(value)
                if normalized:
                    extracted_data["date_of_birth"]["value"] = normalized
                else:
                    logger.warning(f"Could not normalize DOB: {value}")

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
        titles_pattern = r'\b(?:MR|MRS|MS|MISS|DR|PROF|SIR|MADAM|SHRI|SMT|KUM|MM|SH|SMT\.|KM\.)\.?\s*'

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

                confidence_scores[field_name] = float(conf)

        return confidence_scores


# Singleton instance
_instance = None


def get_qwen_pan_extractor() -> QwenPANExtractor:
    """Get the singleton Qwen3-VL PAN Card extractor instance."""
    global _instance
    if _instance is None:
        _instance = QwenPANExtractor()
    return _instance

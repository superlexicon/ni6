"""
Qwen3-VL Direct Passport Extractor

Extracts passport fields directly from image using Qwen3-VL vision model.
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


class QwenPassportExtractor:
    """
    Direct passport extraction using Qwen3-VL vision model.

    Replaces the complex multi-stage pipeline (DocTR OCR + logic-based field extraction)
    with a single vision LLM call that extracts field values directly.

    Flow:
    1. Qwen3-VL extracts all passport fields from image
    2. Post-processing: Add confidence scores (1.0) for backwards compatibility
    3. Normalize dates to ISO format (YYYY-MM-DD)
    4. Return structured data with confidence scores

    JSON Output Format (from LLM):
    {
        "passport_number": "A1234567",
        "full_name": "JOHN DOE",  // OR surname + given_names for separate fields
        "surname": "DOE",  // For passports with separate surname field
        "given_names": "JOHN WILLIAM",  // For passports with separate given names field
        "date_of_birth": "1990-01-15",
        "nationality": "United States",
        "sex": "M",
        "date_of_expiry": "2025-06-20",
        "passport_country": "US",
        "place_of_birth": "New York",
        "issuing_authority": "Department of State",
        "date_of_issue": "2015-06-20"
    }

    Post-processed Format (with confidence added):
    {
        "passport_number": {"value": "A1234567", "confidence": 1.0, "source": "vision_llm"},
        "full_name": {"value": "JOHN DOE", "confidence": 1.0, "source": "vision_llm"},  // Combined from surname+given_names if needed
        "surname": {"value": "DOE", "confidence": 1.0, "source": "vision_llm"},
        "given_names": {"value": "JOHN WILLIAM", "confidence": 1.0, "source": "vision_llm"},
        ...
    }
    """

    # Required fields that must be extracted
    # Note: full_name OR (surname + given_names) must be present
    REQUIRED_FIELDS = [
        "passport_number",
        # "full_name" OR ("surname" + "given_names") is required
        "date_of_birth",
        "nationality",
        "sex"
    ]

    # Optional fields that may or may not be present
    OPTIONAL_FIELDS = [
        "full_name",  # For passports with single name field
        "surname",  # For passports with separate surname field
        "given_names",  # For passports with separate given names field
        "date_of_expiry",
        "passport_country",
        "place_of_birth",
        "issuing_authority",
        "date_of_issue"
    ]

    def __init__(self):
        """Initialize the Qwen3-VL extractor."""
        self.llm_service = LLMService()

    async def extract_fields(
        self,
        image_bytes: bytes,
        country_hint: Optional[str] = None,
        max_retries: int = 4
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Extract passport fields directly from image using Qwen3-VL.

        Args:
            image_bytes: JPEG image data (already preprocessed to meet Qwen3-VL requirements)
            country_hint: Optional country code hint for better extraction
            max_retries: Maximum number of extraction attempts with different strategies

        Returns:
            Tuple of (extracted_data, confidence_data):
            - extracted_data: Dict with field values, structured as:
                {
                    "field_name": {"value": "...", "confidence": 0.9, "source": "..."}
                }
            - confidence_data: Dict with per-field confidence scores
        """
        logger.info(f"Starting Qwen3-VL direct passport extraction (hint: country={country_hint})")

        # Try multiple extraction strategies
        for attempt in range(max_retries):
            try:
                # Build prompts for direct extraction
                system_prompt = self._build_system_prompt()
                user_prompt = self._build_user_prompt(country_hint)

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
                    # Look for a complete JSON object starting with { and ending with }
                    if "{" in thinking and "}" in thinking:
                        # Find the last complete JSON object in thinking
                        # (models often output reasoning followed by JSON)
                        json_start = thinking.rfind("{")
                        json_end = thinking.rfind("}")
                        if json_start >= 0 and json_end > json_start:
                            # Extract just the JSON part
                            potential_json = thinking[json_start:json_end + 1]
                            # Verify it's valid JSON by trying to parse it
                            try:
                                import json
                                json.loads(potential_json)
                                logger.info(f"Successfully extracted JSON from thinking field ({len(potential_json)} chars)")
                                content = potential_json
                            except json.JSONDecodeError:
                                logger.error(f"Extracted text from thinking is not valid JSON")
                                logger.error("This model may not support JSON-only output. Consider using qwen2.5-vl:7b instead.")

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
                        logger.error("This model may not support JSON-only output. Consider using qwen2.5-vl:7b instead.")

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

        # Compute confidence scores from extracted data
        confidence_data = self._compute_confidence_scores(extracted_data)

        logger.info(
            f"Qwen3-VL passport extraction completed: "
            f"{len(extracted_data)} fields extracted, "
            f"avg confidence: {sum(confidence_data.values()) / len(confidence_data) if confidence_data else 0:.2f}"
        )

        return extracted_data, confidence_data

    def _build_system_prompt(self) -> str:
        """
        Build system prompt for Qwen3-VL direct passport extraction.

        CRITICAL: Must enforce JSON-only output to avoid reasoning mode.
        """
        return """Extract passport information and return ONLY JSON.

Your response must be a single JSON object. No explanations, no thinking.

Format:
{
    "passport_number": "...",
    "full_name": "...",  // OR use surname + given_names
    "surname": "...",  // For passports with separate surname field
    "given_names": "...",  // For passports with separate given names field
    "date_of_birth": "YYYY-MM-DD",
    "nationality": "...",
    "sex": "M/F",
    "date_of_expiry": "YYYY-MM-DD",
    "passport_country": "...",
    "place_of_birth": "...",
    "issuing_authority": "...",
    "date_of_issue": "YYYY-MM-DD"
}

Use null for missing fields.

CRITICAL RULE FOR passport_country:
- passport_country MUST be ISO 2-letter country code (e.g., "US", "IN", "SG", "AE", "GB")
- EXTRACT FROM MRZ (Machine Readable Zone) - Look at the bottom of the passport for the two lines with <<<<< patterns
- MRZ Line 1 format: P<COUNTRY_CODE<PASSPORT_NUMBER<<<<<<<<<<<<<<<
- The country code is IMMEDIATELY after "P" or "P<"
- This MRZ country code is the MOST ACCURATE - ignore text from visas, stamps, or endorsements
- Cross-reference with nationality field and issuing authority for confirmation
- Common examples: "INDIA" → "IN", "SINGAPORE" → "SG", "UNITED STATES" → "US", "UNITED ARAB EMIRATES" → "AE"
- DO NOT be fooled by visas, stamps, or endorsements - only use the passport's own country code
- The MRZ country code at the start of line 1 (after document type "P") is the ONLY reliable source

CRITICAL RULE FOR full_name, surname, given_names:

Some passports have a SINGLE name field, others have SEPARATE surname and given name fields.

SINGLE FIELD FORMAT:
- Extract as "full_name": "JOHN DOE"
- Set "surname": null and "given_names": null

SEPARATE FIELDS FORMAT:
- Extract "surname": "DOE" (last name/family name)
- Extract "given_names": "JOHN WILLIAM" (first and middle names)
- Set "full_name": null (will be combined in post-processing)

INDIA PASSPORTS:
- Given names may contain "S/O", "D/O", "W/O" - include these in given_names
- Example: Surname="SHARMA", Given names="RAKESH S/O OM PRAKASH"

DETECTING THE FORMAT:
- Look for EXPLICIT LABELS: "Surname", "Family Name", "Last Name" → this indicates separate fields
- Look for EXPLICIT LABELS: "Given Names", "First Name", "Given Name" → this indicates separate fields
- If you see separate labels with separate values below them, use the SEPARATE FIELDS FORMAT
- If you see a single "Name" label with one value, use SINGLE FIELD FORMAT
- Look ABOVE the MRZ for name fields - they are typically in the upper portion of the passport
- Extract ALL name parts - don't truncate names

Remove ALL salutations, titles, and prefixes from names. Extract ONLY the name.
Examples: "Mr. JOHN DOE" -> "JOHN DOE", "Shri RAJ KUMAR" -> "RAJ KUMAR"

MRZ Parsing:
- MRZ (Machine Readable Zone) is at the bottom of the passport with two lines of <<<<< patterns
- Line 1: Document type + country code + passport number (with check digits)
- Line 2: Date of birth + sex + date of expiry + personal number
- Extract passport_number from positions 0-8 (first line after country code)
- Dates in MRZ are in YYMMDD format - convert to YYYY-MM-DD

Date Normalization:
- All dates must be in ISO format: YYYY-MM-DD
- If year is 2 digits (YY), assume:
  - If YY >= 50, year is 19YY
  - If YY < 50, year is 20YY
- Common date formats to handle:
  - DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY
  - MM/DD/YYYY, MM-DD-YYYY, MM.DD.YYYY
  - YYYY/MM/DD, YYYY-MM-DD, YYYY.MM.DD
  - D MMM YYYY (e.g., "15 Jan 1990")
  - MMM D, YYYY (e.g., "Jan 15, 1990")

Sex:
- Must be single letter: "M" or "F"
- Extract from MRZ or explicit label on passport

ISO country codes: IN, SG, AE, US, GB, MY, MM, TH, VN, PH, ID, BD, PK, LK, NP"""

    def _build_user_prompt(
        self,
        country_hint: Optional[str]
    ) -> str:
        """
        Build user prompt with country hints and extraction instructions.

        CRITICAL: Prepend /no_think to bypass Qwen3's CoT generation.

        Args:
            country_hint: Optional country code hint

        Returns:
            User prompt string
        """
        prompt_parts = [
            "/no_think",  # Qwen3 trigger to skip thinking block generation
            "Extract passport fields from this passport image.",
            "Read MRZ (bottom two lines with <<<<<) for accurate dates and passport number."
        ]

        # Add hints if provided
        if country_hint:
            prompt_parts.append(f"Expected country: {country_hint}")

        prompt_parts.extend([
            "Return JSON with field values. Use null for missing fields.",
            "All dates must be in YYYY-MM-DD format."
        ])

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
                # Try to extract the outermost JSON object
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
                # Add confidence: 1.0 for backwards compatibility
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

        # Clean passport_number: remove spaces and dashes
        if "passport_number" in extracted_data:
            value = extracted_data["passport_number"]["value"]
            if value and isinstance(value, str):
                cleaned = re.sub(r'[\s\-]', '', value)
                extracted_data["passport_number"]["value"] = cleaned

        # Clean full_name: remove salutations/titles
        if "full_name" in extracted_data:
            value = extracted_data["full_name"]["value"]
            if value and isinstance(value, str):
                cleaned = self._remove_salutations(value)
                extracted_data["full_name"]["value"] = cleaned

        # Handle separate surname and given_names fields
        # If we have separate surname and given_names but no full_name, combine them
        if "surname" in extracted_data and "given_names" in extracted_data:
            surname_val = extracted_data["surname"]["value"]
            given_names_val = extracted_data["given_names"]["value"]

            # Debug logging: Show what was extracted
            logger.info(f"Name extraction - surname: '{surname_val}', given_names: '{given_names_val}'")

            # Only combine if both have values and full_name is missing or empty
            if surname_val and given_names_val:
                full_name_val = extracted_data.get("full_name", {}).get("value")

                if not full_name_val or not isinstance(full_name_val, str) or len(full_name_val.strip()) == 0:
                    # Combine: SURNAME + GIVEN_NAMES
                    combined_full_name = f"{surname_val} {given_names_val}"
                    extracted_data["full_name"] = {
                        "value": combined_full_name.strip(),
                        "confidence": 1.0,
                        "source": "vision_llm"
                    }
                    logger.info(f"Combined surname and given_names into full_name: {combined_full_name}")

        # Debug logging: Show all name-related fields that were extracted
        name_fields = ["full_name", "surname", "given_names"]
        extracted_names = {k: extracted_data.get(k, {}).get("value") for k in name_fields if k in extracted_data}
        if extracted_names:
            logger.info(f"Final name extraction result: {extracted_names}")

        # Clean surname: remove salutations/titles
        if "surname" in extracted_data:
            value = extracted_data["surname"]["value"]
            if value and isinstance(value, str):
                cleaned = self._remove_salutations(value)
                extracted_data["surname"]["value"] = cleaned

        # Clean given_names: remove salutations/titles (but keep S/O, D/O, W/O)
        if "given_names" in extracted_data:
            value = extracted_data["given_names"]["value"]
            if value and isinstance(value, str):
                # For given_names, only remove common salutations, keep lineage indicators
                # Pattern: remove salutations but keep S/O, D/O, W/O patterns
                # First, remove common titles
                titles_pattern = r'\b(?:MR|MRS|MS|MISS|DR|PROF|SIR|MADAM|SHRI|SMT|KUM|MME|HERR|FRAU|SIGNOR|SIGNORA)\.?\s*'
                cleaned = re.sub(f'^{titles_pattern}', '', value, flags=re.IGNORECASE)
                # Don't strip S/O, D/O, W/O - these are part of Indian passport names
                cleaned = cleaned.strip().strip('.')
                if cleaned != value:
                    extracted_data["given_names"]["value"] = cleaned

        # Normalize dates to ISO format
        for date_field in ["date_of_birth", "date_of_expiry", "date_of_issue"]:
            if date_field in extracted_data:
                value = extracted_data[date_field]["value"]
                if value and isinstance(value, str):
                    normalized = self._normalize_date(value)
                    if normalized:
                        extracted_data[date_field]["value"] = normalized
                        logger.debug(f"Normalized {date_field}: {value} -> {normalized}")

        # Normalize passport_country to ISO 2-letter code
        if "passport_country" in extracted_data:
            value = extracted_data["passport_country"]["value"]
            if value and isinstance(value, str):
                # Clean up the value
                value = value.strip().upper()

                # If it's already a 2-letter code, keep it
                if len(value) == 2 and value.isalpha():
                    extracted_data["passport_country"]["value"] = value
                # If it's a 3-letter code, convert to 2-letter
                elif len(value) == 3 and value.isalpha():
                    from app.utils.country_code_converter import convert_alpha3_to_alpha2
                    converted = convert_alpha3_to_alpha2(value)
                    if converted:
                        extracted_data["passport_country"]["value"] = converted
                        logger.info(f"Converted 3-letter country code: {value} -> {converted}")
                    else:
                        logger.warning(f"Failed to convert 3-letter code: {value}")
                # Otherwise, try to extract from full country name
                else:
                    from app.utils.country_code_converter import country_name_to_code
                    converted = country_name_to_code(value)
                    if converted:
                        extracted_data["passport_country"]["value"] = converted
                        logger.info(f"Converted country name: {value} -> {converted}")
                    else:
                        logger.warning(f"Failed to convert country name: {value}")

        return extracted_data

    def _compute_confidence_scores(self, extracted_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute per-field confidence scores from extracted data.

        Simple format: field_name -> confidence_float

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

    def _remove_salutations(self, name: str) -> str:
        """
        Remove salutations and titles from passport holder name.

        Handles English, Indian, and international titles with/without dots.

        Args:
            name: Raw name that may contain titles

        Returns:
            Name with titles removed
        """
        if not name:
            return name

        # Common salutations/titles to remove (with and without dots)
        # Pattern matches word boundaries and optional dots
        titles_pattern = r'\b(?:MR|MRS|MS|MISS|DR|PROF|SIR|MADAM|SHRI|SMT|KUM|MME|HERR|FRAU|SIGNOR|SIGNORA)\.?\s*'

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
        - MM/DD/YYYY, MM-DD-YYYY, MM.DD.YYYY
        - YYYY/MM/DD, YYYY-MM-DD, YYYY.MM.DD
        - D MMM YYYY (e.g., "15 Jan 1990")
        - MMM D, YYYY (e.g., "Jan 15, 1990")
        - YYMMDD (MRZ format)

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
        # MRZ format: YYMMDD
        if re.match(r'^\d{6}$', date_str):
            yy = int(date_str[0:2])
            mm = date_str[2:4]
            dd = date_str[4:6]
            # Convert 2-digit year to 4-digit
            yyyy = 1900 + yy if yy >= 50 else 2000 + yy
            return f"{yyyy}-{mm}-{dd}"

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


# Singleton instance
_instance = None


def get_qwen_passport_extractor() -> QwenPassportExtractor:
    """Get the singleton Qwen3-VL passport extractor instance."""
    global _instance
    if _instance is None:
        _instance = QwenPassportExtractor()
    return _instance

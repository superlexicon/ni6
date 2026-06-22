"""
Qwen3-VL Direct Bank Statement Extractor

Extracts bank statement fields directly from image using Qwen3-VL vision model.
No intermediate OCR, layout caching, or spatial coordinate extraction needed.

Qwen3-VL extracts field values directly from the image.
Confidence scores are added post-extraction for backwards compatibility (1.0).
SWIFT codes are fetched from database based on extracted bank metadata.
"""

import json
import re
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

from app.core.logger import get_logger
from app.services.llm_service import LLMService
from app.core.key_injection.bank_database_lookup import get_bank_database_lookup, BankInfo
from app.utils.date_parser import parse_date_to_mariadb, format_date_for_display


logger = get_logger()


@dataclass
class ExtractionResult:
    """Result of direct field extraction with confidence scores."""
    extracted_data: Dict[str, Any]  # Field values with metadata
    confidence_data: Dict[str, float]  # Per-field confidence scores
    extraction_method: str = "qwen3_vl_direct"
    raw_response: Optional[str] = None


class QwenBankStatementExtractor:
    """
    Direct bank statement extraction using Qwen3-VL vision model.

    Replaces the complex multi-stage pipeline (DocTR OCR + GLiNER2 + Layout Cache + Spatial Coordinate Extraction)
    with a single vision LLM call that extracts field values directly.

    Flow:
    1. Qwen3-VL extracts bank metadata (bank_name, bank_country, bank_address) + all fields
    2. Post-processing: Add confidence scores (1.0) for backwards compatibility
    3. Look up SWIFT code from database using extracted bank metadata
    4. Return structured data with confidence scores

    JSON Output Format (from LLM):
    {
        "bank_name": "STATE BANK OF INDIA",
        "bank_country": "IN",
        "bank_address": "NEW DELHI MAIN",
        "account_holder_name": "JOHN DOE",
        "customer_address": "123 Main St\\nCity 12345",
        "account_number": "123456789",
        "currency": "INR",
        "statement_date": "01 Jan 2024"
    }

    Post-processed Format (with confidence added):
    {
        "bank_name": {"value": "STATE BANK OF INDIA", "confidence": 1.0},
        "bank_country": {"value": "IN", "confidence": 1.0},
        ...
    }
    """

    # Required fields that must be extracted
    REQUIRED_FIELDS = [
        "bank_name",
        "bank_country",
        "account_holder_name",
        "account_number",
        "currency"
    ]

    # Optional fields that may or may not be present
    OPTIONAL_FIELDS = [
        "bank_address",
        "customer_address",
        "cif_number",
        "ifsc_code",
        "iban",
        "statement_date",
        "statement_period",
        "opening_balance",
        "closing_balance"
    ]

    def __init__(self):
        """Initialize the Qwen3-VL extractor."""
        self.llm_service = LLMService()
        self.bank_lookup = get_bank_database_lookup()

    async def extract_fields(
        self,
        image_bytes: bytes,
        bank_name_hint: Optional[str] = None,
        country_hint: Optional[str] = None,
        max_retries: int = 4
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Extract bank statement fields directly from image using Qwen3-VL.

        Args:
            image_bytes: JPEG image data (already preprocessed to meet Qwen3-VL requirements)
            bank_name_hint: Optional bank name hint for better extraction
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
        logger.info(f"Starting Qwen3-VL direct extraction (hints: bank={bank_name_hint}, country={country_hint})")

        # Try multiple extraction strategies
        for attempt in range(max_retries):
            try:
                # Build prompts for direct extraction
                system_prompt = self._build_system_prompt()
                user_prompt = self._build_user_prompt(bank_name_hint, country_hint)

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

        # Post-process: Look up SWIFT code from database based on extracted bank metadata
        bank_name = extracted_data.get("bank_name", {}).get("value")
        bank_country = extracted_data.get("bank_country", {}).get("value")

        if bank_name and bank_country:
            logger.info(f"Looking up SWIFT code for bank={bank_name}, country={bank_country}")

            # Try to look up SWIFT code from database using only bank_name and bank_country
            bank_info = self.bank_lookup.lookup_by_name(
                bank_name=bank_name,
                country=bank_country
            )

            if bank_info and bank_info.swift_code:
                extracted_data["swift_code"] = {
                    "value": bank_info.swift_code,
                    "confidence": 1.0,  # Database lookup is authoritative
                    "source": "database"
                }
                logger.info(f"Found SWIFT code from database: {bank_info.swift_code}")
            else:
                logger.debug(f"No SWIFT code found in database for {bank_name}/{bank_country}")

        # Compute confidence scores from extracted data
        confidence_data = self._compute_confidence_scores(extracted_data)

        logger.info(
            f"Qwen3-VL extraction completed: "
            f"{len(extracted_data)} fields extracted, "
            f"avg confidence: {sum(confidence_data.values()) / len(confidence_data) if confidence_data else 0:.2f}"
        )

        return extracted_data, confidence_data

    def _build_system_prompt(self) -> str:
        """
        Build system prompt for Qwen3-VL direct extraction.

        CRITICAL: Must enforce JSON-only output to avoid reasoning mode.
        """
        return """Extract bank statement information and return ONLY JSON.

Your response must be a single JSON object. No explanations, no thinking.

Format:
{
    "bank_name": "...",
    "bank_country": "...",
    "bank_address": "...",
    "account_holder_name": "...",
    "customer_address": "...",
    "account_number": "...",
    "currency": "...",
    "statement_date": "..."
}

Use null for missing fields.

CRITICAL RULE FOR account_holder_name:
Remove ALL salutations, titles, and prefixes. Extract ONLY the name.
Examples: "Mr. JOHN DOE" -> "JOHN DOE", "Shri RAJ KUMAR" -> "RAJ KUMAR"

CRITICAL RULE FOR Separating Name from Address:
The account_holder_name and customer_address are SEPARATE fields.

account_holder_name format:
- May contain "S/O" (Son Of), "D/O" (Daughter Of), "W/O" (Wife Of) - these are part of the name
- These lineage indicators belong in account_holder_name, NOT in customer_address
- Example: "MANOGRAN S/O THANABALAN" is the COMPLETE account_holder_name

customer_address format:
- Extract ONLY the address (street, block, city, postal code)
- DO NOT include the person's name or lineage indicators (S/O, D/O, W/O)
- Address starts with street, block, building, or similar location identifiers
- Example: "BLK 29 MARINE CRESCENT #11-25 SINGAPORE 440029"

When name and address appear together, SPLIT them:
- "MANOGRAN S/O THANABALAN BLK 29 MARINE CRESCENT #11-25 SINGAPORE 440029"
  -> account_holder_name: "MANOGRAN S/O THANABALAN"
  -> customer_address: "BLK 29 MARINE CRESCENT #11-25 SINGAPORE 440029"

CRITICAL RULE FOR customer_address vs bank_address:
- **customer_address** = address belonging to the account holder (nearest to their name)
- **bank_address** = ONLY the bank's physical branch/location address

**SINGLE ADDRESS RULE (MOST IMPORTANT):**
- If there is ONLY ONE address in the entire document:
  - That address is ALWAYS the customer_address
  - Set bank_address to null
  - NEVER classify a single address as bank_address

- If there are TWO addresses:
  - The address NEAREST to account_holder_name is customer_address
  - The other address (if labeled "branch", "head office", etc.) is bank_address

**Keywords indicating customer_address:**
- "Communication Address", "Correspondence Address", "Mailing Address"
- Address appearing near customer name

**Keywords indicating bank_address:**
- "Registered Office", "Branch", "Head Office", "HO"
- Address labeled with bank name

**Examples:**
- Single address "H.No 1-21, SETTYGARIPALLE, Chittoor, Andhra Pradesh 517419"
  -> customer_address: "H.No 1-21, SETTYGARIPALLE, Chittoor, Andhra Pradesh 517419"
  -> bank_address: null

- "Communication Address: 123 Main St" + "Branch: 456 Bank Ave"
  -> customer_address: "123 Main St"
  -> bank_address: "456 Bank Ave"

Address Format Examples:
- India: "H.No 1-21, SETTYGARIPALLE, Chittoor, Andhra Pradesh 517419"
- Singapore: "BLK 29 MARINE CRESCENT #11-25 SINGAPORE 440029"
- Standard: "123 Main Street, City 12345"

ISO country codes: IN, SG, AE, US, GB"""

    def _build_user_prompt(
        self,
        bank_name_hint: Optional[str],
        country_hint: Optional[str]
    ) -> str:
        """
        Build user prompt with bank hints and extraction instructions.

        CRITICAL: Prepend /no_think to bypass Qwen3's CoT generation.

        Args:
            bank_name_hint: Optional bank name hint
            country_hint: Optional country code hint

        Returns:
            User prompt string
        """
        prompt_parts = [
            "/no_think",  # Qwen3 trigger to skip thinking block generation
            "Extract fields from this bank statement image.",
            "CRITICAL: customer_address is the address belonging to the account holder (nearest to their name)."
        ]

        # Add hints if provided
        if bank_name_hint or country_hint:
            hints = []
            if bank_name_hint:
                hints.append(f"Bank: {bank_name_hint}")
            if country_hint:
                hints.append(f"Country: {country_hint}")

            if hints:
                prompt_parts.append(f"Expected: " + ", ".join(hints))

        prompt_parts.extend([
            "Address formats include: '123 Main St, City 12345', 'BLK 29 STREET #01-123 SINGAPORE 123456', etc.",
            "Return JSON with field values. Use null for missing fields."
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
                # Strategy 3: Try to find JSON-like structures in reasoning text
                # Look for patterns like "field_name": "..." or "field_name": {"value": "..."}
                json_pattern = r'(\w+)\s*:\s*(?:"([^"]*)"|\{[^}]*\})'
                matches = re.findall(json_pattern, content)
                if matches and len(matches) >= 3:  # At least 3 fields found
                    # Reconstruct JSON from matches
                    reconstructed = "{"
                    for i, match in enumerate(matches):
                        if i > 0:
                            reconstructed += ","
                        reconstructed += match
                    reconstructed += "}"

                    # Try to add field names by looking for patterns before each match
                    lines = content.split('\n')
                    field_pattern = r'(\w+)\s*:\s*\{'
                    for line in lines:
                        field_match = re.search(field_pattern, line)
                        if field_match:
                            field_name = field_match.group(1)
                            # Add field to reconstructed JSON
                            pass  # Complex reconstruction - skip for now

                    content = reconstructed
                    logger.debug("Extracted JSON using field pattern matching")
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

        # Validate structure - should have field entries with value/confidence
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

        # Clean numeric values (remove spaces, dashes, etc.)
        for field in ["account_number", "opening_balance", "closing_balance"]:
            if field in extracted_data:
                value = extracted_data[field]["value"]
                if value and isinstance(value, str):
                    # Clean account number: remove spaces and dashes, then leading special chars
                    if field == "account_number":
                        # First remove spaces and dashes
                        cleaned = re.sub(r'[\s\-]', '', value)
                        # Then remove any leading special characters (non-alphanumeric)
                        cleaned = re.sub(r'^[^a-zA-Z0-9]+', '', cleaned)
                        # Validation: account numbers must be numeric-only (no letters)
                        # This catches masked numbers like "5524XXXXXXXX" or invalid formats like "ACC123456"
                        if re.search(r'[a-zA-Z]', cleaned):
                            logger.warning(f"Account number contains letters, marking as invalid: '{cleaned}'")
                            extracted_data[field]["value"] = None
                            extracted_data[field]["confidence"] = 0.0
                        else:
                            extracted_data[field]["value"] = cleaned
                    # Clean balance values: remove commas and spaces
                    elif field in ["opening_balance", "closing_balance"]:
                        cleaned = re.sub(r'[,\s]', '', value)
                        extracted_data[field]["value"] = cleaned

        # Clean account_holder_name: remove salutations/titles
        if "account_holder_name" in extracted_data:
            value = extracted_data["account_holder_name"]["value"]
            if value and isinstance(value, str):
                cleaned = self._remove_salutations(value)
                extracted_data["account_holder_name"]["value"] = cleaned

        # Normalize statement_date to ISO format
        if "statement_date" in extracted_data:
            value = extracted_data["statement_date"]["value"]
            if value and isinstance(value, str):
                parsed_date = parse_date_to_mariadb(value)
                if parsed_date:
                    normalized = format_date_for_display(parsed_date)  # Returns YYYY-MM-DD
                    extracted_data["statement_date"]["value"] = normalized
                    logger.debug(f"Normalized statement_date: '{value}' -> '{normalized}'")

        # Fallback: Fix single-address misclassification
        # If bank_address is set but customer_address is null, the model likely
        # misclassified a single address as bank_address instead of customer_address
        bank_addr = extracted_data.get("bank_address", {}).get("value")
        customer_addr = extracted_data.get("customer_address", {}).get("value")

        if bank_addr and not customer_addr:
            # Move bank_address to customer_address and set bank_address to null
            extracted_data["customer_address"] = {
                "value": bank_addr,
                "confidence": 1.0,
                "source": "vision_llm_corrected"
            }
            extracted_data["bank_address"] = {
                "value": None,
                "confidence": 0.0,
                "source": "vision_llm_corrected"
            }
            logger.info(f"Address fallback applied: moved bank_address to customer_address (single address case)")

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
        Remove salutations and titles from account holder name.

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


# Singleton instance
_instance = None


def get_qwen_bank_statement_extractor() -> QwenBankStatementExtractor:
    """Get the singleton Qwen3-VL extractor instance."""
    global _instance
    if _instance is None:
        _instance = QwenBankStatementExtractor()
    return _instance

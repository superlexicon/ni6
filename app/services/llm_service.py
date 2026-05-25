"""
LLM Service for Bank-Specific Prompt Generation

Provides integration with LLM APIs (OpenAI, Anthropic, etc.) for generating
GLiNER2 prompts specific to bank statement formats.

Follows the same pattern as worldcheck_service.py for consistency.
"""

import asyncio
import json
import time
from typing import Dict, Any, Optional, List
import httpx

from app.config.llm_config import llm_settings
from app.core.logger import get_logger


logger = get_logger()


class LLMService:
    """Service for LLM API calls to generate bank-specific prompts"""

    # Class-level shared client (created on first use)
    _client: Optional[httpx.AsyncClient] = None
    _client_lock: Optional[Any] = None

    def __init__(self):
        """Initialize the LLM service."""
        self.settings = llm_settings
        self.api_url = self.settings.api_url.rstrip('/')

    @classmethod
    async def _get_client(cls) -> httpx.AsyncClient:
        """Get or create the shared httpx client."""
        if cls._client is None or cls._client.is_closed:
            timeout = httpx.Timeout(
                connect=10.0,
                read=60.0,
                write=10.0,
                pool=5.0
            )
            cls._client = httpx.AsyncClient(timeout=timeout)
        return cls._client

    @classmethod
    async def close_client(cls):
        """Close the shared client (call on shutdown)."""
        if cls._client is not None and not cls._client.is_closed:
            await cls._client.aclose()
            cls._client = None

    async def generate_prompts(
        self,
        bank_name: str,
        bank_abbrev: str,
        country_code: str,
        ocr_text: str,
        generic_extraction_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate bank-specific GLiNER2 prompts using LLM.

        Args:
            bank_name: Full bank name
            bank_abbrev: Bank abbreviation
            country_code: ISO country code
            ocr_text: OCR text from bank statement (first 5000 chars)
            generic_extraction_result: Results from generic GLiNER2 extraction

        Returns:
            Dictionary with:
                - prompts: List of prompt configurations
                - extraction_config: Extraction configuration
                - reasoning: LLM's explanation
                - token_usage: Token usage statistics
                - error: Error message if generation failed
        """
        start_time = time.time()

        try:
            # Truncate OCR text to first 5000 chars
            ocr_text_sample = ocr_text[:5000] if ocr_text else ""

            # Build the system prompt
            system_prompt = self._build_system_prompt()

            # Build the user prompt with bank info and statement sample
            user_prompt = self._build_user_prompt(
                bank_name=bank_name,
                bank_abbrev=bank_abbrev,
                country_code=country_code,
                ocr_text=ocr_text_sample,
                generic_extraction_result=generic_extraction_result
            )

            # Make the API call
            response = await self._make_api_request(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )

            # Parse the response
            result = self._parse_response(response, time.time() - start_time)

            logger.info(
                f"LLM prompt generation completed for {bank_abbrev}/{country_code}: "
                f"{len(result.get('prompts', []))} prompts generated"
            )

            return result

        except (asyncio.CancelledError, Exception) as e:
            elapsed = time.time() - start_time
            if isinstance(e, asyncio.CancelledError):
                logger.error(f"LLM prompt generation cancelled for {bank_abbrev}/{country_code}")
                return {
                    "prompts": [],
                    "extraction_config": {},
                    "reasoning": "",
                    "token_usage": {},
                    "error": "Prompt generation was cancelled"
                }

            logger.error(
                f"LLM prompt generation failed for {bank_abbrev}/{country_code}: {str(e)}"
            )
            return {
                "prompts": [],
                "extraction_config": {},
                "reasoning": "",
                "token_usage": {},
                "error": f"Prompt generation error: {str(e)}"
            }

    def _build_system_prompt(self) -> str:
        """Build the system prompt for LLM."""
        return """You are a GLiNER2 prompt engineering expert specializing in bank statement analysis.

GLiNER2 is a schema-based information extraction model that uses natural language descriptions to identify and extract entities from text.

Your task is to analyze bank statement formats and create optimized entity descriptions for GLiNER2 extraction.

Key principles:
1. Be SPECIFIC - Describe exactly how each field appears in THIS bank's statements
2. Use OBSERVATIONS - Note label text, positioning, formatting patterns
3. Provide REAL EXAMPLES - Extract actual examples from the provided statement text
4. Suggest VALIDATION - Recommend regex patterns and confidence thresholds

GLiNER2 entity categories include:
- PERSON: Names of individuals
- ORGANIZATION: Company names, bank names
- LOCATION: Addresses, cities, countries
- DATE: Dates in various formats
- NUMBER: Numeric values
- MONEY: Currency amounts
- IDENTIFIER: Account numbers, IDs, codes
- custom: Domain-specific entities

Return your response as valid JSON only, no markdown formatting."""

    def _build_user_prompt(
        self,
        bank_name: str,
        bank_abbrev: str,
        country_code: str,
        ocr_text: str,
        generic_extraction_result: Optional[Dict[str, Any]]
    ) -> str:
        """Build the user prompt with bank information."""
        prompt = f"""Analyze this bank statement format and generate GLiNER2 entity descriptions.

Bank Information:
- Bank: {bank_name} ({bank_abbrev})
- Country: {country_code}

Statement OCR Text (first 5000 chars):
{ocr_text}
"""

        if generic_extraction_result:
            prompt += f"""
Generic Extraction Results (for reference):
{json.dumps(generic_extraction_result, indent=2, default=str)}
"""

        prompt += """

Task: Generate GLiNER2 entity descriptions optimized for this bank's statement format.

For each of the following fields, analyze how it appears in this statement:

1. **account_holder_name**: Customer/account holder name
2. **account_number**: Bank account number
3. **cif_number**: Customer Identification File number (if present)
4. **customer_address**: Customer's residential/postal address
5. **branch_address**: Bank branch address (separate from customer)
6. **branch_name**: Branch name or identifier
7. **currency**: Currency code or display name
8. **statement_date**: Statement date or period

For each field found in the statement:
- **prompt_description**: Clear, specific description for GLiNER2 (e.g., "Customer name appearing after 'Name of Account Holder:' label in uppercase letters")
- **entity_category**: PERSON, ORGANIZATION, LOCATION, DATE, NUMBER, MONEY, IDENTIFIER, or custom
- **examples**: 2-4 actual examples extracted from this statement
- **validation_pattern**: Suggested regex pattern for validation
- **threshold**: Recommended confidence threshold (0.2-0.5)

Return JSON format:
{
  "prompts": [
    {
      "entity_type": "account_holder_name",
      "prompt_description": "Customer name appearing after 'Name of Account Holder:' label",
      "entity_category": "PERSON",
      "examples": ["RAM CHANDRA", "SMT. LAXMI DEVI"],
      "validation_pattern": "^[A-Z\\s\\.]+$",
      "threshold": 0.3
    }
  ],
  "extraction_config": {
    "default_threshold": 0.3,
    "extraction_order": ["account_holder_name", "account_number", "statement_date", "closing_balance"],
    "special_handling": "Notes about multi-page statements, special formatting, etc."
  },
  "reasoning": "Brief explanation of key observations about this bank's statement format"
}
"""
        return prompt

    async def _make_api_request(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> Dict[str, Any]:
        """Make an API request to the LLM service."""
        url = f"{self.api_url}/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.settings.api_key}"
        }

        payload = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "response_format": {"type": "json_object"}
        }

        # Retry logic
        for attempt in range(self.settings.max_retries):
            try:
                client = await self._get_client()
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                if e.response.status_code in (429, 500, 502, 503, 504):
                    # Retry on rate limit or server errors
                    if attempt < self.settings.max_retries - 1:
                        delay = self.settings.retry_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"LLM API error {e.response.status_code}, retrying in {delay}s...")
                        await asyncio.sleep(delay)
                        continue
                raise

            except httpx.TimeoutException as e:
                if attempt < self.settings.max_retries - 1:
                    delay = self.settings.retry_delay * (2 ** attempt)
                    logger.warning(f"LLM API timeout, retrying in {delay}s...")
                    await asyncio.sleep(delay)
                    continue
                raise

    def _parse_response(
        self,
        response: Dict[str, Any],
        generation_time_ms: float
    ) -> Dict[str, Any]:
        """Parse the LLM API response."""
        try:
            # Extract content from response
            choices = response.get("choices", [])
            if not choices:
                return {
                    "prompts": [],
                    "extraction_config": {},
                    "reasoning": "",
                    "token_usage": {},
                    "error": "Empty response from LLM"
                }

            content = choices[0].get("message", {}).get("content", "")

            # Parse JSON content
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code block
                if "```json" in content:
                    start = content.find("```json") + 7
                    end = content.find("```", start)
                    if end > start:
                        parsed = json.loads(content[start:end].strip())
                elif "```" in content:
                    start = content.find("```") + 3
                    end = content.find("```", start)
                    if end > start:
                        parsed = json.loads(content[start:end].strip())
                else:
                    raise

            # Extract token usage
            usage = response.get("usage", {})
            token_usage = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0)
            }

            return {
                "prompts": parsed.get("prompts", []),
                "extraction_config": parsed.get("extraction_config", {}),
                "reasoning": parsed.get("reasoning", ""),
                "token_usage": token_usage,
                "generation_time_ms": int(generation_time_ms * 1000)
            }

        except Exception as e:
            logger.error(f"Failed to parse LLM response: {str(e)}")
            return {
                "prompts": [],
                "extraction_config": {},
                "reasoning": "",
                "token_usage": response.get("usage", {}),
                "error": f"Failed to parse response: {str(e)}"
            }


# Global service instance
llm_service = LLMService()

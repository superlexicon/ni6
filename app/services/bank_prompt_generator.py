"""
Bank Prompt Generator Service

Generates GLiNER2 prompts specific to a bank's statement format using LLM.
Bridges the gap between generic extraction and bank-specific optimization.

Workflow:
1. Analyze statement OCR text and generic extraction results
2. Call LLM to generate bank-specific prompts
3. Return prompts in database-ready format
"""

import json
from typing import Dict, Any, Optional, List
from datetime import datetime

from app.services.llm_service import llm_service
from app.core.logger import get_logger


logger = get_logger()


class BankPromptGenerator:
    """Generates GLiNER2 prompts specific to a bank's statement format."""

    def __init__(self):
        """Initialize the prompt generator."""
        self.llm_service = llm_service

    async def generate_prompts_for_bank(
        self,
        bank_id: int,
        bank_abbrev: str,
        bank_name: str,
        country_code: str,
        ocr_text: str,
        generic_extraction_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate bank-specific GLiNER2 prompts based on statement analysis.

        Args:
            bank_id: Database ID of the bank
            bank_abbrev: Bank abbreviation (e.g., "MAHB")
            bank_name: Full bank name
            country_code: ISO country code
            ocr_text: OCR text from the statement
            generic_extraction_result: Results from generic GLiNER2 extraction

        Returns:
            Dictionary with:
                - prompts: List of prompt configurations for database
                - extraction_config: Extraction configuration for database
                - metadata: Generation metadata (model, tokens, timing)
                - error: Error message if generation failed
        """
        print(f"DEBUG: generate_prompts_for_bank called for bank_id={bank_id}, bank={bank_abbrev}")
        logger.info(
            f"Generating prompts for bank_id={bank_id}, "
            f"bank={bank_abbrev}, country={country_code}"
        )

        try:
            # Step 1: Call LLM service to generate prompts
            llm_result = await self.llm_service.generate_prompts(
                bank_name=bank_name,
                bank_abbrev=bank_abbrev,
                country_code=country_code,
                ocr_text=ocr_text,
                generic_extraction_result=generic_extraction_result
            )

            # Check for errors
            if llm_result.get("error"):
                logger.error(f"LLM generation failed: {llm_result['error']}")
                return {
                    "prompts": [],
                    "extraction_config": {},
                    "metadata": self._build_metadata(llm_result),
                    "error": llm_result["error"]
                }

            # Step 2: Validate and normalize prompts
            prompts = self._validate_and_normalize_prompts(
                llm_result.get("prompts", []),
                bank_id,
                country_code
            )

            # Step 3: Build extraction config
            extraction_config = self._build_extraction_config(
                llm_result.get("extraction_config", {}),
                bank_id,
                country_code
            )

            # Step 4: Build metadata
            metadata = self._build_metadata(llm_result)
            metadata["bank_id"] = bank_id
            metadata["bank_abbrev"] = bank_abbrev
            metadata["country_code"] = country_code

            logger.info(
                f"Successfully generated {len(prompts)} prompts for {bank_abbrev}/{country_code}"
            )

            return {
                "prompts": prompts,
                "extraction_config": extraction_config,
                "metadata": metadata
            }

        except Exception as e:
            logger.error(f"Prompt generation failed for {bank_abbrev}/{country_code}: {str(e)}")
            return {
                "prompts": [],
                "extraction_config": {},
                "metadata": {},
                "error": f"Generation failed: {str(e)}"
            }

    def _validate_and_normalize_prompts(
        self,
        raw_prompts: List[Dict[str, Any]],
        bank_id: int,
        country_code: str
    ) -> List[Dict[str, Any]]:
        """
        Validate and normalize LLM-generated prompts for database storage.

        Args:
            raw_prompts: Raw prompts from LLM
            bank_id: Bank ID
            country_code: Country code

        Returns:
            Validated and normalized prompts
        """
        print(f"DEBUG: _validate_and_normalize_prompts called with {len(raw_prompts)} raw prompts")
        logger.info(f"_validate_and_normalize_prompts called with {len(raw_prompts)} raw prompts")
        validated_prompts = []

        # Required entity types for bank statements
        required_entity_types = {
            "bank_name",
            "bank_country",
            "account_holder_name",
            "account_number",
            "customer_address",
            "currency",
            "statement_date"
        }

        # Optional entity types
        optional_entity_types = {
            "cif_number",
            "branch_address",
            "branch_name"
        }

        # Check for spatial context in prompts
        self._validate_prompts_have_spatial_context(raw_prompts)

        for raw_prompt in raw_prompts:
            try:
                # Extract required fields
                entity_type = raw_prompt.get("entity_type", "").strip().lower()
                prompt_description = raw_prompt.get("prompt_description", "").strip()

                if not entity_type or not prompt_description:
                    logger.warning(f"Skipping prompt with missing entity_type or description")
                    continue

                # Normalize entity category
                entity_category = raw_prompt.get("entity_category", "custom").upper()
                valid_categories = {
                    "PERSON", "ORGANIZATION", "LOCATION", "DATE", "NUMBER",
                    "MONEY", "IDENTIFIER", "CUSTOM"
                }
                if entity_category not in valid_categories:
                    entity_category = "CUSTOM"

                # Normalize threshold
                threshold = float(raw_prompt.get("threshold", 0.3))
                threshold = max(0.1, min(0.9, threshold))  # Clamp between 0.1 and 0.9

                # Normalize examples
                examples = raw_prompt.get("examples", [])
                if isinstance(examples, str):
                    examples = [examples]
                elif not isinstance(examples, list):
                    examples = []

                # Ensure all examples are strings (extract from dict if needed)
                normalized_examples = []
                for e in examples:
                    if isinstance(e, dict):
                        # LLM may have returned [{"text": "example"}] - extract the value
                        # Handle various dict formats: {"text": "..."}, {"value": "..."}, {"example": "..."}
                        for key in ['text', 'value', 'example', 'content']:
                            if key in e:
                                normalized_examples.append(str(e[key]).strip())
                                break
                        else:
                            # Unknown dict format, skip
                            logger.debug(f"Skipping unhandled dict format in examples: {e}")
                    elif e:
                        # Convert to string if not already
                        normalized_examples.append(str(e).strip())

                examples = [e for e in normalized_examples if e]

                # Validate and truncate validation_pattern (VARCHAR(255) in DB)
                validation_pattern = raw_prompt.get("validation_pattern")
                if validation_pattern and isinstance(validation_pattern, str):
                    validation_pattern = validation_pattern.strip()[:255]

                # Build validated prompt
                validated_prompt = {
                    "bank_id": bank_id,
                    "country_code": country_code.upper(),
                    "entity_type": entity_type,
                    "prompt_description": prompt_description,
                    "entity_category": entity_category,
                    "threshold": threshold,
                    "examples": json.dumps(examples) if examples else None,
                    "validation_pattern": validation_pattern,
                    "is_active": 1,
                    "usage_count": 0,
                    "created_by": "llm_auto_generated"
                }

                validated_prompts.append(validated_prompt)

            except Exception as e:
                logger.warning(f"Failed to validate prompt: {str(e)}")
                continue

        # Check for required entity types
        found_types = {p["entity_type"] for p in validated_prompts}
        missing_types = required_entity_types - found_types

        if missing_types:
            logger.warning(
                f"Missing required entity types for bank_id={bank_id}: {missing_types}. "
                f"These will need to be generated from additional samples."
            )

        return validated_prompts

    def _build_extraction_config(
        self,
        raw_config: Dict[str, Any],
        bank_id: int,
        country_code: str
    ) -> Dict[str, Any]:
        """
        Build extraction configuration for database storage.

        Args:
            raw_config: Raw config from LLM
            bank_id: Bank ID
            country_code: Country code

        Returns:
            Validated extraction configuration
        """
        config = {
            "bank_id": bank_id,
            "country_code": country_code.upper(),
            "default_threshold": float(raw_config.get("default_threshold", 0.3)),
            "extraction_order": raw_config.get("extraction_order"),
            "special_handling": raw_config.get("special_handling"),
            "is_active": 1,
            "prompt_generation_status": "completed"
        }

        # Validate threshold
        config["default_threshold"] = max(0.1, min(0.9, config["default_threshold"]))

        # Validate extraction_order
        if config["extraction_order"] and isinstance(config["extraction_order"], list):
            # Ensure all items are strings
            config["extraction_order"] = json.dumps([
                str(item) for item in config["extraction_order"]
            ])
        else:
            config["extraction_order"] = None

        # Validate special_handling
        if config["special_handling"]:
            if isinstance(config["special_handling"], dict):
                config["special_handling"] = json.dumps(config["special_handling"])
            elif isinstance(config["special_handling"], str):
                # Keep only non-empty strings (empty strings are invalid JSON)
                if not config["special_handling"].strip():
                    config["special_handling"] = None
                else:
                    # Wrap plain string in JSON object for database storage
                    config["special_handling"] = json.dumps({"note": config["special_handling"].strip()})
            else:
                config["special_handling"] = None
        else:
            config["special_handling"] = None

        return config

    def _build_metadata(self, llm_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build metadata from LLM result.

        Args:
            llm_result: Result from LLM service

        Returns:
            Metadata dictionary
        """
        metadata = {
            "generation_time_ms": llm_result.get("generation_time_ms"),
            "llm_provider": "openai",  # Default, can be made configurable
            "llm_model": llm_service.settings.model,
            "prompt_tokens": llm_result.get("token_usage", {}).get("prompt_tokens"),
            "completion_tokens": llm_result.get("token_usage", {}).get("completion_tokens"),
            "total_tokens": llm_result.get("token_usage", {}).get("total_tokens"),
            "reasoning": llm_result.get("reasoning", ""),
            "created_at": datetime.utcnow().isoformat()
        }

        return metadata

    async def generate_prompts_from_multiple_samples(
        self,
        bank_id: int,
        bank_abbrev: str,
        bank_name: str,
        country_code: str,
        samples: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate prompts from multiple statement samples for better coverage.

        Args:
            bank_id: Database ID of the bank
            bank_abbrev: Bank abbreviation
            bank_name: Full bank name
            country_code: ISO country code
            samples: List of samples with 'ocr_text' and optional 'generic_result'

        Returns:
            Combined prompts from all samples
        """
        if not samples:
            return {
                "prompts": [],
                "extraction_config": {},
                "metadata": {},
                "error": "No samples provided"
            }

        logger.info(f"Generating prompts from {len(samples)} samples for {bank_abbrev}/{country_code}")

        # For now, use the first sample for generation
        # Future: Implement multi-sample synthesis
        first_sample = samples[0]
        return await self.generate_prompts_for_bank(
            bank_id=bank_id,
            bank_abbrev=bank_abbrev,
            bank_name=bank_name,
            country_code=country_code,
            ocr_text=first_sample.get("ocr_text", ""),
            generic_extraction_result=first_sample.get("generic_result")
        )

    def _validate_prompts_have_spatial_context(self, prompts: List[Dict]) -> List[Dict]:
        """
        Ensure prompts include spatial/layout context keywords.

        Args:
            prompts: List of prompts from LLM

        Returns:
            The same prompts (validation only, no modification)
        """
        logger.info(f"Validating spatial context for {len(prompts)} prompts")
        spatial_keywords = [
            'right', 'left', 'below', 'above', 'near', 'adjacent',
            'top', 'bottom', 'section', 'grouped', 'table', 'header',
            'immediately', 'before', 'after', 'next to', 'with'
        ]

        for prompt in prompts:
            description = prompt.get('prompt_description', '')
            entity_type = prompt.get('entity_type', 'unknown')

            if description and not any(kw in description.lower() for kw in spatial_keywords):
                # Log warning but don't fail - the prompt might still work
                logger.warning(
                    f"Prompt for '{entity_type}' lacks spatial context keywords. "
                    f"Consider adding positional/layout hints like: right, left, below, above, "
                    f"near, adjacent, section, grouped, table, header, immediately, before, after."
                )
            else:
                logger.debug(f"Prompt for '{entity_type}' has spatial context.")

            # Log full details for customer_address prompt for debugging
            entity_type_lower = entity_type.lower() if isinstance(entity_type, str) else str(entity_type).lower()
            if 'customer_address' in entity_type_lower or 'customer address' in entity_type_lower:
                logger.info(f"Address-related prompt found - entity_type: '{entity_type}'")
                logger.info(f"  Description: {description}")
                examples = prompt.get('examples', [])
                logger.info(f"  Examples: {examples}")
                logger.info(f"  Threshold: {prompt.get('threshold', 'N/A')}")

        return prompts


# Global service instance
bank_prompt_generator = BankPromptGenerator()

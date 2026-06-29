"""
LLM Service for Bank-Specific Prompt Generation

Provides integration with LLM APIs (OpenAI, Anthropic, etc.) for generating
GLiNER2 prompts specific to bank statement formats.

Follows the same pattern as worldcheck_service.py for consistency.
"""

import asyncio
import base64
import io
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
    _vision_client: Optional[httpx.AsyncClient] = None
    _client_lock: Optional[Any] = None

    def __init__(self):
        """Initialize the LLM service."""
        self.settings = llm_settings
        self.api_url = self.settings.api_url.rstrip('/')

    @property
    def vision_model(self) -> str:
        """Get the vision model for image-based layout analysis."""
        return getattr(self.settings, 'vision_model', 'qwen2.5-vl:7b')

    @classmethod
    async def _get_client(cls, vision_request: bool = False) -> httpx.AsyncClient:
        """Get or create the shared httpx client."""
        # Use separate client for vision requests with longer timeout
        if vision_request:
            if cls._vision_client is None or cls._vision_client.is_closed:
                timeout = httpx.Timeout(
                    connect=10.0,
                    read=300.0,  # 5 minutes for vision models
                    write=10.0,
                    pool=5.0
                )
                cls._vision_client = httpx.AsyncClient(timeout=timeout)
            return cls._vision_client

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
        if cls._vision_client is not None and not cls._vision_client.is_closed:
            await cls._vision_client.aclose()
            cls._vision_client = None
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

    async def refine_prompts(
        self,
        bank_name: str,
        bank_abbrev: str,
        country_code: str,
        ocr_text: str,
        previous_prompts: Dict[str, Any],
        failed_fields: List[str],
        validation_errors: Dict[str, Any],
        previous_extraction_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Refine GLiNER prompts based on validation feedback.

        Args:
            bank_name: Full bank name
            bank_abbrev: Bank abbreviation
            country_code: ISO country code
            ocr_text: OCR text from bank statement (first 5000 chars)
            previous_prompts: Prompts used in previous extraction attempt
            failed_fields: List of fields that failed validation
            validation_errors: Detailed validation error information
            previous_extraction_result: Previous extraction results for context

        Returns:
            Dictionary with:
                - prompts: Dict of refined prompt configurations
                - reasoning: LLM's explanation of refinements
                - token_usage: Token usage statistics
                - error: Error message if refinement failed
        """
        start_time = time.time()

        try:
            # Truncate OCR text to first 5000 chars
            ocr_text_sample = ocr_text[:5000] if ocr_text else ""

            # Build the system prompt for refinement
            system_prompt = self._build_refinement_system_prompt()

            # Build the user prompt with refinement context
            user_prompt = self._build_refinement_user_prompt(
                bank_name=bank_name,
                bank_abbrev=bank_abbrev,
                country_code=country_code,
                ocr_text=ocr_text_sample,
                previous_prompts=previous_prompts,
                failed_fields=failed_fields,
                validation_errors=validation_errors,
                previous_extraction_result=previous_extraction_result
            )

            # Make the API call
            response = await self._make_refinement_api_request(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )

            # Parse the response
            result = self._parse_refinement_response(response, time.time() - start_time)

            logger.info(
                f"LLM prompt refinement completed for {bank_abbrev}/{country_code}: "
                f"{len(result.get('prompts', {}))} prompts refined, "
                f"targeting {len(failed_fields)} failed fields"
            )

            return result

        except (asyncio.CancelledError, Exception) as e:
            elapsed = time.time() - start_time
            if isinstance(e, asyncio.CancelledError):
                logger.error(f"LLM prompt refinement cancelled for {bank_abbrev}/{country_code}")
                return {
                    "prompts": {},
                    "reasoning": "",
                    "token_usage": {},
                    "error": "Prompt refinement was cancelled"
                }

            logger.error(
                f"LLM prompt refinement failed for {bank_abbrev}/{country_code}: {str(e)}"
            )
            return {
                "prompts": {},
                "reasoning": "",
                "token_usage": {},
                "error": f"Prompt refinement error: {str(e)}"
            }

    async def call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4000,
        response_format: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Call LLM with text-only (no image).

        This method is designed for text-only LLMs that don't require image input.
        It supports both OpenAI-compatible APIs and Ollama.

        Args:
            system_prompt: System instructions
            user_prompt: User message with context
            model: Optional model override (defaults to text_model from config)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
            response_format: Optional response format (e.g., {"type": "json_object"})

        Returns:
            LLM response JSON with content, token_usage, error if any
        """
        start_time = time.time()

        try:
            # Use configured text model or override
            text_model = model or getattr(self.settings, 'text_model', self.settings.model)

            logger.info(f"Using text-only LLM: {text_model}")

            # Check if using Ollama
            if self._is_ollama_url():
                logger.info("Using Ollama native API for text-only request")
                return await self._call_ollama_text(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=text_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    start_time=start_time
                )
            else:
                # Use OpenAI-compatible API
                return await self._call_openai_text(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=text_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    start_time=start_time
                )

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Text-only LLM call failed: {str(e)}")
            return {
                "content": None,
                "token_usage": {},
                "error": f"Text-only LLM call error: {str(e)}",
                "elapsed_ms": int(elapsed * 1000),
                "model": model
            }

    async def _call_ollama_text(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        start_time: float
    ) -> Dict[str, Any]:
        """
        Call Ollama native API for text-only LLM.

        Args:
            system_prompt: System instructions
            user_prompt: User message
            model: Model name
            temperature: Temperature setting
            max_tokens: Maximum tokens
            start_time: Request start time

        Returns:
            LLM response dict with content, token_usage, error if any
        """
        try:
            # Build native Ollama URL
            base_url = self.api_url.replace('/v1', '').replace('/chat/completions', '')
            url = f"{base_url}/api/chat"
            logger.info(f"Using Ollama native API: {url} with model {model}")

            payload = {
                "model": model,
                "stream": False,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }

            client = await self._get_client(vision_request=False)
            response = await client.post(url, json=payload, timeout=120.0)
            response.raise_for_status()

            result = response.json()
            elapsed = (time.time() - start_time) * 1000

            # Extract content from Ollama response
            content = None
            if "message" in result and "content" in result["message"]:
                content = result["message"]["content"]

            # Extract token usage if available
            token_usage = {}
            if "prompt_eval_count" in result:
                token_usage["prompt_tokens"] = result["prompt_eval_count"]
            if "eval_count" in result:
                token_usage["completion_tokens"] = result["eval_count"]

            logger.info(f"Ollama text LLM call completed in {elapsed:.0f}ms")

            return {
                "content": content,
                "token_usage": token_usage,
                "model": model,
                "elapsed_ms": int(elapsed),
                "provider": "ollama"
            }

        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            logger.error(f"Ollama text LLM call failed: {str(e)}")
            return {
                "content": None,
                "token_usage": {},
                "error": f"Ollama text LLM call error: {str(e)}",
                "elapsed_ms": int(elapsed),
                "model": model
            }

    async def _call_openai_text(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        response_format: Optional[Dict[str, str]],
        start_time: float
    ) -> Dict[str, Any]:
        """
        Call OpenAI-compatible API for text-only LLM.

        Args:
            system_prompt: System instructions
            user_prompt: User message
            model: Model name
            temperature: Temperature setting
            max_tokens: Maximum tokens
            response_format: Optional response format
            start_time: Request start time

        Returns:
            LLM response dict with content, token_usage, error if any
        """
        try:
            url = f"{self.api_url}/chat/completions"

            headers = {
                "Content-Type": "application/json",
            }

            # Only add Authorization header if api_key is not empty
            if self.settings.api_key:
                headers["Authorization"] = f"Bearer {self.settings.api_key}"

            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            # Add response format if specified (and not using Ollama)
            if response_format and not self._is_ollama_url():
                payload["response_format"] = response_format

            logger.info(f"Calling OpenAI-compatible API: {url} with model {model}")

            client = await self._get_client(vision_request=False)
            response = await client.post(url, json=payload, headers=headers, timeout=120.0)
            response.raise_for_status()

            result = response.json()
            elapsed = (time.time() - start_time) * 1000

            # Extract content from response
            content = None
            token_usage = {}

            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0].get("message", {}).get("content")

            if "usage" in result:
                token_usage = result["usage"]

            logger.info(f"OpenAI-compatible text LLM call completed in {elapsed:.0f}ms")

            return {
                "content": content,
                "token_usage": token_usage,
                "model": model,
                "elapsed_ms": int(elapsed),
                "provider": "openai_compatible"
            }

        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            logger.error(f"OpenAI-compatible text LLM call failed: {str(e)}")
            return {
                "content": None,
                "token_usage": {},
                "error": f"OpenAI-compatible text LLM call error: {str(e)}",
                "elapsed_ms": int(elapsed),
                "model": model
            }

    def _detect_image_type(self, image_bytes: bytes) -> str:
        """Detect image type from magic bytes."""
        if image_bytes.startswith(b'\x89PNG'):
            return 'png'
        elif image_bytes.startswith(b'\xff\xd8\xff'):
            return 'jpeg'
        elif image_bytes.startswith(b'GIF87a') or image_bytes.startswith(b'GIF89a'):
            return 'gif'
        elif image_bytes.startswith(b'BM'):
            return 'bmp'
        elif image_bytes.startswith(b'II*\x00') or image_bytes.startswith(b'MM\x00*'):
            return 'tiff'
        else:
            # Default to JPEG for most bank statement images
            return 'jpeg'

    def _ensure_jpeg_format(self, image_bytes: bytes) -> bytes:
        """
        Ensure image is in JPEG or PNG format for qwen3-vl compatibility.

        qwen3-vl supports both JPEG and PNG formats, so we pass-through
        images already in these formats and only convert other formats.
        Also handles PDF-to-image conversion for bank statements.

        Args:
            image_bytes: Original image bytes

        Returns:
            JPEG or PNG formatted image bytes
        """
        try:
            # JPEG magic bytes: FF D8 FF
            # PNG magic bytes: 89 50 4E 47
            # PDF magic bytes: 25 50 44 46 (%PDF)
            jpeg_magic = b'\xff\xd8\xff'
            png_magic = b'\x89PNG'
            pdf_magic = b'%PDF'

            # If already JPEG or PNG, return as-is (avoids unnecessary conversion)
            if image_bytes.startswith(jpeg_magic) or image_bytes.startswith(png_magic):
                logger.debug("Image already in JPEG/PNG format, skipping conversion")
                return image_bytes

            # Handle PDF files - convert to image first
            if image_bytes.startswith(pdf_magic):
                logger.debug("Detected PDF file, converting to JPEG image")
                return self._convert_pdf_to_image(image_bytes)

            from PIL import Image

            # Open image
            img = Image.open(io.BytesIO(image_bytes))

            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Save as JPEG
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=95)
            logger.debug(f"Converted image to JPEG format (original: {img.format})")
            return output.getvalue()

        except ImportError:
            logger.warning("PIL not available, returning original image format")
            return image_bytes
        except Exception as e:
            logger.error(f"Failed to convert image to JPEG: {str(e)}")
            return image_bytes

    def _convert_pdf_to_image(self, pdf_bytes: bytes) -> bytes:
        """Convert PDF first page to JPEG image for qwen3-vl processing."""
        try:
            import fitz  # PyMuPDF
            from PIL import Image

            pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            if len(pdf_doc) > 0:
                page = pdf_doc[0]
                pix = page.get_pixmap(dpi=150)

                # Convert to PNG first, then to JPEG via PIL for better quality
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))

                # Convert to RGB for JPEG compatibility
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Save as JPEG
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG', quality=95)
                result = img_byte_arr.getvalue()

                logger.debug(f"Converted PDF to JPEG image: {len(result)} bytes")
                pdf_doc.close()
                return result
            pdf_doc.close()
            return pdf_bytes
        except Exception as e:
            logger.error(f"Failed to convert PDF to image: {str(e)}")
            return pdf_bytes

    def _ensure_token_budget(self, image_bytes: bytes) -> bytes:
        """
        Prepare image for vision LLM by handling format conversions.

        NOTE: Token-aware sizing is now handled by the preprocessing service.
        This method only handles:
        - PDF to image conversion (for PDF inputs)
        - Color space normalization (CMYK, RGBA, etc. → RGB)

        The preprocessing service ensures images are already sized correctly
        for the vision LLM token budget (max 1078px for ~6000 tokens with Qwen3.5).

        Args:
            image_bytes: Image bytes (JPEG/PNG/PDF supported)

        Returns:
            Processed image bytes (JPEG format)
        """
        try:
            from PIL import Image
            import io

            # Convert PDF to image if needed
            if image_bytes.startswith(b'%PDF'):
                image_bytes = self._convert_pdf_to_image(image_bytes)

            img = Image.open(io.BytesIO(image_bytes))

            # Handle CMYK color space (common in scanned documents)
            if img.mode == 'CMYK':
                logger.info("Converting CMYK image to RGB")
                try:
                    # Convert CMYK to RGB
                    img = img.convert('RGB')
                except Exception as e:
                    logger.error(f"CMYK to RGB conversion failed: {e}")
                    # Fallback: try converting through RGB mode directly
                    img = img.convert('RGB')
            elif img.mode != 'RGB':
                # Handle other color modes (RGBA, L, P, etc.)
                img = img.convert('RGB')

            # Save as JPEG
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=95)
            result = output.getvalue()

            return result

        except ImportError:
            logger.warning("PIL not available, returning original image")
            return image_bytes
        except Exception as e:
            logger.error(f"Failed to prepare image for LLM: {str(e)}")
            return image_bytes

    def _resize_image(self, image_bytes: bytes, max_size: int, quality: int = 85) -> bytes:
        """
        Resize image if it exceeds max_size bytes.

        Args:
            image_bytes: Original image bytes
            max_size: Maximum target size in bytes
            quality: JPEG quality (1-100)

        Returns:
            Resized image bytes
        """
        try:
            from PIL import Image

            # Open image
            img = Image.open(io.BytesIO(image_bytes))

            # Convert to RGB if needed (avoid RGBA mode issues)
            if img.mode in ('RGBA', 'LA', 'P'):
                # Create white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                if img.mode in ('RGBA', 'LA'):
                    background.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
                    img = background
                else:
                    img = img.convert('RGB')
            elif img.mode != 'RGB':
                img = img.convert('RGB')

            # Start with original size
            width, height = img.size
            scale_factor = 1.0

            # Calculate scale factor iteratively
            output = io.BytesIO()
            while scale_factor > 0.1:  # Don't scale down more than 90%
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)

                if new_width < 100 or new_height < 100:  # Minimum dimensions
                    break

                resized = img.resize((new_width, new_height), Image.LANCZOS)
                output.seek(0)
                output.truncate()
                resized.save(output, format='JPEG', quality=quality, optimize=True)
                result_bytes = output.getvalue()

                if len(result_bytes) <= max_size:
                    logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}: {len(result_bytes)} bytes")
                    return result_bytes

                # Reduce scale by 10% and try again
                scale_factor -= 0.1

            # If we still can't get it small enough, return last attempt
            logger.warning(f"Could not resize image below {max_size} bytes, returning best effort: {len(result_bytes)} bytes")
            return result_bytes

        except ImportError:
            logger.warning("PIL not available, returning original image")
            return image_bytes
        except Exception as e:
            logger.error(f"Failed to resize image: {str(e)}")
            return image_bytes

    async def call_vision_llm(
        self,
        image_bytes: bytes,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 8000
    ) -> Dict[str, Any]:
        """
        Call vision LLM with image input using Ollama API.

        Args:
            image_bytes: Image data (JPEG/PNG/PDF supported)
            system_prompt: System prompt with instructions
            user_prompt: User prompt with field descriptions
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            {
                "content": str,
                "model": str,
                "token_usage": {...},
                "elapsed_ms": int
            }
        """
        import time
        import base64

        start_time = time.time()

        try:
            # Apply token-based image sizing for qwen3.5+ and modern vision LLMs
            # This replaces the old 28-pixel alignment requirement
            processed_bytes = self._ensure_token_budget(image_bytes)

            # Encode to base64
            image_base64 = base64.b64encode(processed_bytes).decode('utf-8')

            # Build Ollama vision API payload
            # Ollama uses 'images' array with base64 strings, not OpenAI's image_url format
            payload = {
                "model": self.vision_model,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                        "images": [image_base64]
                    }
                ],
                "stream": False,
                "think": False,  # Disable thinking mode at top level
                "format": "json",  # Force JSON output from model
                "options": {
                    "temperature": temperature,
                    "num_predict": 1000,  # Limit output to ~1000 tokens to prevent excessive generation
                    "num_ctx": self.settings.num_ctx,  # Use configured context window size (default 8192)
                    "think": False,  # Also disable thinking mode in options
                    "reasoning": False  # Also disable reasoning mode
                }
            }

            # Use vision client with longer timeout
            client = await self._get_client(vision_request=True)

            logger.info(f"Calling vision LLM: {self.vision_model} with image ({len(processed_bytes)} bytes)")

            # Build Ollama API URL
            base_url = self.api_url.replace('/v1', '').replace('/chat/completions', '')
            url = f"{base_url}/api/chat"

            # Call Ollama API with explicit read timeout to interrupt stuck models
            # Using httpx.Timeout to only override read timeout (not connect/write/pool)
            request_timeout = httpx.Timeout(
                connect=10.0,  # Connection timeout
                read=15.0,     # Read timeout - fail fast if model gets stuck generating Thai text
                write=10.0,    # Write timeout
                pool=5.0        # Pool timeout
            )
            response = await client.post(url, json=payload, timeout=request_timeout)

            elapsed_ms = int((time.time() - start_time) * 1000)

            # Parse response
            response.raise_for_status()
            result = response.json()

            # Log raw response for debugging
            logger.debug(f"=== RAW OLLAMA VISION RESPONSE ===")
            logger.debug(f"Status: {response.status_code}")
            logger.debug(f"Result keys: {list(result.keys()) if result else 'None'}")
            logger.debug(f"Full result (first 1000 chars): {str(result)[:1000]}")
            logger.debug(f"=== END RAW OLLAMA VISION RESPONSE ===")

            if result and 'message' in result:
                content = result['message'].get('content', '')

                # If content is empty, check for alternative response fields
                # DO NOT use 'thinking' field as it contains reasoning, not the final answer
                if not content:
                    # Log the full message structure for debugging
                    logger.debug(f"Content field empty. Message keys: {list(result['message'].keys())}")

                    # Check for alternative fields that might contain the actual response
                    # Some models may use 'response', 'output', or other field names
                    for alt_field in ['response', 'output', 'answer', 'result']:
                        if alt_field in result['message']:
                            content = result['message'].get(alt_field, '')
                            if content:
                                logger.debug(f"Using '{alt_field}' field for content")
                                break

                    # If still empty, check if thinking contains JSON (not just reasoning)
                    if not content and 'thinking' in result['message']:
                        thinking_content = result['message'].get('thinking', '')
                        # Only use thinking if it looks like JSON (starts with {)
                        if thinking_content.strip().startswith('{'):
                            content = thinking_content
                            logger.debug(f"Using 'thinking' field for content (appears to be JSON)")
                        else:
                            logger.warning(
                                f"Content and response fields empty. 'thinking' contains non-JSON reasoning. "
                                f"Model may not be generating JSON output as expected."
                            )

                logger.debug(f"Extracted content (first 500 chars): {content[:500] if content else 'EMPTY'}")

                # Extract token usage if available
                token_usage = result.get('prompt_eval_count', {})

                logger.info(f"Vision LLM call completed in {elapsed_ms}ms")

                return {
                    "content": content,
                    "model": self.vision_model,
                    "token_usage": token_usage,
                    "elapsed_ms": elapsed_ms
                }
            else:
                logger.error(f"Unexpected vision LLM response format")
                return {
                    "error": "Unexpected response format from vision LLM",
                    "elapsed_ms": elapsed_ms
                }

        except httpx.HTTPStatusError as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            # Capture Ollama's error response body for debugging
            error_body = e.response.text[:500] if hasattr(e.response, 'text') else str(e)
            logger.error(f"Vision LLM HTTP {e.response.status_code}: {error_body}")
            return {
                "error": f"HTTP {e.response.status_code}: {error_body}",
                "elapsed_ms": elapsed_ms
            }
        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.error(f"Vision LLM call failed: {str(e)}")
            return {
                "error": str(e),
                "elapsed_ms": elapsed_ms
            }

    def _is_ollama_url(self) -> bool:
        """Check if the current API URL is an Ollama instance."""
        api_url_lower = self.api_url.lower()

        # Check for Ollama-specific terms
        if 'ollama' in api_url_lower:
            return True

        # Check for localhost/127.0.0.1 with typical Ollama ports or OpenAI-compatible endpoints
        # Ollama typically runs on localhost with custom ports
        if 'localhost' in api_url_lower or '127.0.0.1' in api_url_lower:
            # Check for OpenAI-compatible endpoints on localhost (common for Ollama)
            if '/v1/chat/completions' in api_url_lower or '/v1/' in api_url_lower:
                return True
            # Check for common Ollama ports
            if any(port in api_url_lower for port in [':11434', ':1177', ':11435']):
                return True

        return False

    def _build_refinement_system_prompt(self) -> str:
        """Build the system prompt for LLM refinement."""
        return """You are a GLiNER2 prompt refinement expert specializing in bank statement analysis.

GLiNER2 is a schema-based information extraction model that uses natural language descriptions to identify and extract entities from text.

Your task is to ANALYZE WHY previous extraction prompts FAILED and CREATE IMPROVED prompts that will succeed.

Key principles:
1. IDENTIFY THE ROOT CAUSE - Why did the previous prompt fail to extract this field?
2. BE MORE SPECIFIC - Use exact labels, positioning, and formatting patterns
3. PROVIDE BETTER EXAMPLES - Extract actual examples where the field appears
4. ADJUST THRESHOLDS - Lower thresholds if confidence is too high
5. TRY ALTERNATIVE PATTERNS - If the field appears in multiple ways, cover all variations

**CRITICAL: Include Spatial and Layout Context**

When refining prompts, you MUST enhance descriptions with:
1. **Positional hints**: WHERE the value appears relative to its label (right, below, adjacent, near)
2. **Document layout context**: WHICH section of the document (header, summary table, transaction list)
3. **Spatial relationships**: How the field relates to other fields (grouped with, appears near)

**Multi-line Field Extraction:**
When refining prompts for multi-line fields like addresses:
- Emphasize capturing ALL lines of the field, not just the first line
- Describe how lines are GROUPED TOGETHER spatially
- Mention the total number of lines typically present
- Provide examples that show the complete multi-line format

Common spatial issues that cause failures:
- Field appears in unexpected location (e.g., currency inside transaction tables, not just header)
- Label positioned differently (e.g., account number BELOW name, not beside it)
- Multi-line fields (e.g., addresses span multiple lines, need to capture all components)

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

    def _build_refinement_user_prompt(
        self,
        bank_name: str,
        bank_abbrev: str,
        country_code: str,
        ocr_text: str,
        previous_prompts: Dict[str, Any],
        failed_fields: List[str],
        validation_errors: Dict[str, Any],
        previous_extraction_result: Dict[str, Any]
    ) -> str:
        """Build the user prompt with refinement context."""
        # Build prompt parts to avoid f-string nesting issues
        failed_fields_str = ', '.join(failed_fields)
        previous_prompts_str = json.dumps(previous_prompts, indent=2, default=str)
        previous_extraction_str = json.dumps(previous_extraction_result, indent=2, default=str)
        validation_errors_str = json.dumps(validation_errors, indent=2, default=str)

        prompt = f"""Analyze this failed bank statement extraction and refine the GLiNER2 prompts.

BANK CONTEXT:
- Bank: {bank_name} ({bank_abbrev})
- Country: {country_code}

FAILED FIELDS (these need to be extracted):
{failed_fields_str}

PREVIOUS PROMPTS (these failed to extract the required fields):
{previous_prompts_str}

PREVIOUS EXTRACTION RESULT (what was actually extracted):
{previous_extraction_str}

VALIDATION ERRORS:
{validation_errors_str}

STATEMENT OCR TEXT (first 5000 chars):
{ocr_text}

ANALYSIS TASK:
For each FAILED field, analyze WHY the previous prompt failed and create a BETTER prompt.

Common failure reasons:
1. Description too generic - make it more specific to THIS bank's format
2. Examples don't match actual text - provide real examples from the OCR text
3. Threshold too high - lower it to capture more matches
4. Label text not mentioned - include the exact label text that precedes the field
5. **Missing spatial context** - Add WHERE the field appears (right of label, below header, in table, etc.)
6. **Layout not considered** - Describe document section and spatial relationships
7. Field not visible in sample - note if field appears on a different page

**SPATIAL CONTEXT ENHANCEMENT:**
When refining prompts, analyze and include:
- Field position relative to labels: "IMMEDIATELY TO THE RIGHT", "BELOW", "ADJACENT TO"
- Document section: "in HEADER section", "in SUMMARY table at BOTTOM", "in TRANSACTION list"
- Spatial relationships: "GROUPED WITH address components", "NEAR balance figures"
- Layout patterns: "in TABLE with date-description-amount columns", "multi-line address block"

For each failed field, provide:
- **entity_type**: The field name (e.g., "address_country")
- **prompt_description**: MORE SPECIFIC description with spatial context (include exact labels, positioning, layout, formatting)
- **entity_category**: PERSON, ORGANIZATION, LOCATION, DATE, NUMBER, MONEY, IDENTIFIER, or custom
- **examples**: 2-4 REAL examples extracted from this statement's OCR text
- **validation_pattern**: Suggested regex pattern for validation (optional)
- **threshold**: Recommended confidence threshold (try 0.2-0.4 if previous was 0.5+)

CRITICAL FOCUS: Look at the OCR text and find EXACTLY how each failed field appears. Provide real examples from the text.

NOTE: You only need to provide refined prompts for the FAILED fields listed above.
The calling code will merge your refined prompts with the original prompts that succeeded.
This ensures we maintain extraction quality for fields that were already working.

Return JSON format:
{{
  "prompts": {{
    "address_country": {{
      "entity_type": "address_country",
      "prompt_description": "Country code from customer address, appears after postal code or as last line of address",
      "entity_category": "LOCATION",
      "examples": ["THAILAND", "TH"],
      "validation_pattern": "^[A-Z]{{2}}$",
      "threshold": 0.3
    }}
  }},
  "reasoning": "Brief explanation of what was wrong with previous prompts and how the refined prompts fix it"
}}
"""
        return prompt

    async def _make_refinement_api_request(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> Dict[str, Any]:
        """Make an API request to the LLM service for refinement."""
        url = f"{self.api_url}/chat/completions"

        headers = {
            "Content-Type": "application/json",
        }

        # Only add Authorization header if api_key is not empty (for services like OpenAI)
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"

        payload = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.settings.prompt_refinement_temperature,
            "max_tokens": self.settings.prompt_refinement_max_tokens,
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

    def _parse_refinement_response(
        self,
        response: Dict[str, Any],
        refinement_time_ms: float
    ) -> Dict[str, Any]:
        """Parse the LLM API response for refinement."""
        try:
            # Extract content from response
            choices = response.get("choices", [])
            if not choices:
                return {
                    "prompts": {},
                    "reasoning": "",
                    "token_usage": {},
                    "error": "Empty response from LLM"
                }

            content = choices[0].get("message", {}).get("content", "")

            # Parse JSON content
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as e:
                # Log the actual content for debugging
                logger.error(f"Failed to parse LLM refinement JSON: {str(e)}")
                logger.debug(f"LLM refinement content (first 1000 chars): {content[:1000]}")
                # Try to extract JSON from markdown code block
                if "```json" in content:
                    start = content.find("```json") + 7
                    end = content.find("```", start)
                    if end > start:
                        try:
                            parsed = json.loads(content[start:end].strip())
                        except json.JSONDecodeError as e2:
                            logger.error(f"Failed to parse JSON from markdown block: {str(e2)}")
                            logger.debug(f"Markdown block content: {content[start:end]}")
                            raise
                elif "```" in content:
                    start = content.find("```") + 3
                    end = content.find("```", start)
                    if end > start:
                        try:
                            parsed = json.loads(content[start:end].strip())
                        except json.JSONDecodeError as e2:
                            logger.error(f"Failed to parse JSON from code block: {str(e2)}")
                            logger.debug(f"Code block content: {content[start:end]}")
                            raise
                else:
                    raise

            # Extract token usage
            usage = response.get("usage", {})
            token_usage = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0)
            }

            # Convert prompts list to dict format for consistency
            prompts_dict = {}
            prompts_list = parsed.get("prompts", [])
            if isinstance(prompts_list, list):
                for prompt in prompts_list:
                    entity_type = prompt.get("entity_type")
                    if entity_type:
                        prompts_dict[entity_type] = prompt
            elif isinstance(prompts_list, dict):
                prompts_dict = prompts_list

            return {
                "prompts": prompts_dict,
                "reasoning": parsed.get("reasoning", ""),
                "token_usage": token_usage,
                "refinement_time_ms": int(refinement_time_ms * 1000)
            }

        except Exception as e:
            logger.error(f"Failed to parse LLM refinement response: {str(e)}")
            return {
                "prompts": {},
                "reasoning": "",
                "token_usage": response.get("usage", {}),
                "error": f"Failed to parse response: {str(e)}"
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

**CRITICAL: Include Spatial and Layout Context**

For each prompt description, you MUST include:
1. **Positional hints**: WHERE the value appears relative to its label (right, below, adjacent, near)
2. **Document layout context**: WHICH section of the document (header, summary table, transaction list)
3. **Spatial relationships**: How the field relates to other fields (grouped with, appears near)

Examples of spatial-aware prompts:
- "Account number appears IMMEDIATELY TO THE RIGHT of 'Account No:' label in the account information section at the top of the statement"
- "Currency symbol (Rs, $) appears BEFORE or AFTER monetary amounts in transaction tables and summary sections"
- "City name appears BELOW state/region information in the address block, typically grouped with postal code"

**CRITICAL: Multi-line Field Extraction**

For fields that span multiple lines (especially addresses):
- **Capture ALL lines**: Extract the COMPLETE multi-line value, not just the first line
- **Line grouping**: Lines that are GROUPED TOGETHER spatially form a single field value
- **Address blocks**: Addresses typically appear as 3-5 consecutive lines showing street, city, state, postal code, country
- **Combine all lines**: GLiNER should extract and combine all address lines into a single text value

Multi-line field examples:
- "Address: 123 MAIN STREET\nAPT 4B\nNEW YORK NY 10001" → Extract ALL FOUR lines as one address value
- "Customer:\nJOHN SMITH\n123 MAY STREET\nSINGAPORE 560123" → Address is the last TWO lines combined

Your prompts should enable GLiNER2 to find values even when labels are variations of expected patterns.

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

DETECTED BANK CONTEXT:
This is a bank statement from {bank_name} ({bank_abbrev}), a bank registered in {country_code}.
Generate prompts specifically optimized for {bank_name}'s statement format in {country_code}.

Statement OCR Text (first 5000 chars):
{ocr_text}
"""

        if generic_extraction_result:
            # Extract detected bank_name and bank_country if available
            detected_bank_name = None
            detected_bank_country = None

            if 'bank_name' in generic_extraction_result and generic_extraction_result['bank_name']:
                detected_bank_name = generic_extraction_result['bank_name'].get('value', '') if isinstance(generic_extraction_result['bank_name'], dict) else generic_extraction_result['bank_name']

            if 'bank_country' in generic_extraction_result and generic_extraction_result['bank_country']:
                detected_bank_country = generic_extraction_result['bank_country'].get('value', '') if isinstance(generic_extraction_result['bank_country'], dict) else generic_extraction_result['bank_country']

            if detected_bank_name or detected_bank_country:
                prompt += f"""
STAGE 1 IDENTIFICATION RESULTS:
- Detected Bank Name: {detected_bank_name or 'Not detected'}
- Detected Bank Country: {detected_bank_country or 'Not detected'}

Full Generic Extraction Results (for reference):
{json.dumps(generic_extraction_result, indent=2, default=str)}
"""

        prompt += """

IMPORTANT DISTINCTIONS:
- account_holder_name: Person's name (e.g., "JOHN SMITH") - NEVER extract as branch_name
- branch_name: Location/area name (e.g., "Marine Parade") - NEVER extract a person's name
- customer_address: Customer's home address - contains block/street details
- branch_address: Bank's branch location - appears with "Branch Address" labels

**Document Layout Patterns to Consider:**

Bank statements typically follow these spatial patterns:
- **Header section**: Bank name, logo, statement title at TOP
- **Account info**: Account holder name, number, address near TOP LEFT or TOP RIGHT
- **Summary table**: Opening/closing balances, currency in CENTER or BOTTOM sections
- **Transaction lists**: Date, description, amount columns in TABLE format
- **Address blocks**: Multi-line addresses with street, city, state, postal code GROUPED TOGETHER

When generating prompts, consider these spatial patterns even though you only see text.

Task: Generate GLiNER2 entity descriptions optimized specifically for this bank's statement format.

For each of the following fields, analyze how it appears in this {bank_name} statement from {country_code}:

1. **bank_name**: "{bank_name}" or variant (e.g., "{bank_abbrev}") - extract exact bank name as it appears in logo/headers
2. **bank_country**: "{country_code}" - extract the country code for this bank's registered location
3. **account_holder_name**: Customer/account holder name
4. **account_number**: Bank account number
5. **cif_number**: Customer Identification File number (if present)
6. **customer_address**: Customer's residential/postal address
7. **branch_address**: Bank branch address (separate from customer)
8. **branch_name**: Branch name or identifier
9. **currency**: Currency code or display name
10. **statement_date**: Statement date or period
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
      "prompt_description": "Customer's full name appearing IMMEDIATELY TO THE RIGHT OF labels like 'Name:', 'Customer Name:', 'Account Holder:' in the account information section at the TOP of the statement. Format: 2-6 words in ALL CAPS, may include middle initial.",
      "entity_category": "PERSON",
      "examples": ["RAM CHANDRA", "SMT. LAXMI DEVI", "VINEETH NARASIMHAN"],
      "validation_pattern": "^[A-Z\\s\\.]+$",
      "threshold": 0.3
    },
    {
      "entity_type": "currency",
      "prompt_description": "Currency code (INR, USD, EUR, SGD) or symbol (Rs, $, €) appearing BEFORE OR AFTER monetary amounts throughout the statement. Commonly found NEAR balance figures, in transaction tables, and summary sections. Look for patterns like 'Rs 5,000', 'INR 10000.50', 'USD 250.00'",
      "entity_category": "CUSTOM",
      "examples": ["INR", "Rs", "USD", "SGD"],
      "validation_pattern": "^(INR|USD|EUR|SGD|GBP|Rs|\\$|€|£)$",
      "threshold": 0.4
    },
    {
      "entity_type": "customer_address",
      "prompt_description": "COMPLETE multi-line customer address appearing as a GROUPED block of 3-5 consecutive lines. Typically located BELOW account holder name in the account information section. MUST capture ALL lines of the address block combined: street address (plot/flat number, street name, area), city, state/region, postal code, and country. Look for address patterns grouped together on consecutive lines. DO NOT extract just the first line - combine ALL address lines into a single value.",
      "entity_category": "LOCATION",
      "examples": [
        "N 52 MAYA INDRAPRASTHA APARTMENTS J P NAGAR 6TH PHASE BANGALORE KARNATAKA INDIA 560078",
        "123 MAIN STREET APT 4B NEW YORK NY 10001 USA",
        "BLK 29 MARINE CRESCENT #09-112 SINGAPORE 370029"
      ],
      "validation_pattern": "^[A-Z0-9\\s\\-\\.,#]+$",
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
        }

        # Only add Authorization header if api_key is not empty (for services like OpenAI)
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"

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
            except json.JSONDecodeError as e:
                # Log the actual content for debugging
                logger.error(f"Failed to parse LLM JSON response: {str(e)}")
                logger.debug(f"LLM response content (first 1000 chars): {content[:1000]}")
                # Try to extract JSON from markdown code block
                if "```json" in content:
                    start = content.find("```json") + 7
                    end = content.find("```", start)
                    if end > start:
                        try:
                            parsed = json.loads(content[start:end].strip())
                        except json.JSONDecodeError as e2:
                            logger.error(f"Failed to parse JSON from markdown block: {str(e2)}")
                            logger.debug(f"Markdown block content: {content[start:end]}")
                            raise
                elif "```" in content:
                    start = content.find("```") + 3
                    end = content.find("```", start)
                    if end > start:
                        try:
                            parsed = json.loads(content[start:end].strip())
                        except json.JSONDecodeError as e2:
                            logger.error(f"Failed to parse JSON from code block: {str(e2)}")
                            logger.debug(f"Code block content: {content[start:end]}")
                            raise
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

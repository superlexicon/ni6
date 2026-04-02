"""
World Check One API Service

Provides integration with LSEG World Check One API for screening individuals
against watchlists during verification. Uses HMAC-SHA256 signature authentication.

See: https://developers.lseg.com/en/api-catalog/customer-and-third-party-screening/world-check-one-api/quick-start
"""

import asyncio
import hashlib
import hmac
import base64
import uuid
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from urllib.parse import urlparse
import httpx

from app.config.worldcheck_config import worldcheck_settings
from app.core.logger import get_logger
from app.utils.string_matching import jaro_winkler_similarity


logger = get_logger()


class WorldCheckService:
    """Service for screening individuals against World Check One watchlists"""

    # Class-level shared client (created on first use)
    _client: Optional[httpx.AsyncClient] = None
    _client_lock: Optional[Any] = None

    def __init__(self):
        self.settings = worldcheck_settings
        self.base_url = self.settings.api_url.rstrip('/')
        self.gateway_url = self.settings.gateway_url.rstrip('/')

    @classmethod
    async def _get_client(cls) -> httpx.AsyncClient:
        """Get or create the shared httpx client."""
        if cls._client is None or cls._client.is_closed:
            # Use separate timeouts for connect, read, write, and pool
            timeout = httpx.Timeout(
                connect=10.0,    # 10 seconds to establish connection
                read=30.0,       # 30 seconds to read response
                write=10.0,      # 10 seconds to write request
                pool=5.0         # 5 seconds to get connection from pool
            )
            cls._client = httpx.AsyncClient(timeout=timeout)
        return cls._client

    @classmethod
    async def close_client(cls):
        """Close the shared client (call on shutdown)."""
        if cls._client is not None and not cls._client.is_closed:
            await cls._client.aclose()
            cls._client = None

    async def screen_individual(
        self,
        full_name: str,
        date_of_birth: Optional[str] = None,
        country: Optional[str] = None,
        gender: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Screen an individual against World Check One watchlists.

        Args:
            full_name: Full name of the individual
            date_of_birth: Date of birth in YYYY-MM-DD format
            country: Country code (ISO 3166-1 alpha-3)
            gender: Gender (M/F or MALE/FEMALE)

        Returns:
            Dictionary containing:
                - is_match: bool - Whether a watchlist match was found
                - match_details: list - Details of any matches found
                - screening_id: str - Unique ID for this screening
                - raw_response: dict - Full API response for audit
                - error: str (optional) - Error message if screening failed
        """
        screening_id = str(uuid.uuid4())

        try:
            # Build the screening request
            request_body = self._build_screening_request(
                full_name=full_name,
                date_of_birth=date_of_birth,
                country=country,
                gender=gender,
                case_id=screening_id
            )

            # Make the API call (POST /cases?screen=SYNC)
            # Endpoint for signing (without query params per LSEG docs)
            endpoint = f"{self.gateway_url}/cases"
            response = await self._make_api_request(endpoint, request_body, query_params="screen=SYNC")

            # Parse the response with enhanced matching logic
            result = self._parse_screening_response(
                response,
                screening_id,
                submitted_name=full_name,
                submitted_dob=date_of_birth,
                submitted_country=country
            )

            logger.info(
                f"World Check screening completed for '{full_name}': "
                f"is_match={result['is_match']}, screening_id={screening_id}"
            )

            return result

        except (asyncio.CancelledError, Exception) as e:
            # Handle CancelledError separately - this indicates the operation was cancelled
            if isinstance(e, asyncio.CancelledError):
                logger.error(f"World Check screening cancelled for screening_id={screening_id}")
                return {
                    "is_match": None,
                    "match_details": [],
                    "screening_id": screening_id,
                    "raw_response": None,
                    "error": "Watchlist screening was cancelled"
                }
            # Handle other exceptions
            logger.error(
                f"World Check screening failed for screening_id={screening_id}: {str(e)}"
            )
            return {
                "is_match": None,
                "match_details": [],
                "screening_id": screening_id,
                "raw_response": None,
                "error": f"Watchlist screening error: {str(e)}"
            }

    def _normalize_date_to_iso(self, date_input) -> Optional[str]:
        """
        Normalize date to ISO 8601 format (YYYY-MM-DD) for World Check API.

        Handles both date objects and various string formats from passport extraction.
        World Check requires dates in ISO 8601 format (YYYY-MM-DD).
        """
        if not date_input:
            return None

        # Handle date objects directly
        from datetime import date
        if isinstance(date_input, date):
            return date_input.strftime("%Y-%m-%d")

        # Handle strings - first check if already in ISO format
        date_str = str(date_input).strip()

        # If already in YYYY-MM-DD format (valid ISO 8601), return as-is
        if len(date_str) == 10 and date_str[4] == '-' and date_str[7] == '-':
            try:
                # Validate it's a real date
                datetime.strptime(date_str, "%Y-%m-%d")
                return date_str
            except ValueError:
                pass  # Not a valid ISO date, continue parsing

        # Try other formats
        date_formats = [
            "%d-%m-%Y",    # 15-01-1990
            "%Y/%m/%d",    # 1990/01/15
            "%d/%m/%Y",    # 15/01/1990
            "%Y%m%d",      # 19900115
            "%d%m%Y",      # 15011990
            "%d %b %Y",    # 10 NOV 1982 (abbreviated month)
            "%d %B %Y",    # 10 November 1982 (full month)
            "%d-%b-%Y",    # 10-Nov-1982
            "%d/%b/%Y",    # 10/Nov/1982
        ]

        # Try original case first, then title case for month names
        for date_variant in [date_str, date_str.title()]:
            for fmt in date_formats:
                try:
                    parsed = datetime.strptime(date_variant, fmt)
                    return parsed.strftime("%Y-%m-%d")
                except ValueError:
                    continue

        logger.warning(f"Could not normalize date format for World Check: {date_input}")
        return None

    def _normalize_gender(self, gender_input: Optional[str]) -> str:
        """
        Normalize gender to World Check format (MALE/FEMALE/UNSPECIFIED).

        Args:
            gender_input: Gender input (M/F/MALE/FEMALE or None)

        Returns:
            Normalized gender string (MALE, FEMALE, or UNSPECIFIED)
        """
        if not gender_input:
            return "UNSPECIFIED"

        gender_upper = str(gender_input).strip().upper()
        if gender_upper in ['M', 'MALE']:
            return "MALE"
        elif gender_upper in ['F', 'FEMALE']:
            return "FEMALE"
        return "UNSPECIFIED"

    def _build_screening_request(
        self,
        full_name: str,
        date_of_birth: Optional[str],
        country: Optional[str],
        gender: Optional[str],
        case_id: str
    ) -> Dict[str, Any]:
        """Build the screening request payload for World Check One API"""

        # Debug logging for input values
        logger.info(f"Building World Check request:")
        logger.debug(f"  - date_of_birth: {date_of_birth} (type: {type(date_of_birth).__name__})")
        logger.info(f"  - gender: {gender} (type: {type(gender).__name__ if gender else 'None'})")
        logger.info(f"  - country: {country}")

        request = {
            "groupId": self.settings.group_id,
            "entityType": "INDIVIDUAL",
            "caseId": case_id,
            "name": full_name,
            "providerTypes": ["WATCHLIST"]
        }

        # Add secondary fields - always include gender as it's required by group config
        secondary_fields = []

        # SFCT_1 is Gender in World Check One (always send - required by group config)
        normalized_gender = self._normalize_gender(gender)
        logger.info(f"  - normalized_gender: {normalized_gender}")
        secondary_fields.append({
            "typeId": "SFCT_1",
            "value": normalized_gender
        })

        if date_of_birth:
            # SFCT_2 is Date of Birth in World Check One
            # Normalize to ISO 8601 format (YYYY-MM-DD) required by API
            normalized_dob = self._normalize_date_to_iso(date_of_birth)
            logger.debug(f"  - normalized_dob: {normalized_dob}")
            if normalized_dob:
                secondary_fields.append({
                    "typeId": "SFCT_2",
                    "dateTimeValue": normalized_dob
                })

        if country:
            # SFCT_3 is Country in World Check One
            secondary_fields.append({
                "typeId": "SFCT_3",
                "value": country
            })

        if secondary_fields:
            request["secondaryFields"] = secondary_fields

        logger.info(f"  - secondaryFields: {secondary_fields}")
        return request

    async def _make_api_request(
        self,
        endpoint: str,
        request_body: Dict[str, Any],
        query_params: Optional[str] = None
    ) -> Dict[str, Any]:
        """Make an authenticated request to World Check One API using HMAC signature"""

        # URL includes query params, but endpoint for signing does not
        url = f"{self.base_url}{endpoint}"
        if query_params:
            url = f"{url}?{query_params}"
        date_header = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")

        # Generate HMAC signature
        signature = self._sign_request(
            method="POST",
            endpoint=endpoint,
            date=date_header,
            body=request_body
        )

        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Date": date_header,
            "Authorization": (
                f"Signature keyId=\"{self.settings.api_key}\","
                f"algorithm=\"hmac-sha256\","
                f"headers=\"(request-target) host date content-type content-length\","
                f"signature=\"{signature}\""
            )
        }

        # Use exact body string to ensure content-length matches signature
        body_str = json.dumps(request_body, separators=(',', ':'))

        # Use shared client instead of creating new one each time
        client = await self._get_client()
        try:
            response = await client.post(
                url,
                content=body_str,
                headers=headers
            )
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException as e:
            logger.error(f"World Check API timeout: {e}")
            raise
        except httpx.HTTPStatusError as e:
            logger.error(f"World Check API HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"World Check API request failed: {e}")
            raise

    def _sign_request(
        self,
        method: str,
        endpoint: str,
        date: str,
        body: Dict[str, Any]
    ) -> str:
        """
        Generate HMAC-SHA256 signature for World Check One API request.

        The signature is computed over a string containing:
        - (request-target): method and path
        - host: API host
        - date: request date
        - content-type: application/json
        - content-length: body length
        """
        body_str = json.dumps(body, separators=(',', ':'))
        content_length = len(body_str.encode('utf-8'))

        # Extract host from URL
        parsed_url = urlparse(self.base_url)
        host = parsed_url.netloc

        # Build signing string (hyphens, charset, body included per LSEG community post #103139)
        signing_string = (
            f"(request-target): {method.lower()} {endpoint}\n"
            f"host: {host}\n"
            f"date: {date}\n"
            f"content-type: application/json; charset=utf-8\n"
            f"content-length: {content_length}\n"
            f"{body_str}"
        )

        # Generate HMAC-SHA256 signature
        # API secret is used DIRECTLY as HMAC key (no base64 decoding per LSEG docs)
        secret_bytes = self.settings.api_secret.encode('utf-8')

        signature = hmac.new(
            secret_bytes,
            signing_string.encode('utf-8'),
            hashlib.sha256
        ).digest()

        # Return base64 encoded signature
        return base64.b64encode(signature).decode('utf-8')

    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """
        Calculate fuzzy similarity between two names.

        Normalizes names (lowercase, removes punctuation, collapses spaces)
        and returns a similarity ratio between 0.0 and 1.0.
        """
        def normalize(name: str) -> str:
            name = name.lower()
            # Keep only alphanumeric characters and spaces
            name = ''.join(c for c in name if c.isalnum() or c.isspace())
            # Collapse multiple spaces into one
            return ' '.join(name.split())

        n1 = normalize(name1)
        n2 = normalize(name2)
        return jaro_winkler_similarity(n1, n2)

    def _check_secondary_fields_match(
        self,
        result: Dict[str, Any],
        submitted_dob: Optional[str],
        submitted_country: Optional[str]
    ) -> bool:
        """
        Check if secondary fields (DOB, country) match or are unavailable.

        Returns False only if we submitted a field AND the watchlist has it AND they don't match.
        This means DOB/country mismatch indicates a different person (allow through).
        """
        # Extract secondary field results from API response
        secondary_results = result.get("secondaryFieldResults", [])

        for field in secondary_results:
            field_type = field.get("typeId")
            match_status = field.get("fieldResult")  # "MATCHED", "NOT_MATCHED", "NOT_FOUND"

            # SFCT_1 = Date of Birth
            if field_type == "SFCT_1" and submitted_dob:
                if match_status == "NOT_MATCHED":
                    logger.debug(f"DOB mismatch detected - watchlist record is different person")
                    return False  # DOB mismatch - not the same person

            # SFCT_5 = Country
            if field_type == "SFCT_5" and submitted_country:
                if match_status == "NOT_MATCHED":
                    logger.debug(f"Country mismatch detected - watchlist record is different person")
                    return False  # Country mismatch - not the same person

        return True  # No mismatches found

    def _parse_screening_response(
        self,
        response: Dict[str, Any],
        screening_id: str,
        submitted_name: str,
        submitted_dob: Optional[str] = None,
        submitted_country: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Parse World Check One API response and determine if there's a true match.

        A true match requires:
        1. Match strength >= threshold (0.8)
        2. Name similarity >= threshold (0.85)
        3. No DOB/country mismatch (if provided)
        """
        results = response.get("results", [])
        match_details = []
        is_match = False

        for result in results:
            # Ensure match_strength is a float (API may return string)
            match_strength_raw = result.get("matchStrength", 0)
            try:
                match_strength = float(match_strength_raw)
            except (ValueError, TypeError):
                match_strength = 0.0
            matched_name = result.get("matchedName", "")

            # Step 1: Check if match strength meets threshold
            if match_strength < self.settings.match_threshold:
                continue  # Skip results below threshold

            # Step 2: Calculate name similarity between submitted and matched name
            name_similarity = self._calculate_name_similarity(submitted_name, matched_name)

            # Step 3: Check secondary field matches (DOB, country)
            secondary_match = self._check_secondary_fields_match(
                result, submitted_dob, submitted_country
            )

            # Step 4: Determine if this is a true match
            # All conditions must be met: match_strength, name_similarity, and secondary_match
            is_true_match = (
                match_strength >= self.settings.match_threshold and
                name_similarity >= self.settings.name_similarity_threshold and
                secondary_match
            )

            # Log detailed match analysis
            logger.info(
                f"World Check match analysis: "
                f"submitted='{submitted_name}', matched='{matched_name}', "
                f"strength={match_strength:.2f}, name_sim={name_similarity:.2f}, "
                f"secondary_match={secondary_match}, is_true_match={is_true_match}"
            )

            if is_true_match:
                is_match = True

            match_details.append({
                "match_strength": match_strength,
                "matched_name": matched_name,
                "name_similarity": name_similarity,
                "secondary_match": secondary_match,
                "is_true_match": is_true_match,
                "category": result.get("category"),
                "subcategory": result.get("subcategory"),
                "sources": result.get("sources", [])
            })

        return {
            "is_match": is_match,
            "match_details": match_details,
            "screening_id": screening_id,
            "raw_response": response,
            "error": None
        }


# Global service instance
worldcheck_service = WorldCheckService()

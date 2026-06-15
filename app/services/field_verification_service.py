"""
Field Verification Service for verifying expected field values in documents.

This service handles the 'field_verification' document type by:
1. Extracting data from the document using auto-detection (GenericDocumentService)
2. Comparing extracted values against expected values
3. Returning verification results with per-field details

The service is document type agnostic - it uses GenericDocumentService
for detection and extraction, then performs field-level matching.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime

from app.core import get_logger
from app.dto import DocumentErrorCode
from app.dto.verification_session import SequentialJobResponse
from app.services.generic_document_service import GenericDocumentService


logger = get_logger()


class FieldVerificationService:
    """
    Service for verifying expected field values in documents.

    This service is triggered when document_type="field_verification" and:
    1. Extracts expected_values from the file metadata
    2. Uses GenericDocumentService to detect and extract data
    3. Compares each expected value against extracted data
    4. Returns verification results with per-field details

    Matching Rules:
    - String values: case-insensitive, whitespace-normalized comparison
    - Numeric values: exact match after stripping whitespace
    - List/array values: ALL items in expected must exist in extracted (order-independent)
    - Missing extracted field: found=False, matches=False
    - Value mismatch: found=True, matches=False
    """

    def __init__(
        self,
        detailed_analysis_service=None,
        state_service=None,
        user_identity_repo=None,
    ):
        """
        Initialize the field verification service.

        Args:
            detailed_analysis_service: Optional detailed analysis service
            state_service: Optional state service
            user_identity_repo: Repository for fetching user identity (for name matching)
        """
        self.logger = get_logger()
        self.detailed_analysis_service = detailed_analysis_service
        self.state_service = state_service
        self.user_identity_repo = user_identity_repo

    async def verify_fields(
        self,
        request_data: Dict[str, Any],
        job_id: str,
        client_public_key: str
    ) -> SequentialJobResponse:
        """
        Verify expected field values in a document.

        Args:
            request_data: Request data containing files array
            job_id: Job identifier
            client_public_key: Client's public key

        Returns:
            SequentialJobResponse with verification results
        """
        start_time = datetime.now()
        self.logger.info(f"Starting field verification for job {job_id}")

        try:
            # Extract files from request
            files_data = request_data.get("files", [])
            if not files_data or len(files_data) != 1:
                return self._create_error_response(
                    error_code=DocumentErrorCode.TECHNICAL_INVALID_PAYLOAD,
                    error_message="Field verification requires exactly one file",
                    job_id=job_id,
                )

            file_data = files_data[0]

            # Extract expected_values from file metadata
            expected_values = file_data.get("expected_values")

            # Validate expected_values is provided and non-empty
            if not expected_values:
                return self._create_error_response(
                    error_code=DocumentErrorCode.TECHNICAL_MISSING_FIELD,
                    error_message="expected_values is required for field_verification document type",
                    job_id=job_id,
                )

            if not isinstance(expected_values, dict) or len(expected_values) == 0:
                return self._create_error_response(
                    error_code=DocumentErrorCode.TECHNICAL_MISSING_FIELD,
                    error_message="expected_values must be a non-empty dictionary",
                    job_id=job_id,
                )

            self.logger.info(f"Expected values to verify: {list(expected_values.keys())}")

            # Get user identity ID for processing
            user_identity_id = self._get_user_identity_id(client_public_key)

            # Use GenericDocumentService for auto-detection and extraction
            generic_service = GenericDocumentService(
                user_identity_repo=self.user_identity_repo
            )

            # Extract data from document (type-agnostic)
            # Note: Field verification requires document_type to be provided by caller
            # For backward compatibility, default to 'id_card' when not specified
            extraction_response = await generic_service.process_auto_document(
                file_data=file_data,
                client_public_key=client_public_key,
                user_identity_id=user_identity_id,
                document_type='id_card',  # Default for field verification
                country_code=None,
                entity=None
            )

            # Check if extraction was successful
            if not extraction_response.result:
                # Extraction failed - return the error response
                self.logger.warning(f"Document extraction failed: {extraction_response.error}")
                extraction_response.job_id = job_id
                return extraction_response

            # Get extracted data (excluding metadata fields like _detection, _name_match)
            extracted_data = extraction_response.extracted_data or {}
            extracted_data_for_comparison = self._extract_comparison_data(extracted_data)

            # Perform field verification
            verification_results = self._verify_fields(
                expected_values=expected_values,
                extracted_data=extracted_data_for_comparison
            )

            # Calculate overall result (all fields must match)
            all_match = all(
                result.get("matches", False) for result in verification_results.values()
            )

            # Build summary
            total_fields = len(verification_results)
            matched_fields = sum(
                1 for result in verification_results.values() if result.get("matches", False)
            )
            failed_fields = total_fields - matched_fields

            summary = {
                "total_fields": total_fields,
                "matched_fields": matched_fields,
                "failed_fields": failed_fields,
            }

            processing_time = (datetime.now() - start_time).total_seconds()

            self.logger.info(
                f"Field verification complete: {matched_fields}/{total_fields} fields matched "
                f"({'ALL MATCH' if all_match else 'SOME FAILED'})"
            )

            # Build response
            response = SequentialJobResponse(
                result=all_match,
                job_id=job_id,
                verification_state=0,  # Not applicable for this job type
                sequence_no=0,
                processing_time_seconds=processing_time,
                extracted_data=extracted_data,
                other_checks={
                    "field_verification": verification_results,
                    "summary": summary,
                },
                message=f"Field verification complete: {matched_fields}/{total_fields} fields matched",
                # Include detection metadata from extraction
                detected_document_type=extraction_response.detected_document_type,
                detected_country=extraction_response.detected_country,
                detected_entity=extraction_response.detected_entity,
                detection_confidence=extraction_response.detection_confidence,
                selected_schema=extraction_response.selected_schema,
            )

            return response

        except Exception as e:
            self.logger.error(f"Error during field verification: {e}")
            return self._create_error_response(
                error_code=DocumentErrorCode.PROCESSING_ERROR,
                error_message=str(e),
                job_id=job_id,
            )

    def _extract_comparison_data(
        self,
        extracted_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract comparison data by removing metadata fields.

        Args:
            extracted_data: Raw extracted data from GenericDocumentService

        Returns:
            Dictionary with only actual field data (no _metadata fields)
        """
        # Filter out metadata fields (starting with _)
        return {
            k: v for k, v in extracted_data.items()
            if not k.startswith("_")
        }

    def _verify_fields(
        self,
        expected_values: Dict[str, Any],
        extracted_data: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Verify each expected field against extracted data.

        Args:
            expected_values: Dictionary of expected field values
            extracted_data: Dictionary of extracted field values

        Returns:
            Dictionary with verification results for each field
        """
        verification_results = {}

        for field_name, expected_value in expected_values.items():
            result = self._verify_single_field(
                field_name=field_name,
                expected_value=expected_value,
                extracted_data=extracted_data
            )
            verification_results[field_name] = result

        return verification_results

    def _verify_single_field(
        self,
        field_name: str,
        expected_value: Any,
        extracted_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Verify a single field against extracted data.

        Args:
            field_name: Name of the field to verify
            expected_value: Expected value for the field
            extracted_data: Dictionary of extracted field values

        Returns:
            Dictionary with verification result:
            - expected: The expected value
            - found: Whether the field was found in extracted data
            - matches: Whether the value matches
            - extracted: The extracted value (if found)
        """
        # Check if field exists in extracted data
        if field_name not in extracted_data:
            return {
                "expected": expected_value,
                "found": False,
                "matches": False,
                "extracted": None,
            }

        extracted_value = extracted_data[field_name]

        # Perform comparison based on value types
        matches = self._compare_values(expected_value, extracted_value)

        return {
            "expected": expected_value,
            "found": True,
            "matches": matches,
            "extracted": extracted_value,
        }

    def _compare_values(self, expected: Any, extracted: Any) -> bool:
        """
        Compare expected and extracted values.

        Matching Rules:
        - String values: case-insensitive, whitespace-normalized comparison
        - Numeric values: exact match after stripping whitespace
        - List/array values: ALL items in expected must exist in extracted (order-independent)
        - None/empty: direct comparison

        Args:
            expected: Expected value
            extracted: Extracted value

        Returns:
            True if values match according to matching rules
        """
        # Handle None values
        if expected is None:
            return extracted is None
        if extracted is None:
            return False

        # Both strings - case-insensitive, whitespace-normalized
        if isinstance(expected, str) and isinstance(extracted, str):
            return self._normalize_string(expected) == self._normalize_string(extracted)

        # Both lists/arrays - all expected items must exist in extracted
        if isinstance(expected, list) and isinstance(extracted, list):
            # Normalize string items for comparison
            expected_normalized = [
                self._normalize_string(item) if isinstance(item, str) else item
                for item in expected
            ]
            extracted_normalized = [
                self._normalize_string(item) if isinstance(item, str) else item
                for item in extracted
            ]
            # All expected items must exist in extracted
            return all(item in extracted_normalized for item in expected_normalized)

        # Numeric comparison - convert to string and normalize
        try:
            expected_str = self._normalize_string(str(expected))
            extracted_str = self._normalize_string(str(extracted))
            return expected_str == extracted_str
        except Exception:
            pass

        # Direct comparison as fallback
        return expected == extracted

    def _normalize_string(self, value: str) -> str:
        """
        Normalize a string for comparison.

        Normalization includes:
        - Case folding (case-insensitive)
        - Whitespace normalization (collapse multiple spaces)

        Args:
            value: String to normalize

        Returns:
            Normalized string
        """
        if not isinstance(value, str):
            value = str(value)
        # Case fold and normalize whitespace
        return " ".join(value.casefold().split())

    def _get_user_identity_id(self, client_public_key: str) -> str:
        """
        Get user identity ID from client public key.

        Args:
            client_public_key: Client's public key

        Returns:
            User identity ID or placeholder if not found
        """
        try:
            if self.user_identity_repo:
                from app.repositories.user_key_repository import UserKeyRepository
                user_key_repo = UserKeyRepository()
                user_key = user_key_repo.get_key_by_public_key(client_public_key)
                if user_key and user_key.get('user_identity_id'):
                    return user_key['user_identity_id']
        except Exception as e:
            self.logger.debug(f"Could not get user_identity_id: {e}")

        # Return placeholder if not found
        return f"user_{client_public_key[:16]}"

    def _create_error_response(
        self,
        error_code: str,
        error_message: str,
        job_id: str,
    ) -> SequentialJobResponse:
        """Create an error response."""
        return SequentialJobResponse(
            result=False,
            job_id=job_id,
            verification_state=0,
            message=f"Error processing field verification: {error_message}",
            error_code=error_code,
            error=error_message,
            extracted_data={},
            other_checks=None,
        )


__all__ = [
    "FieldVerificationService",
]

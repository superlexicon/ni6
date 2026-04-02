"""
Schema Selector for hierarchical schema selection with fallback.

This module provides intelligent schema selection based on:
1. Detection results (document type, country, entity)
2. Schema specificity preferences
3. Fallback chain: type:country:entity → type:country → type

The selector ensures the most specific applicable schema is always used.
"""

from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass

from app.core import get_logger
from app.schemas.generic import (
    SchemaRegistry,
    DocumentTypeSchema,
    DocumentDetectionResult,
)


logger = get_logger()


@dataclass
class SchemaSelectionResult:
    """
    Result of schema selection process.

    Attributes:
        selected_schema: The schema that was selected
        selection_method: How the schema was selected (exact_match, fallback, etc.)
        fallback_chain: List of schemas that were tried in order
        confidence: Selection confidence (0-1)
    """
    selected_schema: Optional[DocumentTypeSchema]
    selection_method: str
    fallback_chain: List[DocumentTypeSchema]
    confidence: float

    def __post_init__(self):
        if self.fallback_chain is None:
            self.fallback_chain = []


class SchemaSelector:
    """
    Hierarchical schema selector with intelligent fallback.

    Selection priority:
    1. Exact match (type:country:entity)
    2. Country-level match (type:country)
    3. Entity-level match (type:entity)
    4. Generic match (type)

    The selector also considers:
    - Schema enabled status
    - Schema priority
    - Detection confidence
    """

    def __init__(self, registry: Optional[SchemaRegistry] = None):
        """
        Initialize the schema selector.

        Args:
            registry: Optional schema registry (uses global singleton if None)
        """
        self.logger = get_logger()
        self.registry = registry or SchemaRegistry()

    def select(
        self,
        detection_result: Optional[DocumentDetectionResult] = None,
        document_type: Optional[str] = None,
        country_code: Optional[str] = None,
        entity: Optional[str] = None,
        confidence_threshold: float = 0.4,
        prefer_specific: bool = True,
    ) -> SchemaSelectionResult:
        """
        Select the best schema based on detection results or explicit parameters.

        Args:
            detection_result: Document detection result (optional)
            document_type: Document type (required if no detection_result)
            country_code: Country code (optional)
            entity: Entity identifier (optional)
            confidence_threshold: Minimum confidence to use detected values
            prefer_specific: Whether to prefer more specific schemas

        Returns:
            SchemaSelectionResult with selected schema and selection metadata
        """
        # Extract parameters from detection result if provided
        if detection_result:
            doc_type = detection_result.document_type or document_type
            # Only use detected country/entity if confidence is above threshold
            country = (
                detection_result.country_code
                if detection_result.country_confidence >= confidence_threshold
                else country_code
            )
            ent = (
                detection_result.entity
                if detection_result.entity_confidence >= confidence_threshold
                else entity
            )
        else:
            doc_type = document_type
            country = country_code
            ent = entity

        if not doc_type:
            self.logger.warning("No document type provided for schema selection")
            return SchemaSelectionResult(
                selected_schema=None,
                selection_method="no_type",
                fallback_chain=[],
                confidence=0.0,
            )

        self.logger.debug(
            f"Selecting schema for type={doc_type}, country={country}, entity={ent}"
        )

        # Build fallback chain
        fallback_chain = self._build_fallback_chain(doc_type, country, ent)

        # Find the first enabled schema in the chain
        selected_schema = None
        selection_method = "none"

        for schema, method in fallback_chain:
            if schema and schema.enabled:
                selected_schema = schema
                selection_method = method
                break

        # Calculate selection confidence
        if selected_schema:
            confidence = self._calculate_selection_confidence(
                selected_schema, doc_type, country, ent, detection_result
            )
            self.logger.info(
                f"Selected schema: {selected_schema.schema_id} "
                f"(method={selection_method}, confidence={confidence:.2f})"
            )
        else:
            confidence = 0.0
            self.logger.warning(f"No schema found for type={doc_type}, country={country}, entity={ent}")

        # Flatten fallback chain for result
        fallback_schemas = [s for s, _ in fallback_chain]

        return SchemaSelectionResult(
            selected_schema=selected_schema,
            selection_method=selection_method,
            fallback_chain=fallback_schemas,
            confidence=confidence,
        )

    def _build_fallback_chain(
        self,
        document_type: str,
        country_code: Optional[str],
        entity: Optional[str],
    ) -> List[Tuple[Optional[DocumentTypeSchema], str]]:
        """
        Build the fallback chain for schema selection.

        Returns a list of (schema, method) tuples in order of preference.
        """
        chain = []

        # 1. Try exact match (type:country:entity)
        if country_code and entity:
            schema_id = f"{document_type}:{country_code}:{entity}"
            schema = self.registry.get_schema_by_id(schema_id)
            if schema:
                chain.append((schema, "exact_match"))

        # 2. Try country-level (type:country)
        if country_code:
            schema_id = f"{document_type}:{country_code}"
            schema = self.registry.get_schema_by_id(schema_id)
            if schema:
                chain.append((schema, "country_match"))

        # 3. Try entity-level (type:entity) - if entity but no country match
        if entity:
            schema_id = f"{document_type}:__entity__:{entity}"
            schema = self.registry.get_schema_by_id(schema_id)
            if schema:
                chain.append((schema, "entity_match"))

        # 4. Try generic (type only)
        schema = self.registry.get_schema_by_id(document_type)
        if schema:
            chain.append((schema, "generic_match"))

        return chain

    def _calculate_selection_confidence(
        self,
        schema: DocumentTypeSchema,
        document_type: str,
        country_code: Optional[str],
        entity: Optional[str],
        detection_result: Optional[DocumentDetectionResult],
    ) -> float:
        """
        Calculate confidence score for schema selection.

        Higher confidence = better match between schema and detected document.
        """
        confidence = 0.5  # Base confidence

        # Boost based on schema specificity
        if schema.entity and entity:
            if schema.entity.lower() == entity.lower():
                confidence += 0.3
        elif schema.entity and entity and schema.entity.lower() != entity.lower():
            confidence -= 0.1

        if schema.country_code and country_code:
            if schema.country_code.upper() == country_code.upper():
                confidence += 0.2

        # Use detection confidence if available
        if detection_result:
            detection_conf = detection_result.confidence
            confidence = confidence * 0.6 + detection_conf * 0.4

        # Apply priority bonus
        priority_bonus = min(schema.priority / 200, 0.1)
        confidence += priority_bonus

        return min(max(confidence, 0.0), 1.0)

    def find_best_schema_by_text(
        self,
        text: str,
        document_type_hint: Optional[str] = None,
        top_k: int = 3,
    ) -> List[SchemaSelectionResult]:
        """
        Find the best schemas based on text matching.

        Args:
            text: Document text to match against
            document_type_hint: Optional hint to narrow down document type
            top_k: Number of top results to return

        Returns:
            List of SchemaSelectionResult sorted by confidence
        """
        matches = self.registry.find_schemas_by_text(
            text=text,
            document_type_hint=document_type_hint,
            top_k=top_k * 2,  # Get more to filter
        )

        results = []
        for schema, match_score in matches:
            result = SchemaSelectionResult(
                selected_schema=schema,
                selection_method="text_match",
                fallback_chain=[schema],
                confidence=match_score,
            )
            results.append(result)

        # Sort by confidence and return top_k
        results.sort(key=lambda r: r.confidence, reverse=True)
        return results[:top_k]

    def get_all_schemas_for_type(
        self,
        document_type: str,
        country_code: Optional[str] = None,
    ) -> List[DocumentTypeSchema]:
        """
        Get all schemas for a given document type, sorted by specificity.

        Args:
            document_type: The document type
            country_code: Optional country filter

        Returns:
            List of schemas sorted by specificity (most specific first)
        """
        schemas = self.registry.get_all_schemas(
            document_type=document_type,
            country_code=country_code,
        )

        # Sort by specificity and priority
        schemas.sort(key=lambda s: (s.specificity, s.priority), reverse=True)
        return schemas

    def suggest_schema_for_text(
        self,
        text: str,
        detection_result: Optional[DocumentDetectionResult] = None,
    ) -> SchemaSelectionResult:
        """
        Suggest the best schema based on text content and optional detection result.

        This is a convenience method that combines text matching with
        the standard selection process.

        Args:
            text: Document text
            detection_result: Optional detection result to guide selection

        Returns:
            SchemaSelectionResult with the best suggested schema
        """
        # First, try selection using detection result
        if detection_result and detection_result.document_type:
            standard_result = self.select(detection_result=detection_result)

            # If we got a good match, return it
            if standard_result.selected_schema and standard_result.confidence > 0.7:
                return standard_result

        # Fallback to text-based matching
        text_matches = self.find_best_schema_by_text(
            text=text,
            document_type_hint=detection_result.document_type if detection_result else None,
            top_k=1,
        )

        if text_matches and text_matches[0].selected_schema:
            text_result = text_matches[0]
            # Update method to reflect combined approach
            text_result.selection_method = "text_fallback"
            return text_result

        # No good match found
        return SchemaSelectionResult(
            selected_schema=None,
            selection_method="no_match",
            fallback_chain=[],
            confidence=0.0,
        )


# Convenience functions

def select_schema(
    document_type: str,
    country_code: Optional[str] = None,
    entity: Optional[str] = None,
    detection_result: Optional[DocumentDetectionResult] = None,
) -> SchemaSelectionResult:
    """
    Quick schema selection function.

    Args:
        document_type: Document type
        country_code: Optional country code
        entity: Optional entity identifier
        detection_result: Optional detection result

    Returns:
        SchemaSelectionResult with selected schema
    """
    selector = SchemaSelector()
    return selector.select(
        detection_result=detection_result,
        document_type=document_type,
        country_code=country_code,
        entity=entity,
    )


def get_best_schema_for_text(
    text: str,
    document_type_hint: Optional[str] = None,
) -> Optional[DocumentTypeSchema]:
    """
    Get the best matching schema for a given text.

    Args:
        text: Document text
        document_type_hint: Optional hint for document type

    Returns:
        Best matching schema or None
    """
    selector = SchemaSelector()
    result = selector.find_best_schema_by_text(
        text=text,
        document_type_hint=document_type_hint,
        top_k=1,
    )
    return result[0].selected_schema if result else None


__all__ = [
    "SchemaSelector",
    "SchemaSelectionResult",
    "select_schema",
    "get_best_schema_for_text",
]

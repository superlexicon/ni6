"""
Generic Document Schema Library

This module provides a central registry for document schemas with hierarchical lookup.
Schemas are organized by:
1. Document Type (tax_return, id_card, driving_license, utility_bill, etc.)
2. Country Code (SG, IN, US, MY, TH, etc.)
3. Entity (DBS, SBI, IRAS, Chase, etc.)

Schema lookup follows a specific → country-level → generic fallback pattern.

Usage:
    from app.schemas.generic import SchemaRegistry, DocumentTypeSchema

    # Register a schema
    schema = DocumentTypeSchema(
        schema_id="tax_return:SG:iras",
        document_type="tax_return",
        ...
    )
    SchemaRegistry.register(schema)

    # Get the most specific schema for a document
    schema = SchemaRegistry.get_schema(
        document_type="tax_return",
        country_code="SG",
        entity="iras"
    )
"""

from typing import Dict, List, Optional, Set, Tuple, Iterator
from collections import defaultdict
import threading

from app.core import get_logger

from .base import (
    DocumentTypeSchema,
    DocumentDetectionResult,
    GLINER2Schema,
    ExtractionResult,
)


logger = get_logger()


class SchemaRegistry:
    """
    Central registry for document schemas with hierarchical lookup.

    This is a singleton class that manages all document type schemas.
    Schemas are indexed by document_type, country_code, and entity for efficient lookup.

    Thread-safe implementation using locks for concurrent access.
    """

    _instance: Optional['SchemaRegistry'] = None
    _lock = threading.Lock()

    # Indexed storage for efficient lookup
    _schemas: Dict[str, DocumentTypeSchema] = {}  # schema_id -> schema

    # Indexes for fast lookup
    _by_type: Dict[str, Set[str]] = defaultdict(set)  # document_type -> set of schema_ids
    _by_country: Dict[str, Set[str]] = defaultdict(set)  # country_code -> set of schema_ids
    _by_entity: Dict[str, Set[str]] = defaultdict(set)  # entity -> set of schema_ids
    _by_type_country: Dict[Tuple[str, str], Set[str]] = defaultdict(set)  # (type, country) -> schema_ids
    _by_type_entity: Dict[Tuple[str, str], Set[str]] = defaultdict(set)  # (type, entity) -> schema_ids

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(SchemaRegistry, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return
            self._initialized = True
            logger.info("SchemaRegistry initialized")

    @classmethod
    def register(cls, schema: DocumentTypeSchema) -> None:
        """
        Register a document schema in the registry.

        Args:
            schema: The DocumentTypeSchema to register

        Raises:
            ValueError: If a schema with the same schema_id already exists
        """
        with cls._lock:
            # Check for duplicate schema_id
            if schema.schema_id in cls._schemas:
                logger.warning(
                    f"Schema {schema.schema_id} already registered, overwriting"
                )
                # Unregister existing schema first
                cls._unregister_schema_by_id(schema.schema_id)

            # Store the schema
            cls._schemas[schema.schema_id] = schema

            # Update indexes
            cls._by_type[schema.document_type].add(schema.schema_id)

            if schema.country_code:
                cls._by_country[schema.country_code].add(schema.schema_id)
                cls._by_type_country[(schema.document_type, schema.country_code)].add(schema.schema_id)

            if schema.entity:
                cls._by_entity[schema.entity].add(schema.schema_id)
                cls._by_type_entity[(schema.document_type, schema.entity)].add(schema.schema_id)

            logger.debug(
                f"Registered schema: {schema.schema_id} "
                f"(type={schema.document_type}, country={schema.country_code}, entity={schema.entity})"
            )

    @classmethod
    def unregister(cls, schema_id: str) -> bool:
        """
        Unregister a schema by its ID.

        Args:
            schema_id: The schema ID to unregister

        Returns:
            True if the schema was found and removed, False otherwise
        """
        with cls._lock:
            return cls._unregister_schema_by_id(schema_id)

    @classmethod
    def _unregister_schema_by_id(cls, schema_id: str) -> bool:
        """Internal method to unregister a schema (must be called within lock)."""
        if schema_id not in cls._schemas:
            return False

        schema = cls._schemas[schema_id]

        # Remove from indexes
        cls._by_type[schema.document_type].discard(schema.schema_id)

        if schema.country_code:
            cls._by_country[schema.country_code].discard(schema.schema_id)
            cls._by_type_country[(schema.document_type, schema.country_code)].discard(schema.schema_id)

        if schema.entity:
            cls._by_entity[schema.entity].discard(schema.schema_id)
            cls._by_type_entity[(schema.document_type, schema.entity)].discard(schema.schema_id)

        # Remove the schema
        del cls._schemas[schema_id]

        logger.debug(f"Unregistered schema: {schema_id}")
        return True

    @classmethod
    def get_schema(
        cls,
        document_type: str,
        country_code: Optional[str] = None,
        entity: Optional[str] = None
    ) -> Optional[DocumentTypeSchema]:
        """
        Get the most specific schema matching the given criteria.

        Lookup follows a specific → country-level → generic fallback pattern:
        1. Try type:country:entity (most specific)
        2. Try type:country
        3. Try type:entity
        4. Try type (generic)

        Args:
            document_type: The document type (required)
            country_code: ISO country code (optional)
            entity: Entity identifier (optional)

        Returns:
            The most specific matching schema, or None if no match found
        """
        with cls._lock:
            # Try exact match first (type:country:entity)
            if country_code and entity:
                schema_id = f"{document_type}:{country_code}:{entity}"
                if schema := cls._schemas.get(schema_id):
                    return schema

            # Try type:country
            if country_code:
                schema_id = f"{document_type}:{country_code}"
                if schema := cls._schemas.get(schema_id):
                    return schema

            # Try type:entity
            if entity:
                schema_id = f"{document_type}:__entity__:{entity}"
                if schema := cls._schemas.get(schema_id):
                    return schema

            # Try generic (type only)
            schema_id = document_type
            if schema := cls._schemas.get(schema_id):
                return schema

            return None

    @classmethod
    def get_all_schemas(
        cls,
        document_type: Optional[str] = None,
        country_code: Optional[str] = None,
        entity: Optional[str] = None,
        enabled_only: bool = True
    ) -> List[DocumentTypeSchema]:
        """
        Get all schemas matching the given criteria.

        Args:
            document_type: Filter by document type (None = all)
            country_code: Filter by country code (None = all)
            entity: Filter by entity (None = all)
            enabled_only: Only return enabled schemas

        Returns:
            List of matching schemas, sorted by specificity (most specific first)
        """
        with cls._lock:
            results: List[DocumentTypeSchema] = []

            for schema in cls._schemas.values():
                if enabled_only and not schema.enabled:
                    continue
                if document_type and schema.document_type != document_type:
                    continue
                if country_code and schema.country_code != country_code:
                    continue
                if entity and schema.entity != entity:
                    continue
                results.append(schema)

            # Sort by specificity (entity > country > generic)
            results.sort(key=lambda s: (s.specificity, s.priority), reverse=True)
            return results

    @classmethod
    def get_document_types(cls) -> List[str]:
        """Get all registered document types."""
        with cls._lock:
            return sorted(set(cls._by_type.keys()))

    @classmethod
    def get_countries_for_type(cls, document_type: str) -> List[str]:
        """Get all country codes that have schemas for a given document type."""
        with cls._lock:
            country_codes = set()
            for schema_id in cls._by_type.get(document_type, set()):
                if schema := cls._schemas.get(schema_id):
                    if schema.country_code:
                        country_codes.add(schema.country_code)
            return sorted(country_codes)

    @classmethod
    def get_entities_for_type_country(
        cls,
        document_type: str,
        country_code: Optional[str] = None
    ) -> List[str]:
        """Get all entities that have schemas for a given type/country."""
        with cls._lock:
            entities = set()
            for schema_id in cls._by_type.get(document_type, set()):
                if schema := cls._schemas.get(schema_id):
                    if country_code and schema.country_code != country_code:
                        continue
                    if schema.entity:
                        entities.add(schema.entity)
            return sorted(entities)

    @classmethod
    def find_best_match(
        cls,
        document_type: Optional[str] = None,
        country_code: Optional[str] = None,
        entity: Optional[str] = None
    ) -> Optional[DocumentTypeSchema]:
        """
        Find the best matching schema when exact match is not found.

        This method will:
        1. Try exact match first
        2. Try partial matches (e.g., without entity)
        3. Return the best available fallback

        Args:
            document_type: The document type
            country_code: ISO country code
            entity: Entity identifier

        Returns:
            Best matching schema or None
        """
        with cls._lock:
            # If no document type provided, return None
            if not document_type:
                return None

            # Try exact match first
            if schema := cls.get_schema(document_type, country_code, entity):
                return schema

            # Try with country only
            if country_code:
                if schema := cls.get_schema(document_type, country_code, None):
                    return schema

            # Try with document type only
            if schema := cls.get_schema(document_type, None, None):
                return schema

            return None

    @classmethod
    def find_schemas_by_text(
        cls,
        text: str,
        document_type_hint: Optional[str] = None,
        top_k: int = 5
    ) -> List[Tuple[DocumentTypeSchema, float]]:
        """
        Find schemas that match the given text.

        Args:
            text: The document text to match against
            document_type_hint: Optional hint to narrow down document type
            top_k: Maximum number of results to return

        Returns:
            List of (schema, match_score) tuples sorted by score
        """
        with cls._lock:
            results: List[Tuple[DocumentTypeSchema, float]] = []

            for schema in cls._schemas.values():
                if not schema.enabled:
                    continue

                if document_type_hint and schema.document_type != document_type_hint:
                    continue

                match_score = schema.matches_text(text)
                if match_score > 0:
                    results.append((schema, match_score))

            # Sort by match score and specificity
            results.sort(key=lambda x: (x[1], x[0].specificity), reverse=True)
            return results[:top_k]

    @classmethod
    def clear(cls) -> None:
        """Clear all registered schemas. Useful for testing."""
        with cls._lock:
            cls._schemas.clear()
            cls._by_type.clear()
            cls._by_country.clear()
            cls._by_entity.clear()
            cls._by_type_country.clear()
            cls._by_type_entity.clear()
            logger.info("SchemaRegistry cleared")

    @classmethod
    def count(cls) -> int:
        """Get the total number of registered schemas."""
        with cls._lock:
            return len(cls._schemas)

    @classmethod
    def iter_schemas(cls) -> Iterator[DocumentTypeSchema]:
        """Iterate over all registered schemas."""
        with cls._lock:
            return iter(cls._schemas.values())

    @classmethod
    def get_schema_by_id(cls, schema_id: str) -> Optional[DocumentTypeSchema]:
        """Get a schema by its ID."""
        with cls._lock:
            return cls._schemas.get(schema_id)

    @classmethod
    def get_stats(cls) -> Dict[str, any]:
        """Get registry statistics."""
        with cls._lock:
            return {
                "total_schemas": len(cls._schemas),
                "document_types": len(cls._by_type),
                "countries": len(cls._by_country),
                "entities": len(cls._by_entity),
                "by_document_type": {
                    doc_type: len(schema_ids)
                    for doc_type, schema_ids in cls._by_type.items()
                }
            }


# Convenience functions for common operations

def register_schema(schema: DocumentTypeSchema) -> None:
    """Register a schema in the global registry."""
    SchemaRegistry.register(schema)


def get_schema(
    document_type: str,
    country_code: Optional[str] = None,
    entity: Optional[str] = None
) -> Optional[DocumentTypeSchema]:
    """Get the best matching schema from the global registry."""
    return SchemaRegistry.get_schema(document_type, country_code, entity)


def list_document_types() -> List[str]:
    """List all registered document types."""
    return SchemaRegistry.get_document_types()


def list_schemas(
    document_type: Optional[str] = None,
    country_code: Optional[str] = None,
    entity: Optional[str] = None
) -> List[DocumentTypeSchema]:
    """List all schemas matching the given criteria."""
    return SchemaRegistry.get_all_schemas(document_type, country_code, entity)


__all__ = [
    "SchemaRegistry",
    "DocumentTypeSchema",
    "DocumentDetectionResult",
    "GLINER2Schema",
    "ExtractionResult",
    "register_schema",
    "get_schema",
    "list_document_types",
    "list_schemas",
]

# Import schema definitions to populate the registry
# Import in order of specificity: generic -> country -> entity

from . import document_schemas
from . import country_schemas
from . import entity_schemas

# Log registration stats
logger.info(f"SchemaRegistry initialized with {SchemaRegistry.count()} schemas")
logger.debug(f"Registered document types: {SchemaRegistry.get_document_types()}")

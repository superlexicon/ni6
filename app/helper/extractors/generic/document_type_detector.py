"""
Document Type Detector using GLiNER2.

This module provides three-tier document type detection:
1. Document Type (tax_return, id_card, driving_license, utility_bill, etc.)
2. Country (SG, IN, US, MY, TH, etc.)
3. Entity (DBS, SBI, Chase, IRAS, etc.)

The detector uses a hybrid approach:
- GLiNER2 for entity detection (zero-shot NER)
- Pattern matching for known keywords and labels
- Fallback to existing country detection patterns
"""

import re
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field

from app.core import get_logger
from app.core.gliner_ner_model import GLiNERNERModel
from app.helper.extractors.pattern_definitions import (
    detect_country_from_text,
    ISO3_TO_ISO2,
    COUNTRY_PATTERNS,
)
from app.schemas.generic import (
    SchemaRegistry,
    DocumentTypeSchema,
    DocumentDetectionResult,
)


logger = get_logger()


# Document types that are handled by existing services (NOT in generic detector)
EXISTING_DOCUMENT_TYPES = {
    "passport",
    "bank_statement",
    "selfie",
    "video_selfie",
    "id_card",  # Note: id_card has an existing service but we can add generic support
    "resume",
    "secret_share_recovery",
    "add_public_key",
    "remove_public_key",
}


# Keywords for document type detection (Tier 1)
DOCUMENT_TYPE_KEYWORDS = {
    "tax_return": [
        "tax return", "income tax", "tax assessment", "notice of assessment",
        "tax statement", "tax computation", "taxable income", "tax payable",
        "assessment year", "tax year", "federal tax", "state tax",
        "inland revenue", "revenue authority", "income tax department",
        "irs", "internal revenue service", "itra", "lhdn",
    ],
    "tax_residency_certificate": [
        "tax residency certificate", "trc", "tax resident",
        "certificate of residence", "tax residency",
        "residency certificate", "tax domicile",
    ],
    "id_card": [
        "identity card", "national id", "identification card", "id card",
        "national identity", "citizen card", "personal identity",
        "nric", "fin", "pan card", "aadhaar", "mykad",
        "social security card", "ssn",
        # Tax residency and additional ID types
        "tax residency", "trc", "tax certificate", "tax resident",
        "emirates id",
    ],
    "driving_license": [
        "driver's license", "driving licence", "driving license",
        "driver license", "motor vehicle license", "vehicle licence",
        "class", "driving permit", "motor vehicle department",
        "department of motor vehicles", "dmv",
    ],
    "utility_bill": [
        "utility bill", "electricity bill", "water bill", "gas bill",
        "internet bill", "phone bill", "utility statement", "bill statement",
        "amount due", "payment due", "billing period", "service address",
    ],
    "payslip": [
        "payslip", "pay slip", "salary slip", "pay statement",
        "earnings statement", "pay advice", "payroll", "wage statement",
        "net pay", "gross pay", "pay period", "year to date",
    ],
    "insurance_policy": [
        "insurance policy", "policy document", "insurance certificate",
        "policy number", "sum assured", "insurance company",
        "policyholder", "beneficiary", "premium", "coverage",
    ],
    "employment_letter": [
        "employment letter", "offer letter", "employment contract",
        "appointment letter", "job offer", "offer of employment",
        "employment verification", "proof of employment", "work confirmation",
    ],
    "residence_proof": [
        "proof of residence", "address proof", "residence verification",
        "proof of address", "residency proof", "domicile certificate",
    ],
}


# Entity detection keywords by country (Tier 3)
ENTITY_KEYWORDS = {
    # Singapore banks
    "dbs": ["dbs", "development bank of singapore", "posb-dbs"],
    "posb": ["posb", "post office savings bank", "posb bank"],
    "uob": ["uob", "united overseas bank"],
    "ocbc": ["ocbc", "oversea-chinese banking corporation"],
    "citibank": ["citibank", "citi bank", "citi"],
    # India banks
    "sbi": ["sbi", "state bank of india", "state bank"],
    "hdfc": ["hdfc", "hdfc bank"],
    "icici": ["icici", "icici bank"],
    "axis": ["axis", "axis bank"],
    # US banks
    "chase": ["chase", "jpmorgan chase", "jp morgan"],
    "boa": ["bank of america", "boa"],
    "wells_fargo": ["wells fargo", "wells"],
    # Singapore government
    "iras": ["iras", "inland revenue authority of singapore", "inland revenue"],
    "cpf": ["cpf", "central provident fund", "cpf board"],
    "ica": ["ica", "immigration and checkpoints authority"],
    # Singapore utilities
    "sp": ["sp services", "singapore power", "sp group", "s&p"],
    "pub": ["pub", "pub singapore", "national water agency"],
    "singtel": ["singtel", "singapore telecommunications"],
    "starhub": ["starhub", "starhub limited"],
    # Insurance
    "aia": ["aia", "american international assurance"],
    "prudential": ["prudential", "prudential assurance"],
    "great_eastern": ["great eastern", "great eastern life"],
    # UAE government/tax
    "trc": ["ministry of finance", "federal tax authority", "tax residency certificate"],
    "mof": ["ministry of finance", "mof uae"],
    "pan": ["permanent account number", "pan card", "income tax department"],
    "aadhaar": ["uidai", "unique identification authority", "aadhaar"],
}


@dataclass
class EntityMatch:
    """A matched entity with confidence score."""
    entity: str
    confidence: float
    matched_text: str
    method: str  # "keyword", "gliner2", "pattern"


class DocumentTypeDetector:
    """
    GLiNER2-powered three-tier document type detector.

    Detection hierarchy:
    1. Document Type - what kind of document is this?
    2. Country - which country is this document from?
    3. Entity - which specific entity issued this document?

    Uses a hybrid approach combining:
    - GLiNER2 zero-shot NER for flexible entity detection
    - Pattern matching for known keywords and labels
    - Existing country detection patterns
    """

    def __init__(self, gliner_model: Optional[GLiNERNERModel] = None):
        """
        Initialize the detector.

        Args:
            gliner_model: Optional GLiNER model instance (singleton if None)
        """
        self.logger = get_logger()
        self.gliner_model = gliner_model or GLiNERNERModel()
        self.schema_registry = SchemaRegistry()

    async def detect(
        self,
        text: str,
        hint_document_type: Optional[str] = None,
        hint_country: Optional[str] = None,
        hint_entity: Optional[str] = None,
    ) -> DocumentDetectionResult:
        """
        Detect document type, country, and entity using three-tier detection.

        Args:
            text: The document text to analyze
            hint_document_type: Optional hint for document type
            hint_country: Optional hint for country code (ISO 2-letter)
            hint_entity: Optional hint for entity

        Returns:
            DocumentDetectionResult with detected information and confidence scores
        """
        self.logger.info(
            f"Starting document type detection "
            f"(hints: type={hint_document_type}, country={hint_country}, entity={hint_entity})"
        )

        # Tier 1: Document Type Detection
        type_result = await self._detect_document_type(text, hint_document_type)
        detected_type = type_result[0]
        type_confidence = type_result[1]
        type_method = type_result[2]
        type_keywords = type_result[3]

        self.logger.info(f"Detected document type: {detected_type} (confidence: {type_confidence:.2f})")

        # Tier 2: Country Detection
        country_result = self._detect_country(text, hint_country, detected_type)
        detected_country = country_result[0]
        country_confidence = country_result[1]
        country_keywords = country_result[2]

        self.logger.info(f"Detected country: {detected_country} (confidence: {country_confidence:.2f})")

        # Tier 3: Entity Detection (context-aware)
        entity_result = await self._detect_entity(
            text,
            hint_entity,
            detected_type,
            detected_country
        )
        detected_entity = entity_result[0]
        entity_confidence = entity_result[1]
        entity_keywords = entity_result[2]

        if detected_entity:
            self.logger.info(f"Detected entity: {detected_entity} (confidence: {entity_confidence:.2f})")

        # Calculate overall confidence (weighted average)
        overall_confidence = (
            type_confidence * 0.4 +
            country_confidence * 0.3 +
            entity_confidence * 0.3
        )

        # Get the best matching schema
        schema_used = self.schema_registry.get_schema(
            document_type=detected_type,
            country_code=detected_country,
            entity=detected_entity,
        )

        # Build result
        result = DocumentDetectionResult(
            document_type=detected_type,
            document_type_name=schema_used.document_type_name if schema_used else detected_type,
            country_code=detected_country,
            country_name=schema_used.country_name if schema_used else None,
            entity=detected_entity,
            entity_name=schema_used.entity_name if schema_used else None,
            confidence=overall_confidence,
            type_confidence=type_confidence,
            country_confidence=country_confidence,
            entity_confidence=entity_confidence,
            detected_keywords=list(set(type_keywords + country_keywords + entity_keywords)),
            detected_patterns=[],
            schema_used=schema_used,
            detection_method=type_method,
        )

        self.logger.info(
            f"Detection complete: type={detected_type}, country={detected_country}, "
            f"entity={detected_entity}, overall_confidence={overall_confidence:.2f}"
        )

        return result

    async def _detect_document_type(
        self,
        text: str,
        hint: Optional[str] = None,
    ) -> Tuple[Optional[str], float, str, List[str]]:
        """
        Detect document type using keyword matching and GLiNER2.

        Returns:
            Tuple of (document_type, confidence, method, matched_keywords)
        """
        text_lower = text.lower()

        # If hint is provided and valid, use it
        if hint and hint in DOCUMENT_TYPE_KEYWORDS:
            self.logger.debug(f"Using hint for document type: {hint}")
            return hint, 1.0, "hint", [hint]

        # Score each document type by keyword matches
        type_scores: Dict[str, Tuple[int, List[str]]] = {}

        for doc_type, keywords in DOCUMENT_TYPE_KEYWORDS.items():
            matched = [kw for kw in keywords if kw.lower() in text_lower]
            if matched:
                type_scores[doc_type] = (len(matched), matched)

        if not type_scores:
            # Fallback: Use GLiNER2 to detect organization names that might indicate document type
            return await self._detect_type_with_gliner2(text)

        # Get the best scoring type
        best_type, (match_count, matched_keywords) = max(
            type_scores.items(),
            key=lambda x: x[1][0]
        )

        # Calculate confidence based on match count vs total keywords for that type
        total_keywords = len(DOCUMENT_TYPE_KEYWORDS[best_type])
        confidence = min(match_count / max(total_keywords, 1), 1.0)

        # Boost confidence if we have multiple matches
        if match_count >= 3:
            confidence = min(confidence * 1.2, 1.0)

        return best_type, confidence, "keyword", matched_keywords

    async def _detect_type_with_gliner2(
        self,
        text: str,
    ) -> Tuple[Optional[str], float, str, List[str]]:
        """
        Fallback: Use GLiNER2 to detect document type from organization names.

        Returns:
            Tuple of (document_type, confidence, method, matched_keywords)
        """
        try:
            # Use GLiNER2 to detect organizations
            model = await self.gliner_model.get_model_with_gpu()
            GLiNERClass, gliner_version = self.gliner_model.get_gliner_classes()

            # Try to extract organization names
            schema = model.create_schema().entities({
                "organization": "The name of an organization, company, or government agency",
            })

            entities_dict = model.extract(
                text[:2000],  # Limit text length for performance
                schema=schema,
                threshold=0.3,
                include_confidence=True,
            )

            if entities_dict and "organization" in entities_dict:
                orgs = entities_dict["organization"]
                if isinstance(orgs, list) and len(orgs) > 0:
                    org_names = [o.get("value", "").lower() for o in orgs]

                    # Check for tax-related organizations
                    for org in org_names:
                        if any(kw in org for kw in ["revenue", "tax", "irs", "lhdn"]):
                            return "tax_return", 0.7, "gliner2", [org]

                    # Check for utility companies
                    for org in org_names:
                        if any(kw in org for kw in ["power", "electric", "water", "utility"]):
                            return "utility_bill", 0.7, "gliner2", [org]

                    # Check for banks
                    for org in org_names:
                        if any(kw in org for kw in ["bank", "banking"]):
                            # Note: bank_statement is handled by existing service
                            return "bank_statement", 0.6, "gliner2", [org]

        except Exception as e:
            self.logger.warning(f"GLiNER2 document type detection failed: {e}")

        return None, 0.0, "none", []

    def _detect_country(
        self,
        text: str,
        hint: Optional[str] = None,
        document_type: Optional[str] = None,
    ) -> Tuple[Optional[str], float, List[str]]:
        """
        Detect country using existing patterns and context.

        Returns:
            Tuple of (country_code, confidence, matched_keywords)
        """
        # If hint is provided, validate and use it
        if hint:
            hint_upper = hint.upper()
            # Convert ISO3 to ISO2 if needed
            if hint_upper in ISO3_TO_ISO2:
                hint = ISO3_TO_ISO2[hint_upper]
            elif len(hint) == 2 and hint.isalpha():
                hint = hint.upper()
            else:
                hint = None

            if hint:
                self.logger.debug(f"Using hint for country: {hint}")
                return hint, 1.0, [hint]

        # Use existing country detection
        detected = detect_country_from_text(text)

        if detected:
            # Normalize to ISO2
            country_code = detected.upper()
            if country_code in ISO3_TO_ISO2:
                country_code = ISO3_TO_ISO2[country_code]

            # Look for country-specific keywords to boost confidence
            text_lower = text.lower()
            confidence = 0.7  # Base confidence from pattern match

            # Country-specific keyword boosts
            country_boosters = {
                "SG": ["singapore", "s'pore", "sg", "iras", "cpf", "nric", "posb", "dbs"],
                "IN": ["india", "indian", "pan", "aadhaar", "lakh", "crore", "rupee", "₹"],
                "US": ["united states", "usa", "america", "social security", "irs", "dollar", "$"],
                "MY": ["malaysia", "malaysian", "mykad", "ringgit", "rm"],
                "TH": ["thailand", "thai", "baht", "฿"],
                "AE": ["uae", "united arab emirates", "dubai", "abu dhabi", "emirates", "dirham", "dh"],
            }

            if country_code in country_boosters:
                boosters = country_boosters[country_code]
                matches = sum(1 for kw in boosters if kw in text_lower)
                if matches > 0:
                    confidence = min(confidence + (matches * 0.1), 1.0)

            return country_code, confidence, [detected]

        return None, 0.0, []

    async def _detect_entity(
        self,
        text: str,
        hint: Optional[str] = None,
        document_type: Optional[str] = None,
        country_code: Optional[str] = None,
    ) -> Tuple[Optional[str], float, List[str]]:
        """
        Detect entity (bank, institution, organization) using keywords and GLiNER2.

        Returns:
            Tuple of (entity, confidence, matched_keywords)
        """
        text_lower = text.lower()

        # If hint is provided, validate and use it
        if hint:
            hint_lower = hint.lower()
            # Check if this is a known entity
            for entity, keywords in ENTITY_KEYWORDS.items():
                if hint_lower in keywords or hint_lower == entity:
                    self.logger.debug(f"Using hint for entity: {hint}")
                    return entity, 1.0, [hint]

        # Score each entity by keyword matches
        entity_scores: Dict[str, Tuple[int, List[str]]] = {}

        for entity, keywords in ENTITY_KEYWORDS.items():
            matched = [kw for kw in keywords if kw.lower() in text_lower]
            if matched:
                entity_scores[entity] = (len(matched), matched)

        if entity_scores:
            # Get the best scoring entity
            best_entity, (match_count, matched_keywords) = max(
                entity_scores.items(),
                key=lambda x: x[1][0]
            )

            # Calculate confidence
            confidence = min(match_count / 3, 1.0)  # Normalize to 3 keywords = full confidence
            if match_count >= 2:
                confidence = min(confidence * 1.2, 1.0)

            return best_entity, confidence, matched_keywords

        # Fallback: Use GLiNER2 to extract organization names
        try:
            model = await self.gliner_model.get_model_with_gpu()
            GLiNERClass, gliner_version = self.gliner_model.get_gliner_classes()

            schema = model.create_schema().entities({
                "organization": "The name of a bank, financial institution, or government agency",
            })

            entities_dict = model.extract(
                text[:2000],
                schema=schema,
                threshold=0.35,
                include_confidence=True,
            )

            if entities_dict and "organization" in entities_dict:
                orgs = entities_dict["organization"]
                if isinstance(orgs, list) and len(orgs) > 0:
                    best_org = max(orgs, key=lambda o: o.get("confidence", 0))
                    org_name = best_org.get("value", "").lower()
                    org_conf = best_org.get("confidence", 0)

                    # Try to match against known entities
                    for entity, keywords in ENTITY_KEYWORDS.items():
                        if any(kw in org_name for kw in keywords):
                            return entity, org_conf * 0.8, [org_name]

        except Exception as e:
            self.logger.warning(f"GLiNER2 entity detection failed: {e}")

        return None, 0.0, []


# Convenience function for quick detection

async def detect_document_type(
    text: str,
    hint_document_type: Optional[str] = None,
    hint_country: Optional[str] = None,
    hint_entity: Optional[str] = None,
) -> DocumentDetectionResult:
    """
    Quick detection function for document type, country, and entity.

    Args:
        text: The document text to analyze
        hint_document_type: Optional hint for document type
        hint_country: Optional hint for country code (ISO 2-letter)
        hint_entity: Optional hint for entity

    Returns:
        DocumentDetectionResult with detected information
    """
    detector = DocumentTypeDetector()
    return await detector.detect(
        text=text,
        hint_document_type=hint_document_type,
        hint_country=hint_country,
        hint_entity=hint_entity,
    )


__all__ = [
    "DocumentTypeDetector",
    "DocumentDetectionResult",
    "detect_document_type",
    "EXISTING_DOCUMENT_TYPES",
    "ENTITY_KEYWORDS",
]

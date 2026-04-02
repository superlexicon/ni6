"""
GLiNER NER Model Wrapper for Bank Statement Entity Extraction

This module provides a GLiNER-based Named Entity Recognition model
wrapper integrated with GPU resource manager for extracting
entities from bank statements using zero-shot NER.

GLiNER: Generalist and Lightweight Model for Named Entity Recognition
Paper: https://arxiv.org/abs/2311.15268
Repo: https://github.com/urchade/GLiNER

GLiNER2: Unified Schema-Based Information Extraction
GitHub: https://github.com/fastino-ai/GLiNER2
"""

import logging

from typing import Optional, Dict, List, Any, TYPE_CHECKING

# Lazy import for GLiNER2 to avoid dependency issues
def get_gliner_classes():
    """Lazy import GLiNER2 or GLiNER to avoid dependency issues"""
    try:
        from gliner2 import GLiNER2
        return GLiNER2, "gliner2"
    except ImportError:
        try:
            from gliner import GLiNER
            return GLiNER, "gliner"
        except ImportError as e:
            raise ImportError(
                "Neither GLiNER2 nor GLiNER is installed. Install with: poetry add gliner2"
            ) from e

# Type hints for when type checking
try:
    from gliner2 import GLiNER2
except ImportError:
    from gliner import GLiNER

from app.core.logger import get_logger
from app.core.gpu_manager import ModelType, get_gpu_manager
from app.core.framework_coordinator import get_framework_coordinator, FrameworkType

logger = logging.getLogger(__name__)

# Model configuration
# Using GLiNER2 for improved entity extraction (better person name detection)
# GLiNER2 correctly handles Indian names and addresses better than original GLiNER

# Use local weights if available (downloaded via scripts/download_model_weights.py)
# Falls back to HuggingFace hub if local weights not found
import os
GLINER2_LOCAL_PATH = os.path.join(os.path.dirname(__file__), "..", "gliner2_weights")
GLINER2_MODEL_NAME = GLINER2_LOCAL_PATH if os.path.exists(GLINER2_LOCAL_PATH) else "fastino/gliner2-large-v1"

# Alternative models (uncomment to use):
# GLINER2_MODEL_NAME = "fastino/gliner2-multi-v1"  # Multi-lingual version
# GLINER2_MODEL_NAME = "fastino/gliner2-large-v1"  # Better accuracy

# Legacy GLiNER model (fallback)
GLINER_MODEL_NAME = "urchade/gliner_multi-v2.1"


class GLiNERNERModel:
    """
    GLiNER NER model wrapper integrated with GPU resource manager.

    This class manages both GLiNER and GLiNER2 models:
    - GLiNER2: Better person name detection, correct handling of Indian names
    - GLiNER: Fallback for older installations without GLiNER2
    """

    _instance: Optional['GLiNERNERModel'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GLiNERNERModel, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Only initialize once (singleton pattern)
        if not hasattr(self, 'logger'):
            self.logger = get_logger()
            self.framework_coordinator = get_framework_coordinator()
            self.gpu_manager = get_gpu_manager()
            self._setup_environment()

    def _setup_environment(self):
        """Setup environment for GLiNER model."""
        # Set HF_HOME for caching models
        if not os.environ.get('HF_HOME'):
            cache_dir = os.path.expanduser("~/.cache/huggingface")
            os.makedirs(cache_dir, exist_ok=True)
            os.environ['HF_HOME'] = cache_dir

        self.logger.info(f"HuggingFace cache directory: {os.environ.get('HF_HOME')}")

    def get_model(self) -> 'GLiNER':
        """
        Get GLiNER model with GPU resource management (synchronous version).
        Creates model without async GPU manager for module-level initialization.

        Returns:
            GLiNER model instance
        """
        try:
            # Get PyTorch device
            device = self.framework_coordinator.get_device_string(FrameworkType.PYTORCH)
            self.logger.info(f"Using device for GLiNER: {device}")

            # FIX: Pre-warm PyTorch CUDA initialization before GLiNER loads
            # This prevents "No CUDA GPUs are available" error from GLiNER's internal tokenizer loading
            if device.startswith("cuda:"):
                import torch
                # Force CUDA initialization - torch.cuda.init() alone is not enough
                # We need to actually interact with the CUDA device
                torch.cuda.set_device(device)  # Set the specific CUDA device
                _ = torch.cuda.is_available()  # Force CUDA check
                _ = torch.cuda.device_count()  # Force device enumeration
                self.logger.info(f"PyTorch CUDA pre-initialized: available={torch.cuda.is_available()}, device_count={torch.cuda.device_count()}")

            # Lazy import - try GLiNER2 first, fallback to GLiNER
            GLiNERClass, gliner_version = get_gliner_classes()

            self.logger.info(f"Loading {gliner_version} model: {GLINER2_MODEL_NAME}")

            # Load model - handle different APIs for GLiNER2 vs GLiNER
            if gliner_version == "gliner2":
                # GLiNER2 returns results wrapped as {'entities': {entity_type: [values]}}
                model = GLiNERClass.from_pretrained(GLINER2_MODEL_NAME)
                model = model.to(device)  # Move model to GPU
                self.logger.info(f"{gliner_version} model loaded successfully on {device}")
            else:
                # GLiNER v1 uses map_location parameter
                model = GLiNERClass.from_pretrained(GLINER_MODEL_NAME, map_location=device)
                self.logger.info(f"{gliner_version} model loaded successfully on {device}")

            return model

        except Exception as e:
            self.logger.error(f"Failed to get GLiNER model: {e}")
            raise RuntimeError(f"GLiNER model initialization failed: {str(e)}")

    async def get_model_with_gpu(self) -> 'GLiNER':
        """
        Get GLiNER model with async GPU resource management.

        Returns:
            GLiNER model instance with GPU resources allocated
        """
        try:
            result = await self.gpu_manager.get_model_with_gpu(
                model_type=ModelType.GLINER_NER,
                model_class=self._create_model,
            )
            self.logger.debug("GLiNER model retrieved from GPU manager")
            return result
        except Exception as e:
            self.logger.error(f"Failed to get GLiNER model: {e}")
            raise RuntimeError(f"GLiNER model initialization failed: {str(e)}")

    def _create_model(self):
        """
        Create GLiNER or GLiNER2 model instance.

        Returns:
            Model instance (GLiNER2 or GLiNER)
        """
        try:
            # Get PyTorch device
            device = self.framework_coordinator.get_device_string(FrameworkType.PYTORCH)
            self.logger.info(f"Using device for GLiNER: {device}")

            # FIX: Pre-warm PyTorch CUDA initialization before GLiNER loads
            # This prevents "No CUDA GPUs are available" error from GLiNER's internal tokenizer loading
            if device.startswith("cuda:"):
                import torch
                # Force CUDA initialization - torch.cuda.init() alone is not enough
                # We need to actually interact with the CUDA device
                torch.cuda.set_device(device)  # Set the specific CUDA device
                _ = torch.cuda.is_available()  # Force CUDA check
                _ = torch.cuda.device_count()  # Force device enumeration
                self.logger.info(f"PyTorch CUDA pre-initialized: available={torch.cuda.is_available()}, device_count={torch.cuda.device_count()}")

            # Lazy import - try GLiNER2 first, fallback to GLiNER
            GLiNERClass, gliner_version = get_gliner_classes()

            self.logger.info(f"Loading {gliner_version} model: {GLINER2_MODEL_NAME}")

            # Load model - handle different APIs for GLiNER2 vs GLiNER
            if gliner_version == "gliner2":
                # GLiNER2.from_pretrained() returns the model directly
                model = GLiNERClass.from_pretrained(GLINER2_MODEL_NAME)
                model = model.to(device)  # Move model to GPU
                self.logger.info(f"{gliner_version} model loaded successfully on {device}")
            else:
                # GLiNER v1 uses map_location parameter
                model = GLiNERClass.from_pretrained(GLINER_MODEL_NAME, map_location=device)
                self.logger.info(f"{gliner_version} model loaded successfully on {device}")

            return model

        except Exception as e:
            self.logger.error(f"Failed to create GLiNER model: {str(e)}")
            raise

    async def release_model(self) -> bool:
        """
        Release GLiNER model resources.

        Returns:
            True if model was released successfully
        """
        try:
            released = await self.gpu_manager.release_model(ModelType.GLINER_NER)
            if released:
                self.logger.info("GLiNER model resources released")
            return released
        except Exception as e:
            self.logger.error(f"Failed to release GLiNER model: {e}")
            return False

    def extract_bank_statement_entities(
        self,
        text: str,
        labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Extract entities from bank statement text using GLiNER (synchronous).

        Args:
            text: Bank statement OCR text
            labels: Custom entity labels to extract. If None, uses default bank statement labels.

        Returns:
            Dictionary mapping entity types to extracted values with confidence scores
        """
        if labels is None:
            labels = self._get_bank_statement_labels()

        try:
            model = self.get_model()

            # Check which GLiNER version is being used
            GLiNERClass, gliner_version = get_gliner_classes()

            # Run zero-shot NER with threshold for low-confidence filtering
            # GLiNER2 and GLiNER have different APIs
            if gliner_version == "gliner2":
                # GLiNER2 returns results wrapped as {'entities': {entity_type: [values]}}
                entities_dict = model.extract_entities(text, labels, threshold=0.3)
                # Flatten nested dict into separate entities
                entities_list = []
                if 'entities' in entities_dict:
                    for entity_type, values in entities_dict['entities'].items():
                        if isinstance(values, list):
                            for value in values:
                                score = value.get("score", 0.5) if isinstance(value, dict) else 0.5
                                entity_text = value if isinstance(value, str) else value.get("text", "")
                                entities_list.append({"text": entity_text, "label": entity_type, "score": score})
            else:
                # GLiNER v1 returns a List of dicts
                entities_list = model.predict_entities(text, labels, threshold=0.3)

            # Group and filter results - expects dict from GLiNER results
            results = self._group_entities(entities_list, labels)

            # Log special case for multiple addresses
            if results.get('address') and isinstance(results['address'], list):
                self.logger.info(f"GLiNER extracted {len(results['address'])} addresses")

            self.logger.info(f"GLiNER extracted {len([r for r in results.values() if r])} entity types")
            return results

        except Exception as e:
            self.logger.error(f"GLiNER extraction failed: {e}")
            return {label: None for label in labels}

    async def extract_bank_statement_with_schema_async(
        self,
        text: str,
    ) -> Dict[str, Any]:
        """
        Extract entities from bank statement text using GLiNER2 schema-based extraction.

        GLiNER2's schema feature uses natural language descriptions to guide extraction,
        which provides better context for detecting label-value pairs like account numbers.

        Args:
            text: Bank statement OCR text

        Returns:
            Dictionary mapping entity types to extracted values with confidence scores
        """
        try:
            model = await self.get_model_with_gpu()
            GLiNERClass, gliner_version = get_gliner_classes()

            # Only GLiNER2 supports schema-based extraction
            if gliner_version != "gliner2":
                self.logger.warning("Schema extraction requires GLiNER2, falling back to labels")
                return await self.extract_bank_statement_entities_async(text)

            # Build schema using GLiNER2's Schema builder
            # The schema is a dict where keys are field names and values are descriptions
            entity_types = self._get_bank_statement_schema()

            # Create schema using GLiNER2's API
            schema = model.create_schema().entities(entity_types)

            # Run schema-based extraction using extract() method
            entities_dict = model.extract(
                text,
                schema=schema,
                threshold=0.3,  # Lowered from 0.4 to handle image-based OCR quality
                include_confidence=True,
                include_spans=True  # Enable span extraction for boundary detection
            )

            # Process GLiNER2 schema output format
            # Returns: {"entities": {field_name: [{"text": str, "confidence": float}, ...]}}
            results = {}

            # Extract entities from the nested "entities" key
            entities_data = entities_dict.get("entities", entities_dict)

            for field_name, values in entities_data.items():
                if values is None:
                    results[field_name] = None
                    continue

                if isinstance(values, list) and len(values) > 0:
                    # Take the highest confidence value
                    best = max(values, key=lambda v: v.get('confidence', v.get('score', 0)))
                    results[field_name] = {
                        'value': best.get('text', ''),
                        'confidence': best.get('confidence', best.get('score', 0.0))
                    }
                elif isinstance(values, dict):
                    # Single value as dict
                    results[field_name] = {
                        'value': values.get('text', ''),
                        'confidence': values.get('confidence', values.get('score', 0.0))
                    }
                elif isinstance(values, str):
                    # Simple string value
                    results[field_name] = {
                        'value': values,
                        'confidence': 0.5
                    }

            self.logger.info(f"GLiNER2 schema extracted {len(results)} entity types")

            # If schema extraction returned no results, fall back to labels-based extraction
            if len(results) == 0:
                self.logger.warning("Schema extraction returned no entities, falling back to labels-based extraction")
                return await self.extract_bank_statement_entities_async(text)

            return results

        except Exception as e:
            self.logger.error(f"GLiNER2 schema extraction failed: {e}")
            # Fallback to labels-based extraction
            self.logger.info("Falling back to labels-based extraction")
            return await self.extract_bank_statement_entities_async(text)

    def _get_known_bank_names(self) -> str:
        """
        Load known bank names from config.json for GLiNER schema.

        Uses the comprehensive bank configuration with 229+ alternate names
        from reference_templates/bank_statements/config.json.

        Returns a comma-separated string of bank names for the schema description.
        """
        try:
            from app.core.key_injection.bank_lookup import BankLookup

            bl = BankLookup.get_instance()

            # Get all bank names from alternate_names_map (these are the full names)
            all_names = list(bl._alternate_names_lower.keys())

            # Also add abbreviations
            all_names.extend(bl._abbreviations)

            # Remove duplicates and sort
            unique_names = sorted(set(all_names))

            # Prioritize banks from commonly processed countries
            # by including key banks first
            priority_banks = [
                'dbs', 'ocbc', 'uob', 'posb',  # Singapore
                'emirates nbd', 'adcb', 'dib', 'mashreq', 'rakbank',  # UAE
                'hdfc', 'icici', 'axis', 'sbi', 'kotak',  # India
                'maybank', 'cimb', 'public bank',  # Malaysia
                'bangkok bank', 'krung thai', 'kasikorn',  # Thailand
                'hsbc', 'standard chartered', 'citibank',  # International
            ]

            prioritized = []
            remaining = []

            for name in unique_names:
                name_lower = name.lower()
                # Check if any priority bank is a substring of this name
                is_priority = any(pb in name_lower for pb in priority_banks)
                if is_priority:
                    prioritized.append(name)
                else:
                    remaining.append(name)

            # Combine prioritized with remaining, limit to 150 names
            final_names = prioritized + remaining
            final_names = final_names[:150]

            if final_names:
                self.logger.info(f"Loaded {len(final_names)} bank names from config.json")
                return ', '.join(final_names)

            # Fallback if no names found
            return self._get_default_bank_names()

        except Exception as e:
            self.logger.warning(f"Failed to load bank names from config: {e}")
            return self._get_default_bank_names()

    def _get_default_bank_names(self) -> str:
        """Fallback bank names if JSON load fails."""
        return "Emirates NBD, HSBC, HDFC Bank, ICICI Bank, Axis Bank, DBS Bank, " \
               "State Bank of India, Abu Dhabi Commercial Bank, Dubai Islamic Bank, " \
               "Mashreq Bank, RAK Bank, OCBC Bank, UOB, ANZ, Standard Chartered"

    def _get_bank_statement_schema(self) -> Dict[str, str]:
        """
        Get GLiNER2 schema for bank statement extraction.

        Single flexible schema that handles both tabular and letterhead formats.
        Returns a dict where keys are field names and values are natural language descriptions.

        Returns:
            Dictionary mapping field names to description strings
        """
        return {
            # Primary identification fields
            "bank_name": f"The official name of the bank or financial institution, NOT the branch location. This is the institution name like 'Emirates NBD' 'HSBC' 'HDFC Bank' 'DBS Bank'. Common bank names: {self._get_known_bank_names()}. Do NOT include branch location names like 'Ibn Battuta Mall' or 'Connaught Place' in the bank name - those go in branch_name. The bank name appears as a header logo at the top, or in website URLs like www.emiratesnbd.com.",

            # Account holder name - all person names should map here
            "account_holder_name": "The person's full legal name. This appears immediately after labels like 'Name' 'Customer Name' 'Account Holder Name' 'A/C Holder Name' 'Primary Account Holder Name' 'Account Holder' 'Customer' or may appear with a colon prefix like ': JOHN SMITH'. The name may appear with titles like Mr Mrs Ms Dr Shri Smt before the name or without titles. Names are 2-4 words in ALL CAPS or Title Case (e.g., 'JOHN SMITH' 'JANE DOE' 'ROBERT JOHN DOE'). Names with single-letter initials are also valid (e.g., 'A B SMITH' 'J P DOE'). The name may include patronymic markers like S/O D/O A/L meaning Son Of Daughter Of (e.g., 'JOHN SMITH S/O ROBERT DOE'). A valid name contains ONLY letters and spaces - NEVER numbers or prefixes like PAN CIF ACC NO A/C NO. DO NOT extract city names, area names, or location names like 'Bangalore' 'Mumbai' 'Delhi' - these are NOT person names.",


            "account_number": "The unique numeric identifier for the bank account. This appears immediately to the right of or on the next line after labels like Account Number Account No A/C No Savings A/C Current A/C. This is the customer's bank account number NOT the CIF number or Customer ID.",
            "cif_number": "The Customer Identification File number used internally by the bank. This appears after labels like CIF Number or CIF or Cust ID. This is a numeric identifier that is NOT the account number.",

            # Address fields - explicitly distinguish customer vs branch
            "customer_address": "The customer's permanent residential postal mailing address consisting of multiple lines. Contains house number with h no hno or plot number, or block number followed by street name, then city/town, then state/province, then postal/zip/pin code. The address NEVER includes the person's name or title. The address NEVER includes the bank's branch address. Do NOT extract OCR noise patterns like random special characters repeated dots dashes.",

            "branch_address": "The bank branch's physical location address including building name complex name floor number street near landmark area city state pin code. This is the bank's address NOT the customer's address. This appears near labels like Base Branch Branch Address Registered Office Corporate Office Head Office Your Base Branch.",

            # Additional fields
            "branch_name": "The specific branch location or area name ONLY, not the bank name. This appears after the bank name or near labels like 'Branch' 'Location'. Do NOT include the bank institution name in branch_name.",
            "currency": "The 3-letter ISO 4217 currency code. For DBS: SGD. For Indian banks: INR. For international: USD EUR GBP etc. This is ONLY a 3-letter code like SGD INR USD EUR. Do NOT extract serial numbers like S/N: EN05301101135929 or similar identification numbers. Do NOT extract the word Currency itself.",
            "statement_date": "The statement date appearing as a date in formats like DD MMM YYYY or DD/MM/YYYY or MM/DD/YYYY. When a date RANGE appears, extract the LATEST/END date. This appears near labels like 'Statement Date' 'as at' 'Statement for the period'.",
        }

    async def extract_bank_statement_entities_async(
        self,
        text: str,
        labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Extract entities from bank statement text using GLiNER (async).

        Args:
            text: Bank statement OCR text
            labels: Custom entity labels to extract. If None, uses default bank statement labels.

        Returns:
            Dictionary mapping entity types to extracted values with confidence scores
        """
        if labels is None:
            labels = self._get_bank_statement_labels()

        try:
            model = await self.get_model_with_gpu()

            # Check which GLiNER version is being used
            GLiNERClass, gliner_version = get_gliner_classes()

            # Run zero-shot NER with threshold for low-confidence filtering
            # GLiNER2 and GLiNER have different APIs
            if gliner_version == "gliner2":
                # GLiNER2 returns results wrapped as {'entities': {entity_type: [values]}}
                entities_dict = model.extract_entities(text, labels, threshold=0.3)
                # Flatten nested dict into separate entities
                entities_list = []
                if 'entities' in entities_dict:
                    for entity_type, values in entities_dict['entities'].items():
                        if isinstance(values, list):
                            for value in values:
                                score = value.get("score", 0.5) if isinstance(value, dict) else 0.5
                                entity_text = value if isinstance(value, str) else value.get("text", "")
                                entities_list.append({"text": entity_text, "label": entity_type, "score": score})
            else:
                # GLiNER v1 returns a List of dicts
                entities_list = model.predict_entities(text, labels, threshold=0.3)

            # Group and filter results - expects dict from GLiNER results
            results = self._group_entities(entities_list, labels)

            # Log special case for multiple addresses
            if results.get('address') and isinstance(results['address'], list):
                self.logger.info(f"GLiNER extracted {len(results['address'])} addresses")

            self.logger.info(f"GLiNER extracted {len([r for r in results.values() if r])} entity types")
            return results

        except Exception as e:
            self.logger.error(f"GLiNER extraction failed: {e}")
            return {label: None for label in labels}

    def _get_bank_statement_labels(self) -> List[str]:
        """Get default entity labels for bank statement extraction."""
        return [
            # Generic patterns (existing)
            "account holder name including S/O D/O A/L patterns",
            "customer full name with S/O D/O A/L",
            "person name or customer name with title or honorific like Mr Mrs Ms Dr",

            # Label-aware patterns for better extraction from label-value pairs
            # These help GLiNER2 learn that names appearing after specific labels are the account holder
            "name after Account Holder Names or Account Holder label",
            "name after Customer Name label",
            "value after Name label in bank statement",
            "customer name appearing after Account Holder or Name heading",
            "customer name value after Name or Account Holder heading",
            "account holder name value appearing after its label",

            # Bank & Account entities
            "bank name",
            "bank or financial institution name",
            "bank of maharashtra",
            "name of bank at top of statement",
            "account number",
            "currency",
            "ifsc code",
            "customer identification file number or CIF number",

            # GLiNER2: Explicitly distinguish between customer and branch addresses
            # GLiNER2's context understanding allows it to map the correct address to each label
            "customer address",
            "customer residential address",
            "customer permanent address",
            "customer home address",
            "bank branch address",
            "branch address or bank location",
            "communication address or correspondence address",

            # Additional address detection patterns for Indian addresses
            # Help GLiNER recognize addresses with house numbers, landmarks, pin codes
            "address with house number and street name",
            "address with near landmark or temple",
            "address with village or locality name",
            "postal address with pin code or zip code",
            "complete mailing address with area pin code",

            # Positional-aware address labels for formal letter format
            # These labels help GLiNER detect multi-line addresses that appear below account holder name
            "multi line customer address appearing below account holder name in formal letter format",
            "customer address in top left of document with multiple lines",
            "residential address spanning multiple lines below customer name",
            "address blocks directly under account holder name in letter format",
            "full customer address appearing directly after name in letter format",

            # Other entities
            "branch",
            "statement date"
        ]

    def _get_id_card_labels(self) -> List[str]:
        """
        Get default entity labels for ID card extraction.

        These labels are designed for zero-shot NER to extract common ID card fields.
        Labels include descriptive patterns to help GLiNER identify the correct entities.
        """
        return [
            # Personal Information
            "full name or person name including S/O D/O A/L patterns",
            "given name or first name",
            "surname or last name",
            "father's name or father name",
            "mother's name or mother name",
            "date of birth or birth date",
            "gender or sex M F",

            # Document Information
            "identification number or ID number or card number",
            "PAN number or permanent account number",
            "driver license number or driving licence number",
            "national ID number or citizenship number",
            "passport number or document number",

            # Dates
            "issue date or date of issue",
            "expiry date or expiration date or valid until",

            # Location
            "address or residential address",
            "city or town",
            "state or province or region",
            "postal code or zip code",
            "country or nationality",

            # Authority/Issuing
            "issuing authority or issuing office",
            "place of birth or birthplace",

            # Additional Fields (varies by card type)
            "blood group or blood type",
        ]

    async def extract_id_card_entities(
        self,
        text: str,
        labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Extract entities from ID card text using GLiNER (async).

        Similar pattern to extract_bank_statement_entities_async().
        """
        if labels is None:
            labels = self._get_id_card_labels()

        try:
            model = await self.get_model_with_gpu()

            # Check which GLiNER version is being used
            GLiNERClass, gliner_version = get_gliner_classes()

            # Run zero-shot NER with threshold for low-confidence filtering
            # GLiNER2 and GLiNER have different APIs
            if gliner_version == "gliner2":
                # GLiNER2 returns results wrapped as {'entities': {entity_type: [values]}}
                entities_dict = model.extract_entities(text, labels, threshold=0.3)
                # Flatten nested dict into separate entities
                entities = []
                if 'entities' in entities_dict:
                    for entity_type, values in entities_dict['entities'].items():
                        if isinstance(values, list):
                            for value in values:
                                score = value.get("score", 0.5) if isinstance(value, dict) else 0.5
                                entity_text = value if isinstance(value, str) else value.get("text", "")
                                entities.append({"text": entity_text, "label": entity_type, "score": score})
            else:
                # GLiNER v1 returns a List of dicts
                entities = model.predict_entities(text, labels, threshold=0.3)

            # Group and filter results
            results = self._group_id_card_entities(entities, labels)

            self.logger.info(f"GLiNER extracted {len([r for r in results.values() if r])} ID card entity types")
            return results

        except Exception as e:
            self.logger.error(f"GLiNER extraction failed: {e}")
            return {label: None for label in labels}

    def _group_id_card_entities(self, entities: List[Dict], labels: List[str]) -> Dict[str, Any]:
        """
        Group GLiNER entities by label type for ID card extraction.

        Similar to _group_entities but optimized for ID card fields.
        For name-type labels, returns all entities as a list.
        For other single-value labels, returns only the highest confidence entity.
        """
        # Log raw entities for debugging
        self.logger.debug(f"Raw GLiNER entities ({len(entities)}): {entities}")

        # Labels that can have multiple values
        multi_value_labels = {'full name or person name including s/o d/o a/l patterns'}

        # Initialize results with None
        results = {}

        for entity in entities:
            text = entity.get("text", "").strip()
            label = entity.get("label", "").lower()
            score = entity.get("score", 0.0)

            # Log each entity being processed
            self.logger.debug(f"Processing entity: text='{text}', label='{label}', score={score:.3f}")

            # Skip low confidence predictions
            if score < 0.3:
                self.logger.debug(f"  → Skipped (confidence {score:.3f} < 0.3)")
                continue

            # Normalize label to our schema
            normalized_label = self._normalize_label(label, labels)

            if not normalized_label:
                self.logger.debug(f"  → Skipped (label '{label}' could not be normalized)")
                continue

            # For multi-value labels (names), collect all entities
            if normalized_label in multi_value_labels:
                if normalized_label not in results:
                    results[normalized_label] = []
                results[normalized_label].append({
                        "value": text,
                        "confidence": score
                    })
                self.logger.debug(f"  → Accepted as '{normalized_label}' (multi-value)")
            else:
                # For single-value labels, keep highest confidence only
                if normalized_label not in results or score > results[normalized_label].get("confidence", 0):
                    results[normalized_label] = {
                        "value": text,
                            "confidence": score
                        }
                    self.logger.debug(f"  → Accepted as '{normalized_label}'")

        # Ensure all labels have entries
        for label in labels:
            normalized = self._normalize_id_card_label(label.lower(), labels)
            if normalized and normalized not in results:
                results[normalized] = None

        return results

    def _normalize_id_card_label(self, label: str, valid_labels: List[str]) -> Optional[str]:
        """
        Normalize GLiNER label to ID card schema field name.

        Args:
            label: Label from GLiNER
            valid_labels: List of valid ID card labels

        Returns:
            Normalized label name or None if invalid
        """
        label_lower = label.lower()

        # Direct match
        if label_lower in [l.lower() for l in valid_labels]:
            return label_lower

        # Map common variations for ID card fields
        mappings = {
            # Name mappings
            "full name or person name including s/o d/o a/l patterns": "full_name",
            "given name or first name": "given_name",
            "surname or last name": "last_name",
            "father's name or father name": "father_name",
            "mother's name or mother name": "mother_name",
            "person": "full_name",  # Generic person label
            "name": "full_name",  # Short form

            # ID number mappings
            "identification number or id number or card number": "identification_number",
            "pan number or permanent account number": "pan_number",
            "driver license": "driver_license_number",
            "passport number": "passport_number",

            # Date mappings
            "date of birth or birth date": "date_of_birth",
            "dob": "date_of_birth",
            "issue date": "issue_date",
            "expiry date": "expiry_date",

            # Location mappings
            "address": "address",
            "city": "city",
            "state": "state",
            "postal code": "postal_code",

            # Additional
            "blood group": "blood_group",
        }

        for key, value in mappings.items():
            if key in label_lower:
                return value

        # Partial match
        for valid_label in valid_labels:
            if valid_label.lower() in label_lower or label_lower in valid_label.lower():
                return valid_label.lower()

        return None

    def _group_entities(self, entities: List[Dict], labels: List[str]) -> Dict[str, Any]:
        """
        Group GLiNER entities by label type and filter by confidence.

        For multi-value labels like 'address', returns all entities above threshold.
        For single-value labels, returns only the highest confidence entity.

        Args:
            entities: List of entity dicts from GLiNER with 'text', 'label', 'score' keys
            labels: List of valid entity labels

        Returns:
            Dictionary mapping normalized label names to entity/ies with confidence.
            Multi-value labels return a list, single-value labels return a dict.
        """
        # Log raw entities for debugging
        self.logger.debug(f"Raw GLiNER entities ({len(entities)}): {entities}")

        # Labels that can have multiple values
        multi_value_labels = {'address', 'branch'}

        # Initialize results with None
        results = {}

        for entity in entities:
            # GLiNER2 wraps results in {'entities': {...}} format
            # When 'text' value is a dict (nested entities), unwrap and skip it
            entity_text_raw = entity.get("text", "")
            if isinstance(entity_text_raw, dict):
                # GLiNER2 nested dict - skip wrapper entity, process inner entities separately
                continue
            text = entity_text_raw.strip() if isinstance(entity_text_raw, str) else ""
            label = entity.get("label", "").lower()
            score = entity.get("score", 0.0)

            # Log each entity being processed
            self.logger.debug(f"Processing entity: text='{text}', label='{label}', score={score:.3f}")

            # Skip low confidence predictions
            if score < 0.3:
                self.logger.debug(f"  → Skipped (confidence {score:.3f} < 0.3)")
                continue

            # Normalize label to our schema
            normalized_label = self._normalize_label(label, labels)

            if not normalized_label:
                self.logger.debug(f"  → Skipped (label '{label}' could not be normalized)")
                continue

            # For multi-value labels, collect all entities
            if normalized_label in multi_value_labels:
                if normalized_label not in results:
                    results[normalized_label] = []
                results[normalized_label].append({
                        "value": text,
                        "confidence": score
                    })
                self.logger.debug(f"  → Accepted as '{normalized_label}' (multi-value)")
            else:
                # For single-value labels, keep highest confidence only
                if normalized_label not in results or score > results[normalized_label].get("confidence", 0):
                    results[normalized_label] = {
                        "value": text,
                            "confidence": score
                        }
                    self.logger.debug(f"  → Accepted as '{normalized_label}'")

        # Ensure all labels have entries
        for label in labels:
            normalized = self._normalize_label(label.lower(), labels)
            if normalized and normalized not in results:
                results[normalized] = None

        return results

    def _normalize_label(self, label: str, valid_labels: List[str]) -> Optional[str]:
        """
        Normalize GLiNER label to our schema.

        Args:
            label: Label from GLiNER
            valid_labels: List of valid entity labels

        Returns:
            Normalized label name or None if invalid
        """
        label_lower = label.lower()

        # Direct match
        if label_lower in [l.lower() for l in valid_labels]:
            return label_lower

        # Map common variations
        mappings = {
            "account holder": "account holder name",
            "customer": "account holder name",
            "person": "account holder name",
            "holder": "account holder name",
            "name": "account holder name",
            "bank name": "bank name",
        }

        for key, value in mappings.items():
            if key in label_lower:
                return value

        return None

    def _post_process_misclassified_names(
        self,
        entities: List[Dict],
        results: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Post-process GLiNER entities to find misclassified account holder names.

        GLiNER often labels person names as "bank name", "organization", etc.
        This method scans all entities and finds those that look like person names
        but were misclassified.

        S/O Name patterns are HIGHEST PRIORITY - if found, return immediately.

        Args:
            entities: Raw GLiNER entities with 'text', 'label', 'score'
            results: Current grouped results from _group_entities()

        Returns:
            Updated account holder name dict if a misclassified name is found, else None
        """
        import re

        # Only run if we don't already have a high-confidence name
        current_name = results.get("account holder name")
        if current_name and current_name.get("confidence", 0) >= 0.7:
            return None  # Already have a good name

        # HIGHEST PRIORITY - Check for S/O, D/O, A/L patterns first
        for entity in entities:
            text = entity.get("text", "").strip()
            score = entity.get("score", 0.0)

            # Skip low confidence
            if score < 0.3:
                continue

            # Check for S/O, D/O, A/L patterns (Malaysian/Indian name markers)
            if re.search(r'\s+(S/O|D/O|A/L|S/O\.|D/O\.|A/L\.)\s+', text, re.IGNORECASE):
                # Quick validation - make sure it's not just "S/O" alone
                if len(text.strip()) > 10 and text.count(' ') > 2:
                    # Valid name - should have more than just the marker
                    name_score = self._is_misclassified_person_name(text, entity.get("label", ""), score)
                    # If it passes basic validation (not obviously an address), use it
                    if name_score >= 0.3:
                        self.logger.info(
                            f"GLiNER post-processing: Found S/O pattern '{text}' "
                            f"(conf={score:.2f})"
                        )
                        return {
                            "value": self._remove_name_titles(text),
                            "confidence": min(score * 0.95, 0.98),
                            "reclassified": True,
                            "so_pattern": True
                        }

        best_candidate = None
        best_score = 0.0

        for entity in entities:
            text = entity.get("text", "").strip()
            label = entity.get("label", "").lower()
            score = entity.get("score", 0.0)

            # Skip if already labeled as account holder
            if label == "account holder name" or label == "customer name":
                continue

            # Check if this looks like a misclassified person name
            name_score = self._is_misclassified_person_name(text, label, score)
            if name_score > best_score:
                best_score = name_score
                best_candidate = {
                    "value": text,
                    "confidence": min(score * 0.9, 0.98)
                }

        if best_candidate:
            self.logger.info(
                f"GLiNER post-processing: Reclassified '{best_candidate['value']}' "
                f"(orig label was misclassified, conf={best_candidate['confidence']:.2f})"
            )
            return best_candidate

        return None

    def _is_misclassified_person_name(self, text: str, label: str, score: float) -> float:
        """
        Determine if an entity is a misclassified person name.

        Returns a confidence score (0-1) indicating likelihood of being a person name.
        """
        import re

        text_upper = text.upper()
        name_score = 0.0

        # Strong indicators: S/O, D/O, A/L patterns (Malaysian/Indian names)
        if re.search(r'\s+(S/O|D/O|A/L|S/O\.|D/O\.|A/L\.)\s+', text_upper):
            name_score += 0.6

        # Strong indicators: Common name prefixes (MR., MRS., MS., DR.)
        if re.match(r'^(MR\.?|MRS\.?|MS\.?|DR\.?|MISS\.?)\s+[A-Z][A-Z\-\'\.]*(?:\s+[A-Z][A-Z\-\'\.]*)*', text_upper):
            name_score += 0.4

        # Pattern: Title-case person name (2-4 words, no digits)
        if re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}$', text):
            name_score += 0.3

        # Pattern: All caps person name (common in some bank statements)
        if re.match(r'^[A-Z][A-Z\s\-\']{2,50}$', text_upper):
            name_score += 0.2

        # Pattern: Name with S/O marker (Indian naming pattern)
        if re.search(r'(S/O|D/O|A/L)', text_upper):
            name_score += 0.5

        # GLiNER label analysis - boost for labels that commonly misclassify names
        misleading_labels = ['bank name', 'branch', 'organization', 'org', 'company', 'ltd', 'corp']
        if any(ml in label for ml in misleading_labels):
            name_score += 0.2

        # Penalty: Contains typical banking/business terminology
        banking_terms = ['bank', 'branch', 'atm', 'credit', 'debit', 'account', 'balance', 'statement', 'limited', 'finance']
        if any(term in text_upper.split() for term in banking_terms):
            name_score -= 0.3

        # Penalty: Contains numbers (unless S/O pattern)
        if re.search(r'(S/O|D/O|A/L)', text_upper):
            pass  # Don't penalize S/O names
        elif re.search(r'\d', text):
            name_score -= 0.2

        # Penalty: Too short (single word < 3 chars) or too long (> 50 chars)
        if len(text) < 3:
            name_score -= 0.1
        elif len(text) > 50:
            name_score -= 0.1

        return max(0.0, min(name_score, 1.0))

    # ========================================================================
    # SCHEMA REGISTRY HELPER METHODS FOR GENERIC DOCUMENT EXTRACTION
    # ========================================================================

    async def extract_with_schema_async(
        self,
        text: str,
        field_descriptions: Dict[str, str],
        threshold: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Extract entities using GLiNER2 schema-based extraction.

        This is a generic method for any document type using natural language
        field descriptions. Used by the generic document service.

        Args:
            text: OCR text to extract from
            field_descriptions: Dict mapping field names to natural language descriptions
            threshold: Confidence threshold for extraction (0-1)

        Returns:
            Dictionary mapping field names to extracted values with confidence
        """
        try:
            model = await self.get_model_with_gpu()
            GLiNERClass, gliner_version = get_gliner_classes()

            # Only GLiNER2 supports schema-based extraction
            if gliner_version != "gliner2":
                self.logger.warning("Schema extraction requires GLiNER2, returning empty")
                return {}

            # Create schema using GLiNER2's API
            schema = model.create_schema().entities(field_descriptions)

            # Run schema-based extraction
            entities_dict = model.extract(
                text,
                schema=schema,
                threshold=threshold,
                include_confidence=True,
                include_spans=True
            )

            # Process results into standard format
            results = {}
            for field_name, entities in entities_dict.items():
                if entities is None:
                    results[field_name] = None
                    continue

                if isinstance(entities, list) and len(entities) > 0:
                    # Get the highest confidence entity
                    best = max(entities, key=lambda e: e.get('confidence', 0))
                    results[field_name] = {
                        'value': best.get('value', '').strip(),
                        'confidence': best.get('confidence', 0.0),
                    }
                elif isinstance(entities, dict):
                    results[field_name] = {
                        'value': entities.get('value', '').strip(),
                        'confidence': entities.get('confidence', 0.0),
                    }
                else:
                    results[field_name] = {
                        'value': str(entities).strip(),
                        'confidence': 0.5,
                    }

            return results

        except Exception as e:
            self.logger.error(f"GLiNER2 schema extraction failed: {e}")
            return {}

    def get_gliner_classes(self):
        """Get GLiNER class and version. Lazily imports GLiNER2 or GLiNER."""
        return get_gliner_classes()


# Global instance for module-level access
_gliner_ner_model_instance: Optional[GLiNERNERModel] = None


def get_gliner_ner_model() -> GLiNERNERModel:
    """Get lazy-loaded GLiNER NER model instance."""
    global _gliner_ner_model_instance
    if _gliner_ner_model_instance is None:
        _gliner_ner_model_instance = GLiNERNERModel()
    return _gliner_ner_model_instance

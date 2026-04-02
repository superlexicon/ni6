"""
BERT NER Model Wrapper for Resume Entity Extraction

This module provides a BERT-based Named Entity Recognition model
wrapper integrated with the GPU resource manager for extracting
entities from resumes.
"""

import logging
import os
from typing import Optional, Tuple, TYPE_CHECKING

# Import model classes lazily to avoid dependency issues
def get_transforms_classes():
    """Lazy import transformers to avoid dependency issues"""
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    return AutoModelForTokenClassification, AutoTokenizer

# Type hints for when type checking
if TYPE_CHECKING:
    from transformers import AutoModelForTokenClassification, AutoTokenizer

from app.core.logger import get_logger
from app.core.gpu_manager import ModelType, get_gpu_manager
from app.core.framework_coordinator import get_framework_coordinator, FrameworkType

logger = logging.getLogger(__name__)

# Model configuration
BERT_NER_MODEL_NAME = "yashpwr/resume-ner-bert-v2"


class BertNerModel:
    """
    BERT NER model wrapper integrated with GPU resource manager.

    This class manages the BERT-based Named Entity Recognition model
    with centralized GPU resource allocation and memory management.
    """

    _instance: Optional['BertNerModel'] = None
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BertNerModel, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.gpu_manager = get_gpu_manager()
            self.framework_coordinator = get_framework_coordinator()
            self.logger = get_logger()
            self._initialized = True
            self._setup_environment()
            logger.info("BertNerModel initialized with unified framework coordinator")

    def _setup_environment(self):
        """Setup environment for BERT NER model."""
        # Disable parallelism in tokenizers to avoid warnings
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        # Set HF_HOME for caching models
        if not os.environ.get('HF_HOME'):
            cache_dir = os.path.expanduser("~/.cache/huggingface")
            os.makedirs(cache_dir, exist_ok=True)
            os.environ['HF_HOME'] = cache_dir

        self.logger.info(f"HuggingFace cache directory: {os.environ.get('HF_HOME')}")

    def get_model(self) -> Tuple['AutoModelForTokenClassification', 'AutoTokenizer']:
        """
        Get BERT NER model with GPU resource management (synchronous version).
        Creates model without async GPU manager for module-level initialization.

        Returns:
            Tuple of (model, tokenizer)
        """
        try:
            # Direct model creation for module initialization
            # GPU resources will be managed on first use
            model, tokenizer = self._create_model()

            self.logger.debug("BERT NER model created (GPU resources allocated on first use)")
            return model, tokenizer

        except Exception as e:
            self.logger.error(f"Failed to get BERT NER model: {e}")
            raise RuntimeError(f"BERT NER model initialization failed: {str(e)}")

    async def get_model_with_gpu(self) -> Tuple['AutoModelForTokenClassification', 'AutoTokenizer']:
        """
        Get BERT NER model with async GPU resource management.

        Returns:
            Tuple of (model, tokenizer) with GPU resources allocated
        """
        try:
            # Get model with GPU resource management
            result = await self.gpu_manager.get_model_with_gpu(
                model_type=ModelType.BERT_NER,
                model_class=self._create_model_wrapper,
            )

            self.logger.debug("BERT NER model retrieved from GPU manager")
            return result

        except Exception as e:
            self.logger.error(f"Failed to get BERT NER model: {e}")
            raise RuntimeError(f"BERT NER model initialization failed: {str(e)}")

    def _create_model_wrapper(self) -> Tuple['AutoModelForTokenClassification', 'AutoTokenizer']:
        """
        Wrapper for model creation that returns a tuple.
        This is needed because the GPU manager expects a single model instance,
        but we need to return both model and tokenizer.

        Returns:
            Tuple of (model, tokenizer)
        """
        return self._create_model()

    def _create_model(self) -> Tuple['AutoModelForTokenClassification', 'AutoTokenizer']:
        """
        Create BERT NER model instance.

        Returns:
            Tuple of (model, tokenizer)
        """
        try:
            # Get PyTorch device
            device = self.framework_coordinator.get_device_string(FrameworkType.PYTORCH)
            self.logger.info(f"Using device for BERT NER: {device}")

            # Lazy import transformers
            AutoModelForTokenClassification, AutoTokenizer = get_transforms_classes()

            self.logger.info(f"Loading BERT NER model: {BERT_NER_MODEL_NAME}")

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(BERT_NER_MODEL_NAME)

            # Load model
            model = AutoModelForTokenClassification.from_pretrained(BERT_NER_MODEL_NAME)

            # Move model to device
            model = model.to(device)
            model.eval()  # Set to evaluation mode

            self.logger.info(f"BERT NER model loaded successfully on {device}")
            self.logger.info(f"Model num labels: {model.num_labels}")
            self.logger.info(f"Label map: {model.config.id2label}")

            return model, tokenizer

        except Exception as e:
            self.logger.error(f"Failed to create BERT NER model: {str(e)}")
            raise

    async def release_model(self) -> bool:
        """
        Release BERT NER model resources.

        Returns:
            True if model was released successfully
        """
        try:
            released = await self.gpu_manager.release_model(ModelType.BERT_NER)
            if released:
                self.logger.info("BERT NER model resources released")
            return released
        except Exception as e:
            self.logger.error(f"Failed to release BERT NER model: {e}")
            return False

    def get_model_sync(self) -> Tuple['AutoModelForTokenClassification', 'AutoTokenizer']:
        """
        Synchronous wrapper for get_model - use with caution in async contexts.

        Returns:
            Tuple of (model, tokenizer)
        """
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.get_model())
        except RuntimeError:
            # No event loop running, create one
            return asyncio.run(self.get_model())

    def get_label_map(self) -> dict:
        """
        Get the label map from the model config.

        Returns:
            Dictionary mapping label IDs to label names
        """
        try:
            # Load model just to get config (lightweight operation)
            AutoModelForTokenClassification, _ = get_transforms_classes()
            model = AutoModelForTokenClassification.from_pretrained(BERT_NER_MODEL_NAME)
            label_map = model.config.id2label
            del model  # Free memory
            return label_map
        except Exception as e:
            self.logger.warning(f"Failed to get label map: {e}")
            # Return default label map based on known model
            return {
                0: "O",
                1: "B-EMAIL",
                2: "I-EMAIL",
                3: "B-PHONE",
                4: "I-PHONE",
                5: "B-NAME",
                6: "I-NAME",
                7: "B-ORG",
                8: "I-ORG",
                9: "B-DATE",
                10: "I-DATE",
                11: "B-DESIGNATION",
                12: "I-DESIGNATION",
                13: "B-SKILL",
                14: "I-SKILL",
                15: "B-LOCATION",
                16: "I-LOCATION",
                17: "B-EDUCATION_DEGREE",
                18: "I-EDUCATION_DEGREE",
                19: "B-EDUCATION_INSTITUTION",
                20: "I-EDUCATION_INSTITUTION",
            }


# Global instance for module-level access
_bert_ner_model_instance: Optional[BertNerModel] = None


def get_bert_ner_model() -> BertNerModel:
    """Get lazy-loaded BERT NER model instance."""
    global _bert_ner_model_instance
    if _bert_ner_model_instance is None:
        _bert_ner_model_instance = BertNerModel()
    return _bert_ner_model_instance


# For backward compatibility
bert_ner_model = None

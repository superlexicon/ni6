from .logger import get_logger

# Forgery detection models imported lazily to avoid torch dependency issues
class _AIGeneratedAnalyzerLazy:
    _class = None

    @classmethod
    def _load_class(cls):
        if cls._class is None:
            from .forgery_photoshop.forgery.ai_generated_analyzer import AIGeneratedAnalyzer as _Class
            cls._class = _Class
        return cls._class

    def __getattr__(self, name):
        return getattr(self._load_class(), name)

    def __call__(self, *args, **kwargs):
        return self._load_class()(*args, **kwargs)

AIGeneratedAnalyzer = _AIGeneratedAnalyzerLazy()

class _AdaptiveMethodLazy:
    _class = None

    @classmethod
    def _load_class(cls):
        if cls._class is None:
            from .forgery_photoshop.forgery.adaptive_method import AdaptiveMethod as _Class
            cls._class = _Class
        return cls._class

    def __getattr__(self, name):
        return getattr(self._load_class(), name)

    def __call__(self, *args, **kwargs):
        return self._load_class()(*args, **kwargs)

AdaptiveMethod = _AdaptiveMethodLazy()

class _DQMethodLazy:
    _class = None

    @classmethod
    def _load_class(cls):
        if cls._class is None:
            from .forgery_photoshop.forgery.dq_method import DQMethod as _Class
            cls._class = _Class
        return cls._class

    def __getattr__(self, name):
        return getattr(self._load_class(), name)

    def __call__(self, *args, **kwargs):
        return self._load_class()(*args, **kwargs)

DQMethod = _DQMethodLazy()

class _NoiseSnifferMethodLazy:
    _class = None

    @classmethod
    def _load_class(cls):
        if cls._class is None:
            from .forgery_photoshop.photoshopped.noisesniffer_method import NoiseSnifferMethod as _Class
            cls._class = _Class
        return cls._class

    def __getattr__(self, name):
        return getattr(self._load_class(), name)

    def __call__(self, *args, **kwargs):
        return self._load_class()(*args, **kwargs)

NoiseSnifferMethod = _NoiseSnifferMethodLazy()

class _PhotoShoppedAnalyzerLazy:
    _class = None

    @classmethod
    def _load_class(cls):
        if cls._class is None:
            from .forgery_photoshop.photoshopped.photoshopped_analyzer import PhotoShoppedAnalyzer as _Class
            cls._class = _Class
        return cls._class

    def __getattr__(self, name):
        return getattr(self._load_class(), name)

    def __call__(self, *args, **kwargs):
        return self._load_class()(*args, **kwargs)

PhotoShoppedAnalyzer = _PhotoShoppedAnalyzerLazy()

# Psccnet functions imported lazily
class _PsccnetFunctionsLazy:
    _functions = None

    @classmethod
    def _load_functions(cls):
        if cls._functions is None:
            from .psccnet_model import psccnet_get_model, psccnet_get_model_sync, psccnet_get_device, psccnet_get_device_sync
            cls._functions = {
                'psccnet_get_model': psccnet_get_model,
                'psccnet_get_model_sync': psccnet_get_model_sync,
                'psccnet_get_device': psccnet_get_device,
                'psccnet_get_device_sync': psccnet_get_device_sync,
            }
        return cls._functions

    def __getattr__(self, name):
        return self._load_functions()[name]

_psccnet_functions = _PsccnetFunctionsLazy()
psccnet_get_model = lambda *args, **kwargs: _psccnet_functions.psccnet_get_model(*args, **kwargs)
psccnet_get_model_sync = lambda *args, **kwargs: _psccnet_functions.psccnet_get_model_sync(*args, **kwargs)
psccnet_get_device = lambda *args, **kwargs: _psccnet_functions.psccnet_get_device(*args, **kwargs)
psccnet_get_device_sync = lambda *args, **kwargs: _psccnet_functions.psccnet_get_device_sync(*args, **kwargs)

# adaptive_get_model imported lazily
class _AdaptiveGetModelLazy:
    _func = None

    @classmethod
    def _load_func(cls):
        if cls._func is None:
            from .adaptive_model import adaptive_get_model as _func
            cls._func = _func
        return cls._func

    def __call__(self, *args, **kwargs):
        return self._load_func()(*args, **kwargs)

adaptive_get_model = _AdaptiveGetModelLazy()

# PsccnetMethod imported lazily to avoid circular import
class _PsccnetMethodLazy:
    _class = None

    @classmethod
    def _load_class(cls):
        if cls._class is None:
            from .forgery_photoshop.photoshopped.psccnet_method import PsccnetMethod as _PsccnetMethodClass
            cls._class = _PsccnetMethodClass
        return cls._class

    def __getattr__(self, name):
        return getattr(self._load_class(), name)

    def __call__(self, *args, **kwargs):
        return self._load_class()(*args, **kwargs)

PsccnetMethod = _PsccnetMethodLazy()
from .db.database import get_db_connection, Database
from .key.secp256k1 import KeyPair
from .key.scalsa20_crypto import Scalsa20Crypto
# DoctrModel imported lazily to avoid dependency issues
class _DoctrModelLazy:
    _class = None

    @classmethod
    def _load_class(cls):
        if cls._class is None:
            from .doctr_model import DoctrModel as _DoctrModelClass
            cls._class = _DoctrModelClass
        return cls._class

    def __getattr__(self, name):
        return getattr(self._load_class(), name)

    def __call__(self, *args, **kwargs):
        return self._load_class()(*args, **kwargs)

DoctrModel = _DoctrModelLazy()

# BertNerModel imported lazily to avoid dependency issues
class _BertNerModelLazy:
    _class = None

    @classmethod
    def _load_class(cls):
        if cls._class is None:
            from .bert_ner_model import BertNerModel as _BertNerModelClass
            cls._class = _BertNerModelClass
        return cls._class

    def __getattr__(self, name):
        return getattr(self._load_class(), name)

    def __call__(self, *args, **kwargs):
        return self._load_class()(*args, **kwargs)

BertNerModel = _BertNerModelLazy()

# get logger
logger = get_logger()

key_pair = KeyPair()
scalsa20_crypto = Scalsa20Crypto()

# Ollama LLM features have been removed
embeddings = None
llm = None
logger.info("Ollama LLM features not available - integration removed")

# Doctr model will be lazy-loaded to avoid async initialization issues
_doctr_model_instance = None

def get_doctr_model():
    """Get lazy-loaded Doctr model instance."""
    global _doctr_model_instance
    if _doctr_model_instance is None:
        _doctr_model_instance = DoctrModel().get_model()
    return _doctr_model_instance

# Bert NER model will be lazy-loaded to avoid async initialization issues
_bert_ner_model_instance = None

def get_bert_ner_model():
    """Get lazy-loaded Bert NER model instance."""
    global _bert_ner_model_instance
    if _bert_ner_model_instance is None:
        _bert_ner_model_instance = BertNerModel().get_model()
    return _bert_ner_model_instance

# For backward compatibility
doctr_model = None

__all__ = [
    "logger",
    "get_doctr_model",
    "doctr_model",
    "adaptive_get_model",
    "psccnet_get_model",
    "psccnet_get_model_sync",
    "psccnet_get_device",
    "psccnet_get_device_sync",
    "AIGeneratedAnalyzer",
    "AdaptiveMethod",
    "DQMethod",
    "NoiseSnifferMethod",
    "PhotoShoppedAnalyzer",
    "PsccnetMethod",
    "get_db_connection",
    "key_pair",
    "scalsa20_crypto",
    "doctr_model",
    "BertNerModel",
    "get_bert_ner_model",
    "Database"
]

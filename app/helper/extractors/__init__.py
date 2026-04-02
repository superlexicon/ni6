from .passport_extractor import PassportExtractor
from .selfie_otp_extractor import SelfieOTPExtractor
from .tax_statement_extractor import TaxStatementExtractor
from .bert_ner_resume_extractor import BertNerResumeExtractor

# GLiNER extractors imported lazily to avoid segfault during startup
# (gliner2 → gliner → transformers → torch.jit.script segfaults with PyTorch 2.6.0)
class _GLiNERIDCardExtractorLazy:
    _class = None

    @classmethod
    def _load_class(cls):
        if cls._class is None:
            from .gliner_id_card_extractor import GLiNERIDCardExtractor as _Class
            cls._class = _Class
        return cls._class

    def __getattr__(self, name):
        return getattr(self._load_class(), name)

    def __call__(self, *args, **kwargs):
        return self._load_class()(*args, **kwargs)

GLiNERIDCardExtractor = _GLiNERIDCardExtractorLazy()

class _GLiNERBankStatementExtractorLazy:
    _class = None

    @classmethod
    def _load_class(cls):
        if cls._class is None:
            from .gliner_bank_statement_extractor import GLiNERBankStatementExtractor as _Class
            cls._class = _Class
        return cls._class

    def __getattr__(self, name):
        return getattr(self._load_class(), name)

    def __call__(self, *args, **kwargs):
        return self._load_class()(*args, **kwargs)

GLiNERBankStatementExtractor = _GLiNERBankStatementExtractorLazy()

__all__ = [
    "PassportExtractor",
    "SelfieOTPExtractor",
    "TaxStatementExtractor",
    "BertNerResumeExtractor",
    "GLiNERIDCardExtractor",
    "GLiNERBankStatementExtractor"
]

"""
Extractors Package

Provides direct extraction services for bank statements, passports,
and generic document classification/PII extraction.
"""

from app.services.extractors.qwen_bank_statement_extractor import (
    QwenBankStatementExtractor,
    get_qwen_bank_statement_extractor
)
from app.services.extractors.qwen_passport_extractor import (
    QwenPassportExtractor,
    get_qwen_passport_extractor
)
from app.services.extractors.qwen_generic_document_extractor import (
    QwenGenericDocumentExtractor,
    get_qwen_generic_document_extractor
)

__all__ = [
    "QwenBankStatementExtractor",
    "get_qwen_bank_statement_extractor",
    "QwenPassportExtractor",
    "get_qwen_passport_extractor",
    "QwenGenericDocumentExtractor",
    "get_qwen_generic_document_extractor"
]

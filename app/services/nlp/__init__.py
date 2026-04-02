"""
NLP services for enhanced text analysis.

Provides:
- VADER sentiment analysis (fast, rule-based)
- spaCy named entity extraction
- FinBERT financial crime classification (lazy-loaded)
- Article content extraction (newspaper3k)
"""

from app.services.nlp.nlp_service import nlp_service

__all__ = ["nlp_service"]

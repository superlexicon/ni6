"""
NLP Service for enhanced text analysis.

Provides:
- VADER sentiment analysis (fast, rule-based)
- spaCy named entity extraction
- FinBERT financial crime classification (lazy-loaded)
"""

# Set NLTK data path BEFORE importing nltk modules
import nltk
import os

# Use relative path from working directory (/app)
# Dockerfile copies to /app/app/nltk_data/, working dir is /app
nltk_data_dir = os.path.abspath('app/nltk_data')
nltk.data.path.insert(0, nltk_data_dir)  # Insert at beginning to search first

from typing import Dict, Any, List, Optional
from app.core.logger import get_logger


class NLPService:
    """Centralized NLP service for text analysis."""

    def __init__(self):
        self.logger = get_logger()
        self.vader_analyzer = None
        self.finbert_pipeline = None
        self.finbert_tokenizer = None
        self.nlp = None
        self._initialized = False

    async def initialize(self):
        """Initialize NLP models (lazy load FinBERT)."""
        if self._initialized:
            return

        try:
            # VADER (instant, rule-based)
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            self.vader_analyzer = SentimentIntensityAnalyzer()
            self.logger.info("VADER sentiment analyzer initialized")

            # spaCy (fast NER) - load bundled model from repository
            import spacy
            from pathlib import Path

            # Try bundled model first, fallback to system-installed
            bundled_path = Path(__file__).parent.parent.parent / "spacy_models" / "en_core_web_sm"
            if bundled_path.exists():
                self.nlp = spacy.load(str(bundled_path))
                self.logger.info(f"spaCy NER loaded from bundled model: {bundled_path}")
            else:
                self.nlp = spacy.load("en_core_web_sm")
                self.logger.info("spaCy NER loaded from system installation")

            self._initialized = True
            self.logger.info("NLP service initialized (FinBERT will load on first use)")
        except Exception as e:
            self.logger.error(f"NLP service initialization failed: {e}")
            raise

    def analyze_sentiment_vader(self, text: str) -> Dict[str, float]:
        """Fast sentiment analysis using VADER.

        Args:
            text: Text to analyze

        Returns:
            {'neg': 0.0-1.0, 'neu': 0.0-1.0, 'pos': 0.0-1.0, 'compound': -1.0 to 1.0}

        Interpretation:
        - compound >= 0.05: Positive
        - compound between -0.05 and 0.05: Neutral
        - compound <= -0.05: Negative
        """
        if not self.vader_analyzer:
            return {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}

        scores = self.vader_analyzer.polarity_scores(text)
        return scores

    def extract_entities_spacy(self, text: str) -> Dict[str, List]:
        """Extract named entities using spaCy.

        Args:
            text: Text to extract entities from

        Returns:
            {
                'persons': ['John Smith', 'Jane Doe'],
                'organizations': ['SEC', 'FBI'],
                'dates': ['2024-01-15'],
                'locations': ['New York'],
                'money': ['$1.2 million'],
                'gpe': ['USA']  # Geopolitical entities
            }
        """
        if not self.nlp:
            return {
                'persons': [], 'organizations': [], 'dates': [],
                'locations': [], 'money': [], 'gpe': []
            }

        doc = self.nlp(text)

        entities = {
            'persons': [],
            'organizations': [],
            'dates': [],
            'locations': [],
            'money': [],
            'gpe': []
        }

        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                entities['persons'].append(ent.text)
            elif ent.label_ in ['ORG', 'NORP']:
                entities['organizations'].append(ent.text)
            elif ent.label_ == 'DATE':
                entities['dates'].append(ent.text)
            elif ent.label_ == 'GPE':
                entities['gpe'].append(ent.text)
            elif ent.label_ == 'LOC':
                entities['locations'].append(ent.text)
            elif ent.label_ == 'MONEY':
                entities['money'].append(ent.text)

        return entities

    async def classify_financial_crime_finbert(self, text: str) -> Dict[str, Any]:
        """Classify financial crime using FinBERT.

        Args:
            text: Article text to classify

        Returns:
            {
                'category': 'fraud' | 'corruption' | 'money_laundering' | 'none',
                'confidence': 0.0-1.0,
                'event_type': 'allegation' | 'investigation' | 'conviction' | 'cleared',
                'sentiment': 'positive' | 'neutral' | 'negative'
            }
        """
        # Lazy load FinBERT on first use
        if not self.finbert_pipeline:
            await self._load_finbert()

        if not self.finbert_pipeline:
            return {'category': 'none', 'confidence': 0.0, 'event_type': 'unknown', 'sentiment': 'neutral'}

        try:
            # Classify using FinBERT
            result = self.finbert_pipeline(text[:512])  # Truncate to 512 tokens

            # Determine category from labels
            label = result[0]['label']
            score = result[0]['score']

            # Map to our categories
            category = self._map_finbert_label(label)
            event_type = self._extract_event_type(text)

            return {
                'category': category,
                'confidence': score,
                'event_type': event_type,
                'sentiment': self._get_finbert_sentiment(label)
            }
        except Exception as e:
            self.logger.error(f"FinBERT classification failed: {e}")
            return {'category': 'none', 'confidence': 0.0, 'event_type': 'unknown', 'sentiment': 'neutral'}

    async def _load_finbert(self):
        """Load FinBERT model (lazy, on first use)."""
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
            from pathlib import Path

            # Use local weights if available (downloaded via scripts/download_model_weights.py)
            local_path = Path(__file__).parent.parent.parent / "finbert_weights"
            model_name = str(local_path) if local_path.exists() else "ProsusAI/finbert"

            self.logger.info(f"Loading FinBERT model from: {model_name}")
            self.finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.finbert_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=self.finbert_tokenizer)
            self.logger.info("FinBERT model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load FinBERT: {e}")
            self.finbert_pipeline = None

    def _map_finbert_label(self, label: str) -> str:
        """Map FinBERT label to our categories."""
        label_map = {
            'positive': 'none',
            'neutral': 'none',
            'negative': 'fraud'  # Default negative to fraud
        }
        return label_map.get(label.lower(), 'none')

    def _extract_event_type(self, text: str) -> str:
        """Extract event type from text using keyword patterns."""
        text_lower = text.lower()

        # Check for CIVIL keywords FIRST (priority over criminal)
        # Civil lawsuits should NOT be classified as criminal convictions
        civil_keywords = ['civil lawsuit', 'civil action', 'civil litigation', 'civil suit', 'civil case']
        if any(kw in text_lower for kw in civil_keywords):
            return 'civil_lawsuit'

        # Check for conviction keywords
        conviction_keywords = ['convicted', 'pleaded guilty', 'found guilty', 'sentenced', 'criminal conviction']
        if any(kw in text_lower for kw in conviction_keywords):
            return 'conviction'

        # Check for cleared keywords
        cleared_keywords = ['cleared', 'acquitted', 'exonerated', 'charges dropped', 'not guilty']
        if any(kw in text_lower for kw in cleared_keywords):
            return 'cleared'

        # Check for investigation keywords
        investigation_keywords = ['investigation', 'probe', 'investigating', 'under investigation']
        if any(kw in text_lower for kw in investigation_keywords):
            return 'investigation'

        # Default to allegation
        allegation_keywords = ['accused', 'alleged', 'allegations', 'charged with', 'sued']
        if any(kw in text_lower for kw in allegation_keywords):
            return 'allegation'

        return 'unknown'

    def _get_finbert_sentiment(self, label: str) -> str:
        """Get sentiment from FinBERT label."""
        if label.lower() == 'negative':
            return 'negative'
        elif label.lower() == 'positive':
            return 'positive'
        else:
            return 'neutral'


# Global instance
nlp_service = NLPService()

"""
NLP Service for enhanced text analysis.

Provides:
- Qwen/LLM sentiment analysis (replaces VADER)
- spaCy named entity extraction
- FinBERT financial crime classification (lazy-loaded)
"""

from typing import Dict, Any, List, Optional
from app.core.logger import get_logger


class NLPService:
    """Centralized NLP service for text analysis."""

    def __init__(self):
        self.logger = get_logger()
        self.llm_service = None
        self.finbert_pipeline = None
        self.finbert_tokenizer = None
        self.nlp = None
        self._initialized = False

    async def initialize(self):
        """Initialize NLP models (lazy load FinBERT)."""
        if self._initialized:
            return

        try:
            # Qwen/LLM for sentiment analysis
            from app.services.llm_service import LLMService
            self.llm_service = LLMService()
            self.logger.info("LLM service initialized for sentiment analysis")

            # spaCy (fast NER) - load bundled model from repository
            import spacy
            from pathlib import Path

            # Try bundled model first, fallback to system-installed
            bundled_path = Path(__file__).parent.parent.parent / "models" / "spacy" / "en_core_web_sm"
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

    async def analyze_sentiment_llm(self, text: str) -> Dict[str, float]:
        """Sentiment analysis using Qwen/LLM (replaces VADER).

        Args:
            text: Text to analyze

        Returns:
            {'neg': 0.0-1.0, 'neu': 0.0-1.0, 'pos': 0.0-1.0, 'compound': -1.0 to 1.0}

        Interpretation:
            - compound >= 0.05: Positive
            - compound between -0.05 and 0.05: Neutral
            - compound <= -0.05: Negative
        """
        if not self.llm_service:
            return {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}

        try:
            import json
            import httpx

            # Use text_model for faster sentiment analysis
            from app.config.llm_config import llm_settings

            url = f"{llm_settings.api_url}/chat/completions"

            headers = {"Content-Type": "application/json"}
            if llm_settings.api_key:
                headers["Authorization"] = f"Bearer {llm_settings.api_key}"

            system_prompt = """You are a sentiment analysis expert. Analyze the sentiment of the given text and respond with a JSON object containing:
- "sentiment": "positive", "neutral", or "negative"
- "compound": a float score from -1.0 (very negative) to 1.0 (very positive)
- "confidence": confidence score from 0.0 to 1.0

For compound score:
- -0.8 to -1.0: Extremely negative
- -0.5 to -0.8: Very negative
- -0.2 to -0.5: Somewhat negative
- -0.05 to -0.2: Slightly negative
- -0.05 to 0.05: Neutral
- 0.05 to 0.2: Slightly positive
- 0.2 to 0.5: Somewhat positive
- 0.5 to 0.8: Very positive
- 0.8 to 1.0: Extremely positive

Respond ONLY with valid JSON, no other text."""

            user_prompt = f"Analyze the sentiment of this text:\n\n{text}"

            payload = {
                "model": llm_settings.text_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 100,
                "format": "json"  # For Ollama JSON mode
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                result = response.json()

                # Parse the LLM response
                content = result['choices'][0]['message']['content']
                sentiment_data = json.loads(content)

                # Convert to VADER-compatible format
                sentiment = sentiment_data.get('sentiment', 'neutral')
                compound = sentiment_data.get('compound', 0.0)

                # Map sentiment to neg/neu/pos format
                if sentiment == 'positive':
                    pos = min(0.5 + abs(compound) / 2, 1.0)
                    neg = 0.0
                    neu = 1.0 - pos
                elif sentiment == 'negative':
                    neg = min(0.5 + abs(compound) / 2, 1.0)
                    pos = 0.0
                    neu = 1.0 - neg
                else:
                    neu = 1.0
                    pos = 0.0
                    neg = 0.0

                return {
                    'neg': neg,
                    'neu': neu,
                    'pos': pos,
                    'compound': compound
                }

        except Exception as e:
            self.logger.warning(f"LLM sentiment analysis failed: {e}. Returning neutral sentiment.")
            return {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}

    async def analyze_sentiment_vader(self, text: str) -> Dict[str, float]:
        """VADER compatibility method - redirects to LLM analysis.

        This method is kept for backward compatibility. Use analyze_sentiment_llm() directly.
        """
        return await self.analyze_sentiment_llm(text)

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
            local_path = Path(__file__).parent.parent.parent / "models" / "finbert"
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

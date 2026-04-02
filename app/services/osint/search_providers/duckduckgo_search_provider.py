"""
DuckDuckGo Search Provider for OSINT Background Search.

Enhanced version of the original DuckDuckGo search with:
- Caching layer
- Negative news keyword detection
- Query variations for comprehensive search
- OSINT-specific result structure
"""

import asyncio
import time
import hashlib
import re
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from ddgs import DDGS
from app.core.logger import get_logger
from app.config.osint_config import osint_settings


class DuckDuckGoSearchProvider:
    """
    DuckDuckGo search provider for OSINT background screening.

    Features:
    - Rate limiting (respectful scraping)
    - Result caching (24 hours)
    - Negative news keyword detection
    - Query variations for comprehensive search
    """

    # Negative news keywords for risk assessment
    NEGATIVE_KEYWORDS = {
        'critical': [
            'fraud', 'scam', 'money laundering', 'embezzlement', 'bribery',
            'corruption', 'terrorist', 'terrorism', 'sanctioned', 'convicted',
            'arrested', 'indicted', 'charged', 'lawsuit', 'criminal', 'ponzi',
            'pyramid scheme', 'identity theft', 'cybercrime', 'hacking'
        ],
        'high': [
            'investigation', 'probe', 'audit', 'complaint', 'lawsuit',
            'allegation', 'accused', 'sued', 'settled', 'controversy',
            'misconduct', 'violation', 'penalty', 'fine', 'regulator'
        ],
        'medium': [
            'dispute', 'conflict', 'issue', 'concern', 'problem',
            'failure', 'bankrupt', 'liquidation', 'delisted', 'warning'
        ]
    }

    def __init__(self):
        self.logger = get_logger()
        self.last_search_time = 0
        self.min_search_interval = osint_settings.min_delay_seconds
        self.cache = {}  # Simple in-memory cache (production: use Redis)

    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for query."""
        return hashlib.md5(query.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get results from cache if available and not expired."""
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            cache_time = cached.get('timestamp')
            if cache_time and datetime.now() - cache_time < timedelta(hours=osint_settings.cache_results_hours):
                self.logger.info(f"Cache hit for query: {cache_key[:8]}...")
                return cached.get('results')
        return None

    def _save_to_cache(self, cache_key: str, results: Dict[str, Any]):
        """Save results to cache."""
        self.cache[cache_key] = {
            'timestamp': datetime.now(),
            'results': results
        }

    async def search(
        self,
        full_name: str,
        date_of_birth: Optional[str] = None,
        country: Optional[str] = None,
        max_results: int = 10
    ) -> Dict[str, Any]:
        """
        Perform comprehensive web search using DuckDuckGo.

        Args:
            full_name: Person's full name
            date_of_birth: Date of birth (YYYY-MM-DD)
            country: Country name or code
            max_results: Maximum number of results per query

        Returns:
            {
                "queries_performed": int,
                "results_count": int,
                "negative_news_count": int,
                "sources": list,
                "results": list
            }
        """
        self.logger.debug(f"Starting DuckDuckGo search for: {full_name}")

        # Build query variations
        queries = self._build_search_queries(full_name, date_of_birth, country)

        all_results = []
        negative_news_count = 0
        sources = []

        for query in queries:
            # Check cache first
            cache_key = self._get_cache_key(query)
            cached_results = self._get_from_cache(cache_key)
            if cached_results:
                all_results.extend(cached_results.get('results', []))
                negative_news_count += cached_results.get('negative_news_count', 0)
                sources.extend(cached_results.get('sources', []))
                continue

            # Perform search with retry logic
            results = await self._search_with_retry(query, max_results)
            if results:
                # Analyze for negative news
                analyzed = await self._analyze_results(results, query, full_name, country)
                all_results.extend(results)
                negative_news_count += analyzed['negative_news_count']
                sources.extend(analyzed['sources'])

                # Cache results
                self._save_to_cache(cache_key, analyzed)

        # Remove duplicates
        seen_urls = set()
        unique_results = []
        unique_sources = []
        for result in all_results:
            url = result.get('href', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)
                if url not in unique_sources:
                    unique_sources.append(url)

        return {
            "queries_performed": len(queries),
            "results_count": len(unique_results),
            "negative_news_count": negative_news_count,
            "sources": unique_sources,
            "results": unique_results
        }

    def _build_search_queries(
        self,
        full_name: str,
        date_of_birth: Optional[str],
        country: Optional[str]
    ) -> List[str]:
        """
        Build query variations for comprehensive search.

        Query variations:
        1. Full name only
        2. Full name + country
        3. Full name + "fraud" or "scam"
        4. Full name + "arrested" or "charged"
        5. Last name + first initial (for name variations)
        """
        queries = []

        # Basic query
        queries.append(f'"{full_name}"')

        # Name + country
        if country:
            queries.append(f'"{full_name}" {country}')

        # Negative news queries
        queries.append(f'"{full_name}" fraud')
        queries.append(f'"{full_name}" scam')
        queries.append(f'"{full_name}" arrested')
        queries.append(f'"{full_name}" charged')

        # Name variation (Last, First)
        name_parts = full_name.split()
        if len(name_parts) >= 2:
            last_name = name_parts[-1]
            first_name = name_parts[0]
            queries.append(f'"{last_name}, {first_name}"')

            # Last name + fraud keywords
            queries.append(f'"{last_name}" fraud')
            queries.append(f'"{last_name}" scandal')

        # Date of birth + name (if available)
        if date_of_birth:
            # Handle both str and datetime.date types
            from datetime import date as date_type
            if isinstance(date_of_birth, date_type):
                year = str(date_of_birth.year)
            elif isinstance(date_of_birth, str):
                year = date_of_birth[:4] if len(date_of_birth) >= 4 else ''
            else:
                year = ''

            if year:
                queries.append(f'"{full_name}" {year}')

        return queries

    async def _search_with_retry(
        self,
        query: str,
        max_results: int,
        retries: int = 3
    ) -> Optional[List[Dict]]:
        """Perform search with exponential backoff retry logic."""
        for attempt in range(retries):
            try:
                # Enforce rate limiting
                elapsed = time.time() - self.last_search_time
                if elapsed < self.min_search_interval:
                    await asyncio.sleep(self.min_search_interval - elapsed)

                with DDGS() as ddgs:
                    results = list(
                        ddgs.text(
                            query=query,
                            max_results=max_results,
                        )
                    )
                    self.last_search_time = time.time()
                    self.logger.info(f"Search '{query}' returned {len(results)} results")
                    return results

            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for '{query}': {str(e)}")
                if attempt < retries - 1:
                    await asyncio.sleep(2**attempt)  # Exponential backoff
                else:
                    self.logger.error(f"All search attempts failed for query: {query}")
                    return None

    async def _analyze_results(self, results: List[Dict], query: str, full_name: str = None, country: str = None) -> Dict[str, Any]:
        """
        Analyze search results for negative news using NLP-enhanced detection.

        PHASE 1: Fast VADER sentiment + spaCy NER on all results
        PHASE 2: Deep FinBERT analysis on top negative results (if NLP enabled)

        Args:
            results: List of search results
            query: The search query used
            full_name: The person's full name (for relevance filtering)
            country: The person's country (for relevance filtering)

        Returns:
            {
                "results": list,
                "negative_news_count": int,
                "sources": list
            }
        """
        from app.services.nlp.nlp_service import nlp_service
        from app.services.nlp.article_extractor import article_extractor

        negative_count = 0
        sources = []
        analyzed_results = []
        needs_deep_analysis = []

        # Check if NLP enhanced analysis is enabled
        nlp_enabled = getattr(osint_settings, 'enable_nlp_enhanced_analysis', False)

        for result in results:
            title = result.get('title', '')
            body = result.get('body', '')
            url = result.get('href', '')
            combined_text = f"{title} {body}"

            # PHASE 1: Fast NLP analysis (if enabled)
            sentiment = None
            entities = None
            is_negative = False

            if nlp_enabled and nlp_service._initialized:
                # VADER sentiment analysis
                sentiment = nlp_service.analyze_sentiment_vader(combined_text)

                # Extract entities with spaCy
                entities = nlp_service.extract_entities_spacy(combined_text)

                # Check if negative sentiment AND relevant to person
                is_negative = sentiment['compound'] < -0.3  # VADER negative threshold

            # Fallback to keyword-based detection
            if not is_negative:
                risk_level = self._check_negative_keywords(combined_text)
                is_negative = risk_level is not None

            # Check relevance to person
            is_relevant = not full_name or self._is_relevant_negative_news(result, full_name)

            # Only set NLP keys if they have actual values (avoid setting None)
            if sentiment is not None:
                result['sentiment'] = sentiment
            if entities is not None:
                result['entities'] = entities

            if is_negative and is_relevant:
                # ADDITIONAL CHECKS: Verify this is actually about the right person
                # Fuzzy name matching to ensure it's the same person
                if full_name:
                    name_match_score = self._fuzzy_match_name(title, body, full_name)
                    if name_match_score < 0.5:  # Threshold for same person
                        # Not same person, skip this result
                        result['risk_level'] = None
                        analyzed_results.append(result)
                        continue

                # Country matching to ensure it's from the right country
                if country:
                    combined_text = f"{title} {body}"
                    if not self._match_country(combined_text, country):
                        # Wrong country, skip this result
                        result['risk_level'] = None
                        analyzed_results.append(result)
                        continue

                # All checks passed - this is relevant negative news
                negative_count += 1
                result['risk_level'] = 'negative'
                if nlp_enabled:
                    result['needs_deep_analysis'] = True
                    needs_deep_analysis.append(result)
            elif sentiment and sentiment['compound'] < 0:
                # Slightly negative but below threshold
                result['risk_level'] = 'low_risk'
            else:
                result['risk_level'] = None

            # Extract source domain
            if url:
                source = self._extract_domain(url)
                if source and source not in sources:
                    sources.append(source)

            analyzed_results.append(result)

        # PHASE 2: Deep analysis for top negative articles (async, if NLP enabled)
        if nlp_enabled and nlp_service._initialized:
            # Limit to configured number for performance
            deep_analysis_limit = getattr(osint_settings, 'nlp_deep_analysis_limit', 5)
            enable_fallback = getattr(osint_settings, 'enable_article_extraction_fallback', True)

            for result in needs_deep_analysis[:deep_analysis_limit]:
                url = result.get('href')
                if url:
                    article = await article_extractor.extract_article(url)
                    if article:
                        # Successfully extracted article - use FinBERT for classification
                        classification = await nlp_service.classify_financial_crime_finbert(article['text'])
                        result['deep_analysis'] = {
                            'article': article,
                            'classification': classification
                        }
                    elif enable_fallback:
                        # Fallback: use search result snippet when article extraction fails
                        # This handles 403 errors, paywalls, and other extraction failures
                        title = result.get('title', '')
                        body = result.get('body', '')

                        # Combine title and body for analysis
                        fallback_text = f"{title}. {body}" if title and body else (title or body)

                        if fallback_text:
                            self.logger.info(f"Article extraction failed for {url}, using search snippet fallback ({len(fallback_text)} chars)")

                            # Use FinBERT on the snippet
                            classification = await nlp_service.classify_financial_crime_finbert(fallback_text)

                            result['deep_analysis'] = {
                                'article': {
                                    'title': title,
                                    'text': fallback_text,
                                    'url': url,
                                    'is_fallback': True  # Mark as fallback data
                                },
                                'classification': classification
                            }
                        else:
                            self.logger.warning(f"No fallback text available for {url}")

            # After deep analysis, apply threshold to adjust negative_count
            # Only count results that meet the threshold:
            # - Conviction with confidence >= 0.7
            # - Investigation with confidence >= 0.5
            adjusted_count = 0
            for result in analyzed_results:
                if result.get('risk_level') == 'negative':
                    deep_analysis = result.get('deep_analysis')
                    if deep_analysis:
                        classification = deep_analysis.get('classification') or {}
                        event_type = classification.get('event_type', 'unknown')
                        confidence = classification.get('confidence', 0.0)

                        # Check if meets threshold
                        meets_threshold = False
                        if event_type == 'conviction' and confidence >= 0.7:
                            meets_threshold = True
                        elif event_type == 'investigation' and confidence >= 0.5:
                            meets_threshold = True
                        elif event_type == 'allegation' and confidence >= 0.4:
                            meets_threshold = True

                        if not meets_threshold:
                            # Doesn't meet threshold - remove negative status
                            result['risk_level'] = None
                            result['excluded_reason'] = f'Below threshold (event_type={event_type}, confidence={confidence:.2f})'
                        else:
                            adjusted_count += 1
                    else:
                        # No deep analysis performed - count it anyway (old behavior for compatibility)
                        adjusted_count += 1

            # Update negative_count to only include threshold-passing results
            negative_count = adjusted_count

        return {
            'results': analyzed_results,
            'negative_news_count': negative_count,
            'sources': sources
        }

    def _check_negative_keywords(self, text: str) -> Optional[str]:
        """
        Check text for negative news keywords.

        Returns:
            'critical', 'high', 'medium', or None if no negative keywords found
        """
        text_lower = text.lower()

        # Check critical keywords first
        for keyword in self.NEGATIVE_KEYWORDS['critical']:
            if keyword in text_lower:
                return 'critical'

        # Then high
        for keyword in self.NEGATIVE_KEYWORDS['high']:
            if keyword in text_lower:
                return 'high'

        # Finally medium
        for keyword in self.NEGATIVE_KEYWORDS['medium']:
            if keyword in text_lower:
                return 'medium'

        return None

    def _is_relevant_negative_news(self, result: dict, search_name: str) -> bool:
        """
        Check if negative news is actually about the searched person.

        Reduces false positives by requiring the searched name to appear
        prominently in the result (in title or first 200 characters of body).

        Args:
            result: Search result with title and body
            search_name: Person's name being searched for

        Returns:
            True if this appears to be relevant negative news about the person
        """
        title = result.get('title', '').lower()
        body = result.get('body', '').lower()
        search_name_lower = search_name.lower()

        # Name must appear in title (strongest signal)
        if search_name_lower in title:
            return True

        # Or name must appear in first 200 characters of body
        # (indicates the article is actually about this person)
        body_start = body[:200]
        if search_name_lower in body_start:
            return True

        # Otherwise, it's probably not about this person
        return False

    def _fuzzy_match_name(self, title: str, body: str, search_name: str) -> float:
        """
        Perform fuzzy name matching to determine if result is about the same person.

        Uses token-based Jaccard similarity + coverage algorithm (same as sanctions checker).

        Args:
            title: Result title
            body: Result body text
            search_name: Person's name being searched for

        Returns:
            Score 0.0-1.0 where >=0.5 indicates likely match
        """
        import unicodedata

        def remove_accents(text: str) -> str:
            """Remove diacritics/accents from text."""
            normalized = unicodedata.normalize('NFD', text)
            return ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')

        def tokenize_name(name: str) -> set:
            """Extract name tokens."""
            if not name:
                return set()
            name = remove_accents(name)
            # Remove common titles
            import re
            name = re.sub(r'\b(H\.E\.|Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)\s?', '', name)
            name = name.replace(',', ' ')
            name = re.sub(r'\s+', ' ', name).strip().lower()
            return set(name.split())

        def calculate_match_score(search_tokens: set, entry_tokens: set) -> float:
            """Calculate match score between token sets."""
            if not search_tokens or not entry_tokens:
                return 0.0
            intersection = search_tokens & entry_tokens
            union = search_tokens | entry_tokens
            jaccard = len(intersection) / len(union) if union else 0.0
            coverage = len(intersection) / len(search_tokens)
            return (coverage * 0.7) + (jaccard * 0.3)

        # Combine title and body for analysis
        combined_text = f"{title} {body}".lower()
        search_tokens = tokenize_name(search_name)
        result_tokens = tokenize_name(combined_text)

        return calculate_match_score(search_tokens, result_tokens)

    def _match_country(self, text: str, search_country: str) -> bool:
        """
        Check if the text mentions the searched country.

        Handles country code variations (SG, Singapore, SGP).

        Args:
            text: Text to search in
            search_country: Country code or name to match

        Returns:
            True if country is found in text, False otherwise
        """
        if not search_country or not text:
            return True  # If no country specified, don't filter out

        text_lower = text.lower()
        search_lower = search_country.lower().strip()

        # Direct match
        if search_lower in text_lower:
            return True

        # Common country code/name variations
        country_mappings = {
            'sg': ['singapore', 'sgp', 'sg'],
            'us': ['united states', 'usa', 'us', 'america'],
            'uk': ['united kingdom', 'uk', 'gb', 'great britain', 'england'],
            'my': ['malaysia', 'my'],
            'th': ['thailand', 'th'],
            'id': ['indonesia', 'id'],
            'ph': ['philippines', 'ph'],
            'in': ['india', 'in'],
        }

        # Check if search country is in mappings
        for code, variations in country_mappings.items():
            if search_lower in variations:
                # Check if any variation appears in text
                for variation in variations:
                    if variation in text_lower:
                        return True

        return False

    def _extract_domain(self, url: str) -> Optional[str]:
        """Extract domain name from URL."""
        try:
            # Remove protocol and path
            domain = re.sub(r'^https?://', '', url)
            domain = re.sub(r'/.*$', '', domain)
            # Remove www.
            domain = re.sub(r'^www\.', '', domain)
            return domain
        except Exception:
            return None


# Global instance
duckduckgo_provider = DuckDuckGoSearchProvider()

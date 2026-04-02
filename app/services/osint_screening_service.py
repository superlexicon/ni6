"""
OSINT Screening Service - Free background search using web scraping.

Performs comprehensive OSINT screening using:
- Web search (DuckDuckGo, Google scraping)
- Social media (LinkedIn, Twitter/X scraping)
- Public records (country-specific scrapers)
- Sanctions lists (OFAC, EU, UN free downloads)

Alternative to World-Check for identity verification and risk assessment.
"""

import asyncio
import uuid
from typing import Dict, Any, Optional, List, Set
from datetime import datetime
from app.core.logger import get_logger
from app.config.osint_config import osint_settings
from app.services.osint.search_providers.duckduckgo_search_provider import duckduckgo_provider
from app.services.osint.search_providers.google_scraper import google_scraper
from app.services.osint.search_providers.public_image_search_provider import public_image_search_provider
from app.services.osint.sanctions import sanctions_list_checker
from app.services.osint.pep import pep_checker
from app.services.osint.risk_scorer import risk_scorer
from app.repositories import face_biometrics_repository
from app.services.nlp.nlp_service import nlp_service
import httpx
import trafilatura


class OSINTScreeningService:
    """
    Free OSINT background search service using web scraping.

    Target Markets: India, Malaysia, Thailand, Indonesia, Philippines
    """

    def __init__(self):
        self.logger = get_logger()
        self.settings = osint_settings

        # Search providers
        self.duckduckgo_provider = duckduckgo_provider
        self.google_scraper = google_scraper
        self.risk_scorer = risk_scorer
        self.public_image_search_provider = public_image_search_provider

        # NLP service
        self.nlp_service = nlp_service

    async def screen_individual(
        self,
        full_name: str,
        date_of_birth: Optional[str] = None,
        country: Optional[str] = None,
        gender: Optional[str] = None,
        address: Optional[str] = None,
        user_identity_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive OSINT screening using free sources.

        Args:
            full_name: Full name from passport/document
            date_of_birth: Date of birth (YYYY-MM-DD format)
            country: ISO country code or country name
            gender: M/F/Other
            address: Address from bank statement or document
            user_identity_id: User identity ID for face-verified search (optional)

        Returns:
            {
                "is_match": bool,  # True if risk_score >= threshold
                "overall_risk_score": float,  # 0-100
                "risk_category": str,  # LOW, MEDIUM, HIGH, CRITICAL
                "match_details": list,
                "screening_id": str,
                "raw_response": dict,
                "error": str or None
            }
        """
        screening_id = f"osint_{uuid.uuid4().hex[:12]}"
        start_time = datetime.now()

        self.logger.debug(f"Starting OSINT screening {screening_id} for: {full_name}")

        # Initialize results structure
        results = {
            "web_search": None,
            "public_records": None,
            "sanctions": None,
            "pep": None
        }

        errors = []

        # PHASE 1: Web Search with Face Verification (Critical)
        # Skip web search entirely if max_images is 0
        if self.settings.enable_web_search and self.settings.face_verified_max_images > 0:
            try:
                # Only face-verified search (no regular web search fallback)
                if user_identity_id:
                    self.logger.info("Using face-verified negative news search")
                    results["web_search"] = await self._get_face_verified_negative_news(
                        full_name=full_name,
                        country=country,
                        user_identity_id=user_identity_id
                    )
                else:
                    self.logger.info("No user_identity_id provided, skipping web search")
            except Exception as e:
                self.logger.error(f"Web search failed: {type(e).__name__}")
                errors.append(f"web_search: {type(e).__name__}")
                results["web_search"] = {}  # Initialize with empty dict to prevent NoneType errors

        # PHASE 2: Sanctions Check (Critical)
        if self.settings.enable_sanctions_check:
            try:
                results["sanctions"] = await self._check_sanctions(full_name, date_of_birth, country)
            except Exception as e:
                self.logger.error(f"Sanctions check failed: {e}")
                errors.append(f"sanctions: {str(e)}")
                results["sanctions"] = {}  # Initialize with empty dict to prevent NoneType errors

        # PHASE 2.5: PEP Check (Critical)
        if self.settings.enable_pep_check:
            try:
                results["pep"] = await self._check_pep(full_name, date_of_birth, country)
            except Exception as e:
                self.logger.error(f"PEP check failed: {e}")
                errors.append(f"pep: {str(e)}")
                results["pep"] = {}  # Initialize with empty dict to prevent NoneType errors

        # PHASE 5: Calculate Risk Score
        try:
            risk_assessment = await self._calculate_risk(results, full_name, date_of_birth, country, address)
        except Exception as e:
            self.logger.error(f"Risk calculation failed: {e}")
            risk_assessment = {
                "result": "ERROR",
                "reason": f"Risk calculation failed: {str(e)}",
                "overall_risk_score": 0.0,
                "risk_category": "UNKNOWN",
                "component_scores": {},
                "error": str(e)
            }

        # Determine if this is a match (high risk) based on PASS/FAIL result
        # NEW: Use binary PASS/FAIL result instead of score threshold
        is_match = risk_assessment.get("result") == "FAIL"

        # Build match_details for World-Check compatibility
        match_details = self._build_match_details(results, risk_assessment)

        processing_time = (datetime.now() - start_time).total_seconds()

        # Add web_search to check_breakdown
        check_breakdown = risk_assessment.get("check_breakdown") or {}
        check_breakdown["web_search_checks"] = results.get("web_search") or {}

        response = {
            "is_match": is_match,
            "result": risk_assessment.get("result"),  # NEW: "PASS" or "FAIL"
            "reason": risk_assessment.get("reason"),  # NEW: Explanation for result
            "overall_risk_score": risk_assessment["overall_risk_score"],
            "risk_category": risk_assessment["risk_category"],
            "match_details": match_details,
            "screening_id": screening_id,
            "component_scores": risk_assessment.get("component_scores") or {},
            "check_breakdown": check_breakdown,
            "public_records": {
                "sanctions": results.get("sanctions") or {}
            },
            "raw_response": results,
            "processing_time_seconds": processing_time,
            "error": "; ".join(errors) if errors else None
        }

        self.logger.info(
            f"OSINT screening {screening_id} completed: "
            f"Result={risk_assessment.get('result')}, "
            f"Reason={risk_assessment.get('reason')}, "
            f"Score={risk_assessment['overall_risk_score']:.1f}, "
            f"Category={risk_assessment['risk_category']}, "
            f"Match={is_match}"
        )

        return response

    async def _search_web(
        self,
        full_name: str,
        date_of_birth: Optional[str],
        country: Optional[str]
    ) -> Dict[str, Any]:
        """Search web using DuckDuckGo (primary) and Google scraper (fallback)."""
        self.logger.debug(f"Starting web search for: {full_name}")

        # Primary: DuckDuckGo search
        ddg_results = await self.duckduckgo_provider.search(
            full_name=full_name,
            date_of_birth=date_of_birth,
            country=country,
            max_results=10
        )

        # Fallback: Google scraper if DuckDuckGo returns limited results
        google_results = None
        if ddg_results.get('results_count', 0) < 3:
            self.logger.info("DuckDuckGo returned limited results, trying Google scraper")
            try:
                google_results = await self.google_scraper.search(
                    full_name=full_name,
                    date_of_birth=date_of_birth,
                    country=country,
                    max_results=10
                )
            except Exception as e:
                self.logger.warning(f"Google scraper failed: {e}")

        # Combine results
        combined_results = {
            "queries_performed": ddg_results.get('queries_performed', 0),
            "results_count": ddg_results.get('results_count', 0),
            "negative_news_count": ddg_results.get('negative_news_count', 0),
            "sources": ddg_results.get('sources', []),
            "results": ddg_results.get('results', []),  # NLP-enhanced results with sentiment data
            "duckduckgo_results": ddg_results
        }

        if google_results:
            combined_results['results_count'] += google_results.get('results_count', 0)
            combined_results['negative_news_count'] += google_results.get('negative_news_count', 0)
            combined_results['sources'].extend(google_results.get('sources', []))
            combined_results['google_results'] = google_results

        self.logger.info(
            f"Web search completed: {combined_results['results_count']} results, "
            f"{combined_results['negative_news_count']} negative news items"
        )

        return combined_results

    async def _check_sanctions(
        self,
        full_name: str,
        date_of_birth: Optional[str],
        country: Optional[str]
    ) -> Dict[str, Any]:
        """Check against free sanctions lists (OFAC, EU, UN)."""
        # Access checker dynamically from module (initialized at runtime in app/__init__.py)
        checker = sanctions_list_checker.sanctions_checker
        if checker is None:
            self.logger.warning("Sanctions checker not initialized")
            return {}
        return await checker.check_individual(
            full_name=full_name,
            date_of_birth=date_of_birth,
            country=country
        )

    async def _check_pep(
        self,
        full_name: str,
        date_of_birth: Optional[str],
        country: Optional[str]
    ) -> Dict[str, Any]:
        """Check against PEP (Politically Exposed Persons) database."""
        # Access checker dynamically from module (initialized at runtime in app/__init__.py)
        checker = pep_checker.pep_checker
        if checker is None:
            self.logger.warning("PEP checker not initialized")
            return {}
        return await checker.check_individual(
            full_name=full_name,
            date_of_birth=date_of_birth,
            country=country
        )

    async def _calculate_risk(
        self,
        results: Dict[str, Any],
        full_name: str,
        date_of_birth: Optional[str],
        country: Optional[str],
        address: Optional[str]
    ) -> Dict[str, Any]:
        """Calculate overall risk score from all search results."""
        return self.risk_scorer.calculate_risk(
            results=results,
            full_name=full_name,
            date_of_birth=date_of_birth,
            country=country,
            address=address
        )

    def _build_match_details(
        self,
        results: Dict[str, Any],
        risk_assessment: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Build match_details list compatible with World-Check format."""
        details = []

        # Add sanctions matches
        sanctions = results.get("sanctions") or {}
        if sanctions:
            # OFAC Sanctions
            if sanctions.get("ofac_match"):
                details.append(self._build_sanctions_match_detail(
                    "ofac", sanctions, "OFAC Specially Designated Nationals (SDN) List", "OFAC SDN List"
                ))

            # EU Sanctions
            if sanctions.get("eu_sanctions_match"):
                details.append(self._build_sanctions_match_detail(
                    "eu", sanctions, "EU Financial Sanctions Files (FSF)", "EU Consolidated List of Sanctions"
                ))

            # UN Sanctions
            if sanctions.get("un_sanctions_match"):
                details.append(self._build_sanctions_match_detail(
                    "un", sanctions, "UN Security Council Consolidated List", "UN Security Council Sanctions List"
                ))

        # Add negative news
        web_search = results.get("web_search") or {}
        if web_search and web_search.get("negative_news_count", 0) > 0:
            details.append({
                "match_strength": 0.7,
                "matched_name": "Negative News Found",
                "name_similarity": 0.8,
                "secondary_match": False,
                "is_true_match": True,
                "category": "negative_news",
                "subcategory": "web_search",
                "sources": web_search.get("sources", []),
                "description": f"{web_search.get('negative_news_count', 0)} negative news articles found"
            })

        return details

    def _build_sanctions_match_detail(
        self,
        source_key: str,
        sanctions: Dict[str, Any],
        list_name: str,
        source_short_name: str
    ) -> Dict[str, Any]:
        """
        Build sanctions match detail with binary confidence information.

        Args:
            source_key: 'ofac', 'eu', or 'un'
            sanctions: Sanctions results dict
            list_name: Full name of the sanctions list
            source_short_name: Short name for sources array

        Returns:
            Match detail dict with binary confidence and matched fields
        """
        # Get binary confidence for this source
        binary_confidence_key = f"{source_key}_binary_confidence"
        binary_confidence = sanctions.get(binary_confidence_key, 0.0)

        # Get details to extract matched name
        details_key = f"{source_key}_details"
        details = sanctions.get(details_key, [])

        # Extract matched name from the first (highest confidence) match
        matched_name = "Unknown"
        matched_fields = {'name': False, 'country': False, 'dob': False}

        if details and len(details) > 0:
            first_match = details[0]
            matched_name = first_match.get('name', matched_name)
            # Preserve the matched_fields from the database entry for reference
            matched_fields = first_match.get('matched_fields', {'name': True, 'country': False, 'dob': False})

        # Determine field match description based on binary confidence
        # This is more accurate than relying on matched_fields which might be inconsistent
        confidence_pct = int(binary_confidence * 100)

        if binary_confidence >= 0.98:
            fields_desc = "Name + Country + DOB"
        elif binary_confidence >= 0.65:
            fields_desc = "Name + Country"
        elif binary_confidence >= 0.32:
            fields_desc = "Name"
        else:
            fields_desc = "Unknown"

        return {
            "match_strength": binary_confidence,  # 0.33, 0.66, or 0.99
            "matched_name": matched_name,
            "name_similarity": binary_confidence,
            "binary_confidence": binary_confidence,
            "matched_fields": matched_fields,  # Preserve original for reference
            "secondary_match": False,
            "is_true_match": True,
            "category": "sanctions",
            "subcategory": f"{source_key}_sanctions_list",
            "sources": [source_short_name],
            "description": f"Match found on {list_name} ({fields_desc} match: {confidence_pct}% confidence)"
        }

    async def _get_face_verified_negative_news(
        self,
        full_name: str,
        country: Optional[str],
        user_identity_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get negative news from face-verified articles only.

        Uses image search to find articles with photos matching the user's selfie,
        then analyzes only those articles for negative sentiment.

        Returns:
            Dict with negative_news_count, results, sources (same format as web_search)
            or None if face-verified search disabled/unavailable
        """
        if not user_identity_id:
            self.logger.info("No user_identity_id provided, skipping face-verified search")
            return None

        # Retrieve selfie embedding from database
        embeddings = await asyncio.to_thread(
            face_biometrics_repository.get_embeddings_by_user_identity_ordered,
            user_identity_id=user_identity_id,
            limit=1
        )

        if not embeddings:
            self.logger.info(f"No selfie embedding found for user {user_identity_id}")
            return None

        # Get the embedding vector
        selfie_embedding = embeddings[0].get('embedding')

        if not selfie_embedding:
            self.logger.warning(f"Empty embedding for user {user_identity_id}")
            return None

        # Perform image search with face matching
        self.logger.debug(f"Starting face-verified image search for: {full_name}")
        try:
            image_results = await self.public_image_search_provider.search(
                full_name=full_name,
                country=country,
                max_results=self.settings.face_verified_max_images,
                selfie_embedding=selfie_embedding
            )
            self.logger.info(f"Image search completed: {image_results.get('images_downloaded', 0)} downloaded, {image_results.get('matches_found', 0)} matched")
        except Exception as e:
            self.logger.error(f"Face-verified image search failed: {type(e).__name__}")
            # Fall back to regular web search on error
            raise

        if not image_results or image_results.get('matches_found', 0) == 0:
            self.logger.debug(f"No face matches found for {full_name}")
            # Return empty result (not None) to indicate search was done but no matches
            return {
                "results_count": 0,
                "negative_news_count": 0,
                "sources": [],
                "average_sentiment": 0.0,
                "results": []
            }

        # Store matched pages with face match details
        matched_pages = image_results.get('matched_page_urls', [])

        # Extract page URLs from matched results
        page_urls = [match['page_url'] for match in matched_pages]

        self.logger.info(
            f"Face-verified search found {len(page_urls)} page URLs. "
            f"Analyzing content for negative sentiment..."
        )

        # Analyze these page URLs for negative sentiment
        analysis_results = await self._analyze_face_verified_pages(
            page_urls=page_urls,
            full_name=full_name,
            country=country
        )

        # Add face_match details to each result
        results_with_faces = []
        for result in analysis_results.get('results', []):
            # Find the corresponding face match for this URL
            face_match = next((m for m in matched_pages if m['page_url'] == result['url']), None)
            results_with_faces.append({
                **result,
                'face_match': {
                    'image_url': face_match.get('image_url'),
                    'similarity': face_match.get('similarity'),
                    'confidence': face_match.get('confidence'),
                    'source_type': face_match.get('source_type')
                } if face_match else None
            })

        # Calculate average sentiment score
        sentiment_scores = [(r.get('sentiment') or {}).get('compound', 0) for r in results_with_faces]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0

        return {
            "results_count": analysis_results.get('results_count'),
            "negative_news_count": analysis_results.get('negative_news_count'),
            "sources": analysis_results.get('sources'),
            "average_sentiment": avg_sentiment,
            "results": results_with_faces
        }

    async def _analyze_face_verified_pages(
        self,
        page_urls: List[str],
        full_name: str,
        country: Optional[str]
    ) -> Dict[str, Any]:
        """
        Analyze face-verified page URLs for negative sentiment.

        Fetches content from each URL and runs VADER + FinBERT analysis.
        """
        results = []
        negative_count = 0
        sources = []

        for url in page_urls:
            try:
                # Fetch page content
                async with httpx.AsyncClient(timeout=15.0) as client:
                    response = await client.get(url)
                    response.raise_for_status()

                # Extract article content using trafilatura
                content = trafilatura.extract_content(response.text)

                if not content:
                    self.logger.warning(f"Could not extract content from {url}")
                    continue

                # Get title from HTML
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')
                title_tag = soup.find('title')
                title_text = title_tag.get_text() if title_tag else ""

                # Analyze with NLP
                nlp_result = await self._analyze_with_nlp(
                    title=title_text,
                    body=content,
                    url=url,
                    full_name=full_name,
                    country=country
                )

                if nlp_result.get('risk_level') == 'negative':
                    negative_count += 1
                    sources.append(url)

                results.append({
                    "title": title_text,
                    "url": url,
                    "body": content[:500] if content else "",  # Truncate for storage
                    "risk_level": nlp_result.get('risk_level'),
                    "sentiment": nlp_result.get('sentiment'),
                    "deep_analysis": nlp_result.get('deep_analysis')
                })

            except Exception as e:
                self.logger.warning(f"Failed to analyze {url}: {str(e)}")
                continue

        self.logger.info(
            f"Face-verified analysis complete: {len(results)} pages analyzed, "
            f"{negative_count} negative news items"
        )

        return {
            "results_count": len(results),
            "negative_news_count": negative_count,
            "sources": sources,
            "results": results
        }

    async def _analyze_with_nlp(
        self,
        title: str,
        body: str,
        url: str,
        full_name: str,
        country: Optional[str]
    ) -> Dict[str, Any]:
        """
        Analyze text with NLP (VADER + FinBERT).
        """
        combined_text = f"{title} {body}"
        result = {
            "url": url,
            "title": title,
            "risk_level": None,
            "sentiment": None,
            "deep_analysis": None
        }

        # Check if NLP service is initialized
        if not self.nlp_service or not self.nlp_service._initialized:
            return result

        # VADER sentiment analysis
        sentiment = self.nlp_service.analyze_sentiment_vader(combined_text)

        if sentiment['compound'] < self.settings.nlp_sentiment_threshold:  # Default -0.3
            result['risk_level'] = 'negative'
            result['sentiment'] = sentiment

            # FinBERT deep analysis for negative items
            if self.settings.enable_nlp_enhanced_analysis:
                try:
                    deep_analysis = self.nlp_service.extract_event_type(body)
                    result['deep_analysis'] = deep_analysis
                except Exception as e:
                    self.logger.debug(f"FinBERT analysis failed for {url}: {e}")

        return result


# Global service instance
osint_screening_service = OSINTScreeningService()

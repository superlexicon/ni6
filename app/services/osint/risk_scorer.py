"""
Risk Scoring Algorithm for OSINT Background Search.

Calculates overall risk scores (0-100) based on multiple components:
- Negative news (29% weight)
- Criminal records (15% weight)
- Sanctions (22% weight)
- PEP - Politically Exposed Persons (29% weight)
- Digital footprint (0% weight - informational only, not scored)
"""

from typing import Dict, Any, Optional
from datetime import datetime, date
from app.core.logger import get_logger
from app.config.osint_config import osint_settings


class RiskScorer:
    """
    Calculates risk scores for OSINT background screening.

    Risk Categories:
    - LOW (0-24): Auto-approve
    - MEDIUM (25-49): Manual review optional
    - HIGH (50-74): Manual review required
    - CRITICAL (75-100): Auto-reject
    """

    # Component weights (must sum to 1.0)
    WEIGHTS = {
        "negative_news": 0.29,
        "criminal_records": 0.15,
        "sanctions": 0.22,
        "pep": 0.29,
        "digital_footprint": 0.00  # Informational only, not included in scoring
    }

    # Negative news keywords with point values
    NEGATIVE_NEWS_KEYWORDS = {
        'critical': {
            'keywords': ['fraud', 'scam', 'money laundering', 'embezzlement', 'bribery',
                        'corruption', 'terrorist', 'terrorism', 'sanctioned', 'convicted',
                        'arrested', 'indicted', 'charged', 'criminal', 'ponzi',
                        'pyramid scheme', 'identity theft', 'cybercrime', 'hacking'],
            'points': 50,
            'recency_multiplier': 2.0  # Double if within 1 year
        },
        'high': {
            'keywords': ['investigation', 'probe', 'audit', 'complaint', 'allegation',
                        'accused', 'sued', 'settled', 'controversy', 'misconduct',
                        'violation', 'penalty', 'fine', 'regulator'],
            'points': 30,
            'recency_multiplier': 1.5
        },
        'medium': {
            'keywords': ['dispute', 'conflict', 'issue', 'concern', 'problem',
                         'failure', 'bankrupt', 'liquidation', 'delisted', 'warning'],
            'points': 10,
            'recency_multiplier': 1.0
        }
    }

    def __init__(self):
        self.logger = get_logger()

    def calculate_risk(
        self,
        results: Dict[str, Any],
        full_name: str,
        date_of_birth: Optional[str],
        country: Optional[str],
        address: Optional[str]
    ) -> Dict[str, Any]:
        """
        Calculate overall risk score from OSINT search results.

        Args:
            results: Combined search results from all providers
            full_name: Expected full name
            date_of_birth: Expected date of birth
            country: Expected country
            address: Expected address

        Returns:
            {
                "overall_risk_score": float,  # 0-100
                "risk_category": str,  # LOW, MEDIUM, HIGH, CRITICAL
                "component_scores": dict,
                "sanctions_binary_confidence": float,  # NEW: 0.0, 0.33, 0.66, or 0.99
                "sanctions_override": bool  # NEW: True if sanctions determined final score
            }
        """
        self.logger.debug(f"Calculating risk score for: {full_name}")

        # Calculate component scores
        component_scores = {
            "negative_news": self._calculate_negative_news_risk(results.get('web_search') or {}),
            "criminal_records": self._calculate_criminal_risk(results),
            "sanctions": self._calculate_sanctions_risk(results),
            "pep": self._calculate_pep_risk(results),
            "digital_footprint": self._calculate_footprint_risk(results)
        }

        # Calculate overall result using sanctions override model
        risk_result = self._calculate_overall_risk_with_sanctions_override(
            results, component_scores
        )

        # Extract result components
        result = risk_result["result"]  # "PASS" or "FAIL"
        reason = risk_result["reason"]
        overall_score = risk_result["overall_score"]

        # Determine if sanctions override was used
        sanctions = results.get('sanctions') or {}
        sanctions_override = sanctions.get('binary_confidence', 0.0) > 0

        # Map result to risk category for backward compatibility
        # FAIL -> HIGH (or CRITICAL if very high score), PASS -> LOW
        if result == "FAIL":
            risk_category = "CRITICAL" if overall_score >= 75 else "HIGH"
        else:
            risk_category = "LOW"

        self.logger.info(
            f"Risk assessment: {result} - {reason} - "
            f"Score: {overall_score:.1f} ({risk_category}) - "
            f"Sanctions override: {sanctions_override}, "
            f"Components: {component_scores}"
        )

        # Build detailed check breakdown
        check_breakdown = self._build_check_breakdown(
            sanctions,
            results.get('pep') or {},
            results.get('web_search') or {}
        )

        return {
            "result": result,  # NEW: "PASS" or "FAIL"
            "reason": reason,  # NEW: Explanation for result
            "overall_risk_score": round(overall_score, 2),
            "risk_category": risk_category,
            "component_scores": component_scores,
            "sanctions_binary_confidence": sanctions.get('binary_confidence', 0.0),
            "sanctions_override": sanctions_override,
            "check_breakdown": check_breakdown
        }

    def _calculate_negative_news_risk(self, web_search: Dict[str, Any]) -> float:
        """
        Calculate risk score from negative news using NLP analysis.

        Scoring:
        - Conviction: +50 points (adjusted by confidence)
        - Investigation: +30 points
        - Allegation: +15 points
        - Cleared: 0 points
        - Very negative sentiment (< -0.5): +30 points
        - Somewhat negative sentiment (< -0.3): +15 points
        - Mildly negative sentiment (< -0.1): +5 points
        - Keyword-based fallback (when NLP unavailable): +10-50 points

        Returns risk score (0-100)
        """
        if not web_search or not isinstance(web_search, dict):
            return 0.0

        results = web_search.get('results') or []
        if not results:
            return 0.0

        total_score = 0.0

        for result in results:
            # DEFENSIVE: Only count results actually marked as negative
            if result.get('risk_level') != 'negative':
                continue

            deep_analysis = result.get('deep_analysis')

            if deep_analysis:
                # Use FinBERT classification
                classification = deep_analysis.get('classification') or {}
                event_type = classification.get('event_type', 'unknown')
                confidence = classification.get('confidence', 0.5)

                if event_type == 'conviction':
                    total_score += 50 * confidence
                elif event_type == 'investigation':
                    total_score += 30 * confidence
                elif event_type == 'civil_lawsuit':
                    # Civil lawsuits are much lower risk than criminal matters
                    total_score += 10 * confidence
                elif event_type == 'allegation':
                    total_score += 15 * confidence
                # 'cleared' adds 0 points
            else:
                # Use sentiment analysis if available
                sentiment = result.get('sentiment') or {}
                if sentiment:
                    compound = sentiment.get('compound', 0)

                    # Calculate score based on sentiment strength
                    if compound < -0.5:  # Very negative
                        total_score += 30
                    elif compound < -0.3:  # Somewhat negative
                        total_score += 15
                    elif compound < -0.1:  # Mildly negative
                        total_score += 5
                else:
                    # FALLBACK: Keyword-based scoring when NLP unavailable
                    # This handles cases where NLP service failed to initialize
                    # but the search provider still detected negative keywords
                    title = result.get('title', '')
                    body = result.get('body', '')
                    combined_text = f'{title} {body}'.lower()

                    # Use the existing keyword scoring system
                    keyword_score = self._score_by_keywords(combined_text)
                    total_score += keyword_score

        return min(total_score, 100.0)

    def _score_by_keywords(self, text: str) -> float:
        """
        Score text using keyword-based detection (fallback when NLP unavailable).

        Returns score (0-50) based on keyword severity.
        """
        if not text:
            return 0.0

        max_score = 0.0

        # Check each severity level
        for severity, config in self.NEGATIVE_NEWS_KEYWORDS.items():
            keywords = config['keywords']
            points = config['points']

            for keyword in keywords:
                if keyword in text:
                    max_score = max(max_score, points)

        return max_score

    def _calculate_criminal_risk(self, results: Dict[str, Any]) -> float:
        """
        Calculate risk from criminal records (0-100).

        Risk factors:
        - Felony convictions: +60 points each
        - Misdemeanor convictions: +30 points each
        - Pending charges: +40 points each
        - Country-specific criminal records: +60 points
        - Country-specific court cases: +30 points

        Note: Sanctions are now calculated separately in _calculate_sanctions_risk()
        """
        if not results:
            return 0.0

        risk_score = 0.0

        # Check country-specific records (NOT sanctions)
        country_records = results.get('country_specific') or {}
        if country_records.get('criminal_records'):
            risk_score += 60
        if country_records.get('court_cases'):
            risk_score += 30

        return min(risk_score, 100.0)

    def _calculate_sanctions_risk(self, results: Dict[str, Any]) -> float:
        """
        Calculate risk from sanctions using binary confidence model.

        Uses binary confidence from sanctions checker:
        - Name + Country + DOB match: 99 points
        - Name + Country match: 66 points
        - Name only match: 33 points
        - No match: 0 points

        Args:
            results: Combined search results

        Returns:
            Risk score (0-99)
        """
        if not results:
            return 0.0

        sanctions = results.get('sanctions') or {}
        if not sanctions:
            return 0.0

        # Get the highest binary confidence across all sanctions lists
        highest_confidence = sanctions.get('binary_confidence', 0.0)

        # Convert to 0-100 scale
        return highest_confidence * 100

    def _calculate_pep_risk(self, results: Dict[str, Any]) -> float:
        """
        Calculate risk from PEP status using binary confidence model.

        Scoring:
        - Current PEP (high confidence): 60 points
        - Former PEP (high confidence): 30 points
        - Adjusted by binary confidence (0.33, 0.66, 0.99)

        Args:
            results: Combined search results

        Returns:
            Risk score (0-60)
        """
        if not results:
            return 0.0

        pep = results.get('pep') or {}
        if not pep:
            return 0.0

        # Check if PEP match found
        if not pep.get('is_pep', False):
            return 0.0

        # Get binary confidence
        binary_confidence = pep.get('binary_confidence', 0.0)
        if binary_confidence == 0.0:
            return 0.0

        # Base score depends on current vs former
        if pep.get('current_pep_match', False):
            base_score = 60.0  # Higher risk for current PEPs
        elif pep.get('former_pep_match', False):
            base_score = 30.0  # Lower risk for former PEPs
        else:
            return 0.0

        # Adjust by binary confidence (0.33, 0.66, or 0.99)
        # This prevents false positives from low-confidence matches
        if binary_confidence >= 0.98:
            return base_score  # Full score for high confidence
        elif binary_confidence >= 0.65:
            return base_score * 0.8  # 80% for medium confidence
        else:  # 0.33
            return base_score * 0.5  # 50% for low confidence

    def _calculate_overall_risk_with_sanctions_override(
        self,
        results: Dict[str, Any],
        component_scores: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Calculate PASS/FAIL using "any failure = fail" model.

        Strategy:
        - FAIL if high-confidence sanctions (≥65%)
        - FAIL if high-confidence PEP (≥65%)
        - FAIL if high negative news (≥50)
        - PASS only if ALL checks are low-confidence/low-risk

        Args:
            results: Combined search results
            component_scores: Individual component scores

        Returns:
            {
                "result": "PASS" or "FAIL",
                "reason": str (explanation),
                "overall_score": float (max component score for backward compatibility)
            }
        """
        sanctions = results.get('sanctions') or {}
        sanctions_confidence = sanctions.get('binary_confidence', 0.0)

        pep = results.get('pep') or {}
        is_pep = pep.get('is_pep', False)
        pep_confidence = pep.get('binary_confidence', 0.0)

        negative_news = component_scores.get('negative_news', 0.0)

        # FAIL Condition 1: High-confidence sanctions (≥65%)
        if sanctions_confidence >= 0.65:
            return {
                "result": "FAIL",
                "reason": f"High-confidence sanctions match ({sanctions_confidence*100:.0f}%)",
                "overall_score": sanctions_confidence * 100
            }

        # FAIL Condition 2: High-confidence PEP (≥65%)
        if is_pep and pep_confidence >= 0.65:
            pep_score = component_scores.get('pep', 0.0)
            return {
                "result": "FAIL",
                "reason": f"High-confidence PEP match ({pep_confidence*100:.0f}%)",
                "overall_score": pep_score
            }

        # FAIL Condition 3: High negative news (≥50 - serious criminal indicators)
        if negative_news >= 50:
            return {
                "result": "FAIL",
                "reason": f"High negative news score ({negative_news:.0f}/100)",
                "overall_score": max(negative_news, 60.0)
            }

        # All checks passed - return PASS with max score
        max_score = max(component_scores.values()) if component_scores else 0.0
        return {
            "result": "PASS",
            "reason": "All checks within acceptable thresholds",
            "overall_score": max_score
        }

    def _build_check_breakdown(
        self,
        sanctions_result: Dict[str, Any],
        pep_result: Dict[str, Any],
        web_search_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build detailed breakdown of all checks performed.

        Args:
            sanctions_result: Sanctions check result from sanctions checker
            pep_result: PEP check result from PEP checker
            web_search_result: Web search results with negative sentiment analysis

        Returns:
            {
                "sanctions_checks": {status, score, binary_confidence, matched_lists, details},
                "pep_checks": {status, score, is_pep, pep_type, binary_confidence, details},
                "negative_sentiment_checks": {status, score, total_analyzed, negative_count, classification, top_matches}
            }
        """
        # Build sanctions checks breakdown
        sanctions_checks = self._build_sanctions_breakdown(sanctions_result)

        # Build PEP checks breakdown
        pep_checks = self._build_pep_breakdown(pep_result)

        # Build negative sentiment checks breakdown
        negative_sentiment_checks = self._build_negative_sentiment_breakdown(web_search_result)

        return {
            "sanctions_checks": sanctions_checks,
            "pep_checks": pep_checks,
            "negative_sentiment_checks": negative_sentiment_checks
        }

    def _build_sanctions_breakdown(self, sanctions_result: Dict[str, Any]) -> Dict[str, Any]:
        """Build detailed sanctions check breakdown."""
        if not sanctions_result:
            return {
                "status": "PASS",
                "score": 0.0,
                "binary_confidence": 0.0,
                "matched_lists": [],
                "total_lists_checked": 0,
                "details": []
            }

        binary_confidence = sanctions_result.get('binary_confidence', 0.0)
        score = binary_confidence * 100
        status = "FAIL" if binary_confidence >= 0.65 else "PASS"

        # Build matched lists
        matched_lists = []
        if sanctions_result.get('sanctions_match'):
            lists_checked = sanctions_result.get('lists_checked', [])
            matched_lists = [lst.replace('_', ' ').title() for lst in lists_checked]

        # Build details from sanctions_details
        details = []

        sanctions_details = sanctions_result.get('sanctions_details') or []
        for detail in sanctions_details:
            # Determine the list name from source_key
            source_key = detail.get('source_key', 'crime_entities')
            list_name = source_key.replace('_', ' ').title()
            if list_name == 'Crime Entities':
                list_name = 'Crime/Sanctions Watchlist'

            details.append({
                "list": list_name,
                "matched_name": detail.get('name', ''),
                "match_type": detail.get('match_type', 'unknown'),
                "confidence": detail.get('name_match_confidence', 0.0),
                "matched_fields": {
                    "name": (detail.get('matched_fields') or {}).get('name', False),
                    "country": (detail.get('matched_fields') or {}).get('country', False),
                    "dob": (detail.get('matched_fields') or {}).get('dob', False)
                }
            })

        return {
            "status": status,
            "score": round(score, 2),
            "binary_confidence": binary_confidence,
            "matched_lists": matched_lists,
            "total_lists_checked": len(sanctions_result.get('lists_checked') or []),
            "details": details
        }

    def _build_pep_breakdown(self, pep_result: Dict[str, Any]) -> Dict[str, Any]:
        """Build detailed PEP check breakdown."""
        if not pep_result:
            return {
                "status": "PASS",
                "score": 0.0,
                "is_pep": False,
                "pep_type": None,
                "binary_confidence": 0.0,
                "positions_found": [],
                "details": []
            }

        is_pep = pep_result.get('is_pep', False)
        binary_confidence = pep_result.get('binary_confidence', 0.0)
        status = "FAIL" if is_pep and binary_confidence >= 0.65 else "PASS"

        # Determine PEP type
        pep_type = None
        if pep_result.get('current_pep_match'):
            pep_type = "current"
        elif pep_result.get('former_pep_match'):
            pep_type = "former"

        # Calculate score (same logic as _calculate_pep_risk)
        if is_pep and binary_confidence > 0:
            if pep_type == "current":
                base_score = 60.0
            elif pep_type == "former":
                base_score = 30.0
            else:
                base_score = 30.0

            if binary_confidence >= 0.98:
                score = base_score
            elif binary_confidence >= 0.65:
                score = base_score * 0.8
            else:
                score = base_score * 0.5
        else:
            score = 0.0

        # Build details from PEP details
        details = []
        pep_details = pep_result.get('pep_details') or []
        for detail in pep_details:
            details.append({
                "name": detail.get('name', ''),
                "position": detail.get('position', ''),
                "country": detail.get('country'),
                "match_type": detail.get('match_type', 'unknown'),
                "confidence": detail.get('confidence', 0.0)
            })

        return {
            "status": status,
            "score": round(score, 2),
            "is_pep": is_pep,
            "pep_type": pep_type,
            "binary_confidence": binary_confidence,
            "positions_found": pep_result.get('positions_found') or [],
            "details": details
        }

    def _build_negative_sentiment_breakdown(self, web_search_result: Dict[str, Any]) -> Dict[str, Any]:
        """Build detailed negative sentiment check breakdown."""
        if not web_search_result:
            return {
                "status": "PASS",
                "score": 0.0,
                "total_analyzed": 0,
                "negative_count": 0,
                "classification": {
                    "conviction": {"count": 0, "avg_confidence": 0.0},
                    "investigation": {"count": 0, "avg_confidence": 0.0},
                    "civil_lawsuit": {"count": 0, "avg_confidence": 0.0},
                    "allegation": {"count": 0, "avg_confidence": 0.0},
                    "cleared": {"count": 0, "avg_confidence": 0.0}
                },
                "top_matches": []
            }

        results = web_search_result.get('results') or []
        total_analyzed = len(results)
        negative_count = web_search_result.get('negative_news_count', 0)

        # Calculate score using same logic as _calculate_negative_news_risk
        score = self._calculate_negative_news_risk(web_search_result)
        status = "FAIL" if score >= 50 else "PASS"

        # Build classification breakdown
        classification = {
            "conviction": {"count": 0, "avg_confidence": 0.0},
            "investigation": {"count": 0, "avg_confidence": 0.0},
            "civil_lawsuit": {"count": 0, "avg_confidence": 0.0},
            "allegation": {"count": 0, "avg_confidence": 0.0},
            "cleared": {"count": 0, "avg_confidence": 0.0}
        }

        # Build top matches (limit to top 10 for response size)
        top_matches = []
        for result in results[:50]:  # Analyze up to 50 results
            deep_analysis = result.get('deep_analysis')
            if deep_analysis:
                classification_data = deep_analysis.get('classification') or {}
                event_type = classification_data.get('event_type', 'unknown')
                confidence = classification_data.get('confidence', 0.5)

                # Update classification counts
                if event_type in classification:
                    classification[event_type]["count"] += 1
                    # Update average confidence
                    current_avg = classification[event_type]["avg_confidence"]
                    current_count = classification[event_type]["count"]
                    classification[event_type]["avg_confidence"] = round(
                        (current_avg * (current_count - 1) + confidence) / current_count, 3
                    )

                # Add to top matches if it has relevant data
                if event_type in ['conviction', 'investigation', 'civil_lawsuit', 'allegation', 'cleared']:
                    top_matches.append({
                        "title": result.get('title', ''),
                        "url": result.get('href', ''),
                        "event_type": event_type,
                        "confidence": round(confidence, 3),
                        "sentiment_score": round((result.get('sentiment') or {}).get('compound', 0.0), 3)
                    })

        # Limit top matches to 10 most relevant (highest confidence)
        top_matches.sort(key=lambda x: x['confidence'], reverse=True)
        top_matches = top_matches[:10]

        return {
            "status": status,
            "score": round(score, 2),
            "total_analyzed": total_analyzed,
            "negative_count": negative_count,
            "classification": classification,
            "top_matches": top_matches
        }

    def _calculate_footprint_risk(self, results: Dict[str, Any]) -> float:
        """
        Calculate risk from digital footprint analysis (0-100).

        Risk factors:
        - No web presence at all: +30 points
        - Only recent web presence: +20 points
        - Consistent web presence: 0 points
        """
        web_search = results.get('web_search') or {}
        social_media = results.get('social_media') or {}

        total_results = web_search.get('results_count', 0)
        profiles_found = len(social_media.get('profiles_found') or {})

        if total_results == 0 and profiles_found == 0:
            return 30.0  # No web presence at all

        if total_results < 5:
            return 20.0  # Very limited web presence

        return 0.0  # Normal web presence

    def _get_risk_category(self, score: float) -> str:
        """Convert risk score to category."""
        if score >= 75:
            return "CRITICAL"
        elif score >= 50:
            return "HIGH"
        elif score >= 25:
            return "MEDIUM"
        else:
            return "LOW"

    def analyze_news_content(self, text: str, publish_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze news content for risk keywords.

        Args:
            text: News content to analyze
            publish_date: Publication date (for recency multiplier)

        Returns:
            {
                "risk_level": str,  # critical, high, medium, none
                "keywords_found": list,
                "score_contribution": float
            }
        """
        text_lower = text.lower()
        keywords_found = []
        total_score = 0.0

        # Check each keyword category
        for category, config in self.NEGATIVE_NEWS_KEYWORDS.items():
            for keyword in config['keywords']:
                if keyword in text_lower:
                    keywords_found.append({
                        'keyword': keyword,
                        'category': category
                    })

                    # Calculate score with recency multiplier
                    points = config['points']
                    if publish_date and config.get('recency_multiplier', 1.0) > 1.0:
                        # Check if news is recent (<1 year)
                        try:
                            pub_date = datetime.fromisoformat(publish_date.replace('Z', '+00:00'))
                            if (datetime.now() - pub_date).days < 365:
                                points *= config['recency_multiplier']
                        except:
                            pass

                    total_score += points

        # Determine risk level
        if any(kw['category'] == 'critical' for kw in keywords_found):
            risk_level = 'critical'
        elif any(kw['category'] == 'high' for kw in keywords_found):
            risk_level = 'high'
        elif any(kw['category'] == 'medium' for kw in keywords_found):
            risk_level = 'medium'
        elif keywords_found:
            risk_level = 'low'
        else:
            risk_level = 'none'

        return {
            'risk_level': risk_level,
            'keywords_found': keywords_found,
            'score_contribution': min(total_score, 100.0)
        }


# Global instance
risk_scorer = RiskScorer()

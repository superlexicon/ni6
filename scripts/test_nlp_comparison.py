"""
Test script comparing NLP-enhanced negative news detection vs keyword-only approach.

This script runs the same screening scenarios with both approaches and compares:
- Overall risk scores
- Negative news counts
- Risk categories
- Component scores
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.osint_screening_service import osint_screening_service
from app.config.osint_config import osint_settings
from app.services.osint.risk_scorer import RiskScorer


def initialize_sanctions_checker():
    """Initialize sanctions checker for test script usage."""
    from app.services.osint.sanctions.sanctions_list_checker import SanctionsListChecker
    import app.services.osint.sanctions.sanctions_list_checker as checker_module
    from app.repositories import sanctions_repository
    from app.services.osint.sanctions.sanctions_sync_service import SanctionsSyncService
    from app.core import get_db_connection

    try:
        database = get_db_connection()
        sync_service = SanctionsSyncService(sanctions_repository, database)

        # Create and assign the sanctions_checker instance
        sanctions_checker_instance = SanctionsListChecker(
            sanctions_repository=sanctions_repository,
            sync_service=sync_service
        )
        checker_module.sanctions_checker = sanctions_checker_instance
        osint_screening_service.sanctions_checker = sanctions_checker_instance

        print("[INFO] Sanctions checker initialized")
        return True
    except Exception as e:
        print(f"[WARNING] Failed to initialize sanctions checker: {e}")
        return False


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_comparison_table(scenarios: list):
    """Print a comparison table of results."""
    print_section("COMPARISON TABLE: NLP-Enhanced vs Keyword-Only")

    # Header
    print(f"\n{'Scenario':<40} {'Keyword':<10} {'NLP':<10} {'Diff':<10} {'Category Change'}")
    print("-" * 100)

    for scenario in scenarios:
        name = scenario['name'][:38] + '..' if len(scenario['name']) > 40 else scenario['name']
        keyword_score = scenario['keyword_score']
        nlp_score = scenario['nlp_score']
        diff = nlp_score - keyword_score

        # Color indicators
        diff_str = f"{diff:+.1f}"
        if diff > 10:
            diff_str = f"⚠️ {diff_str}"
        elif diff < -10:
            diff_str = f"✅ {diff_str}"

        keyword_cat = scenario['keyword_category']
        nlp_cat = scenario['nlp_category']

        cat_change = ""
        if keyword_cat != nlp_cat:
            cat_change = f"{keyword_cat} → {nlp_cat}"

        print(f"{name:<40} {keyword_score:<10.1f} {nlp_score:<10.1f} {diff_str:<10} {cat_change}")


def print_detailed_analysis(scenario: dict, keyword_result: dict, nlp_result: dict):
    """Print detailed analysis for a single scenario."""
    print_section(f"Detailed Analysis: {scenario['name']}")

    print(f"\nDescription: {scenario.get('description', 'N/A')}")

    # Score comparison
    print(f"\n{'Method':<20} {'Score':<10} {'Category':<12} {'Action'}")
    print("-" * 70)

    keyword_score = keyword_result['overall_risk_score']
    keyword_cat = keyword_result['risk_category']
    keyword_action = "REJECT" if keyword_score >= osint_settings.risk_threshold else "APPROVE"

    nlp_score = nlp_result['overall_risk_score']
    nlp_cat = nlp_result['risk_category']
    nlp_action = "REJECT" if nlp_score >= osint_settings.risk_threshold else "APPROVE"

    print(f"{'Keyword-Only':<20} {keyword_score:<10.1f} {keyword_cat:<12} {keyword_action}")
    print(f"{'NLP-Enhanced':<20} {nlp_score:<10.1f} {nlp_cat:<12} {nlp_action}")

    diff = nlp_score - keyword_score
    print(f"\nScore Difference: {diff:+.1f} ({'NLP higher' if diff > 0 else 'Keyword higher' if diff < 0 else 'Same'})")

    # Component scores comparison
    print(f"\n{'Component':<25} {'Keyword':<10} {'NLP':<10} {'Difference'}")
    print("-" * 70)

    keyword_components = keyword_result.get('component_scores', {})
    nlp_components = nlp_result.get('component_scores', {})

    all_components = set(keyword_components.keys()) | set(nlp_components.keys())

    for component in sorted(all_components):
        keyword_val = keyword_components.get(component, 0)
        nlp_val = nlp_components.get(component, 0)
        diff_comp = nlp_val - keyword_val

        diff_str = f"{diff_comp:+.1f}"
        if abs(diff_comp) > 5:
            diff_str += " ⚠️"

        print(f"{component.replace('_', ' ').title():<25} {keyword_val:<10.1f} {nlp_val:<10.1f} {diff_str}")

    # Negative news analysis
    print_section("Negative News Analysis")

    keyword_web = keyword_result.get('search_sources', {}).get('web_search', {})
    nlp_web = nlp_result.get('search_sources', {}).get('web_search', {})

    keyword_neg = keyword_web.get('negative_news_count', 0)
    nlp_neg = nlp_web.get('negative_news_count', 0)
    keyword_results = keyword_web.get('results_count', 0)
    nlp_results = nlp_web.get('results_count', 0)

    print(f"\n{'Metric':<30} {'Keyword':<15} {'NLP':<15}")
    print("-" * 70)
    print(f"{'Total Search Results':<30} {keyword_results:<15} {nlp_results:<15}")
    print(f"{'Negative News Count':<30} {keyword_neg:<15} {nlp_neg:<15}")

    # Show sample results with NLP analysis
    if nlp_web.get('results'):
        print_section("Sample NLP-Analyzed Results")

        results = nlp_web['results'][:3]  # Show first 3
        for i, result in enumerate(results, 1):
            title = result.get('title', 'N/A')[:60]
            sentiment = result.get('sentiment', {})
            compound = sentiment.get('compound', 0) if sentiment else 0

            sentiment_label = "NEGATIVE" if compound < -0.3 else "NEUTRAL" if compound < 0.3 else "POSITIVE"
            deep_analysis = result.get('deep_analysis', {})

            print(f"\n{i}. {title}")
            print(f"   Sentiment: {sentiment_label} (compound: {compound:.2f})")

            if deep_analysis:
                classification = deep_analysis.get('classification', {})
                event_type = classification.get('event_type', 'unknown')
                confidence = classification.get('confidence', 0)
                print(f"   Deep Analysis: {event_type} (confidence: {confidence:.2f})")


async def screen_with_method(scorer: RiskScorer, full_name: str, date_of_birth: str = None,
                             country: str = None, address: str = None) -> dict:
    """Perform screening with a specific scorer configuration."""
    # Get web search results
    web_search = await osint_screening_service.duckduckgo_provider.search(
        full_name=full_name,
        date_of_birth=date_of_birth,
        country=country,
        max_results=10
    )

    # Get other results (simplified for comparison)
    results = {
        'web_search': web_search,
        'sanctions': {},
        'pep': {},
        'social_media': {},
        'country_specific': {}
    }

    # Calculate risk scores using the provided scorer
    risk_result = scorer.calculate_risk(
        results=results,
        full_name=full_name,
        date_of_birth=date_of_birth,
        country=country,
        address=address
    )

    # Build full result structure
    return {
        'overall_risk_score': risk_result['overall_risk_score'],
        'risk_category': risk_result['risk_category'],
        'component_scores': risk_result['component_scores'],
        'search_sources': {
            'web_search': web_search
        }
    }


async def run_comparison_tests():
    """Run comparison tests between NLP and keyword-only approaches."""
    load_dotenv()
    initialize_sanctions_checker()

    # Initialize NLP service for comparison tests
    print("Initializing NLP service...")
    from app.services.nlp.nlp_service import nlp_service
    try:
        await nlp_service.initialize()
        print("✓ NLP service initialized (VADER + spaCy)")
    except Exception as e:
        print(f"⚠ Failed to initialize NLP service: {e}")
        print("  Tests will run with keyword-only analysis for both methods")
        return

    print_section("NLP vs Keyword-Only Negative News Detection Comparison")

    # Test scenarios that demonstrate the differences
    test_scenarios = [
        {
            "name": "Cleared of Charges (False Positive Test)",
            "description": "Person cleared of fraud should NOT be counted as negative",
            "data": {
                "full_name": "John Smith",
                "country": "United States"
            }
            # Note: Using a generic name, but the pattern is what matters
            # In real testing, you'd use specific cases with known outcomes
        },
        {
            "name": "Actual Fraud Conviction",
            "description": "Person convicted of fraud should be high risk",
            "data": {
                "full_name": "Bernie Madoff",
                "country": "United States",
                "date_of_birth": "1938-04-29"
            }
        },
        {
            "name": "Business Controversy (Civil)",
            "description": "Civil allegations should be lower risk than criminal",
            "data": {
                "full_name": "Elizabeth Holmes",
                "country": "United States",
                "date_of_birth": "1984-02-03"
            }
        }
    ]

    comparison_results = []
    detailed_results = []

    # Create scorers
    nlp_scorer = RiskScorer()

    # Simulate keyword-only by forcing fallback
    # (We'll modify results to remove NLP data)

    for scenario in test_scenarios:
        print(f"\n{'='*70}")
        print(f"Testing: {scenario['name']}")
        print(f"{'='*70}")

        data = scenario['data']

        # Run with NLP enabled
        print("\n[1/2] Running NLP-enhanced analysis...")
        nlp_result = await screen_with_method(
            nlp_scorer,
            full_name=data['full_name'],
            date_of_birth=data.get('date_of_birth'),
            country=data.get('country'),
            address=data.get('address')
        )

        # Simulate keyword-only by removing NLP data
        print("[2/2] Running keyword-only analysis...")

        # First get the result with same search data
        keyword_result = await screen_with_method(
            nlp_scorer,
            full_name=data['full_name'],
            date_of_birth=data.get('date_of_birth'),
            country=data.get('country'),
            address=data.get('address')
        )

        # Now strip NLP data from results to simulate keyword-only
        web_search = keyword_result['search_sources']['web_search']
        keyword_only_web_search = {
            **web_search,
            'results': [
                {k: v for k, v in r.items() if k not in ['sentiment', 'entities', 'deep_analysis']}
                for r in web_search.get('results', [])
            ]
        }
        keyword_result['search_sources']['web_search'] = keyword_only_web_search

        # Re-calculate risk score without NLP data
        results_for_keyword = {
            'web_search': keyword_only_web_search,
            'sanctions': {},
            'pep': {},
            'social_media': {},
            'country_specific': {}
        }

        keyword_risk = nlp_scorer.calculate_risk(
            results=results_for_keyword,
            full_name=data['full_name'],
            date_of_birth=data.get('date_of_birth'),
            country=data.get('country'),
            address=data.get('address')
        )

        keyword_result['overall_risk_score'] = keyword_risk['overall_risk_score']
        keyword_result['risk_category'] = keyword_risk['risk_category']
        keyword_result['component_scores'] = keyword_risk['component_scores']

        # Store results
        comparison_results.append({
            'name': scenario['name'],
            'keyword_score': keyword_result['overall_risk_score'],
            'nlp_score': nlp_result['overall_risk_score'],
            'keyword_category': keyword_result['risk_category'],
            'nlp_category': nlp_result['risk_category']
        })

        detailed_results.append({
            'scenario': scenario,
            'keyword_result': keyword_result,
            'nlp_result': nlp_result
        })

        print(f"\n✓ Keyword-Only Score: {keyword_result['overall_risk_score']:.1f} ({keyword_result['risk_category']})")
        print(f"✓ NLP-Enhanced Score: {nlp_result['overall_risk_score']:.1f} ({nlp_result['risk_category']})")

    # Print comparison table
    print_comparison_table(comparison_results)

    # Print detailed analysis for each scenario
    for detail in detailed_results:
        print_detailed_analysis(
            detail['scenario'],
            detail['keyword_result'],
            detail['nlp_result']
        )

    # Summary
    print_section("Summary")

    total_scenarios = len(comparison_results)
    score_differences = [r['nlp_score'] - r['keyword_score'] for r in comparison_results]
    avg_diff = sum(score_differences) / len(score_differences) if score_differences else 0

    category_changes = sum(1 for r in comparison_results if r['keyword_category'] != r['nlp_category'])

    print(f"\nTotal Scenarios Tested: {total_scenarios}")
    print(f"Average Score Difference: {avg_diff:+.1f}")
    print(f"Category Changes: {category_changes}")

    print(f"\n{'Method':<20} {'Characteristics'}")
    print("-" * 70)
    print(f"{'Keyword-Only':<20} Prone to false positives, no context understanding")
    print(f"{'NLP-Enhanced':<20} Semantic analysis, allegation vs conviction detection")

    print("\n" + "=" * 70)
    print("  Test Complete")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(run_comparison_tests())

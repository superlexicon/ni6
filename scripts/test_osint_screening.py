"""
Test script for OSINT Background Screening Service.

Tests the complete OSINT screening flow:
- Web search (DuckDuckGo + Google scraper)
- Sanctions checking (OFAC, EU, UN)
- Country-specific public records (if country is supported)
- Risk scoring
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.osint_screening_service import osint_screening_service
from app.config.osint_config import osint_settings


def initialize_sanctions_checker():
    """Initialize sanctions checker for test script usage."""
    from app.services.osint.sanctions.sanctions_list_checker import SanctionsListChecker
    import app.services.osint.sanctions.sanctions_list_checker as checker_module
    from app.repositories import crime_repository

    try:
        # Syncing is now handled by separate OSSPEP Quarkus service
        # This application only reads from database
        sync_service = None

        # Create and assign the sanctions_checker instance
        sanctions_checker_instance = SanctionsListChecker(
            sanctions_repository=crime_repository,
            sync_service=sync_service
        )
        checker_module.sanctions_checker = sanctions_checker_instance

        # IMPORTANT: Also update the service's reference since it was already initialized
        osint_screening_service.sanctions_checker = sanctions_checker_instance

        print("[INFO] Sanctions checker initialized successfully")
        return True
    except Exception as e:
        print(f"[WARNING] Failed to initialize sanctions checker: {e}")
        import traceback
        traceback.print_exc()
        return False


def initialize_pep_checker():
    """Initialize PEP checker for test script usage."""
    from app.services.osint.pep.pep_checker import PEPChecker
    import app.services.osint.pep.pep_checker as pep_checker_module
    from app.repositories import pep_repository

    try:
        # Syncing is now handled by separate OSSPEP Quarkus service
        # This application only reads from database
        sync_service = None

        # Create and assign the pep_checker instance
        pep_checker_instance = PEPChecker(
            pep_repository=pep_repository,
            sync_service=sync_service
        )
        pep_checker_module.pep_checker = pep_checker_instance

        # IMPORTANT: Also update the service's reference since it was already initialized
        osint_screening_service.pep_checker = pep_checker_instance

        print("[INFO] PEP checker initialized successfully")
        return True
    except Exception as e:
        print(f"[WARNING] Failed to initialize PEP checker: {e}")
        import traceback
        traceback.print_exc()
        return False


def initialize_nlp_service():
    """Initialize NLP service for test script usage."""
    from app.services.nlp.nlp_service import nlp_service

    try:
        # Use asyncio to run the async initialization
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is running, create a task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, nlp_service.initialize())
                future.result()
        else:
            asyncio.run(nlp_service.initialize())

        print("[INFO] NLP service initialized successfully")
        return True
    except Exception as e:
        print(f"[WARNING] Failed to initialize NLP service: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(key: str, value: any, indent: int = 0):
    """Print a formatted result."""
    prefix = "  " * indent
    if isinstance(value, dict):
        print(f"{prefix}{key}:")
        for k, v in value.items():
            print_result(k, v, indent + 1)
    elif isinstance(value, list):
        print(f"{prefix}{key}:")
        for item in value:
            if isinstance(item, dict):
                for k, v in item.items():
                    print_result(k, v, indent + 1)
            else:
                print(f"{prefix}  - {item}")
    else:
        print(f"{prefix}{key}: {value}")


async def test_osint_screening(test_case_index: int = 1, custom_data: dict = None, run_all_risk_profiles: bool = False, run_medium_risk: bool = False):
    """Test OSINT screening with sample data or custom parameters."""
    load_dotenv()

    # Initialize services for test script usage
    initialize_sanctions_checker()
    initialize_pep_checker()
    initialize_nlp_service()

    print_section("OSINT Background Screening Test")

    # High-risk profile test cases (PEP, Sanctioned, Criminal)
    high_risk_test_cases = [
        {
            "name": "HIGH RISK: PEP (Regional Politician)",
            "description": "Test screening against a mid-level politically exposed person",
            "data": {
                "full_name": "Teodoro Nguema Obiang Mangue",
                "country": "Equatorial Guinea",
            }
        },
        {
            "name": "HIGH RISK: Sanctions (Oligarch)",
            "description": "Test screening against a sanctioned business figure",
            "data": {
                "full_name": "Oleg Deripaska",
                "country": "Russia",
            }
        },
        {
            "name": "HIGH RISK: Criminal (Ponzi Schemer)",
            "description": "Test screening for white-collar criminal/fraud indicators",
            "data": {
                "full_name": "Scott Rothstein",
                "country": "United States",
                "date_of_birth": "1962-07-10"
            }
        },
        {
            "name": "HIGH RISK: PEP (Government Minister)",
            "description": "Test screening against a government official under sanctions",
            "data": {
                "full_name": "Nicolás Maduro",
                "country": "Venezuela",
            }
        },
        {
            "name": "HIGH RISK: Sanctions (Designated Individual)",
            "description": "Test screening against OFAC designated individual",
            "data": {
                "full_name": "Slobodan Tesic",
                "country": "Serbia",
            }
        },
        {
            "name": "HIGH RISK: Criminal (Corruption/Fraud)",
            "description": "Test screening for corruption and fraud indicators",
            "data": {
                "full_name": "Eduardo Cunha",
                "country": "Brazil",
                "date_of_birth": "1958-09-20"
            }
        }
    ]

    # Medium-risk profile test cases (Business controversies, civil issues, regulatory problems)
    medium_risk_test_cases = [
        {
            "name": "MEDIUM RISK: Business Fraud (Unconvicted)",
            "description": "Test screening for business fraud allegations (civil case)",
            "data": {
                "full_name": "Elizabeth Holmes",
                "country": "United States",
                "date_of_birth": "1984-02-03"
            }
        },
        {
            "name": "MEDIUM RISK: Regulatory Violations",
            "description": "Test screening for financial misconduct/regulatory issues",
            "data": {
                "full_name": "Martin Shkreli",
                "country": "United States",
                "date_of_birth": "1983-03-17"
            }
        },
        {
            "name": "MEDIUM RISK: Civil Fraud Settlement",
            "description": "Test screening for civil fraud and business controversies",
            "data": {
                "full_name": "Donald Trump",
                "country": "United States",
                "date_of_birth": "1946-06-14"
            }
        }
    ]

    # Country-specific test cases
    country_test_cases = [
        {
            "name": "India Test (MCA)",
            "data": {
                "full_name": "Rajesh Kumar",
                "date_of_birth": "1975-05-20",
                "country": "India",
                "address": "Mumbai, Maharashtra"
            }
        },
        {
            "name": "Malaysia Test (SSM)",
            "data": {
                "full_name": "Ahmad Abdullah",
                "date_of_birth": "1985-08-10",
                "country": "Malaysia",
                "address": "Kuala Lumpur"
            }
        },
        {
            "name": "Thailand Test (DBD)",
            "data": {
                "full_name": "Somchai Wong",
                "date_of_birth": "1990-03-25",
                "country": "Thailand",
                "address": "Bangkok"
            }
        },
        {
            "name": "Indonesia Test (AHU)",
            "data": {
                "full_name": "Budi Santoso",
                "date_of_birth": "1988-11-05",
                "country": "Indonesia",
                "address": "Jakarta"
            }
        },
        {
            "name": "Philippines Test (SEC)",
            "data": {
                "full_name": "Jose Reyes",
                "date_of_birth": "1982-07-12",
                "country": "Philippines",
                "address": "Manila"
            }
        }
    ]

    # Combine all test cases
    all_test_cases = high_risk_test_cases + medium_risk_test_cases + country_test_cases

    # Use custom data if provided
    if custom_data:
        test_cases_to_run = [{
            "name": "Custom Screening",
            "data": custom_data
        }]
    elif run_medium_risk:
        # Run only the medium-risk profile test cases
        test_cases_to_run = medium_risk_test_cases
    elif run_all_risk_profiles:
        # Run only the high-risk profile test cases
        test_cases_to_run = high_risk_test_cases
    else:
        # Select specific test case (1-indexed)
        idx = test_case_index - 1
        if 0 <= idx < len(all_test_cases):
            selected_case = all_test_cases[idx]
        else:
            selected_case = all_test_cases[0]
        test_cases_to_run = [selected_case]

    # Run each test case
    for i, selected_case in enumerate(test_cases_to_run):
        if len(test_cases_to_run) > 1:
            print(f"\n{'='*60}")
            print(f"Test {i+1}/{len(test_cases_to_run)}: {selected_case['name']}")
            print(f"{'='*60}")
        else:
            print(f"\nRunning: {selected_case['name']}")

        if selected_case.get('description'):
            print(f"Description: {selected_case['description']}")

        print(f"Name: {selected_case['data']['full_name']}")
        if selected_case['data'].get('country'):
            print(f"Country: {selected_case['data']['country']}")
        if selected_case['data'].get('date_of_birth'):
            print(f"Date of Birth: {selected_case['data']['date_of_birth']}")
        if selected_case['data'].get('address'):
            print(f"Address: {selected_case['data']['address']}")

        # Perform screening
        print_section("Starting OSINT Screening...")

        result = await osint_screening_service.screen_individual(
            full_name=selected_case['data']['full_name'],
            date_of_birth=selected_case['data'].get('date_of_birth'),
            country=selected_case['data'].get('country'),
            address=selected_case['data'].get('address')
        )

        # Display results for this test case
        print_section("Screening Results")

        print_result("Screening ID", result.get('screening_id'))
        # NEW: Show binary PASS/FAIL result
        result_value = result.get('result')
        if result_value:
            print_result("Result", result_value)
        reason_value = result.get('reason')
        if reason_value:
            print_result("Reason", reason_value)
        print_result("Match Found", result.get('is_match'))
        print_result("Overall Risk Score", result.get('overall_risk_score'))
        print_result("Risk Category", result.get('risk_category'))
        print_result("Processing Time", f"{result.get('processing_time_seconds', 0):.2f} seconds")

        if result.get('error'):
            print_result("Error", result.get('error'))

        # Component scores
        print_section("Component Scores")
        component_scores = result.get('component_scores', {})
        for component, score in component_scores.items():
            print(f"  {component.replace('_', ' ').title()}: {score:.1f}/100")

        # Search sources summary
        print_section("Search Sources Summary")

        web_search = result.get('search_sources', {}).get('web_search', {})
        print_result("Web Search Results", web_search.get('results_count', 0))
        print_result("Negative News Count", web_search.get('negative_news_count', 0))

        sanctions = result.get('search_sources', {}).get('public_records', {}).get('sanctions', {})
        print_result("OFAC Match", sanctions.get('ofac_match', False))
        print_result("EU Sanctions Match", sanctions.get('eu_sanctions_match', False))
        print_result("UN Sanctions Match", sanctions.get('un_sanctions_match', False))

        country_records = result.get('raw_response', {}).get('country_specific', {})
        if country_records:
            print_result("Country", country_records.get('country'))
            print_result("Records Found", country_records.get('records_found', 0))

        # Match details
        if result.get('match_details'):
            print_section("Match Details")
            for detail in result.get('match_details', []):
                print(f"\n  Category: {detail.get('category')}")
                print(f"  Subcategory: {detail.get('subcategory')}")
                print(f"  Match Strength: {detail.get('match_strength')}")
                print(f"  Description: {detail.get('description')}")

        # Summary
        print_section("Summary")

        # NEW: Use binary result instead of score threshold
        result_value = result.get('result')
        reason_value = result.get('reason')
        risk_score = result.get('overall_risk_score', 0)
        risk_category = result.get('risk_category', 'UNKNOWN')

        print(f"\n  Final Risk Assessment:")
        if result_value:
            print(f"    Result: {result_value}")
        if reason_value:
            print(f"    Reason: {reason_value}")
        print(f"    Score: {risk_score:.1f}/100")
        print(f"    Category: {risk_category}")

        if result_value == "FAIL":
            print(f"    Action: REJECT")
        else:
            print(f"    Action: APPROVE")

    print("\n" + "=" * 60)
    print("  Test Complete")
    print("=" * 60 + "\n")


async def test_configuration():
    """Test configuration settings."""
    load_dotenv()

    print_section("Configuration Test")

    print("\nOSINT Settings:")
    print(f"  Risk Threshold: {osint_settings.risk_threshold}")
    print(f"  Enable Web Search: {osint_settings.enable_web_search}")
    print(f"  Enable Sanctions Check: {osint_settings.enable_sanctions_check}")
    print(f"  Cache Results Hours: {osint_settings.cache_results_hours}")
    print(f"  Headless Browser: {osint_settings.headless_browser}")
    print(f"  Browser Type: {osint_settings.browser_type}")
    print(f"  Min Delay Seconds: {osint_settings.min_delay_seconds}")
    print(f"  Max Delay Seconds: {osint_settings.max_delay_seconds}")
    print(f"  Rotate User Agents: {osint_settings.rotate_user_agents}")

    print("\nEnvironment:")
    screening_provider = os.getenv("SCREENING_PROVIDER", "worldcheck")
    print(f"  SCREENING_PROVIDER: {screening_provider}")

    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Test OSINT Background Screening Service',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run high-risk profile tests (PEP, Sanctions, Criminal) - DEFAULT
  python scripts/test_osint_screening.py

  # Run high-risk profiles explicitly
  python scripts/test_osint_screening.py --risk-profiles

  # Run medium-risk profiles (business controversies, civil issues)
  python scripts/test_osint_screening.py --medium-risk

  # Run custom screening with full name only
  python scripts/test_osint_screening.py --name "John Doe"

  # Run custom screening with all details
  python scripts/test_osint_screening.py --name "Jane Smith" --dob "1990-05-15" --country "United Kingdom" --address "London, UK"

  # Run specific predefined test case (1-14)
  python scripts/test_osint_screening.py --test-case 2
        """
    )
    parser.add_argument('--config-only', action='store_true', help='Only test configuration')
    parser.add_argument('--risk-profiles', action='store_true',
                        help='Run all high-risk profile tests (PEP, Sanctions, Criminal)')
    parser.add_argument('--medium-risk', action='store_true',
                        help='Run all medium-risk profile tests (business fraud, regulatory issues)')
    parser.add_argument('--test-case', type=int, choices=range(1, 15), default=1,
                        help='Specific test case (1-14). 1-6: High risk, 7-9: Medium risk, 10-14: Country-specific')

    # Custom screening parameters
    parser.add_argument('--name', '--full-name', dest='full_name', type=str,
                        help='Full name of the individual to screen')
    parser.add_argument('--dob', '--date-of-birth', dest='date_of_birth', type=str,
                        help='Date of birth (YYYY-MM-DD format)')
    parser.add_argument('--country', type=str,
                        help='Country name')
    parser.add_argument('--address', type=str,
                        help='Address information')

    args = parser.parse_args()

    if args.config_only:
        asyncio.run(test_configuration())
    elif args.full_name:
        # Build custom data dict from provided arguments
        custom_data = {'full_name': args.full_name}
        if args.date_of_birth:
            custom_data['date_of_birth'] = args.date_of_birth
        if args.country:
            custom_data['country'] = args.country
        if args.address:
            custom_data['address'] = args.address
        asyncio.run(test_osint_screening(custom_data=custom_data))
    elif args.medium_risk:
        # Run medium-risk profile tests
        asyncio.run(test_osint_screening(run_medium_risk=True))
    elif args.risk_profiles or not args.full_name and not args.medium_risk and args.test_case == 1:
        # Default: run all high-risk profile tests
        asyncio.run(test_osint_screening(run_all_risk_profiles=True))
    else:
        asyncio.run(test_osint_screening(args.test_case))

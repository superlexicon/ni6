#!/usr/bin/env python3
"""
Bank Prompt Review CLI Tool

Provides command-line interface for reviewing, managing, and regenerating
bank-specific GLiNER2 prompts.

Usage:
    poetry run python scripts/review_bank_prompts.py --list
    poetry run python scripts/review_bank_prompts.py --bank MAHB --country IN
    poetry run python scripts/review_bank_prompts.py --regenerate --bank MAHB --country IN
    poetry run python scripts/review_bank_prompts.py --stats
"""

import argparse
import asyncio
import json
import sys
from typing import Optional

# Add parent directory to path for imports
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.bank_prompt_database_service import bank_prompt_database_service
from app.services.bank_prompt_generator import bank_prompt_generator
from app.core.key_injection.bank_database_lookup import get_bank_database_lookup
from app.core.logger import get_logger

logger = get_logger()


def format_prompt_for_display(prompt: dict) -> str:
    """Format a prompt for display."""
    lines = [
        f"  Entity: {prompt['entity_type']}",
        f"    Description: {prompt['prompt_description'][:80]}...",
        f"    Category: {prompt['entity_category']}",
        f"    Threshold: {prompt['threshold']}",
        f"    Usage: {prompt['usage_count']} times",
    ]
    if prompt.get('examples'):
        try:
            examples = json.loads(prompt['examples']) if isinstance(prompt['examples'], str) else prompt['examples']
            if examples:
                lines.append(f"    Examples: {', '.join(str(e) for e in examples[:3])}")
        except:
            pass
    if prompt.get('validation_pattern'):
        lines.append(f"    Pattern: {prompt['validation_pattern']}")
    return '\n'.join(lines)


def list_all_banks_with_prompts():
    """List all banks that have prompts."""
    print("\n" + "=" * 80)
    print("BANKS WITH CUSTOM PROMPTS")
    print("=" * 80 + "\n")

    banks = bank_prompt_database_service.get_all_banks_with_prompts(is_active=True)

    if not banks:
        print("No banks with custom prompts found.")
        return

    print(f"Found {len(banks)} banks with custom prompts:\n")

    for bank in banks:
        print(f"  {bank['abbrev']} - {bank['full_name']} ({bank['country_code']})")
        print(f"    Prompts: {bank['prompt_count']}")
        print(f"    Total Usage: {bank['total_usage'] or 0}")
        if bank['last_used']:
            print(f"    Last Used: {bank['last_used']}")
        print()


def view_bank_prompts(bank_abbrev: str, country_code: str):
    """View prompts for a specific bank."""
    print("\n" + "=" * 80)
    print(f"PROMPTS FOR {bank_abbrev}/{country_code}")
    print("=" * 80 + "\n")

    # Get bank info
    bank_db = get_bank_database_lookup()
    bank_info = bank_db.lookup_by_name(bank_abbrev, country_code)

    if not bank_info:
        print(f"Error: Bank '{bank_abbrev}' not found in country '{country_code}'")
        return

    # Get prompts
    prompts = bank_prompt_database_service.get_bank_prompts(
        bank_id=bank_db.get_bank_by_name_and_country(bank_info.full_name, country_code)['bank_id'],
        country_code=country_code
    )

    if not prompts:
        print(f"No custom prompts found for {bank_abbrev}/{country_code}")
        return

    print(f"Bank: {bank_info.full_name}")
    print(f"Country: {country_code}")
    print(f"Default Threshold: {prompts.get('default_threshold', 0.3)}")
    print(f"Total Prompts: {len(prompts.get('prompts', {}))}\n")

    for entity_type, config in prompts.get('prompts', {}).items():
        print(f"  Entity: {entity_type}")
        print(f"    Description: {config.get('description', '')[:80]}...")
        print(f"    Category: {config.get('entity', 'custom')}")
        print(f"    Threshold: {config.get('threshold', 0.3)}")
        if config.get('examples'):
            print(f"    Examples: {', '.join(str(e) for e in config['examples'][:3])}")
        if config.get('pattern'):
            print(f"    Pattern: {config.get('pattern')}")
        print()


async def regenerate_prompts_for_bank(bank_abbrev: str, country_code: str, sample_text: Optional[str] = None):
    """Regenerate prompts for a specific bank."""
    print("\n" + "=" * 80)
    print(f"REGENERATING PROMPTS FOR {bank_abbrev}/{country_code}")
    print("=" * 80 + "\n")

    # Get bank info
    bank_db = get_bank_database_lookup()
    bank_info = bank_db.lookup_by_name(bank_abbrev, country_code)

    if not bank_info:
        print(f"Error: Bank '{bank_abbrev}' not found in country '{country_code}'")
        return

    # If sample text not provided, use a default
    if not sample_text:
        print("Warning: No sample text provided. Using generic sample.")
        sample_text = f"""
        {bank_info.full_name} Bank Statement
        Account Holder: JOHN DOE
        Account Number: 1234567890
        Statement Date: 01/01/2024
        """

    # Generate prompts
    print("Generating prompts with LLM...")
    generated = await bank_prompt_generator.generate_prompts_for_bank(
        bank_id=bank_db.get_bank_by_name_and_country(bank_info.full_name, country_code)['bank_id'],
        bank_abbrev=bank_abbrev,
        bank_name=bank_info.full_name,
        country_code=country_code,
        ocr_text=sample_text,
        generic_extraction_result=None
    )

    if generated.get('error'):
        print(f"Error generating prompts: {generated['error']}")
        return

    # Save to database
    print(f"Generated {len(generated['prompts'])} prompts. Saving to database...")
    save_success = bank_prompt_database_service.save_bank_prompts(
        bank_id=bank_db.get_bank_by_name_and_country(bank_info.full_name, country_code)['bank_id'],
        country_code=country_code,
        prompts=generated['prompts'],
        extraction_config=generated['extraction_config'],
        metadata=generated.get('metadata')
    )

    if save_success:
        print("Prompts saved successfully!")
        print(f"\nReasoning: {generated.get('metadata', {}).get('reasoning', 'N/A')}")
    else:
        print("Error: Failed to save prompts to database")


def show_statistics():
    """Show prompt statistics."""
    print("\n" + "=" * 80)
    print("PROMPT STATISTICS")
    print("=" * 80 + "\n")

    stats = bank_prompt_database_service.get_prompt_statistics()

    print("Overall Statistics:")
    print(f"  Total Prompts: {stats['total_prompts']}")
    print(f"  Total Usage: {stats['total_usage']}")
    print(f"  Average Usage: {stats['avg_usage']:.2f}")
    print(f"  Banks with Prompts: {stats['bank_count']}")
    print(f"  Countries Covered: {stats['country_count']}\n")

    print("Entity Type Breakdown:")
    for entity in stats.get('entity_breakdown', []):
        print(f"  {entity['entity_type']}: {entity['count']} prompts (avg threshold: {entity['avg_threshold']:.2f})")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Review and manage bank-specific GLiNER2 prompts"
    )

    # Actions
    parser.add_argument('--list', action='store_true', help='List all banks with custom prompts')
    parser.add_argument('--bank', type=str, help='Bank abbreviation (e.g., MAHB, HDFC)')
    parser.add_argument('--country', type=str, help='ISO country code (e.g., IN, AE, SG)')
    parser.add_argument('--regenerate', action='store_true', help='Regenerate prompts for the specified bank')
    parser.add_argument('--sample', type=str, help='Sample OCR text for prompt generation (use with --regenerate)')
    parser.add_argument('--stats', action='store_true', help='Show prompt statistics')

    args = parser.parse_args()

    # Validate arguments
    if args.regenerate and not (args.bank and args.country):
        print("Error: --regenerate requires --bank and --country")
        sys.exit(1)

    if (args.bank or args.country) and not (args.bank and args.country):
        print("Error: Both --bank and --country are required when viewing specific bank")
        sys.exit(1)

    # Execute action
    try:
        if args.list:
            list_all_banks_with_prompts()

        elif args.bank and args.country:
            if args.regenerate:
                asyncio.run(regenerate_prompts_for_bank(args.bank, args.country, args.sample))
            else:
                view_bank_prompts(args.bank, args.country)

        elif args.stats:
            show_statistics()

        else:
            parser.print_help()

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Bank Pattern Analysis Script

Analyzes sample bank statement PDFs to determine which pattern type
(bank name OR URL) works best for each bank.

This helps optimize the bank_identifiers_map in config.json by keeping
only the most effective pattern type for each bank.

Usage:
    poetry run python scripts/analyze_bank_patterns.py [--data-dir PATH]
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz  # PyMuPDF


def load_config() -> Dict:
    """Load bank statement configuration."""
    config_path = Path(__file__).parent.parent / "app" / "reference_templates" / "bank_statements" / "config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_text_from_pdf(pdf_path: str, max_pages: int = 1) -> str:
    """Extract text from PDF using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        text_parts = []

        pages_to_process = min(max_pages, doc.page_count)

        for page_num in range(pages_to_process):
            page = doc[page_num]
            text = page.get_text()
            text_parts.append(text)

        doc.close()
        return "\n".join(text_parts)
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""


def analyze_pdf_for_patterns(
    pdf_path: str,
    bank_identifiers_map: Dict[str, str]
) -> Tuple[Set[str], Set[str]]:
    """
    Analyze a PDF for bank name and URL patterns.

    Returns:
        Tuple of (matched_bank_names, matched_urls)
    """
    text = extract_text_from_pdf(pdf_path)
    text_upper = text.upper()

    matched_bank_names = set()
    matched_urls = set()

    # Check each identifier in the map
    for identifier, abbrev in bank_identifiers_map.items():
        identifier_upper = identifier.upper()

        # Check if identifier appears in text
        if identifier_upper in text_upper:
            # Determine if this is a URL or a name
            if identifier.startswith(("www.", "http://", "https://")):
                # This is a URL/domain pattern
                # Normalize for matching (remove www., http://, etc.)
                normalized = identifier_upper.replace("WWW.", "").replace("HTTP://", "").replace("HTTPS://", "")
                if normalized in text_upper.replace("WWW.", "").replace("HTTP://", "").replace("HTTPS://", ""):
                    matched_urls.add(abbrev)
            else:
                # This is a name pattern
                matched_bank_names.add(abbrev)

    return matched_bank_names, matched_urls


def find_sample_pdfs(data_dir: Path) -> Dict[str, List[Path]]:
    """
    Find sample bank statement PDFs organized by bank.

    Expected directory structure:
        data_dir/
            dbs/
                sample1.pdf
                sample2.pdf
            hdfc/
                sample1.pdf
            ...

    Returns:
        Dict mapping bank abbreviation to list of PDF paths
    """
    bank_samples = defaultdict(list)

    if not data_dir.exists():
        print(f"Warning: Data directory {data_dir} does not exist")
        return bank_samples

    # Look for subdirectories named after banks
    for subdir in data_dir.iterdir():
        if subdir.is_dir():
            bank_name = subdir.name.lower()

            # Find PDF files in this directory
            pdf_files = list(subdir.glob("*.pdf"))
            bank_samples[bank_name] = pdf_files

    return bank_samples


def analyze_bank_patterns(
    data_dir: Path,
    bank_identifiers_map: Dict[str, str]
) -> Dict[str, Dict]:
    """
    Analyze patterns for all banks.

    Returns:
        Dict mapping bank abbreviation to analysis results:
        {
            "bank_abbrev": {
                "name_patterns_found": int,
                "url_patterns_found": int,
                "recommended_pattern": "name" | "url",
                "confidence": float
            }
        }
    """
    results = {}
    bank_samples = find_sample_pdfs(data_dir)

    # Group identifiers by bank
    bank_to_identifiers = defaultdict(lambda: {"names": [], "urls": []})
    for identifier, abbrev in bank_identifiers_map.items():
        if identifier.startswith(("www.", "http://", "https://")):
            bank_to_identifiers[abbrev]["urls"].append(identifier)
        else:
            bank_to_identifiers[abbrev]["names"].append(identifier)

    # Analyze each bank
    for bank_abbrev, identifiers in bank_to_identifiers.items():
        name_patterns = identifiers["names"]
        url_patterns = identifiers["urls"]

        # Get sample PDFs for this bank
        pdf_files = bank_samples.get(bank_abbrev.lower(), [])

        if not pdf_files:
            # No samples found, use heuristics
            results[bank_abbrev] = {
                "name_patterns_count": len(name_patterns),
                "url_patterns_count": len(url_patterns),
                "name_patterns_found": 0,
                "url_patterns_found": 0,
                "samples_analyzed": 0,
                "recommended_pattern": "name" if len(name_patterns) >= len(url_patterns) else "url",
                "confidence": 0.0,
                "reason": "No sample PDFs available for analysis"
            }
            continue

        # Analyze samples
        name_match_count = 0
        url_match_count = 0

        for pdf_path in pdf_files:
            matched_names, matched_urls = analyze_pdf_for_patterns(pdf_path, bank_identifiers_map)

            if bank_abbrev in matched_names:
                name_match_count += 1
            if bank_abbrev in matched_urls:
                url_match_count += 1

        total_samples = len(pdf_files)
        name_success_rate = name_match_count / total_samples if total_samples > 0 else 0
        url_success_rate = url_match_count / total_samples if total_samples > 0 else 0

        # Recommend the pattern type with higher success rate
        if name_success_rate > url_success_rate:
            recommended = "name"
            confidence = name_success_rate
        elif url_success_rate > name_success_rate:
            recommended = "url"
            confidence = url_success_rate
        else:
            # Equal success rates, prefer name (more explicit)
            recommended = "name"
            confidence = name_success_rate

        results[bank_abbrev] = {
            "name_patterns_count": len(name_patterns),
            "url_patterns_count": len(url_patterns),
            "name_patterns_found": name_match_count,
            "url_patterns_found": url_match_count,
            "samples_analyzed": total_samples,
            "recommended_pattern": recommended,
            "confidence": confidence,
            "reason": f"Based on analysis of {total_samples} sample(s)"
        }

    return results


def print_analysis_results(results: Dict[str, Dict], bank_identifiers_map: Dict[str, str]) -> None:
    """Print analysis results in a readable format."""
    print("\n" + "=" * 100)
    print("BANK PATTERN ANALYSIS RESULTS")
    print("=" * 100)

    # Sort by bank abbreviation
    sorted_banks = sorted(results.items(), key=lambda x: x[0])

    print(f"\n{'Bank':<10} {'Name Pats':<10} {'URL Pats':<10} {'Name %':<10} {'URL %':<10} {'Recommend':<12} {'Confidence':<12} {'Samples':<10} {'Reason'}")
    print("-" * 100)

    for bank_abbrev, analysis in sorted_banks:
        name_pats = analysis["name_patterns_count"]
        url_pats = analysis["url_patterns_count"]
        name_found = analysis["name_patterns_found"]
        url_found = analysis["url_patterns_found"]
        samples = analysis["samples_analyzed"]

        if samples > 0:
            name_pct = (name_found / samples) * 100
            url_pct = (url_found / samples) * 100
        else:
            name_pct = 0
            url_pct = 0

        recommend = analysis["recommended_pattern"].upper()
        confidence = f"{analysis['confidence']:.1%}" if analysis['confidence'] > 0 else "N/A"
        reason = analysis["reason"][:30] if len(analysis["reason"]) > 30 else analysis["reason"]

        print(f"{bank_abbrev:<10} {name_pats:<10} {url_pats:<10} {name_pct:<9.1f}% {url_pct:<9.1f}% {recommend:<12} {confidence:<12} {samples:<10} {reason}")

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    # Count recommendations
    name_recommended = sum(1 for a in results.values() if a["recommended_pattern"] == "name")
    url_recommended = sum(1 for a in results.values() if a["recommended_pattern"] == "url")
    total_banks = len(results)

    print(f"\nTotal banks analyzed: {total_banks}")
    print(f"Recommended to use NAME patterns: {name_recommended} ({name_recommended/total_banks*100:.1f}%)")
    print(f"Recommended to use URL patterns: {url_recommended} ({url_recommended/total_banks*100:.1f}%)")

    # Banks with no samples
    no_samples = [bank for bank, analysis in results.items() if analysis["samples_analyzed"] == 0]
    if no_samples:
        print(f"\nBanks with NO sample PDFs analyzed ({len(no_samples)}):")
        print(", ".join(sorted(no_samples)))
        print("\n⚠️  These banks need sample PDFs for accurate pattern recommendation!")

    # Print pattern cleanup recommendations
    print("\n" + "=" * 100)
    print("PATTERN CLEANUP RECOMMENDATIONS")
    print("=" * 100)

    print("\nFor each bank, keep ONLY the recommended pattern type:")
    print("- For 'name' recommendations: Keep name patterns, remove URL patterns")
    print("- For 'url' recommendations: Keep URL patterns, remove name patterns")

    # Group identifiers by bank
    bank_to_identifiers = defaultdict(lambda: {"names": [], "urls": []})
    for identifier, abbrev in bank_identifiers_map.items():
        if identifier.startswith(("www.", "http://", "https://")):
            bank_to_identifiers[abbrev]["urls"].append(identifier)
        else:
            bank_to_identifiers[abbrev]["names"].append(identifier)

    print("\nRecommended removals (add to removals list in config.json):")
    print("-" * 80)

    removal_count = 0
    for bank_abbrev, analysis in results.items():
        if analysis["samples_analyzed"] == 0:
            continue  # Skip banks without samples

        recommended = analysis["recommended_pattern"]
        identifiers = bank_to_identifiers.get(bank_abbrev, {"names": [], "urls": []})

        if recommended == "name":
            # Remove URL patterns
            for url in identifiers["urls"]:
                print(f'  "{url}": "{bank_abbrev}",  # Remove: Use name patterns instead')
                removal_count += 1
        else:
            # Remove name patterns
            for name in identifiers["names"]:
                print(f'  "{name}": "{bank_abbrev}",  # Remove: Use URL patterns instead')
                removal_count += 1

    if removal_count == 0:
        print("  (No removals recommended - all patterns are optimal)")
    else:
        print(f"\nTotal recommended removals: {removal_count}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze bank statement PDFs to determine optimal pattern types"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="scripts/test_data/bank_statements",
        help="Directory containing sample bank statement PDFs organized by bank"
    )

    args = parser.parse_args()

    # Load configuration
    print("Loading bank configuration...")
    config = load_config()
    bank_identifiers_map = config.get("bank_identifiers_map", {})

    print(f"Loaded {len(bank_identifiers_map)} bank identifiers")

    # Analyze patterns
    data_dir = Path(args.data_dir)
    print(f"\nAnalyzing PDFs in: {data_dir}")

    results = analyze_bank_patterns(data_dir, bank_identifiers_map)

    # Print results
    print_analysis_results(results, bank_identifiers_map)

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()

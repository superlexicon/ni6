#!/usr/bin/env python3
"""
Download all ML/DL model weights for the OSINT application from Google Drive zip files.

The zip files contain folders that are copied to the correct app directories:
- deepface → app/deepface_weights
- finbert → app/finbert_weights
- nltk → app/nltk_data
- photoholmes → app/photoholmes/weights
- spacy → app/spacy_models

Usage:
    python scripts/download_model_weights.py                    # Download all
    python scripts/download_model_weights.py --dry-run         # Show what would be downloaded
    python scripts/download_model_weights.py --force           # Re-download existing files
"""

import argparse
import logging
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import List

import gdown

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

# Google Drive file IDs for the 5 zip files containing all model weights
# Using direct download URL format for large files
MODELS_ZIP_URLS = [
    os.environ.get("MODELS_ZIP_URL_1", "https://drive.google.com/uc?id=1gSGwguV4ItxNcuXz2row5ev85Yo5fxnH"),
    os.environ.get("MODELS_ZIP_URL_2", "https://drive.google.com/uc?id=1CdDaVQ7JXa6Pyem9tQjetHDHqEDkeudk"),
    os.environ.get("MODELS_ZIP_URL_3", "https://drive.google.com/uc?id=1-z64KGwN8RNCixMHAS4f-MSfwwx5JJ1L"),
    os.environ.get("MODELS_ZIP_URL_4", "https://drive.google.com/uc?id=1Z7OGBDWYhLJlh7lNnF8hZH4ZxMe3j-Jg"),
    os.environ.get("MODELS_ZIP_URL_5", "https://drive.google.com/uc?id=1Y5HivG9Eu6pZTQ4LtJjDN02OCTa2vpgm"),
]

# Zip file extracts to this root folder
ZIP_ROOT_FOLDER = "ni6-models"

# Mapping from folder names in the zip to target directories in the app
# Format: "zip_folder_name": "app/target_directory"
FOLDER_MAPPING = {
    "deepface": "app/deepface_weights",
    "finbert": "app/finbert_weights",
    "nltk": "app/nltk_data",
    "photoholmes": "app/photoholmes/weights",
    "spacy": "app/spacy_models",
}

# ============================================================================
# Utility Functions
# ============================================================================

def log_info(msg: str):
    """Log info message."""
    logger.info(f"INFO: {msg}")

def log_success(msg: str):
    """Log success message."""
    logger.info(f"✓ {msg}")

def log_error(msg: str):
    """Log error message."""
    logger.error(f"✗ {msg}")

def log_warning(msg: str):
    """Log warning message."""
    logger.warning(f"⚠ {msg}")

def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent

# ============================================================================
# Download and Extract Functions
# ============================================================================

def download_and_extract_models(
    zip_urls: List[str],
    project_root: Path,
    force: bool = False,
    dry_run: bool = False
) -> dict:
    """Download the models zip files and copy folders to correct locations.

    Args:
        zip_urls: List of Google Drive URLs for the zip files
        project_root: Project root directory
        force: Force re-download even if files exist
        dry_run: If True, only show what would be done

    Returns:
        Dict mapping folder names to success status
    """
    results = {}

    # Check if URLs are configured
    if any("YOUR_FILE_ID" in url for url in zip_urls):
        log_error("Please update MODELS_ZIP_URLS with the actual Google Drive URLs")
        log_error("Either edit the script or set MODELS_ZIP_URL_1 through MODELS_ZIP_URL_5 environment variables")
        return results

    # Download to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()

        if dry_run:
            log_info("Dry run mode - would download and extract:")
            for i, url in enumerate(zip_urls, 1):
                log_info(f"  Part {i}: {url}")
            for folder, target in FOLDER_MAPPING.items():
                target_dir = project_root / target
                exists = "✓ Exists" if target_dir.exists() and any(target_dir.iterdir()) else "→ Would copy"
                log_info(f"  {folder} → {target} [{exists}]")
            return {k: True for k in FOLDER_MAPPING.keys()}

        # Download all zip files
        log_info(f"Downloading {len(zip_urls)} zip file(s)...")
        for i, zip_url in enumerate(zip_urls, 1):
            zip_path = tmp_path / f"models_part{i}.zip"

            log_info(f"  Part {i}/{len(zip_urls)}: {zip_url}")

            try:
                gdown.download(
                    zip_url,
                    output=str(zip_path),
                    quiet=False,
                    fuzzy=True,
                    use_cookies=False
                )
                if not zip_path.exists():
                    log_error(f"    Failed to download part {i}")
                    return results
                log_success(f"    Part {i} downloaded")

                # Extract this zip file
                log_info(f"    Extracting part {i}...")
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)
                except Exception as e:
                    log_error(f"    Failed to extract part {i}: {e}")
                    return results

            except Exception as e:
                log_error(f"    Failed to download part {i}: {e}")
                return results

        log_success("All parts downloaded and extracted")

        # List what was extracted
        extracted_items = list(extract_dir.iterdir())
        log_info(f"Extracted {len(extracted_items)} top-level item(s)")

        # Copy folders to their target locations
        for folder_name, target_rel_path in FOLDER_MAPPING.items():
            source_dir = extract_dir / ZIP_ROOT_FOLDER / folder_name
            target_dir = project_root / target_rel_path

            # Check if source exists
            if not source_dir.exists():
                log_warning(f"  {folder_name}: Not found in zip (skipped)")
                results[folder_name] = False
                continue

            # Check if already exists
            if not force and target_dir.exists() and any(target_dir.iterdir()):
                log_info(f"  {folder_name}: Already exists at {target_rel_path} (skipped)")
                results[folder_name] = True
                continue

            # Create target directory
            target_dir.mkdir(parents=True, exist_ok=True)

            # Copy contents
            log_info(f"  {folder_name}: Copying to {target_rel_path}...")
            try:
                # Remove existing contents if force
                if force and target_dir.exists():
                    for item in target_dir.iterdir():
                        if item.is_dir():
                            shutil.rmtree(item)
                        else:
                            item.unlink()

                # Copy all contents
                for item in source_dir.iterdir():
                    dest = target_dir / item.name
                    if item.is_dir():
                        shutil.copytree(item, dest, dirs_exist_ok=True)
                    else:
                        shutil.copy2(item, dest)

                # Count files
                file_count = len(list(target_dir.rglob("*")))
                log_success(f"  {folder_name}: Copied {file_count} file(s)")
                results[folder_name] = True

            except Exception as e:
                log_error(f"  {folder_name}: Failed to copy - {e}")
                results[folder_name] = False

    return results


# ============================================================================
# CLI Interface
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download ML/DL model weights for the OSINT application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all weights
  python scripts/download_model_weights.py

  # Dry run to see what would be downloaded
  python scripts/download_model_weights.py --dry-run

  # Force re-download existing files
  python scripts/download_model_weights.py --force

  # Provide URLs via command line
  python scripts/download_model_weights.py --url URL1 --url URL2 --url URL3 --url URL4 --url URL5

Environment Variables:
  MODELS_ZIP_URL_1 through MODELS_ZIP_URL_5
                Google Drive URLs for the 5 model zip files
                (can also be set by editing the script)
        """
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without downloading"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist"
    )
    parser.add_argument(
        "--url",
        type=str,
        action="append",
        help="Override zip file URLs (can be specified multiple times)"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Show banner
    logger.info("=" * 60)
    logger.info("Model Weights Download Script")
    logger.info("OSINT Application - ML/DL Model Weights")
    logger.info("=" * 60)
    logger.info("")

    # Get URLs (command line overrides environment variables)
    if args.url:
        zip_urls = args.url
    else:
        zip_urls = MODELS_ZIP_URLS

    # Get project root
    project_root = get_project_root()

    # Download and extract
    results = download_and_extract_models(
        zip_urls=zip_urls,
        project_root=project_root,
        force=args.force,
        dry_run=args.dry_run
    )

    # Show summary
    if not args.dry_run and results:
        logger.info("")
        logger.info("=" * 60)
        success_count = sum(1 for v in results.values() if v)
        total_count = len(results)

        if success_count == total_count:
            logger.info(f"✓ All {total_count} models downloaded successfully!")
        else:
            failed = [k for k, v in results.items() if not v]
            logger.info(f"⚠ {success_count}/{total_count} succeeded. Failed: {', '.join(failed)}")
        logger.info("=" * 60)

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

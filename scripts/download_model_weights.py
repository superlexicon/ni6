#!/usr/bin/env python3
"""
Download all ML/DL model weights for the OSINT application from Google Drive zip files.

The zip files contain folders that are copied to the correct app directories:
- deepface → app/deepface/.deepface/weights
- doctr → app/models/hub/checkpoints
- finbert → app/models/finbert
- gliner2 → app/models/gliner2
- mediapipe → app/models/mediapipe
- nltk → app/models/nltk
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
    os.environ.get("MODELS_ZIP_URL_1", "https://drive.google.com/uc?export=download&id=16M5puSMn4o7ahwYHYTIODy42k2VB4D_H"),
    os.environ.get("MODELS_ZIP_URL_2", "https://drive.google.com/uc?export=download&id=1hoIUbFn3TjjAJSk05W-Ia9o42LG-tlKp"),
    os.environ.get("MODELS_ZIP_URL_3", "https://drive.google.com/uc?export=download&id=1fcW9Q9frxx8J_vtMaHjHwKt5Lei8ahqX"),
    os.environ.get("MODELS_ZIP_URL_4", "https://drive.google.com/uc?export=download&id=1CQbhGBJVyeNoWAVO2QUXH0FdfmEu7UQr"),
    os.environ.get("MODELS_ZIP_URL_5", "https://drive.google.com/uc?export=download&id=1TKtfYVE6FRvvFA_LIfGAwSKrfkn8NPse"),
]

# Zip file extracts to this root folder
ZIP_ROOT_FOLDER = "ni6-models"

# Mapping from folder names in the zip to target directories in the app
# Format: "zip_folder_name": "app/target_directory"
FOLDER_MAPPING = {
    "deepface": "app/deepface/.deepface/weights",
    "doctr": "app/models/hub/checkpoints",
    "finbert": "app/models/finbert",
    "gliner2": "app/models/gliner2",
    "mediapipe": "app/models/mediapipe",
    "nltk": "app/models/nltk",
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
                # Use Python's requests library to handle Google Drive downloads
                # This handles the virus scan warning page automatically
                import requests
                import io

                log_info(f"    Downloading with requests library...")

                # Use requests to download, handling redirects automatically
                # For large files, stream the download
                response = requests.get(zip_url, stream=True, timeout=300)

                # Check if we got the virus scan warning page
                if "virus scan warning" in response.text.lower() or response.url.startswith("https://drive.google.com/"):
                    # Parse HTML to get the confirm link
                    import re
                    uuid_match = re.search(r'name="uuid" value="([^"]+)"', response.text)
                    file_id = zip_url.split("id=")[1].split("&")[0]

                    if uuid_match:
                        uuid_value = uuid_match.group(1)
                        # Construct the direct download URL
                        download_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t&uuid={uuid_value}"
                        response = requests.get(download_url, stream=True, timeout=300)

                # Download with progress indicator
                downloaded_size = 0
                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)

                if not zip_path.exists() or zip_path.stat().st_size < 1000:
                    log_error(f"    Failed to download part {i} (file too small: {zip_path.stat().st_size if zip_path.exists() else 0} bytes)")
                    return results
                log_success(f"    Part {i} downloaded ({zip_path.stat().st_size / (1024*1024):.1f} MB)")

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

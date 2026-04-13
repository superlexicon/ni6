#!/usr/bin/env python3
"""
Download all ML/DL model weights using their native library download mechanisms.

This script downloads weights directly from the official sources for each library:
- DeepFace → Google Drive via deepface library (~500 MB)
- DocTR → PyTorch hub (~200 MB)
- FinBERT → HuggingFace Hub (~500 MB)
- GLiNER2 → HuggingFace Hub (~700 MB)
- MediaPipe → Google Cloud Storage (~30 MB)
- NLTK → NLTK data repository (~5 MB)
- PhotoHolmes → GitHub releases (~150 MB)
- spaCy → spaCy models (~50 MB)
- InsightFace → InsightFace servers (~435 MB)

Total size: ~2.5 GB

Usage:
    python scripts/download_model_weights.py                    # Download all
    python scripts/download_model_weights.py --deepface         # Download only DeepFace
    python scripts/download_model_weights.py --dry-run         # Show what would be downloaded
    python scripts/download_model_weights.py --force           # Re-download existing files
"""

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any
import shutil

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

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
APP_DIR = PROJECT_ROOT / "app"

# Model directories
DEEPFACE_TARGET = APP_DIR / "deepface" / ".deepface" / "weights"
DOCTR_TARGET = APP_DIR / "models" / "doctr" / "models"
FINBERT_TARGET = APP_DIR / "models" / "finbert"
GLINER2_TARGET = APP_DIR / "models" / "gliner2"
MEDIAPIPE_TARGET = APP_DIR / "models" / "mediapipe"
NLTK_TARGET = APP_DIR / "models" / "nltk"
PHOTOHOLMES_TARGET = APP_DIR / "photoholmes" / "weights"
SPACY_TARGET = APP_DIR / "models" / "spacy"
INSIGHTFACE_TARGET = APP_DIR / "models" / "insightface"

# HuggingFace cache directory
HF_CACHE_DIR = APP_DIR / ".cache" / "huggingface"


# ============================================================================
# Utility Functions
# ============================================================================

def log_info(msg: str):
    logger.info(f"INFO: {msg}")

def log_success(msg: str):
    logger.info(f"✓ {msg}")

def log_error(msg: str):
    logger.error(f"✗ {msg}")

def log_warning(msg: str):
    logger.warning(f"⚠ {msg}")

def log_section(title: str):
    logger.info("")
    logger.info("=" * 60)
    logger.info(f" {title}")
    logger.info("=" * 60)


def check_exists(path: Path, name: str) -> bool:
    """Check if a directory exists and has content."""
    if path.exists() and any(path.iterdir()):
        return True
    return False


# ============================================================================
# DeepFace Download
# ============================================================================

def download_deepface(force: bool = False, dry_run: bool = False) -> bool:
    """
    Download DeepFace weights using the library's native download mechanism.

    DeepFace downloads from Google Drive URLs defined in the library.
    We set DEEPFACE_HOME to our custom target directory.
    """
    log_section("DeepFace")

    if not force and check_exists(DEEPFACE_TARGET, "DeepFace"):
        log_info(f"Already exists at {DEEPFACE_TARGET.relative_to(PROJECT_ROOT)} (skipped)")
        return True

    if dry_run:
        log_info(f"Would download DeepFace weights to {DEEPFACE_TARGET.relative_to(PROJECT_ROOT)}")
        return True

    try:
        # Set DeepFace home to our custom location
        deepface_home = APP_DIR / "deepface"
        os.environ['DEEPFACE_HOME'] = str(deepface_home)

        # Import and trigger download for each model
        from app.deepface.DeepFace import build_model as build_deepface_model

        models_to_download = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace', 'Dlib', 'SFace']

        for model_name in models_to_download:
            log_info(f"  Downloading {model_name} weights...")
            try:
                build_deepface_model(model_name)
                log_success(f"  {model_name} downloaded")
            except Exception as e:
                log_warning(f"  {model_name}: {str(e)}")
                # Continue with next model

        # Verify files were downloaded
        if check_exists(DEEPFACE_TARGET, "DeepFace"):
            file_count = len(list(DEEPFACE_TARGET.rglob("*")))
            total_size = sum(f.stat().st_size for f in DEEPFACE_TARGET.rglob("*")) / (1024 * 1024)
            log_success(f"Downloaded {file_count} file(s) ({total_size:.1f} MB)")
            return True
        else:
            log_error("Failed to download DeepFace weights")
            return False

    except Exception as e:
        log_error(f"Failed to download DeepFace weights: {e}")
        return False


# ============================================================================
# DocTR Download
# ============================================================================

def download_doctr(force: bool = False, dry_run: bool = False) -> bool:
    """
    Download DocTR weights using PyTorch hub.

    DocTR uses TORCH_HOME environment variable for custom cache location.
    """
    log_section("DocTR")

    if not force and check_exists(DOCTR_TARGET, "DocTR"):
        log_info(f"Already exists at {DOCTR_TARGET.relative_to(PROJECT_ROOT)} (skipped)")
        return True

    if dry_run:
        log_info(f"Would download DocTR weights to {DOCTR_TARGET.relative_to(PROJECT_ROOT)}")
        return True

    try:
        # Set TORCH_HOME to our custom location
        torch_home = APP_DIR / "models"
        os.environ['TORCH_HOME'] = str(torch_home)

        import torch
        from doctr.models import detection, recognition

        # Download detection model (db_resnet50)
        log_info("  Downloading detection model (db_resnet50)...")
        det_model = detection.model_pretrained_registry('db_resnet50', pretrained=True)
        log_success("  Detection model downloaded")

        # Download recognition model (crnn_vgg16_bn)
        log_info("  Downloading recognition model (crnn_vgg16_bn)...")
        rec_model = recognition.model_pretrained_registry('crnn_vgg16_bn', pretrained=True)
        log_success("  Recognition model downloaded")

        # Verify files
        if check_exists(DOCTR_TARGET, "DocTR"):
            file_count = len(list(DOCTR_TARGET.rglob("*")))
            total_size = sum(f.stat().st_size for f in DOCTR_TARGET.rglob("*")) / (1024 * 1024)
            log_success(f"Downloaded {file_count} file(s) ({total_size:.1f} MB)")
            return True
        else:
            log_error("Failed to download DocTR weights")
            return False

    except Exception as e:
        log_error(f"Failed to download DocTR weights: {e}")
        return False


# ============================================================================
# HuggingFace Models (FinBERT, GLiNER2, BERT NER)
# ============================================================================

def download_huggingface_models(force: bool = False, dry_run: bool = False) -> Dict[str, bool]:
    """
    Download all HuggingFace models using the transformers library.

    Models:
    - FinBERT (ProsusAI/finbert)
    - GLiNER2 (fastino/gliner2-large-v1)
    - BERT NER (yashpwr/resume-ner-bert-v2)
    """
    log_section("HuggingFace Models")

    results = {}

    if dry_run:
        log_info("Would download HuggingFace models:")
        log_info("  - FinBERT (ProsusAI/finbert)")
        log_info("  - GLiNER2 (fastino/gliner2-large-v1)")
        log_info("  - BERT NER (yashpwr/resume-ner-bert-v2)")
        return {"finbert": True, "gliner2": True, "bert_ner": True}

    try:
        # Set HuggingFace cache directory
        os.environ['HF_HOME'] = str(HF_CACHE_DIR)
        HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification

        # FinBERT
        log_info("  Downloading FinBERT (ProsusAI/finbert)...")
        try:
            tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

            # Copy to target directory
            FINBERT_TARGET.mkdir(parents=True, exist_ok=True)
            tokenizer.save_pretrained(str(FINBERT_TARGET))
            model.save_pretrained(str(FINBERT_TARGET))
            log_success("  FinBERT downloaded")
            results["finbert"] = True
        except Exception as e:
            log_error(f"  FinBERT: {e}")
            results["finbert"] = False

        # GLiNER2
        log_info("  Downloading GLiNER2 (fastino/gliner2-large-v1)...")
        try:
            # GLiNER uses its own download mechanism
            from app.core.gliner_ner_model import GLINER2_MODEL_NAME
            GLINER2_TARGET.mkdir(parents=True, exist_ok=True)

            # Download using GLiNER library
            try:
                from gliner import GLiNER
                model = GLiNER.from_pretrained("fastino/gliner2-large-v1", cache_dir=str(HF_CACHE_DIR))

                # Save to target directory
                model.save_model(str(GLINER2_TARGET))
                log_success("  GLiNER2 downloaded")
                results["gliner2"] = True
            except ImportError:
                # Fallback: download via huggingface hub
                from huggingface_hub import snapshot_download
                snapshot_download(
                    repo_id="fastino/gliner2-large-v1",
                    local_dir=str(GLINER2_TARGET),
                    local_dir_use_symlinks=False
                )
                log_success("  GLiNER2 downloaded")
                results["gliner2"] = True
        except Exception as e:
            log_error(f"  GLiNER2: {e}")
            results["gliner2"] = False

        # BERT NER
        log_info("  Downloading BERT NER (yashpwr/resume-ner-bert-v2)...")
        try:
            tokenizer = AutoTokenizer.from_pretrained("yashpwr/resume-ner-bert-v2")
            model = AutoModelForTokenClassification.from_pretrained("yashpwr/resume-ner-bert-v2")
            log_success("  BERT NER downloaded")
            results["bert_ner"] = True
        except Exception as e:
            log_error(f"  BERT NER: {e}")
            results["bert_ner"] = False

        # Summary
        success_count = sum(1 for v in results.values() if v)
        log_success(f"HuggingFace: {success_count}/{len(results)} models downloaded")

        return results

    except Exception as e:
        log_error(f"Failed to download HuggingFace models: {e}")
        return {"finbert": False, "gliner2": False, "bert_ner": False}


# ============================================================================
# MediaPipe Download
# ============================================================================

def download_mediapipe(force: bool = False, dry_run: bool = False) -> bool:
    """
    Download MediaPipe hand landmark model.

    Uses the custom download function from the hand gesture detector.
    """
    log_section("MediaPipe")

    if not force and check_exists(MEDIAPIPE_TARGET, "MediaPipe"):
        log_info(f"Already exists at {MEDIAPIPE_TARGET.relative_to(PROJECT_ROOT)} (skipped)")
        return True

    if dry_run:
        log_info(f"Would download MediaPipe model to {MEDIAPIPE_TARGET.relative_to(PROJECT_ROOT)}")
        return True

    try:
        from app.helper.hand_gesture_detector import HandGestureDetector

        # Initialize detector which triggers download
        log_info("  Downloading hand_landmarker.task...")
        detector = HandGestureDetector()
        log_success("  MediaPipe model downloaded")

        # Verify
        if check_exists(MEDIAPIPE_TARGET, "MediaPipe"):
            file_size = (MEDIAPIPE_TARGET / "hand_landmarker.task").stat().st_size / (1024 * 1024)
            log_success(f"Downloaded hand_landmarker.task ({file_size:.1f} MB)")
            return True
        else:
            log_error("Failed to download MediaPipe model")
            return False

    except Exception as e:
        log_error(f"Failed to download MediaPipe model: {e}")
        return False


# ============================================================================
# NLTK Download
# ============================================================================

def download_nltk(force: bool = False, dry_run: bool = False) -> bool:
    """
    Download NLTK data using nltk.download().

    Sets nltk.data.path to custom location.
    """
    log_section("NLTK")

    if not force and check_exists(NLTK_TARGET, "NLTK"):
        log_info(f"Already exists at {NLTK_TARGET.relative_to(PROJECT_ROOT)} (skipped)")
        return True

    if dry_run:
        log_info(f"Would download NLTK data to {NLTK_TARGET.relative_to(PROJECT_ROOT)}")
        return True

    try:
        import nltk

        # Set NLTK data path
        NLTK_TARGET.mkdir(parents=True, exist_ok=True)
        nltk.data.path.insert(0, str(NLTK_TARGET))

        # Download required data
        log_info("  Downloading VADER lexicon...")
        nltk.download('vader_lexicon', download_dir=str(NLTK_TARGET))
        log_success("  VADER lexicon downloaded")

        # Verify
        if check_exists(NLTK_TARGET, "NLTK"):
            file_count = len(list(NLTK_TARGET.rglob("*")))
            log_success(f"Downloaded {file_count} file(s)")
            return True
        else:
            log_error("Failed to download NLTK data")
            return False

    except Exception as e:
        log_error(f"Failed to download NLTK data: {e}")
        return False


# ============================================================================
# PhotoHolmes Download
# ============================================================================

def download_photoholmes(force: bool = False, dry_run: bool = False) -> bool:
    """
    Download PhotoHolmes weights using the CLI tool.

    Downloads weights for adaptive_cfa_net and psccnet methods.
    """
    log_section("PhotoHolmes")

    if not force and check_exists(PHOTOHOLMES_TARGET, "PhotoHolmes"):
        log_info(f"Already exists at {PHOTOHOLMES_TARGET.relative_to(PROJECT_ROOT)} (skipped)")
        return True

    if dry_run:
        log_info(f"Would download PhotoHolmes weights to {PHOTOHOLMES_TARGET.relative_to(PROJECT_ROOT)}")
        return True

    try:
        # Import and use the PhotoHolmes CLI
        import subprocess

        methods = ['adaptive_cfa_net', 'psccnet']
        results = []

        for method in methods:
            log_info(f"  Downloading {method} weights...")
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "app.photoholmes.cli", "download_weights", method],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                if result.returncode == 0:
                    log_success(f"  {method} downloaded")
                    results.append(True)
                else:
                    log_warning(f"  {method}: {result.stderr}")
                    results.append(False)
            except subprocess.TimeoutExpired:
                log_warning(f"  {method}: Download timed out")
                results.append(False)
            except Exception as e:
                log_warning(f"  {method}: {e}")
                results.append(False)

        # Verify
        if check_exists(PHOTOHOLMES_TARGET, "PhotoHolmes"):
            file_count = len(list(PHOTOHOLMES_TARGET.rglob("*")))
            total_size = sum(f.stat().st_size for f in PHOTOHOLMES_TARGET.rglob("*")) / (1024 * 1024)
            log_success(f"Downloaded {file_count} file(s) ({total_size:.1f} MB)")
            return True
        else:
            log_error("Failed to download PhotoHolmes weights")
            return False

    except Exception as e:
        log_error(f"Failed to download PhotoHolmes weights: {e}")
        return False


# ============================================================================
# spaCy Download
# ============================================================================

def download_spacy(force: bool = False, dry_run: bool = False) -> bool:
    """
    Download spaCy language model.

    Downloads en_core_web_sm model.
    """
    log_section("spaCy")

    spacy_model_path = SPACY_TARGET / "en_core_web_sm"

    if not force and check_exists(spacy_model_path, "spaCy"):
        log_info(f"Already exists at {spacy_model_path.relative_to(PROJECT_ROOT)} (skipped)")
        return True

    if dry_run:
        log_info(f"Would download spaCy model to {SPACY_TARGET.relative_to(PROJECT_ROOT)}")
        return True

    try:
        import subprocess

        log_info("  Downloading en_core_web_sm...")
        result = subprocess.run(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode != 0:
            # Try direct download via URL
            log_warning("  spaCy CLI failed, trying direct download...")
            import urllib.request
            import zipfile

            url = "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"

            # Download to temp
            with tempfile.NamedTemporaryFile(suffix=".whl", delete=False) as tmp:
                wheel_path = tmp.name
                log_info(f"  Downloading from {url}...")
                urllib.request.urlretrieve(url, wheel_path)

            # Extract wheel
            log_info("  Extracting wheel...")
            SPACY_TARGET.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(wheel_path, 'r') as zip_ref:
                # Extract en_core_web_sm folder
                for member in zip_ref.namelist():
                    if member.startswith("en_core_web_sm/"):
                        zip_ref.extract(member, SPACY_TARGET)

            os.unlink(wheel_path)

        # Verify
        if check_exists(spacy_model_path, "spaCy"):
            file_count = len(list(spacy_model_path.rglob("*")))
            total_size = sum(f.stat().st_size for f in spacy_model_path.rglob("*")) / (1024 * 1024)
            log_success(f"Downloaded {file_count} file(s) ({total_size:.1f} MB)")
            return True
        else:
            log_error("Failed to download spaCy model")
            return False

    except Exception as e:
        log_error(f"Failed to download spaCy model: {e}")
        return False


# ============================================================================
# InsightFace Download
# ============================================================================

def download_insightface(force: bool = False, dry_run: bool = False) -> bool:
    """
    Download InsightFace buffalo_l model weights.

    Uses the library's built-in download with custom root directory.
    """
    log_section("InsightFace")

    insightface_model_path = INSIGHTFACE_TARGET / "buffalo_l"

    if not force and check_exists(insightface_model_path, "InsightFace"):
        log_info(f"Already exists at {insightface_model_path.relative_to(PROJECT_ROOT)} (skipped)")
        return True

    if dry_run:
        log_info(f"Would download InsightFace model to {insightface_model_path.relative_to(PROJECT_ROOT)}")
        return True

    try:
        from insightface.app import FaceAnalysis

        # Create target directory
        INSIGHTFACE_TARGET.mkdir(parents=True, exist_ok=True)

        # Initialize FaceAnalysis with custom root
        log_info("  Downloading buffalo_l model weights...")
        face_app = FaceAnalysis(name="buffalo_l", root=str(INSIGHTFACE_TARGET))
        face_app.prepare(ctx_id=-1, det_size=(640, 640))
        log_success("  buffalo_l model downloaded")

        # Verify
        onnx_files = list(insightface_model_path.glob("*.onnx"))
        if onnx_files:
            total_size = sum(f.stat().st_size for f in onnx_files) / (1024 * 1024)
            log_success(f"Downloaded {len(onnx_files)} file(s) ({total_size:.1f} MB)")
            for onnx_file in onnx_files:
                size_mb = onnx_file.stat().st_size / (1024 * 1024)
                log_info(f"    - {onnx_file.name} ({size_mb:.1f} MB)")
            return True
        else:
            log_error("Failed to download InsightFace model")
            return False

    except Exception as e:
        log_error(f"Failed to download InsightFace model: {e}")
        return False


# ============================================================================
# CLI Interface
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download ML/DL model weights from official sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all weights
  python scripts/download_model_weights.py

  # Download specific weights
  python scripts/download_model_weights.py --deepface
  python scripts/download_model_weights.py --insightface --huggingface

  # Dry run to see what would be downloaded
  python scripts/download_model_weights.py --dry-run

  # Force re-download existing files
  python scripts/download_model_weights.py --force

Available downloaders:
  --all              Download all weights (default)
  --deepface         DeepFace face recognition models
  --doctr            DocTR document text recognition
  --huggingface      FinBERT, GLiNER2, BERT NER
  --mediapipe        MediaPipe hand gesture detector
  --nltk             NLTK sentiment lexicon
  --photoholmes      PhotoHolmes forgery detection
  --spacy            spaCy language models
  --insightface      InsightFace face recognition
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

    # Individual downloaders
    parser.add_argument("--all", action="store_true", help="Download all weights (default)")
    parser.add_argument("--deepface", action="store_true", help="Download DeepFace weights")
    parser.add_argument("--doctr", action="store_true", help="Download DocTR weights")
    parser.add_argument("--huggingface", action="store_true", help="Download HuggingFace models")
    parser.add_argument("--mediapipe", action="store_true", help="Download MediaPipe model")
    parser.add_argument("--nltk", action="store_true", help="Download NLTK data")
    parser.add_argument("--photoholmes", action="store_true", help="Download PhotoHolmes weights")
    parser.add_argument("--spacy", action="store_true", help="Download spaCy model")
    parser.add_argument("--insightface", action="store_true", help="Download InsightFace model")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Show banner
    logger.info("")
    logger.info("=" * 60)
    logger.info(" Model Weights Download Script")
    logger.info(" Using native library download mechanisms")
    logger.info("=" * 60)
    logger.info("")

    # Determine which downloaders to run
    downloaders = []

    # If no specific downloader selected, run all
    if not any([args.deepface, args.doctr, args.huggingface, args.mediapipe,
                args.nltk, args.photoholmes, args.spacy, args.insightface]):
        args.all = True

    if args.all or args.deepface:
        downloaders.append(("DeepFace", lambda: download_deepface(args.force, args.dry_run)))
    if args.all or args.doctr:
        downloaders.append(("DocTR", lambda: download_doctr(args.force, args.dry_run)))
    if args.all or args.huggingface:
        downloaders.append(("HuggingFace", lambda: download_huggingface_models(args.force, args.dry_run)))
    if args.all or args.mediapipe:
        downloaders.append(("MediaPipe", lambda: download_mediapipe(args.force, args.dry_run)))
    if args.all or args.nltk:
        downloaders.append(("NLTK", lambda: download_nltk(args.force, args.dry_run)))
    if args.all or args.photoholmes:
        downloaders.append(("PhotoHolmes", lambda: download_photoholmes(args.force, args.dry_run)))
    if args.all or args.spacy:
        downloaders.append(("spaCy", lambda: download_spacy(args.force, args.dry_run)))
    if args.all or args.insightface:
        downloaders.append(("InsightFace", lambda: download_insightface(args.force, args.dry_run)))

    # Run downloaders
    results = {}
    for name, downloader in downloaders:
        try:
            result = downloader()
            if isinstance(result, dict):
                results.update(result)
            else:
                results[name.lower()] = result
        except Exception as e:
            log_error(f"{name} failed with exception: {e}")
            results[name.lower()] = False

    # Show summary
    if not args.dry_run:
        logger.info("")
        logger.info("=" * 60)
        logger.info(" Summary")
        logger.info("=" * 60)
        success_count = sum(1 for v in results.values() if v)
        total_count = len(results)

        for name, success in results.items():
            status = "✓" if success else "✗"
            logger.info(f" {status} {name}")

        logger.info("")
        if success_count == total_count:
            logger.info(f"✓ All {total_count} downloaders succeeded!")
        else:
            failed = [k for k, v in results.items() if not v]
            logger.info(f"⚠ {success_count}/{total_count} succeeded. Failed: {', '.join(failed)}")
        logger.info("=" * 60)
        logger.info("")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

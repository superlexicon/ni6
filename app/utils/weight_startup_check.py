"""
PhotoHolmes Weights Startup Verification

Run during application startup to ensure all required weights are available.
Automatically downloads missing weights if needed.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List

# Integrated PhotoHolmes weights are included in app/photoholmes/weights/
# No external PhotoHolmes library needed for startup checks

from app.utils.weight_checker import PhotoHolmesWeightChecker
from app.core.logger import get_logger

logger = get_logger()


class PhotoHolmesWeightManager:
    """Manages PhotoHolmes weights during application startup"""

    def __init__(self, base_path: str = None, auto_download: bool = True):
        """
        Initialize weight manager

        Args:
            base_path: Base path for weights (defaults to current working directory)
            auto_download: Whether to automatically download missing weights
        """
        self.base_path = base_path or os.getcwd()
        self.auto_download = auto_download
        self.weight_checker = PhotoHolmesWeightChecker()

    def verify_and_prepare_weights(self) -> Dict[str, any]:
        """
        Verify all weights are present and download missing ones if auto_download is enabled

        Returns:
            Dictionary with verification results and status
        """
        logger.info("🔍 Verifying PhotoHolmes weights availability...")

        # Check current weight status
        all_present, weight_status = self.weight_checker.verify_all_weights(self.base_path)

        result = {
            "initial_status": "complete" if all_present else "incomplete",
            "weight_status": weight_status,
            "missing_weights": [],
            "downloaded_weights": [],
            "final_status": None,
            "ready_for_use": False
        }

        if all_present:
            logger.info("✅ All PhotoHolmes weights are available and verified")
            result["final_status"] = "complete"
            result["ready_for_use"] = True
        else:
            missing_weights = self.weight_checker.get_missing_weights(self.base_path)
            result["missing_weights"] = missing_weights

            if self.auto_download:
                logger.info(f"📥 Auto-downloading missing weights: {missing_weights}")
                downloaded = self._download_missing_weights(missing_weights)
                result["downloaded_weights"] = downloaded

                # Re-verify after downloads
                all_present_after, weight_status_after = self.weight_checker.verify_all_weights(self.base_path)
                result["final_status"] = "complete" if all_present_after else "incomplete"
                result["ready_for_use"] = all_present_after
                result["weight_status"] = weight_status_after

                if all_present_after:
                    logger.info("✅ All weights now available after downloads")
                else:
                    still_missing = self.weight_checker.get_missing_weights(self.base_path)
                    logger.warning(f"⚠️ Still missing weights after download attempts: {still_missing}")
            else:
                logger.warning(f"⚠️ Missing weights (auto-download disabled): {missing_weights}")
                result["final_status"] = "incomplete"
                result["ready_for_use"] = False

        return result

    def _download_missing_weights(self, missing_weights: List[str]) -> List[str]:
        """
        Download missing weights using PhotoHolmes built-in download functionality

        Args:
            missing_weights: List of missing weight names

        Returns:
            List of successfully downloaded weight names
        """
        downloaded = []

        try:
            # Integrated PhotoHolmes weights are pre-included in app/photoholmes/weights/
            # No download functionality needed for production builds
            logger.info("📦 Using pre-included PhotoHolmes weights from app/photoholmes/weights/")

            for weight_name in missing_weights:
                logger.warning(f"⚠️ Missing weights: {weight_name}")
                logger.info(f"💡 Weights should be pre-included in app/photoholmes/weights/{weight_name}/")

        except Exception as e:
            logger.error(f"❌ Failed to check PhotoHolmes weights: {str(e)}")
            logger.info("💡 Please ensure app/photoholmes/weights/ contains all required weights")

        return downloaded

    def get_weight_summary(self) -> Dict[str, any]:
        """
        Get comprehensive summary of all weights

        Returns:
            Dictionary with detailed weight information
        """
        return self.weight_checker.get_weight_summary(self.base_path)


def verify_photoholmes_weights_on_startup(auto_download: bool = True) -> bool:
    """
    Convenience function to verify weights during application startup

    Args:
        auto_download: Whether to automatically download missing weights

    Returns:
        True if all weights are ready for use, False otherwise
    """
    manager = PhotoHolmesWeightManager(auto_download=auto_download)
    result = manager.verify_and_prepare_weights()

    if result["ready_for_use"]:
        logger.info("🚀 PhotoHolmes weights ready - application can proceed")
        return True
    else:
        logger.error("❌ PhotoHolmes weights not ready - application may have limited functionality")
        return False


def check_spacy_model() -> Dict[str, any]:
    """
    Verify bundled spaCy model exists and is valid.

    Returns:
        Dictionary with spaCy model status
    """
    from pathlib import Path

    spacy_model_path = Path("app/spacy_models/en_core_web_sm")

    if not spacy_model_path.exists():
        return {
            'name': 'spaCy en_core_web_sm',
            'status': 'MISSING',
            'path': str(spacy_model_path),
            'size_mb': 0,
            'message': 'Run: python scripts/download_spacy_model.py'
        }

    # Calculate size
    size_mb = sum(
        f.stat().st_size for f in spacy_model_path.rglob('*') if f.is_file()
    ) / 1024 / 1024

    # Verify model can load
    try:
        import spacy
        nlp = spacy.load(str(spacy_model_path))
        return {
            'name': 'spaCy en_core_web_sm',
            'status': 'OK',
            'path': str(spacy_model_path),
            'size_mb': round(size_mb, 1),
            'version': nlp.meta.get('version', 'unknown')
        }
    except Exception as e:
        return {
            'name': 'spaCy en_core_web_sm',
            'status': f'ERROR: {e}',
            'path': str(spacy_model_path),
            'size_mb': round(size_mb, 1)
        }


if __name__ == "__main__":
    # Standalone execution for testing
    print("PhotoHolmes Weight Verification")
    print("=" * 40)

    manager = PhotoHolmesWeightManager(auto_download=True)
    result = manager.verify_and_prepare_weights()

    print(f"\nFinal Status: {result['final_status']}")
    print(f"Ready for Use: {result['ready_for_use']}")

    if result['downloaded_weights']:
        print(f"Downloaded: {result['downloaded_weights']}")

    if result['missing_weights']:
        print(f"Still Missing: {result['missing_weights']}")

    # Show detailed summary
    summary = manager.get_weight_summary()
    print(f"\nWeight Summary:")
    print(f"Total: {summary['total_weights']}")
    print(f"Available: {summary['available_weights']}")
    print(f"Completeness: {summary['completeness_percentage']}%")
    print(f"Total Size: {summary['total_size_mb']}MB")
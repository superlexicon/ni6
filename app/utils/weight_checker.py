"""
PhotoHolmes Weights Verification Utility

Ensures all required PhotoHolmes model weights are present before initialization.
Prevents unnecessary downloads and verifies weight integrity.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple
from app.core.logger import get_logger

logger = get_logger()


class PhotoHolmesWeightChecker:
    """Utility to check and verify PhotoHolmes model weights presence"""

    # Define required weights with their expected file sizes (approximate)
    REQUIRED_WEIGHTS = {
        "adaptive_cfa_net": {
            "files": ["weights.pth"],
            "min_size_mb": 0.1,  # 140KB
            "path": "app/photoholmes/weights/adaptive_cfa_net"
        },
        "psccnet": {
            "files": ["FENet.pth", "SegNet.pth", "ClsNet.pth"],
            "min_size_mb": 10,  # ~14MB total
            "path": "app/photoholmes/weights/psccnet"
        },
        "focal": {
            "files": ["VIT_weights.pth", "HRNet_weights.pth"],
            "min_size_mb": 400,  # ~452MB total
            "path": "app/photoholmes/weights/focal"
        },
        "exif_as_language": {
            "files": ["weights.pth"],
            "min_size_mb": 300,  # ~349MB
            "path": "app/photoholmes/weights/exif_as_language"
        }
    }

    @classmethod
    def check_weights_exist(cls, base_path: str = None) -> Dict[str, bool]:
        """
        Check if all required PhotoHolmes weights exist and meet minimum size requirements

        Args:
            base_path: Base path for weights (defaults to current working directory)

        Returns:
            Dictionary mapping weight names to their availability status
        """
        if base_path is None:
            base_path = os.getcwd()

        results = {}

        for weight_name, config in cls.REQUIRED_WEIGHTS.items():
            weight_path = Path(base_path) / config["path"]
            files_exist = True
            total_size = 0

            for file_name in config["files"]:
                file_path = weight_path / file_name
                if file_path.exists():
                    try:
                        file_size_mb = file_path.stat().st_size / (1024 * 1024)
                        total_size += file_size_mb
                        logger.debug(f"Weight file found: {file_path} ({file_size_mb:.1f}MB)")
                    except OSError as e:
                        logger.warning(f"Could not read size of {file_path}: {e}")
                        files_exist = False
                        break
                else:
                    logger.warning(f"Missing weight file: {file_path}")
                    files_exist = False
                    break

            # Check if total size meets minimum requirement
            size_ok = total_size >= config["min_size_mb"]
            results[weight_name] = files_exist and size_ok

            if files_exist and size_ok:
                logger.info(f"✅ {weight_name} weights verified ({total_size:.1f}MB)")
            else:
                logger.warning(f"❌ {weight_name} weights missing or insufficient ({total_size:.1f}MB < {config['min_size_mb']}MB)")

        return results

    @classmethod
    def get_missing_weights(cls, base_path: str = None) -> List[str]:
        """
        Get list of missing or insufficient weights

        Args:
            base_path: Base path for weights (defaults to current working directory)

        Returns:
            List of missing weight names
        """
        weight_status = cls.check_weights_exist(base_path)
        missing = [name for name, exists in weight_status.items() if not exists]
        return missing

    @classmethod
    def verify_all_weights(cls, base_path: str = None) -> Tuple[bool, Dict[str, bool]]:
        """
        Verify all required weights are present and return overall status

        Args:
            base_path: Base path for weights (defaults to current working directory)

        Returns:
            Tuple of (overall_status, detailed_status)
        """
        weight_status = cls.check_weights_exist(base_path)
        all_present = all(weight_status.values())

        if all_present:
            logger.info("✅ All PhotoHolmes weights verified and ready")
        else:
            missing = cls.get_missing_weights(base_path)
            logger.warning(f"❌ Missing PhotoHolmes weights: {missing}")

        return all_present, weight_status

    @classmethod
    def get_weight_summary(cls, base_path: str = None) -> Dict[str, any]:
        """
        Get a summary of all weights including their status and sizes

        Args:
            base_path: Base path for weights (defaults to current working directory)

        Returns:
            Dictionary with weight summary information
        """
        if base_path is None:
            base_path = os.getcwd()

        summary = {
            "total_weights": len(cls.REQUIRED_WEIGHTS),
            "available_weights": 0,
            "missing_weights": [],
            "total_size_mb": 0,
            "weights_detail": {}
        }

        for weight_name, config in cls.REQUIRED_WEIGHTS.items():
            weight_path = Path(base_path) / config["path"]
            weight_info = {
                "path": str(weight_path),
                "required_files": config["files"],
                "files_exist": [],
                "files_missing": [],
                "total_size_mb": 0
            }

            total_size = 0
            for file_name in config["files"]:
                file_path = weight_path / file_name
                if file_path.exists():
                    file_size_mb = file_path.stat().st_size / (1024 * 1024)
                    weight_info["files_exist"].append({
                        "name": file_name,
                        "size_mb": round(file_size_mb, 2)
                    })
                    total_size += file_size_mb
                else:
                    weight_info["files_missing"].append(file_name)

            weight_info["total_size_mb"] = round(total_size, 2)
            weight_info["is_complete"] = len(weight_info["files_missing"]) == 0 and total_size >= config["min_size_mb"]

            summary["weights_detail"][weight_name] = weight_info
            summary["total_size_mb"] += total_size

            if weight_info["is_complete"]:
                summary["available_weights"] += 1
            else:
                summary["missing_weights"].append(weight_name)

        summary["total_size_mb"] = round(summary["total_size_mb"], 2)
        summary["completeness_percentage"] = round((summary["available_weights"] / summary["total_weights"]) * 100, 1)

        return summary
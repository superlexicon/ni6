"""
Command-line interface for PhotoHolmes operations within IM-OSINT.

This module provides CLI commands for downloading weights and managing PhotoHolmes models,
adapted to work within the integrated IM-OSINT application structure.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional
import click
import wget

from app.core.logger import get_logger

logger = get_logger()

# Define weight URLs and paths
WEIGHTS_CONFIG = {
    "adaptive_cfa_net": {
        # Downloaded from raw GitHub content as pretrained.pt, renamed to weights.pth
        "url": "https://raw.githubusercontent.com/qbammey/adaptive_cfa_forensics/master/src/models/pretrained.pt",
        "filename": "weights.pth",
        "description": "Adaptive CFA Net weights for forgery detection"
    },
    "psccnet": {
        # PSCCNet weights cannot be downloaded via direct URL
        # They require manual download from the repository
        "manual_download": True,
        "weights": {
            "fenet": {
                "source": "PSCC-Net/checkpoint/HRNet_checkpoint/HRNet.pth",
                "filename": "FENet.pth"
            },
            "segnet": {
                "source": "PSCC-Net/checkpoint/NLCDetection_checkpoint/NLCDetection.pth",
                "filename": "SegNet.pth"
            },
            "clsnet": {
                "source": "PSCC-Net/checkpoint/DetectionHead_checkpoint/DetectionHead.pth",
                "filename": "ClsNet.pth"
            }
        },
        "description": "PSCCNet weights for forgery detection (manual download required)",
        "repo_url": "https://github.com/proteus1991/PSCC-Net",
        "instructions": "Clone the repository or download from Baidu Cloud (password: js74)"
    }
}


def get_weights_dir() -> Path:
    """Get the weights directory for PhotoHolmes models."""
    photoholmes_path = os.getenv("PHOTOHOLMES_PATH", "")
    if not photoholmes_path:
        # Default to project root
        project_root = Path(__file__).parent.parent.parent.parent
        photoholmes_path = str(project_root)

    weights_dir = Path(photoholmes_path) / "photoholmes" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    return weights_dir


def download_file(url: str, filename: str, target_dir: Path) -> Path:
    """
    Download a file from URL to target directory.

    Args:
        url: URL to download from
        filename: Name of the file to save
        target_dir: Directory to save the file

    Returns:
        Path to the downloaded file
    """
    target_path = target_dir / filename

    if target_path.exists():
        logger.info(f"File already exists: {target_path}")
        return target_path

    logger.info(f"Downloading {filename} from {url}")

    try:
        # Use wget with progress bar
        wget.download(url, str(target_path))
        logger.info(f"Successfully downloaded {filename}")
        return target_path
    except Exception as e:
        logger.error(f"Failed to download {filename}: {str(e)}")
        # Clean up partial download
        if target_path.exists():
            target_path.unlink()
        raise


@click.group()
def cli():
    """PhotoHolmes CLI for IM-OSINT."""
    pass


@cli.command()
@click.argument('method', type=click.Choice(['adaptive_cfa_net', 'psccnet']))
@click.option('--force', is_flag=True, help='Force download even if files exist')
def download_weights(method: str, force: bool = False):
    """Download PhotoHolmes model weights."""
    weights_dir = get_weights_dir()

    if method not in WEIGHTS_CONFIG:
        click.echo(f"Unknown method: {method}")
        click.echo(f"Available methods: {', '.join(WEIGHTS_CONFIG.keys())}")
        sys.exit(1)

    config = WEIGHTS_CONFIG[method]
    method_dir = weights_dir / method
    method_dir.mkdir(exist_ok=True)

    try:
        if method == "adaptive_cfa_net":
            target_path = method_dir / config["filename"]
            if not force and target_path.exists():
                click.echo(f"Weights already exist at {target_path}")
                return

            # Download from raw GitHub URL (saves as pretrained.pt, needs rename to weights.pth)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
                tmp_path = Path(tmp.name)

            logger.info(f"Downloading from {config['url']}")
            wget.download(config["url"], str(tmp_path))
            tmp_path.rename(target_path)
            click.echo(f"\n✅ Adaptive CFA Net weights downloaded to {target_path}")

        elif method == "psccnet":
            # PSCCNet requires manual download
            if config.get("manual_download"):
                click.echo("⚠️  PSCCNet weights cannot be downloaded automatically")
                click.echo(f"\nRepository: {config['repo_url']}")
                click.echo(f"\n{config['instructions']}")
                click.echo("\nManual download instructions:")
                click.echo("  1. Clone the repository:")
                click.echo("     git clone https://github.com/proteus1991/PSCC-Net.git /tmp/psccnet")
                click.echo(f"\n  2. Copy the checkpoint files to {method_dir}:")
                for weight_name, weight_config in config["weights"].items():
                    click.echo(f"     cp /tmp/psccnet/{weight_config['source']} \\")
                    click.echo(f"        {method_dir}/{weight_config['filename']}")
                click.echo("\n  3. Or download from Baidu Cloud (password: js74)")
                click.echo("     See the repository README for details")
                return

            downloaded_files = []
            for weight_name, weight_config in config["weights"].items():
                target_path = method_dir / weight_config["filename"]
                if not force and target_path.exists():
                    click.echo(f"⏭️  {weight_name} already exists")
                    downloaded_files.append(target_path)
                    continue

                download_file(weight_config["url"], weight_config["filename"], method_dir)
                downloaded_files.append(target_path)
                click.echo(f"✅ {weight_name} weights downloaded")

            click.echo(f"✅ All PSCCNet weights downloaded to {method_dir}")

    except Exception as e:
        click.echo(f"❌ Failed to download {method} weights: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
def list_weights():
    """List available and downloaded weights."""
    weights_dir = get_weights_dir()

    click.echo(f"PhotoHolmes weights directory: {weights_dir}")
    click.echo()

    for method, config in WEIGHTS_CONFIG.items():
        method_dir = weights_dir / method
        click.echo(f"📁 {method}: {config['description']}")

        if not method_dir.exists():
            click.echo("   ❌ Not downloaded")
        else:
            if method == "adaptive_cfa_net":
                weight_file = method_dir / config["filename"]
                if weight_file.exists():
                    size_mb = weight_file.stat().st_size / (1024 * 1024)
                    click.echo(f"   ✅ {config['filename']} ({size_mb:.1f} MB)")
                else:
                    click.echo(f"   ❌ {config['filename']}")

            elif method == "psccnet":
                for weight_name, weight_config in config["weights"].items():
                    weight_file = method_dir / weight_config["filename"]
                    if weight_file.exists():
                        size_mb = weight_file.stat().st_size / (1024 * 1024)
                        click.echo(f"   ✅ {weight_config['filename']} ({size_mb:.1f} MB)")
                    else:
                        click.echo(f"   ❌ {weight_config['filename']}")

        click.echo()


@cli.command()
@click.option('--method', help='Clean weights for specific method')
def clean_weights(method: Optional[str] = None):
    """Clean downloaded weights."""
    weights_dir = get_weights_dir()

    if method:
        if method not in WEIGHTS_CONFIG:
            click.echo(f"Unknown method: {method}")
            sys.exit(1)

        method_dir = weights_dir / method
        if method_dir.exists():
            import shutil
            shutil.rmtree(method_dir)
            click.echo(f"✅ Cleaned {method} weights")
        else:
            click.echo(f"No weights found for {method}")
    else:
        if weights_dir.exists():
            import shutil
            shutil.rmtree(weights_dir)
            click.echo(f"✅ Cleaned all PhotoHolmes weights")
        else:
            click.echo("No weights directory found")


# Add backward compatibility functions for existing code
def download_adaptive_cfa_weights():
    """Download Adaptive CFA Net weights (backward compatibility)."""
    weights_dir = get_weights_dir()
    config = WEIGHTS_CONFIG["adaptive_cfa_net"]
    method_dir = weights_dir / "adaptive_cfa_net"
    method_dir.mkdir(exist_ok=True)

    target_path = method_dir / config["filename"]
    if not target_path.exists():
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            tmp_path = Path(tmp.name)

        wget.download(config["url"], str(tmp_path))
        tmp_path.rename(target_path)

    return target_path


def download_psccnet_weights():
    """Download PSCCNet weights (backward compatibility)."""
    weights_dir = get_weights_dir()
    config = WEIGHTS_CONFIG["psccnet"]

    # PSCCNet requires manual download
    if config.get("manual_download"):
        logger.warning("PSCCNet weights cannot be downloaded automatically")
        logger.info(f"See {config['repo_url']} for manual download instructions")
        return []

    method_dir = weights_dir / "psccnet"
    method_dir.mkdir(exist_ok=True)

    downloaded_files = []
    for weight_name, weight_config in config["weights"].items():
        target_path = method_dir / weight_config["filename"]
        if not target_path.exists():
            download_file(weight_config["url"], weight_config["filename"], method_dir)
        downloaded_files.append(target_path)

    return downloaded_files


if __name__ == "__main__":
    cli()
#!/usr/bin/env python3
"""
Download GLINER2 model weights using huggingface_hub.

This is necessary because GLINER2's model.safetensors is stored in HuggingFace's
Xet storage, which returns 504 Gateway Timeout when accessed via curl.

Usage:
    python scripts/download_gliner2.py [target_dir]
    python scripts/download_gliner2.py app/models/gliner2
"""

import sys
import os
from pathlib import Path

def main():
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "app/models/gliner2"

    # Convert to absolute path
    if not os.path.isabs(target_dir):
        script_dir = Path(__file__).parent
        target_dir = script_dir.parent / target_dir

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading GLINER2 to: {target_dir}")

    try:
        from huggingface_hub import hf_hub_download

        # All required files for GLiNER2 model
        # Source: https://huggingface.co/fastino/gliner2-large-v1/tree/main
        required_files = [
            "config.json",
            "encoder_config/config.json",
            "added_tokens.json",
            "special_tokens_map.json",
            "spm.model",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors",  # Large file (~1.9GB) - download last
        ]

        for filename in required_files:
            # Skip model.safetensors for now (download separately)
            if filename == "model.safetensors":
                continue

            print(f"  Downloading {filename}...")
            file_path = hf_hub_download(
                repo_id="fastino/gliner2-large-v1",
                filename=filename,
                local_dir=str(target_dir)
            )
            size_bytes = os.path.getsize(file_path)
            size_kb = size_bytes / 1024
            print(f"    → {filename} ({size_kb:.1f} KB)")

        # Download model.safetensors (large file, ~1.9GB)
        print("  Downloading model.safetensors (~1.9GB, this may take several minutes)...")
        model_path = hf_hub_download(
            repo_id="fastino/gliner2-large-v1",
            filename="model.safetensors",
            local_dir=str(target_dir)
        )
        size_mb = os.path.getsize(model_path) / 1024 / 1024
        print(f"    → model.safetensors ({size_mb:.1f} MB)")

        print("✓ GLINER2 download complete")
        return 0

    except ImportError:
        print("✗ Error: huggingface_hub not installed")
        print("  Install with: poetry add huggingface_hub")
        return 1
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

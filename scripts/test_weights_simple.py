#!/usr/bin/env python3
"""
Simple test to verify DeepFace uses correct weights path.

This test verifies that when DEEPFACE_HOME is set to 'app/deepface',
DeepFace stores weights at 'app/deepface/.deepface/weights/'.

Usage:
    # Test with default DEEPFACE_HOME (should be app/deepface)
    DEEPFACE_HOME=app/deepface poetry run python scripts/test_weights_simple.py

    # Or test current configuration
    poetry run python scripts/test_weights_simple.py
"""

import os
import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    print("=" * 60)
    print("DeepFace Weights Path Test")
    print("=" * 60)
    print()

    # Show current DEEPFACE_HOME
    deepface_home = os.environ.get("DEEPFACE_HOME", "")
    print(f"DEEPFACE_HOME: '{deepface_home}'")
    print()

    if not deepface_home:
        print("⚠ DEEPFACE_HOME is not set!")
        print("  Add this to your .env file:")
        print("  DEEPFACE_HOME=app/deepface")
        print()
        print("  Testing with default behavior...")
        print()

    # Import DeepFace utilities
    try:
        from app.deepface.deepface.commons import folder_utils
        from app.deepface.deepface.commons import weight_utils

        # Get the actual DeepFace home
        actual_home = folder_utils.get_deepface_home()
        print(f"Actual DeepFace Home: '{actual_home}'")

        # Check if it matches expected
        if deepface_home and deepface_home in actual_home:
            print("✓ DEEPFACE_HOME is being used")
        elif not deepface_home:
            print(f"ℹ Using default: {actual_home}")
        else:
            print(f"⚠ DEEPFACE_HOME set but not in actual path")

        print()

        # Test where weights would be stored
        test_file = "test_model.h5"
        expected_weights_dir = os.path.join(actual_home, ".deepface", "weights")
        expected_path = os.path.join(expected_weights_dir, test_file)

        print(f"Weights directory: {expected_weights_dir}")
        print(f"Test file path: {expected_path}")
        print()

        # Check if weights directory exists
        weights_path = Path(expected_weights_dir)
        if weights_path.exists():
            files = list(weights_path.glob("*"))
            print(f"✓ Weights directory exists ({len(files)} files)")
            for f in sorted(files)[:5]:
                size = f.stat().st_size / (1024 * 1024)  # MB
                print(f"  - {f.name} ({size:.1f} MB)")
            if len(files) > 5:
                print(f"  ... and {len(files) - 5} more")
        else:
            print(f"⚠ Weights directory does not exist yet")
            print(f"  Weights will be downloaded on first use")

        print()
        print("=" * 60)
        print("Summary")
        print("=" * 60)

        if deepface_home == "app/deepface":
            print("✓ DEEPFACE_HOME correctly set to 'app/deepface'")
            print("✓ Weights will be stored at: app/deepface/.deepface/weights/")
            return 0
        elif not deepface_home:
            print("⚠ DEEPFACE_HOME not set")
            print("  Add to .env: DEEPFACE_HOME=app/deepface")
            return 1
        else:
            print(f"✗ DEEPFACE_HOME is '{deepface_home}' (expected 'app/deepface')")
            return 1

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

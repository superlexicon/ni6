#!/usr/bin/env python3
"""
Generate passport test images for testing the image preprocessing pipeline.

This script creates a comprehensive set of passport test images derived from
the base Indian passport image, covering various formats, color modes, and size scenarios.
"""

import os
from pathlib import Path
from PIL import Image, ImageCms, ImageDraw

# Configuration
BASE_PASSPORT_PATH = "scripts/test_data/user_005/passport.png"
OUTPUT_DIR = "scripts/test_data/user_014"


def load_base_passport() -> Image.Image:
    """Load the base passport image."""
    img = Image.open(BASE_PASSPORT_PATH)
    print(f"Loaded base passport: {img.size} pixels, mode: {img.mode}")
    return img


def ensure_output_dir():
    """Ensure the output directory exists."""
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def convert_to_format(img: Image.Image, format_name: str, output_path: str, **kwargs):
    """Convert image to specified format and save."""
    format_map = {
        "JPEG": ("JPEG", ".jpg"),
        "PNG": ("PNG", ".png"),
        "TIFF": ("TIFF", ".tiff"),
        "WEBP": ("WEBP", ".webp"),
        "BMP": ("BMP", ".bmp"),
    }

    pil_format, ext = format_map[format_name]
    img.save(output_path, format=pil_format, **kwargs)
    print(f"Created: {output_path}")


def convert_color_mode(img: Image.Image, mode: str, output_path: str, format: str = "JPEG"):
    """Convert image to specified color mode and save."""
    if mode == "RGB":
        converted = img.convert("RGB")
    elif mode == "RGBA":
        converted = img.convert("RGBA")
    elif mode == "L":
        # Grayscale
        converted = img.convert("L")
    elif mode == "P":
        # Palette mode
        converted = img.convert("P", palette=Image.Palette.ADAPTIVE)
    elif mode == "CMYK":
        # Convert RGB to CMYK using color profile
        rgb_img = img.convert("RGB")
        # Create a simple CMYK conversion
        converted = rgb_img.convert("CMYK")
    else:
        converted = img.convert(mode)

    save_format = "PNG" if mode in ["RGBA", "P"] else format
    converted.save(output_path, format=save_format)
    print(f"Created: {output_path} (mode: {mode}, format: {save_format})")


def scale_image(img: Image.Image, scale_factor: float, output_path: str):
    """Scale image by factor and save."""
    if scale_factor == 1.0:
        scaled = img
    else:
        new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
        scaled = img.resize(new_size, Image.Resampling.LANCZOS)

    scaled.save(output_path, format="JPEG", quality=95)
    print(f"Created: {output_path} (scale: {scale_factor}x, size: {scaled.size})")


def create_aspect_ratio_variations(img: Image.Image, output_dir: str):
    """Create wide and tall aspect ratio variants."""
    # Get the original image data
    img_rgb = img.convert("RGB")

    # Wide aspect ratio (2000x400px) - fit height, center horizontally
    wide_width, wide_height = 2000, 400
    wide = Image.new("RGB", (wide_width, wide_height), color=(255, 255, 255))

    # Scale to fit the height, then center horizontally
    scale = wide_height / img.height
    new_width = int(img.width * scale)
    resized = img_rgb.resize((new_width, wide_height), Image.Resampling.LANCZOS)
    x = (wide_width - new_width) // 2

    wide.paste(resized, (x, 0))
    wide_path = os.path.join(output_dir, "passport_special_aspect_wide.jpg")
    wide.save(wide_path, format="JPEG", quality=95)
    print(f"Created: {wide_path} (wide: {wide.size}, passport size: {resized.size})")

    # Tall aspect ratio (400x2000px) - fit width, center vertically
    tall_width, tall_height = 400, 2000
    tall = Image.new("RGB", (tall_width, tall_height), color=(255, 255, 255))

    # Scale to fit the width, then center vertically
    scale = tall_width / img.width
    new_height = int(img.height * scale)
    resized = img_rgb.resize((tall_width, new_height), Image.Resampling.LANCZOS)
    y = (tall_height - new_height) // 2

    tall.paste(resized, (0, y))
    tall_path = os.path.join(output_dir, "passport_special_aspect_tall.jpg")
    tall.save(tall_path, format="JPEG", quality=95)
    print(f"Created: {tall_path} (tall: {tall.size}, passport size: {resized.size})")


def generate_all_images():
    """Generate all passport test images."""
    ensure_output_dir()

    # Load base passport
    base = load_base_passport()
    base_rgb = base.convert("RGB")

    print("\n=== Generating Format Variations ===")
    # Format variations
    convert_to_format(base_rgb, "JPEG", os.path.join(OUTPUT_DIR, "passport_format_jpeg.jpg"))
    convert_to_format(base, "PNG", os.path.join(OUTPUT_DIR, "passport_format_png.png"))
    convert_to_format(base, "PNG", os.path.join(OUTPUT_DIR, "passport_format_png_alpha.png"))
    convert_to_format(base, "TIFF", os.path.join(OUTPUT_DIR, "passport_format_tiff.tiff"))
    convert_to_format(base, "WEBP", os.path.join(OUTPUT_DIR, "passport_format_webp.webp"), quality=95)
    convert_to_format(base, "BMP", os.path.join(OUTPUT_DIR, "passport_format_bmp.bmp"))

    print("\n=== Generating Color Mode Variations ===")
    # Color mode variations
    convert_color_mode(base_rgb, "RGB", os.path.join(OUTPUT_DIR, "passport_colormode_rgb.jpg"), format="JPEG")
    convert_color_mode(base_rgb, "CMYK", os.path.join(OUTPUT_DIR, "passport_colormode_cmyk.jpg"), format="JPEG")
    convert_color_mode(base, "RGBA", os.path.join(OUTPUT_DIR, "passport_colormode_rgba.png"), format="PNG")
    convert_color_mode(base, "L", os.path.join(OUTPUT_DIR, "passport_colormode_grayscale.png"), format="PNG")
    convert_color_mode(base, "P", os.path.join(OUTPUT_DIR, "passport_colormode_palette.png"), format="PNG")

    print("\n=== Generating Size Variations ===")
    # Size variations
    scale_image(base_rgb, 1.0, os.path.join(OUTPUT_DIR, "passport_size_original.jpg"))
    scale_image(base_rgb, 2.0, os.path.join(OUTPUT_DIR, "passport_size_2x_scaled.jpg"))
    scale_image(base_rgb, 3.0, os.path.join(OUTPUT_DIR, "passport_size_3x_scaled.jpg"))
    scale_image(base_rgb, 4.0, os.path.join(OUTPUT_DIR, "passport_size_4x_scaled.jpg"))

    print("\n=== Generating Special Cases ===")
    # Special cases - aspect ratio variations
    create_aspect_ratio_variations(base, OUTPUT_DIR)

    print("\n=== Summary ===")
    output_files = sorted(
        list(Path(OUTPUT_DIR).glob("passport_*.jpg")) +
        list(Path(OUTPUT_DIR).glob("passport_*.png")) +
        list(Path(OUTPUT_DIR).glob("passport_*.tiff")) +
        list(Path(OUTPUT_DIR).glob("passport_*.webp")) +
        list(Path(OUTPUT_DIR).glob("passport_*.bmp"))
    )
    print(f"\nTotal test images created: {len(output_files)}")
    for f in output_files:
        print(f"  - {f.name}")


if __name__ == "__main__":
    generate_all_images()

#!/usr/bin/env python3
"""
Generate placeholder passport template images.

Run this script once to create placeholder images for all countries
defined in config.json. These should be replaced with actual sanitized
passport template images.
"""

import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Directory containing this script and config.json
TEMPLATES_DIR = Path(__file__).parent / "passports"


def create_placeholder_image(country_code: str, country_name: str, width: int = 880, height: int = 1240) -> Image.Image:
    """
    Create a placeholder passport template image.

    Args:
        country_code: ISO 3-letter country code
        country_name: Full country name
        width: Image width (default passport data page width)
        height: Image height (default passport data page height)

    Returns:
        PIL Image object
    """
    # Create base image with a light gray background
    img = Image.new('RGB', (width, height), color=(240, 240, 245))
    draw = ImageDraw.Draw(img)

    # Add border
    border_color = (100, 100, 120)
    draw.rectangle([5, 5, width-6, height-6], outline=border_color, width=3)

    # Add header area (simulating passport header)
    header_y = 30
    draw.rectangle([20, header_y, width-20, header_y + 80], fill=(220, 220, 230), outline=border_color)

    # Add "PASSPORT" text in header
    try:
        font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
        font_medium = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except:
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Header text
    text = "PASSPORT"
    bbox = draw.textbbox((0, 0), text, font=font_large)
    text_width = bbox[2] - bbox[0]
    draw.text(((width - text_width) // 2, header_y + 25), text, fill=(50, 50, 70), font=font_large)

    # Country name below header
    country_text = country_name.upper()
    bbox = draw.textbbox((0, 0), country_text, font=font_medium)
    text_width = bbox[2] - bbox[0]
    draw.text(((width - text_width) // 2, header_y + 95), country_text, fill=(50, 50, 70), font=font_medium)

    # Add photo placeholder area (right side)
    photo_x, photo_y = width - 280, 200
    photo_w, photo_h = 240, 300
    draw.rectangle([photo_x, photo_y, photo_x + photo_w, photo_y + photo_h],
                   fill=(200, 200, 210), outline=border_color, width=2)
    draw.text((photo_x + 60, photo_y + 140), "[PHOTO]", fill=(150, 150, 160), font=font_small)

    # Add ghost photo placeholder (left side, smaller)
    ghost_x, ghost_y = 40, 350
    ghost_w, ghost_h = 100, 130
    draw.rectangle([ghost_x, ghost_y, ghost_x + ghost_w, ghost_y + ghost_h],
                   fill=(210, 210, 220), outline=(180, 180, 190), width=1)
    draw.text((ghost_x + 10, ghost_y + 55), "[GHOST]", fill=(170, 170, 180), font=font_small)

    # Add guilloche pattern area (left side)
    guilloche_x, guilloche_y = 40, 500
    guilloche_w, guilloche_h = 350, 400
    draw.rectangle([guilloche_x, guilloche_y, guilloche_x + guilloche_w, guilloche_y + guilloche_h],
                   fill=(230, 230, 240), outline=border_color, width=1)

    # Draw simple guilloche-like pattern
    for i in range(0, guilloche_w, 20):
        for j in range(0, guilloche_h, 20):
            x = guilloche_x + i + 10
            y = guilloche_y + j + 10
            draw.ellipse([x-5, y-5, x+5, y+5], outline=(200, 200, 210))
            draw.arc([x-8, y-8, x+8, y+8], 0, 180, fill=(190, 190, 200))

    # Add security thread placeholder (vertical line)
    thread_x = width // 2
    draw.line([(thread_x, 0), (thread_x, height)], fill=(180, 160, 140), width=10)
    draw.line([(thread_x, 0), (thread_x, height)], fill=(200, 180, 160), width=6)

    # Add data fields placeholder (left side)
    fields_y = 180
    field_labels = [
        "Surname:",
        "Given Names:",
        "Nationality:",
        "Date of Birth:",
        "Sex:",
        "Place of Birth:",
        "Date of Issue:",
        "Date of Expiry:",
        "Passport No.:"
    ]

    for i, label in enumerate(field_labels):
        y_pos = fields_y + i * 35
        draw.text((40, y_pos), label, fill=(100, 100, 120), font=font_small)
        # Field value line
        draw.line([(200, y_pos + 20), (photo_x - 20, y_pos + 20)], fill=(180, 180, 190), width=1)

    # Add MRZ placeholder (bottom)
    mrz_y = height - 120
    draw.rectangle([20, mrz_y, width-20, height-20], fill=(220, 220, 230), outline=border_color)
    mrz_text = f"P<{country_code}{'<'*37}"
    draw.text((30, mrz_y + 15), mrz_text[:44], fill=(80, 80, 100), font=font_small)
    draw.text((30, mrz_y + 45), f"{'<'*44}", fill=(80, 80, 100), font=font_small)

    # Add watermark
    watermark = "PLACEHOLDER TEMPLATE"
    bbox = draw.textbbox((0, 0), watermark, font=font_large)
    text_width = bbox[2] - bbox[0]
    draw.text(((width - text_width) // 2, height // 2), watermark,
              fill=(220, 220, 225), font=font_large)

    return img


def main():
    """Generate placeholder images for all countries in config.json."""
    config_path = TEMPLATES_DIR / "config.json"

    if not config_path.exists():
        print(f"Error: config.json not found at {config_path}")
        return

    with open(config_path, 'r') as f:
        config = json.load(f)

    countries = config.get('countries', {})
    print(f"Found {len(countries)} countries in config.json")

    generated = 0
    skipped = 0

    for country_code, country_config in countries.items():
        country_name = country_config.get('name', country_code)

        # Check if image already exists
        jpg_path = TEMPLATES_DIR / f"{country_code}.jpg"
        png_path = TEMPLATES_DIR / f"{country_code}.png"

        if jpg_path.exists() or png_path.exists():
            print(f"  [SKIP] {country_code}: Image already exists")
            skipped += 1
            continue

        # Generate placeholder
        print(f"  [GEN]  {country_code}: Generating placeholder for {country_name}")
        img = create_placeholder_image(country_code, country_name)

        # Save as JPEG
        img.save(jpg_path, "JPEG", quality=95)
        generated += 1

    print(f"\nDone! Generated {generated} images, skipped {skipped}")
    print(f"\nNOTE: These are placeholder images. Replace them with actual sanitized")
    print(f"      passport template images before production use.")


if __name__ == "__main__":
    main()

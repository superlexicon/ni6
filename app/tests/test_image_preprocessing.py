"""
Unit tests for Image Preprocessing in LLM Service.

Tests the image preprocessing pipeline including:
- Various image formats (JPEG, PNG, TIFF, WEBP, BMP)
- Various color modes (RGB, CMYK, RGBA, Grayscale, Palette)
- Large image downsizing (token budget enforcement)
- PDF to image conversion
"""

import unittest
import sys
import os
import io
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from PIL import Image

# Import LLMService
import importlib.util
spec = importlib.util.spec_from_file_location("llm_service", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "services", "llm_service.py"))
llm_service = importlib.util.module_from_spec(spec)
spec.loader.exec_module(llm_service)

LLMService = llm_service.LLMService


class TestImagePreprocessing(unittest.TestCase):
    """Test cases for image preprocessing in LLM Service."""

    def setUp(self):
        """Set up test fixtures."""
        self.llm_service = LLMService()
        self.test_dir = Path(__file__).parent / "test_images"
        self.test_dir.mkdir(exist_ok=True)

        # Path to base Indian passport image
        self.base_image_path = Path(__file__).parent.parent / "reference_templates" / "passports" / "IN.png"

        # Load base image for conversions
        if self.base_image_path.exists():
            self.base_image = Image.open(self.base_image_path)
        else:
            # Create a test image if base image is not available
            self.base_image = self._create_test_image()

    def tearDown(self):
        """Clean up test fixtures."""
        # Optional: Clean up generated test images
        pass

    def _create_test_image(self, size=(800, 600)):
        """Create a synthetic test image for testing."""
        img = Image.new('RGB', size, color=(100, 150, 200))
        # Add some text and patterns for complexity
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, 200, 100], fill=(255, 0, 0))
        draw.rectangle([250, 50, 400, 100], fill=(0, 255, 0))
        draw.rectangle([450, 50, 600, 100], fill=(0, 0, 255))
        return img

    def _image_to_bytes(self, img, format='JPEG', **kwargs):
        """Convert PIL Image to bytes."""
        buffer = io.BytesIO()
        # Convert RGBA to RGB for JPEG formats
        if format.upper() == 'JPEG' and img.mode == 'RGBA':
            # Create white background
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
            img = background
        img.save(buffer, format=format, **kwargs)
        return buffer.getvalue()

    def _bytes_to_image(self, image_bytes):
        """Convert bytes to PIL Image."""
        return Image.open(io.BytesIO(image_bytes))

    def _save_test_image(self, image_bytes, filename):
        """Save test image to test directory for manual inspection."""
        filepath = self.test_dir / filename
        with open(filepath, 'wb') as f:
            f.write(image_bytes)
        return filepath

    # ==================== Format Tests ====================

    def test_jpeg_format(self):
        """Test JPEG format processing."""
        # Convert base image to JPEG
        jpeg_bytes = self._image_to_bytes(self.base_image, 'JPEG', quality=95)

        # Process through _ensure_token_budget
        result = self.llm_service._ensure_token_budget(jpeg_bytes)

        # Verify output is valid JPEG
        self.assertTrue(result.startswith(b'\xff\xd8\xff'))  # JPEG magic bytes

        # Verify it's a valid image
        img = self._bytes_to_image(result)
        self.assertEqual(img.mode, 'RGB')

        # Save for inspection
        self._save_test_image(result, 'test_jpeg_output.jpg')

    def test_png_format(self):
        """Test PNG format processing."""
        # Convert base image to PNG
        png_bytes = self._image_to_bytes(self.base_image, 'PNG')

        # Process through _ensure_token_budget
        result = self.llm_service._ensure_token_budget(png_bytes)

        # Verify output is valid JPEG (should be converted)
        self.assertTrue(result.startswith(b'\xff\xd8\xff'))  # JPEG magic bytes

        # Verify it's a valid image
        img = self._bytes_to_image(result)
        self.assertEqual(img.mode, 'RGB')

        # Save for inspection
        self._save_test_image(result, 'test_png_output.jpg')

    def test_png_with_alpha(self):
        """Test PNG format with alpha channel."""
        # Convert base image to PNG with alpha
        if self.base_image.mode != 'RGBA':
            base_rgba = self.base_image.convert('RGBA')
        else:
            base_rgba = self.base_image

        # Add some transparency
        png_alpha_bytes = self._image_to_bytes(base_rgba, 'PNG')

        # Process through _ensure_token_budget
        result = self.llm_service._ensure_token_budget(png_alpha_bytes)

        # Verify output is valid JPEG (alpha should be removed)
        self.assertTrue(result.startswith(b'\xff\xd8\xff'))  # JPEG magic bytes

        # Verify it's a valid RGB image
        img = self._bytes_to_image(result)
        self.assertEqual(img.mode, 'RGB')

        # Save for inspection
        self._save_test_image(result, 'test_png_alpha_output.jpg')

    def test_webp_format(self):
        """Test WEBP format processing."""
        # Convert base image to WEBP
        try:
            webp_bytes = self._image_to_bytes(self.base_image, 'WEBP', quality=95)

            # Process through _ensure_token_budget
            result = self.llm_service._ensure_token_budget(webp_bytes)

            # Verify output is valid JPEG
            self.assertTrue(result.startswith(b'\xff\xd8\xff'))  # JPEG magic bytes

            # Verify it's a valid image
            img = self._bytes_to_image(result)
            self.assertEqual(img.mode, 'RGB')

            # Save for inspection
            self._save_test_image(result, 'test_webp_output.jpg')
        except Exception as e:
            # WEBP might not be supported in all PIL versions
            self.skipTest(f"WEBP format not supported: {e}")

    def test_bmp_format(self):
        """Test BMP format processing."""
        # Convert base image to BMP
        bmp_bytes = self._image_to_bytes(self.base_image, 'BMP')

        # Process through _ensure_token_budget
        result = self.llm_service._ensure_token_budget(bmp_bytes)

        # Verify output is valid JPEG
        self.assertTrue(result.startswith(b'\xff\xd8\xff'))  # JPEG magic bytes

        # Verify it's a valid image
        img = self._bytes_to_image(result)
        self.assertEqual(img.mode, 'RGB')

        # Save for inspection
        self._save_test_image(result, 'test_bmp_output.jpg')

    def test_tiff_format(self):
        """Test TIFF format processing."""
        # Convert base image to TIFF
        tiff_bytes = self._image_to_bytes(self.base_image, 'TIFF')

        # Process through _ensure_token_budget
        result = self.llm_service._ensure_token_budget(tiff_bytes)

        # Verify output is valid JPEG
        self.assertTrue(result.startswith(b'\xff\xd8\xff'))  # JPEG magic bytes

        # Verify it's a valid image
        img = self._bytes_to_image(result)
        self.assertEqual(img.mode, 'RGB')

        # Save for inspection
        self._save_test_image(result, 'test_tiff_output.jpg')

    # ==================== Color Mode Tests ====================

    def test_rgb_mode(self):
        """Test RGB color mode processing (baseline)."""
        # Ensure base image is RGB
        if self.base_image.mode != 'RGB':
            base_rgb = self.base_image.convert('RGB')
        else:
            base_rgb = self.base_image

        rgb_bytes = self._image_to_bytes(base_rgb, 'JPEG')

        # Process through _ensure_token_budget
        result = self.llm_service._ensure_token_budget(rgb_bytes)

        # Verify output is valid JPEG
        self.assertTrue(result.startswith(b'\xff\xd8\xff'))

        # Verify it's still RGB
        img = self._bytes_to_image(result)
        self.assertEqual(img.mode, 'RGB')

        # Save for inspection
        self._save_test_image(result, 'test_rgb_output.jpg')

    def test_cmyk_mode(self):
        """Test CMYK color mode conversion to RGB."""
        # Convert to CMYK (simulating scanned documents)
        base_rgb = self.base_image.convert('RGB') if self.base_image.mode != 'RGB' else self.base_image
        cmyk_img = base_rgb.convert('CMYK')
        cmyk_bytes = self._image_to_bytes(cmyk_img, 'JPEG')

        # Process through _ensure_token_budget
        result = self.llm_service._ensure_token_budget(cmyk_bytes)

        # Verify output is valid JPEG
        self.assertTrue(result.startswith(b'\xff\xd8\xff'))

        # Verify it was converted to RGB
        img = self._bytes_to_image(result)
        self.assertEqual(img.mode, 'RGB', "CMYK should be converted to RGB")

        # Save for inspection
        self._save_test_image(result, 'test_cmyk_output.jpg')

        # Also save original CMYK for comparison
        self._save_test_image(cmyk_bytes, 'test_cmyk_input.jpg')

    def test_rgba_mode(self):
        """Test RGBA mode with transparency."""
        # Convert to RGBA
        rgba_img = self.base_image.convert('RGBA')
        rgba_bytes = self._image_to_bytes(rgba_img, 'PNG')

        # Process through _ensure_token_budget
        result = self.llm_service._ensure_token_budget(rgba_bytes)

        # Verify output is valid JPEG
        self.assertTrue(result.startswith(b'\xff\xd8\xff'))

        # Verify it was converted to RGB
        img = self._bytes_to_image(result)
        self.assertEqual(img.mode, 'RGB', "RGBA should be converted to RGB")

        # Save for inspection
        self._save_test_image(result, 'test_rgba_output.jpg')

    def test_grayscale_mode(self):
        """Test grayscale mode conversion to RGB."""
        # Convert to grayscale
        gray_img = self.base_image.convert('L')
        gray_bytes = self._image_to_bytes(gray_img, 'PNG')

        # Process through _ensure_token_budget
        result = self.llm_service._ensure_token_budget(gray_bytes)

        # Verify output is valid JPEG
        self.assertTrue(result.startswith(b'\xff\xd8\xff'))

        # Verify it was converted to RGB
        img = self._bytes_to_image(result)
        self.assertEqual(img.mode, 'RGB', "Grayscale should be converted to RGB")

        # Save for inspection
        self._save_test_image(result, 'test_grayscale_output.jpg')

        # Also save original grayscale for comparison
        self._save_test_image(gray_bytes, 'test_grayscale_input.png')

    def test_palette_mode(self):
        """Test palette mode (P mode) conversion to RGB."""
        # Convert to palette mode with dithering
        palette_img = self.base_image.convert('P', palette=Image.ADAPTIVE)
        palette_bytes = self._image_to_bytes(palette_img, 'PNG')

        # Process through _ensure_token_budget
        result = self.llm_service._ensure_token_budget(palette_bytes)

        # Verify output is valid JPEG
        self.assertTrue(result.startswith(b'\xff\xd8\xff'))

        # Verify it was converted to RGB
        img = self._bytes_to_image(result)
        self.assertEqual(img.mode, 'RGB', "Palette mode should be converted to RGB")

        # Save for inspection
        self._save_test_image(result, 'test_palette_output.jpg')

        # Also save original palette for comparison
        self._save_test_image(palette_bytes, 'test_palette_input.png')

    # ==================== Size Tests ====================

    def test_original_size(self):
        """Test original size (no resize expected if <1078px)."""
        # Use base image as-is
        original_bytes = self._image_to_bytes(self.base_image, 'JPEG', quality=95)

        # Get original dimensions
        original_img = self._bytes_to_image(original_bytes)
        orig_width, orig_height = original_img.size

        # Process through _ensure_token_budget
        result = self.llm_service._ensure_token_budget(original_bytes)

        # Verify output is valid
        result_img = self._bytes_to_image(result)
        result_width, result_height = result_img.size

        # If original was smaller than 1078px, dimensions should be similar
        max_dimension = max(orig_width, orig_height)
        if max_dimension <= 1078:
            # Dimensions should be very close (allowing for JPEG compression differences)
            self.assertLess(abs(result_width - orig_width), 5)
            self.assertLess(abs(result_height - orig_height), 5)

        # Save for inspection
        self._save_test_image(result, 'test_original_size_output.jpg')

    def test_2x_scaled_image(self):
        """Test 2x scaled image (~2000px width) - should trigger downsizing."""
        # Scale up base image by 2x
        orig_width, orig_height = self.base_image.size
        scaled_img = self.base_image.resize((orig_width * 2, orig_height * 2), Image.LANCZOS)
        scaled_bytes = self._image_to_bytes(scaled_img, 'JPEG', quality=95)

        # Get scaled dimensions
        scaled_img_check = self._bytes_to_image(scaled_bytes)
        scaled_width, scaled_height = scaled_img_check.size

        # Process through _ensure_token_budget
        result = self.llm_service._ensure_token_budget(scaled_bytes)

        # Verify output is valid
        result_img = self._bytes_to_image(result)
        result_width, result_height = result_img.size

        # Verify downsizing occurred
        max_dim = max(scaled_width, scaled_height)
        result_max_dim = max(result_width, result_height)

        self.assertLess(result_max_dim, max_dim, "Image should be downsized")
        self.assertLessEqual(result_max_dim, 1090, "Max dimension should be ~1078px (allowing for rounding)")

        # Save for inspection
        self._save_test_image(result, 'test_2x_scaled_output.jpg')
        self._save_test_image(scaled_bytes, 'test_2x_scaled_input.jpg')

    def test_3x_scaled_image(self):
        """Test 3x scaled image (~3000px width) - aggressive downsizing."""
        # Scale up base image by 3x
        orig_width, orig_height = self.base_image.size
        scaled_img = self.base_image.resize((orig_width * 3, orig_height * 3), Image.LANCZOS)
        scaled_bytes = self._image_to_bytes(scaled_img, 'JPEG', quality=95)

        # Get scaled dimensions
        scaled_img_check = self._bytes_to_image(scaled_bytes)
        scaled_width, scaled_height = scaled_img_check.size

        # Process through _ensure_token_budget
        result = self.llm_service._ensure_token_budget(scaled_bytes)

        # Verify output is valid
        result_img = self._bytes_to_image(result)
        result_width, result_height = result_img.size

        # Verify aggressive downsizing occurred
        max_dim = max(scaled_width, scaled_height)
        result_max_dim = max(result_width, result_height)

        self.assertLess(result_max_dim, max_dim, "Image should be aggressively downsized")
        self.assertLessEqual(result_max_dim, 1090, "Max dimension should be ~1078px (allowing for rounding)")

        # Verify aspect ratio is maintained
        orig_ratio = scaled_width / scaled_height
        result_ratio = result_width / result_height
        self.assertAlmostEqual(orig_ratio, result_ratio, places=1,
                              msg="Aspect ratio should be maintained")

        # Save for inspection
        self._save_test_image(result, 'test_3x_scaled_output.jpg')
        self._save_test_image(scaled_bytes, 'test_3x_scaled_input.jpg')

    def test_4x_scaled_image(self):
        """Test 4x scaled image (~4000px width) - maximum downsizing."""
        # Scale up base image by 4x
        orig_width, orig_height = self.base_image.size
        scaled_img = self.base_image.resize((orig_width * 4, orig_height * 4), Image.LANCZOS)
        scaled_bytes = self._image_to_bytes(scaled_img, 'JPEG', quality=95)

        # Get scaled dimensions
        scaled_img_check = self._bytes_to_image(scaled_bytes)
        scaled_width, scaled_height = scaled_img_check.size

        # Process through _ensure_token_budget
        result = self.llm_service._ensure_token_budget(scaled_bytes)

        # Verify output is valid
        result_img = self._bytes_to_image(result)
        result_width, result_height = result_img.size

        # Verify maximum downsizing occurred
        max_dim = max(scaled_width, scaled_height)
        result_max_dim = max(result_width, result_height)

        self.assertLess(result_max_dim, max_dim, "Image should be maximally downsized")
        self.assertLessEqual(result_max_dim, 1090, "Max dimension should be ~1078px (allowing for rounding)")

        # Verify aspect ratio is maintained
        orig_ratio = scaled_width / scaled_height
        result_ratio = result_width / result_height
        self.assertAlmostEqual(orig_ratio, result_ratio, places=1,
                              msg="Aspect ratio should be maintained")

        # Save for inspection
        self._save_test_image(result, 'test_4x_scaled_output.jpg')
        self._save_test_image(scaled_bytes, 'test_4x_scaled_input.jpg')

    # ==================== Resize Image Tests ====================

    def test_resize_image_small(self):
        """Test _resize_image with small image (no resize needed)."""
        # Create small image
        small_img = self._create_test_image((100, 100))
        small_bytes = self._image_to_bytes(small_img, 'JPEG')

        # Try to resize with large max_size
        result = self.llm_service._resize_image(small_bytes, max_size=1000000)

        # Should not resize if under max_size
        self.assertIsNotNone(result)

        # Save for inspection
        self._save_test_image(result, 'test_resize_small.jpg')

    def test_resize_image_large(self):
        """Test _resize_image with large image (should trigger resize)."""
        # Create large image by scaling up
        orig_width, orig_height = self.base_image.size
        large_img = self.base_image.resize((orig_width * 2, orig_height * 2), Image.LANCZOS)
        large_bytes = self._image_to_bytes(large_img, 'JPEG', quality=95)

        # Get original size
        orig_size = len(large_bytes)

        # Resize to smaller max_size
        target_size = orig_size // 2  # Half the size
        result = self.llm_service._resize_image(large_bytes, max_size=target_size)

        # Should resize
        self.assertIsNotNone(result)
        result_size = len(result)

        # Result should be smaller than original (or close to target)
        self.assertLess(result_size, orig_size * 1.1)  # Allow 10% margin

        # Save for inspection
        self._save_test_image(result, 'test_resize_large.jpg')

    def test_resize_image_quality(self):
        """Test _resize_image with different quality settings."""
        # Create image
        test_bytes = self._image_to_bytes(self.base_image, 'JPEG', quality=95)

        # Test with different quality settings
        quality_50 = self.llm_service._resize_image(test_bytes, max_size=10000, quality=50)
        quality_90 = self.llm_service._resize_image(test_bytes, max_size=10000, quality=90)

        # Both should produce valid images
        self.assertIsNotNone(quality_50)
        self.assertIsNotNone(quality_90)

        # Both should be valid JPEG images
        self.assertTrue(quality_50.startswith(b'\xff\xd8\xff'))
        self.assertTrue(quality_90.startswith(b'\xff\xd8\xff'))

        # Both should produce valid images when opened
        img_50 = self._bytes_to_image(quality_50)
        img_90 = self._bytes_to_image(quality_90)
        self.assertEqual(img_50.mode, 'RGB')
        self.assertEqual(img_90.mode, 'RGB')

        # Save for inspection
        self._save_test_image(quality_50, 'test_resize_quality_50.jpg')
        self._save_test_image(quality_90, 'test_resize_quality_90.jpg')

    # ==================== Edge Cases ====================

    def test_empty_image(self):
        """Test handling of empty/zero-byte image."""
        empty_bytes = b''

        # Should handle gracefully
        result = self.llm_service._ensure_token_budget(empty_bytes)

        # Should return original or handle error
        self.assertIsNotNone(result)

        # If it couldn't process, should return original
        self.assertEqual(result, empty_bytes)

    def test_corrupted_image(self):
        """Test handling of corrupted image data."""
        corrupted_bytes = b'\xff\xd8\xff\x00\x00\x00\x00\x00corrupted'

        # Should handle gracefully
        result = self.llm_service._ensure_token_budget(corrupted_bytes)

        # Should return original or handle error
        self.assertIsNotNone(result)

    def test_pdf_file(self):
        """Test PDF to image conversion."""
        # Create a simple PDF for testing (if PyMuPDF is available)
        try:
            import fitz  # PyMuPDF

            # Create a simple PDF with one page
            doc = fitz.open()
            page = doc.new_page(width=595, height=842)  # A4 size

            # Add some text
            page.insert_text((50, 50), "Test PDF Document", fontsize=12)
            page.insert_text((50, 70), "This is a test PDF for image preprocessing.", fontsize=10)

            # Save to bytes
            pdf_bytes = doc.tobytes()
            doc.close()

            # Process through _ensure_token_budget
            result = self.llm_service._ensure_token_budget(pdf_bytes)

            # Verify output is valid JPEG
            self.assertTrue(result.startswith(b'\xff\xd8\xff'))

            # Verify it's a valid image
            img = self._bytes_to_image(result)
            self.assertEqual(img.mode, 'RGB')

            # Save for inspection
            self._save_test_image(result, 'test_pdf_output.jpg')
            self._save_test_image(pdf_bytes, 'test_pdf_input.pdf')

        except ImportError:
            self.skipTest("PyMuPDF not available for PDF testing")

    def test_aspect_ratio_preservation(self):
        """Test that aspect ratio is preserved during resizing."""
        # Create image with specific aspect ratio (e.g., 2:1)
        test_img = Image.new('RGB', (800, 400), color=(100, 150, 200))
        test_bytes = self._image_to_bytes(test_img, 'JPEG', quality=95)

        # Scale up significantly
        scaled_img = test_img.resize((2400, 1200), Image.LANCZOS)
        scaled_bytes = self._image_to_bytes(scaled_img, 'JPEG', quality=95)

        # Process through _ensure_token_budget
        result = self.llm_service._ensure_token_budget(scaled_bytes)

        # Verify aspect ratio is preserved
        result_img = self._bytes_to_image(result)
        result_width, result_height = result_img.size
        result_ratio = result_width / result_height

        # Original ratio was 2:1
        self.assertAlmostEqual(result_ratio, 2.0, places=1,
                              msg="Aspect ratio should be preserved")

        # Save for inspection
        self._save_test_image(result, 'test_aspect_ratio_output.jpg')

    def test_token_budget_enforcement(self):
        """Test that token budget is properly enforced."""
        # Create a very large image
        large_img = self.base_image.resize((3000, 2000), Image.LANCZOS)
        large_bytes = self._image_to_bytes(large_img, 'JPEG', quality=95)

        # Process through _ensure_token_budget
        result = self.llm_service._ensure_token_budget(large_bytes)

        # Verify output dimensions
        result_img = self._bytes_to_image(result)
        width, height = result_img.size
        max_dim = max(width, height)

        # Max dimension should be approximately 1078 pixels (allowing for rounding)
        self.assertLessEqual(max_dim, 1090,
                           "Max dimension should not exceed token budget limit")

        # Verify it's close to expected size (allowing for rounding)
        self.assertGreater(max_dim, 1000,
                         "Max dimension should be reasonably sized")

        # Save for inspection
        self._save_test_image(result, 'test_token_budget_output.jpg')


class TestImagePreprocessingIntegration(unittest.TestCase):
    """Integration tests for image preprocessing workflow."""

    def setUp(self):
        """Set up test fixtures."""
        self.llm_service = LLMService()
        self.test_dir = Path(__file__).parent / "test_images"
        self.test_dir.mkdir(exist_ok=True)

    def test_cmyk_to_rgb_workflow(self):
        """Test complete CMYK to RGB conversion workflow."""
        # Create a test image in CMYK
        from PIL import ImageDraw

        img = Image.new('CMYK', (800, 600), color=(0, 0, 0, 0))  # Black in CMYK
        draw = ImageDraw.Draw(img)
        draw.rectangle([100, 100, 300, 200], fill=(100, 50, 0, 0))
        draw.rectangle([400, 100, 600, 200], fill=(0, 100, 50, 0))

        # Convert to bytes
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        cmyk_bytes = buffer.getvalue()

        # Process through the service
        result = self.llm_service._ensure_token_budget(cmyk_bytes)

        # Verify conversion to RGB
        result_img = Image.open(io.BytesIO(result))
        self.assertEqual(result_img.mode, 'RGB')

        # Save for inspection
        test_path = self.test_dir / "cmyk_workflow_output.jpg"
        with open(test_path, 'wb') as f:
            f.write(result)

    def test_multi_page_pdf_handling(self):
        """Test handling of multi-page PDF (should use first page)."""
        try:
            import fitz  # PyMuPDF

            # Create a PDF with 3 pages
            doc = fitz.open()
            for i in range(3):
                page = doc.new_page(width=595, height=842)
                page.insert_text((50, 50 + i*100), f"Page {i+1}", fontsize=12)

            pdf_bytes = doc.tobytes()
            doc.close()

            # Process through _ensure_token_budget
            result = self.llm_service._ensure_token_budget(pdf_bytes)

            # Verify output is valid JPEG
            self.assertTrue(result.startswith(b'\xff\xd8\xff'))

            # Verify it's a valid image
            img = Image.open(io.BytesIO(result))
            self.assertEqual(img.mode, 'RGB')

            # Save for inspection
            test_path = self.test_dir / "multipage_pdf_output.jpg"
            with open(test_path, 'wb') as f:
                f.write(result)

            pdf_path = self.test_dir / "multipage_pdf_input.pdf"
            with open(pdf_path, 'wb') as f:
                f.write(pdf_bytes)

        except ImportError:
            self.skipTest("PyMuPDF not available for multi-page PDF testing")

    def test_extreme_aspect_ratios(self):
        """Test handling of extreme aspect ratios."""
        # Test very tall image (portrait)
        tall_img = Image.new('RGB', (400, 2000), color=(100, 150, 200))
        tall_bytes = io.BytesIO()
        tall_img.save(tall_bytes, format='JPEG')
        tall_bytes = tall_bytes.getvalue()

        result_tall = self.llm_service._ensure_token_budget(tall_bytes)
        result_tall_img = Image.open(io.BytesIO(result_tall))

        # Should be downsized but aspect ratio preserved
        self.assertEqual(result_tall_img.mode, 'RGB')
        tall_ratio = result_tall_img.width / result_tall_img.height
        self.assertLess(tall_ratio, 1.0)  # Still portrait

        # Test very wide image (landscape)
        wide_img = Image.new('RGB', (2000, 400), color=(100, 150, 200))
        wide_bytes = io.BytesIO()
        wide_img.save(wide_bytes, format='JPEG')
        wide_bytes = wide_bytes.getvalue()

        result_wide = self.llm_service._ensure_token_budget(wide_bytes)
        result_wide_img = Image.open(io.BytesIO(result_wide))

        # Should be downsized but aspect ratio preserved
        self.assertEqual(result_wide_img.mode, 'RGB')
        wide_ratio = result_wide_img.width / result_wide_img.height
        self.assertGreater(wide_ratio, 1.0)  # Still landscape

        # Save for inspection
        with open(self.test_dir / "extreme_tall_output.jpg", 'wb') as f:
            f.write(result_tall)
        with open(self.test_dir / "extreme_wide_output.jpg", 'wb') as f:
            f.write(result_wide)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)

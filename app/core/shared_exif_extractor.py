"""
Shared EXIF extractor to eliminate duplicate EXIF processing across different components.
"""
import io
from typing import Dict, Any, Optional
from PIL import Image
from PIL.ExifTags import TAGS
from app.core.logger import get_logger


class SharedExifExtractor:
    """Shared EXIF extractor that provides EXIF data to multiple components."""

    def __init__(self):
        self.logger = get_logger()

    def extract_exif_data(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Extract EXIF data from image bytes once and return it in multiple formats.

        Args:
            image_bytes: Raw image bytes

        Returns:
            Dictionary with different EXIF data formats for various consumers
        """
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                # Get raw EXIF data
                raw_exif = img._getexif()

                if raw_exif is None:
                    self.logger.info("No EXIF data found in image")
                    return {
                        'has_exif': False,
                        'raw_exif': {},
                        'readable_exif': {},
                        'photoholmes_exif': {},
                        'validation_exif': {}
                    }

                # Extract readable EXIF data (for validation)
                readable_exif = {}
                for tag_id, value in raw_exif.items():
                    tag_name = TAGS.get(tag_id, str(tag_id))
                    # Convert bytes to string if needed
                    if isinstance(value, bytes):
                        try:
                            value = value.decode('utf-8', errors='ignore')
                        except:
                            value = str(value)
                    readable_exif[tag_name] = value

                # Prepare EXIF for PhotoHolmes EXIF As Language method
                photoholmes_exif = {}
                for tag_id, value in raw_exif.items():
                    tag_name = str(tag_id)  # PhotoHolmes expects string keys
                    try:
                        # Test serialization for PhotoHolmes
                        import json
                        json.dumps(value)
                        photoholmes_exif[tag_name] = str(value)
                    except (TypeError, ValueError):
                        photoholmes_exif[tag_name] = f"non_serializable_{type(value).__name__}"

                # Prepare EXIF for validation
                validation_exif = readable_exif.copy()

                self.logger.info(f"EXIF data extracted successfully: {len(readable_exif)} tags")

                return {
                    'has_exif': True,
                    'raw_exif': raw_exif,
                    'readable_exif': readable_exif,
                    'photoholmes_exif': photoholmes_exif,
                    'validation_exif': validation_exif
                }

        except Exception as e:
            self.logger.error(f"Failed to extract EXIF data: {str(e)}")
            return {
                'has_exif': False,
                'raw_exif': {},
                'readable_exif': {},
                'photoholmes_exif': {},
                'validation_exif': {},
                'error': str(e)
            }


# Global instance for shared use
_shared_extractor = None

def get_shared_exif_extractor() -> SharedExifExtractor:
    """Get global shared EXIF extractor instance."""
    global _shared_extractor
    if _shared_extractor is None:
        _shared_extractor = SharedExifExtractor()
    return _shared_extractor
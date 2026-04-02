from pathlib import Path
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from app.core import logger

class ImageValidationService:
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
    SUPPORTED_CONTENT_TYPES = {'image/jpeg', 'image/png', 'image/webp'}
    RGB_MODE = 'RGB'
    
    def __init__(self):
        self.logger = logger
    
    async def validate_and_convert_image(
        self, 
        image_bytes: bytes, 
    ) -> bytes:
        try:
            try:
                img = Image.open(BytesIO(image_bytes))
                img.verify()  
                img = Image.open(BytesIO(image_bytes))
            except (UnidentifiedImageError, Exception) as e:
                raise ValueError(f"Invalid image file: {str(e)}")        
            if img.mode != self.RGB_MODE:
                img = img.convert(self.RGB_MODE)
            
            output = BytesIO()
            img.save(output, format='JPEG', quality=95)
            processed_bytes = output.getvalue()
            
            return processed_bytes
            
        except Exception as e:
            self.logger.error(f"Image validation/processing failed: {str(e)}")
            raise ValueError(f"Failed to process image: {str(e)}")
    
    def validate_content_type(self, content_type: str) -> None:
        if content_type.lower() not in self.SUPPORTED_CONTENT_TYPES:
            raise ValueError(f"Unsupported content type: {content_type}. "
                           f"Supported types: {', '.join(self.SUPPORTED_CONTENT_TYPES)}")
    
    def get_file_extension(self, filename: str) -> str:
        return Path(filename).suffix.lower()
    
    def is_extension_supported(self, filename: str) -> bool:
        ext = self.get_file_extension(filename)
        return ext in self.SUPPORTED_EXTENSIONS

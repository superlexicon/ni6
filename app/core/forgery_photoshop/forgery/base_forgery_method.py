import os
import tempfile
from abc import ABC, abstractmethod
from typing import Any, Dict, TypeVar, Generic

from app.core.logger import get_logger

# Enable MPS fallback for unsupported PyTorch operations on Apple Silicon
# This is needed for AdaptiveCFANet which uses dilated convolutions not supported on MPS
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

T = TypeVar('T')

class BaseForgeryMethod(ABC, Generic[T]):
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg'}
    
    def __init__(self):
        self.logger = get_logger()
        self.logger.info(f"{self.__class__.__name__} initialized")
    
    async def _save_temp_image(self, image_bytes: bytes) -> str:
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        try:
            temp_file.write(image_bytes)
            return temp_file.name
        except Exception as e:
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
            self.logger.error(f"Failed to save temporary image: {e}")
            raise
        finally:
            temp_file.close()
    
    @abstractmethod
    async def _process_image(self, image_bytes: bytes) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def _analyze(self, processed_data: Dict[str, Any]) -> T:
        pass
    
    async def analyze_image(self, image_bytes: bytes) -> T:
        self.logger.info(f"Processing image with {self.__class__.__name__}")

        try:
            processed_data = await self._process_image(image_bytes)
            # Run the synchronous _analyze method in a thread to prevent blocking
            import asyncio
            result = await asyncio.to_thread(self._analyze, processed_data)

            self.logger.info(f"{self.__class__.__name__} analysis completed successfully")
            return result

        except Exception as e:
            self.logger.error(f"Error in {self.__class__.__name__}: {str(e)}")
            raise ValueError(f"Image analysis failed: {str(e)}")
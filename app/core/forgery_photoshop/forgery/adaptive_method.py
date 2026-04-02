import os
from typing import Dict, Any, Optional
import numpy as np
import torch
from app.photoholmes.utils.image import read_image
from app.photoholmes.methods.adaptive_cfa_net import adaptive_cfa_net_preprocessing

from app.core import adaptive_get_model
from app.dto import AdaptiveMethodData
from .base_forgery_method import BaseForgeryMethod

# Enable MPS fallback for unsupported operations (dilated convolutions)
# This allows AdaptiveCFANet to run on Apple Silicon with CPU fallback for unsupported ops
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

class AdaptiveMethod(BaseForgeryMethod[AdaptiveMethodData]):
    TAMPERED_THRESHOLD = 0.55
    
    def __init__(self):
        super().__init__()
        self.method = adaptive_get_model()
    
    async def _process_image(self, image_bytes: bytes) -> Dict[str, Any]:
        try:
            temp_path = await self._save_temp_image(image_bytes)
            try:
                # read_image already returns a tensor in the correct format
                image = read_image(temp_path)
                # No conversion needed - read_image handles the conversion
                return {"image": image}
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        except Exception as e:
            self.logger.error(f"Error during image processing: {e}")
            raise ValueError(f"Image processing failed: {e}")
    
    def _analyze(self, processed_data: Dict[str, Any]) -> AdaptiveMethodData:
        try:
            input_data = adaptive_cfa_net_preprocessing(**processed_data)

            # Ensure tensor has batch dimension
            image_tensor = input_data['image']
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0)

            # Move tensor to same device as model (FIX: device mismatch error)
            model_device = next(self.method.parameters()).device
            image_tensor = image_tensor.to(model_device)

            # Forward pass
            pred = self.method.forward(image_tensor)
            pred = torch.exp(pred)

            # Apply channel permutation (fix device placement)
            device = pred.device
            pred[:, 1] = pred[torch.tensor([1, 0, 3, 2], device=device), 1]
            pred[:, 2] = pred[torch.tensor([2, 3, 0, 1], device=device), 2]
            pred[:, 3] = pred[torch.tensor([3, 2, 1, 0], device=device), 3]
            pred = torch.mean(pred, dim=1)

            # Calculate confidence
            best_grid = torch.argmax(torch.mean(pred, dim=(1, 2)))
            authentic = torch.argmax(pred, dim=0) == best_grid
            confidence = 1 - torch.max(pred, dim=0).values
            confidence = torch.clamp(confidence, 0, 1)
            confidence[authentic] = 1
            output = 1 - confidence

            tampered_ratio = self._calculate_tampered_ratio(output)
            self.logger.info(f"Calculated tampered ratio: {tampered_ratio:.4f}")
            return AdaptiveMethodData(tampered_ratio=tampered_ratio)
        except Exception as e:
            self.logger.error(f"Error during analysis: {e}")
            raise
    
    def _calculate_tampered_ratio(self, output: torch.Tensor) -> float:
        tampered_mask = (output > self.TAMPERED_THRESHOLD).float()
        return (tampered_mask.sum() / tampered_mask.numel()).item()
    
    async def adaptive_method(self, image_bytes: bytes, pre_decoded_image: Optional[np.ndarray] = None) -> AdaptiveMethodData:
        # Note: pre_decoded_image not used - Adaptive requires temp file pipeline
        return await self.analyze_image(image_bytes)
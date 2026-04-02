"""
Optimized FOCAL processing with ViT-friendly configurations.

ViT comes in many variants, but pre-trained weights are tied to specific
img_size/patch_size combinations. Common ViT variants include:
- ViT-Base: 224x224, 16x16 patches, 768 dim
- ViT-Large: 224x224, 16x16 patches, 1024 dim
- ViT-Huge: 224x224, 14x14 patches, 1280 dim
- Custom: Various img_size/patch_size combinations

The FOCAL method uses a SAM-style ViT with img_size=1024, patch_size=16.
Instead of changing the ViT architecture (which breaks pre-trained weights),
we'll use the same configuration but optimize other aspects.
"""
from typing import Optional, Tuple, Type
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode
from app.photoholmes.methods.focal import Focal
from app.core.logger import get_logger


class OptimizedViTFocal:
    """Performance-optimized FOCAL wrapper using original ViT with smart optimizations."""

    def __init__(self, focal_instance: Focal, logger):
        self.focal_instance = focal_instance
        self.logger = logger
        self.device = focal_instance.device

    def predict(self, image: torch.Tensor):
        """
        Optimized prediction using original ViT configuration with processing optimizations.

        Args:
            image (torch.Tensor): Input image of shape (C, H, W)

        Returns:
            Tensor: Binary mask of shape (H, W)
        """
        if len(image.shape) != 3:
            raise ValueError("Input image should be of shape (C, H, W)")
        _, im_H, im_W = image.shape

        # Use original 1024x1024 for ViT compatibility but optimize the processing
        target_size = [1024, 1024]
        self.logger.info(f"OptimizedViT: Using original size {target_size} (ViT pre-trained requirements)")

        # Optimized preprocessing with performance improvements
        # Use bilinear instead of bicubic for faster processing
        image_processed = resize(
            image,
            target_size,
            interpolation=InterpolationMode.BILINEAR,
            antialias=False  # Disable antialiasing for speed
        ).to(self.device) / 255.0

        # Use mixed precision if available (but NOT on ROCm - causes type mismatch)
        is_cuda = str(self.device).startswith('cuda')
        is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
        use_autocast = is_cuda and not is_rocm

        with torch.cuda.amp.autocast(enabled=use_autocast):
            with torch.no_grad():
                # Run forward pass with original ViT
                Fo = self.focal_instance.forward(image_processed[None, :])
                _, W, H, _ = Fo.shape
                Fo = Fo.flatten(1, 2)

        # Perform clustering
        result = self.focal_instance.clustering(x=Fo, k=2)

        Lo = result.labels
        if torch.sum(Lo) > torch.sum(1 - Lo):
            Lo = 1 - Lo
        Lo = Lo.view(H, W)

        # Optimized resize back to original dimensions
        mask = resize(
            Lo.unsqueeze(0),
            [im_H, im_W],
            interpolation=InterpolationMode.NEAREST,  # Use nearest for binary mask
        ).squeeze(0).float()

        return mask
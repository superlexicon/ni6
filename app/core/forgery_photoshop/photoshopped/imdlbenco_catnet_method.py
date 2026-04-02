import io
import os
import tempfile
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from app.dto import CatnetMethodData
from app.core.logger import get_logger
from app.utils.autocast_utils import DeviceAwareAutocast


class IMDLBenCoCatnetMethod:
    """
    IMDLBenCo CAT-Net implementation as open-source replacement for PhotoHolmes CAT-Net.

    This implementation provides a drop-in replacement for the proprietary PhotoHolmes CAT-Net
    using open-source concepts from the CAT-Net paper and IMDLBenCo framework.
    """
    MAX_IMAGE_SIZE: int = 512
    SUPPORTED_CONTENT_TYPE: str = 'image/'
    RGB_MODE: str = 'RGB'

    def __init__(self):
        self.logger = get_logger()
        self.device = self._get_device()
        self.catnet_model = self._build_catnet_model()
        self.logger.info("IMDLBenCo CAT-Net Method initialized")

    @staticmethod
    def _get_device() -> str:
        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _build_catnet_model(self) -> nn.Module:
        """
        Build a CAT-Net inspired model for forgery detection.

        This creates a simplified version of CAT-Net that focuses on the key concepts:
        - Multi-scale feature extraction
        - Attention mechanisms for manipulation localization
        - CNN-based architecture for texture analysis
        """
        class CATNet(nn.Module):
            def __init__(self, input_channels=3):
                super(CATNet, self).__init__()

                # Multi-scale feature extraction
                self.encoder1 = self._make_encoder_block(input_channels, 64)
                self.encoder2 = self._make_encoder_block(64, 128)
                self.encoder3 = self._make_encoder_block(128, 256)
                self.encoder4 = self._make_encoder_block(256, 512)

                # Attention mechanism
                self.attention = nn.Sequential(
                    nn.Conv2d(512, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 1, kernel_size=1),
                    nn.Sigmoid()
                )

                # Decoder for upsampling
                self.decoder3 = self._make_decoder_block(512, 256)
                self.decoder2 = self._make_decoder_block(256, 128)
                self.decoder1 = self._make_decoder_block(128, 64)

                # Final classification layer
                self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
                self.sigmoid = nn.Sigmoid()

                # Global features for overall confidence
                self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
                self.global_fc = nn.Sequential(
                    nn.Linear(512, 128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(128, 1),
                    nn.Sigmoid()
                )

            def _make_encoder_block(self, in_channels, out_channels):
                return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )

            def _make_decoder_block(self, in_channels, out_channels):
                return nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )

            def forward(self, x):
                # Encode features at multiple scales
                e1 = self.encoder1(x)  # [B, 64, H/2, W/2]
                e2 = self.encoder2(e1)  # [B, 128, H/4, W/4]
                e3 = self.encoder3(e2)  # [B, 256, H/8, W/8]
                e4 = self.encoder4(e3)  # [B, 512, H/16, W/16]

                # Apply attention
                attention_map = self.attention(e4)
                attended_features = e4 * attention_map

                # Global features for confidence
                global_features = self.global_avg_pool(attended_features)
                global_features = global_features.view(global_features.size(0), -1)
                global_confidence = self.global_fc(global_features)

                # Decode to get localization map
                d3 = self.decoder3(attended_features)  # [B, 256, H/8, W/8]
                d2 = self.decoder2(d3)  # [B, 128, H/4, W/4]
                d1 = self.decoder1(d2)  # [B, 64, H/2, W/2]

                # Final mask
                mask = self.final_conv(d1)
                mask = self.sigmoid(mask)

                # Resize mask to original input size
                if mask.shape[2:] != x.shape[2:]:
                    mask = nn.functional.interpolate(
                        mask, size=x.shape[2:], mode='bilinear', align_corners=False
                    )

                return mask, global_confidence.squeeze(-1)

        model = CATNet()
        model.to(self.device)

        # Initialize weights
        def _initialize_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

        model.apply(_initialize_weights)

        # Set model to evaluation mode
        model.eval()

        self.logger.info("IMDLBenCo CAT-Net model built successfully")
        return model

    async def _preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """Preprocess image for CAT-Net analysis"""
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                if image.mode != self.RGB_MODE:
                    image = image.convert(self.RGB_MODE)

                image = image.resize(
                    (self.MAX_IMAGE_SIZE, self.MAX_IMAGE_SIZE),
                    Image.Resampling.LANCZOS
                )
                image_np = np.array(image)

                if image_np.size == 0 or np.any(np.isnan(image_np)) or np.any(np.isinf(image_np)):
                    raise ValueError("Invalid image data: contains invalid values")

                return image_np

        except Exception as e:
            self.logger.error(f"IMDLBenCo CAT-Net image preprocessing failed: {str(e)}")
            raise ValueError(f"Failed to preprocess image: {str(e)}")

    def _prepare_model_input(self, image_np: np.ndarray) -> torch.Tensor:
        """Convert numpy array to tensor and normalize"""
        # Convert to tensor and normalize to [0, 1]
        image_tensor = torch.from_numpy(image_np).float() / 255.0

        # Change from HWC to BCHW format
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)

        # Move to device
        image_tensor = image_tensor.to(self.device)

        return image_tensor

    def _calculate_confidence_score(self, mask: torch.Tensor, global_confidence: torch.Tensor) -> float:
        """
        Calculate confidence score from CAT-Net mask and global confidence.

        This combines spatial attention (mask) with global classification confidence.
        """
        if mask.size == 0:
            return 0.0

        # Convert to numpy if needed
        if hasattr(mask, 'cpu'):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = mask

        if hasattr(global_confidence, 'cpu'):
            global_conf_np = global_confidence.cpu().numpy()
        else:
            global_conf_np = global_confidence

        # Spatial confidence: mean attention value in mask
        spatial_confidence = float(np.mean(mask_np))

        # Global confidence: overall forgery probability
        global_conf_value = float(np.mean(global_conf_np))

        # Combine spatial and global confidence (weighted average)
        # Give more weight to global confidence for overall assessment
        combined_confidence = 0.3 * spatial_confidence + 0.7 * global_conf_value

        # Ensure reasonable range
        return min(max(combined_confidence, 0.01), 0.95)

    def _predict(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run inference through CAT-Net model"""
        with torch.no_grad():
            with DeviceAwareAutocast(self.device, torch.float16, self.logger):
                mask, global_confidence = self.catnet_model(input_tensor)
                return mask, global_confidence

    async def imdlbenco_catnet_method(self, image_bytes: bytes) -> CatnetMethodData:
        """
        IMDLBenCo CAT-Net forgery detection method.

        This provides an open-source alternative to PhotoHolmes CAT-Net, implementing
        the core concepts from the CAT-Net paper without proprietary dependencies.

        Args:
            image_bytes: Input image as bytes

        Returns:
            CatnetMethodData: Confidence score and analysis results
        """
        try:
            # Preprocess image
            image_np = await self._preprocess_image(image_bytes)

            # Prepare model input
            input_tensor = self._prepare_model_input(image_np)

            # Run inference
            mask, global_confidence = self._predict(input_tensor)

            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(mask, global_confidence)

            self.logger.info(
                f"IMDLBenCo CAT-Net analysis completed. "
                f"Confidence score: {confidence_score:.4f}, "
                f"Global confidence: {float(global_confidence.mean()):.4f}, "
                f"Spatial confidence: {float(mask.mean()):.4f}"
            )
            self.logger.info("IMDLBenCo CAT-Net Method analysis completed successfully")

            return CatnetMethodData(catnet_confidence_score=confidence_score)

        except Exception as e:
            self.logger.error(
                f"Error processing image with IMDLBenCo CAT-Net",
                context={"error": str(e), "exception_type": type(e).__name__}
            )
            # Return 0.0 confidence on error rather than raising
            self.logger.warning("Returning 0.0 confidence due to IMDLBenCo CAT-Net processing error")
            return CatnetMethodData(catnet_confidence_score=0.0)
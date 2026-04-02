# Code derived from
# https://github.com/HighwayWu/FOCAL/tree/main
from typing import Any, Dict, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.transforms.functional import resize, to_tensor
from typing_extensions import TypedDict
from PIL import Image

from ...base import BaseTorchMethod, BenchmarkOutput
from app.core.gpu_manager import ModelType

from .utils import load_weights

try:
    from torch_kmeans import KMeans
except ModuleNotFoundError:
    raise ImportError(
        "To use the Focal method, you need to install the `torch_kmeans` package."
    )


class FocalWeights(TypedDict):
    ViT: Union[str, Dict[str, Any]]
    HRNet: Union[str, Dict[str, Any]]


class Focal(BaseTorchMethod):
    """
    Implementation of Focal method [Wu et al., 2023].

    Focal is an end to end neural network.
    """

    def __init__(
        self,
        weights: FocalWeights,
        device: str = "cuda",
        **kwargs,
    ):
        """
        Args:
            weights (FocalWeights): Weights for the Focal model. The weights
                should be a dictionary with keys "ViT" and "HRNet". The values
                should be the path to the weights file or a dictionary with
                the weights.
            device (str): Device to run the model on.
        """
        # Initialize the GPU-enabled base class
        super().__init__(device=device, model_type=ModelType.PHOTOHOLMES, **kwargs)

        self.network_list = nn.ModuleList()

        for net_name, w in weights.items():
            if net_name == "HRNet":
                from .models.hrnet import HRNet

                net = HRNet()
                load_weights(net, w)  # type: ignore[arg-type]

                self.network_list.append(net)

            elif net_name == "ViT":
                from .models.vit import ImageEncoderViT

                net = ImageEncoderViT()
                load_weights(net, w)  # type: ignore[arg-type]

                self.network_list.append(net)
            else:
                raise ValueError(f"Unknown network {net_name}")

            self.clustering = KMeans(verbose=False)

        for net in self.network_list:
            net.eval()

        # Move all submodules (network_list) to the target device
        self.to(self.device)

    def load_model(self) -> None:
        """
        Load model weights.

        For Focal, models are already loaded in __init__, so this is a no-op.
        """
        pass  # Models are loaded in __init__

    def preprocess(self, image: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Preprocess input image for Focal inference.

        Args:
            image: Input image (path, numpy array, or PIL Image)

        Returns:
            Preprocessed tensor of shape (C, H, W)
        """
        # Load image to PIL format
        pil_image = self.load_image(image)

        # Convert to tensor (C, H, W) with values in [0, 255]
        tensor = to_tensor(pil_image) * 255.0

        return tensor

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input image of shape (B, C, H, W)

        Returns:
            Tensor: Output of the network
        """
        Fo = self.network_list[0](x)
        Fo = Fo.permute(0, 2, 3, 1)

        _, H, W, _ = Fo.shape
        Fo = F.normalize(Fo, dim=3)
        Fo_list = [Fo]

        for additional_net in self.network_list[1:]:
            Fo_add = additional_net(x)
            Fo_add = F.interpolate(Fo_add, (H, W))
            Fo_add = Fo_add.permute(0, 2, 3, 1)
            Fo_add = F.normalize(Fo_add, dim=3)
            Fo_list.append(Fo_add)

        Fo = torch.cat(Fo_list, dim=3)

        return Fo

    def predict(self, image: torch.Tensor):  # type: ignore[override]
        """
        Run a prediction over a preprocessed image. You can use the pipeline
        `focal_preprocessing` provied in `photoholmes.methods.focal.preprocessing`.

        Args:
            image (torch.Tensor): Input image of shape (C, H, W)

        Returns:
            Tensor: Binary mask of shape (H, W)
        """
        if len(image.shape) != 3:
            raise ValueError("Input image should be of shape (C, H, W)")
        _, im_H, im_W = image.shape

        # This operation destroys traces that typically indicate the presence of a
        # forgery. This indicates the method is most likely overfitted to
        # the dataset or to semantic forgery.
        image = resize(image, [1024, 1024]).to(self.device) / 255.0

        with torch.no_grad():
            Fo = self.forward(image[None, :])
            _, W, H, _ = Fo.shape
            Fo = Fo.flatten(1, 2)

        result = self.clustering(x=Fo, k=2)

        Lo = result.labels
        if torch.sum(Lo) > torch.sum(1 - Lo):
            Lo = 1 - Lo
        Lo = Lo.view(H, W)
        mask = resize(Lo.unsqueeze(0), [im_H, im_W]).squeeze(0).float()

        return mask

    def benchmark(  # type: ignore[override]
        self, image: torch.Tensor
    ) -> BenchmarkOutput:
        """
        Benchmarks the Focal method using the provided image.

        Args:
            image (Tensor): Input image tensor.

        Returns:
            BenchmarkOutput: Contains the mask and placeholders for heatmap and
            detection.
        """
        mask = self.predict(image)
        return {"mask": mask, "heatmap": None, "detection": None}

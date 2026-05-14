from typing import Any, Dict, Union

import torch
import torch.nn as nn


def load_weights(
    model: nn.Module, weights: Union[str, Dict[str, Any]], device: str = "cuda"
):
    """
    Load weights into a model.

    Args:
        model (nn.Module): model to load the weights into.
        weights (str | dict): path to the weights file or the weights themselves.
        device (str): device to run the model on.

    Note:
        Handles weights saved with DataParallel wrapper and/or nested model structure.
        For example, VIT weights from SAM checkpoint have 'module.net.image_encoder.*' prefix.
    """
    if isinstance(weights, str):
        weights_ = torch.load(weights, map_location=torch.device(device))
    else:
        weights_ = weights

    # Get the model's expected keys to filter irrelevant weights
    model_keys = set(model.state_dict().keys())

    # Handle weights saved with DataParallel wrapper and/or nested model structure
    # The VIT weights were saved from SAM model with structure: module.net.image_encoder.*
    # But we're loading into a bare ImageEncoderViT class
    new_state_dict = {}
    for k, v in weights_.items():
        new_key = k
        # Remove 'module.' prefix if present (DataParallel wrapper)
        if new_key.startswith("module."):
            new_key = new_key[7:]  # len("module.") = 7
        # Remove 'net.image_encoder.' prefix if present (SAM model structure)
        if new_key.startswith("net.image_encoder."):
            new_key = new_key[20:]  # len("net.image_encoder.") = 20
        # Remove 'net.' prefix if present (for HRNet or other models)
        if new_key.startswith("net."):
            new_key = new_key[4:]  # len("net.") = 4
        # Only include keys that match the model's state dict
        if new_key in model_keys:
            new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)

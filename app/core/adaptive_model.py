import os

from app.photoholmes.methods.adaptive_cfa_net import AdaptiveCFANet
import torch
from typing import Optional


class AdaptiveModel:
    _instance: Optional['AdaptiveModel'] = None
    _model: Optional[AdaptiveCFANet] = None
    _initialized: bool = False

    ARCH_CONFIG = "pretrained"
    # Use integrated photoholmes weights path
    ADAPTIVE_WEIGHTS = "app/photoholmes/weights/adaptive_cfa_net/weights.pth"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AdaptiveModel, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.device = self._get_device()
            self._initialized = True

    def _get_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda:0"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def get_model(self) -> AdaptiveCFANet:
        if self._model is None:
            try:
                self._model = AdaptiveCFANet(
                    arch_config=self.ARCH_CONFIG,
                    weights=self.ADAPTIVE_WEIGHTS,
                )
                self._model.to_device(self.device)
                self._model.eval()
            except Exception as e:
                raise RuntimeError(f"Model initialization failed: {str(e)}")
        return self._model


def adaptive_init_model():
    pass


def adaptive_get_model() -> AdaptiveCFANet:
    return AdaptiveModel().get_model()

"""Warpper for ImageNet ResNet model."""

from __future__ import annotations

import logging

from torch import nn
from torchvision import models

from src.models.timm_model import TimmModel
from src.utils.types import BatchImages, BatchLogits

logger = logging.getLogger(__name__)


class ResNet50(nn.Module):
    """Wrapper for ImageNet ResNet model."""

    def __init__(self, **kwargs):
        """Initialize ResNet50 model."""
        super().__init__()
        _ = kwargs  # Unused
        self._wrapped_model = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT
        )

    def forward(self, inputs: BatchImages) -> BatchLogits:
        """Forward pass."""
        return self._wrapped_model(inputs)


class SwinL(TimmModel):

    def __init__(self, **kwargs):
        super().__init__(dataset="imagenet", arch="timm-swin-l", **kwargs)


class SwinB(nn.Module):

    def __init__(self, **kwargs):
        """Initialize Swin-B model."""
        super().__init__()
        _ = kwargs  # Unused
        self._wrapped_model = models.swin_b(
            weights=models.Swin_B_Weights.IMAGENET1K_V1
        )

    def forward(self, inputs: BatchImages) -> BatchLogits:
        """Forward pass."""
        return self._wrapped_model(inputs)

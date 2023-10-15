"""Input normalization layer."""

import torch
from torch import nn

from src.utils.types import BatchImages, BatchLogits


class Normalize(nn.Module):
    """Normalize images by mean and std."""

    def __init__(self, mean, std, *args, **kwargs) -> None:
        """Initialize Normalize.

        Args:
            mean: Mean of images per-chaneel.
            std: Std of images per-channel.
        """
        _ = args, kwargs  # Unused
        super().__init__()
        if mean is None or std is None:
            self.mean, self.std = None, None
        else:
            self.register_buffer(
                "mean", torch.tensor(mean)[None, :, None, None]
            )
            self.register_buffer("std", torch.tensor(std)[None, :, None, None])

    def forward(self, images: BatchImages) -> BatchImages:
        """Normalize images."""
        if self.mean is None:
            return images
        return (images - self.mean) / self.std


class Postprocess(nn.Module):
    """Postprocess logits by temperature scaling."""

    def __init__(self, temperature: float = 1.0) -> None:
        """Initialize Postprocess.

        Args:
            temperature: Temperature scaling. Defaults to 1.0.
        """
        super().__init__()
        if temperature <= 0:
            raise ValueError(
                f"Temperature must be positive, got {temperature}!"
            )
        self._temperature: float = temperature

    def forward(self, logits: BatchLogits) -> BatchLogits:
        """Scale logits by temperature."""
        logits = logits / self._temperature
        # Clamp to avoid overflow/underflow
        logits.clamp_(-32, 32)
        return logits

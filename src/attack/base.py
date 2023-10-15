"""Base Attack module."""

from __future__ import annotations

from typing import Callable, Tuple

import torch
from torch import nn

from src.utils.types import BatchImages, BatchLabels

_LossFn = Callable[[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]


class AttackModule(nn.Module):
    """Base Attack module."""

    def __init__(
        self,
        attack_config: dict[str, str | int | float],
        core_model: nn.Module,
        loss_fn: _LossFn,
        norm: str = "Linf",
        eps: float = 8 / 255,
        num_classes: int = 10,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        if norm not in ("L2", "Linf"):
            raise NotImplementedError("norm must be either L2 or Linf!")
        _ = attack_config, kwargs  # Unused
        self._core_model: nn.Module = core_model
        self._loss_fn: _LossFn = loss_fn
        self._eps: float = eps
        self._norm: str = norm
        self._verbose: bool = verbose
        self._num_classes: int = num_classes
        self.dual_losses: bool = False

    def forward(self, inputs: BatchImages, targets: BatchLabels) -> BatchImages:
        """Forward pass of the attack."""
        raise NotImplementedError("Attack module must implement forward!")

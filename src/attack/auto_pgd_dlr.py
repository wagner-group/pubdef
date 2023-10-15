"""Auto-PGD Attack with DLR loss from AutoAttack."""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from src.attack.auto import AutoAttack


class AutoPGDDLRAttack(AutoAttack):
    """Auto-PGD Attack with DLR loss."""

    def __init__(
        self,
        attack_config: dict[str, str | int | float],
        core_model: nn.Module,
        loss_fn: Callable[[tuple[torch.Tensor, torch.Tensor]], torch.Tensor],
        norm: str = "Linf",
        eps: float = 8 / 255,
        **kwargs,
    ) -> None:
        """Initialize Auto-PGD Attack with DLR loss."""
        super().__init__(
            attack_config,
            core_model,
            loss_fn,
            norm,
            eps,
            attacks_to_run=["apgd-dlr"],
            **kwargs,
        )

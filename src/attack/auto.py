"""Wrapper AutoAttack."""

from __future__ import annotations

import logging
from typing import Callable

import torch
from autoattack import AutoAttack as OrigAutoAttack
from torch import nn

from src.attack.base import AttackModule
from src.utils.types import BatchImages, BatchLabels

logger = logging.getLogger(__name__)


class AutoAttack(AttackModule):
    """Wrapper for AutoAttack."""

    def __init__(
        self,
        attack_config: dict[str, str | int | float],
        core_model: nn.Module,
        loss_fn: Callable[[tuple[torch.Tensor, torch.Tensor]], torch.Tensor],
        norm: str = "Linf",
        eps: float = 8 / 255,
        attacks_to_run: list[str] | None = None,
        **kwargs,
    ) -> None:
        """Initialize AutoAttack wrapper."""
        super().__init__(
            attack_config, core_model, loss_fn, norm, eps, **kwargs
        )
        version = "standard"
        if isinstance(attacks_to_run, (list, tuple)):
            logger.info("Running %s from AutoAttack", str(attacks_to_run))
            version = "custom"
        logger.info("Running AutoAttack version: %s", version)
        self._adversary = OrigAutoAttack(
            self._core_model,
            norm=self._norm,
            eps=self._eps,
            version=version,
            verbose=self._verbose,
            attacks_to_run=attacks_to_run,
        )

    def forward(self, inputs: BatchImages, targets: BatchLabels) -> BatchImages:
        """Runs attack and returns adversarial examples."""
        mode = self._core_model.training
        self._core_model.eval()
        x_adv = self._adversary.run_standard_evaluation(
            inputs, targets.argmax(1), bs=inputs.shape[0]
        )
        self._core_model.train(mode)
        return x_adv

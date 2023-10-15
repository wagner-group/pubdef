"""Momentum PGD Attack.

References:
- Dong et al., Boosting Adversarial Attacks with Momentum, 2018.
  https://arxiv.org/abs/1710.06081
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from src.attack.pgd import PGDAttack


class MomentumPGDAttack(PGDAttack):
    """Momentum PGD Attack."""

    def __init__(
        self,
        attack_config: dict[str, str | int | float],
        core_model: nn.Module,
        loss_fn: Callable[[tuple[torch.Tensor, torch.Tensor]], torch.Tensor],
        norm: str = "Linf",
        eps: float = 8 / 255,
        **kwargs,
    ):
        """Initialize MomentumPGDAttack."""
        super().__init__(
            attack_config, core_model, loss_fn, norm, eps, **kwargs
        )
        # Momentum decaying term
        self._mu: float = attack_config.get("mpgd_mu", 1.0)

    def _forward_linf(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        mode = self._core_model.training
        self._core_model.eval()

        # Initialize worst-case inputs
        x_adv_worst = inputs.clone().detach()
        worst_losses = torch.zeros(len(inputs), 1, 1, 1, device=inputs.device)

        # Repeat PGD for specified number of restarts
        for _ in range(self._num_restarts):
            x_adv = inputs.clone().detach()
            momentum = torch.zeros_like(x_adv)

            # Initialize adversarial inputs
            x_adv += torch.zeros_like(x_adv).uniform_(-self._eps, self._eps)
            x_adv.clamp_(0, 1)

            # Run PGD on inputs for specified number of steps
            for _ in range(self._num_steps):
                grads = self._get_grad(x_adv, targets)
                with torch.no_grad():
                    x_adv.detach_()
                    grads /= grads.abs().mean((1, 2, 3), keepdim=True)
                    momentum.mul_(self._mu).add_(grads)
                    # Perform gradient update, project to norm ball
                    x_adv += self._step_size * torch.sign(momentum)
                    x_adv = self._project_linf(x_adv, inputs)

            x_adv_worst, worst_losses = self._get_best_adv(
                x_adv, targets, x_adv_worst, worst_losses
            )

        # Return worst-case perturbed input logits
        self._core_model.train(mode)
        return x_adv_worst.detach()

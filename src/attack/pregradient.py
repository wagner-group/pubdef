"""Pregradient-PGD Attack.

References:
- Xiaosen Wang, Jiadong Lin, Han Hu, Jingdong Wang, Kun He, "Boosting
  Adversarial Transferability through Enhanced Momentum," 2021.
  https://arxiv.org/abs/2103.10609
- https://github.com/ZhengyuZhao/TransferAttackEval/blob/main/attacks/gradient_stabilization_attacks.py#L182
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from src.attack.pgd import PGDAttack


class PregradientPGDAttack(PGDAttack):
    """Pregradient-PGD Attack."""

    def __init__(
        self,
        attack_config: dict[str, str | int | float],
        core_model: nn.Module,
        loss_fn: Callable[[tuple[torch.Tensor, torch.Tensor]], torch.Tensor],
        norm: str = "Linf",
        eps: float = 8 / 255,
        **kwargs,
    ):
        """Initialize PregradientPGDAttack."""
        super().__init__(
            attack_config, core_model, loss_fn, norm, eps, **kwargs
        )
        # Momentum decaying term
        self._mu: float = attack_config.get("pregradient_mu", 1.0)

    def _forward_linf(self, x, y):
        mode = self._core_model.training
        self._core_model.eval()

        # Initialize worst-case inputs
        x_adv_worst = x.clone().detach()
        worst_losses = torch.zeros(len(x), 1, 1, 1, device=x.device)

        # Repeat PGD for specified number of restarts
        for _ in range(self._num_restarts):
            x_adv = x.clone().detach()
            grad_prev = torch.zeros_like(x_adv)
            grad_t = grad_prev.clone()

            # Initialize adversarial inputs
            x_adv += torch.zeros_like(x_adv).uniform_(-self._eps, self._eps)
            x_adv.clamp_(0, 1)

            # Run PGD on inputs for specified number of steps
            for _ in range(self._num_steps):
                x_adv_t = x_adv + self._step_size * self._mu * grad_t
                grads = self._get_grad(x_adv_t, y)
                with torch.no_grad():
                    grad_t = grads / grads.abs().mean((1, 2, 3), keepdim=True)
                    grads = grad_t + self._mu * grad_prev
                    grad_prev = grads
                    # Perform gradient update, project to norm ball
                    x_adv += self._step_size * torch.sign(grads)
                    x_adv = self._project_linf(x_adv, x)

            x_adv_worst, worst_losses = self._get_best_adv(
                x_adv, y, x_adv_worst, worst_losses
            )

        # Return worst-case perturbed input logits
        self._core_model.train(mode)
        return x_adv_worst.detach()

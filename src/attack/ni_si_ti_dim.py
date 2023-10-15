"""Diverse-Input Attack."""

from __future__ import annotations

import random
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

from src.attack.di import RandomResizedPad
from src.attack.ti import TIAttack


class CombinedAttack(TIAttack):
    """NI-SI-TI-DIM Attack.

    Reference: https://arxiv.org/abs/1908.06281.
    """

    def __init__(
        self,
        attack_config: dict[str, str | int | float],
        core_model: nn.Module,
        loss_fn: Callable[[tuple[torch.Tensor, torch.Tensor]], torch.Tensor],
        norm: str = "Linf",
        eps: float = 8 / 255,
        dataset: str = "cifar10",
        **kwargs,
    ):
        """Initialize NI-SI-TI-DIM Attack."""
        super().__init__(
            attack_config, core_model, loss_fn, norm, eps, **kwargs
        )
        # Momentum decaying term
        self._mu: float = attack_config.get("mpgd_mu", 1.0)
        self._di_num_copies: int = attack_config.get("di_num_copies", 5)
        self._di_prop: float = attack_config.get("di_prop", 0.7)
        self._si_num_copies: int = attack_config.get("si_num_copies", 5)
        if "cifar" in dataset:
            self._transform = RandomResizedPad(28)
        elif "imagenet" in dataset:
            self._transform = RandomResizedPad(196)
        else:
            raise NotImplementedError("Only cifar and imagenet are supported.")

    def _get_grad(
        self, x_adv: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute logits, loss, gradients."""
        x_adv.requires_grad_()
        apply_tf: bool = random.random() >= self._di_prop

        with torch.enable_grad():
            # DI
            if apply_tf:
                x_tf = x_adv.repeat(self._di_num_copies, 1, 1, 1)
                targets = targets.repeat(self._di_num_copies, 1)
                # Apply random transform
                x_tf = self._transform(x_tf)
            else:
                x_tf = x_adv

            # SI
            sum_grads = 0
            for i in range(self._si_num_copies):
                logits = self._core_model(x_tf / (2**i), **self._forward_args)
                loss = self._loss_fn(logits, targets).sum()
                grads = torch.autograd.grad(loss, x_adv)[0]
                grads.detach_()
                sum_grads += grads
            sum_grads /= self._si_num_copies * self._di_num_copies

        # TI
        grads = F.conv2d(
            sum_grads,
            self._gaussian_kernel,
            bias=None,
            stride=1,
            padding=(self._pad, self._pad),
            groups=3,
        )

        return grads

    def _forward_linf(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        mode = self._core_model.training
        self._core_model.eval()

        # Initialize worst-case inputs
        x_adv_worst = inputs.clone()
        x_adv_worst.detach_()
        worst_losses = torch.zeros(len(inputs), 1, 1, 1, device=inputs.device)

        # Repeat PGD for specified number of restarts
        for _ in range(self._num_restarts):
            x_adv = inputs.clone()
            x_adv.detach_()
            momentum = torch.zeros_like(x_adv)
            prev_momentum = momentum.clone()

            # Initialize adversarial inputs
            x_adv += torch.zeros_like(x_adv).uniform_(-self._eps, self._eps)
            x_adv.clamp_(0, 1)
            x_nes = x_adv.clone()

            # Run PGD on inputs for specified number of steps
            for _ in range(self._num_steps):
                # NI: Nesterov sample is not clipped
                x_nes = x_adv + self._step_size * self._mu * prev_momentum
                grads = self._get_grad(x_nes, targets)
                with torch.no_grad():
                    # Momentum
                    grads /= grads.abs().mean((1, 2, 3), keepdim=True)
                    momentum.mul_(self._mu).add_(grads)
                    # Perform gradient update, project to norm ball
                    x_adv += self._step_size * torch.sign(momentum)
                    x_adv = self._project_linf(x_adv, inputs)
                    prev_momentum = momentum

            x_adv_worst, worst_losses = self._get_best_adv(
                x_adv, targets, x_adv_worst, worst_losses
            )

        # Return worst-case perturbed input logits
        self._core_model.train(mode)
        return x_adv_worst.detach()

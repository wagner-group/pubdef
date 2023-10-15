"""Diverse-Input Attack."""

from __future__ import annotations

import random
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

from src.attack.di import RandomResizedPad
from src.attack.ti import TIAttack
from src.utils.types import BatchHardLabels, BatchImages, BatchLabels


class AdmixCombinedAttack(TIAttack):
    """NI-Admix-TI-DIM Attack."""

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
        """Initialize NI-Admix-TI-DIM Attack."""
        super().__init__(
            attack_config, core_model, loss_fn, norm, eps, **kwargs
        )
        # Momentum decaying term
        self._mu: float = attack_config.get("mpgd_mu", 1.0)
        self._di_num_copies: int = attack_config.get("di_num_copies", 5)
        self._di_prop: float = attack_config.get("di_prop", 0.7)
        # m1 from the paper
        self._admix_num_copies: int = attack_config.get("admix_num_copies", 5)
        # m2 from the paper
        self._admix_num_others: int = attack_config.get("admix_num_others", 3)
        if "cifar" in dataset:
            self._transform = RandomResizedPad(28)
        elif "imagenet" in dataset:
            self._transform = RandomResizedPad(196)
        else:
            raise NotImplementedError("Only cifar and imagenet are supported.")

    # This method is only called internally so different signature is fine.
    # pylint: disable=arguments-differ
    def _get_grad(
        self,
        x_adv: BatchImages,
        targets: BatchLabels,
        hard_targets: BatchHardLabels,
        x_orig: BatchImages,
        target_to_other_idx: dict[int, torch.Tensor],
    ) -> BatchImages:
        """Compute logits, loss, gradients."""
        x_adv.requires_grad_()
        apply_tf: bool = random.random() >= self._di_prop
        batch_size: int = x_adv.shape[0]

        # Get indices of other classes for Admix
        other_idx = torch.zeros(
            (self._num_classes, self._admix_num_others),
            dtype=torch.long,
            requires_grad=False,
            device=x_adv.device,
        )
        for i in range(self._num_classes):
            idx = target_to_other_idx[i]
            others = torch.randperm(len(idx))[: self._admix_num_others]
            # There might be fewer other samples than self._admix_num_others
            num_others = min(len(idx), self._admix_num_others)
            other_idx[i, :num_others] = idx[others]
            # If we don't have enough other samples, fill with random ones
            other_idx[i, num_others:] = torch.randperm(batch_size)[
                : self._admix_num_others - num_others
            ]
        other_idx = other_idx[hard_targets[:batch_size]]
        other_idx = other_idx.transpose(0, 1).reshape(-1)

        x_adv.requires_grad_()
        x_orig.requires_grad_(False)
        with torch.enable_grad():
            # Admix
            x_admix = x_adv.repeat(self._admix_num_others, 1, 1, 1)
            x_admix += 0.2 * x_orig[other_idx]

            # DI
            if apply_tf:
                x_tf = x_admix.repeat(self._di_num_copies, 1, 1, 1)
                # Apply random transform
                x_tf = self._transform(x_tf)
                targets = targets.repeat(self._di_num_copies, 1)
            else:
                x_tf = x_admix

            sum_grads = 0
            # SI
            for i in range(self._admix_num_copies):
                logits = self._core_model(x_tf / (2**i), **self._forward_args)
                loss = self._loss_fn(logits, targets).sum()
                grads = torch.autograd.grad(loss, x_adv)[0]
                grads.detach_()
                sum_grads += grads

            sum_grads /= (
                self._admix_num_copies
                * self._di_num_copies
                * self._admix_num_others
            )

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
        self, inputs: BatchImages, targets: BatchLabels
    ) -> BatchImages:
        mode = self._core_model.training
        self._core_model.eval()

        # Initialize worst-case inputs
        x_adv_worst = inputs.clone()
        x_adv_worst.detach_()
        worst_losses = torch.zeros(len(inputs), 1, 1, 1, device=inputs.device)
        hard_targets = targets.argmax(1)

        # Gather indices of samples from other classes
        target_to_other_idx: dict[int, torch.Tensor] = {}
        for i in range(self._num_classes):
            idx = torch.where(hard_targets != i)[0]
            target_to_other_idx[i] = idx
        targets_dp = targets.repeat(self._admix_num_others, 1)

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
                grads = self._get_grad(
                    x_nes, targets_dp, hard_targets, inputs, target_to_other_idx
                )
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
        x_adv_worst.detach_()
        return x_adv_worst

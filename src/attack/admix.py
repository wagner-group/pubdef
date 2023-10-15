"""Admix Attack."""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from src.attack.pgd import PGDAttack
from src.utils.types import BatchImages, BatchLabels, BatchHardLabels


class AdmixAttack(PGDAttack):
    """Admix Attack.

    This Admix subsumes SI but excludes everything else (DI, TI, Momentum).
    """

    def __init__(
        self,
        attack_config: dict[str, str | int | float],
        core_model: nn.Module,
        loss_fn: Callable[[tuple[torch.Tensor, torch.Tensor]], torch.Tensor],
        norm: str = "Linf",
        eps: float = 8 / 255,
        **kwargs,
    ):
        """Initialize AdmixAttack."""
        super().__init__(
            attack_config, core_model, loss_fn, norm, eps, **kwargs
        )
        # m1 from the paper
        self._admix_num_copies: int = attack_config.get("admix_num_copies", 5)
        # m2 from the paper
        self._admix_num_others: int = attack_config.get("admix_num_others", 3)
        # Momentum decaying term
        self._mu: float = attack_config.get("mpgd_mu", 1.0)

    # This method is only called internally so different signature is fine.
    # pylint: disable=arguments-differ
    def _get_grad(
        self,
        x_adv: BatchImages,
        targets: BatchLabels,
        hard_targets: BatchHardLabels,
        x_orig: BatchImages,
        target_to_other_idx: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Compute logits, loss, gradients.

        Assume that each batch contains all classes (randomly sampled) with at
        least self._admix_num_others samples on each class.
        """
        batch_size: int = x_adv.shape[0]
        other_idx = torch.zeros(
            (self._num_classes, self._admix_num_others),
            dtype=torch.long,
            requires_grad=False,
            device=x_adv.device,
        )
        for i in range(self._num_classes):
            idx = target_to_other_idx[i]
            others = torch.randperm(len(idx))[: self._admix_num_others]
            assert others.shape[0] == self._admix_num_others
            other_idx[i] = others
        other_idx = other_idx[hard_targets[:batch_size]]
        other_idx = other_idx.transpose(0, 1).reshape(-1)

        x_adv.requires_grad_()
        x_orig.requires_grad_(False)
        with torch.enable_grad():
            x_admix = x_adv.repeat(self._admix_num_others, 1, 1, 1)
            x_admix += 0.2 * x_orig[other_idx]
            sum_grads = 0
            for i in range(self._admix_num_copies):
                # Admix already combines scaling-invariant component (2**j)
                # Different SI copies use the same admix sample.
                # https://github.com/JHL-HUST/Admix/blob/main/mi_admix_ti_di.py#L156
                logits = self._core_model(x_admix / 2**i, **self._forward_args)
                loss = self._loss_fn(logits, targets).sum()
                grads = torch.autograd.grad(loss, x_adv)[0]
                grads.detach_()
                sum_grads += grads
        sum_grads /= self._admix_num_others * self._admix_num_copies
        return sum_grads

    def _forward_linf(
        self, inputs: BatchImages, targets: BatchLabels
    ) -> BatchImages:
        mode = self._core_model.training
        self._core_model.eval()

        # Initialize worst-case inputs
        x_adv_worst = inputs.clone()
        x_adv_worst.detach_()
        worst_losses = torch.zeros(len(inputs), 1, 1, 1, device=inputs.device)

        # Gather indices of samples from other classes
        target_to_other_idx: dict[int, torch.Tensor] = {}
        hard_targets = targets.argmax(1)
        for i in range(self._num_classes):
            idx = torch.where(hard_targets != i)[0]
            target_to_other_idx[i] = idx
        targets_dp = targets.repeat(self._admix_num_others, 1)

        # Repeat PGD for specified number of restarts
        for _ in range(self._num_restarts):
            x_adv = inputs.clone().detach()

            # Initialize adversarial inputs
            x_adv += torch.zeros_like(x_adv).uniform_(-self._eps, self._eps)
            x_adv.clamp_(0, 1)

            # Run PGD on inputs for specified number of steps
            for _ in range(self._num_steps):
                grads = self._get_grad(
                    x_adv, targets_dp, hard_targets, inputs, target_to_other_idx
                )
                with torch.no_grad():
                    # Perform gradient update, project to norm ball
                    x_adv += self._step_size * torch.sign(grads)
                    x_adv = self._project_linf(x_adv, inputs)

            x_adv_worst, worst_losses = self._get_best_adv(
                x_adv, targets, x_adv_worst, worst_losses
            )

        # Return worst-case perturbed input logits
        self._core_model.train(mode)
        return x_adv_worst.detach()

"""Translation-Invariant Attack."""

from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.stats
import torch
import torch.nn.functional as F
from torch import nn

from src.attack.pgd import PGDAttack


class TIAttack(PGDAttack):
    """Translation-Invariant (TI) Attack.

    Reference: https://arxiv.org/abs/1904.02884.
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
        """Initialize TIAttack."""
        super().__init__(
            attack_config, core_model, loss_fn, norm, eps, **kwargs
        )
        self._num_copies: int = attack_config.get("ti_num_copies", 1)
        self._kernel_size: int = attack_config.get("ti_kernel_size", 5)
        self._radius: int = attack_config.get("ti_radius", 3)
        self._pad = (self._kernel_size - 1) // 2
        self._get_kernel()

    def _get_kernel(self) -> np.ndarray:
        grid = np.linspace(-self._radius, self._radius, self._kernel_size)
        kern1d = scipy.stats.norm.pdf(grid)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        kernel = kernel.astype(np.float32)
        gaussian_kernel = np.stack([kernel] * 3)
        gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
        gaussian_kernel = torch.from_numpy(gaussian_kernel).to("cuda")
        self._gaussian_kernel = gaussian_kernel

    def _get_grad(
        self, x_adv: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute logits, loss, gradients."""
        with torch.enable_grad():
            x_adv.requires_grad_()
            logits = self._core_model(x_adv, **self._forward_args)
            loss = self._loss_fn(logits, targets).mean()
            grads = torch.autograd.grad(loss, x_adv)
            grads = grads[0].detach()
            grads = F.conv2d(
                grads,
                self._gaussian_kernel,
                bias=None,
                stride=1,
                padding=(self._pad, self._pad),
                groups=3,
            )
        return grads

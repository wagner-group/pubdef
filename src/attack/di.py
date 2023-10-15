"""Diverse-Input Attack."""

from __future__ import annotations

import random
from typing import Callable

import torch
import torchvision.transforms.functional as T
from torch import nn
from torchvision.transforms import InterpolationMode

from src.attack.pgd import PGDAttack


class RandomResizedPad:
    """Randomly reszie inputs then pad to original size."""

    def __init__(self, min_size: int) -> None:
        """Initialize RandomResizedPad.

        Args:
            min_size: Minimum size to randomly resize to.
        """
        assert isinstance(min_size, int)
        self._min_size: int = min_size

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """Randomly reszie inputs then pad to original size."""
        orig_shape = inputs.shape
        orig_size = orig_shape[-1]
        rnd_size = torch.randint(
            low=self._min_size, high=orig_size, size=(1,), requires_grad=False
        )
        pad_top, pad_left = torch.randint(
            low=0,
            high=int(orig_size - rnd_size),
            size=(2,),
            requires_grad=False,
        )
        pad_bot = orig_size - rnd_size - pad_top
        pad_right = orig_size - rnd_size - pad_left
        inputs = T.resize(
            inputs,
            (rnd_size, rnd_size),
            interpolation=InterpolationMode.NEAREST,
        )
        inputs = T.pad(
            inputs,
            (pad_left, pad_top, pad_right, pad_bot),
            fill=0,
            padding_mode="constant",
        )
        assert inputs.shape == orig_shape
        return inputs


class DIAttack(PGDAttack):
    """Diverse-Input (DI) Attack.

    Reference: https://arxiv.org/abs/1803.06978.
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
        """Initialize DIAttack."""
        super().__init__(
            attack_config, core_model, loss_fn, norm, eps, **kwargs
        )
        self._di_num_copies: int = attack_config.get("di_num_copies", 5)
        self._di_prop: float = attack_config.get("di_prop", 0.7)
        # TODO(feature): other dataset
        if "cifar" in dataset:
            self._transform = RandomResizedPad(28)  # 32*7/8
        elif "imagenet" in dataset:
            self._transform = RandomResizedPad(196)  # 224*7/8
        else:
            raise NotImplementedError("Only cifar and imagenet are supported.")

    def _get_grad(
        self, x_adv: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute logits, loss, gradients."""
        x_adv.requires_grad_()
        apply_tf = random.random() >= self._di_prop

        with torch.enable_grad():
            if apply_tf:
                x_tf = x_adv.repeat(self._di_num_copies, 1, 1, 1)
                # Apply random transform
                x_tf = self._transform(x_tf)
                targets = targets.repeat(self._di_num_copies, 1)
            else:
                x_tf = x_adv

            logits = self._core_model(x_tf, **self._forward_args)
            loss = self._loss_fn(logits, targets).sum()
            grads = torch.autograd.grad(loss, x_adv)[0]
            grads.detach_()

        grads /= self._di_num_copies
        return grads

"""PGD Attack."""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import autograd, nn

from src.attack.base import AttackModule
from src.utils.loss import magnitude

_EPS = 1e-9


class PGDAttack(AttackModule):
    """Projected Gradient Descent (PGD) or Iterative FGSM Attack."""

    def __init__(
        self,
        attack_config: dict[str, str | int | float],
        core_model: nn.Module,
        loss_fn: Callable[[tuple[torch.Tensor, torch.Tensor]], torch.Tensor],
        norm: str = "Linf",
        eps: float = 8 / 255,
        forward_args: dict[str, Any] | None = None,
        **kwargs,
    ):
        """Initialize PGDAttack."""
        super().__init__(
            attack_config, core_model, loss_fn, norm, eps, **kwargs
        )
        self._num_steps: int = attack_config["pgd_steps"]
        self._step_size: float = attack_config["pgd_step_size"]
        self._num_restarts: int = attack_config["num_restarts"]
        self._forward_args: dict[str, Any] = (
            forward_args if forward_args is not None else {}
        )

    def _get_grad(
        self, x_adv: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute logits, loss, gradients."""
        with torch.enable_grad():
            x_adv.requires_grad_()
            logits = self._core_model(x_adv, **self._forward_args)
            loss = self._loss_fn(logits, targets).mean()
            grads = torch.autograd.grad(loss, x_adv, allow_unused=True)
            grads = grads[0].detach()
        return grads

    def _get_best_adv(
        self,
        x_adv: torch.Tensor,
        targets: torch.Tensor,
        x_adv_worst: torch.Tensor,
        worst_losses: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._num_restarts == 1:
            x_adv_worst = x_adv
        else:
            # Update worst-case inputs with itemized final losses
            fin_losses = self._loss_fn(
                self._core_model(x_adv), targets
            ).reshape(worst_losses.shape)
            up_mask = (fin_losses >= worst_losses).float()
            x_adv_worst = x_adv * up_mask + x_adv_worst * (1 - up_mask)
            worst_losses = fin_losses * up_mask + worst_losses * (1 - up_mask)
        return x_adv_worst, worst_losses

    def _project_l2(self, inputs: torch.Tensor, eps: float) -> torch.Tensor:
        dims = [-1] + [1] * (inputs.ndim - 1)
        norm = inputs.view(len(inputs), -1).norm(2, 1).view(dims)
        inputs *= eps / (norm + _EPS)
        return inputs

    def _forward_l2(
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

            # Initialize adversarial inputs
            x_adv += self._project_l2(torch.randn_like(x_adv), self._eps)
            x_adv.clamp_(0, 1)

            # Run PGD on inputs for specified number of steps
            for _ in range(self._num_steps):
                grads = self._get_grad(x_adv, targets)
                with torch.no_grad():
                    # Perform gradient update, project to norm ball
                    delta = (
                        x_adv
                        - inputs
                        + self._project_l2(grads, self._step_size)
                    )
                    x_adv = inputs + self._project_l2(delta, self._eps)
                    # Clip perturbed inputs to image domain
                    x_adv.clamp_(0, 1)

            x_adv_worst, worst_losses = self._get_best_adv(
                x_adv, targets, x_adv_worst, worst_losses
            )

        # Return worst-case perturbed input logits
        self._core_model.train(mode)
        return x_adv_worst.detach()

    def _project_linf(
        self, x_adv: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """Project x_adv to L-inf epsilon ball around x."""
        x_adv = torch.min(torch.max(x_adv, x - self._eps), x + self._eps)
        # Clip perturbed inputs to image domain
        x_adv.clamp_(0, 1)
        return x_adv

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

            # Initialize adversarial inputs
            x_adv += torch.zeros_like(x_adv).uniform_(-self._eps, self._eps)
            x_adv = torch.clamp(x_adv, 0, 1)

            # Run PGD on inputs for specified number of steps
            for _ in range(self._num_steps):
                grads = self._get_grad(x_adv, targets)
                with torch.no_grad():
                    # Perform gradient update, project to norm ball
                    x_adv = x_adv.detach() + self._step_size * torch.sign(grads)
                    x_adv = self._project_linf(x_adv, inputs)

            x_adv_worst, worst_losses = self._get_best_adv(
                x_adv, targets, x_adv_worst, worst_losses
            )

        # Return worst-case perturbed input logits
        self._core_model.train(mode)
        return x_adv_worst.detach()

    def forward(self, *args) -> torch.Tensor:
        """Generate adversarial examples."""
        if self._norm == "L2":
            return self._forward_l2(*args)
        return self._forward_linf(*args)


def smooth_adv(models, inputs, labels, eps):
    steps = 6
    alpha = eps / 3.0

    noise = torch.zeros_like(inputs).uniform_(-eps, eps)
    adv = inputs.detach() + noise
    adv.clamp_(0, 1)

    for _ in range(steps):
        adv.requires_grad_()
        grad_loss = 0
        for _, model in enumerate(models):
            loss = F.cross_entropy(model(adv), labels)
            grad = autograd.grad(loss, adv, create_graph=True)[0]
            grad = grad.flatten(start_dim=1)
            grad_loss += magnitude(grad)

        grad_loss /= len(models)
        grads = torch.autograd.grad(grad_loss, adv, allow_unused=True)
        grads = grads[0].detach()
        with torch.no_grad():
            adv = adv.detach() + alpha * torch.sign(grads)
            adv = torch.min(torch.max(adv, inputs - eps), inputs + eps)
            adv.clamp_(0, 1)

    adv.grad = None
    adv.detach_()
    return adv

"""Ensemble of models."""

from __future__ import annotations

import logging

import torch
from torch import nn

from src.models.ensemble.ensemble import Ensemble
from src.utils.types import BatchImages, BatchLabels, BatchLogits, StackLogits

logger = logging.getLogger(__name__)


class DVGEnsemble(Ensemble):
    """DVERGE ensemble from https://arxiv.org/abs/2104.00671."""

    _default_reduction: str = "dvg"

    def __init__(
        self,
        members: list[nn.Module],
        dverge_reg_coeff: float = 1.0,
        **kwargs,
    ) -> None:
        """Initialize DVERGE ensemble.

        Args:
            members: A list of member models.
            dverge_reg_coeff: DVERGE regularization coefficient. Defaults to 1.0.
        """
        super().__init__(members, **kwargs)
        self._dverge_reg_coeff: float = dverge_reg_coeff

    def aggregate_outputs(
        self, outputs: StackLogits, reduction: str | None = None
    ) -> BatchLogits | StackLogits:
        """Aggregate ensemble outputs."""
        reduction = reduction or self._default_reduction
        if reduction == self._default_reduction:
            reduction = "prob_mean"
        outputs = super().aggregate_outputs(outputs, reduction=reduction)
        if (
            reduction == self._default_reduction
            and not self._freeze_ensemble_but_first
        ):
            outputs = torch.clamp(outputs, min=1e-40)
            outputs = outputs.log()
        return outputs

    def _distill(
        self,
        model,
        inputs,
        epsilon,
        alpha,
        steps,
        layer,
        before_relu=True,
        mu=1,
        momentum=True,
        rand_start=False,
    ):
        x_nat = inputs.clone().detach()
        x_adv = None
        if rand_start:
            x_adv = (
                inputs.clone().detach()
                + torch.FloatTensor(inputs.shape)
                .uniform_(-epsilon, epsilon)
                .cuda()
            )
        else:
            x_adv = inputs.clone().detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        g = torch.zeros_like(x_adv)
        criterion = nn.MSELoss()

        for _ in range(steps):
            x_adv.requires_grad = True
            out = model[1].get_features(
                x=x_adv, layer=layer, before_relu=before_relu
            )
            target = torch.roll(out.detach(), shifts=1, dims=0)
            loss = criterion(out, target)
            model.zero_grad()
            loss.backward()

            grad = x_adv.grad.detach()
            with torch.no_grad():
                if momentum:  # line 86
                    flat_grad = grad.view(grad.shape[0], -1)
                    l1_grad = torch.norm(flat_grad, 1, dim=1)
                    grad /= torch.clamp(l1_grad, min=1e-12).view(
                        grad.shape[0], 1, 1, 1
                    )
                    new_grad = mu * g + grad
                    g = new_grad
                else:
                    new_grad = grad
                x_adv = x_adv - alpha * new_grad.sign()
                x_adv = torch.max(
                    torch.min(x_adv, x_nat + epsilon), x_nat - epsilon
                )
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        # print(f"DEBUG 3 START")
        # print(f"distill() is being called!")
        # print(f"DEBUG 3 END")
        return x_adv.detach()

    def compute_loss(
        self,
        outputs: BatchLogits,
        inputs: BatchImages | None = None,
        targets: BatchLabels | None = None,
        criterion: torch.nn.Module | None = None,
        distill_layer: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        """Compute Distillation loss term for DVERGE."""
        steps = 10
        distill_eps = 0.07
        alpha = 0.007
        distilled_data_list = []
        for model in self.members:
            distilled_data_list.append(
                self._distill(
                    model,
                    inputs,
                    distill_eps,
                    alpha,
                    steps,
                    distill_layer,
                    before_relu=True,
                    mu=1,
                    momentum=True,
                    rand_start=False,
                )
            )

        distill_loss = 0
        for i, member in enumerate(self.members):
            for j, distilled_data in enumerate(distilled_data_list):
                if i == j:
                    continue
                outputs = member(distilled_data)
                model_loss, _ = criterion(outputs, targets)
                distill_loss += model_loss
        distill_loss *= self._dverge_reg_coeff
        return distill_loss

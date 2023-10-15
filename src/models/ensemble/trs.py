"""Ensemble of models."""

from __future__ import annotations

import logging

import torch
from torch import autograd, nn

from src.attack.pgd import smooth_adv
from src.models.ensemble.ensemble import Ensemble
from src.utils.loss import abs_cosine, magnitude
from src.utils.types import BatchImages, BatchLabels, BatchLogits, StackLogits

logger = logging.getLogger(__name__)


class TRSEnsemble(Ensemble):
    """TRS ensemble from https://arxiv.org/abs/2104.00671."""

    _default_reduction: str = "trs"

    def __init__(
        self,
        members: list[nn.Module],
        trs_lambda_a: float = 40,
        trs_lambda_b: float = 2.5,
        **kwargs,
    ) -> None:
        """Initialize TRS ensemble.

        For detail, see https://github.com/AI-secure/Transferability-Reduced-Smooth-Ensemble/issues/1.
        Loss = lambda_a * cos_loss + lambda_b * smooth_loss
        CIFAR-10: lambda_a=40, lambda_b=2.5

        Args:
            members: A list of member models.
            trs_lambda_a: See above for detail. Defaults to 40.
            trs_lambda_b: See above for detail. Defaults to 2.5.
        """
        super().__init__(members, **kwargs)
        self._lambda_a = trs_lambda_a
        self._lambda_b = trs_lambda_b

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

    def compute_loss(
        self,
        outputs: BatchLogits,
        targets: BatchLabels | None = None,
        inputs: BatchImages | None = None,
        criterion: torch.nn.Module | None = None,
        epsilon: float = 8 / 255,
        plus_adv: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Compute TRS loss in addition to the usual classification loss."""
        _ = kwargs  # Unused
        inputs.requires_grad_(True)
        # Collect gradients from all models
        logger.debug("Computing cosine sim grad loss...")
        grads = []
        for i, member in enumerate(self.members):
            outputs: BatchLogits = member(inputs)
            curr_loss, _ = criterion(outputs, targets)
            grad = autograd.grad(
                curr_loss,
                inputs,
                create_graph=not (i > 0 and self._freeze_ensemble_but_first),
            )[0]
            if i > 0 and self._freeze_ensemble_but_first:
                grad = grad.detach()
            grad = grad.flatten(start_dim=1)
            grads.append(grad)
        inputs.requires_grad_(False)

        # Compute pairwise cosine similarity between gradients
        cos_loss, smooth_loss = 0, 0
        num_grads = len(grads)
        num_pairs = 0
        for i in range(num_grads):
            if i > 0 and self._freeze_ensemble_but_first:
                break
            for j in range(i + 1, num_grads):
                cos_loss += abs_cosine(grads[i], grads[j])
                num_pairs += 1
        cos_loss /= num_pairs

        # Compute smoothness loss
        loss_inputs = inputs
        if plus_adv:
            half_bs = inputs.shape[0] // 2
            logger.debug("Running smooth_adv...")
            adv_inputs = smooth_adv(
                self.members,
                inputs[half_bs:],
                targets[half_bs:, 0].argmax(1),  # Hard label from clean src
                epsilon,
            )
            adv_inputs.detach_()
            loss_inputs = torch.cat([inputs[:half_bs], adv_inputs])
        loss_inputs.requires_grad_(True)
        logger.debug("Computing smooth loss...")
        num_models = 0
        for i, model in enumerate(self.members):
            if i > 0 and self._freeze_ensemble_but_first:
                break
            outputs = model(loss_inputs)
            loss, _ = criterion(outputs, targets)
            grad = autograd.grad(loss, loss_inputs, create_graph=True)[0]
            grad = grad.flatten(start_dim=1)
            smooth_loss += magnitude(grad)
            num_models += 1
        smooth_loss /= num_models
        loss_inputs.requires_grad_(False)
        return self._lambda_a * cos_loss + self._lambda_b * smooth_loss

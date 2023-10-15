"""Ensemble of models."""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch import nn

from src.utils.types import BatchImages, BatchLogits, StackLogits

logger = logging.getLogger(__name__)


class Ensemble(nn.Module):
    """Ensemble Module."""
    _default_reduction: str = "mean"

    def __init__(
        self,
        members: list[nn.Module],
        weights: torch.Tensor | np.ndarray | list[float] | None = None,
        freeze_ensemble_but_first: bool = False,
        **kwargs,
    ) -> None:
        """Initialize Ensemble.

        Args:
            members: A list of member models.
            weights: Ensemble weights for averaging outputs. Must be 1D tensor,
                array, or list of the same length as members. Defaults to None
                (equal weights).
            freeze_ensemble_but_first: Whether to freeze all ensemble members
                during training and only train the first member.

        Raises:
            IndexError: members and weights are of different lenght.
            ValueError: weights contain negative value(s).
        """
        logger.debug("Started initializing Ensemble!")
        _ = kwargs  # Unused
        super().__init__()
        self._freeze_ensemble_but_first: bool = freeze_ensemble_but_first
        self.members = nn.ModuleList(members)
        self._weights = None
        if weights is not None:
            if len(weights) != len(members):
                raise IndexError(
                    "weights must be None or have the same length as members "
                    f"({len(weights)} vs {len(members)})!"
                )
            if not isinstance(weights, torch.Tensor):
                weights = torch.tensor(weights, dtype=torch.float32)
            if any(weights < 0):
                raise ValueError("weights cannot be negative!")
            weights /= weights.sum()
            # Register ensemble weights as non-trainable weights
            self.register_buffer("_weights", weights[None, :, None])
        logger.debug("Completed Ensemble Initialization!")

    def _weighted_avg(self, outputs: torch.Tensor) -> torch.Tensor:
        if self._weights is not None:
            outputs = torch.einsum("ijk,ijk->ik", outputs, self._weights)
        else:
            outputs = outputs.mean(1)
        return outputs

    def compute_loss(self, outputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """Default doesn't do anything to loss."""
        _ = kwargs  # Unused
        return torch.tensor(0, device=outputs.device, dtype=outputs.dtype)

    def aggregate_outputs(
        self, outputs: StackLogits, reduction: str | None = None
    ) -> BatchLogits | StackLogits:
        """Aggregate ensemble outputs."""
        reduction = reduction or self._default_reduction
        if reduction not in (
            "none",
            "prob",
            "mean",
            "prob_mean",
            "mean_prob",
            "mean_logits",
            "logits_mean",
        ):
            raise NotImplementedError(
                f"Reduction mode {reduction} is not implemented!"
            )

        if reduction == "prob_mean":
            outputs = torch.softmax(outputs, dim=-1)
            outputs = self._weighted_avg(outputs)
        else:
            if "mean" in reduction:
                # mean, mean_prob, mean_logits, logits_mean
                outputs = self._weighted_avg(outputs)
            if "prob" in reduction:
                # prob, mean_prob
                outputs = torch.softmax(outputs, dim=-1)
        return outputs

    def forward(
        self,
        inputs: BatchImages,
        reduction: str | None = None,
        compute_loss: bool = False,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of all members and then combine outputs.

        Args:
            inputs: Input tensors in batch.
            return_logits: If True, return logits instead of softmax. Defaults
                to True.
            reduction: Reduction mode. Options: "none" or combinations of "mean"
                and "prob" or "logits". See code for details. Defaults to "mean"
            compute_loss: If True, compute ensemble-specific loss. Defaults to
                False.

        Returns:
            Ensemble outputs: Shape [batch_size, num_members, num_classes] or
                [batch_size, num_classes] depending on `reduction` when model is
                in training mode. In eval mode, it is expected that ensemble
                returns output of shape [batch_size, num_classes].
            Loss: Extra loss terms specific to different types of ensemble.
                Only returned when training mode is on.
        """
        if self._freeze_ensemble_but_first and not compute_loss:
            outputs = self.members[0](inputs)
            return outputs

        outputs = []
        for model in self.members:
            outputs.append(model(inputs))
        outputs = torch.stack(outputs, dim=1)
        outputs = self.aggregate_outputs(outputs, reduction=reduction)

        if compute_loss:
            # We choose to compute special ensemble loss inside forward() to
            # utilize DataParallel or DistributedDataParallel mode.
            ensemble_loss = self.compute_loss(outputs, inputs=inputs, **kwargs)
            if self._freeze_ensemble_but_first:
                outputs = outputs[:, 0:1]
            return outputs, ensemble_loss
        return outputs

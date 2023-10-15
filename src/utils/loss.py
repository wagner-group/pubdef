"""Define all loss functions."""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from src.dataloader.cutmix import soft_cross_entropy
from src.utils.types import BatchLabels, BatchLogits, StackLabels, StackLogits

logger = logging.getLogger(__name__)


def abs_cosine(g1, g2):
    """Absolute cosine similarity."""
    return torch.abs(F.cosine_similarity(g1, g2)).mean()


def magnitude(g1):
    """TRS codebase method of calculating magnitude."""
    return (torch.sum(g1**2, 1)).mean() * 2


def _assert_shape(logits, targets, is_dual_loss=False):
    if logits.ndim == 2:
        logits = logits.unsqueeze(1)
    if targets.ndim == 2:
        targets = targets.unsqueeze(1)
    assert logits.ndim == 3, (
        "Invalid logits shape! Expected stacked logits with shape [B, S, C]"
        f" but got {(logits.shape)}."
    )
    assert targets.ndim == 3, (
        "Invalid targets shape! Expected stacked targets with shape "
        f"[B, S, C] but got {(targets.shape)}."
    )
    if not is_dual_loss:
        assert logits.shape == targets.shape, (
            f"Expected same shape for logits ({logits.shape}) and labels "
            f"({targets.shape})!"
        )
    else:
        assert logits.shape[0] == targets.shape[0], (
            f"Batch dimension does not match {logits.shape} (logits) vs "
            f"{targets.shape} (targets)!"
        )
        assert logits.shape[-1] == targets.shape[-1], (
            f"Label dimension does not match {logits.shape} (logits) vs "
            f"{targets.shape} (targets)!"
        )
        assert logits.shape[1] == targets.shape[1] * 2, (
            "When dual loss is used, dataset dimension of logits must be twice "
            f"of that of targets, but got {logits.shape} (logits) vs "
            f"{targets.shape} (targets)!"
        )
    return logits, targets


class Loss(nn.Module):
    """Base class for loss functions."""

    is_dual_loss: bool = False

    def __init__(
        self,
        beta: float = 0.5,
        num_datasets: int = 1,
        dataset_dropout: list[float] | None = None,
        dataset_weights: list[float] | None = None,
        worst_k_transfer_loss: int | None = None,
        use_lse: bool = False,
        **kwargs,
    ) -> None:
        """Initialize Mixed Adversarial Training loss.

        Args:
            beta: Beta parameter weighting adversarial and clean losses.
                Defaults to 0.5.
            num_datasets: Number of sub-datasets. Defaults to 1.
            dataset_dropout: Dropout rate for each sub-dataset.
                Defaults to None (no dropout).
            dataset_weights: Weights for each sub-dataset. Defaults to None
            worst_k_transfer_loss: Number of worst-k transfer losses to use.
                Defaults to None (use all losses).
            use_lse: If True, use least square error instead of cross entropy.
                Defaults to False.
        """
        _ = kwargs  # Unused
        super().__init__()
        self._beta: float = beta
        self._use_lse: bool = use_lse

        if dataset_dropout is not None and len(dataset_dropout) != num_datasets:
            raise ValueError(
                f"Invalid dataset_dropout! Expected {num_datasets} dropout "
                f"rates but got {len(dataset_dropout)}!"
            )
        self._dataset_dropout: list[float] | None = dataset_dropout

        if (
            worst_k_transfer_loss is not None
            and worst_k_transfer_loss >= num_datasets
        ):
            raise ValueError(
                f"Invalid worst_k_transfer_loss! Expected a value < "
                f"{num_datasets} but got {worst_k_transfer_loss}!"
            )
        self._worst_k_transfer_loss: int | None = worst_k_transfer_loss

        if dataset_weights is None:
            dataset_weights = [1.0] * num_datasets
        dataset_weights = torch.tensor(dataset_weights) / (sum(dataset_weights))
        if len(dataset_weights) != num_datasets:
            raise ValueError(
                f"Invalid dataset_weights! Expected {num_datasets} weights but "
                f"got {len(dataset_weights)}!"
            )
        # Original dataset weights
        self._dataset_weights: torch.Tensor  # To silence linter
        self.register_buffer("_dataset_weights", dataset_weights)
        # Current dataset weights (maybe dynamically updated)
        self.weights: torch.Tensor
        self.register_buffer("weights", dataset_weights)

        logger.info("Loss function: %s", self.__class__.__name__)
        logger.info("    beta: %s", self._beta)
        logger.info("    dataset_dropout: %s", self._dataset_dropout)
        logger.info("    dataset_weights: %s", dataset_weights.tolist())
        logger.info(
            "    worst_k_transfer_loss: %s", self._worst_k_transfer_loss
        )

    @abstractmethod
    def forward(
        self,
        logits: StackLogits | BatchLogits,
        targets: StackLabels | BatchLabels,
    ) -> tuple[torch.Tensor, torch.Tenor]:
        """Compute the loss.

        Args:
            logits: Stacked logits from all sub-datasets, shape: [B, S, C].
            targets: Stacked targets, shape: [B, N].

        Returns:
            torch.Tensor: Final loss value. Shape: [1,]
            torch.Tensor: Losses for each sub-dataset before applying dataset
                weights. Shape: [num_datasetes,]
        """
        raise NotImplementedError("Loss.forward() is not implemented.")


class DualLoss(Loss):
    """Dual loss computed on both clean and adversarial examples."""

    is_dual_loss: bool = True

    def forward(
        self,
        logits: StackLogits | BatchLogits,
        targets: StackLabels | BatchLabels,
    ) -> tuple[torch.Tensor, torch.Tenor]:
        """Compute the loss. See `Loss.forward` for details."""
        logits, targets = _assert_shape(
            logits, targets, is_dual_loss=self.is_dual_loss
        )
        mean_losses = []
        batch_size, num_datasets = logits.shape[:2]
        assert num_datasets % 2 == 0
        num_datasets = int(num_datasets / 2)

        for i in range(num_datasets):
            cl_logits = logits[:, i]
            adv_logits = logits[:, num_datasets + i]
            cl_loss = self._cl_loss_fn(
                cl_logits, targets[:, i], reduction="none"
            )
            adv_loss = self._adv_loss_fn(
                cl_logits, adv_logits, targets[:, i], reduction="none"
            )
            loss = cl_loss + self._beta * adv_loss
            if self._dataset_dropout is not None:
                drop_idx = torch.randperm(batch_size)[
                    : int(batch_size * self._dataset_dropout[i])
                ]
                loss[drop_idx] *= 0
            loss = loss.sum() / batch_size
            mean_losses.append(loss)

        mean_losses = torch.stack(mean_losses)
        loss_weights = self.weights.to(mean_losses.device)
        total_loss = (mean_losses * loss_weights).sum()
        return total_loss, mean_losses


class TradesLoss(DualLoss):
    """TRADES loss."""

    def __init__(self, **kwargs) -> None:
        """Initialize TradesLoss. See Loss.__init__ for details."""
        super().__init__(**kwargs)

        def _loss_fn(cl_logits, adv_logits, targets, reduction="none"):
            _ = targets  # Unused
            assert reduction == "none"
            adv_lprobs = F.log_softmax(adv_logits, dim=1)
            cl_probs = F.softmax(cl_logits, dim=1)
            return F.kl_div(adv_lprobs, cl_probs, reduction="none").sum(1)

        self._cl_loss_fn = soft_cross_entropy
        self._adv_loss_fn = _loss_fn


class TradesLseLoss(DualLoss):
    """TRADES with LSE loss."""

    def __init__(self, **kwargs) -> None:
        """Initialize TradesLseLoss. See Loss.__init__ for details."""
        super().__init__(**kwargs)

        def _cl_loss_fn(cl_logits, targets, reduction="none"):
            assert reduction == "none"
            softmax = F.softmax(cl_logits, dim=1)
            return torch.sum((softmax - targets) ** 2, dim=-1)

        def _adv_loss_fn(cl_logits, adv_logits, targets, reduction="none"):
            _ = cl_logits  # Unused
            assert reduction == "none"
            softmax = F.softmax(adv_logits, dim=1)
            return torch.sum((softmax - targets) ** 2, dim=-1)

        self._cl_loss_fn = _cl_loss_fn
        self._adv_loss_fn = _adv_loss_fn


class MatLoss(DualLoss):
    """MAT loss."""

    def __init__(self, **kwargs) -> None:
        """Initialize MatLoss. See Loss.__init__ for details."""
        super().__init__(**kwargs)

        def _loss_fn(cl_logits, adv_logits, targets, reduction="none"):
            _ = cl_logits  # Unused
            assert reduction == "none"
            return soft_cross_entropy(adv_logits, targets, reduction=reduction)

        self._cl_loss_fn = soft_cross_entropy
        self._adv_loss_fn = _loss_fn


class SingleLoss(Loss):
    """Single loss computed only on adversarial examples."""

    is_dual_loss: bool = False

    def _get_losses(
        self, logits: StackLogits, targets: StackLabels
    ) -> tuple[torch.Tensor, torch.Tenso, list[torch.Tenor]]:
        """Compute losses for each sub-dataset."""
        losses, mean_losses, mean_accs = [], [], []
        num_samples_to_drop = 0
        batch_size, num_datasets = logits.shape[:2]
        for i in range(num_datasets):
            cur_logits, cur_targets = logits[:, i], targets[:, i]
            acc = (cur_logits.argmax(1) == cur_targets.argmax(1)).float()
            loss = self._loss_fn(cur_logits, cur_targets, reduction="none")
            if self._dataset_dropout is not None:
                # Get a random indices of the batch
                rand_idx = torch.randperm(batch_size)
                num_samples_to_drop = int(batch_size * self._dataset_dropout[i])
                rand_idx = rand_idx[num_samples_to_drop:]
                loss = loss[rand_idx]
                acc = acc[rand_idx]
                losses.append(loss)
            mean_losses.append(loss.mean())
            mean_accs.append(acc.mean())
        mean_losses = torch.stack(mean_losses)
        mean_accs = torch.stack(mean_accs)
        return mean_losses, mean_accs, losses

    def forward(
        self,
        logits: StackLogits | BatchLogits,
        targets: StackLabels | BatchLabels,
    ) -> tuple[torch.Tensor, torch.Tenor]:
        """Compute the loss. See `Loss.forward` for details."""
        logits, targets = _assert_shape(
            logits, targets, is_dual_loss=self.is_dual_loss
        )
        k: int = self._worst_k_transfer_loss
        mean_losses, _, losses = self._get_losses(logits, targets)
        loss_weights = self.weights.to(mean_losses.device)

        if k is None:
            total_loss = (mean_losses * loss_weights).sum()
            return total_loss, mean_losses

        losses_without_clean = torch.stack(losses[1:], dim=1)
        # Obtaining top k values from each individual sample in batch
        topk_values, topk_indices = torch.topk(losses_without_clean, k, dim=1)
        # Setting the indices to correspond with which source model with
        # picked starting from index 1 since index 0 refers to clean samples
        topk_indices = topk_indices + 1  # [B, k]

        # Concencating clean loss with top-k losses: [B, k + 1]
        topk_losses = torch.cat((losses[0].unsqueeze(1), topk_values), dim=1)
        # Get corresponding dataset weights for clean + top-k losses
        topk_indices = torch.cat(
            (torch.zeros_like(topk_indices[:, 0:1]), topk_indices), dim=1
        )
        topk_weights = loss_weights[topk_indices]
        # Renormalize weights to sum to 1
        topk_weights = topk_weights / topk_weights.sum(1, keepdim=True)
        total_loss = (topk_weights * topk_losses).sum(1).mean()
        return total_loss, mean_losses


class CELoss(SingleLoss):
    """Cross-entropy loss."""

    def __init__(self, **kwargs) -> None:
        """Initialize CELoss."""
        super().__init__(**kwargs)
        self._loss_fn = soft_cross_entropy


class LseLoss(SingleLoss):
    """SCORE LSE loss."""

    def __init__(self, **kwargs) -> None:
        """Initialize LseLoss."""
        super().__init__(**kwargs)

        def _lse_loss_fn(logits, targets, reduction="none"):
            assert reduction == "none"
            probs = F.softmax(logits, dim=1)
            return torch.sum((probs - targets) ** 2, dim=-1)

        self._loss_fn = _lse_loss_fn


class WeightedExpLoss(CELoss):
    """Custom Weighted Exponential Loss (soft top k version)."""

    is_dual_loss: bool = False

    def __init__(
        self,
        alpha_exp_smooth: float = 0.99,
        include_clean_samples: bool = False,
        track_metric: Literal["loss", "acc"] = "loss",
        **kwargs,
    ) -> None:
        """Initialize WeightedExpLoss.

        Args:
            alpha_exp_smooth: Smoothing parameter for exponential moving
                average. Defaults to 0.99.
            include_clean_samples: If True, also weigh clean samples with the
                exponential moiving average. Defaults to False.
            track_metric: Metric to track for exponential moving average.
                Options are "loss" and "acc". Defaults to "loss".
        """
        super().__init__(**kwargs)
        self._alpha: float = alpha_exp_smooth
        self._include_clean_samples: bool = include_clean_samples
        self._track_metric: Literal["loss", "acc"] = track_metric
        # Initialize moving_average to be uniform
        moving_average = torch.ones(len(self._dataset_weights))
        moving_average /= moving_average.sum()
        # Register moving_average as a buffer
        self._moving_average: torch.Tensor  # To silence linter
        self.register_buffer("_moving_average", moving_average)
        logger.info("    alpha_exp_smooth: %s", self._alpha)
        logger.info("    include_clean_samples: %s", include_clean_samples)

    def forward(
        self,
        logits: StackLogits | BatchLogits,
        targets: StackLabels | BatchLabels,
    ) -> tuple[torch.Tensor, torch.Tenor]:
        """Compute the loss. See `Loss.forward` for details."""
        logits, targets = _assert_shape(
            logits, targets, is_dual_loss=self.is_dual_loss
        )
        mean_losses, mean_accs, _ = self._get_losses(logits, targets)

        # Update exponential moving average
        self._moving_average *= 1 - self._alpha
        if self._track_metric == "loss":
            self._moving_average += self._alpha * mean_losses.detach()
        else:
            # Has to do 1 - accuracy because higher accuracy means we should
            # allocate a smaller weight for it.
            self._moving_average += self._alpha * (1 - mean_accs.detach())

        # Normalize moving average to sum to 1
        if self._include_clean_samples:
            self.weights = self._moving_average * self._dataset_weights
            self.weights /= self.weights.sum()
        else:
            # Keep weight for clean data unchanged to self._dataset_weights[0]
            non_clean_weights = (
                self._moving_average[1:] * self._dataset_weights[1:]
            )
            non_clean_weights /= non_clean_weights.sum()
            non_clean_weights *= 1 - self._dataset_weights[0]
            self.weights[1:] = non_clean_weights

        total_loss = (mean_losses * self.weights).sum()
        return total_loss, mean_losses


class ZeroLoss(SingleLoss):
    """Loss that always returns 0. Used with DVERGE only."""

    def forward(
        self,
        logits: StackLogits | BatchLogits,
        targets: StackLabels | BatchLabels,
    ) -> tuple[torch.Tensor, torch.Tenor]:
        """Compute the loss. See `Loss.forward` for details."""
        logits, targets = _assert_shape(
            logits, targets, is_dual_loss=self.is_dual_loss
        )
        num_datasets = logits.shape[1]
        total_loss = torch.zeros((1,), device=logits.device)[0]
        mean_losses = torch.zeros((num_datasets,), device=logits.device)
        return total_loss, mean_losses


def get_criteria(
    cuda_device: int = 0,
    train_loss: str = "ce",
    adv_beta: float | None = 0.5,
    **kwargs,
) -> tuple[Loss, Loss]:
    """Get train/test and adversarial training criteria."""
    criterion = CELoss(**kwargs)
    if train_loss not in LOSS_DICT:
        raise NotImplementedError(
            f"Invalid train_loss mode: {train_loss} (valid losses include "
            f"{list(LOSS_DICT.keys())})!"
        )
    train_criterion = LOSS_DICT[train_loss](beta=adv_beta, **kwargs)
    return criterion.cuda(cuda_device), train_criterion.cuda(cuda_device)


LOSS_DICT = {
    "ce": CELoss,
    "trades": TradesLoss,
    "mat": MatLoss,
    "weighted-exp": WeightedExpLoss,
    "zero": ZeroLoss,
    "lse": LseLoss,
    "trades-lse": TradesLseLoss,
}

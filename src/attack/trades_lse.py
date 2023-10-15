"""TRADES with LSE loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.attack.trades import TRADESAttack


class _TradesLseLoss(torch.nn.Module):
    """TRADES with LSE loss."""

    def __init__(self, reduction: str = "none") -> None:
        """Initialize TradesLseLoss."""
        super().__init__()
        self._reduction = reduction

    def forward(self, cl_logits, adv_logits):
        """Compute LSE loss."""
        cl_probs = F.softmax(cl_logits, dim=1)
        adv_probs = F.softmax(adv_logits, dim=1)
        loss = torch.sum((cl_probs - adv_probs) ** 2, dim=-1)
        if self._reduction == "none":
            return loss
        if self._reduction == "mean":
            return torch.mean(loss)
        if self._reduction == "sum":
            return torch.sum(loss)
        raise ValueError(f"Invalid reduction: {self._reduction}")


class TradesLseAttack(TRADESAttack):
    """TRADES attack with LSE loss."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize LseAttack."""
        super().__init__(*args, **kwargs)
        self._trades_loss_fn = _TradesLseLoss(reduction="none")

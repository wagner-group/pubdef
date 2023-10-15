"""LSE Attack."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.attack.pgd import PGDAttack


class _LseLoss(torch.nn.Module):
    """LSE loss."""

    def __init__(self, reduction: str = "none") -> None:
        """Initialize LseLoss."""
        super().__init__()
        self._reduction = reduction

    def forward(self, logits, targets):
        """Compute LSE loss."""
        probs = F.softmax(logits, dim=1)
        loss = torch.sum((probs - targets) ** 2, dim=-1)
        if self._reduction == "none":
            return loss
        if self._reduction == "mean":
            return torch.mean(loss)
        if self._reduction == "sum":
            return torch.sum(loss)
        raise ValueError(f"Invalid reduction: {self._reduction}")


class LseAttack(PGDAttack):
    """LSE Attack."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize LseAttack."""
        super().__init__(*args, **kwargs)
        self._loss_fn = _LseLoss(reduction="none")

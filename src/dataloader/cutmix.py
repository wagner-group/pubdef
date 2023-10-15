"""Custom CutMix Dataset to handle index.

Code adapted from https://github.com/ildoonet/cutmix.
"""

from __future__ import annotations

import random
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data.dataset import Dataset

from src.utils.types import BatchImages, BatchLabels, BatchLogits


class CutMix(Dataset):
    """CutMix wrapper Dataset."""

    def __init__(
        self,
        dataset: Dataset,
        num_class: int,
        num_mix: int = 1,
        beta: float = 1.0,
        prob: float = 1.0,
    ) -> None:
        self.dataset = dataset
        self.num_class = num_class
        self.num_mix = num_mix
        self.beta = beta
        self.prob = prob

    def __getitem__(self, index: int) -> tuple[BatchImages, BatchLabels, int]:
        """Get item from dataset."""
        img, label, idx = self.dataset[index]
        lb_onehot = _onehot(self.num_class, label)

        for _ in range(self.num_mix):
            r = np.random.rand(1)
            if self.beta <= 0 or r > self.prob:
                continue

            # generate mixed sample
            lam = np.random.beta(self.beta, self.beta)
            rand_index = random.choice(range(len(self)))

            img2, lb2, _ = self.dataset[rand_index]
            lb2_onehot = _onehot(self.num_class, lb2)

            bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
            img[:, bbx1:bbx2, bby1:bby2] = img2[:, bbx1:bbx2, bby1:bby2]
            lam = 1 - (
                (bbx2 - bbx1)
                * (bby2 - bby1)
                / (img.size()[-1] * img.size()[-2])
            )
            lb_onehot = lb_onehot * lam + lb2_onehot * (1.0 - lam)

        return img, lb_onehot, idx

    def __len__(self):
        return len(self.dataset)


class SoftCrossEntropyLoss(nn.Module):
    def __init__(self, reduction: Literal["none", "mean"] = "none") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, logits: BatchLogits, targets: BatchLabels):
        return soft_cross_entropy(logits, targets, reduction=self.reduction)


def soft_cross_entropy(
    logits: BatchLogits,
    targets: BatchLabels,
    reduction: Literal["none", "mean"] = "none",
) -> torch.Tensor:
    """Cross entropy that accepts soft targets.

    Args:
        logits: predictions for neural network.
        targets: targets, can be soft.
        reduction: Loss reduction method. "none" or "mean".

    Examples::

        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)

        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(input, target)
        loss.backward()
    """
    if len(targets.size()) == 1:
        targets = torch.nn.functional.one_hot(
            targets, num_classes=logits.size(-1)
        )
        targets = targets.float()

    if targets.squeeze(1).shape != logits.shape:
        # (torch.Size([32, 10])) must match logits shape (torch.Size([32, 10]))!
        raise ValueError(
            f"Target shape ({targets.shape}) must match logits shape "
            f"({logits.shape})!"
        )
    loss = torch.sum(-targets * F.log_softmax(logits, dim=1), dim=1)
    if reduction == "mean":
        return torch.mean(loss)
    return loss


def _onehot(size, target):
    if not isinstance(target, int):
        return target
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.0
    return vec


def rand_bbox(size, lam):
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception

    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

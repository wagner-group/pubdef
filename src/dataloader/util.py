"""Utility for dataloader."""

from __future__ import annotations

from src.dataloader.cifar_multi_datasets import (
    CIFAR10MultiDataset,
    CIFAR100MultiDataset,
)
from src.dataloader.imagenet_multi_datasets import ImageNetMultiDataset


def load_dataset(dataset: str = "cifar10", **kwargs):
    """Load dataset.

    Args:
        dataset: Dataset name to load.

    Returns:
        Dataset object.
    """
    dataset_fn = {
        "cifar10": CIFAR10MultiDataset,
        "cifar100": CIFAR100MultiDataset,
        "imagenet": ImageNetMultiDataset,
    }
    return dataset_fn[dataset](dataset=dataset, **kwargs)

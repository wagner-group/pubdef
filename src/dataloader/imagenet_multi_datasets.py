"""Dataset for tensor version of CIFAR-10/100."""

from __future__ import annotations

import logging
from typing import Callable

import torch
import torchvision.transforms.v2 as transforms
from timm.data import create_transform
from torch.utils.data import Dataset

from src.dataloader.cifar_multi_datasets import (
    CIFARMultiDataset,
    CustomImageDataset,
)
from src.dataloader.cutmix import CutMix
from src.dataloader.imagenet import ImageNetDataset

logger = logging.getLogger(__name__)


class ImageNetMultiDataset(CIFARMultiDataset):
    """ImageNet dataset."""

    # Normalize param is used as part of model, not preprocessing
    normalize_params: dict[str, tuple[float, float, float]] = {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
    num_classes: int = 1000
    input_dim: tuple[int, int, int] = (3, 224, 224)
    available_augments: tuple[str] = (
        "default",
        "randaug",
        "cutmix",
        "noaug",
        "cj",
        "erase",
        "gen1",
    )

    # pylint: disable=super-init-not-called
    def __init__(
        self,
        dataset: str = "imagenet",
        root_dir: str = "~/data/",
        train_load_paths: list[str] | None = None,
        val_load_paths: list[str] | None = None,
        test_load_paths: list[str] | None = None,
        num_workers: int = 4,
        batch_size: int = 128,
        distributed: bool = True,
        val_ratio: float = 0.1,
        shuffle: bool = True,
        seed: int = 0,
        augments: list[str] | None = None,
        num_copies: int = 100000,
        **kwargs,
    ) -> None:
        """Load ImageNet dataset.

        Args:
            dataset: Dataset to load. Defaults to "cifar10".
            root_dir: Path to root directory.
            train_load_paths: List of paths to train samples.
            val_load_paths: List of paths to val samples.
            test_load_paths: List of paths to test samples.
            num_workers: Number of workers for data loader. Defaults to 4.
            batch_size: Batch size. Defaults to 128.
            distributed: Whether to use distributed sampler. Defaults to True.
            val_ratio: Ratio of validation set to training set. Defaults to 0.1.
            shuffle: Whether to shuffle the dataset. Defaults to True.
            seed: Random seed. Defaults to 0.
            augments: List of data augmentation to apply to train set.
                Defaults to ["default"].
            num_copies: Number of copies per sample. Defaults to 100000.
            kwargs: Additional arguments to pass to dataset.
        """
        assert dataset == "imagenet", "Only ImageNet is supported."
        logger.info("Loading multi-dataset ImageNet dataset...")
        self._dataset: str = dataset
        self._root_dir: str = root_dir
        self._val_ratio: float = val_ratio
        self._shuffle: bool = shuffle
        self._seed: int = seed
        self._distributed: bool = distributed
        self._batch_size: int = batch_size
        self._num_workers: int = num_workers
        # Augmentations for original dataset (will be parsed by CIFARDataset)
        self._orig_augments: list[str] = augments
        self._augments: list[str] = self._parase_augments(augments or ["noaug"])
        self._num_datasets: int | None = None
        self._kwargs = kwargs

        # Load tensors from paths
        trainset, valset, testset = [], [], []
        if train_load_paths:
            logger.info("Loading train samples from %s...", train_load_paths)
            trainset = self._gather_loads(train_load_paths)
        if val_load_paths:
            logger.info("Loading val samples from %s...", val_load_paths)
            valset = self._gather_loads(val_load_paths)
        if test_load_paths:
            logger.info("Loading test samples from %s...", test_load_paths)
            testset = self._gather_loads(test_load_paths)

        train_transform, transform = self._get_transforms()
        self.trainset: list[Dataset] = [
            CustomImageDataset(
                *samples, transform=train_transform, num_copies=num_copies
            )
            for samples in trainset
        ]
        if "cutmix" in augments:
            # Apply CutMix to each training dataset separately which means there
            # is no mixing across datasets.
            cutmix_params = {
                "num_class": self.num_classes,
                "beta": 1.0,
                "prob": 0.5,
                "num_mix": 2,
            }
            self.trainset = [CutMix(t, **cutmix_params) for t in self.trainset]

        self.valset: list[Dataset] = [
            CustomImageDataset(
                *samples, transform=transform, num_copies=num_copies
            )
            for samples in valset
        ]
        self.testset: list[Dataset] = [
            CustomImageDataset(
                *samples, transform=transform, num_copies=num_copies
            )
            for samples in testset
        ]

    def _get_transforms(
        self,
    ) -> tuple[Callable[..., torch.Tensor], Callable[..., torch.Tensor]]:
        """Get transforms for transfer adversrial examples.

        # TODO(enhance): We may want to make this different from the default
        # transforms on original images.
        """
        train_augs = create_transform(
            self.input_dim,
            is_training=True,
            use_prefetcher=True,
            no_aug=False,
            scale=None,
            ratio=None,
            hflip=0.5,
            vflip=0.0,
            color_jitter=0.1 if "cj" in self._augments else 0.0,
            interpolation="bilinear",
            mean=(0, 0, 0),
            std=(1, 1, 1),
        ).transforms[:-1]

        if "randaug" in self._augments:
            logger.info("Using randaug data augmentation...")
            train_augs.append(
                transforms.RandAugment(
                    interpolation=transforms.InterpolationMode.BILINEAR
                )
            )
        if "erase" in self._augments:
            train_augs.append(transforms.RandomErasing(p=0.25))

        test_transforms = transforms.Compose(
            [
                transforms.ToImageTensor(),
                transforms.ConvertImageDtype(),
            ]
        )
        train_transforms = transforms.Compose(
            [
                *train_augs,
                transforms.ToImageTensor(),
                transforms.ConvertImageDtype(),
            ]
        )
        return train_transforms, test_transforms

    def _load_default(self) -> tuple[Dataset, Dataset, Dataset]:
        # Load default CIFAR dataset
        default_imagenet_dataset = ImageNetDataset(
            dataset=self._dataset,
            root_dir=self._root_dir,
            num_workers=self._num_workers,
            batch_size=self._batch_size,
            distributed=self._distributed,
            val_ratio=self._val_ratio,
            shuffle=self._shuffle,
            seed=self._seed,
            augments=self._orig_augments,
            **self._kwargs,
        )
        return (
            default_imagenet_dataset.trainset,
            default_imagenet_dataset.valset,
            default_imagenet_dataset.testset,
        )

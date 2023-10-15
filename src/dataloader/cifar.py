"""Dataset for CIFAR-10/100."""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import torch
import torch.utils.data
import torchvision.datasets
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms

from src.dataloader.cutmix import CutMix
from src.utils.eval_sampler import DistributedEvalSampler

logger = logging.getLogger(__name__)


class CIFAR10DatasetwithIndex(torchvision.datasets.CIFAR10):
    """CIFAR10 dataset that also returns sample indices."""

    def __getitem__(self, index: int) -> tuple[Any, Any, Any]:
        """Get item from dataset with index."""
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class CIFAR100DatasetwithIndex(torchvision.datasets.CIFAR100):
    """CIFAR10 dataset that also returns sample indices."""

    def __getitem__(self, index: int) -> tuple[Any, Any, Any]:
        """Get item from dataset with index."""
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class CIFARDataset:
    """Wrapper around PyTorch CIFAR-10/100 datasets."""

    # Normalize param is used as part of model, not preprocessing
    normalize_params: dict[str, tuple[float, float, float]] = {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
    input_dim: tuple[int, int, int] = (3, 32, 32)
    available_augments: tuple[str] = (
        "default",
        "pad_crop",
        "randaug",
        "cutmix",
        "noaug",
    )

    def __init__(
        self,
        dataset: str = "cifar10",
        root_dir: str = "~/data/",
        num_workers: int = 4,
        batch_size: int = 128,
        distributed: bool = True,
        val_ratio: float = 0.1,
        shuffle: bool = True,
        seed: int = 0,
        augments: list[str] | None = None,
        **kwargs,
    ) -> None:
        """Load CIFAR-10/CIFAR-100 dataset.

        Args:
            dataset: Dataset to load. Defaults to "cifar10".
            root_dir: Path to root directory.
            num_workers: Number of workers for data loader. Defaults to 4.
            batch_size: Batch size. Defaults to 128.
            distributed: Whether to use distributed sampler. Defaults to True.
            val_ratio: Ratio of validation set to training set. Defaults to 0.1.
            shuffle: Whether to shuffle the dataset. Defaults to True.
            seed: Random seed. Defaults to 0.
            augments: List of data augments to apply to train set.
                Defaults to ["default"].
            **kwargs: Additional arguments to pass to dataset.
        """
        logger.info("Loading default CIFAR-10/100 dataset...")
        _ = kwargs  # Unused
        self._root_dir: str = root_dir
        self._shuffle: bool = shuffle
        self._seed: int = seed
        self._distributed: bool = distributed
        self._batch_size: int = batch_size
        self._num_workers: int = num_workers
        # Parse data augments
        self._augments: list[str] = self._parase_augments(
            augments or ["pad_crop"]
        )

        dataset_func: Callable[..., Any]
        if dataset == "cifar10":
            dataset_func = CIFAR10DatasetwithIndex
        elif dataset == "cifar100":
            dataset_func = CIFAR100DatasetwithIndex
        else:
            raise NotImplementedError(
                f"Invalid dataset for CIFARDataset ({dataset}); Only "
                '"cifar10", "cifar100" are available!'
            )

        transform_train, transform = self._get_transforms()
        self.trainset = dataset_func(
            root=self._root_dir,
            train=True,
            download=True,
            transform=transform_train,
        )
        self.valset = dataset_func(
            root=self._root_dir, train=True, download=True, transform=transform
        )
        self.testset = dataset_func(
            root=self._root_dir, train=False, download=True, transform=transform
        )

        # Random split train and validation sets and shuffle them if needed
        num_train_total: int = len(self.trainset)
        self.num_valid: int = int(np.floor(val_ratio * num_train_total))
        self.num_train: int = num_train_total - self.num_valid
        self.num_test: int = len(self.testset)

        # Random split train/val sets
        indices: list[int] = list(range(num_train_total))
        # Remove random split from CIFAR to make sure we have the same split
        # every time
        # if self._shuffle:
        #     with np_temp_seed(self._seed):
        #         np.random.shuffle(indices)
        train_idx: list[int] = indices[self.num_valid :]
        val_idx: list[int] = indices[: self.num_valid]
        self.trainset: torch.utils.data.Dataset = torch.utils.data.Subset(
            self.trainset, train_idx
        )
        self.valset: torch.utils.data.Dataset = torch.utils.data.Subset(
            self.valset, val_idx
        )
        if "cutmix" in self._augments:
            logger.info("Using cutmix data augmentation...")
            cutmix_params = {
                "num_class": int(dataset.replace("cifar", "")),
                "beta": 1.0,
                "prob": 0.5,
                "num_mix": 2,
            }
            self.trainset = CutMix(self.trainset, **cutmix_params)

    def _parase_augments(self, augments: list[str]) -> list[str]:
        """Parse list of data augmentations."""
        if any(aug not in self.available_augments for aug in augments):
            raise ValueError(
                f"Invalid augment ({augments})! Available augments are "
                f"{self.available_augments}."
            )
        # Replace default with pad_crop
        return list(map(lambda x: x.replace("default", "pad_crop"), augments))

    def _get_transforms(
        self,
    ) -> tuple[Callable[..., torch.Tensor], Callable[..., torch.Tensor]]:
        train_augs = []
        if "pad_crop" in self._augments:
            logger.info("Using pad_crop data augmentation...")
            train_augs.extend(
                [
                    transforms.RandomCrop(self.input_dim[-1], padding=4),
                    transforms.RandomHorizontalFlip(0.5),
                ]
            )
        if "randaug" in self._augments:
            logger.info("Using randaug data augmentation...")
            train_augs.append(
                transforms.RandAugment(
                    interpolation=transforms.InterpolationMode.BILINEAR
                )
            )

        test_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        train_transforms = transforms.Compose(
            [
                *train_augs,
                transforms.ToTensor(),
            ]
        )
        return train_transforms, test_transforms

    def _get_loader_sampler(
        self, dataset: Dataset, split: str
    ) -> tuple[DataLoader, Sampler]:
        sampler: Sampler | None = None
        shuffle: bool | None = self._shuffle if split == "train" else False

        if self._distributed:
            shuffle = None  # When sampler is used, shuffle must be None
            if split == "train":
                sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset,
                    shuffle=self._shuffle,
                    seed=self._seed,
                    drop_last=False,
                )
            else:
                # Use distributed sampler for validation but not testing
                # due to AutoAttack.
                sampler = DistributedEvalSampler(
                    dataset, shuffle=False, seed=self._seed
                )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=shuffle,
            num_workers=self._num_workers,
            pin_memory=True,
            sampler=sampler,
            persistent_workers=True,
            prefetch_factor=4,
        )
        return loader, sampler

    def get_data_loader(self, mode: str = "load", **kwargs):
        """Get train/val/test dataloaders."""
        _ = kwargs, mode  # Unused
        train_loader, train_sampler = self._get_loader_sampler(
            self.trainset, "train"
        )
        val_loader, _ = self._get_loader_sampler(self.valset, "val")
        test_loader, _ = self._get_loader_sampler(self.testset, "test")
        return train_loader, train_sampler, val_loader, test_loader


class CIFAR10Dataset(CIFARDataset):
    """CIFAR-10 dataset."""

    num_classes: int = 10


class CIFAR100Dataset(CIFARDataset):
    """CIFAR-100 dataset."""

    num_classes: int = 100

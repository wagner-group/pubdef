"""Dataset for ImageNet."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.utils.data
import torchvision
import torchvision.transforms.v2 as transforms
from timm.data import create_transform

from src.dataloader.cifar import CIFARDataset
from src.dataloader.cutmix import CutMix
from src.utils.random_seed import np_temp_seed

logger = logging.getLogger(__name__)


class ImageNetDatasetwithIndex(torchvision.datasets.ImageFolder):
    """ImageNet dataset that also returns sample indices."""

    def __getitem__(self, index: int):
        """Get item from dataset with index."""
        sample = super().__getitem__(index)
        return *sample, index


class ImageNetDataset(CIFARDataset):
    """Wrapper around PyTorch ImageNet dataset."""

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
        num_workers: int = 4,
        batch_size: int = 128,
        distributed: bool = True,
        val_ratio: float = 0.1,
        shuffle: bool = True,
        seed: int = 0,
        augments: list[str] | None = None,
        selected_file_paths: str | None = None,
        **kwargs,
    ) -> None:
        """Load ImageNet dataset.

        Args:
            dataset: Dataset to load. Defaults to "imagenet".
            root_dir: Path to root directory.
            num_workers: Number of workers for data loader. Defaults to 4.
            batch_size: Batch size. Defaults to 128.
            distributed: Whether to use distributed sampler. Defaults to True.
            val_ratio: Ratio of validation set to training set. Defaults to 0.1.
            shuffle: Whether to shuffle the dataset. Defaults to True.
            seed: Random seed. Defaults to 0.
            augments: List of data augments to apply to train set.
                Defaults to ["default"].
            selected_file_paths: Path to txt file contains file paths to load
                (used with validation of ImageNet only). Defaults to None (load
                all files).
            **kwargs: Additional arguments to pass to dataset.
        """
        assert dataset == "imagenet", "Only support ImageNet dataset."
        logger.info("Loading default ImageNet dataset...")
        _ = kwargs  # Unused
        self._root_dir: str = root_dir + "/imagenet/"
        self._shuffle: bool = shuffle
        self._seed: int = seed
        self._distributed: bool = distributed
        self._batch_size: int = batch_size
        self._num_workers: int = num_workers
        # Parse data augments
        self._augments: list[str] = self._parase_augments(
            augments or ["default"]
        )

        # Filter validation set by selected_file_paths
        if selected_file_paths is not None:
            selected_file_paths = Path(selected_file_paths).expanduser()
            if not selected_file_paths.is_file():
                raise FileNotFoundError(
                    f"selected_file_paths {selected_file_paths} is not found!"
                )
            selected_file_paths = selected_file_paths.read_text(
                encoding="utf-8"
            ).splitlines()
            selected_file_paths = set(selected_file_paths)
            num_selected_files = len(selected_file_paths)

        def is_selected(file_path):
            file_path = "/".join(file_path.split("/")[-2:])
            return file_path in selected_file_paths

        transform_train, transform = self._get_transforms()
        self.trainset = ImageNetDatasetwithIndex(
            root=self._root_dir + "train", transform=transform_train
        )
        self.valset = ImageNetDatasetwithIndex(
            root=self._root_dir + "train", transform=transform
        )
        self.testset = ImageNetDatasetwithIndex(
            root=self._root_dir + "val",
            transform=transform,
            is_valid_file=None if selected_file_paths is None else is_selected,
        )
        if selected_file_paths is not None:
            assert len(self.testset) == num_selected_files, (
                f"Number of selected files {len(self.testset)} does not match "
                f"the given selected_file_paths ({num_selected_files})!"
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
        if self._shuffle:
            with np_temp_seed(self._seed):
                np.random.shuffle(indices)
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
                "num_class": self.num_classes,
                "beta": 1.0,
                "prob": 0.5,
                "num_mix": 2,
            }
            self.trainset = CutMix(self.trainset, **cutmix_params)

    def _get_transforms(
        self,
    ) -> tuple[Callable[..., torch.Tensor], Callable[..., torch.Tensor]]:
        if "gen1" in self._augments:
            train_aug_prams = {
                "scale": (0.81, 1.0),  # Prevent very small crop
                "ratio": (3. / 4., 4. / 3.),
                "hflip": 0.5,
                "vflip": 0.0,
                "color_jitter": 0.1 if "cj" in self._augments else 0.0,
            }
        else:
            train_aug_prams = {
                "scale": None,  # Default
                "ratio": None,  # Default
                "hflip": 0.5,
                "vflip": 0.0,
                "color_jitter": 0.1 if "cj" in self._augments else 0.0,
            }
        train_augs = create_transform(
            self.input_dim,
            is_training=True,
            use_prefetcher=True,
            no_aug=False,
            interpolation="bilinear",
            mean=(0, 0, 0),
            std=(1, 1, 1),
            **train_aug_prams,
        ).transforms[:-1]

        if "randaug" in self._augments:
            logger.info("Using randaug data augmentation...")
            train_augs.append(
                transforms.RandAugment(
                    interpolation=transforms.InterpolationMode.BILINEAR
                )
            )
        # if "mixup" in self._augments:
        #     train_augs.append(
        #         transforms.MixUp(
        #             alpha=1.0,
        #             num_classes=self.num_classes,
        #             labels_getter="default",
        #         )
        #     )

        test_transforms = transforms.Compose(
            [
                transforms.Resize(int(self.input_dim[-1] * 256 / 224)),
                transforms.CenterCrop(self.input_dim[-1]),
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

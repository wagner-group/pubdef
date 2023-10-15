"""Dataset for tensor version of CIFAR-10/100."""

from __future__ import annotations

import logging
import pathlib
import random

import numpy as np
import torch
import torch.utils.data
import torchvision
from PIL import Image
from safetensors import safe_open
from torch.utils.data import Dataset

from src.dataloader.cifar import CIFARDataset
from src.dataloader.cutmix import CutMix
from src.utils.types import BatchLabels

logger = logging.getLogger(__name__)


class StackDataset(Dataset):
    """Stack multiple datasets together."""

    def __init__(
        self,
        datasets: list[Dataset],
        num_classes: int,
        mix_idx: list[int] | None = None,
    ) -> None:
        """Initialize StackDataset.

        Args:
            datasets: List of datasets to stack.
            num_classes: Number of classes in dataset.
            mix_idx: List of dataset indices to mix together. The rest of the
                datasets will be stacked as usual. To mix all datasets as one
                stream (stack dim of 1), use mix_idx = range(len(datasets)).
                Defaults to None (stack all datasets).
        """
        self.datasets: list[Dataset] = datasets
        self.num_classes: int = num_classes
        self.mix_idx: set[int] = mix_idx or {}

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        """Get item at index."""
        # TODO(speed): We can avoid reading from disk here by selecting the
        # random indices to load first.
        samples = [d[index] for d in self.datasets]

        # Each sample is (image, target, index)
        mix_samples, stack_samples = [], []
        for i, sample in enumerate(samples):
            if i in self.mix_idx:
                mix_samples.append(sample)
            else:
                stack_samples.append(sample)

        # Choose only one sample among mixed samples
        if mix_samples:
            sample = random.choice(mix_samples)
            stack_samples.append(sample)

        stacked_images = torch.stack([s[0] for s in stack_samples], dim=0)
        stacked_targets = []
        for _, sample in enumerate(stack_samples):
            if isinstance(sample[1], int):
                # Convert to one-hot
                target = torch.zeros(self.num_classes, dtype=torch.long)
                target[sample[1]] = 1
            else:
                target = sample[1]
            stacked_targets.append(target)
        stacked_targets = torch.stack(stacked_targets, dim=0)
        return stacked_images, stacked_targets, index

    def __len__(self):
        """Get length of dataset."""
        return min(len(d) for d in self.datasets)


class CustomImageDataset(Dataset):
    """TensorDataset with support of transforms."""

    def __init__(
        self,
        load_paths: list[str],
        targets: list[int],
        indices: list[int],
        soft_targets: BatchLabels | None = None,
        transform: torchvision.transforms.Transform | None = None,
        num_copies: int | None = 100000,
    ) -> None:
        """Initialize CustomImageDataset."""
        if not load_paths:
            raise ValueError("load_paths must not be empty!")
        if not all(len(load_paths) == len(lst) for lst in (targets, indices)):
            raise ValueError(
                "Different number of datasets given (load_paths: "
                f"{len(load_paths)}, targets: {len(targets)}, indices: "
                f"{len(indices)})!"
            )
        if isinstance(num_copies, int) and num_copies < 1:
            raise ValueError(
                f"num_copies must be at least 1, but is {num_copies}!"
            )
        self._num_samples_per_copy: int = len(np.unique(indices))
        # self._samples: [num_samples, num_dataset_copies]
        self._samples = [[] for _ in range(self._num_samples_per_copy)]
        self._targets = [0] * self._num_samples_per_copy
        self._soft_targets: BatchLabels | None = soft_targets
        for load_path, target, index in zip(load_paths, targets, indices):
            self._samples[index].append(load_path)
            self._targets[index] = target
        if num_copies is not None:
            for i in range(self._num_samples_per_copy):
                self._samples[i] = self._samples[i][:num_copies]
        logger.info("Loaded %d samples.", sum(len(s) for s in self._samples))
        self._transform = transform

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        """Get item at index."""
        image_paths = self._samples[index]
        # Sample one random image path (pick a random copy)
        image_path = random.choice(image_paths)
        image = Image.open(image_path)
        if self._transform:
            image = self._transform(image)
        # Load soft labels if exist
        if self._soft_targets is not None:
            label = self._soft_targets[index]
        else:
            label = self._targets[index]
        return image, label, index

    def __len__(self) -> int:
        """Get length of dataset."""
        return self._num_samples_per_copy


class CIFARMultiDataset(CIFARDataset):
    """Wrapper around PyTorch CIFAR-10/100 datasets."""

    # pylint: disable=super-init-not-called
    def __init__(
        self,
        dataset: str = "cifar10",
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
        """Load CIFAR-10/CIFAR-100 dataset. For args, see `CIFARDataset`.

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
        logger.info("Loading tensor version of CIFAR-10/100 dataset...")
        _ = kwargs  # Unused
        if dataset not in ("cifar10", "cifar100"):
            raise NotImplementedError(
                'dataset must be either "cifar10" or "cifar100" to use '
                f"CIFARTensorDataset, but it is {dataset}!"
            )
        self.num_classes: int = 10 if dataset == "cifar10" else 100
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
        # 1 Dataset = 1 src of transfer attack
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
                "num_class": int(dataset.replace("cifar", "")),
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

        # TODO(feature): If randomly split train/val sets, have to make sure
        # that they are consistent with the original datasets
        # Random split train and validation sets and shuffle them if needed
        # num_train_total: int = len(self.trainset)
        # self.num_train: int = num_train_total
        # self.num_valid: int = len(self.valset)
        # self.num_test: int = len(self.testset)
        # if not val_given:
        #     self.num_valid: int = int(np.floor(val_ratio * num_train_total))
        #     self.num_train: int = num_train_total - self.num_valid
        #     # Random split train/val sets
        #     indices: list[int] = list(range(self.num_train))
        #     if self._shuffle:
        #         with np_temp_seed(self._seed):
        #             np.random.shuffle(indices)
        #     train_idx: list[int] = indices[self.num_valid :]
        #     val_idx: list[int] = indices[: self.num_valid]
        #     self.trainset = Subset(self.trainset, train_idx)
        #     self.valset = Subset(self.valset, val_idx)

    def _parase_augments(self, augments: list[str]) -> list[str]:
        """Parse list of data augmentations."""
        if any(aug not in self.available_augments for aug in augments):
            raise ValueError(
                f"Invalid augment ({augments})! Available augments are "
                f"{self.available_augments}."
            )
        # Replace default with pad_crop
        return map(lambda x: x.replace("default", "noaug"), augments)

    def _gather_loads(
        self, paths: list[str]
    ) -> list[tuple[list[str], list[int], list[int], BatchLabels | None]]:
        if not isinstance(paths, (list, tuple)):
            raise ValueError(
                f"paths must be a list of strings, but it is {paths}!"
            )
        if not paths:
            raise ValueError(f"`paths` cannot be empty ({paths})!")

        samples = []
        # Gather data from each of the given paths
        for path in paths:
            path = pathlib.Path(path).expanduser()
            if not path.is_dir():
                raise NotADirectoryError(
                    f"Path {str(path)} is not a directory or does not exist!"
                )

            load_paths, targets, indices = [], [], []
            for load_path in path.glob("*/*.png"):
                # load_path: rep_index.pt
                load_path = str(load_path)
                target, name = load_path.split("/")[-2:]
                target = int(target)
                index = int(name.split("_", maxsplit=1)[0])
                load_paths.append(load_path)
                targets.append(target)
                indices.append(index)

            if not load_paths:
                raise ValueError(
                    f"There are no valid data at {path}! Please check if data "
                    f"exist at {path}/<CLASS_ID>/<IMG>.png, or remove this path"
                    " from load_paths."
                )

            # Get soft labels from safetensors if exist and cutmix was used
            soft_target_path = path / "soft_targets.safetensors"
            if soft_target_path.is_file() and "cutmix" in path.name:
                with safe_open(
                    str(soft_target_path), framework="pt", device="cpu"
                ) as file:
                    soft_targets = file.get_tensor("targets")
                if len(soft_targets) != len(load_paths):
                    raise ValueError(
                        f"Size of soft targets ({len(soft_targets)}) does not "
                        f"match the number of samples ({len(load_paths)})! The "
                        "soft target file may have been overwritten."
                    )
                samples.append((load_paths, targets, indices, soft_targets))
            else:
                samples.append((load_paths, targets, indices, None))

        for i, per_dataset_samples in enumerate(samples):
            logger.info(
                "%d samples found under path %s",
                len(per_dataset_samples[0]),
                paths[i],
            )
        return samples

    def _load_default(self) -> tuple[Dataset, Dataset, Dataset]:
        # Load default CIFAR dataset
        default_cifar_dataset = CIFARDataset(
            dataset=self._dataset,
            root_dir=self._root_dir,
            num_workers=self._num_workers,
            batch_size=self._batch_size,
            distributed=self._distributed,
            val_ratio=self._val_ratio,
            shuffle=self._shuffle,
            seed=self._seed,
            augments=self._orig_augments,
        )
        return (
            default_cifar_dataset.trainset,
            default_cifar_dataset.valset,
            default_cifar_dataset.testset,
        )

    def _get_datasets(
        self,
        datasets: list[Dataset],
        mode: str = "stack",
        orig_dataset: Dataset | None = None,
    ) -> Dataset:
        if mode in ("stack", "mix"):
            datasets = [orig_dataset, *datasets]
        elif mode == "orig":
            datasets = [orig_dataset]
        elif mode == "load" or orig_dataset is None:
            pass
        else:
            raise NotImplementedError(f"Mode {mode} is not implemented!")
        if not datasets:
            logger.warning(
                "No dataset given for mode %s! Returning None...", mode
            )
            return None

        # Concatenate tensor version with default CIFAR dataset
        stacked_dataset = StackDataset(
            datasets=datasets,
            num_classes=self.num_classes,
            mix_idx=list(range(1, len(datasets))) if mode == "mix" else None,
        )

        # Set num_datasets
        self._num_datasets = 2 if mode == "mix" else len(datasets)
        return stacked_dataset

    def get_num_datasets(self) -> int:
        """Returns the number of final datasets loaded in each batch."""
        if self._num_datasets is None:
            raise ValueError(
                "num_datasets is not initialized! Please call get_data_loader()"
                "at least once with the desired mode."
            )
        return self._num_datasets

    def get_data_loader(self, mode: str = "stack", **kwargs):
        """Get train/val/test dataloaders.

        Args:
            mode: Mode to get data loaders.
                "orig": Only includes original dataset.
                "load': Only includes loaded dataset, used for evaluating
                on transfer adversarial examples.
                "stack": Stack all datasets (original and loaded ones).
                "mix": Mix all loaded datasets and stack with the original.
                Defaults to "stack".
        """
        _ = kwargs  # Unused
        logger.info("Getting data loaders with mode: %s", mode)

        # Load default CIFAR dataset
        orig_trainset, orig_valset, orig_testset = self._load_default()

        # Stack multiple datasets
        trainset = self._get_datasets(
            self.trainset, mode=mode, orig_dataset=orig_trainset
        )
        valset = self._get_datasets(
            self.valset, mode=mode, orig_dataset=orig_valset
        )
        testset = self._get_datasets(
            self.testset, mode=mode, orig_dataset=orig_testset
        )

        # Get data loaders from datasets
        train_loader, val_loader, test_loader = None, None, None
        train_sampler = None
        if trainset is not None:
            train_loader, train_sampler = self._get_loader_sampler(
                trainset, "train"
            )
        if valset is not None:
            val_loader, _ = self._get_loader_sampler(valset, "val")
        if testset is not None:
            test_loader, _ = self._get_loader_sampler(testset, "test")
        return train_loader, train_sampler, val_loader, test_loader


class CIFAR10MultiDataset(CIFARMultiDataset):
    """CIFAR-10 multi-dataset."""

    num_classes: int = 10


class CIFAR100MultiDataset(CIFARMultiDataset):
    """CIFAR-100 multi-dataset."""

    num_classes: int = 100

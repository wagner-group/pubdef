"""Main script for both training and evaluation."""

from __future__ import annotations

import timm

if not timm.__version__.startswith("0.6"):
    from timm.models import xcit

    xcit.XCiT = xcit.Xcit  # pylint: disable=no-member

# pylint: disable=wrong-import-order,wrong-import-position
import argparse
import logging
import pprint
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
import wandb
from frozendict import frozendict
from safetensors.torch import save_file
from torch import nn
from torch.backends import cudnn

from src.attack.base import AttackModule
from src.attack.util import get_saved_adv_name, setup_default_attacks
from src.dataloader.util import load_dataset
from src.models.util import build_classifier
from src.utils.argparser import get_args_parser, setup_output_dir
from src.utils.distributed import (
    get_rank,
    init_distributed_mode,
    is_main_process,
)
from src.utils.loss import get_criteria
from src.utils.metric import AverageMeter, ProgressMeter, accuracy
from src.utils.types import BatchLogits, StackLogits


def save_adv_main(
    config: dict[str, Any]  # pylint: disable=redefined-outer-name
) -> None:
    """Main function."""
    config["evaluate"] = True
    init_distributed_mode(config)
    cudnn.benchmark = True
    pprint.pprint(config)
    output_dir: Path = setup_output_dir(config)
    # Make sure that config is immutable
    config = frozendict(config)
    print("-" * shutil.get_terminal_size().columns)
    gpu: int = config["gpu"]
    save_adv_partition: str = config["save_adv_partition"]

    if config["distributed"]:
        raise NotImplementedError(
            "Saving adversarial examples is not compatible with "
            "distributed mode!"
        )

    # Fix the seed for reproducibility
    seed: int = config["seed"] + get_rank()
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Data loading code
    print("=> Creating dataset...")
    dataset = load_dataset(
        dataset=config["dataset"],
        root_dir=config["data_dir"],
        distributed=config["distributed"],
        batch_size=config["batch_size"],
        num_workers=config["workers"],
        val_ratio=config["val_ratio"],
        shuffle=config["shuffle"],
        seed=config["seed"],
        augments=config["augments"],
        selected_file_paths=config["selected_file_paths"],
    )
    num_classes: int = dataset.num_classes
    # Default dataloader
    data_order: int = {
        "train": 0,  # Skip train sampler at 1.
        "val": 2,
        "test": 3,
    }[save_adv_partition]
    dataloader = dataset.get_data_loader()[data_order]

    # Create model
    print("=> Creating model...")
    config = dict(config)
    model, _, _ = build_classifier(config)
    config = frozendict(config)
    model.eval()

    # Define loss function
    criterion, _ = get_criteria(gpu, config["train_loss"], config["adv_beta"])

    # Logging
    if is_main_process():
        logfile = open(str(output_dir / "log.txt"), "a", encoding="utf-8")
        logfile.write(str(config) + "\n")
        logfile.flush()
        if config["wandb"]:
            wandb_id = output_dir.name
            wandb.init(
                project="transfer", id=wandb_id, config=config, resume="allow"
            )
            print("wandb step:", wandb.run.step)

    eval_attack: list[tuple[str, AttackModule]] = setup_default_attacks(
        config, model, num_classes, num_restarts=1
    )[0]
    attack_name, attack = eval_attack[1]  # Skip no_attack

    if config["save_adv_path"] is not None:
        save_adv_path = config["save_adv_path"]
    else:
        save_adv_path = output_dir / get_saved_adv_name(
            config, attack=attack_name, partition=save_adv_partition
        )
    save_adv_path = Path(save_adv_path)
    if not save_adv_path.exists():
        save_adv_path.mkdir(parents=True, exist_ok=True)
    if not save_adv_path.is_dir():
        raise NotADirectoryError("save_adv_path must be a directory!")
    print(f"Generated adversarial examples will be saved at {save_adv_path}...")

    for i in range(num_classes):
        (save_adv_path / f"{i:05d}").mkdir(parents=True, exist_ok=True)

    print(f"=> Running {attack_name} attack...")
    _gen_adv(
        config,
        dataloader,
        model,
        criterion,
        attack,
        save_adv_path=save_adv_path,
        num_datasets=config["num_dataset_copies"],
    )


def _gen_adv(
    config: dict[str, Any],  # pylint: disable=redefined-outer-name
    dataloader: torch.utils.data.DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    attack: AttackModule,
    save_adv_path: Path | str | None = None,
    num_datasets: int | float | None = None,
):
    gpu: int = config["gpu"]
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    mem = AverageMeter("Mem (GB)", ":6.1f")
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, losses, top1, mem],
        prefix="Test: ",
    )
    num_saved_adv: int = 0  # Num adv saved so far
    num_data: int = len(dataloader.dataset)
    if isinstance(num_datasets, int) and num_datasets >= 1:
        num_data *= num_datasets
    else:
        raise ValueError(
            "num_dataset_copies must be an integer >= 1, but it is "
            f"{num_datasets}!"
        )

    num_repeats: int = int(np.ceil(num_data / len(dataloader.dataset)))
    print(f"   {num_data} adversarial examples will be generated...")
    saved_advs: dict[str, list[torch.Tensor]] = {
        "images": [],
        "targets": [],
        "hard_targets": [],
        "filenames": [],
    }

    # Switch to evaluate mode
    model.eval()

    end = time.time()
    for rep in range(num_repeats):
        for i, (images, targets, indices) in enumerate(dataloader):
            hard_targets = targets.argmax(-1).squeeze(1)
            assert hard_targets.ndim == 1
            cur_rep = rep + config["rep_offset"]
            filepaths = [
                save_adv_path / f"{tgt:05d}/{index:08d}_{cur_rep:02d}.png"
                for index, tgt in zip(indices, hard_targets.tolist())
            ]
            if all(f.exists() for f in filepaths):
                continue

            # Measure data loading time
            data_time.update(time.time() - end)
            images = images.cuda(gpu, non_blocking=True)
            targets = targets.cuda(gpu, non_blocking=True)
            batch_size, _num_datasets, num_classes = targets.shape
            assert _num_datasets == 1, (
                "Invalid targets shape! The 2nd dim must be 1 but get shape "
                f"{targets.shape}!"
            )

            # Compute attack and output
            with torch.no_grad():
                images = images.view(-1, *images.shape[2:])
                images = attack(images, targets.view(-1, num_classes))
                outputs: BatchLogits = model(images)
                # Restack outputs
                outputs: StackLogits = outputs.view(batch_size, 1, num_classes)
                loss = criterion(outputs, targets)[0]

            # Measure accuracy and record loss
            acc1, _ = accuracy(outputs[:, 0], targets[:, 0].argmax(1))
            losses.update(loss.item(), batch_size)
            top1.update(acc1.item(), batch_size)

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            mem.update(torch.cuda.max_memory_allocated() // 1e9)
            if i % config["print_freq"] == 0:
                progress.display(i)

            # Save adv images and targets
            # saved_advs["images"].extend(list(images.cpu()))
            targets.squeeze_(1)
            # saved_advs["hard_targets"].extend(hard_targets.tolist())
            saved_advs["targets"].append(targets.cpu())
            # filenames = [
            #     f"{index:08d}_{rep + config['rep_offset']:02d}.png"
            #     for index in indices
            # ]
            # saved_advs["filenames"].extend(filenames)

            for j in range(batch_size):
                torchvision.utils.save_image(images[j], str(filepaths[j]))

            num_saved_adv += batch_size
            if num_saved_adv >= num_data:
                break

    progress.synchronize()
    print(f" * Acc@1 {top1.avg:.3f}")

    # Create directories for each target class
    # for target in np.unique(saved_advs["hard_targets"]):
    #     (save_adv_path / f"{target:05d}").mkdir(parents=True, exist_ok=True)

    # Save each sample one by one
    # for i in range(num_data):
    #     image = saved_advs["images"][i]
    #     hard_target = saved_advs["hard_targets"][i]
    #     filename = saved_advs["filenames"][i]
    #     path = save_adv_path / f"{hard_target:05d}" / filename
    #     torchvision.utils.save_image(image, path)
    # Save target tensor
    targets = torch.cat(saved_advs["targets"])
    save_file({"targets": targets}, save_adv_path / "soft_targets.safetensors")
    return {"acc1": top1.avg, "loss": losses.avg}


if __name__ == "__main__":
    main_parser = argparse.ArgumentParser(
        "Generate and save adversarial examples",
        parents=[get_args_parser()],
    )
    main_parser.add_argument(
        "--rep-offset",
        default=0,
        type=int,
        help=(
            "Each saved file name contains image id and num_copies. This "
            "argument offsets num_copies. Useful for generating multiple copies"
            "in parallel (don't forget to change seed!)."
        ),
    )
    args = main_parser.parse_args()
    config: dict[str, Any] = vars(args)

    # Set logging config
    logging.basicConfig(
        stream=sys.stdout,
        format="[%(asctime)s - %(name)s - %(levelname)s]: %(message)s",
        level=logging.DEBUG if args.debug else logging.INFO,
        force=True,
    )
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    save_adv_main(config)

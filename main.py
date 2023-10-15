"""Main script for both training and evaluation."""

from __future__ import annotations

import timm

if not timm.__version__.startswith("0.6"):
    from timm.models import xcit

    xcit.XCiT = xcit.Xcit  # pylint: disable=no-member

# pylint: disable=wrong-import-order,wrong-import-position
import argparse
import json
import logging
import math
import os
import pprint
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb
from frozendict import frozendict
from torch.backends import cudnn
from torch.cuda import amp
from wandb.sdk.lib.mailbox import MailboxError
from wandb.sdk.service.service import ServiceStartTimeoutError

from src.attack.base import AttackModule
from src.attack.util import setup_default_attacks
from src.dataloader.util import load_dataset
from src.models.ensemble.dverge import DVGEnsemble
from src.models.ensemble.ensemble import Ensemble
from src.models.util import build_classifier
from src.utils.argparser import (
    get_args_parser,
    setup_load_paths,
    setup_output_dir,
)
from src.utils.distributed import (
    dist_barrier,
    get_rank,
    init_distributed_mode,
    is_main_process,
    save_on_master,
)
from src.utils.loss import get_criteria
from src.utils.metric import (
    AverageMeter,
    ProgressMeter,
    accuracy,
    adjust_learning_rate,
    safe_pickle,
)
from src.utils.types import (
    BatchImages,
    BatchLogits,
    StackImages,
    StackLabels,
    StackLogits,
)

BEST_ACC: float = -1.0
logger = logging.getLogger(__name__)


def train_test_main(
    config: dict[str, Any]  # pylint: disable=redefined-outer-name
) -> dict[str, Any]:
    """Main function."""
    init_distributed_mode(config)
    cudnn.benchmark = True
    config = setup_load_paths(config)
    output_dir = setup_output_dir(config)
    pprint.pprint(config)
    # Make sure that config is not mutable
    config = frozendict(config)
    print("-" * shutil.get_terminal_size().columns)
    gpu: int = config["gpu"]

    global BEST_ACC  # pylint: disable=global-statement

    # Fix the seed for reproducibility
    seed: int = config["seed"] + get_rank()
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Data loading code
    logger.info("=> Creating dataset...")
    dataset = load_dataset(
        dataset=config["dataset"],
        root_dir=config["data_dir"],
        distributed=config["distributed"],
        batch_size=config["batch_size"],
        num_workers=config["workers"],
        val_ratio=config["val_ratio"],
        shuffle=True,
        seed=config["seed"],
        train_load_paths=config["train_load_paths"],
        test_load_paths=config["test_load_paths"],
        val_load_paths=config["val_load_paths"],
        augments=config["augments"],
        num_copies=config["num_dataset_copies"],
        selected_file_paths=config["selected_file_paths"],
    )
    num_classes: int = dataset.num_classes
    data = dataset.get_data_loader(mode=config["data_load_mode"])
    train_loader, train_sampler, val_loader, test_loader = data

    # Get number of stacked datasets
    num_datasets = dataset.get_num_datasets()

    # Create model
    logger.info("=> Creating model...")
    # Resume requires updating "start_epoch" so we need to unfreeze dict
    config = dict(config)
    model, optimizer, scaler = build_classifier(config)
    config = frozendict(config)
    cudnn.benchmark = True

    # Define loss function
    criterion, train_criterion = get_criteria(
        cuda_device=gpu,
        train_loss=config["train_loss"],
        adv_beta=config["adv_beta"],
        num_datasets=num_datasets,
        dataset_dropout=config["dataset_dropout"],
        dataset_weights=config["dataset_weights"],
        worst_k_transfer_loss=config["worst_k_transfer_loss"],
        alpha_exp_smooth=config["alpha_exp_smooth"],
        include_clean_samples=config["include_clean_samples"],
        track_metric=config["exp_loss_track_metric"],
    )

    # Logging
    if is_main_process():
        logfile = open(str(output_dir / "log.txt"), "a", encoding="utf-8")
        logfile.write(str(config) + "\n")
        logfile.flush()
        if config["wandb"]:
            try:
                # wandb id cannot be longer than 128 characters
                wid = f"{output_dir.name[:120]}-{hash(output_dir.name) % 10000}"
                wandb.init(
                    entity="transfer-def",
                    project=config["dataset"],
                    id=wid,
                    config=dict(config),
                    resume="allow",
                )
                logger.info("wandb step: %d", wandb.run.step)
            except (ServiceStartTimeoutError, MailboxError):
                logger.warning("wandb init timed out. Disabling wandb.")
                config = dict(config)
                config["wandb"] = False
                config = frozendict(config)

    # Set up default attack modules
    eval_attacks_list, train_attack, val_attack = setup_default_attacks(
        config, model, num_classes
    )

    no_attack: AttackModule = eval_attacks_list[0][1]
    save_metrics = {"train": [], "test": []}

    if config["evaluate"]:
        if config["resume"]:
            load_path = config["resume"]
        else:
            load_path = str(output_dir / "checkpoint_best.pt")
    else:
        logger.info("=> Beginning training")
        val_stats = {}
        for epoch in range(config["start_epoch"], config["epochs"]):
            is_best: bool = False
            if config["distributed"]:
                # This is neede to make shuffling work correctly
                train_sampler.set_epoch(epoch)
            lr: float = adjust_learning_rate(config, optimizer, epoch)
            logger.info("=> lr @ epoch %d: %.2e", epoch, lr)

            # Train for one epoch
            train_stats = _train(
                config,
                train_loader,
                model,
                train_criterion,
                train_attack,
                optimizer,
                scaler,
                epoch,
                num_datasets=num_datasets,
                eval_criterion=criterion,
            )

            # Only validate once every two epochs
            if (epoch + 1) % 2 == 0:
                val_stats = _validate(
                    config,
                    val_loader,
                    model,
                    criterion,
                    no_attack,
                    num_datasets=num_datasets,
                )
                clean_acc: float = val_stats["weighted_acc"]
                main_acc: float = clean_acc
                is_best: bool = clean_acc > BEST_ACC
                if config["train_attack"] != "none":
                    # Validate with attacks
                    adv_val_stats = _validate(
                        config,
                        val_loader,
                        model,
                        criterion,
                        val_attack,
                        num_datasets=num_datasets,
                    )
                    val_stats = {
                        **val_stats,
                        **{f"adv_{k}": v for k, v in adv_val_stats.items()},
                    }
                    main_acc = adv_val_stats["weighted_acc"]
                    # If acc1 > clean_acc1, we likely have gradient obfuscation
                    # and so we should not save this model.
                    is_best = clean_acc >= main_acc > BEST_ACC

                if config["freeze_ensemble_but_first"]:
                    state_dict = model.module.members[0].state_dict()
                else:
                    state_dict = model.state_dict()
                save_dict = {
                    "epoch": epoch + 1,
                    "state_dict": state_dict,
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_acc": BEST_ACC,
                    "args": config,
                }

                if is_best:
                    logger.info("=> Saving new best checkpoint...")
                    save_on_master(save_dict, output_dir, is_best=True)
                    BEST_ACC = main_acc
                save_epoch = epoch + 1 if config["save_all_epochs"] else None
                save_on_master(
                    save_dict, output_dir, is_best=False, epoch=save_epoch
                )
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"val_{k}": v for k, v in val_stats.items()},
                "epoch": epoch,
            }

            if is_main_process():
                save_metrics["train"].append(log_stats)
                if config["wandb"]:
                    wandb.log(log_stats)
                logfile.write(json.dumps(log_stats) + "\n")
                logfile.flush()

        # Compute stats of best model after training
        dist_barrier()
        load_path = str(output_dir / "checkpoint_best.pt")

        logger.info("=> Loading checkpoint from %s...", load_path)
        if gpu is None:
            checkpoint = torch.load(load_path)
        else:
            # Map model to be loaded to specified single gpu
            checkpoint = torch.load(load_path, map_location=f"cuda:{gpu}")
        if config["freeze_ensemble_but_first"]:
            model.module.members[0].load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint["state_dict"])

    # Running evaluation
    for attack_name, attack in eval_attacks_list:
        # Use DataParallel (not distributed) model for AutoAttack.
        # Otherwise, DDP model can get timeout or c10d failure.
        # This can be fixed by using autograd instead of backward().
        stats, idx_correct = _validate(
            config,
            test_loader,
            model,
            criterion,
            attack,
            return_idx_correct=True,
            num_datasets=num_datasets,
        )
        logger.info("=> %s: %s", attack_name, str(stats))
        stats["attack"] = str(attack_name)
        dist_barrier()
        if is_main_process():
            save_metrics["test"].append({"idx_correct": idx_correct, **stats})
            if config["wandb"]:
                wandb.log(stats)
            logfile.write(json.dumps(stats) + "\n")

    if is_main_process():
        # Save metrics to pickle file if not exists else append
        metric_path: Path = output_dir / "metrics.pkl"
        if metric_path.exists():
            metrics = safe_pickle(metric_path, load=True)
            metrics.append(save_metrics)
        else:
            metrics = [save_metrics]
        safe_pickle(metric_path, obj=metrics, load=False)

        if config["data_load_mode"] != "load":
            # Save clean accuracy
            result_path: Path = output_dir / "results.pkl"
            logger.info("Saving result to %s", str(result_path))
            results = {}
            if result_path.exists():
                results = safe_pickle(result_path, load=True)
            if "test" not in results:
                results["test"] = {}
            for stats in save_metrics["test"]:
                results["test"][stats["attack"]] = stats["acc0"]
            safe_pickle(result_path, obj=results, load=False)

        last_path = output_dir / "checkpoint_last.pt"
        if last_path.exists():
            os.remove(str(last_path))
        logfile.close()

    return save_metrics


def _train(
    config: dict[str, Any],  # pylint: disable=redefined-outer-name
    train_loader,
    model,
    criterion,
    attack,
    optimizer,
    scaler,
    epoch,
    num_datasets: int = 1,
    eval_criterion: torch.nn.Module | None = None,
):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    mlosses = [AverageMeter(f"Loss{i}", ":.4e") for i in range(num_datasets)]
    maccs = [AverageMeter(f"Acc{i}", ":6.2f") for i in range(num_datasets)]
    weighted_loss = AverageMeter("Weighted Loss", ":.4e")
    weighted_acc = AverageMeter("Weighted Acc", ":6.2f")
    weighted_loss = AverageMeter("Weighted Loss", ":.4e")
    mem = AverageMeter("Mem (GB)", ":6.1f")
    progress = ProgressMeter(
        len(train_loader),
        [
            batch_time,
            data_time,
            weighted_loss,
            weighted_acc,
            *mlosses,
            *maccs,
            mem,
        ],
        prefix=f"Epoch: [{epoch}]",
    )

    gpu: int = config["gpu"]
    is_dual_loss: bool = criterion.is_dual_loss

    # Switch to train mode
    num_models = (
        1 if config["freeze_ensemble_but_first"] else len(config["arch"])
    )
    model.train()

    distill_layer = config["distill_layer"]
    if isinstance(model.module, DVGEnsemble):
        if not config["distill_fixed_layer"]:
            logger.info("[DVERGE] Randomly choosing a layer for distillation")
            distill_layer = random.randint(1, config["depth"])
        logger.info("Distillation layer selected is %s", distill_layer)

    def _restack_outputs(out, bs, nd, nc):
        return out.view(bs, nd * (2 if is_dual_loss else 1), nc)

    end = time.time()
    for i, (images, targets, _) in enumerate(train_loader):
        images: StackImages = images
        targets: StackLabels = targets
        if epoch == 0 and config["warmup_iters"] > 0:
            adjust_learning_rate(config, optimizer, epoch, iteration=i)

        # Measure data loading time
        data_time.update(time.time() - end)
        batch_size, _num_datasets, num_classes = targets.shape
        assert images.shape[1] == _num_datasets == num_datasets, (
            "Invalid images and/or targets shape! The 2nd dim must be "
            f"num_datasets ({num_datasets}) but got shape {images.shape} "
            f"(images) and {targets.shape} (targets)!"
        )
        images = images.cuda(gpu, non_blocking=True)
        targets = targets.cuda(gpu, non_blocking=True)
        # Flatten stack of images
        image_shape = images.shape[-3:]
        images: BatchImages = images.view(-1, *image_shape)

        # Compute output
        with amp.autocast(enabled=not config["full_precision"]):
            # Run training attack
            adv_images = attack(images, targets.view(-1, num_classes))
            clean_images = images
            if is_dual_loss:
                # Restack and then flatten adversarial images
                images = images.view(batch_size, -1, *image_shape)
                adv_images = adv_images.view(batch_size, -1, *image_shape)
                images = torch.cat([images, adv_images], dim=1)
                images = images.view(-1, *image_shape)
            else:
                images = adv_images

            if isinstance(model.module, Ensemble):
                # Linear warmup of epsilon for TRS
                # NOTE: It is not mentioned in TRS paper how init_eps is set.
                # We follow the official code that sets init_eps = 0.1 and
                # final_eps = 0.2 for MNIST.
                init_eps = 0.5 * config["epsilon"]
                inc_eps = config["epsilon"] - init_eps
                # Need to explicitely specify reduction == "none" for ensemble
                # trs_loss = ce loss (clean samples) + smoothness (clean samples) + cosine (clean samples)
                outputs, total_loss = model(
                    clean_images,
                    reduction="none",
                    compute_loss=True,
                    targets=targets,
                    criterion=eval_criterion,
                    epsilon=inc_eps * epoch / config["epochs"] + init_eps,
                    distill_layer=distill_layer,
                )
                # Average across DataParallel models
                total_loss = total_loss.mean()
                # If adversarial training is used, we need to compute the loss
                # on adversarial samples as well by updating the outputs.
                if config["train_attack"] != "none":
                    outputs: StackLogits = model(images, reduction="none")
                    if config["freeze_ensemble_but_first"]:
                        outputs = outputs.unsqueeze(1)

                for j in range(num_models):
                    # Restack outputs
                    model_output: StackLogits = _restack_outputs(
                        outputs[:, j], batch_size, num_datasets, num_classes
                    )
                    model_total_loss, loss_list = criterion(
                        model_output, targets
                    )
                    total_loss += model_total_loss
                outputs: BatchLogits = model.module.aggregate_outputs(outputs)
                outputs: StackLogits = _restack_outputs(
                    outputs, batch_size, num_datasets, num_classes
                )
            else:
                # Compute outputs and loss for a single model
                outputs: BatchLogits = model(images)
                outputs: StackLogits = _restack_outputs(
                    outputs, batch_size, num_datasets, num_classes
                )
                total_loss, loss_list = criterion(outputs, targets)

        # Measure accuracy and record loss
        wacc = 0
        for j in range(num_datasets):
            loss = loss_list[j]
            if not math.isfinite(loss.item()):
                logger.error("Loss%d is %f, stopping training", j, loss.item())
                sys.exit(1)
            # When is_dual_loss, we compute acc on adversarial samples
            acc, _ = accuracy(
                outputs[:, num_datasets + j] if is_dual_loss else outputs[:, j],
                targets[:, j].argmax(1),
            )
            mlosses[j].update(loss.item(), batch_size)
            maccs[j].update(acc.item(), batch_size)
            wacc += criterion.weights[j] * acc
        weighted_acc.update(wacc.item(), batch_size)
        weighted_loss.update(total_loss.item(), batch_size)

        # Compute gradient and do SGD step
        scaler.scale(total_loss).backward()
        if config["clip_grad_norm"] is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["clip_grad_norm"]
            )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        mem.update(torch.cuda.max_memory_allocated() // 1e9)

        if i % config["print_freq"] == 0:
            if is_main_process() and config["wandb"]:
                loss_weights = criterion.weights.tolist()
                stats = {
                    **{f"acc{i}": acc.avg for i, acc in enumerate(maccs)},
                    **{f"loss{i}": loss.avg for i, loss in enumerate(mlosses)},
                    "weighted_acc": weighted_acc.avg,
                    "scaler": scaler.get_scale(),
                    **{
                        f"loss_weight{i}": w for i, w in enumerate(loss_weights)
                    },
                }
                wandb.log(stats)
            progress.display(i)

    progress.synchronize()
    return {
        **{f"acc{i}": acc.avg for i, acc in enumerate(maccs)},
        **{f"loss{i}": loss.avg for i, loss in enumerate(mlosses)},
        "weighted_acc": weighted_acc.avg,
        "lr": optimizer.param_groups[0]["lr"],
    }


def _validate(
    config: dict[str, Any],  # pylint: disable=redefined-outer-name
    val_loader,
    model,
    criterion,
    attack,
    return_idx_correct: bool = False,
    num_datasets: int = 1,
):
    gpu: int = config["gpu"]
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    mlosses = [AverageMeter(f"Loss{i}", ":.4e") for i in range(num_datasets)]
    maccs = [AverageMeter(f"Acc{i}", ":6.2f") for i in range(num_datasets)]
    weighted_loss = AverageMeter("Weighted Loss", ":.4e")
    weighted_acc = AverageMeter("Weighted Acc", ":6.2f")
    weighted_loss = AverageMeter("Weighted Loss", ":.4e")
    mem = AverageMeter("Mem (GB)", ":6.1f")
    progress = ProgressMeter(
        len(val_loader),
        [
            batch_time,
            data_time,
            weighted_loss,
            weighted_acc,
            *mlosses,
            *maccs,
            mem,
        ],
        prefix="Test: ",
    )
    idx_correct = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (images, targets, _) in enumerate(val_loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        images: StackImages = images.cuda(gpu, non_blocking=True)
        targets: StackLabels = targets.cuda(gpu, non_blocking=True)
        image_shape = images.shape[-3:]
        batch_size, _num_datasets, num_classes = targets.shape
        assert _num_datasets == num_datasets, (
            "Invalid targets shape! The 2nd dim must be num_datasets "
            f"({num_datasets}) but get shape {targets.shape}!"
        )
        image_shape = images.shape[-3:]
        # Flatten stack of images
        images: BatchImages = images.view(-1, *image_shape)

        # Compute output
        with torch.no_grad():
            images = attack(images, targets.view(-1, num_classes))
            outputs = model(images)
            outputs = outputs.view(batch_size, num_datasets, num_classes)
            total_loss, loss_list = criterion(outputs, targets)

        # Measure accuracy and record loss
        wacc = 0
        for j in range(num_datasets):
            loss = loss_list[j]
            if not math.isfinite(loss.item()):
                logger.error("Loss%d is %f, stopping eval", j, loss.item())
                sys.exit(1)
            acc, idx = accuracy(outputs[:, j], targets[:, j].argmax(1))
            idx_correct.append(idx.cpu().numpy())
            mlosses[j].update(loss.item(), batch_size)
            maccs[j].update(acc.item(), batch_size)
            wacc += criterion.weights[j] * acc
        weighted_acc.update(wacc.item(), batch_size)
        weighted_loss.update(total_loss.item(), batch_size)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        mem.update(torch.cuda.max_memory_allocated() // 1e9)
        if i % config["print_freq"] == 0:
            progress.display(i)

    progress.synchronize()
    logger.info(" * Acc %s", " ".join([f"{acc.avg:.3f}" for acc in maccs]))
    logger.info(" * Weighted Acc %.3f", weighted_acc.avg)

    return_dict = {
        **{f"acc{i}": acc.avg for i, acc in enumerate(maccs)},
        **{f"loss{i}": loss.avg for i, loss in enumerate(mlosses)},
        "weighted_acc": weighted_acc.avg,
    }
    if return_idx_correct:
        idx_correct = np.concatenate(idx_correct, axis=0)
        return return_dict, idx_correct
    return return_dict


if __name__ == "__main__":
    main_parser = argparse.ArgumentParser(
        "Transfer attack defense", parents=[get_args_parser()]
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

    train_test_main(config)

"""PyTorch distributed training functionality."""

from __future__ import annotations

import datetime
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


def setup_for_distributed(is_master: bool) -> None:
    """This function disables printing when not in master process."""
    import builtins as __builtin__  # pylint: disable=import-outside-toplevel

    builtin_print = __builtin__.print

    # pylint: disable=redefined-builtin
    def print(*args, **kwargs) -> None:
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized() -> bool:
    """Check whether distributed mode is available and initialized."""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size() -> int:
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def dist_barrier():
    if is_dist_avail_and_initialized():
        torch.distributed.barrier()


def is_main_process():
    return get_rank() == 0


def save_on_master(
    state: Any, output_dir: Path, is_best: bool = True, epoch: int | None = None
) -> None:
    if is_main_process():
        path = str(
            output_dir
            / ("checkpoint_best.pt" if is_best else "checkpoint_last.pt")
        )
        torch.save(state, path)
        # Save to best model if not exist
        if not os.path.exists(path):
            torch.save(state, path)

        if epoch is not None:
            path = str(output_dir / f"checkpoint_epoch{epoch}.pt")
            torch.save(state, path)


def init_distributed_mode(args):
    if args["no_distributed"]:
        args["distributed"] = False
        rank = 0
        args["gpu"] = None
        return
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        args["world_size"] = int(os.environ["WORLD_SIZE"])
        args["gpu"] = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        args["gpu"] = rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args["distributed"] = False
        rank = 0
        args["gpu"] = None
        return

    args["distributed"] = True
    torch.cuda.set_device(args["gpu"])
    print(
        f"| distributed init (rank {rank}): {args['dist_url']}",
        flush=True,
    )
    torch.distributed.init_process_group(
        backend=args["dist_backend"],
        init_method=args["dist_url"],
        world_size=args["world_size"],
        rank=rank,
        # Set timeout to 2 hours for gloo
        timeout=datetime.timedelta(hours=1),
    )
    torch.distributed.barrier()
    setup_for_distributed(rank == 0)

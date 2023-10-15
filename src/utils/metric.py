"""Metric utilities."""

import fcntl
import math
import pathlib
import pickle
import random
import time
from typing import Any

import torch
import torch.distributed as dist


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def synchronize(self):
        t = torch.tensor(
            [self.sum, self.count], dtype=torch.float64, device="cuda"
        )
        if dist.is_initialized():
            dist.barrier()
            dist.all_reduce(t)
        t = t.tolist()
        self.sum = int(t[0])
        self.count = t[1]
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def synchronize(self):
        for meter in self.meters:
            meter.synchronize()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def adjust_learning_rate(
    config, optimizer, epoch: int, iteration: int | None = None
) -> float:
    """Decay the learning rate based on schedule."""
    learning_rate = config["lr"]
    if iteration is not None and iteration < config["warmup_iters"]:
        learning_rate *= (iteration + 1) / config["warmup_iters"]
    elif epoch < config["warmup_epochs"]:
        # Linear warmup
        learning_rate *= (epoch + 1) / config["warmup_epochs"]

    if config["lr_schedule"] == "step":
        factor = sum(epoch >= step for step in config["lr_steps"])
        learning_rate *= config["lr_step_size"]**factor
    elif config["lr_schedule"] == "cosine":
        learning_rate *= 0.5 * (
            1.0 + math.cos(math.pi * epoch / config["epochs"])
        )
    else:
        raise NotImplementedError(
            f"Unknown lr schedule: {config['lr_schedule']}"
        )

    # Apply new learning rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate
    return learning_rate


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        # Return only top1
        return res[0], correct[0]


def safe_pickle(
    pkl_path: pathlib.Path | str,
    obj: Any | None = None,
    load: bool = True,
    timeout: int = 10,
    retry: int = 200,
    rand_timeout: bool = True,
) -> Any | None:
    """Pickle dump/load with retry on failure."""
    if isinstance(pkl_path, str):
        pkl_path = pathlib.Path(pkl_path)

    for _ in range(retry):
        try:
            with pkl_path.open("rb" if load else "wb") as pkl_file:
                fcntl.flock(pkl_file, fcntl.LOCK_EX)
                if load:
                    obj = pickle.load(pkl_file)
                else:
                    assert obj is not None
                    pickle.dump(obj, pkl_file)
                fcntl.flock(pkl_file, fcntl.LOCK_UN)
            return obj
        except (EOFError, pickle.UnpicklingError):
            time.sleep(timeout * random.random() if rand_timeout else timeout)
    raise RuntimeError(
        f"Failed to pickle {str(pkl_path)} after {retry} trials!"
    )

"""Utility for printing results."""

from __future__ import annotations

import os
import sys

import numpy as np
from tabulate import tabulate

from exp_params import ATTACKS, PUBLIC_MODELS
from src.utils.metric import safe_pickle


class HiddenPrints:
    def __init__(self, enable: bool = True) -> None:
        self._enable: bool = enable
        self._original_stdout = None

    def __enter__(self):
        if self._enable:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._enable:
            sys.stdout.close()
            sys.stdout = self._original_stdout


class Aggregator:
    def __init__(self, num_axes: int = 2, dataset: str = "cifar10") -> None:
        self.num_axes: int = num_axes
        self.table: dict[str | tuple[str, ...], list[float]] = {}
        self.keys = [set() for _ in range(num_axes)]
        self.dataset = dataset

    def update(
        self, key: str | tuple[str, ...], vals: np.ndarray | None = None
    ) -> None:
        """Update the aggregator with a new value."""
        if vals is None:
            return
        key = key if isinstance(key, tuple) else (key,)
        if len(key) != self.num_axes:
            raise ValueError(
                f"key length {len(key)} must be the equal to num_axes "
                f"({self.num_axes})!"
            )
        for i, k in enumerate(key):
            self.keys[i].add(k)
        self.table[key] = vals

    def get_all(
        self, name: str | None = None, axis: int | None = None
    ) -> dict[str | tuple[str, ...], float]:
        return_dict = {}
        for key, val in self.table.items():
            if axis is not None and name is not None and key[axis] != name:
                continue
            remain_key = key
            if axis is not None:
                remain_key = key[:axis] + key[axis + 1 :]
            if len(remain_key) == 1:
                remain_key = remain_key[0]
            return_dict[remain_key] = np.mean(val)
        return return_dict

    def get_avg(
        self, name: str | list[str] | None = None, axis: int = 0
    ) -> float:
        means = []
        name = [name] if isinstance(name, str) else name
        if name is not None and not isinstance(name, (list, tuple)):
            raise ValueError("name must be a string or list of string or None!")
        for key, val in self.table.items():
            if name is None or key[axis] in name:
                means.append(np.mean(val))
        if not means:
            return None
        return np.mean(means)

    def get_best(
        self, name: str | list[str] | None = None, axis: int = 0
    ) -> dict[str, float]:
        best = None
        name = [name] if isinstance(name, str) else name
        if name is not None and not isinstance(name, (list, tuple)):
            raise ValueError("name must be a string or list of string or None!")
        for key, val in self.table.items():
            if name is not None and key[axis] not in name:
                continue
            if best is None:
                best = np.copy(val)
            else:
                best |= val
        best = np.mean(best) if best is not None else None
        return best

    def get_group_best(self, group: str) -> float:
        """Get the best result for a group of models."""
        results = []
        for src in PUBLIC_MODELS[self.dataset][group]:
            for key, val in self.table.items():
                if src not in key:
                    continue
                results.append(np.mean(val))
        return np.max(results)

    def get_keys(self, axis: int = 0) -> list[str]:
        return sorted(list(self.keys[axis]))


def _format_row(row: list[str | float]) -> list[str]:
    for i, col in enumerate(row):
        if isinstance(col, str):
            col += ","
        elif isinstance(col, float):
            col = f"{col * 100:.2f},"
        row[i] = col
    return row


def print_result(
    attacks: list[str],
    source_models: list[str],
    aggregator: Aggregator,
    dataset: str = "cifar10",
) -> tuple[float, float, float]:
    """Print results in a table."""
    # Sort models based on PUBLIC_MODELS
    are_models_public = [False] * len(source_models)
    public_models, other_models, groups = [], [], []
    for group, models in PUBLIC_MODELS[dataset].items():
        for model in models:
            if model not in source_models:
                continue
            public_models.append(model)
            if group not in groups:
                groups.append(group)
            else:
                groups.append(None)
            are_models_public[source_models.index(model)] = True
    for is_public, model in zip(are_models_public, source_models):
        if not is_public:
            other_models.append(model)
            groups.append(None)
    source_models = [*public_models, *other_models]
    headers = ["Sources", *attacks, "Avg.", "Best", "Best (group)"]

    print("===================== Results: src/attack =====================")
    table = []
    for src, grp in zip(source_models, groups):
        vals = ["r"] * len(attacks)
        for atk, val in aggregator.get_all(src, axis=0).items():
            vals[attacks.index(atk)] = val
        avg = aggregator.get_avg(src, axis=0)
        best = aggregator.get_best(src, axis=0)
        row = [src, *vals, avg, best]
        if grp is not None:
            row.append(aggregator.get_best(PUBLIC_MODELS[dataset][grp], axis=0))
        else:
            row.append("")
        row = _format_row(row)
        table.append(row)
    print(tabulate(table, headers=headers))
    print()
    print("===================== Results: attack/src =====================")
    table = []
    for atk in attacks:
        vals = ["r"] * len(source_models)
        for src, val in aggregator.get_all(atk, axis=1).items():
            vals[source_models.index(src)] = val
        avg = aggregator.get_avg(atk, axis=1)
        best = aggregator.get_best(atk, axis=1)
        row = _format_row([atk, *vals, avg, best])
        table.append(row)
    headers = ["Attacks", *source_models, "Avg.", "Best"]
    print(tabulate(table, headers=headers[:-1]))

    if not aggregator.table:
        return None, None, None

    overall_best_err = aggregator.get_best() * 100
    avg_err = aggregator.get_avg() * 100
    best_err = max(aggregator.get_all().values()) * 100
    print(
        f"Error rate (avg, best, per-sample best): {avg_err:.2f}, "
        f"{best_err:.2f}, {overall_best_err:.2f}"
    )
    print(
        f"Accuracy (avg, best, per-sample best): {100 - avg_err:.2f}, "
        f"{100 - best_err:.2f}, {100 - overall_best_err:.2f}"
    )
    return avg_err, best_err, overall_best_err


def get_result_aggregator(
    model_name: str, attack_suffix: str = ""
) -> Aggregator:
    """Get the result aggregator for the given model_name."""
    tgt_path = f"results/{model_name}/results.pkl"
    results = safe_pickle(tgt_path, load=True)
    aggregator = Aggregator(num_axes=2, dataset=model_name.split("_")[0])
    attacks = {}  # Use dict instead of set to preserver order
    srcs = []
    for src in results.keys():
        # pylint: disable=unsupported-membership-test
        # if _SRCS is not None and src not in _SRCS:
        #     continue
        if src == "test":
            continue
        srcs.append(src)
        for attack in ATTACKS:
            attacks[attack] = None
            idx_success = results[src].get(attack + attack_suffix)
            aggregator.update((src, attack), idx_success)
    return aggregator

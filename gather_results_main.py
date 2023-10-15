"""Script to gather results from multiple runs and print them in tables."""

from __future__ import annotations

from pathlib import Path

from tabulate import tabulate

from src.utils.metric import safe_pickle
from src.utils.print_result import Aggregator, print_result

# Specify attacks to include in the table
_ATTACKS = [
    "pgd",
    "mpgd",
    "pregradient",
    "di",
    "ti",
    "admix",
    "naa",
    "ni-si-ti-dim",
    "ni-admix-ti-dim",
    "auto-pgd-dlr",
    "auto-pgd-ce",
]
_ATTACK_SUFFIX = "_test_temp1.0_10step"

# Specify target model(s)
# regex = "cifar10_*_pgd_ce_seed0_bs1024*"
regex = "cifar10_*_trades_trades_seed0_bs*"
# regex = "cifar10_resnext29-8x64d_pgd_ce_seed0_bs512*"
# regex = "cifar10_trs_resnet18+resnet18+resnet18+resnet18*"
# regex = "cifar10_trs_freeze_wideresnet34-10+a6+0a+78+c0_none_ce*"
_TGTS = [p.name for p in sorted(Path("results").glob(regex))]
# _TGTS = [
tgts = [
    # "cifar10_dla_none_ce_seed0_lr0.1_wd0.0005_sgd",
    # "cifar10_dla_pgd_ce_seed0_lr0.1_wd0.0005_sgd_eps8",
    # "cifar10_efficientnetb0_none_ce_seed0_lr0.1_wd0.0005_sgd",
    # "cifar10_efficientnetb0_pgd_ce_seed0_lr0.1_wd0.0005_sgd_eps8",
    # "cifar10_mobilenetv2_none_ce_seed0_lr0.1_wd0.0005_sgd",
    # "cifar10_mobilenetv2_pgd_ce_seed0_lr0.1_wd0.0005_sgd_eps8",
    # "cifar10_resnet18_none_ce_seed0_lr0.1_wd0.0005_sgd",
    # "cifar10_resnet18_none_ce_seed1_lr0.1_wd0.0005_sgd",
    # "cifar10_resnet18_pgd_ce_seed0_lr0.1_wd0.0005_sgd_eps8",
    # "cifar10_resnext29-2x64d_none_ce_seed0_lr0.1_wd0.0005_sgd",
    # "cifar10_resnext29-2x64d_pgd_ce_seed0_lr0.1_wd0.0005_sgd_eps8",
    # "cifar10_senet18_none_ce_seed0_lr0.1_wd0.0005_sgd",
    # "cifar10_senet18_pgd_ce_seed0_lr0.1_wd0.0005_sgd_eps8",
    # "cifar10_clip",
    # "cifar10_resnet18_none_mat_seed0_src-8e57_atk-b06e_path-a571_lr0.1_wd0.0005_sgd_beta0.5_dw1.0-0.1-0.9_do0.0-0.0-0.0",
    # "cifar10_resnet18_none_mat_seed0_src-8e57_atk-b06e_path-a571_lr0.1_wd0.0005_sgd_beta0.5_dw1.0-0.2-0.8_do0.0-0.0-0.0",
    # "cifar10_resnet18_none_mat_seed0_src-8e57_atk-b06e_path-a571_lr0.1_wd0.0005_sgd_beta0.5_dw1.0-0.3-0.7_do0.0-0.0-0.0",
    # "cifar10_resnet18_none_mat_seed0_src-8e57_atk-b06e_path-a571_lr0.1_wd0.0005_sgd_beta0.5_dw1.0-0.4-0.6_do0.0-0.0-0.0",
    # "cifar10_resnet18_none_mat_seed0_src-8e57_atk-b06e_path-a571_lr0.1_wd0.0005_sgd_beta0.5_dw1.0-0.5-0.5_do0.0-0.0-0.0",
    # "cifar10_resnet18_none_mat_seed0_src-8e57_atk-b06e_path-a571_lr0.1_wd0.0005_sgd_beta0.5_dw1.0-0.6-0.4_do0.0-0.0-0.0",
    # "cifar10_resnet18_none_mat_seed0_src-8e57_atk-b06e_path-a571_lr0.1_wd0.0005_sgd_beta0.5_dw1.0-0.7-0.3_do0.0-0.0-0.0",
    # "cifar10_resnet18_none_mat_seed0_src-8e57_atk-b06e_path-a571_lr0.1_wd0.0005_sgd_beta0.5_dw1.0-0.8-0.2_do0.0-0.0-0.0",
    # "cifar10_resnet18_none_mat_seed0_src-8e57_atk-b06e_path-a571_lr0.1_wd0.0005_sgd_beta0.5_dw1.0-0.9-0.1_do0.0-0.0-0.0",
    # "cifar10_efficientnetb0_none_mat_seed0_src-8e57_atk-b06e_path-a571_lr0.1_wd0.0005_sgd_beta0.5_dw1.0-0.1-0.9_do0.0-0.0-0.0",
    # "cifar10_efficientnetb0_none_mat_seed0_src-8e57_atk-b06e_path-a571_lr0.1_wd0.0005_sgd_beta0.5_dw1.0-0.3-0.7_do0.0-0.0-0.0",
    # "cifar10_efficientnetb0_none_mat_seed0_src-8e57_atk-b06e_path-a571_lr0.1_wd0.0005_sgd_beta0.5_dw1.0-0.5-0.5_do0.0-0.0-0.0",
    # "cifar10_efficientnetb0_none_mat_seed0_src-8e57_atk-b06e_path-a571_lr0.1_wd0.0005_sgd_beta0.5_dw1.0-0.7-0.3_do0.0-0.0-0.0",
    # "cifar10_efficientnetb0_none_mat_seed0_src-8e57_atk-b06e_path-a571_lr0.1_wd0.0005_sgd_beta0.5_dw1.0-0.9-0.1_do0.0-0.0-0.0",
    # "cifar10_resnet18_none_mat_seed0_src-e8be_atk-b06e_path-208d_lr0.1_wd0.0005_sgd_beta0.5_dw1.0-1.0_do0.0-0.0",
    # "cifar10_resnet18_none_mat_seed0_src-041e_atk-b06e_path-d84e_lr0.1_wd0.0005_sgd_beta0.5_dw1.0-1.0_do0.0-0.0",
    # "cifar10_resnet18_none_mat_seed0_src-8e57_atk-b06e_path-a571_lr0.1_wd0.0005_sgd_beta0.5_dw1.0-0.05-0.95_do0.0-0.0-0.0",
    # "cifar10_resnet18_none_mat_seed0_src-8e57_atk-b06e_path-a571_lr0.1_wd0.0005_sgd_beta0.5_dw1.0-0.02-0.98_do0.0-0.0-0.0",
    # "cifar10_resnet18_none_mat_seed0_src-8e57_atk-b06e_path-a571_lr0.1_wd0.0005_sgd_beta0.5_dw1.0-0.01-0.99_do0.0-0.0-0.0",
    # "cifar10_resnet18_none_mat_seed0_src-8e57_atk-b06e_path-a571_lr0.1_wd0.0005_sgd_beta0.5_dw1.0-0.001-0.999_do0.0-0.0-0.0",
    # "cifar10_resnet18_none_mat_seed0_src-6f26_atk-b06e_path-9bab_lr0.1_wd0.0005_sgd_beta0.5_dw1.0-1.0_do0.0-0.0",
    # "cifar10_resnet18_none_mat_seed0_src-35e1_atk-b06e_path-9584_lr0.1_wd0.0005_sgd_beta0.5_dw1.0-0.3333-0.3333-0.3333_do0.0-0.0-0.0-0.0",
    # "cifar10_resnet18_none_mat_seed0_src-48c4_atk-b06e_path-e023_lr0.1_wd0.0005_sgd_beta0.5_dw1.0-0.3333-0.3333-0.3333_do0.0-0.0-0.0-0.0",
    # "cifar10_resnet18_none_mat_seed0_src-3749_atk-b06e_path-3850_lr0.1_wd0.0005_sgd_beta0.5_dw1.0-0.25-0.25-0.25-0.25_do0.0-0.0-0.0-0.0-0.0",
]

# Specify source model(s). None to include all that are available.
_SRCS = None
# _SRCS = [
#     "cifar10_dla_none_ce_seed0_lr0.1_wd0.0005_sgd",
#     "cifar10_dla_pgd_ce_seed0_lr0.1_wd0.0005_sgd_eps8",
#     "cifar10_efficientnetb0_none_ce_seed0_lr0.1_wd0.0005_sgd",
#     "cifar10_efficientnetb0_pgd_ce_seed0_lr0.1_wd0.0005_sgd_eps8",
#     "cifar10_mobilenetv2_none_ce_seed0_lr0.1_wd0.0005_sgd",
#     "cifar10_mobilenetv2_pgd_ce_seed0_lr0.1_wd0.0005_sgd_eps8",
#     # "cifar10_resnet18_none_ce_seed0_lr0.1_wd0.0005_sgd",
#     # "cifar10_resnet18_none_ce_seed1_lr0.1_wd0.0005_sgd",
#     # "cifar10_resnet18_pgd_ce_seed0_lr0.1_wd0.0005_sgd_eps8",
#     "cifar10_resnext29-2x64d_none_ce_seed0_lr0.1_wd0.0005_sgd",
#     "cifar10_resnext29-2x64d_pgd_ce_seed0_lr0.1_wd0.0005_sgd_eps8",
#     "cifar10_senet18_none_ce_seed0_lr0.1_wd0.0005_sgd",
#     "cifar10_senet18_pgd_ce_seed0_lr0.1_wd0.0005_sgd_eps8",
# ]
# _SRCS = [
#     "cifar10_resnet18_none_ce_seed1_lr0.1_wd0.0005_sgd",
#     "cifar10_resnet18_pgd_ce_seed0_lr0.1_wd0.0005_sgd_eps8",
# ]


def main():
    """Main function."""
    table = []
    for tgt in _TGTS:
        tgt_path = f"results/{tgt}/results.pkl"
        metrics_path = f"results/{tgt}/metrics.pkl"
        try:
            results = safe_pickle(tgt_path, load=True)
            metrics = safe_pickle(metrics_path, load=True)
        except FileNotFoundError:
            continue
        aggregator = Aggregator(num_axes=2)
        attacks = {}  # Use dict instead of set to preserver order
        srcs = []
        for src in results.keys():
            # pylint: disable=unsupported-membership-test
            if _SRCS is not None and src not in _SRCS:
                continue
            if src == "test":
                continue
            srcs.append(src)
            for attack in _ATTACKS:
                attacks[attack] = None
                idx_success = results[src].get(attack + _ATTACK_SUFFIX)
                aggregator.update((src, attack), idx_success)

        print(f"Target: {tgt}")
        avg_err, best_err, overall_best_err = print_result(
            _ATTACKS, srcs, aggregator
        )
        # clean_acc = metrics[0]["test"]["acc0"]
        # clean_metric = metrics[0]["test"][0]
        # clean_acc, clean_atk = clean_metric["acc0"], clean_metric["attack"]
        # assert clean_atk == "none"
        clean_acc = results["test"]["none"]  # Gather new result with this line
        row = [tgt, clean_acc] + [""] * 4
        if avg_err is not None:
            row[2:5] = [100 - avg_err, 100 - best_err, 100 - overall_best_err]
            print(
                "Overall best, avg:",
                aggregator.get_best() * 100,
                aggregator.get_avg() * 100,
            )
        if "pgd" in results["test"]:
            row[5] = results["test"]["pgd"]
        table.append([f"{r}," for r in row])
        print("\n\n")

    print("Summary:")
    table = tabulate(
        table, headers=["Target", "Clean", "Avg", "Best", "Overall Best", "PGD"]
    )
    print(table)


if __name__ == "__main__":
    main()

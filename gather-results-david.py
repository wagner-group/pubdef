import pickle

from src.utils.metric import safe_pickle
from src.utils.print_result import Aggregator, print_result

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
]

_TGTS = [
    "cifar10_resnet18_none_ce_seed1_src-318d_atk-b06e_path-b34d_lr0.1_wd0.0005_sgd_dw1.0-0.5-0.5_do0.0-0.0-0.0",
    "cifar10_resnet18_none_ce_seed0_src-318d_atk-b06e_path-b34d_lr0.1_wd0.0005_sgd_dw1.0-1.0-1.0_do0.0-0.3-0.7",
    "cifar10_resnet18_none_ce_seed0_src-318d_atk-b06e_path-b34d_lr0.1_wd0.0005_sgd_dw1.0-1.0-1.0_do0.0-0.5-0.5",
    "cifar10_resnet18_none_ce_seed0_src-318d_atk-b06e_path-b34d_lr0.1_wd0.0005_sgd_dw1.0-1.0-1.0_do0.0-1.0-0.0",
    "cifar10_resnet18_none_ce_seed0_src-318d_atk-b06e_path-b34d_lr0.1_wd0.0005_sgd_dw1.0-1.0-1.0_do0.0-0.1-0.9",
    "cifar10_resnet18_none_ce_seed0_src-318d_atk-b06e_path-b34d_lr0.1_wd0.0005_sgd_dw1.0-1.0-1.0_do0.0-0.0-1.0",
    "cifar10_resnet18_none_ce_seed0_src-318d_atk-b06e_path-b34d_lr0.1_wd0.0005_sgd_dw1.0-1.0-1.0_do0.0-0.2-0.8",
    "cifar10_resnet18_none_ce_seed0_src-318d_atk-b06e_path-b34d_lr0.1_wd0.0005_sgd_dw1.0-1.0-1.0_do0.0-0.3-0.7",
    "cifar10_resnet18_none_ce_seed0_src-318d_atk-b06e_path-b34d_lr0.1_wd0.0005_sgd_dw1.0-1.0-1.0_do0.0-0.4-0.6",
    "cifar10_resnet18_none_ce_seed0_src-318d_atk-b06e_path-b34d_lr0.1_wd0.0005_sgd_dw1.0-1.0-1.0_do0.0-0.6-0.4",
    "cifar10_resnet18_none_ce_seed0_src-318d_atk-b06e_path-b34d_lr0.1_wd0.0005_sgd_dw1.0-1.0-1.0_do0.0-0.7-0.3",
    "cifar10_resnet18_none_ce_seed0_src-318d_atk-b06e_path-b34d_lr0.1_wd0.0005_sgd_dw1.0-1.0-1.0_do0.0-0.8-0.2",
    "cifar10_resnet18_none_ce_seed0_src-318d_atk-b06e_path-b34d_lr0.1_wd0.0005_sgd_dw1.0-1.0-1.0_do0.0-0.9-0.1",
    "cifar10_resnet18_none_ce_seed0_src-318d_atk-b06e_path-b34d_lr0.1_wd0.0005_sgd_dw1.0-1.0-1.0_do0.0-0.4-0.6",
    "cifar10_resnet18_none_ce_seed1_src-318d_atk-b06e_path-b34d_lr0.1_wd0.0005_sgd_dw1.0-1.0-1.0_do0.0-0.4-0.6",
    "cifar10_resnet18_none_ce_seed1_src-318d_atk-b06e_path-b34d_lr0.1_wd0.0005_sgd_dw1.0-0.0-1.0_do0.0-0.0-0.0",
    "cifar10_resnet18_none_ce_seed1_src-318d_atk-b06e_path-b34d_lr0.1_wd0.0005_sgd_dw1.0-1.0-0.0_do0.0-0.0-0.0",
    "cifar10_resnet18_none_mat_seed1_src-318d_atk-b06e_path-b34d_lr0.1_wd0.0005_sgd_beta0.5_dw1.0-1.0-1.0_do0.0-0.0-0.0",
    "cifar10_resnet18_none_ce_seed1_src-318d_atk-b06e_path-b34d_lr0.1_wd0.0005_sgd_dw1.0-0.5-0.5_do0.0-0.0-0.0",
    "cifar10_resnet18_none_mat_seed1_src-318d_atk-b06e_path-b34d_lr0.1_wd0.0005_sgd_beta0.5_dw1.0-0.0-1.0_do0.0-0.0-0.0",
    "cifar10_resnet18_none_mat_seed1_src-318d_atk-b06e_path-b34d_lr0.1_wd0.0005_sgd_beta0.5_dw1.0-1.0-0.0_do0.0-0.0-0.0",
    "cifar10_resnet18_none_ce_seed0_src-c8f7_atk-b06e_path-f0b8_lr0.1_wd0.0005_sgd_dw1.0-1.0-1.0_do0.0-0.0-0.0"
]

_SRCS = [
#     "cifar10_resnet18_none_ce_seed1_lr0.1_wd0.0005_sgd",
#     "cifar10_resnet18_pgd_ce_seed0_lr0.1_wd0.0005_sgd_eps8",
      #"cifar10_resnet18_none_seed1",
      #"cifar10_resnet18_pgd_seed2",
      "cifar10_resnet18_none_seed9",
      "cifar10_resnet18_pgd_seed9",
]


def main():
    for tgt in _TGTS:
        tgt_path = f"results/{tgt}/results.pkl"
        results = safe_pickle(tgt_path, load=True)
        src_agg = Aggregator()
        atk_agg = Aggregator()
        best_agg = Aggregator()
        attacks = {} # Use dict instead of set to preserver order
        srcs = []
        for src in results.keys():
            # pylint: disable=unsupported-membership-test
            if _SRCS is not None and src not in _SRCS:
                continue
            srcs.append(src)
            for attack in _ATTACKS:
                attacks[attack] = None
                idx_success = results[src].get(attack)
                src_agg.update(src, idx_success)
                atk_agg.update(attack, idx_success)
                best_agg.update("best", idx_success)

        print(f"Target: {tgt}")
        print_result(_ATTACKS, srcs, src_agg, atk_agg)
        print("Best:", best_agg.best["best"] * 100)
        print("\n\n")


if __name__ == "__main__":
    main()

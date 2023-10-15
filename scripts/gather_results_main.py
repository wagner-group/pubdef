"""Script for gathering transfer attack accuracy for a given target model."""

import argparse
import pickle

from src.utils.print_result import Aggregator, print_result


def main():
    """Main function."""
    tgt_path = f"results/{args.tgt_model}/results.pkl"
    with open(tgt_path, "rb") as file:
        results = pickle.load(file)

    src_agg = Aggregator()
    atk_agg = Aggregator()
    attacks = {}  # Use dict instead of set to preserver order
    for src in results.keys():
        for attack in results[src].keys():
            attacks[attack] = None
            idx_success = results[src][attack]
            src_agg.update(src, idx_success)
            atk_agg.update(attack, idx_success)

    print(f"Target: {args.tgt_model}")
    attacks = list(attacks.keys())
    print_result(attacks, list(results.keys()), src_agg, atk_agg)


if __name__ == "__main__":
    # Specify target model here or as command-line argument
    TGT = "cifar10_resnet18_none_ce_seed0_lr0.1_wd0.0005_sgd"

    parser = argparse.ArgumentParser(
        description="Target model to get results.", add_help=False
    )
    parser.add_argument("tgt_model", default=TGT, type=str)
    args = parser.parse_args()
    main()

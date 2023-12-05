"""Script for running transfer attack experiment."""

from __future__ import annotations

import argparse
from typing import Any

from src.utils.argparser import get_args_parser
from transfer_exp_main import run_exp


def main(config) -> None:
    """Run transfer experiments for multiple sources, targets, and attacks.

    WARNING: metrics["idx_correct"] returned from train_test_main() should have
    a fixed length (= num test samples), but there is no guarantee that the
    order of the samples remained the same across runs. We fix random seed in
    all the scripts so the ordering should be correct when run in the same
    environment (same package version and machine), but beyond that is unclear.
    """
    # ================= Specify sources, targets, and attacks ================ #
    source_models = [
        # "cifar10_rn18_normal_seed0",
        # "cifar10_rn18_normal_seed1",
        # "cifar10_rn18_normal_seed2",
        # "cifar10_rn18_normal_seed3",
        # "cifar10_rn18_tdef_seed0-src-cifar10_rn18_normal_seed0",
        # "cifar10_rn18_tdef_seed0-src-cifar10_rn18_normal_seed1",
        # "cifar10_rn18_tdef_seed0-src-cifar10_rn18_normal_seed2",
        # "cifar10_rn18_tdef_seed0-src-cifar10_rn18_normal_seed3",
        # "cifar10_rn18_tdef_seed1-src-cifar10_rn18_normal_seed1",
        # "cifar10_rn18_pgd_seed0",
        # "cifar10_rn18_pgd_seed1",
        # "cifar10_efficientnetb0_normal_seed0",
        # "cifar10_mobilenetv2_normal_seed0",
        # "cifar10_rn18_tdef_seed1-src-cifar10_rn18_pgd-2_seed1",
        # "cifar10_rn18_tdef_seed0-src-cifar10_rn18_pgd-2_seed0",
        # "cifar10_rn18_pgd-2_seed1",
        # "cifar10_rn18_normal_seed1",
        # "cifar10_efficientnetb0_tdef_seed0-src-cifar10_rn18_pgd-2_seed0-normal_init",
        # "cifar10_mobilenetv2_tdef_seed0-src-cifar10_rn18_pgd-2_seed0-normal_init",
        "cifar10_senet18_tdef_seed8-src-3_pgd_2-init_none_seed8",
    ]
    # target_models = source_models
    target_models = [
        # "cifar10_rn18_tdef_seed0-src-cifar10_rn18_normal_seed1",
        # "cifar10_rn18_normal_seed0",
        # "cifar10_rn18_tdef_seed1-src-cifar10_rn18_pgd-2_seed0-normal_init",
        # "cifar10_rn18_pgd-2_seed0",
        "cifar10_resnet18_tdef_seed0-src-3_pgd_2-init_normal_seed0",
    ]
    attacks = [
        "pgd",
        "mpgd",
        "pregradient",
        "di",
        "ti",
        "admix",
        "naa",
        # "iaa",
        # "ni-si-ti-dim",
        # "ni-admix-ti-dim",
    ]
    # ======================================================================= #
    run_exp(config, source_models, target_models, attacks)


if __name__ == "__main__":
    main_parser = argparse.ArgumentParser(
        "Run multiple transfer attacks across set of source and target models.",
        parents=[get_args_parser()],
    )
    args = main_parser.parse_args()
    _config: dict[str, Any] = vars(args)
    main(_config)

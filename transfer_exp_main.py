"""Script for running transfer attack experiment."""

from __future__ import annotations

import timm

if not timm.__version__.startswith("0.6"):
    from timm.models import xcit

    xcit.XCiT = xcit.Xcit  # pylint: disable=no-member

# pylint: disable=wrong-import-order,wrong-import-position
import argparse
import copy
import logging
import sys
import time
from pathlib import Path
from typing import Any

from main import train_test_main
from src.attack.util import get_saved_adv_name
from src.models.util import ENSEMBLE_MODELS
from src.utils.argparser import get_args_parser, setup_output_dir
from src.utils.metric import safe_pickle
from src.utils.print_result import Aggregator, HiddenPrints, print_result


def _new_config(
    orig_config: dict[str, Any], model: str
) -> tuple[dict[str, Any], Path]:
    new_config = copy.deepcopy(orig_config)
    # "model" format is DATASET_ARCH_TRAIN_SEED
    # ARCH can be
    #   - single model: resnet18
    #   - ensemble: trs_resnet18+resnet50
    #   - ensemble with freeze: trs_freeze_resnet18+resnet50

    ensemble_type = None
    mod_model = model
    for ens in ENSEMBLE_MODELS:
        if f"_{ens}_" in model:
            mod_model = mod_model.replace(f"_{ens}_", "_")
            ensemble_type = ens
            break

    if ensemble_type is not None and "_freeze_" in model:
        # Handle freeze option
        mod_model = mod_model.replace("_freeze_", "_")
        ensemble_type = None

    new_config["ensemble_type"] = ensemble_type
    new_config["arch"] = mod_model.split("_")[1].split("+")
    if ensemble_type is None:
        new_config["arch"] = [new_config["arch"][0]]
    new_config["output_dir"] = f"results/{model}"
    new_config["evaluate"] = True
    output_path = setup_output_dir(new_config)
    return new_config, output_path


def main() -> None:
    """Main function."""
    source_models = config["src_models"]
    target_models = config["tgt_models"]
    attacks = config["transfer_attacks"]
    if not source_models:
        raise ValueError("No source models specified via --src-models.")
    if not target_models:
        raise ValueError("No target models specified via --tgt-models.")
    if not attacks:
        raise ValueError("No attacks specified via --transfer-attacks.")
    config["src_models"] = None
    config["tgt_models"] = None
    config["transfer_attacks"] = None
    run_exp(config, source_models, target_models, attacks)


def run_exp(
    config,  # pylint: disable=redefined-outer-name
    source_models,
    target_models,
    attacks,
) -> None:
    """Run transfer experiments for multiple sources, targets, and attacks.

    WARNING: metrics["idx_correct"] returned from train_test_main() should have
    a fixed length (= num test samples), but there is no guarantee that the
    order of the samples remained the same across runs. We fix random seed in
    all the scripts so the ordering should be correct when run in the same
    environment (same package version and machine), but beyond that is unclear.
    """
    results = {}

    for _, tgt in enumerate(target_models):
        aggregator = Aggregator()
        results[tgt] = {}
        tgt_config, tgt_path = _new_config(config, tgt)

        for src in source_models:
            results[tgt][src] = {}
            src_config, src_path = _new_config(config, src)

            for attack in attacks:
                print(f"{src} -> {tgt} using {attack} attack.")

                if (
                    (attack == "naa" and "robustbench-diffenderfer21" in src)
                    or (attack == "naa" and "imagenet_timm-vgg-11" in src)
                ):
                    print(f"[WARNING] Skipping {attack} on {src}")
                    continue

                start_time = time.time()
                saved_adv_path = src_path / get_saved_adv_name(
                    src_config, attack=attack, partition="test"
                )
                # Check if attack has been generated
                if not saved_adv_path.is_dir():
                    raise NotADirectoryError(
                        f"Given path {saved_adv_path} does not exist or is not "
                        "a directory."
                    )
                if not any(saved_adv_path.iterdir()):
                    raise FileNotFoundError(
                        f"No file found in {saved_adv_path}."
                    )
                # Get full attack name
                adv_name = saved_adv_path.name.replace("saved_", "")

                print("  Evaluating...")
                tgt_config["temperature"] = 1.0  # Reset eval temperature to 1
                tgt_config["test_load_paths"] = [str(saved_adv_path)]
                # Need to set train_load_paths because dataloader requires it
                tgt_config["train_load_paths"] = [str(saved_adv_path)]
                with HiddenPrints(not config["debug"]):
                    metrics = train_test_main(copy.deepcopy(tgt_config))
                # TODO(enhance): Collect metrics from last dataset assuming that
                # it is the adversarial one. Perhaps we can clean this up a bit.
                metrics = metrics["test"][-1]
                elapsed_time = (time.time() - start_time) / 60
                print(
                    f'  accuracy: {metrics["acc0"]:.2f}, time: '
                    f"{elapsed_time:.1f} min",
                    flush=True,
                )

                # Collect successes for best attack across sources and atk algo
                idx_success = ~metrics["idx_correct"]
                aggregator.update((src, attack), idx_success)
                results[tgt][src][adv_name] = idx_success

                # Dump all results after every run
                safe_pickle("results/all_results.pkl", obj=results, load=False)

                # Dump target-specific results
                tgt_result_path = tgt_path / "results.pkl"
                if tgt_result_path.exists():
                    # If results already exist, append to it
                    tgt_results = safe_pickle(tgt_result_path, load=True)
                    if src not in tgt_results:
                        tgt_results[src] = {}
                    tgt_results[src][adv_name] = idx_success
                else:
                    tgt_results = {src: {adv_name: idx_success}}
                safe_pickle(tgt_result_path, obj=tgt_results, load=False)

        print(f"Target: {tgt}")
        errs = print_result(attacks, source_models, aggregator)
        avg_acc, best_acc, overall_best_acc = [100 - err for err in errs]
        # metrics = safe_pickle(f"results/{tgt}/metrics.pkl", load=True)
        # The first metric should be from training
        # metrics = metrics[0]["test"][-1]
        # assert metrics["attack"] in ("none", "no_attack"), metrics
        # clean_acc = metrics["acc0"]
        clean_acc = tgt_results["test"]["none"]  # Gather new result with this line
        print(
            f"Accuracy (clean, avg, best, per-sample best): {clean_acc:.2f}, "
            f"{avg_acc:.2f}, {best_acc:.2f}, {overall_best_acc:.2f}"
        )

    safe_pickle("results/all_results.pkl", obj=results, load=False)
    print("Finished.")


if __name__ == "__main__":
    main_parser = argparse.ArgumentParser(
        "Run multiple transfer attacks across set of source and target models.",
        parents=[get_args_parser()],
    )
    main_parser.add_argument(
        "--tgt-models",
        default=None,
        nargs="+",
        type=str,
        help="List of target models used to evaluate against transfer attacks.",
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

    main()

"""List and parse common args."""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path
from typing import Any

from src.attack.util import get_saved_adv_name
from src.models.util import is_pretrained
from src.utils.loss import LOSS_DICT

logger = logging.getLogger(__name__)


def get_args_parser():
    """List and parse common args."""
    parser = argparse.ArgumentParser(
        description="Transfer attack defense", add_help=False
    )
    parser.add_argument("--data-dir", default="~/data/", type=str)
    parser.add_argument(
        "--arch",
        default=None,
        nargs="+",
        type=str,
        help=(
            "List of archiectures to use. If more than one is provided, an "
            "ensemble will be used."
        ),
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Load pretrained model on ImageNet-1k.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        type=str,
        help="Path to output dir.",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="Number of data loading workers per process",
    )
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--start-epoch", default=0, type=int)
    parser.add_argument(
        "--batch-size",
        default=128,
        type=int,
        help="Mini-batch size per device.",
    )
    parser.add_argument("--full-precision", action="store_true")
    parser.add_argument("--warmup-epochs", default=0, type=int)
    parser.add_argument("--warmup-iters", default=0, type=int)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--wd", default=1e-4, type=float)
    parser.add_argument("--optim", default="sgd", type=str)
    parser.add_argument("--betas", default=(0.9, 0.999), nargs=2, type=float)
    parser.add_argument("--eps", default=1e-8, type=float)
    parser.add_argument("--clip-grad-norm", default=None, type=float)
    parser.add_argument(
        "--lr-schedule",
        default="cosine",
        type=str,
        help="Learning rate schedule. Options: 'cosine' (default), 'step'.",
    )
    parser.add_argument(
        "--lr-steps",
        default=(40, 45),
        nargs="+",
        type=int,
        help=(
            "List of epochs to decay learning rate by a factor of 10 when "
            "'step' lr_schedule is used."
        ),
    )
    parser.add_argument(
        "--lr-step-size",
        default=0.1,
        type=float,
        help="Decay factor for step learning rate schedule (defaults to 0.1).",
    )
    parser.add_argument(
        "--print-freq", default=10, type=int, help="print frequency"
    )
    parser.add_argument(
        "--resume",
        default="",
        nargs="+",
        type=str,
        help="Path to checkpoint to load at beginning of training or testing.",
    )
    parser.add_argument(
        "--load-weight-only",
        action="store_true",
        help="Resume checkpoint by loading model weights only",
    )
    parser.add_argument("--evaluate", action="store_true", help="Evaluate only")
    parser.add_argument(
        "--world-size",
        default=1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--depth",
        default=20,
        type=int,
        help="Depth of the model. Used for DVERGE training.",
    )
    parser.add_argument(
        "--distill-layer",
        default=20,
        type=int,
        help=(
            "Which layer is used for distillation (DVERGE), only used when "
            "--distill-fixed-layer is True"
        ),
    )
    parser.add_argument(
        "--distill-fixed-layer",
        default=False,
        action="store_true",
        help="Whether fixing the layer for distillation (DVERGE)",
    )
    parser.add_argument(
        "--rank", default=0, type=int, help="node rank for distributed training"
    )
    parser.add_argument(
        "--dist-url",
        default="tcp://localhost:10001",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--no-distributed",
        action="store_true",
        help="Disable distributed mode.",
    )
    parser.add_argument("--dist-backend", default="nccl", type=str)
    parser.add_argument("--seed", default=0, type=int)
    # parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
    parser.add_argument("--wandb", action="store_true", help="Enable WandB")
    parser.add_argument(
        "--resume-if-exist",
        action="store_true",
        help=(
            "Override --resume option and resume from the current best "
            "checkpoint in the same dir if exists."
        ),
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode."
    )
    parser.add_argument(
        "--val-ratio",
        default=0.1,
        type=float,
        help="Ratio of training data to use as validation.",
    )
    parser.add_argument("--save-all-epochs", action="store_true")
    parser.add_argument(
        "--dataset", required=True, type=str, help="Name of dataset to use."
    )
    parser.add_argument(
        "--experiment",
        required=True,
        type=str,
        help="Type of experiment to run",
    )
    parser.add_argument(
        "--epsilon",
        default=8 / 255,
        type=float,
        help="Perturbation norm for attacks (default: 8/255)",
    )
    # Adversarial training
    parser.add_argument(
        "--train-attack",
        default="none",
        type=str,
        help="Use adversarial training (default: none = normal training)",
    )
    parser.add_argument(
        "--train-loss",
        type=str,
        default="ce",
        choices=list(LOSS_DICT.keys()),
        help="Loss function for training (default: ce)",
    )
    parser.add_argument(
        "--ensemble-type",
        default=None,
        type=str,
        choices=[None, "default", "trs", "dverge"],
        help=(
            "Type of ensemble to use (default: None = no ensemble). Options are"
            " default (average), trs, dverge."
        ),
    )
    parser.add_argument("--trs-lambda-a", default=40.0, type=float)
    parser.add_argument("--trs-lambda-b", default=2.5, type=float)
    parser.add_argument("--dverge-reg-coeff", default=1.0, type=float)
    parser.add_argument(
        "--trn-atk-steps",
        default=10,
        type=int,
        help="Number of attack iterations during adversarial training.",
    )
    parser.add_argument(
        "--eval-atk-steps",
        default=100,
        type=int,
        help="Number of attack iterations during evaluation.",
    )
    parser.add_argument(
        "--atk-norm",
        default="Linf",
        type=str,
        help="Lp-norm of adversarial perturbation (default: Linf).",
    )
    parser.add_argument(
        "--adv-beta",
        default=6.0,
        type=float,
        help="Beta parameter for TRADES or MAT (default: 6).",
    )
    parser.add_argument(
        "--eval-attack",
        default=[],
        nargs="+",
        type=str,
        help="Attacks to run during evaluation. Default is none (no attack).",
    )
    parser.add_argument(
        "--atk-opt",
        default="",
        type=str,
        help=(
            "Key-value pair for setting custom attack options (comma-separated)"
            ', e.g., "--atk-opt mpgd_mu=1.0".'
        ),
    )
    # Transfer related args
    parser.add_argument(
        "--save-adv-partition",
        default="test",
        type=str,
        help="Data partition to generate adversarial examples from.",
    )
    parser.add_argument(
        "--save-adv-path",
        default=None,
        type=str,
        help=(
            "Path to save generated adversarial examples. If not specified, "
            "will use the same path as output_dir."
        ),
    )
    parser.add_argument(
        "--worst-k-transfer-loss",
        default=None,
        type=int,
        help=(
            "Choose top-k worst losses for transfer-defense training (defaults "
            "to None: k = num_dataset_copies)."
        ),
    )
    parser.add_argument(
        "--num-dataset-copies",
        default=None,
        type=int,
        help=(
            "Number of adversarial examples to generate or load as a multiple "
            "of the original dataset size. Defaults to None (use all)."
        ),
    )
    parser.add_argument(
        "--ensemble-adv-training-one-model",
        default=False,
        type=bool,
        help=("Adversarial Training done for one model in ensemble."),
    )
    parser.add_argument(
        "--alpha-exp-smooth",
        default=None,
        type=float,
        help=(
            "Smoothing coefficient of the exponential moving average weights."
            "Must be between 0 and 1 (higher = weighing current losses more)."
        ),
    )
    parser.add_argument(
        "--include-clean-samples",
        action="store_true",
        help="If True, weigh clean loss with dynamically adjusted weights",
    )
    parser.add_argument(
        "--freeze-ensemble-but-first",
        action="store_true",
        help="If true, we train only the first model in the ensemble.",
    )
    parser.add_argument(
        "--exp-loss-track-metric",
        default="loss",
        choices=["loss", "acc"],
        type=str,
        help=(
            'Metric to track for exponential moving average. Options are "loss"'
            ' and "acc". Defaults to "loss".'
        ),
    )
    parser.add_argument(
        "--data-load-mode",
        default="orig",
        type=str,
        choices=["orig", "load", "mix", "stack"],
        help=(
            "Mode for dataloader. See options under get_data_loader() in "
            "src.dataloader.cifar_multi_datasets.py. Defaults to None (set "
            "automatically by main function).",
        ),
    )
    parser.add_argument(
        "--train-load-paths",
        default=None,
        nargs="+",
        type=str,
        help="Paths to saved samples to load as training set.",
    )
    parser.add_argument(
        "--val-load-paths",
        default=None,
        nargs="+",
        type=str,
        help="Paths to saved samples to load as validation set.",
    )
    parser.add_argument(
        "--test-load-paths",
        default=None,
        nargs="+",
        type=str,
        help="Paths to saved samples to load as test set.",
    )
    parser.add_argument(
        "--src-models",
        default=None,
        nargs="+",
        type=str,
        help="List of sources models used to generate transfer attacks.",
    )
    parser.add_argument(
        "--transfer-attacks",
        default=None,
        nargs="+",
        type=str,
        help="List of transfer attack algorithms to run.",
    )
    parser.add_argument(
        "--augments",
        default=["default"],
        nargs="+",
        type=str,
        help=(
            "List of data augmentations to apply to train set. Options are "
            "'default', 'pad_crop', 'cutmix', 'randaug'."
        ),
    )
    parser.add_argument(
        "--temperature",
        default=1.0,
        type=float,
        help="Logits temperature scaling (default: 1.0).",
    )
    parser.add_argument(
        "--dataset-weights",
        default=None,
        nargs="+",
        type=float,
        help="Loss weights for each dataset (defaults to None).",
    )
    parser.add_argument(
        "--dataset-dropout",
        default=None,
        nargs="+",
        type=float,
        help="Dropout rate for each dataset (defaults to None).",
    )
    parser.add_argument(
        "--no-orig-data",
        action="store_true",
        help="Do not load the original data and only use the saved samples.",
    )
    parser.add_argument(
        "--huyvnphan-weight-path",
        default="./weights/state_dicts/",
        type=str,
        help="Path to pre-trained weights from huyvnphan/PyTorch_CIFAR10.",
    )
    parser.add_argument(
        "--suffix",
        default="",
        type=str,
        help="Custom suffix to append to output_dir name.",
    )
    parser.add_argument(
        "--selected-file-paths",
        default="imagenet_val_samples.txt",
        type=str,
        help=(
            "Path to txt file contains file paths to load (used with validation"
            " of ImageNet only). Defaults to 'imagenet_val_samples.txt'."
        ),
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle data in save_adv_main.py",
    )
    return parser


def naming_hash(obj: str, length: int = 4) -> str:
    """Generate a hash from a string.

    Args:
        obj: Object to hash. Must be string.
        length: Length of hash to return. Defaults to 4.

    Returns:
        Hash string.
    """
    return hashlib.sha512(obj.encode("utf-8")).hexdigest()[:length]


def get_model_name(config: dict[str, Any]) -> str:
    """Generate a name for the model based on the config.

    Args:
        config: Config dictionary.

    Returns:
        Model name.
    """
    # Example model name: cifar10_resnet18_none_mat_seed0_src-f59e_atk-b06e_path-dc6b_lr0.1_wd0.0005_sgd_beta0.5_dw1.0-0.1-0.9_do0.0-0.0-0.0
    tokens = [config["dataset"]]

    # Get ensemble type if applicable and concatenate archs with "+"
    is_ensemble = False
    if len(config["arch"]) > 1:
        tokens.append(config["ensemble_type"])
        if config["freeze_ensemble_but_first"]:
            tokens.append("freeze")
        is_ensemble = True

        arch_names = []
        for arch in config["arch"]:
            if is_pretrained(arch):
                # Shorten pretrained model names
                arch_names.append(naming_hash(arch, length=2))
            else:
                arch_names.append(arch)
        tokens.append("+".join(arch_names))
    else:
        tokens.append(config["arch"][0])

    if is_pretrained(config["arch"][0]):
        # TODO(feature): This would fail for ensembles or fine-tuning
        # Not sure if this is necessary anymore
        pass
    else:
        tokens.extend(
            [
                config["train_attack"],
                config["train_loss"],
            ]
        )
    tokens.append(f"seed{config['seed']}")

    # Additional tokens for transfer def training
    splits = ["train", "val", "test"]
    load_path_is_given = any(config[f"{s}_load_paths"] for s in splits)
    if config["src_models"]:
        # If we are doing transfer def training, add source models hash
        tokens.extend(
            [
                "src-" + naming_hash(str(sorted(config["src_models"]))),
                "atk-" + naming_hash(str(sorted(config["transfer_attacks"]))),
            ]
        )
    elif load_path_is_given:
        # If data are specified by load paths, add one of their hashes
        for split in splits:
            if config[f"{split}_load_paths"]:
                phash = naming_hash(str(sorted(config[f"{split}_load_paths"])))
                tokens.append(f"path-{phash}")
                break

    # Add other training args
    # Ensembles will always involve pretrained models
    if not is_pretrained(config["arch"][0]):
        tokens.extend(
            [
                f"bs{config['batch_size']}",
                f"lr{config['lr']}",
                f"wd{config['wd']}",
                config["optim"],
                f"{config['epochs']}ep",
            ]
        )
        if config["train_attack"] != "none":
            if config["atk_norm"] == "Linf":
                eps = round(config["epsilon"] * 255)
            else:
                eps = config["epsilon"]
            tokens.append(f"eps{eps}")
        if config["train_loss"] in ("mat", "trades", "trades-lse"):
            tokens.append(f"beta{config['adv_beta']}")
        if config["train_loss"] == "weighted-exp":
            tokens.append(f"alpha{config['alpha_exp_smooth']}")
            tokens.append(config["exp_loss_track_metric"])
        # if config["dataset_weights"]:
        #     tokens.append(
        #         "dw" + "-".join([str(dw) for dw in config["dataset_weights"]])
        #     )
        # if config["dataset_dropout"]:
        #     tokens.append(
        #         "do" + "-".join([str(do) for do in config["dataset_dropout"]])
        #     )
        if config["data_load_mode"] is not None:
            tokens.append(config["data_load_mode"])
        if config["worst_k_transfer_loss"] is not None:
            tokens.append(f"top{config['worst_k_transfer_loss']}")

        if is_ensemble:
            if config["ensemble_type"] == "trs":
                tokens.append(f"la{config['trs_lambda_a']}")
                tokens.append(f"lb{config['trs_lambda_b']}")
            elif config["ensemble_type"] == "dverge":
                tokens.append(f"c{config['dverge_reg_coeff']}")

    num_copies = config["num_dataset_copies"]
    if num_copies is not None:
        tokens.append(f"copy{num_copies}")
    if len(config["augments"]) > 1 or config["augments"][0] != "default":
        tokens.extend(config["augments"])
    if config["suffix"]:
        tokens.append(config["suffix"])

    replacement_value = "None"
    tokens = [item or replacement_value for item in tokens]
    model_name = "_".join(tokens)
    return model_name


def setup_output_dir(config: dict[str, Any]) -> Path:
    """Set up distributed mode and create output dir."""
    if config["output_dir"] is None:
        logger.info("Creating output dir automatically...")
        # Automatically create output dir if not specified. This should be used
        # during training only.
        base_path = Path("./results/")
        model_name = get_model_name(config)
        config["output_dir"] = str(base_path / model_name)
        logger.info("output_dir is set to %s.", config["output_dir"])

    output_path: str = config["output_dir"]
    output_dir: Path = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def setup_load_paths(config: dict[str, Any]):
    """Update train/val/test load paths in config."""
    splits = ["train", "val", "test"]
    # Verify that we are not using both src_models and load_paths
    load_path_is_given = any(config[f"{s}_load_paths"] for s in splits)
    if load_path_is_given and config["src_models"]:
        raise ValueError(
            "Cannot specify both src_models and train/val/test_load_paths! "
            "src_models and transfer_attacks are used to automatically set up "
            "the load paths."
        )
    if load_path_is_given:
        logging.info("Using specified load_paths.")
        return config
    if not config["src_models"]:
        return config

    base_path = Path("./results/")
    load_paths = {split: [] for split in splits}

    for src_model in config["src_models"]:
        found = False
        for attack in config["transfer_attacks"]:
            for split in splits:
                name = get_saved_adv_name(
                    config, attack=attack, partition=split
                )
                load_path = base_path / src_model / name
                if load_path.exists():
                    load_paths[split].append(str(load_path))
                    found = True
        if not found:
            raise FileNotFoundError(
                f"No load found for src_model={src_model}, transfer_attacks="
                f'{config["transfer_attacks"]}'
            )

    for split, load_paths in load_paths.items():
        if load_paths:
            logging.info("Setting up new %s_load_paths: %s", split, load_paths)
            config[f"{split}_load_paths"] = load_paths

    return config

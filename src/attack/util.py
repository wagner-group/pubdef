"""Utility functions for setting up attack."""

from __future__ import annotations

import ast
import logging
from typing import Any, Literal

from torch import nn

from src.attack.admix import AdmixAttack
from src.attack.auto import AutoAttack
from src.attack.auto_pgd_ce import AutoPGDCEAttack
from src.attack.auto_pgd_dlr import AutoPGDDLRAttack
from src.attack.base import AttackModule
from src.attack.di import DIAttack
from src.attack.lse import LseAttack
from src.attack.mpgd import MomentumPGDAttack
from src.attack.naa import NAAttack
from src.attack.ni_admix_ti_dim import AdmixCombinedAttack
from src.attack.ni_si_ti_dim import CombinedAttack
from src.attack.none import NoAttack
from src.attack.pgd import PGDAttack
from src.attack.pregradient import PregradientPGDAttack
from src.attack.ti import TIAttack
from src.attack.trades import TRADESAttack
from src.attack.trades_lse import TradesLseAttack
from src.dataloader.cutmix import SoftCrossEntropyLoss

logger = logging.getLogger(__name__)


def _get_loss(option: str) -> nn.Module:
    """Get loss module from config."""
    loss: nn.Module = {
        "ce": nn.CrossEntropyLoss(reduction="none"),
        "soft_ce": SoftCrossEntropyLoss(reduction="none"),
    }[option]
    return loss.cuda()


def setup_default_attacks(
    config: dict[str, str | int | float],
    model: nn.Module,
    num_classes: int,
    **kwargs,
) -> tuple[list[tuple[str, AttackModule]], AttackModule, AttackModule]:
    """Setup default attacks for testing, training, and validation.

    Args:
        config: Overall config.
        model: Target model.
        num_classes: Number of classes.
        kwargs: Custom attack parameters passed to setup_attack().

    Returns:
        Tuple of eval_attacks, train_attack, and val_attack. eval_attacks is a
        list of tuples of attack name and attack module. train_attack and
        val_attack are just attack modules.
    """
    attack_config = {}
    if config["atk_opt"]:
        for key_value in config["atk_opt"].split(","):
            key, value = key_value.split("=")
            attack_config[key] = ast.literal_eval(value)
    attack_config["dataset"] = config["dataset"]
    attack_config["arch"] = config["arch"]

    eval_attacks: list[tuple[str, AttackModule]] = setup_attack(
        mode="test",
        attack_names=config["eval_attack"],
        model=model,
        num_classes=num_classes,
        norm=config["atk_norm"],
        epsilon=config["epsilon"],
        num_steps=config["eval_atk_steps"],
        **attack_config,
        **kwargs,
    )
    # First attack is no_attack so we take the last (second) one
    train_attack: AttackModule = setup_attack(
        mode="train",
        attack_names=[config["train_attack"]],
        model=model,
        norm=config["atk_norm"],
        epsilon=config["epsilon"],
        num_steps=config["trn_atk_steps"],
        **attack_config,
        **kwargs,
    )[-1][1]
    val_attack: AttackModule = setup_attack(
        mode="val",
        attack_names=["pgd"],
        model=model,
        norm=config["atk_norm"],
        epsilon=config["epsilon"],
        num_steps=config["eval_atk_steps"],
        **attack_config,
        **kwargs,
    )[-1][1]
    return eval_attacks, train_attack, val_attack


def setup_attack(
    mode: Literal["train", "test", "val"],
    attack_names: list[str] | None = None,
    model: nn.Module | None = None,
    num_classes: int = 10,
    norm: str = "Linf",
    epsilon: float = 8 / 255,
    num_steps: int = 100,
    num_restarts: int | None = None,
    **kwargs,
) -> list[tuple[str, AttackModule]]:
    """Setup attacker for evaluation.

    Args:
        mode: One of ("train", "test", "val").
        attack_names: List of attack names. Defaults to None.
        model: Model to attack. Defaults to None.
        num_classes: Number of classes. Defaults to 10.
        norm: Choice of perturbation norm ("Linf", "L2"). Defaults to "Linf".
        epsilon: Perturbation norm. Defaults to 8/255.
        num_steps: Number of attack steps. Defaults to 100.
        atk_opt: Additional attack params. Comma-separated pairs of "KEY=VAL".
            Defaults to "".

    Returns:
        A list of tuples of attack name and attack module.
    """
    # If attack_names is None, use NoAttack
    attack_names = attack_names or []
    if isinstance(attack_names, str):
        attack_names = [attack_names]

    # Populate attack config
    attack_config: dict[str, str | float | int] = {"pgd_steps": num_steps}
    if mode == "test":
        attack_config["pgd_step_size"] = max(
            0.001, epsilon / 4 / (num_steps / 10)
        )
        attack_config["num_restarts"] = num_restarts or 3
    elif mode == "train":
        attack_config["pgd_step_size"] = epsilon / num_steps * 1.25
        attack_config["num_restarts"] = num_restarts or 1
        assert (
            len(attack_names) == 1
        ), f"Only 1 attack allowed in train mode, but {attack_names} is given!"
    elif mode == "val":
        attack_config["pgd_step_size"] = 0.002
        attack_config["num_restarts"] = num_restarts or 1
        assert (
            len(attack_names) == 1
        ), f"Only 1 attack allowed in val mode, but {attack_names} is given!"
    else:
        raise ValueError(
            f"Unknown mode: {mode}! mode must be in ('train', 'test', 'val')."
        )
    logger.info("Attack config in %s mode: %s", mode, attack_config)

    common_init = {
        "none": NoAttack,
        "trades": TRADESAttack,
        "mat": PGDAttack,
        "pgd": PGDAttack,
        "pregradient": PregradientPGDAttack,
        "naa": NAAttack,
        "admix": AdmixAttack,
        "di": DIAttack,
        "ti": TIAttack,
        "iaa": PGDAttack,
        "mpgd": MomentumPGDAttack,
        "ni-si-ti-dim": CombinedAttack,
        "ni-admix-ti-dim": AdmixCombinedAttack,
        "auto": AutoAttack,
        "auto-pgd-dlr": AutoPGDDLRAttack,
        "auto-pgd-ce": AutoPGDCEAttack,
        "lse": LseAttack,
        "trades-lse": TradesLseAttack,
    }

    no_attack: AttackModule = NoAttack(None, None, None, norm, epsilon)
    attack_list: list[tuple[str, AttackModule]] = [("none", no_attack)]
    if not attack_names:
        return attack_list

    for atk in attack_names:
        if atk in common_init:
            logger.info("Setting up %s attack", atk)
            attack = common_init[atk](
                attack_config,
                model,
                _get_loss("soft_ce"),
                norm,
                epsilon,
                num_classes=num_classes,
                **kwargs,
            )
        elif atk == "longer-pgd":
            num_steps = 300
            attack_config = {
                "pgd_steps": num_steps,
                "pgd_step_size": max(0.001, epsilon / 4 / (num_steps / 10)),
                "num_restarts": 2,
            }
            attack = PGDAttack(
                attack_config, model, _get_loss("soft_ce"), norm, epsilon
            )
        attack_list.append((atk, attack))

    logger.info(
        "Set up %s attack(s): %d in total.",
        [atk[0] for atk in attack_list],
        len(attack_list),
    )
    return attack_list


def get_saved_adv_name(
    config: dict[str, Any], attack: str = "pgd", partition: str = "test"
) -> str:
    """Standardized way to get saved adversarial example name."""
    epsilon = config["epsilon"]
    if config["atk_norm"] == "Linf":
        epsilon *= 255
    tokens = [
        "saved",
        attack,
        partition,
        f"temp{config['temperature']}",
    ]

    # Backward compatibility with old saved adv names
    if config["atk_norm"] != "Linf" or abs(epsilon - 8) > 1e-6:
        tokens.append(f"eps{epsilon:.1f}")
    if config["eval_atk_steps"] != 100:
        tokens.append(f"{config['eval_atk_steps']}step")

    if config["arch"] == "clip":
        tokens.append(
            f"clip-{config['clip_model']}-{config['clip_prompt_template']}"
        )
    augments = config["augments"]
    if augments is not None and (len(augments) > 1 or augments[0] != "default"):
        # Only include if augments is given and not ["default"]
        augments = sorted(set(augments))
        tokens.extend(augments)
    name = "_".join(tokens)
    return name

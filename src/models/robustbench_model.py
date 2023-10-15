"""Wrapper for RobustBench model.

Models from https://github.com/RobustBench/robustbench.
"""

from __future__ import annotations

import logging

from robustbench.utils import load_model
from torch import nn

from src.utils.types import BatchImages, BatchLogits

logger = logging.getLogger(__name__)


class RobustBenchModel(nn.Module):
    """Wrapper for RobustBench model."""

    def __init__(self, dataset: str, arch: str, **kwargs):
        """Initialize RobustBench model."""
        super().__init__()
        _ = kwargs  # Unused
        model_name, threat_model = {
            "cifar10": {
                "wang23-wrn-70-16": ("Wang2023Better_WRN-70-16", "Linf"),  # Best wrn-70-16
                "rebuffi21-wrn-70-16": ("Rebuffi2021Fixing_70_16_cutmix_extra", "Linf"),
                "gowal21-wrn-70-16": ("Gowal2021Improving_70_16_ddpm_100m", "Linf"),
                "huang22-wrn-a4": ("Huang2022Revisiting_WRN-A4", "Linf"),  # Only wrn-a4
                "xu23-wrn-28-10": ("Xu2023Exploring_WRN-28-10", "Linf"),  # 2nd best wrn-28-10 (best is from wang23)
                "rade21-wrn-34-10": ("Rade2021Helper_extra", "Linf"),  # Best wrn-34-10
                "sehwag21-rn-152": ("Sehwag2021Proxy_ResNest152", "Linf"),  # Best rn-152
                "dai21-wrn-28-10": ("Dai2021Parameterizing", "Linf"),  # Different activation function (PSSiLU)
                "carmon19-wrn-28-10": ("Carmon2019Unlabeled", "Linf"),  # Unlabeled data
                "gowal21-rn-18": ("Gowal2021Improving_R18_ddpm_100m", "Linf"),  # Best PreActResNet-18
                "debenedetti22-xcit-l12": ("Debenedetti2022Light_XCiT-L12", "Linf"),  # Best XCiT-L
                "sehwag20-wrn-28-10": ("Sehwag2020Hydra", "Linf"),  # Pruning
                "andriushchenko20-rn-18": ("Andriushchenko2020Understanding", "Linf"),  # Fast adversarial training
                "engstrom19-rn-50": ("Engstrom2019Robustness", "Linf"),  # Baseline ResNet-50
                "andriushchenko20-prn-18": ("Andriushchenko2020Understanding", "Linf"),  # Baseline PreActResNet-18
                "addepalli21-wrn-34-10": ("Addepalli2021Towards_WRN34", "Linf"),  # Baseline WideResNet-34-10
                # Common corruption
                "diffenderfer21-wrn-18-2": ("Diffenderfer2021Winning_LRR_CARD_Deck", "corruptions"),
                "kireev21-rn-18": ("Kireev2021Effectiveness_RLATAugMix", "corruptions"),
                "hendrycks20-resnext-29-32x4d": ("Hendrycks2020AugMix_ResNeXt", "corruptions"),
                "modas21-rn-18": ("Modas2021PRIMEResNet18", "corruptions"),
                # L2 adversarially trained models
                "wang23-wrn-70-16-l2": ("Wang2023Better_WRN-70-16", "L2"),  # Best wrn-70-16
                "rebuffi21-wrn-70-16-l2": ("Rebuffi2021Fixing_70_16_cutmix_extra", "L2"),
                "gowal20-wrn-70-16-l2": ("Gowal2020Uncovering_extra", "L2"),
                "augustin20-wrn-34-10-l2": ("Augustin2020Adversarial_34_10_extra", "L2"),  # Best wrn-34-10
                "sehwag21-wrn-34-10-l2": ("Sehwag2021Proxy", "L2"),
                "wu20-wrn-34-10-l2": ("Wu2020Adversarial", "L2"),
                "rony19-wrn-28-10-l2": ("Rony2019Decoupling", "L2"),
            },
            "cifar100": {
                # Linf adversarially trained models
                "wang23-wrn-70-16": ("Wang2023Better_WRN-70-16", "Linf"),  # Best wrn-70-16
                "cui23-wrn-28-10": ("Cui2023Decoupled_WRN-28-10", "Linf"),  # Best wrn-28-10
                "bai23-mix": ("Bai2023Improving_edm", "Linf"),  # ResNet-152 + WideResNet-70-16 + mixing network
                "debenedetti22-xcit": ("Debenedetti2022Light_XCiT-L12", "Linf"),  # XCiT-L12
                "jia22-wrn-34-20": ("Jia2022LAS-AT_34_20", "Linf"),  # Best wrn-34-20
                "rade21-rn-18": ("Rade2021Helper_R18_ddpm", "Linf"),  # Best rn-18
                # Common corruption
                "diffenderfer21-wrn-18-2": ("Diffenderfer2021Winning_LRR_CARD_Deck", "corruptions"),  # Best overall
                "modas21-rn-18": ("Modas2021PRIMEResNet18", "corruptions"),  # Best rn-18
                "hendrycks20-resnext-29-32x4d": ("Hendrycks2020AugMix_ResNeXt", "corruptions"),  # Best resnext-29-32x4d
                "addepalli22-wrn-34-10": ("Addepalli2022Efficient_WRN_34_10", "corruptions"),  # Best wrn-34-10
                "gowal20-wrn-70-16": ("Gowal2020Uncovering_extra_Linf", "corruptions"),  # Best wrn-70-16
                "diffenderfer21-bin-wrn-18-2": ("Diffenderfer2021Winning_Binary", "corruptions"),  # Best model that is different from Diffenderfer2021Winning_LRR_CARD_Deck
            },
            "imagenet": {
                # Linf adversarially trained models
                "liu23-swin-l": ("Liu2023Comprehensive_Swin-L", "Linf"),
                "liu23-swin-b": ("Liu2023Comprehensive_Swin-B", "Linf"),
                "singh23-convnext-t": ("Singh2023Revisiting_ConvNeXt-T-ConvStem", "Linf"),
                "singh23-vit-s": ("Singh2023Revisiting_ViT-S-ConvStem", "Linf"),
                "debenedetti22-xcit-s12": ("Debenedetti2022Light_XCiT-S12", "Linf"),
                "salman20-wrn-50-2": ("Salman2020Do_50_2", "Linf"),
                "salman20-rn-50": ("Salman2020Do_R50", "Linf"),
                # Common corruption
                "tian22-deit-b": ("Tian2022Deeper_DeiT-B", "corruptions"),
                "tian22-deit-s": ("Tian2022Deeper_DeiT-S", "corruptions"),
                "erichson22-rn-50": ("Erichson2022NoisyMix_new", "corruptions"),
                "hendrycks20many-rn-50": ("Hendrycks2020Many", "corruptions"),
                "hendrycks20augmix-rn-50": ("Hendrycks2020AugMix", "corruptions"),
                "geirhos18-rn-50": ("Geirhos2018_SIN_IN", "corruptions"),
            },
        }[dataset][arch.replace("robustbench-", "")]
        logger.info("Loading RobustBench model: %s", model_name)
        # Load a model from the model zoo
        self._wrapped_model = load_model(
            model_name=model_name, dataset=dataset, threat_model=threat_model
        )
        logger.debug(self._wrapped_model)

    def forward(self, inputs: BatchImages) -> BatchLogits:
        """Forward pass."""
        return self._wrapped_model(inputs)

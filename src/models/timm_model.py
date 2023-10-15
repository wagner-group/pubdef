"""Warpper for HuggingFace model."""

from __future__ import annotations

import logging

import timm
from torch import nn

from src.utils.types import BatchImages, BatchLogits

logger = logging.getLogger(__name__)


class TimmModel(nn.Module):
    """Wrapper for HuggingFace model."""

    def __init__(self, dataset: str, arch: str, **kwargs):
        """Initialize HuggingFace model."""
        super().__init__()
        _ = kwargs  # Unused
        model_name = {
            "imagenet": {
                "efficientnet-b3": "hf-hub:timm/efficientnet_b3.ra2_in1k",
                "mnasnet-100": "hf-hub:timm/mnasnet_100.rmsp_in1k",
                "inception-v3": "inception_v3",
                "mixer-b16": "hf-hub:timm/mixer_b16_224.goog_in21k_ft_in1k",
                "rexnet-100": "hf-hub:timm/rexnet_100.nav_in1k",
                "hrnet-w18": "hf-hub:timm/hrnet_w18.ms_aug_in1k",
                "vgg-11": "hf-hub:timm/vgg11.tv_in1k",
                "mobilenet-v3-l": "hf-hub:timm/mobilenetv3_large_100.ra_in1k",
                "convmixer-768-32": "hf-hub:timm/convmixer_768_32.in1k",
                "inception-rn-v2": "hf-hub:timm/inception_resnet_v2.tf_in1k",
                # "ghostnetv2-160": "hf-hub:timm/ghostnetv2_160.in1k",
                # "fastvit-t12": "hf-hub:timm/fastvit_t12.apple_dist_in1k",
                "swin-l": "hf-hub:timm/swin_large_patch4_window7_224.ms_in22k_ft_in1k",
            },
        }[dataset][arch.replace("timm-", "")]
        logger.info("Loading timm model: %s", model_name)
        self._wrapped_model = timm.create_model(model_name, pretrained=True)

    def forward(self, inputs: BatchImages) -> BatchLogits:
        """Forward pass."""
        return self._wrapped_model(inputs)

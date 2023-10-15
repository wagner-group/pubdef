"""Warpper for HuggingFace model."""

from __future__ import annotations

import logging

import torch
import torchvision.transforms.functional as F
from torch import nn
from torchvision.transforms import InterpolationMode
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

from src.utils.types import BatchImages, BatchLogits

logger = logging.getLogger(__name__)


class HuggingFaceModel(nn.Module):
    """Wrapper for HuggingFace model."""

    def __init__(self, dataset: str, arch: str, **kwargs):
        """Initialize HuggingFace model."""
        super().__init__()
        _ = kwargs  # Unused
        model_name = {
            "cifar10": {
                "vit-base": "aaraki/vit-base-patch16-224-in21k-finetuned-cifar10",
                "beit": "jadohu/BEiT-finetuned",
                "convnext": "ahsanjavid/conVneXt-tiny-finetuned-cifar10",
                "swint": "microsoft/swin-tiny-patch4-window7-224",
            },
            "cifar100": {
                # "resnet50": "edadaltocg/resnet50_cifar100",
                # "vit-base": "edumunozsala/vit_base-224-in21k-ft-cifar100",
                "vit-base": "Ahmed9275/Vit-Cifar100",
                "swin-tiny": "MazenAmria/swin-tiny-finetuned-cifar100",
                # "densenet121": "edadaltocg/densenet121_cifar100",
            },
            "imagenet": {
                "rn-50": "microsoft/resnet-50",
                "vit-b": "google/vit-base-patch16-224",
                # "mobilenet-v1": "google/mobilenet_v1_0.75_192",
                # "convnext-v2": "facebook/convnextv2-tiny-1k-224",
                "swin-t": "microsoft/swin-tiny-patch4-window7-224",
                # "dino-vit-s": "facebook/dino-vits16",
                # "mobilevit-s": "apple/mobilevit-small",
                # "focalnet-s": "microsoft/focalnet-small",
                # "mobilevitv2-1.0": "apple/mobilevitv2-1.0-imagenet1k-256",
                "convnext-t": "facebook/convnext-tiny-224",
            },
        }[dataset][arch.replace("huggingface-", "")]
        logger.info("Loading HuggingFace model: %s", model_name)

        # Preprocessing
        extractor = AutoFeatureExtractor.from_pretrained(model_name)
        logger.debug(extractor)
        if "shortest_edge" in extractor.size:
            final_size = extractor.size["shortest_edge"]
        else:
            final_size = (extractor.size["height"], extractor.size["width"])
        # Determine crop size
        crop_pct = 1
        self._crop_size = None
        if hasattr(extractor, "crop_pct"):
            crop_pct = extractor.crop_pct
            self._crop_size = final_size
        # Determine resize size
        if "shortest_edge" in extractor.size:
            self._resize_size = round(final_size / crop_pct)
        else:
            self._resize_size = tuple(round(s / crop_pct) for s in final_size)
        self._interp_mode = {
            0: InterpolationMode.NEAREST,
            1: InterpolationMode.LANCZOS,
            2: InterpolationMode.BILINEAR,
            3: InterpolationMode.BICUBIC,
            4: InterpolationMode.BOX,
            5: InterpolationMode.HAMMING,
        }[extractor.resample]
        self.register_buffer(
            "_mean", torch.tensor(extractor.image_mean)[None, :, None, None]
        )
        self.register_buffer(
            "_std", torch.tensor(extractor.image_std)[None, :, None, None]
        )

        self._wrapped_model = AutoModelForImageClassification.from_pretrained(
            model_name
        )
        logger.debug(self._wrapped_model)

    def forward(self, inputs: BatchImages) -> BatchLogits:
        """Forward pass."""
        inputs = inputs.sub(self._mean).div(self._std)
        inputs = F.resize(
            inputs,
            self._resize_size,
            interpolation=self._interp_mode,
            antialias=True,
        )
        if self._crop_size is not None:
            inputs = F.center_crop(inputs, self._crop_size)
        logits = self._wrapped_model(pixel_values=inputs).logits
        return logits

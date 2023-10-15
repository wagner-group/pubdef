"""Utility functions for building models."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Tuple

import torch
from torch import nn
from torch.cuda import amp

from src.dataloader import cifar, imagenet
from src.models import (
    chenyaofo_util,
    cifar_clip,
    cifar_convmixer,
    cifar_densenet,
    cifar_dla,
    cifar_efficientnet,
    cifar_googlenet,
    cifar_inception,
    cifar_mobilenetv2,
    cifar_resnet,
    cifar_resnext,
    cifar_senet,
    cifar_simplevit,
    cifar_vgg,
    cifar_wideresnet,
    common,
    huggingface_model,
    imagenet_resnet,
    robustbench_model,
    timm_model,
)
from src.models.ensemble import dverge, ensemble, trs

_NormParams = Dict[str, Tuple[float, float, float]]
logger = logging.getLogger(__name__)

PRETRAINED_SRCS = (
    "huggingface",
    "huyvnphan",
    "chenyaofo",
    "robustbench",
    "clip",
    "timm",
)
HUYVNPHAN_CIFAR10_MODELS = {
    "densenet121": cifar_densenet.densenet121,
    "googlenet": cifar_googlenet.googlenet,
    "inception-v3": cifar_inception.inception_v3,
    "vgg11-bn": cifar_vgg.vgg11_bn,
    "vgg13-bn": cifar_vgg.vgg13_bn,
    "vgg16-bn": cifar_vgg.vgg16_bn,
    "vgg19-bn": cifar_vgg.vgg19_bn,
}
ENSEMBLE_MODELS = {
    "default": ensemble.Ensemble,
    "trs": trs.TRSEnsemble,
    "dverge": dverge.DVGEnsemble,
}


def is_pretrained(arch: str) -> bool:
    """Check if the model is pretrained."""
    if not isinstance(arch, str):
        raise ValueError(f"arch must be a string, but it is {arch}!")
    return any(name in arch for name in PRETRAINED_SRCS)


def _wrap_distributed(model, distributed, cuda_device):
    model = model.cuda()
    if distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[cuda_device]
        )
    else:
        model = torch.nn.parallel.DataParallel(model)
    return model


def _build_one_model(
    config: dict[str, Any],
    arch: str | None = "resnet50",
) -> nn.Module:
    """Build one model based on config."""
    arch: str = config["arch"] if arch is None else arch
    dataset: str = config["dataset"]
    # Get dataset-specific params
    dataset_cls = {
        "cifar10": cifar.CIFAR10Dataset,
        "cifar100": cifar.CIFAR100Dataset,
        "imagenet": imagenet.ImageNetDataset,
    }[dataset]
    num_classes: int = dataset_cls.num_classes

    # Define normalizing params
    if "clip" in arch:
        normalize_params = {
            "mean": [0.48145466, 0.4578275, 0.40821073],
            "std": [0.26862954, 0.26130258, 0.27577711],
        }
    elif "huyvnphan" in arch:
        normalize_params = {
            "mean": [0.4914, 0.4822, 0.4465],
            "std": [0.2471, 0.2435, 0.2616],
        }
    elif any(name in arch for name in ("huggingface", "robustbench")):
        # Normalization is done in HuggingFace model
        normalize_params = {
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
        }
    else:
        normalize_params: _NormParams = dataset_cls.normalize_params

    # IAA attack modifies source model
    iaa_params: dict[str, Any] = {}
    if "iaa" in config["eval_attack"]:
        iaa_params = {"decays": [1.0, 0.85, 0.65, 0.15], "beta_value": 25.0}
        if "cifar" not in config["dataset"]:
            raise NotImplementedError(
                "IAA is only implemented for CIFAR models!"
            )

    logger.info("=> Building a classifier...")
    if "chenyaofo-vgg" in arch and "cifar" in dataset:
        dst = chenyaofo_util.download_chenyaofo_weight(arch, dataset)
        model_fn = HUYVNPHAN_CIFAR10_MODELS[arch.replace("chenyaofo-", "")]
        model = model_fn(
            pretrained=True,
            num_classes=num_classes,
            weight_path=dst,
            dim=512,
            **iaa_params,
        )
    elif "chenyaofo" in arch:
        # Load public pre-trained models from
        # https://github.com/chenyaofo/pytorch-cifar-models
        model = torch.hub.load(
            "chenyaofo/pytorch-cifar-models",
            f"{dataset}_{arch.replace('chenyaofo-', '').replace('-', '_')}",
            pretrained=True,
        )
    elif "huyvnphan" in arch:
        # Load public pre-trained models from
        # https://github.com/huyvnphan/PyTorch_CIFAR10
        model_fn = HUYVNPHAN_CIFAR10_MODELS[arch.replace("huyvnphan-", "")]
        model = model_fn(
            pretrained=True,
            num_classes=num_classes,
            weight_path=config["huyvnphan_weight_path"],
            **iaa_params,
        )
    elif "huggingface" in arch:
        # Load public pre-trained models from HuggingFace
        model = huggingface_model.HuggingFaceModel(
            dataset=dataset, arch=arch, **iaa_params
        )
    elif "timm" in arch:
        # Load public pre-trained models from HuggingFace
        model = timm_model.TimmModel(dataset=dataset, arch=arch, **iaa_params)
    elif "robustbench" in arch:
        # Load public pre-trained models from RobustBench
        model = robustbench_model.RobustBenchModel(
            dataset=dataset, arch=arch, **iaa_params
        )
    elif "clip" in arch:
        _, clip_model, prompt_template = arch.split("-")
        model = cifar_clip.CLIPClassifier(
            num_classes=num_classes,
            clip_model=clip_model,
            prompt_template=prompt_template,
            **iaa_params,
        )
    else:
        _dataset = "cifar" if "cifar" in dataset else dataset
        # create an instance of ensemble? not too sure about this tho
        model_fn = {
            "cifar": {
                "resnet18": cifar_resnet.ResNet18,
                "resnet34": cifar_resnet.ResNet34,
                "resnet50": cifar_resnet.ResNet50,
                "resnet101": cifar_resnet.ResNet101,
                "resnet152": cifar_resnet.ResNet152,
                "efficientnetb0": cifar_efficientnet.EfficientNetB0,
                "mobilenetv2": cifar_mobilenetv2.MobileNetV2,
                "wideresnet28-10": cifar_wideresnet.wideresnet28_10,
                "wideresnet34-10": cifar_wideresnet.wideresnet34_10,
                "wideresnet34-20": cifar_wideresnet.wideresnet34_20,
                "wideresnet70-16": cifar_wideresnet.wideresnet70_16,
                "dla": cifar_dla.DLA,
                "densenet121": cifar_densenet.densenet121,
                "resnext29-2x64d": cifar_resnext.ResNeXt29_2x64d,
                "resnext29-4x64d": cifar_resnext.ResNeXt29_4x64d,
                "resnext29-8x64d": cifar_resnext.ResNeXt29_8x64d,
                "resnext29-32x4d": cifar_resnext.ResNeXt29_32x4d,
                "inception-v3": cifar_inception.inception_v3,
                "senet18": cifar_senet.SENet18,
                "simplevit": cifar_simplevit.simple_vit,
                "convmixer": cifar_convmixer.build_conv_mixer,
                **HUYVNPHAN_CIFAR10_MODELS,
            },
            "imagenet": {
                "resnet50": imagenet_resnet.ResNet50,
                "swin-l": imagenet_resnet.SwinL,
                "swin-b": imagenet_resnet.SwinB,
            },
        }[_dataset][arch]
        model = model_fn(num_classes=num_classes, **iaa_params)

    model = nn.Sequential(
        common.Normalize(**normalize_params),
        model,
        common.Postprocess(temperature=config["temperature"]),
    )
    return model.to("cuda")


def build_classifier(config: dict[str, Any]):
    """Build a classifier and optimizer given config in config."""
    model: nn.Module
    all_archs = config["arch"]
    ensemble_type = config["ensemble_type"]
    freeze_ensemble_but_first = config["freeze_ensemble_but_first"]
    if len(all_archs) > 1:
        # Build ensemble
        members = [_build_one_model(config, arch=arch) for arch in all_archs]
        if freeze_ensemble_but_first:
            # Exclude public models from being updated
            for member in members[1:]:
                for param in member.parameters():
                    param.requires_grad = False

        if ensemble_type not in ENSEMBLE_MODELS:
            raise NotImplementedError(
                f"Ensemble type ({ensemble_type}) is not valid! Available "
                f"ensemble types are: {list(ENSEMBLE_MODELS.keys())}."
            )

        logger.info("Creating %s Ensemble instance...", ensemble_type)
        model = ENSEMBLE_MODELS[ensemble_type](
            members,
            weights=None,
            freeze_ensemble_but_first=freeze_ensemble_but_first,
            trs_lambda_a=config["trs_lambda_a"],
            trs_lambda_b=config["trs_lambda_b"],
            dverge_reg_coeff=config["dverge_reg_coeff"],
        )
    else:
        model = _build_one_model(config, arch=all_archs[0])

    n_model = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("=> Total params: %.2fM", n_model)
    n_trn = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info("=> Total training params: %.2fM", n_trn)
    model = _wrap_distributed(model, config["distributed"], config["gpu"])

    p_wd, p_non_wd = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            if "bias" in n or "ln" in n or "bn" in n:
                p_non_wd.append(p)
            else:
                p_wd.append(p)

    optim_params = [
        {"params": p_wd, "weight_decay": config["wd"]},
        {"params": p_non_wd, "weight_decay": 0},
    ]

    if config["optim"] == "sgd":
        optimizer = torch.optim.SGD(
            optim_params,
            lr=config["lr"],
            momentum=config["momentum"],
            weight_decay=config["wd"],
        )
    else:
        optimizer = torch.optim.AdamW(
            optim_params,
            lr=config["lr"],
            betas=config["betas"],
            eps=config["eps"],
            weight_decay=config["wd"],
        )

    scaler = amp.GradScaler(enabled=not config["full_precision"])

    # Optionally resume from a checkpoint
    if config["resume"] or config["evaluate"]:
        resume_exists = False

        if not config["evaluate"]:
            model_paths = [f"{config['output_dir']}/checkpoint_last.pt"]
            if config["resume_if_exist"] and os.path.isfile(model_paths[0]):
                resume_exists = True
            else:
                # Use --resume when there is no checkpoint_last.pt, i.e., resume
                # from pre-trained models and not from crashed training.
                model_paths = config["resume"]
            logger.info("Trying to resume training from %s", model_paths)
        else:
            if config["resume"]:
                model_paths = config["resume"]
            else:
                model_paths = [f"{config['output_dir']}/checkpoint_best.pt"]
            logger.info("Trying to load model from %s", model_paths)

        if len(model_paths) == 1:
            _resume_model(
                config,
                model_paths[0],
                model,
                optimizer,
                scaler,
                resume_exists=resume_exists,
                pretrained_exists=is_pretrained(config["arch"][0]),
            )
        else:
            assert isinstance(
                model.module, ensemble.Ensemble
            ), "model must be ensemble to have multiple model paths!"
            for model_path, member, arch in zip(
                model_paths, model.module.members, config["arch"]
            ):
                if not model_path:
                    continue
                _resume_model(
                    config,
                    model_path,
                    member,
                    optimizer,
                    scaler,
                    resume_exists=resume_exists,
                    pretrained_exists=is_pretrained(arch),
                )
    elif not is_pretrained(config["arch"][0]):
        logger.info("=> Model is randomly initialized.")

    return model, optimizer, scaler


def _resume_model(
    config,
    model_path,
    model,
    optimizer,
    scaler,
    resume_exists: bool = False,
    pretrained_exists: bool = False,
):
    gpu: int = config["gpu"]
    if os.path.isfile(model_path):
        logger.info("=> Loading resume checkpoint %s...", model_path)
        if gpu is None:
            checkpoint = torch.load(model_path)
        else:
            # Map model to be loaded to specified single gpu.
            checkpoint = torch.load(model_path, map_location=f"cuda:{gpu}")

        state_dict = {}
        for name, value in checkpoint["state_dict"].items():
            if name.startswith("module.") and not hasattr(model, "module"):
                # Remove "module." prefix from state_dict
                name = name[7:]
            elif not name.startswith("module.") and hasattr(model, "module"):
                name = f"module.{name}"
            state_dict[name] = value

        model.load_state_dict(state_dict, strict=True)
        if not config["load_weight_only"] or resume_exists:
            config["start_epoch"] = checkpoint["epoch"]
            optimizer.load_state_dict(checkpoint["optimizer"])
            scaler.load_state_dict(checkpoint["scaler"])
        logger.info(
            "=> Loaded resume checkpoint (epoch %d)",
            checkpoint["epoch"],
        )
    elif pretrained_exists:
        logger.info(
            "=> Tried to resume but found no checkpoint. Using pre-trained "
            "weight instead."
        )
    elif model_path:
        raise FileNotFoundError(f"=> No checkpoint found at {model_path}")
    else:
        logger.info("=> Tried to resume if exist but found no checkpoint.")

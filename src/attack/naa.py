"""Neuron Attribution-based (NA) Attack.

Reference: https://arxiv.org/abs/2204.00008.
Code adapted from:
https://github.com/ZhengyuZhao/TransferAttackEval/blob/main/attacks/feature_disruption_attacks.py#L586.
"""

from __future__ import annotations

import logging
from typing import Callable

import torch
from torch import nn

from src.attack.pgd import PGDAttack
from src.utils.types import BatchImages, BatchLabels

logger = logging.getLogger(__name__)


def _get_source_layers(model: nn.Module) -> dict[str, nn.Module]:
    """Get all base layers from `model`."""
    layer_names = []
    for name, _ in model.named_modules():
        layer_names.append(name)
    logger.info("All layer names: %s", layer_names)
    return {name: model.get_submodule(name) for name in layer_names}


class NAAttack(PGDAttack):
    """Neuron Attribution-based (NA) Attack."""

    # These may not be an ideal layer!
    LAYERS_DICT = {
        # ImageNet
        "timm-efficientnet-b3": "_wrapped_model.blocks.1",
        "timm-mnasnet-100": "_wrapped_model.blocks.1",
        "timm-inception-v3": "_wrapped_model.Mixed_5d",
        "timm-mixer-b16": "_wrapped_model.blocks.2",
        "timm-rexnet-100": "_wrapped_model.features.3",
        "timm-hrnet-w18": "_wrapped_model.stage2.0.branches.0",
        "timm-vgg-11": "_wrapped_model.features.6",
        "timm-mobilenet-v3-l": "_wrapped_model.blocks.1",
        "robustbench-liu23-swin-b": "_wrapped_model.model.layers.0",
        "robustbench-singh23-convnext-t": "_wrapped_model.stages.1",
        "robustbench-singh23-vit-s": "_wrapped_model.model.blocks.4",
        "robustbench-debenedetti22-xcit-s12": "_wrapped_model.blocks.3",
        "robustbench-salman20-wrn-50-2": "_wrapped_model.model.layer1",
        "robustbench-salman20-rn-50": "_wrapped_model.model.layer1",
        "robustbench-tian22-deit-b": "_wrapped_model.model.blocks.2",
        "robustbench-tian22-deit-s": "_wrapped_model.model.blocks.2",
        "robustbench-erichson22-rn-50": "_wrapped_model.model.layer1",
        "robustbench-hendrycks20many-rn-50": "_wrapped_model.model.layer1",
        "robustbench-hendrycks20augmix-rn-50": "_wrapped_model.model.layer1",
        "robustbench-geirhos18-rn-50": "_wrapped_model.model.layer1",
        # CIFAR
        "chenyaofo-vgg11-bn": "features.6",
        "chenyaofo-mobilenetv2-x0-5": "features.1",
        "chenyaofo-shufflenetv2-x0-5": "stage2",
        "chenyaofo-repvgg-a0": "stage1",
        "huggingface-rn-50": "_wrapped_model.resnet.encoder.stages.0",
        "huyvnphan-vgg11-bn": "features.6",
        "huyvnphan-googlenet": "inception3a",
        "inception-v3": "Mixed_5d",
        "densenet121": "features.denseblock1",
        "robustbench-gowal21-rn-18": "_wrapped_model.layer_0",
        "robustbench-xu23-wrn-28-10": "_wrapped_model.layer.0",
        "robustbench-huang22-wrn-a4": "_wrapped_model.blocks.0",
        "robustbench-debenedetti22-xcit": "_wrapped_model.model.blocks.3",
        "robustbench-diffenderfer21-wrn-18-2": "ensemble",
        "robustbench-hendrycks20-resnext-29-32x4d": "_wrapped_model.stage_1",
        "robustbench-rade21-wrn-34-10": "_wrapped_model.layer.0",
        "robustbench-jia22-wrn-34-20": "_wrapped_model.block1",
        "robustbench-bai23-mix": "_wrapped_model.comp_model.models.0.body.block2",
        "robustbench-rade21-rn-18": "_wrapped_model.layer_0",
        "robustbench-cui23-wrn-28-10" : "_wrapped_model.layer.0",
        # Others
        "wrn-70-16": "_wrapped_model.layer.0",
        "wrn-34-10": "_wrapped_model.block1",  # sehwag21, augustin20, wu20
        "wrn-28-10": "_wrapped_model.block1",  # dai21, carmon19, sehwag20
        "rn-18": "_wrapped_model.layer1",
        "rn-152": "_wrapped_model.layer1",
        "resnet": "layer1",
        "efficientnet": "nonlin",
        "dla": "layer2",
        "mobilenet": "nonlin1",
        "resnext": "layer1",
        "senet": "layer1",
        "convmixer": "4",
        "vit-b": "_wrapped_model.vit.encoder.layer.4",
        "beit": "_wrapped_model.beit.encoder.layer.4",
        "convnext": "_wrapped_model.convnext.encoder.stages.1",
        "swint": "_wrapped_model.swin.encoder.layers.1",
        "swin-t": "_wrapped_model.swin.encoder.layers.1",
        "clip-RN50": "_wrapped_model.visual.layer1",
    }

    def __init__(
        self,
        attack_config: dict[str, str | int | float],
        core_model: nn.Module,
        loss_fn: Callable[[tuple[torch.Tensor, torch.Tensor]], torch.Tensor],
        norm: str = "Linf",
        eps: float = 8 / 255,
        dataset: str = "cifar10",
        arch: str | list[str] = "resnet18",
        **kwargs,
    ) -> None:
        """Initialize NAAttack."""
        super().__init__(
            attack_config, core_model, loss_fn, norm, eps, **kwargs
        )
        # Select 2nd module in core_model since the first is normalization
        source_layers = _get_source_layers(self._core_model.module[1])

        # Search over archs
        assert isinstance(arch, (str, list))
        # TODO(ensemble): Support ensemble
        arch = arch if isinstance(arch, str) else arch[0]
        for arch_key, value in self.LAYERS_DICT.items():
            if arch_key in arch:
                layer_name = value
                break
        if layer_name is None:
            raise NotImplementedError(
                f"arch {arch} not supported in NAAttack! Please define"
                "the layer name in `NAAttack.LAYERS_DICT[dataset]`."
            )
        if layer_name == "ensemble":
            raise NotImplementedError(
                "Ensemble models not supported in NAAttack!"
            )
        logger.info("Using layer %s for NAAttack.", layer_name)
        self._feature_layer = source_layers[layer_name]

    @torch.enable_grad()
    def _grad_aggregate(self, x: BatchImages, y: BatchLabels) -> BatchImages:
        num_grad_aggregate_steps = 30
        hard_y = y.argmax(1)

        def get_mid_grad(m, i, o):
            global mid_grad
            mid_grad = o

        grad_hook = self._feature_layer.register_full_backward_hook(
            get_mid_grad
        )

        agg_grad = 0
        for i in range(1, num_grad_aggregate_steps + 1):
            self._core_model.zero_grad()
            x_i = x * i / num_grad_aggregate_steps
            x_i.requires_grad_()
            output = self._core_model(x_i)
            output = torch.softmax(output, dim=1)
            loss = torch.index_select(output, 1, hard_y).sum()
            loss.backward()
            agg_grad += mid_grad[0].detach()
        grad_hook.remove()
        agg_grad /= num_grad_aggregate_steps
        agg_grad.detach_()
        return agg_grad

    # This method is only called internally so different signature is fine.
    # pylint: disable=arguments-differ
    def _get_grad(
        self,
        x_adv: torch.Tensor,
        output_prime: torch.Tensor,
        agg_grad: torch.Tensor,
    ) -> torch.Tensor:
        """Compute logits, loss, gradients."""
        with torch.enable_grad():
            x_adv.requires_grad_()
            self._core_model(x_adv, **self._forward_args)
            loss = ((mid_output - output_prime) * agg_grad).sum()
            grads = torch.autograd.grad(loss, x_adv)[0]
            grads.detach_()
        return grads

    def _forward_linf(self, x: BatchImages, y: BatchLabels) -> BatchImages:
        mode = self._core_model.training
        self._core_model.eval()
        x = x.detach()
        y = y.detach()

        agg_grad = self._grad_aggregate(x, y)

        def get_mid_output(m, i, o):
            global mid_output
            if isinstance(o, tuple):
                mid_output = o[0]
            else:
                mid_output = o

        output_hook = self._feature_layer.register_forward_hook(get_mid_output)
        self._core_model(torch.zeros_like(x))
        output_prime = mid_output.detach()

        # Initialize worst-case inputs
        x_adv_worst = x.clone()
        worst_losses = torch.zeros(len(x), 1, 1, 1, device=x.device)

        # Repeat PGD for specified number of restarts
        for _ in range(self._num_restarts):
            x_adv = x.clone()

            # Initialize adversarial inputs
            x_adv += torch.zeros_like(x_adv).uniform_(-self._eps, self._eps)
            x_adv.clamp_(0, 1)

            # Run PGD on inputs for specified number of steps
            for _ in range(self._num_steps):
                grads = self._get_grad(x_adv, output_prime, agg_grad)
                with torch.no_grad():
                    x_adv.detach_()
                    # Perform gradient update, project to norm ball
                    x_adv += self._step_size * torch.sign(grads)
                    x_adv = self._project_linf(x_adv, x)

            x_adv_worst, worst_losses = self._get_best_adv(
                x_adv, y, x_adv_worst, worst_losses
            )

        output_hook.remove()
        # Return worst-case perturbed input logits
        self._core_model.train(mode)
        return x_adv_worst.detach()

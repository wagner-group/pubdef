"""WideResNet model for CIFAR dataset from DeepMind.

Code is taken from
https://github.com/deepmind/deepmind-research/blob/master/adversarial_robustness/pytorch/model_zoo.py.
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


class _Block(nn.Module):
    """WideResNet Block."""

    def __init__(self, in_planes, out_planes, stride, activation_fn=nn.ReLU):
        super().__init__()
        self.batchnorm_0 = nn.BatchNorm2d(in_planes)
        self.relu_0 = activation_fn()
        # We manually pad to obtain the same effect as `SAME` (necessary when
        # `stride` is different than 1).
        self.conv_0 = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=0,
            bias=False,
        )
        self.batchnorm_1 = nn.BatchNorm2d(out_planes)
        self.relu_1 = activation_fn()
        self.conv_1 = nn.Conv2d(
            out_planes,
            out_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.has_shortcut = in_planes != out_planes
        if self.has_shortcut:
            self.shortcut = nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )
        else:
            self.shortcut = None
        self._stride = stride

    def forward(self, x, before_relu=False, intermediate=False):
        """Forward pass."""
        if self.has_shortcut:
            x = self.relu_0(self.batchnorm_0(x))
        else:
            out = self.relu_0(self.batchnorm_0(x))
        v = x if self.has_shortcut else out
        if self._stride == 1:
            v = F.pad(v, (1, 1, 1, 1))
        elif self._stride == 2:
            v = F.pad(v, (0, 1, 0, 1))
        else:
            raise ValueError("Unsupported `stride`.")
        out = self.conv_0(v)
        out = self.batchnorm_1(out)
        if intermediate:
            return out if before_relu else self.relu_1(out)
        out = self.relu_1(out)
        out = self.conv_1(out)
        out = torch.add(self.shortcut(x) if self.has_shortcut else x, out)
        return out


class _BlockGroup(nn.Module):
    """WideResNet block group."""

    def __init__(
        self, num_blocks, in_planes, out_planes, stride, activation_fn=nn.ReLU
    ):
        super().__init__()
        block = []
        for i in range(num_blocks):
            block.append(
                _Block(
                    i == 0 and in_planes or out_planes,
                    out_planes,
                    i == 0 and stride or 1,
                    activation_fn=activation_fn,
                )
            )
        self.block = nn.Sequential(*block)

    def forward(self, x, **kwargs):
        """Forward pass."""
        return self.block(x, **kwargs)


class WideResNet(nn.Module):
    """WideResNet."""

    def __init__(
        self,
        num_classes: int = 10,
        depth: int = 28,
        width: int = 10,
        activation_fn: nn.Module = nn.ReLU,
        mean: Union[Tuple[float, ...], float] = CIFAR10_MEAN,
        std: Union[Tuple[float, ...], float] = CIFAR10_STD,
        padding: int = 0,
        num_input_channels: int = 3,
    ):
        """Initialize a WideResNet model."""
        super().__init__()
        self.mean = torch.tensor(mean).view(num_input_channels, 1, 1)
        self.std = torch.tensor(std).view(num_input_channels, 1, 1)
        self.mean_cuda = None
        self.std_cuda = None
        self.padding = padding
        num_channels = [16, 16 * width, 32 * width, 64 * width]
        assert (depth - 4) % 6 == 0
        num_blocks = (depth - 4) // 6
        self.num_blocks = num_blocks  # 34: 5
        self.init_conv = nn.Conv2d(
            num_input_channels,
            num_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.layer = nn.Sequential(
            _BlockGroup(
                num_blocks,
                num_channels[0],
                num_channels[1],
                1,
                activation_fn=activation_fn,
            ),
            _BlockGroup(
                num_blocks,
                num_channels[1],
                num_channels[2],
                2,
                activation_fn=activation_fn,
            ),
            _BlockGroup(
                num_blocks,
                num_channels[2],
                num_channels[3],
                2,
                activation_fn=activation_fn,
            ),
        )
        self.batchnorm = nn.BatchNorm2d(num_channels[3])
        self.relu = activation_fn()
        self.logits = nn.Linear(num_channels[3], num_classes)
        self.num_channels = num_channels[3]

    def get_features(self, x, layer, before_relu=False):
        multiplier = self.num_blocks * 2
        layer_offset = [multiplier] * 3
        layer_offset = [1] + layer_offset
        layer_offset = np.cumsum(layer_offset)

        if self.padding > 0:
            x = F.pad(x, (self.padding,) * 4)
        if x.is_cuda:
            if self.mean_cuda is None:
                self.mean_cuda = self.mean.cuda()
                self.std_cuda = self.std.cuda()
            out = (x - self.mean_cuda) / self.std_cuda
        else:
            out = (x - self.mean) / self.std
        x = self.init_conv(x)

        if layer == 1:
            return x

        layers = self.layer
        for i in range(1, len(layer_offset)):
            if layer <= layer_offset[i]:
                relative_layer = layer - layer_offset[i - 1]
                x = self.layer_block_forward(
                    x, layers[i - 1], relative_layer, before_relu=before_relu
                )
                return x
            x = layers[i - 1](x)

        out = x
        out = self.relu(self.batchnorm(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.num_channels)
        return self.logits(out)

    def layer_block_forward(
        self, x, layer_block, relative_layer, before_relu=False
    ):
        out = x
        layer_block = layer_block.block
        total_layers = (relative_layer + 1) // 2
        for i in range(total_layers):
            if i == total_layers - 1:
                out = layer_block[i](
                    out, before_relu, intermediate=bool(relative_layer % 2)
                )
            else:
                out = layer_block[i](out)
        return out

    def forward(self, x):
        """Forward pass."""
        if self.padding > 0:
            x = F.pad(x, (self.padding,) * 4)
        if x.is_cuda:
            if self.mean_cuda is None:
                self.mean_cuda = self.mean.cuda()
                self.std_cuda = self.std.cuda()
            out = (x - self.mean_cuda) / self.std_cuda
        else:
            out = (x - self.mean) / self.std
        out = self.init_conv(out)
        out = self.layer(out)
        out = self.relu(self.batchnorm(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.num_channels)
        return self.logits(out)


def wideresnet28_10(**kwargs):
    """WideResNet-28-10 model."""
    return WideResNet(depth=28, width=10, **kwargs)


def wideresnet34_10(**kwargs):
    """WideResNet-34-10 model."""
    return WideResNet(depth=34, width=10, **kwargs)


def wideresnet34_20(**kwargs):
    """WideResNet-34-20 model."""
    return WideResNet(depth=34, width=20, **kwargs)


def wideresnet70_16(**kwargs):
    """WideResNet-70-16 model."""
    return WideResNet(depth=70, width=16, **kwargs)

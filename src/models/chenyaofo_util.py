"""Utility functions for building CIFAR models from chenyaofo.

Link: https://github.com/chenyaofo/pytorch-cifar-models.
"""

from __future__ import annotations

import torch
from pathlib import Path

_WEIGHT_URL = {
    "cifar10": {
        'vgg11_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar10_vgg11_bn-eaeebf42.pt',
        'vgg13_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar10_vgg13_bn-c01e4a43.pt',
        'vgg16_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar10_vgg16_bn-6ee7ea24.pt',
        'vgg19_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar10_vgg19_bn-57191229.pt',
        'mobilenetv2_x0_5': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/mobilenetv2/cifar10_mobilenetv2_x0_5-ca14ced9.pt',
        'mobilenetv2_x0_75': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/mobilenetv2/cifar10_mobilenetv2_x0_75-a53c314e.pt',
        'mobilenetv2_x1_0': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/mobilenetv2/cifar10_mobilenetv2_x1_0-fe6a5b48.pt',
        'mobilenetv2_x1_4': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/mobilenetv2/cifar10_mobilenetv2_x1_4-3bbbd6e2.pt',
        'repvgg_a0': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/repvgg/cifar10_repvgg_a0-ef08a50e.pt',
        'repvgg_a1': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/repvgg/cifar10_repvgg_a1-38d2431b.pt',
        'repvgg_a2': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/repvgg/cifar10_repvgg_a2-09488915.pt',
        'resnet20': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet20-4118986f.pt',
        'resnet32': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet32-ef93fc4d.pt',
        'resnet44': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet44-2a3cabcb.pt',
        'resnet56': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet56-187c023a.pt',
        'shufflenetv2_x0_5': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/shufflenetv2/cifar10_shufflenetv2_x0_5-1308b4e9.pt',
        'shufflenetv2_x1_0': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/shufflenetv2/cifar10_shufflenetv2_x1_0-98807be3.pt',
        'shufflenetv2_x1_5': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/shufflenetv2/cifar10_shufflenetv2_x1_5-296694dd.pt',
        'shufflenetv2_x2_0': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/shufflenetv2/cifar10_shufflenetv2_x2_0-ec31611c.pt',
    },
    "cifar100": {
        'vgg11_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar100_vgg11_bn-57d0759e.pt',
        'vgg13_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar100_vgg13_bn-5ebe5778.pt',
        'vgg16_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar100_vgg16_bn-7d8c4031.pt',
        'vgg19_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar100_vgg19_bn-b98f7bd7.pt',
        'mobilenetv2_x0_5': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/mobilenetv2/cifar100_mobilenetv2_x0_5-9f915757.pt',
        'mobilenetv2_x0_75': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/mobilenetv2/cifar100_mobilenetv2_x0_75-d7891e60.pt',
        'mobilenetv2_x1_0': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/mobilenetv2/cifar100_mobilenetv2_x1_0-1311f9ff.pt',
        'mobilenetv2_x1_4': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/mobilenetv2/cifar100_mobilenetv2_x1_4-8a269f5e.pt',
        'repvgg_a0': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/repvgg/cifar100_repvgg_a0-2df1edd0.pt',
        'repvgg_a1': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/repvgg/cifar100_repvgg_a1-c06b21a7.pt',
        'repvgg_a2': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/repvgg/cifar100_repvgg_a2-8e71b1f8.pt',
        'resnet20': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet20-23dac2f1.pt',
        'resnet32': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet32-84213ce6.pt',
        'resnet44': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet44-ffe32858.pt',
        'resnet56': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet56-f2eff4c8.pt',
        'shufflenetv2_x0_5': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/shufflenetv2/cifar100_shufflenetv2_x0_5-1977720f.pt',
        'shufflenetv2_x1_0': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/shufflenetv2/cifar100_shufflenetv2_x1_0-9ae22beb.pt',
        'shufflenetv2_x1_5': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/shufflenetv2/cifar100_shufflenetv2_x1_5-e2c85ad8.pt',
        'shufflenetv2_x2_0': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/shufflenetv2/cifar100_shufflenetv2_x2_0-e7e584cd.pt',
    }
}


def download_chenyaofo_weight(arch, dataset, dst=None):
    model_name = arch.replace('chenyaofo-', '').replace('-', '_')
    url = _WEIGHT_URL[dataset][model_name]
    dst = dst or f"weights/chenyaofo/{dataset}"
    path = Path(dst) / f"{model_name}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        return dst
    torch.hub.download_url_to_file(url, dst, hash_prefix=None, progress=True)
    return dst

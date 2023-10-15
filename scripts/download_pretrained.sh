#!/bin/bash

# Install gdown
pip install gdown

mkdir -p weights && cd weights || exit 1

# Download CIFAR-10 pre-trained models from https://github.com/huyvnphan/PyTorch_CIFAR10
gdown https://drive.google.com/uc?id=17fmN8eQdLpq2jIMQ_X0IXDPXfI9oVWgq
unzip state_dicts.zip && rm state_dicts.zip

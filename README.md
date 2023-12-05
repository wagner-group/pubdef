# PubDef: Defending Against Transfer Attacks From Public Models

> TLDR: Training neural networks against a small subset of transfer attacks generalizes surprisingly well to all the attacks across multiple source models and attack algorithms.

Chawin Sitawarin, Jaewon Chang*, David Huang*, Wesson Altoyan, David Wagner  
[[ArXiv](https://arxiv.org/abs/2310.17645)]

<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

## Abstract

Adversarial attacks have been a looming and unaddressed threat in the industry. However, through a decade-long history of the robustness evaluation literature, we have learned that mounting a strong or optimal attack is challenging. It requires both machine learning and domain expertise. In other words, the white-box threat model, religiously assumed by a large majority of the past literature, is unrealistic. In this paper, we propose a new practical threat model where the adversary relies on transfer attacks through publicly available surrogate models. We argue that this setting will become the most prevalent for security-sensitive applications in the future. We evaluate the transfer attacks in this setting and propose a specialized defense method based on a game-theoretic perspective. The defenses are evaluated under 24 public models and 11 attack algorithms across three datasets (CIFAR-10, CIFAR-100, and ImageNet). Under this threat model, our defense, PubDef, outperforms the state-of-the-art white-box adversarial training by a large margin with almost no loss in the normal accuracy. For instance, on ImageNet, our defense achieves 62% accuracy under the strongest transfer attack vs only 36% of the best adversarially trained model. Its accuracy when not under attack is only 2% lower than that of an undefended model (78% vs 80%).

## Package Dependencies

Recommended versions:

- `python >= 3.8`.
- `cuda >= 11.2`.
- See `requirements.txt` or `environment.yml` for all packages' version (*@chawins: maybe outdated*)

```bash
# Install dependencies with pip
pip install -r requirements.txt

# OR install dependencies with conda
conda env create -f environment.yml

# OR install dependencies manually with latest packages
# Install pytorch 1.10 (or later)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -U h5py scikit-image timm torchmetrics matplotlib kornia wandb foolbox termcolor jaxtyping frozendict tabulate transformers einops safetensors
pip install 'git+https://github.com/fra31/auto-attack'
pip install 'git+https://github.com/openai/CLIP.git'
pip install 'git+https://github.com/RobustBench/robustbench.git'
# For development
pip install -U black pydocstyle pycodestyle
```

NOTE: Installing `robustbench` after `timm` will downgrade it to a compatible version (`0.6.13`). This should not be a problem.

## File Organization

## Public Models

The public models used in the experiments are defined in `exp_params.py` under a global variable called `PUBLIC_MODELS` which is a dictionary organized by the dataset name and the training scheme.

Most of the models are automatically downloaded and cached when they are first used.
Only a few models have to be downloaded manually where we provide a script at `scripts/download_pretrained.sh` to download them.

## Example Scripts

### Evaluation Script

We have prepared pre-trained PubDef models for CIFAR-10, CIFAR-100, and ImageNet as well as some transfer attacks we generated for download from Kaggle ([link](https://www.kaggle.com/datasets/csitawarin/pubdef-defending-against-transfer-attacks/)).
The compressed file size is about 2GB.

```bash
# Install kaggle CLI
pip install kaggle
# Get a Kaggle API token. Place it at .kaggle/kaggle.json
...
# Download the pre-trained models and transfer attacks
kaggle datasets download csitawarin/pubdef-defending-against-transfer-attacks
# Unzip and move results dir to the root directory of this repository
# (e.g., ...pubdef/results/cifar10_wideresnet34-10_pubdef/...)
unzip pubdef-defending-against-transfer-attacks.zip && mv results/results .
# Run the evaluation script
bash scripts/example_test.sh
```

The script above should produce the following results:

CIFAR-10 (attack: `mpgd`, source model: `cifar10_huggingface-vit-base`)

| Models                           | Clean Accuracy | M-PGD Accuracy |
| -------------------------------- | -------------: | -------------: |
| `cifar10_wideresnet34-10_pubdef` |          96.10 |          95.46 |
| `cifar10_huggingface-convnext`   |          97.35 |          55.03 |

CIFAR-100 (attack: `mpgd`, source model: `cifar100_huggingface-swin-tiny`)

| Models                              | Clean Accuracy | M-PGD Accuracy |
| ----------------------------------- | -------------: | -------------: |
| `cifar100_wideresnet34-10_pubdef`   |          76.53 |          65.09 |
| `cifar100_robustbench-rade21-rn-18` |          61.50 |          59.81 |

ImageNet (attack: `mpgd`, source model: `imagenet_huggingface-swin-t`)

| Models                         | Clean Accuracy | M-PGD Accuracy |
| ------------------------------ | -------------: | -------------: |
| `imagenet_resnet50_pubdef`     |          78.65 |          71.94 |
| `imagenet_timm-mobilenet-v3-l` |          74.35 |          61.55 |

### Full Script

A full usage example is provided in `scripts/example_full.sh`. This script has five steps, training base model, training source model, generating transfer adversarial examples, re-training base model with the defense, and evaluating it against other transfer attacks. To run it, simply execute the following command (some parameters may need to be modified):

```bash
bash scripts/example_full.sh
```

The following script is found in `scripts/example_train_with_public_models.sh`; it also runs the full training defense process but rather than having to locally train source models (this script directly loads two which can be configured to load more public models as a proxy for step 2 in the `scripts/example_full.sh` script)

```bash
bash scripts/example_train_with_public_models.sh
```

## Experiment Setup

Lists of models and attack algorithms used in the experiments can also be found/edited at [`exp_params.py`](https://github.com/wagner-group/pubdef/blob/main/exp_params.py).

### CIFAR-10

Standard training models:

- `chenyaofo-resnet20`
- `chenyaofo-vgg11-bn`: Need to use code from `huyvnphan` to set ReLU in-place=False so we can run NAAttack. Seems to create attacks with an unusually high ASR.
- `chenyaofo-mobilenetv2-x0-5`
- `chenyaofo-shufflenetv2-x0-5`
- `chenyaofo-repvgg-a0`
- `huyvnphan-densenet121`
- `huyvnphan-inception-v3`
- `huggingface-vit-base`: `"aaraki/vit-base-patch16-224-in21k-finetuned-cifar10"` (T)
- `huggingface-beit`: `"jadohu/BEiT-finetuned"`
- `huggingface-convnext`: `"ahsanjavid/convnext-tiny-finetuned-cifar10"`
- `cifar10_convmixer_none_ce_seed0_lr0.05_wd0.005_adamw`: Only locally trained model
- `clip-RN50-laion`

Linf adversarially trained models:

- `robustbench-wang23-wrn-70-16` (T)
- `robustbench-xu23-wrn-28-10`
- `robustbench-debenedetti22-xcit-l12`
- `robustbench-sehwag20-wrn-28-10`

L2 adversarially trained models:

- `robustbench-wang23-wrn-70-16-l2` (T)
- `robustbench-rebuffi21-wrn-70-16-l2`
- `robustbench-augustin20-wrn-34-10-l2`
- `robustbench-rony19-wrn-28-10-l2`

Corruption trained models:

- `robustbench-diffenderfer21-wrn-18-2` (T)
- `robustbench-kireev21-rn-18`
- `robustbench-hendrycks20-resnext-29-32x4d`
- `robustbench-modas21-rn-18`

### CIFAR-100

Standard training models:

- `chenyaofo-resnet20` (T)
- `chenyaofo-vgg11-bn`
- `chenyaofo-mobilenetv2-x0-5`
- `chenyaofo-shufflenetv2-x0-5`
- `chenyaofo-repvgg-a0`
- `huggingface-vit-base`: `"edumunozsala/vit_base-224-in21k-ft-cifar100"`
- `huggingface-swin-tiny`: `"MazenAmria/swin-tiny-finetuned-cifar100"`
- `densenet121_local`: Locally trained
- `senet18_local`: Locally trained
- `inception-v3_local`: Locally trained
- `convmixer_local`: Locally trained
- `clip-RN50-laion`

Linf adversarially trained models:

- `robustbench-wang23-wrn-70-16` (T)
- `robustbench-cui23-wrn-28-10`
- `robustbench-bai23-mix`
- `robustbench-debenedetti22-xcit`
- `robustbench-jia22-wrn-34-20`
- `robustbench-rade21-rn-18`

Corruption trained models:

- `robustbench-diffenderfer21-wrn-18-2` (T)
- `robustbench-modas21-rn-18`
- `robustbench-hendrycks20-resnext-29-32x4d`
- `robustbench-addepalli22-wrn-34-10`
- `robustbench-gowal20-wrn-70-16`
- `robustbench-diffenderfer21-bin-wrn-18-2`

### ImageNet

Standard training models:

- `huggingface-rn-50` (T)
- `huggingface-vit-b`
- `huggingface-swin-t`
- `huggingface-convnext-t`
- `timm-efficientnet-b3`
- `timm-mnasnet-100`
- `timm-inception-v3`
- `timm-mixer-b16`
- `timm-rexnet-100`
- `timm-hrnet-w18`
- `timm-vgg-11`
- `timm-mobilenet-v3-l`

Linf adversarially trained models:

- `robustbench-liu23-swin-b` (T)
- `robustbench-singh23-convnext-t`
- `robustbench-singh23-vit-s`
- `robustbench-debenedetti22-xcit-s12`
- `robustbench-salman20-wrn-50-2`
- `robustbench-salman20-rn-50`

Corruption trained models:

- `robustbench-tian22-deit-b`
- `robustbench-tian22-deit-s` (T)
- `robustbench-erichson22-rn-50`
- `robustbench-hendrycks20many-rn-50`
- `robustbench-hendrycks20augmix-rn-50`
- `robustbench-geirhos18-rn-50`

If some models from `robustbench` or `timm` are not found (`KeyError`), please try upgrading these packages.

```bash
pip install --force-reinstall --no-deps git+https://github.com/RobustBench/robustbench.git
pip install -U timm
```

### List of Transfer Attack Algorithms

Attacks are defined in [`src/attack/`](https://github.com/wagner-group/pubdef/tree/main/src/attack).

- `pgd`
- `mpgd`
- `pregradient`
- `di`
- `ti`
- `naa`
- `auto-pgd-dlr`
- `auto-pgd-ce`
- `admix`
- `ni-si-ti-dim`
- `ni-admix-ti-dim`

## Contributors

See `TEAM.md` for a list of contributors.

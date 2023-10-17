# Defenses against Transfer Attack Threats from Public Models

Defending ML models against black-box transfer attacks

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

TODO(chawins@): verify package installation

## File Organization

## Public Models

The public models used in the experiments are defined in `exp_params.py` under a global variable called `PUBLIC_MODELS` which is a dictionary organized by the dataset name and the training scheme.

Most of the models are automatically downloaded and cached when they are first used.
Only a few models have to be downloaded manually where we provide a script at `scripts/download_pretrained.sh` to download them.

## Example Scripts

A full usage example is provided in `scripts/example_full.sh`. This script has five steps, training base model, training source model, generating transfer adversarial examples, re-training base model with the defense, and evaluating it against other transfer attacks. To run it, simply execute the following command (some parameters may need to be modified):

```bash
bash scripts/example_full.sh
```

The following script is found in `scripts/training-with-public-models.sh`; it also runs the full training defense process but rather than having to locally train source models (this script directly loads two which can be configured to load more public models as a proxy for step 2 in the `scripts/example_full.sh` script)

```bash
bash scripts/training-with-public-models.sh
```

## List of Public Models Used in the Experiments

### CIFAR-10

Standard training models:

- `chenyaofo-resnet20`
- `chenyaofo-vgg11-bn`: Need to use code from `huyvnphan` to set ReLU in-place=False so we can run NAAttack. Seems to create attacks with an unusually high ASR.
- `chenyaofo-mobilenetv2-x0-5`
- `chenyaofo-shufflenetv2-x0-5`
- `chenyaofo-repvgg-a0`
- `huyvnphan-densenet121`
- `huyvnphan-inception-v3`
- `huggingface-vit-base`: `"aaraki/vit-base-patch16-224-in21k-finetuned-cifar10"`
- `huggingface-beit`: `"jadohu/BEiT-finetuned"`
- `huggingface-convnext`: `"ahsanjavid/convnext-tiny-finetuned-cifar10"`
- `cifar10_convmixer_none_ce_seed0_lr0.05_wd0.005_adamw`: Only locally trained model
- `clip-RN50-laion`

Linf adversarially trained models:

L2 adversarially trained models:

Corruption trained models:

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

If some models from `robustbench` or `timm` are not found (`KeyError`), please try upgrading it.

```bash
pip install --force-reinstall --no-deps git+https://github.com/RobustBench/robustbench.git
pip install -U timm
```

## Attacks

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

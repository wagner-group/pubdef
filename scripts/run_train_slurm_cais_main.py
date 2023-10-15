"""Submit slurm jobs for training models."""

import subprocess
import sys
import time

NUM_GPUS = 1
TRAIN_LOSS = "ce"
TRAIN_ATTACK = "none"
SEED = 1
ARCHS = [
    "resnet18",
    # "dla",
    # "mobilenetv2",
    # "resnext29-2x64d",
    # "senet18",
    # "efficientnetb0",
    # "huggingface-vit-base",
    # "huggingface-beit",
    # "huggingface-convnext",
    # "huggingface-swint",
    # "chenyaofo-resnet20",
    # "chenyaofo-vgg11-bn",
    # "chenyaofo-mobilenetv2-x0-5",
    # "chenyaofo-shufflenetv2-x0-5",
    # "chenyaofo-repvgg-a0",
    # "huyvnphan-vgg11-bn",
    # "huyvnphan-densenet121",
    # "huyvnphan-googlenet",
    # "huyvnphan-inception-v3",
    # "convmixer",
]
DO_EVAL = True

_OUTPUT_NAME = (
    f"cifar10_train_#ARCH#_{TRAIN_ATTACK}_{TRAIN_LOSS}_s{SEED}_hash#HASH#"
)
_EVALUATE = "--evaluate" if DO_EVAL else ""

_BASE_SCRIPT = f"""#!/bin/bash
#SBATCH --job-name=transfer-defense-{_OUTPUT_NAME}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={8 * NUM_GPUS}
#SBATCH --gpus={NUM_GPUS}
#SBATCH --time=48:00:00
#SBATCH --output %j-{_OUTPUT_NAME}.log

NUM_GPU={NUM_GPUS}
BS=128
NUM_WORKERS={8 * NUM_GPUS}
DATASET="cifar10"
DATAPATH="$HOME/data/"
EPS=0.03137254901

source $HOME/.bashrc
source activate transfer-python3.10

echo #SCRIPT_PATH#

python main.py --no-distributed \
--full-precision --pretrained --data-dir $DATAPATH --dataset $DATASET \\
--workers $NUM_WORKERS --print-freq 50 --epochs 50 --batch-size $BS \\
--lr 1e-1 --wd 5e-4 --epsilon $EPS --atk-norm Linf --experiment none \\
--adv-beta 0.5 --load-weight-only \\
--arch #ARCH# \\
--seed {SEED} \\
--train-attack {TRAIN_ATTACK} \\
--train-loss {TRAIN_LOSS} {_EVALUATE}
"""


def main():
    """Submit slurm jobs for transfer attacks evaluation."""
    num_jobs = 0
    for arch in ARCHS:
        script = _BASE_SCRIPT.strip()
        script = script.replace("#ARCH#", arch)
        hash_id = f"{hash(script) % 100000:05d}"
        script_path = (
            f"tmp/tmp_cifar10_train_{arch}_{TRAIN_ATTACK}_{TRAIN_LOSS}_s{SEED}_"
            f"hash{hash_id}.sh"
        )
        script = script.replace("#SCRIPT_PATH#", script_path)
        script = script.replace("#HASH#", hash_id).strip()

        with open(script_path, "w", encoding="utf-8") as file:
            file.write(script)
        print(f"Submitting job at {script_path}")
        output = subprocess.run(["sbatch", script_path], check=True)
        time.sleep(1)
        if output.returncode != 0:
            print("Failed to submit job!")
            sys.exit(1)
        num_jobs += 1
    print(f"All {num_jobs} jobs submitted successfully!")


if __name__ == "__main__":
    main()

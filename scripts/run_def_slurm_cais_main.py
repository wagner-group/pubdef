"""Submit slurm jobs for training defense models."""

import subprocess
import sys
import time

from src.utils.argparser import naming_hash

NUM_GPUS = 1
TRAIN_LOSS = "mat"
TRAIN_ATTACK = "none"
SEED = 0
ARCH = "resnet18"
RESUME = "cifar10_resnet18_none_ce_seed0_lr0.1_wd0.0005_sgd"
# RESUME = "cifar10_efficientnetb0_none_ce_seed0_lr0.1_wd0.0005_sgd"

# Either specify LOAD_PATHS or ATTACKS + SRC_MODELS but not both
LOAD_PATHS = [
    # "cifar10_resnet18_pgd_seed0/saved_pgd",
    # "cifar10_resnet18_pgd_seed0/saved_ti",
    # "cifar10_resnet18_pgd_seed0/saved_naa",
    # "cifar10_resnet18_none_seed1/saved_pgd",
    # "cifar10_resnet18_none_seed1/saved_ti",
    # "cifar10_resnet18_none_seed1/saved_naa",
]
# pgd mpgd pregradient di ti admix naa ni-si-ti-dim ni-admix-ti-dim
ATTACKS = [
    "pgd",
    # "ti",
]
SRC_MODELS = [
    "cifar10_resnet18_none_ce_seed1_lr0.1_wd0.0005_sgd",
    # "cifar10_resnet18_pgd_ce_seed0_lr0.1_wd0.0005_sgd_eps8",
    "cifar10_senet18_none_ce_seed0_lr0.1_wd0.0005_sgd",
    "cifar10_dla_none_ce_seed0_lr0.1_wd0.0005_sgd",
]

ADV_BETA = 0.5
# DATASET_WEIGHTS = "1.0 0.25 0.25 0.25 0.25"
DATASET_WEIGHTS = "1.0 0.3333 0.3333 0.3333"
# DATASET_WEIGHTS = "1.0 0.001 0.999"
DATASET_DROPOUT = " ".join(["0"] * (len(SRC_MODELS) + 1))
ATTACKS = " ".join(ATTACKS)
SRC_MODELS = " ".join(SRC_MODELS)

if LOAD_PATHS:
    # Auto-generated variables
    LOAD_PATHS = [f"results/{path}" for path in LOAD_PATHS]
    TRAIN_LOAD_PATHS = " ".join([f"{path}_train.pt" for path in LOAD_PATHS])
    TEST_LOAD_PATHS = " ".join([f"{path}_test.pt" for path in LOAD_PATHS])
    VAL_LOAD_PATHS = " ".join([f"{path}_val.pt" for path in LOAD_PATHS])
    NAME = naming_hash(str(sorted(TRAIN_LOAD_PATHS)))
    TRANSFER_ARGS = f"""
--train-load-paths {TRAIN_LOAD_PATHS} \\
--test-load-paths {TEST_LOAD_PATHS} \\
--val-load-paths {VAL_LOAD_PATHS}"""
else:
    TRANSFER_ARGS = f"""
--transfer-attacks {ATTACKS} \\
--src-models {SRC_MODELS}"""
    NAME = f"{naming_hash(str(sorted(SRC_MODELS)))}_{naming_hash(str(sorted(ATTACKS)))}"
TRANSFER_ARGS = TRANSFER_ARGS.strip()
_OUTPUT_NAME = f"cifar10_train_{ARCH}_{TRAIN_ATTACK}_{TRAIN_LOSS}_s{SEED}_{NAME}_hash#HASH#"

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
--adv-beta {ADV_BETA} --load-weight-only \\
--arch {ARCH} \\
--resume "results/{RESUME}/checkpoint_best.pt" \\
--seed {SEED} \\
--train-attack {TRAIN_ATTACK} \\
--train-loss {TRAIN_LOSS} \\
--dataset-weights {DATASET_WEIGHTS} \\
--dataset-dropout {DATASET_DROPOUT} \\
{TRANSFER_ARGS}
"""


def main():
    """Submit slurm jobs for transfer attacks evaluation."""
    num_jobs = 0
    script = _BASE_SCRIPT
    hash_id = f"{hash(script) % 100000:05d}"
    script_path = (
        f"tmp/tmp_cifar10_train_{ARCH}_{TRAIN_ATTACK}_{TRAIN_LOSS}_seed{SEED}_"
        f"{NAME}_hash{hash_id}.sh"
    )
    script = script.replace("#SCRIPT_PATH#", script_path)
    script = script.replace("#HASH#", hash_id)

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

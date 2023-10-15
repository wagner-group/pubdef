"""Submit slurm jobs for generating adversarial examples."""

import subprocess
import sys
import time

NUM_GPUS = 1
TRAIN_MULT = 10
VAL_MULT = 1
TEST_MULT = 1

# CLIP parameters
TEMPERATURE = 1.0
CLIP_MODEL = "RN50"
CLIP_TEMPLATE = "laion"

# Source models
SRC = [
    # "cifar10_dla_none_ce_seed0_lr0.1_wd0.0005_sgd",
    # "cifar10_dla_pgd_ce_seed0_lr0.1_wd0.0005_sgd_eps8",
    # "cifar10_efficientnetb0_none_ce_seed0_lr0.1_wd0.0005_sgd",
    # "cifar10_efficientnetb0_pgd_ce_seed0_lr0.1_wd0.0005_sgd_eps8",
    # "cifar10_mobilenetv2_none_ce_seed0_lr0.1_wd0.0005_sgd",
    # "cifar10_mobilenetv2_pgd_ce_seed0_lr0.1_wd0.0005_sgd_eps8",
    # "cifar10_resnet18_none_ce_seed0_lr0.1_wd0.0005_sgd",
    # "cifar10_resnet18_none_ce_seed1_lr0.1_wd0.0005_sgd",
    # "cifar10_resnet18_pgd_ce_seed0_lr0.1_wd0.0005_sgd_eps8",
    # "cifar10_resnext29-2x64d_none_ce_seed0_lr0.1_wd0.0005_sgd",
    # "cifar10_resnext29-2x64d_pgd_ce_seed0_lr0.1_wd0.0005_sgd_eps8",
    # "cifar10_senet18_none_ce_seed0_lr0.1_wd0.0005_sgd",
    # "cifar10_senet18_pgd_ce_seed0_lr0.1_wd0.0005_sgd_eps8",
    # "cifar10_clip",
    # "cifar10_huggingface-vit-base",
    # "cifar10_huggingface-beit",
    # "cifar10_huggingface-convnext",
    "cifar10_chenyaofo-resnet20",
    # "cifar10_chenyaofo-vgg11-bn",
    # "cifar10_chenyaofo-mobilenetv2-x0-5",
    # "cifar10_chenyaofo-shufflenetv2-x0-5",
    # "cifar10_chenyaofo-repvgg-a0",
    # "cifar10_huyvnphan-vgg11-bn",
    # "cifar10_huyvnphan-densenet121",
    # "cifar10_huyvnphan-googlenet",
    # "cifar10_huyvnphan-inception-v3",
]

# Attacks (pgd mpgd pregradient di ti admix naa ni-si-ti-dim ni-admix-ti-dim)
ATTACK = [
    # "pgd",
    "ti",
    # "naa",
    # "ni-admix-ti-dim",
]

_BASE_SCRIPT = f"""#!/bin/bash
#SBATCH --job-name=chawin-transfer-defense
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={8 * NUM_GPUS}
#SBATCH --gpus={NUM_GPUS}
#SBATCH --time=48:00:00
#SBATCH --output %j-cifar10-gen-#ATK#-#SRC#.log

NUM_GPU={NUM_GPUS}
AA_BS=#BATCH_SIZE#
NUM_WORKERS={8 * NUM_GPUS}
DATASET="cifar10"
DATAPATH="$HOME/data/"
EPS=0.03137254901

source $HOME/.bashrc
source activate transfer-python3.10

echo #SCRIPT_PATH#

GEN_ADV="python -u save_adv_main.py --no-distributed \\
    --seed 0 --full-precision --data-dir $DATAPATH --dataset $DATASET \\
    --workers $NUM_WORKERS --print-freq 50 --batch-size $AA_BS --epsilon $EPS \\
    --atk-norm Linf --experiment none \\
    --clip-model {CLIP_MODEL} --clip-prompt-template {CLIP_TEMPLATE} \\
    --temperature {TEMPERATURE} \\
    --eval-attack #ATK# \\
    --arch #ARCH# \\
    --output-dir results/#SRC#"

# Number of adversarial examples to generate per training sample

"""

if TRAIN_MULT > 0:
    _BASE_SCRIPT += f"$GEN_ADV --num-dataset-copies {TRAIN_MULT} --save-adv-partition train \n"
if VAL_MULT > 0:
    _BASE_SCRIPT += f"$GEN_ADV --num-dataset-copies {VAL_MULT} --save-adv-partition val \n"
if TEST_MULT > 0:
    _BASE_SCRIPT += f"$GEN_ADV --num-dataset-copies {TEST_MULT} --save-adv-partition test \n"


def main():
    """Submit slurm jobs for transfer attacks evaluation."""
    num_jobs = 0
    for src in SRC:
        for atk in ATTACK:
            arch = src.split("_")[1]
            script = _BASE_SCRIPT.replace("#SRC#", src).strip()
            script = script.replace("#ATK#", atk).strip()
            script = script.replace("#ARCH#", arch).strip()
            batch_size = 64 if "huggingface" in src else 128
            batch_size *= NUM_GPUS
            script = script.replace("#BATCH_SIZE#", batch_size).strip()
            hash_id = f"{hash(script) % 100000:05d}"

            script_path = f"tmp/tmp_cifar10_gen_{atk}_{src}_{hash_id}.sh"
            script = script.replace("#SCRIPT_PATH#", script_path).strip()

            with open(script_path, "w", encoding="utf-8") as file:
                file.write(script)
            print(f"Submitting job for {src} at {script_path}")
            output = subprocess.run(["sbatch", script_path], check=True)
            time.sleep(1)
            if output.returncode != 0:
                print(f"Failed to submit job for {src}")
                sys.exit(1)
            num_jobs += 1
    print(f"All {num_jobs} jobs submitted successfully!")


if __name__ == "__main__":
    main()

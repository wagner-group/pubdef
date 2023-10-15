"""Submit slurm jobs for transfer attacks evaluation."""

import subprocess
import sys
import time

NUM_GPUS: int = 1
# Clip parameters
TEMPERATURE = 1.0
CLIP_MODEL = "RN50"
CLIP_TEMPLATE = "laion"

# Target models
TGT: list[str] = [
    "cifar10_resnet18_none_ce_seed0_lr0.1_wd0.0005_sgd",
]

# Source models
SRC: list[str] = [
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
    # f"cifar10_clip_{CLIP_MODEL.replace('/', '')}_{CLIP_TEMPLATE}",
    # "cifar10_resnet18_none_mat_seed0_src-f59e_atk-b06e_path-dc6b_lr0.1_wd0.0005_sgd_beta0.5_dw1.0-0.1-0.9_do0.0-0.0-0.0",
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

# Attacks
ATTACKS: list[str] = [
    # "pgd",
    # "mpgd",
    # "pregradient",
    # "di",
    # "ti",
    "admix",
    # "naa",
    # "ni-si-ti-dim",
    # "ni-admix-ti-dim",
]

_BASE_SCRIPT = f"""#!/bin/bash
#SBATCH --job-name=transfer-defense
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={8 * NUM_GPUS}
#SBATCH --gpus={NUM_GPUS}
#SBATCH --time=48:00:00
#SBATCH --output %j-cifar10-eval-#TGT#-hash#HASH#-#SRC#.log

GPU=0
NUM_GPU={NUM_GPUS}
BS=#BATCH_SIZE#  # batch sisze per GPU
PORT=1001$ID
BACKEND="nccl"
NUM_WORKERS={8 * NUM_GPUS}
DATASET="cifar10"
DATAPATH="$HOME/data/"
EPS=0.03137254901

source $HOME/.bashrc
source activate transfer-python3.10

echo #SCRIPT_PATH#

python -u transfer_exp_main.py \\
--no-distributed \\
--seed 0 --full-precision --data-dir "$DATAPATH" \\
--dataset $DATASET --workers $NUM_WORKERS --print-freq 50 \\
--batch-size $BS --epsilon $EPS --atk-norm "Linf" --experiment "none" \\
--clip-model {CLIP_MODEL} --clip-prompt-template {CLIP_TEMPLATE} \\
--temperature {TEMPERATURE} --no-orig-data \\
--transfer-attacks #ATK# \\
--tgt-models #TGT# --src-models #SRC# --debug
"""


def main() -> None:
    """Submit slurm jobs for transfer attacks evaluation."""
    num_jobs = 0
    for src in SRC:
        for tgt in TGT:
            if src == tgt:
                continue
            batch_size = 64 if "huggingface" in src else 128
            batch_size *= NUM_GPUS
            for atk in ATTACKS:
                # Update script
                script: str = _BASE_SCRIPT.replace("#SRC#", src)
                script = script.replace("#TGT#", tgt)
                script = script.replace("#BATCH_SIZE#", str(batch_size))
                script = script.replace("#ATK#", atk).strip()
                script_hash = str(hash(script) % 100000)
                script = script.replace("#HASH#", script_hash).strip()

                # Write script to file
                script_path: str = (
                    f"tmp/tmp_{tgt}_{src}_{atk}_hash{script_hash}.sh"
                )
                script = script.replace("#SCRIPT_PATH#", script_path).strip()
                with open(script_path, "w", encoding="utf-8") as file:
                    file.write(script)

                print(f"Submitting job for {src} at {script_path}")
                output = subprocess.run(["sbatch", script_path], check=True)
                num_jobs += 1
                time.sleep(1)
                if output.returncode != 0:
                    print(f"Failed to submit job for {src}")
                    sys.exit(1)

    print(f"All {num_jobs} jobs submitted successfully!")


if __name__ == "__main__":
    main()

#!/bin/bash
GPU=1
BS=128 # batch sisze per GPU
AA_BS=16 
ADV_BETA=0.5
NUM_WORKERS=8
DATASET="cifar10"
DATAPATH="$HOME/data/" # Set path to base data directory

# 0.0156862745, 0.03137254901, 0.06274509803
EPS=0.03137254901 # 8/255

SEED=0 # Random seed
TRAIN_MODE="none"
TRAIN_MODE2="pgd"
ARCH="resnet18"
# ARCH="efficientnetb0"
# ARCH="mobilenetv2"
# ARCH="dla"
# ARCH="resnext29-2x64d"
# ARCH="senet18"
#MODEL_NAME="cifar10_resnet18_none_ce_seed0_src-318d_atk-b06e_path-b34d_lr0.1_wd0.0005_sgd_dw1.0-1.0-1.0_do0.0-0.3-0.7"
#MODEL_NAME="cifar10_resnet18_none_ce_seed0_src-318d_atk-b06e_path-b34d_lr0.1_wd0.0005_sgd_dw1.0-1.0-1.0_do0.0-0.5-0.5"
#MODEL_NAME="cifar10_resnet18_none_ce_seed0_src-318d_atk-b06e_path-b34d_lr0.1_wd0.0005_sgd_dw1.0-1.0-1.0_do0.0-1.0-0.0"
#MODEL_NAME="cifar10_resnet18_none_ce_seed0_src-318d_atk-b06e_path-b34d_lr0.1_wd0.0005_sgd_dw1.0-1.0-1.0_do0.0-0.1-0.9"
#MODEL_NAME="cifar10_resnet18_none_ce_seed0_src-318d_atk-b06e_path-b34d_lr0.1_wd0.0005_sgd_dw1.0-1.0-1.0_do0.0-0.0-1.0"
#MODEL_NAME="cifar10_resnet18_none_ce_seed0_src-318d_atk-b06e_path-b34d_lr0.1_wd0.0005_sgd_dw1.0-1.0-1.0_do0.0-0.2-0.8"
#MODEL_NAME="cifar10_resnet18_none_ce_seed0_src-318d_atk-b06e_path-b34d_lr0.1_wd0.0005_sgd_dw1.0-1.0-1.0_do0.0-0.3-0.7"
#MODEL_NAME="cifar10_resnet18_none_ce_seed0_src-318d_atk-b06e_path-b34d_lr0.1_wd0.0005_sgd_dw1.0-1.0-1.0_do0.0-0.4-0.6"
#MODEL_NAME="cifar10_resnet18_none_ce_seed0_src-318d_atk-b06e_path-b34d_lr0.1_wd0.0005_sgd_dw1.0-1.0-1.0_do0.0-0.6-0.4"
#MODEL_NAME="cifar10_resnet18_none_ce_seed0_src-318d_atk-b06e_path-b34d_lr0.1_wd0.0005_sgd_dw1.0-1.0-1.0_do0.0-0.7-0.3"
#MODEL_NAME="cifar10_resnet18_none_ce_seed0_src-318d_atk-b06e_path-b34d_lr0.1_wd0.0005_sgd_dw1.0-1.0-1.0_do0.0-0.8-0.2"
#MODEL_NAME="cifar10_resnet18_none_ce_seed0_src-318d_atk-b06e_path-b34d_lr0.1_wd0.0005_sgd_dw1.0-1.0-1.0_do0.0-0.9-0.1"
#MODEL_NAME="cifar10_resnet18_none_ce_seed0_src-318d_atk-b06e_path-b34d_lr0.1_wd0.0005_sgd_dw1.0-1.0-1.0_do0.0-0.4-0.6"
#MODEL_NAME="cifar10_resnet18_none_ce_seed1_src-318d_atk-b06e_path-b34d_lr0.1_wd0.0005_sgd_dw1.0-1.0-1.0_do0.0-0.4-0.6"
#MODEL_NAME="cifar10_resnet18_none_ce_seed1_src-318d_atk-b06e_path-b34d_lr0.1_wd0.0005_sgd_dw1.0-0.0-1.0_do0.0-0.0-0.0"
#MODEL_NAME="cifar10_resnet18_none_ce_seed1_src-318d_atk-b06e_path-b34d_lr0.1_wd0.0005_sgd_dw1.0-1.0-0.0_do0.0-0.0-0.0"
#MODEL_NAME="cifar10_resnet18_none_mat_seed1_src-318d_atk-b06e_path-b34d_lr0.1_wd0.0005_sgd_beta0.5_dw1.0-1.0-1.0_do0.0-0.0-0.0"
#MODEL_NAME="cifar10_resnet18_none_ce_seed1_src-318d_atk-b06e_path-b34d_lr0.1_wd0.0005_sgd_dw1.0-0.5-0.5_do0.0-0.0-0.0"
#MODEL_NAME="cifar10_resnet18_none_mat_seed1_src-318d_atk-b06e_path-b34d_lr0.1_wd0.0005_sgd_beta0.5_dw1.0-1.0-0.0_do0.0-0.0-0.0"
#MODEL_NAME="cifar10_resnet18_none_mat_seed1_src-318d_atk-b06e_path-b34d_lr0.1_wd0.0005_sgd_beta0.5_dw1.0-0.0-1.0_do0.0-0.0-0.0"
#SRC_NAME="cifar10_resnet18_none_seed1"
#SRC_NAME2="cifar10_resnet18_pgd_seed2"
#MODEL_NAME="cifar10_resnet18_none_ce_seed0_src-c8f7_atk-b06e_path-f0b8_lr0.1_wd0.0005_sgd_dw1.0-1.0-1.0_do0.0-0.0-0.0"
#MODEL_NAME="cifar10_resnet18_none_ce_seed1_src-c8f7_atk-b06e_path-f0b8_lr0.1_wd0.0005_sgd_dw1.0-1.0-1.0_do0.0-0.0-0.0"
#MODEL_NAME="cifar10_resnet18_none_ce_seed2_src-c8f7_atk-b06e_path-f0b8_lr0.1_wd0.0005_sgd_dw1.0-1.0-1.0_do0.0-0.0-0.0"
#MODEL_NAME="cifar10_resnet18_none_ce_seed3_src-c8f7_atk-b06e_path-f0b8_lr0.1_wd0.0005_sgd_dw1.0-1.0-1.0_do0.0-0.0-0.0"
#MODEL_NAME="cifar10_resnet18_pgd_seed2"
#MODEL_NAME="cifar10_resnet18_none_ce_seed3_src-a5e4_atk-b06e_bs128_lr0.1_wd0.0005_sgd_50ep_dw1.0-1.0-1.0_do0.0-0.0-0.0"
#MODEL_NAME="cifar10_resnet18_none_ce_seed3_src-4564_atk-b06e_bs128_lr0.1_wd0.0005_sgd_50ep_dw1.0-0.0-1.0_do0.0-0.0-0.0"
#MODEL_NAME="cifar10_resnet18_none_ce_seed3_src-c8f7_atk-b06e_bs128_lr0.1_wd0.0005_sgd_50ep_dw1.0-1.0-1.0_do0.0-0.0-0.0"
MODEL_NAME="cifar10_resnet18_none_ce_seed3_src-c8f7_atk-b06e_bs128_lr0.1_wd0.0005_sgd_50ep_dw1.0-1.0-1.0_do0.0-0.0-0.0"
SRC_NAME="cifar10_resnet18_none_seed9"
SRC_NAME2="cifar10_resnet18_pgd_seed9"

ATTACK="pgd" # Attack used to generate adversarial examples

echo "(5) Testing against other transfer attacks..."

# Can also replace --src-models with other models
# SRC_NAME=...
# Target model from Step (4). Name is autogenerated with hash of source model
# and attack name. Step (4) will print the name of the target model.
TGT_MODEL=$MODEL_NAME

CUDA_VISIBLE_DEVICES=$GPU python -u transfer_exp_main.py --no-distributed \
    --seed $SEED --full-precision --data-dir "$DATAPATH" \
    --dataset $DATASET --workers $NUM_WORKERS --print-freq 50 \
    --batch-size "$AA_BS" --epsilon $EPS --atk-norm "Linf" --experiment "none" \
    --transfer-attacks pgd mpgd pregradient di ti admix naa iaa ni-si-ti-dim ni-admix-ti-dim \
    --no-orig-data \
    --src-models "$SRC_NAME2" \
    --tgt-models $TGT_MODEL

echo "Finished."



#!/bin/bash
GPU=0
BS=32 # batch sisze per GPU
AA_BS=16 
ADV_BETA=0.5
NUM_WORKERS=8
DATASET="cifar10"
DATAPATH="$HOME/data/" # Set path to base data directory

# 0.0156862745, 0.03137254901, 0.06274509803
EPS=0.03137254901 # 8/255

SEED=3 # Random seed
TRAIN_MODE="none"
TRAIN_MODE2="pgd"
ARCH="resnet18"
ARCH2="resnet18"
ARCH3="resnet18"
# ARCH="dla"
# ARCH="resnext29-2x64d"
# ARCH="senet18"
#MODEL_NAME="cifar10_resnet18_none_seed0"
MODEL_NAME="cifar10_resnet18_none_ce_seed3_src-c8f7_atk-b06e_bs128_lr0.1_wd0.0005_sgd_50ep_dw1.0-1.0-1.0_do0.0-0.0-0.0"
#MODEL_NAME="cifar10_resnet18_none_mat_seed1_src-318d_atk-b06e_path-b34d_lr0.1_wd0.0005_sgd_beta0.5_dw1.0-1.0-1.0_do0.0-0.0-0.0"
#MODEL_NAME="cifar10_resnet18_none_ce_seed1_src-318d_atk-b06e_path-b34d_lr0.1_wd0.0005_sgd_dw1.0-1.0-0.0_do0.0-0.0-0.0"
#SRC_NAME="cifar10_resnet18_none_seed1"
#SRC_NAME2="cifar10_resnet18_pgd_seed2"
#MODEL_NAME="cifar10_resnet18_none_ce_seed8_lr0.1_wd0.0005_sgd"
#MODEL_NAME="cifar10_resnet18_none_ce_seed0_src-c8f7_atk-b06e_path-f0b8_lr0.1_wd0.0005_sgd_dw1.0-1.0-1.0_do0.0-0.0-0.0"
#MODEL_NAME="cifar10_resnet18_none_ce_seed8_lr0.1_wd0.0005_sgd"
SRC_NAME="cifar10_resnet18_none_seed9"
SRC_NAME2="cifar10_resnet18_pgd_seed9"

ATTACK="pgd" # Attack used to generate adversarial examples

echo "(4) Training a target model with transfer defense..."
# Resume from original model in Step (1)
#RESUME=""results/$MODEL_NAME/checkpoint_best.pt""
#TRAIN_MODE="mat" # Mixed Adversarial Training for training with defense

# Training with transfer defense
CUDA_VISIBLE_DEVICES=$GPU python main.py --no-distributed \
--full-precision --pretrained --data-dir $DATAPATH --dataset $DATASET \
--workers $NUM_WORKERS --print-freq 50 --epochs 50 --batch-size $BS \
--lr 1e-1 --wd 5e-4 --epsilon $EPS --atk-norm Linf --experiment none \
--adv-beta $ADV_BETA --load-weight-only --ensemble-type "trs" \
--arch $ARCH $ARCH2 $ARCH3\
--seed $SEED \
--train-attack "pgd" \
--train-loss "mat" \
--num-dataset-copies 1 \

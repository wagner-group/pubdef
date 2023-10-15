#!/bin/bash
GPU=0
BS=128 # batch sisze per GPU
AA_BS=16 
ADV_BETA=6
NUM_WORKERS=8
DATASET="cifar10"
DATAPATH="$HOME/data/" # Set path to base data directory

# 0.0156862745, 0.03137254901, 0.06274509803
EPS=0.03137254901 # 8/255

SEED=12 # Random seed
TRAIN_MODE="none"
ARCH="resnet18"
# ARCH="efficientnetb0"
# ARCH="mobilenetv2"
# ARCH="dla"
# ARCH="resnext29-2x64d"
# ARCH="senet18"
MODEL_NAME="${DATASET}_${ARCH}_${TRAIN_MODE}_seed${SEED}"
#MODEL_NAME="cifar10_resnet18_none_ce_seed3_src-c8f7_atk-b06e_bs128_lr0.1_wd0.0005_sgd_50ep_dw1.0-1.0-1.0_do0.0-0.0-0.0"

ATTACK="pgd" # Attack used to generate adversarial examples

echo "(1) Training an original model..."

# Normal training
CUDA_VISIBLE_DEVICES=$GPU python -u main.py --no-distributed \
    --full-precision --pretrained --data-dir "$DATAPATH" --dataset $DATASET \
    --workers $NUM_WORKERS --print-freq 50 --epochs 50 --batch-size $BS \
    --lr 1e-1 --wd 5e-4 --epsilon $EPS --atk-norm Linf --experiment none \
    --seed $SEED --arch $ARCH \
    --train-attack $TRAIN_MODE \
    --train-loss "ce"
    --output-dir "results/$MODEL_NAME"
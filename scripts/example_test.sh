#!/bin/bash
GPU=0                  # GPU to use
BS=128                 # batch sisze per GPU
NUM_WORKERS=2          # Number of data loading workers
DATAPATH="$HOME/data/" # Set path to base data directory
SEED=0                 # Random seed

# Below are example configs for the three datasets.
# Comment out the two you don't want to run.

DATASET="cifar10"
EPS=0.03137254901 # 8/255
ATK_STEPS=100
TGT_MODEL="cifar10_wideresnet34-10_pubdef" # Use PubDef
# TGT_MODEL="cifar10_huggingface-convnext"  # Use HuggingFace model
SRC_MODEL="cifar10_huggingface-vit-base"

DATASET="cifar100"
EPS=0.03137254901 # 8/255
ATK_STEPS=100
TGT_MODEL="cifar100_wideresnet34-10_pubdef" # Use PubDef
# TGT_MODEL="cifar100_robustbench-rade21-rn-18"  # Use RobustBench model
SRC_MODEL="cifar100_huggingface-swin-tiny"

DATASET="imagenet"
EPS=0.0156862745 # 4/255
ATK_STEPS=50
TGT_MODEL="imagenet_resnet50_pubdef" # Use PubDef
# TGT_MODEL="imagenet_timm-mobilenet-v3-l"  # Use timm model
SRC_MODEL="imagenet_huggingface-swin-t"

# Run evaluation on clean data
CUDA_VISIBLE_DEVICES=$GPU python -u transfer_exp_main.py --no-distributed \
    --seed $SEED --full-precision --data-dir "$DATAPATH" \
    --dataset $DATASET --workers $NUM_WORKERS --print-freq 50 \
    --batch-size $BS --epsilon $EPS --atk-norm "Linf" --experiment "none" \
    --eval-atk-steps $ATK_STEPS --data-load-mode "orig" \
    --transfer-attacks "mpgd" \
    --src-models $SRC_MODEL \
    --tgt-models $TGT_MODEL

# Run evaluation on mpgd attack
CUDA_VISIBLE_DEVICES=$GPU python -u transfer_exp_main.py --no-distributed \
    --seed $SEED --full-precision --data-dir "$DATAPATH" \
    --dataset $DATASET --workers $NUM_WORKERS --print-freq 50 \
    --batch-size $BS --epsilon $EPS --atk-norm "Linf" --experiment "none" \
    --eval-atk-steps $ATK_STEPS --data-load-mode "load" \
    --transfer-attacks "mpgd" \
    --src-models $SRC_MODEL \
    --tgt-models $TGT_MODEL

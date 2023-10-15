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

SEED=10 # Random seed
TRAIN_MODE="pgd"
ARCH="resnet18"
# ARCH="efficientnetb0"
# ARCH="mobilenetv2"
# ARCH="dla"
# ARCH="resnext29-2x64d"
# ARCH="senet18"
ATTACK="pgd mpgd pregradient di ti admix naa iaa ni-si-ti-dim ni-admix-ti-dim" # Attack used to generate adversarial examples

echo "(4) Training a normally-trained source model..."

# Our source model here is just the same model but with different seed
SRC_NAME2="${DATASET}_${ARCH}_${TRAIN_MODE}_seed${SEED}"
#SRC_NAME2="cifar10_resnet18_pgd_seed9"

#CUDA_VISIBLE_DEVICES=$GPU python -u main.py --no-distributed \
    --full-precision --pretrained --data-dir "$DATAPATH" --dataset $DATASET \
    --workers $NUM_WORKERS --print-freq 50 --epochs 50 --batch-size $BS \
    --lr 1e-1 --wd 5e-4 --epsilon $EPS --atk-norm "Linf" --experiment "none" \
    --seed $SEED --arch $ARCH \
    --train-attack "pgd" \
    --train-loss "ce" \
    --output-dir results/$SRC_NAME2

echo "(3) Generating and saving adversarial examples from source model..."

GEN_ADV="python -u save_adv_main.py --no-distributed \
    --seed 6 --full-precision --data-dir $DATAPATH --dataset $DATASET \
    --workers $NUM_WORKERS --print-freq 50 --batch-size $AA_BS --epsilon $EPS \
    --atk-norm Linf --experiment none --eval-attack $ATTACK \
    --arch $ARCH --output-dir results/$SRC_NAME2"

# Number of adversarial examples to generate per training sample
TRAIN_MULT=1
#CUDA_VISIBLE_DEVICES=$GPU $GEN_ADV --num-dataset-copies $TRAIN_MULT --save-adv-partition "train"
#CUDA_VISIBLE_DEVICES=$GPU $GEN_ADV --num-dataset-copies 1 --save-adv-partition "val"
CUDA_VISIBLE_DEVICES=$GPU $GEN_ADV --num-dataset-copies 1 --save-adv-partition "test"


#!/bin/bash
GPU=0
BS=128
ATTACK="pgd"
AA_BS=16
ADV_BETA=0.5
NUM_WORKERS=8
DATASET="cifar10"
DATAPATH="$HOME/data/" # Set path to base data directory

# 0.0156862745, 0.03137254901, 0.06274509803
EPS=0.03137254901 # 8/255


ARCH="resnet18"
ARCH2="resnet18"
ARCH3="resnet18"

SEED=3 # Random seed

ATTACK="pgd" # Attack used to generate adversarial examples

echo "(4) Training a target model with transfer defense..."
# Resume from original model in Step (1)
#RESUME=""results/$MODEL_NAME/checkpoint_best.pt""
#TRAIN_MODE="mat" # Mixed Adversarial Training for training with defense

# Training with transfer defense
CUDA_VISIBLE_DEVICES=$GPU python main.py --no-distributed \
--full-precision --pretrained --data-dir $DATAPATH --dataset $DATASET \
--workers $NUM_WORKERS --print-freq 50 --epochs 200 --batch-size $BS \
--lr 0.1 --epsilon $EPS --atk-norm Linf --experiment none \
--ensemble-type "dverge" \
--lr-schedule "step" --lr-steps 100 150 \
--lr-step-size 0.1 \
--optim "sgd" \
--arch $ARCH $ARCH2 $ARCH3 \
--seed $SEED \
--num-dataset-copies 1 \
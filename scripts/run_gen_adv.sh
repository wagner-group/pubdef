#!/bin/bash
BS=2
NUM_WORKERS=8
DATASET="cifar10"
DATAPATH="$HOME/data/"
EPS=0.03137254901
ARCH="robustbench-kireev21-rn-18"

run_split() {
    CUDA_VISIBLE_DEVICES=0 python -u save_adv_main.py --no-distributed \
        --seed 0 --full-precision --data-dir "$DATAPATH" --dataset $DATASET \
        --workers $NUM_WORKERS --print-freq 50 --batch-size $BS --epsilon $EPS \
        --atk-norm Linf --experiment none \
        --temperature 1.0 \
        --eval-attack pgd \
        --augments noaug \
        --arch $ARCH \
        --output-dir results/${DATASET}_${ARCH} \
        --save-adv-partition "$1" \
        --num-dataset-copies "$2"
}

run_split train 1
run_split val 1
run_split test 1

#!/bin/bash
GPU=0
BS=128 # batch sisze per GPU
NUM_WORKERS=2
DATASET="cifar10"
DATAPATH="$HOME/data/" # Set path to base data directory
RESULT_DIR="results"

# 0.0156862745, 0.03137254901, 0.06274509803
EPS=0.03137254901 # 8/255

SEED=0 # Random seed
ARCH="resnet18"
# ARCH="efficientnetb0"
# ARCH="mobilenetv2"
# ARCH="dla"
# ARCH="resnext29-2x64d"
# ARCH="senet18"
MODEL_NAME="${DATASET}_${ARCH}_seed${SRC_SEED}"

ATTACK="pgd" # Attack used to generate adversarial examples

echo "(1) Training an original model..."

# --train-attack none: "none" means normal training (i.e., not running
#                      Madry et al. adversarial training).
# --train-loss ce: Use cross-entropy loss.

# Normal training
CUDA_VISIBLE_DEVICES=$GPU python -u main.py --no-distributed \
    --full-precision --pretrained --data-dir "$DATAPATH" --dataset $DATASET \
    --workers $NUM_WORKERS --print-freq 50 --epochs 5 --batch-size $BS \
    --lr 1e-1 --wd 5e-4 --epsilon $EPS --atk-norm Linf --experiment none \
    --lr-steps 60 120 160 --lr-schedule step --lr-step-size 0.2 \
    --seed $SEED --arch $ARCH --train-attack none --train-loss ce \
    --output-dir "$RESULT_DIR/$MODEL_NAME"



echo "(2) Training a source model..."
# This step can be skipped if you are using public models.
# To skip this step, specify the name of the source model in Step (3) as arch,
# i.e., "--arch cifar10_robustbench-wang23-wrn-70-16", and set output_dir
# accordingly, e.g., --output-dir results/cifar10_robustbench-wang23-wrn-70-16.
# See src/models/robustbench_model.py, src/models/huggingface_model.py for the
# complete list of included models.

# Our source model here is just the same model but with different seed
SRC_SEED=1
SRC_NAME="${DATASET}_${ARCH}_seed${SRC_SEED}"

CUDA_VISIBLE_DEVICES=$GPU python -u main.py --no-distributed \
    --full-precision --pretrained --data-dir "$DATAPATH" --dataset $DATASET \
    --workers $NUM_WORKERS --print-freq 50 --epochs 5 --batch-size $BS \
    --lr 1e-1 --wd 5e-4 --epsilon $EPS --atk-norm "Linf" --experiment "none" \
    --seed $SRC_SEED --arch $ARCH --train-attack none --train-loss ce \
    --output-dir "$RESULT_DIR/$SRC_NAME"



echo "(3) Generating and saving adversarial examples from source model..."

gen_adv () {
    CUDA_VISIBLE_DEVICES=$GPU python -u save_adv_main.py --no-distributed \
        --seed 0 --full-precision --data-dir "$DATAPATH" --dataset $DATASET \
        --workers $NUM_WORKERS --print-freq 50 --batch-size $BS \
        --epsilon $EPS --atk-norm Linf --experiment none --eval-attack $ATTACK \
        --output-dir $RESULT_DIR/$SRC_NAME \
        --arch $ARCH \
        --eval-atk-steps $3 \
        --num-dataset-copies $1 \
        --save-adv-partition $2
}

# Generate one copies of adversarial examples for training, validation and testing
gen_adv 4 "train" 10
gen_adv 1 "val" 10
# This will be used for evaluation in Step (5). We use 100 steps for test set
gen_adv 1 "test" 100


echo "(4) Training a target model with transfer defense..."

# Optionally resume from original model in Step (1)
RESUME="$RESULT_DIR/$MODEL_NAME/checkpoint_best.pt"

# Arguments
# --load-weight-only: Load weights only from the checkpoint
# --augments default: Use default data augmentation (pad and crop).
# --data-load-mode mix: Mix training adversarial examples from all sources
#                       together. Clean samples are always separted.
# --dataset-weights: Weights for each dataset. This can be a list of floats for
#                    each of the sources of adversarial examples specified by
#                    src_models and transfer_attacks. The first number is
#                    reserved for the clean dataset.
# --dataset-dropout: Dropout rate for each dataset. Similar to dataset_weights,
#                    this can be a list of floats for each of the sources. Used
#                    to randomly drop samples from each source.
# --num-dataset-copies: Number of copies of adversarial examples for each
#                       dataset. If larger than existing number of copies, load
#                       all copies available.

# There are two ways to specify the adversarial examples to train against:
# Option 1:
# Specify --train-load-paths, --val-load-paths, and --test-load-paths pointing
# to directories containing the generated adversarial examples from Step (3).
# This is used in the example below. Each argument can take a list of strings.
# This option is clear and simple, and it is the recommended way. The downside
# is it can get really long when there are many sources of adversarial examples.
# Option 2:
# --src-models: Source model(s) of the adversarial examples to train the defense
#               against. This comes from Step (3).
# --transfer-attacks: Transfer attacks used to generate adversarial examples to
#                     train the defense against. This comes from Step (3).
#                     src_models and transfer_attacks should be specified
#                     togethe, and they can take a list of strings.
# --src-models "$SRC_NAME" \
# --transfer-attacks "$ATTACK" \

# Training with transfer defense
CUDA_VISIBLE_DEVICES=$GPU python -u main.py --no-distributed \
    --full-precision --pretrained --data-dir "$DATAPATH" --dataset $DATASET \
    --workers $NUM_WORKERS --print-freq 50 --epochs 5 --batch-size $BS \
    --lr 1e-1 --wd 5e-4 --epsilon $EPS --atk-norm Linf --experiment none \
    --load-weight-only \
    --resume "$RESUME" \
    --seed $SEED \
    --arch $ARCH \
    --train-attack none \
    --train-loss ce \
    --augments default \
    --data-load-mode mix \
    --train-load-paths "$RESULT_DIR/$SRC_NAME"/saved_pgd_train_temp1.0_10step \
    --test-load-paths "$RESULT_DIR/$SRC_NAME"/saved_pgd_test_temp1.0_10step \
    --val-load-paths "$RESULT_DIR/$SRC_NAME"/saved_pgd_val_temp1.0_10step \
    --dataset-weights 1.0 1.0 \
    --dataset-dropout 0.0 0.0 \
    --num-dataset-copies 4



echo "(5) Testing against other transfer attacks..."

# This step assumes that you already have adversarial examples generated from
# different source models and attacjs like in Step (3).
# Other choices of attacks: mpgd pregradient di ti admix naa iaa ni-si-ti-dim
# ni-admix-ti-dim auto-pgd-dlr auto-pgd-ce

# Can also replace --src-models with other models
# SRC_NAME=...
# Target model from Step (4). Name is autogenerated with hash of source model
# and attack name. Step (4) will print the name of the target model.
TGT_MODEL="cifar10_resnet18_none_ce_seed0_path-43ad_bs128_lr0.1_wd0.0005_sgd_5ep_dw1.0-1.0_do0.0-0.0_mix_copy1"


CUDA_VISIBLE_DEVICES=$GPU python -u transfer_exp_main.py --no-distributed \
    --seed $SEED --full-precision --data-dir "$DATAPATH" \
    --dataset $DATASET --workers $NUM_WORKERS --print-freq 50 \
    --batch-size $BS --epsilon $EPS --atk-norm "Linf" --experiment "none" \
    --tgt-models $TGT_MODEL \
    --transfer-attacks pgd \
    --src-models $SRC_NAME

echo "Finished."

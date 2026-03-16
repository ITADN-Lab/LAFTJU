#!/bin/bash
# Re-run ATJU experiments that failed due to parameter name bug
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

SEED=42
EPOCHS=200
BATCH=128
SAVE_DIR="./results"
DATA_DIR="./dataset"
LOG="./results/experiment_log_atju_rerun.txt"

echo "============================================" | tee "$LOG"
echo "ATJU Re-run (fixed parameter names)" | tee -a "$LOG"
echo "Start: $(date)" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"

for DATASET in cifar10 cifar100; do
    echo "" | tee -a "$LOG"
    echo ">>> $DATASET / ResNet18 / ATJU — $(date)" | tee -a "$LOG"
    python train_laktju.py \
        --dataset $DATASET --model resnet18 --optimizer ATJU \
        --epochs $EPOCHS --batch_size $BATCH --seed $SEED \
        --save_dir $SAVE_DIR --data_dir $DATA_DIR --workers 4 2>&1 | tee -a "$LOG"
    echo "<<< $DATASET / ATJU done — $(date)" | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"
echo "ATJU re-run complete: $(date)" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"

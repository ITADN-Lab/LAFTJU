#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

SEED=42
EPOCHS=200
BATCH=128
SAVE_DIR="./results"
DATA_DIR="./dataset"
LOG="./results/experiment_log.txt"

mkdir -p "$SAVE_DIR"

echo "============================================" | tee "$LOG"
echo "LAKTJU Experiment Suite" | tee -a "$LOG"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)" | tee -a "$LOG"
echo "Start: $(date)" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"

for DATASET in cifar10 cifar100; do
    for OPT in SGD Adam AdamW ATJU LAKTJU; do
        echo "" | tee -a "$LOG"
        echo ">>> $DATASET / ResNet18 / $OPT — $(date)" | tee -a "$LOG"
        python train_laktju.py \
            --dataset $DATASET --model resnet18 --optimizer $OPT \
            --epochs $EPOCHS --batch_size $BATCH --seed $SEED \
            --save_dir $SAVE_DIR --data_dir $DATA_DIR --workers 4 2>&1 | tee -a "$LOG"
        echo "<<< $DATASET / $OPT done — $(date)" | tee -a "$LOG"
    done
done

echo "" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"
echo "All experiments complete: $(date)" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"

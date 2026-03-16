#!/bin/bash
# LAKTJU vs Baselines: CIFAR-10 + CIFAR-100 with ResNet18
# Run all optimizers sequentially on single GPU

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

SEED=42
EPOCHS=200
BATCH=128
SAVE_DIR="./results"
DATA_DIR="./dataset"

echo "============================================"
echo "LAKTJU Experiment Suite"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"
echo "============================================"

# --- CIFAR-10 ResNet18 ---
for OPT in SGD Adam AdamW ATJU LAKTJU; do
    echo ""
    echo ">>> CIFAR-10 / ResNet18 / $OPT"
    python train_laktju.py \
        --dataset cifar10 --model resnet18 --optimizer $OPT \
        --epochs $EPOCHS --batch_size $BATCH --seed $SEED \
        --save_dir $SAVE_DIR --data_dir $DATA_DIR --workers 4
done

# --- CIFAR-100 ResNet18 ---
for OPT in SGD Adam AdamW ATJU LAKTJU; do
    echo ""
    echo ">>> CIFAR-100 / ResNet18 / $OPT"
    python train_laktju.py \
        --dataset cifar100 --model resnet18 --optimizer $OPT \
        --epochs $EPOCHS --batch_size $BATCH --seed $SEED \
        --save_dir $SAVE_DIR --data_dir $DATA_DIR --workers 4
done

echo ""
echo "============================================"
echo "All experiments complete: $(date)"
echo "============================================"

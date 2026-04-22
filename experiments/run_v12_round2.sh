#!/bin/bash
# =============================================================================
# LAKTJU V12 Round 2: Full 200-epoch Training
# Based on Round 1 results: lr=0.01 best (71.39%), lr=0.005 second (69.47%)
# Multi-seed + SAM + additional LR points
# 10 parallel jobs to maximize RTX 5090 utilization
# =============================================================================

set -e
cd "$(dirname "$0")"

RESULTS_DIR="./results/v12_round2"
mkdir -p "$RESULTS_DIR"

MAX_PARALLEL=10
EPOCHS=200
DATASET_DIR="./dataset"

job_count=0

run_job() {
    local cmd="$1"
    local name="$2"
    echo "[Launch] $name"
    eval "$cmd" > "${RESULTS_DIR}/${name}.log" 2>&1 &
    job_count=$((job_count + 1))
    if [ "$job_count" -ge "$MAX_PARALLEL" ]; then
        wait -n 2>/dev/null || wait
        job_count=$((job_count - 1))
    fi
}

echo "============================================"
echo "  LAKTJU V12 Round 2: 200ep Full Training"
echo "  Max parallel: $MAX_PARALLEL"
echo "  Results: $RESULTS_DIR"
echo "============================================"

# Round 1 winner: lr=0.01, wd=5e-4, kf_warmup=300
BEST_LR=0.01
BEST_WD=5e-4
BEST_KFW=300
BEST_HS=2.0

# -----------------------------------------------
# Group A: Best config (lr=0.01) multi-seed, both datasets
# -----------------------------------------------
for SEED in 42 123 456; do
    NAME="c10_v12_lr01_s${SEED}"
    run_job "python train_laktju.py \
        --optimizer LAKTJU_V12 --dataset cifar10 --model resnet18 \
        --lr $BEST_LR --weight_decay $BEST_WD --epochs $EPOCHS --seed $SEED \
        --label_smoothing 0.1 --homotopy_speed $BEST_HS --kf_warmup $BEST_KFW \
        --kf_damping 1e-3 --kf_clip_max 50 --warmup 20 \
        --save_dir $RESULTS_DIR --data_dir $DATASET_DIR" "$NAME"

    NAME="c100_v12_lr01_s${SEED}"
    run_job "python train_laktju.py \
        --optimizer LAKTJU_V12 --dataset cifar100 --model resnet18 \
        --lr $BEST_LR --weight_decay $BEST_WD --epochs $EPOCHS --seed $SEED \
        --label_smoothing 0.1 --homotopy_speed $BEST_HS --kf_warmup $BEST_KFW \
        --kf_damping 1e-3 --kf_clip_max 50 --warmup 20 \
        --save_dir $RESULTS_DIR --data_dir $DATASET_DIR" "$NAME"
done

# -----------------------------------------------
# Group B: lr=0.005 (stable runner-up) multi-seed
# -----------------------------------------------
for SEED in 42 123; do
    NAME="c100_v12_lr005_s${SEED}"
    run_job "python train_laktju.py \
        --optimizer LAKTJU_V12 --dataset cifar100 --model resnet18 \
        --lr 0.005 --weight_decay $BEST_WD --epochs $EPOCHS --seed $SEED \
        --label_smoothing 0.1 --homotopy_speed $BEST_HS --kf_warmup $BEST_KFW \
        --kf_damping 1e-3 --kf_clip_max 50 --warmup 20 \
        --save_dir $RESULTS_DIR --data_dir $DATASET_DIR" "$NAME"
done

# -----------------------------------------------
# Group C: V12 + SAM (rho=0.05) with best lr
# -----------------------------------------------
for SEED in 42 123; do
    NAME="c100_v12_sam_s${SEED}"
    run_job "python train_laktju.py \
        --optimizer LAKTJU_V12 --dataset cifar100 --model resnet18 \
        --lr $BEST_LR --weight_decay $BEST_WD --epochs $EPOCHS --seed $SEED \
        --label_smoothing 0.1 --homotopy_speed $BEST_HS --kf_warmup $BEST_KFW \
        --kf_damping 1e-3 --kf_clip_max 50 --warmup 20 \
        --sam_rho 0.05 \
        --save_dir $RESULTS_DIR --data_dir $DATASET_DIR" "$NAME"
done

echo ""
echo "Waiting for all Round 2 jobs to finish..."
wait
echo "============================================"
echo "  Round 2 complete! Results:"
echo "============================================"

echo ""
echo "=== ROUND 2 RESULTS ==="
for f in ${RESULTS_DIR}/*.log; do
    if [ -f "$f" ]; then
        name=$(basename "$f" .log)
        best_valid=$(grep "Best Valid Acc" "$f" 2>/dev/null | grep -oP '[0-9.]+%' || echo "N/A")
        best_test=$(grep "Best Test Acc" "$f" 2>/dev/null | grep -oP '[0-9.]+%' || echo "N/A")
        echo "  $name: valid=$best_valid test=$best_test"
    fi
done

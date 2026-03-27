#!/bin/bash
# =============================================================================
# LAKTJU V13 Round 2: 正确学习率 lr=0.01 (V8最佳配置)
# Round 1用了lr=0.001（太低），Round 2修正为lr=0.01
# 目标: CIFAR-10 ≥96.2%, CIFAR-100 ≥78.3%
# =============================================================================

set -e
cd "$(dirname "$0")"

RESULTS_DIR="./results/v13_round2"
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
echo "  LAKTJU V13 Round 2: lr=0.01 正确配置"
echo "  目标: C10≥96.2%, C100≥78.3%"
echo "  Max parallel: $MAX_PARALLEL"
echo "============================================"

# V8默认最佳: tju_lr=0.01, a_lr=0.001 (ratio=0.1), wd=1e-4, hs=2.0, kf_damping=1e-3

# -----------------------------------------------
# Group A: V8 lr=0.01 baseline (多种子)
# -----------------------------------------------
for SEED in 42 123 456; do
    NAME="c10_lr01_s${SEED}"
    run_job "python train_laktju.py \
        --optimizer LAKTJU --dataset cifar10 --model resnet18 \
        --lr 0.01 --a_lr_ratio 0.1 --weight_decay 1e-4 \
        --epochs $EPOCHS --seed $SEED \
        --label_smoothing 0.1 --homotopy_speed 2.0 --kf_damping 1e-3 \
        --warmup 20 --c_base 1.0 --kappa 5.0 \
        --save_dir $RESULTS_DIR --data_dir $DATASET_DIR" "$NAME"

    NAME="c100_lr01_s${SEED}"
    run_job "python train_laktju.py \
        --optimizer LAKTJU --dataset cifar100 --model resnet18 \
        --lr 0.01 --a_lr_ratio 0.1 --weight_decay 1e-4 \
        --epochs $EPOCHS --seed $SEED \
        --label_smoothing 0.1 --homotopy_speed 2.0 --kf_damping 1e-3 \
        --warmup 20 --c_base 1.0 --kappa 5.0 \
        --save_dir $RESULTS_DIR --data_dir $DATASET_DIR" "$NAME"
done

# -----------------------------------------------
# Group B: V8 lr=0.01 + SAM (rho=0.05)
# -----------------------------------------------
for SEED in 42 123; do
    NAME="c10_lr01_sam_s${SEED}"
    run_job "python train_laktju.py \
        --optimizer LAKTJU --dataset cifar10 --model resnet18 \
        --lr 0.01 --a_lr_ratio 0.1 --weight_decay 1e-4 \
        --epochs $EPOCHS --seed $SEED \
        --label_smoothing 0.1 --homotopy_speed 2.0 --kf_damping 1e-3 \
        --warmup 20 --sam_rho 0.05 \
        --save_dir $RESULTS_DIR --data_dir $DATASET_DIR" "$NAME"

    NAME="c100_lr01_sam_s${SEED}"
    run_job "python train_laktju.py \
        --optimizer LAKTJU --dataset cifar100 --model resnet18 \
        --lr 0.01 --a_lr_ratio 0.1 --weight_decay 1e-4 \
        --epochs $EPOCHS --seed $SEED \
        --label_smoothing 0.1 --homotopy_speed 2.0 --kf_damping 1e-3 \
        --warmup 20 --sam_rho 0.05 \
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
        best_valid=$(grep "Best Valid Acc" "$f" 2>/dev/null | grep -oP '[0-9.]+(?=%)' || echo "N/A")
        best_test=$(grep "Best Test Acc" "$f" 2>/dev/null | grep -oP '[0-9.]+(?=%)' || echo "N/A")
        echo "$name: valid=$best_valid% test=$best_test%"
    fi
done | sort -t= -k3 -rn

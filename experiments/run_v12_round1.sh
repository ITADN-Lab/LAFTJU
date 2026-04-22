#!/bin/bash
# =============================================================================
# LAKTJU V12 Experiment Script - Round 1: Fast Hyperparameter Sweep (50 epochs)
# Maximize RTX 5090 utilization: ~3GB per job, 10 parallel jobs
# =============================================================================

set -e
cd "$(dirname "$0")"

RESULTS_DIR="./results/v12_round1"
mkdir -p "$RESULTS_DIR"

MAX_PARALLEL=10
EPOCHS=50
DATASET_DIR="./dataset"

# Job counter for parallel control
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
echo "  LAKTJU V12 Round 1: 50ep Fast Sweep"
echo "  Max parallel: $MAX_PARALLEL"
echo "  Results: $RESULTS_DIR"
echo "============================================"

# -----------------------------------------------
# Group A: CIFAR-100 LR sweep (core search)
# -----------------------------------------------
for LR in 0.001 0.003 0.005 0.01; do
    NAME="c100_lr${LR}_s42"
    run_job "python train_laktju.py \
        --optimizer LAKTJU_V12 --dataset cifar100 --model resnet18 \
        --lr $LR --weight_decay 5e-4 --epochs $EPOCHS --seed 42 \
        --label_smoothing 0.1 --homotopy_speed 2.0 --kf_warmup 300 \
        --kf_damping 1e-3 --kf_clip_max 50 --warmup 20 \
        --save_dir $RESULTS_DIR --data_dir $DATASET_DIR" "$NAME"
done

# -----------------------------------------------
# Group B: CIFAR-100 KF tuning
# -----------------------------------------------
for KF_WARMUP in 200 500 1000; do
    NAME="c100_kfw${KF_WARMUP}_s42"
    run_job "python train_laktju.py \
        --optimizer LAKTJU_V12 --dataset cifar100 --model resnet18 \
        --lr 0.003 --weight_decay 5e-4 --epochs $EPOCHS --seed 42 \
        --label_smoothing 0.1 --homotopy_speed 2.0 --kf_warmup $KF_WARMUP \
        --kf_damping 1e-3 --kf_clip_max 50 --warmup 20 \
        --save_dir $RESULTS_DIR --data_dir $DATASET_DIR" "$NAME"
done

# -----------------------------------------------
# Group C: CIFAR-100 Weight decay sweep
# -----------------------------------------------
for WD in 1e-4 5e-4 1e-3; do
    NAME="c100_wd${WD}_s42"
    run_job "python train_laktju.py \
        --optimizer LAKTJU_V12 --dataset cifar100 --model resnet18 \
        --lr 0.003 --weight_decay $WD --epochs $EPOCHS --seed 42 \
        --label_smoothing 0.1 --homotopy_speed 2.0 --kf_warmup 500 \
        --kf_damping 1e-3 --kf_clip_max 50 --warmup 20 \
        --save_dir $RESULTS_DIR --data_dir $DATASET_DIR" "$NAME"
done

echo ""
echo "Waiting for all Round 1 jobs to finish..."
wait
echo "============================================"
echo "  Round 1 complete! Checking results..."
echo "============================================"

# Parse results
echo ""
echo "=== ROUND 1 RESULTS ==="
for f in ${RESULTS_DIR}/*.json; do
    if [ -f "$f" ]; then
        name=$(basename "$f" .json)
        best=$(python3 -c "import json; d=json.load(open('$f')); print(f\"best_test={d['best_test_acc']:.2f}%\")" 2>/dev/null || echo "parse_error")
        echo "  $name: $best"
    fi
done

echo ""
echo "Round 1 done. Use run_v12_round2.sh for full 200ep training with best configs."

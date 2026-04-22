#!/bin/bash
# =============================================================================
# V11 Round 3: Aggressive KF + higher weight decay + SAM combinations
# Based on Round 1 findings: higher alpha_kf and wd=1e-3 performed best
# =============================================================================

set -e
cd "$(dirname "$0")"

PYTHON=python3
SCRIPT=train_laktju.py
COMMON="--model resnet18 --batch_size 128 --workers 2 --label_smoothing 0.1"

echo "=============================================="
echo "LAKTJU V11 Round 3: Aggressive Sweep"
echo "Started at: $(date)"
echo "=============================================="

mkdir -p ./results/v11_round3

# R3-01: alpha_kf=0.5, wd=1e-3, CIFAR-100, 200ep (top R1 combo)
$PYTHON $SCRIPT --optimizer LAKTJU_V11 --dataset cifar100 --epochs 200 --seed 42 \
    --lr 0.001 --alpha_kf 0.5 --kf_warmup 500 --weight_decay 1e-3 \
    --warmup 100 $COMMON \
    --save_dir ./results/v11_round3 2>&1 | tee ./results/v11_round3/R01_kf05_wd1e3.log &

# R3-02: alpha_kf=0.5, wd=1e-3, CIFAR-10, 200ep
$PYTHON $SCRIPT --optimizer LAKTJU_V11 --dataset cifar10 --epochs 200 --seed 42 \
    --lr 0.001 --alpha_kf 0.5 --kf_warmup 500 --weight_decay 1e-3 \
    --warmup 100 $COMMON \
    --save_dir ./results/v11_round3 2>&1 | tee ./results/v11_round3/R02_kf05_wd1e3_c10.log &

# R3-03: alpha_kf=0.5, wd=1e-3, SAM rho=0.03, CIFAR-100
$PYTHON $SCRIPT --optimizer LAKTJU_V11 --dataset cifar100 --epochs 200 --seed 42 \
    --lr 0.001 --alpha_kf 0.5 --kf_warmup 500 --weight_decay 1e-3 \
    --sam_rho 0.03 --warmup 100 $COMMON \
    --save_dir ./results/v11_round3 2>&1 | tee ./results/v11_round3/R03_kf05_wd1e3_sam003.log &

# R3-04: alpha_kf=0.3, wd=1e-3, SAM rho=0.05, CIFAR-100
$PYTHON $SCRIPT --optimizer LAKTJU_V11 --dataset cifar100 --epochs 200 --seed 42 \
    --lr 0.001 --alpha_kf 0.3 --kf_warmup 500 --weight_decay 1e-3 \
    --sam_rho 0.05 --warmup 100 $COMMON \
    --save_dir ./results/v11_round3 2>&1 | tee ./results/v11_round3/R04_kf03_wd1e3_sam005.log &

wait
echo "Round 3 complete at $(date)"

echo ""
echo "=== Round 3 Results ==="
for f in ./results/v11_round3/R*.log; do
    name=$(basename "$f" .log)
    best=$(grep "Best Test Acc" "$f" 2>/dev/null | tail -1)
    echo "$name: $best"
done

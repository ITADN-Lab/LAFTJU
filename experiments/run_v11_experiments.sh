#!/bin/bash
# =============================================================================
# LAKTJU V11 大规模并行实验
# 架构: KF-Enhanced AdamW (不再混合两个优化器)
# GPU: RTX 5090 (32GB VRAM, 600W) - 目标利用率 >80%
# 策略: 10个并行实验, 每个~3GB VRAM
# =============================================================================

set -e
cd "$(dirname "$0")"

PYTHON=python3
SCRIPT=train_laktju.py
COMMON="--model resnet18 --batch_size 128 --workers 2 --label_smoothing 0.1"

echo "=============================================="
echo "LAKTJU V11 Parallel Experiment Suite"
echo "Started at: $(date)"
echo "GPU Status:"
nvidia-smi --query-gpu=memory.total,memory.used,power.draw --format=csv,noheader
echo "=============================================="

mkdir -p ./results/v11_round1
mkdir -p ./results/v11_round2

# =============================================================================
# Round 1: 50ep快速扫描 (10个并行, ~30min)
# 扫描: alpha_kf × lr × weight_decay × kf_warmup
# =============================================================================
echo ""
echo ">>> Round 1: Fast sweep (CIFAR-100, 50 epochs, 10 parallel jobs)"
echo "=============================================="

# --- Batch 1: 10 parallel jobs ---

# R1-01: Pure AdamW baseline (alpha_kf=0)
$PYTHON $SCRIPT --optimizer LAKTJU_V11 --dataset cifar100 --epochs 50 --seed 42 \
    --lr 0.001 --alpha_kf 0.0 --kf_warmup 500 --weight_decay 5e-4 \
    --warmup 100 $COMMON \
    --save_dir ./results/v11_round1 2>&1 | tee ./results/v11_round1/R01_adamw_baseline.log &

# R1-02: alpha_kf=0.1, lr=0.001
$PYTHON $SCRIPT --optimizer LAKTJU_V11 --dataset cifar100 --epochs 50 --seed 42 \
    --lr 0.001 --alpha_kf 0.1 --kf_warmup 500 --weight_decay 5e-4 \
    --warmup 100 $COMMON \
    --save_dir ./results/v11_round1 2>&1 | tee ./results/v11_round1/R02_kf01_lr001.log &

# R1-03: alpha_kf=0.2, lr=0.001
$PYTHON $SCRIPT --optimizer LAKTJU_V11 --dataset cifar100 --epochs 50 --seed 42 \
    --lr 0.001 --alpha_kf 0.2 --kf_warmup 500 --weight_decay 5e-4 \
    --warmup 100 $COMMON \
    --save_dir ./results/v11_round1 2>&1 | tee ./results/v11_round1/R03_kf02_lr001.log &

# R1-04: alpha_kf=0.3, lr=0.001
$PYTHON $SCRIPT --optimizer LAKTJU_V11 --dataset cifar100 --epochs 50 --seed 42 \
    --lr 0.001 --alpha_kf 0.3 --kf_warmup 500 --weight_decay 5e-4 \
    --warmup 100 $COMMON \
    --save_dir ./results/v11_round1 2>&1 | tee ./results/v11_round1/R04_kf03_lr001.log &

# R1-05: alpha_kf=0.5, lr=0.001 (aggressive KF)
$PYTHON $SCRIPT --optimizer LAKTJU_V11 --dataset cifar100 --epochs 50 --seed 42 \
    --lr 0.001 --alpha_kf 0.5 --kf_warmup 500 --weight_decay 5e-4 \
    --warmup 100 $COMMON \
    --save_dir ./results/v11_round1 2>&1 | tee ./results/v11_round1/R05_kf05_lr001.log &

# R1-06: alpha_kf=0.2, lr=0.003 (higher lr)
$PYTHON $SCRIPT --optimizer LAKTJU_V11 --dataset cifar100 --epochs 50 --seed 42 \
    --lr 0.003 --alpha_kf 0.2 --kf_warmup 500 --weight_decay 5e-4 \
    --warmup 100 $COMMON \
    --save_dir ./results/v11_round1 2>&1 | tee ./results/v11_round1/R06_kf02_lr003.log &

# R1-07: alpha_kf=0.2, lr=0.0005 (lower lr)
$PYTHON $SCRIPT --optimizer LAKTJU_V11 --dataset cifar100 --epochs 50 --seed 42 \
    --lr 0.0005 --alpha_kf 0.2 --kf_warmup 500 --weight_decay 5e-4 \
    --warmup 100 $COMMON \
    --save_dir ./results/v11_round1 2>&1 | tee ./results/v11_round1/R07_kf02_lr0005.log &

# R1-08: alpha_kf=0.2, kf_warmup=200 (earlier KF)
$PYTHON $SCRIPT --optimizer LAKTJU_V11 --dataset cifar100 --epochs 50 --seed 42 \
    --lr 0.001 --alpha_kf 0.2 --kf_warmup 200 --weight_decay 5e-4 \
    --warmup 100 $COMMON \
    --save_dir ./results/v11_round1 2>&1 | tee ./results/v11_round1/R08_kf02_warm200.log &

# R1-09: alpha_kf=0.2, weight_decay=1e-3 (higher WD)
$PYTHON $SCRIPT --optimizer LAKTJU_V11 --dataset cifar100 --epochs 50 --seed 42 \
    --lr 0.001 --alpha_kf 0.2 --kf_warmup 500 --weight_decay 1e-3 \
    --warmup 100 $COMMON \
    --save_dir ./results/v11_round1 2>&1 | tee ./results/v11_round1/R09_kf02_wd1e3.log &

# R1-10: alpha_kf=0.2, cos_sim_threshold=0.1 (stricter gate)
$PYTHON $SCRIPT --optimizer LAKTJU_V11 --dataset cifar100 --epochs 50 --seed 42 \
    --lr 0.001 --alpha_kf 0.2 --kf_warmup 500 --weight_decay 5e-4 \
    --cos_sim_threshold 0.1 --warmup 100 $COMMON \
    --save_dir ./results/v11_round1 2>&1 | tee ./results/v11_round1/R10_kf02_cosgate01.log &

wait
echo ">>> Round 1 complete at $(date). Checking results..."

# Print round 1 summary
echo ""
echo "=== Round 1 Results Summary ==="
for f in ./results/v11_round1/R*.log; do
    name=$(basename "$f" .log)
    best=$(grep "Best Test Acc" "$f" 2>/dev/null | tail -1)
    if [ -n "$best" ]; then
        echo "$name: $best"
    else
        last=$(grep "Valid Acc" "$f" 2>/dev/null | tail -1)
        echo "$name: $last"
    fi
done
echo "================================"

# =============================================================================
# Round 2: 200ep完整训练 (基于Round 1最佳 + 多数据集, 8并行)
# 根据Round 1选择top-3配置 × {CIFAR-10, CIFAR-100} + baselines
# =============================================================================
echo ""
echo ">>> Round 2: Full training (200 epochs, 8 parallel jobs)"
echo "=============================================="

# --- 使用Round 1默认最优猜测, 后续可根据结果调整 ---
# Top config candidates (update after checking Round 1):
BEST_KF=0.2
BEST_LR=0.001
BEST_WD=5e-4
BEST_WARM=500

# R2-01: V11 best config, CIFAR-100, seed=42
$PYTHON $SCRIPT --optimizer LAKTJU_V11 --dataset cifar100 --epochs 200 --seed 42 \
    --lr $BEST_LR --alpha_kf $BEST_KF --kf_warmup $BEST_WARM --weight_decay $BEST_WD \
    --warmup 100 $COMMON \
    --save_dir ./results/v11_round2 2>&1 | tee ./results/v11_round2/R01_v11_c100_s42.log &

# R2-02: V11 best config, CIFAR-10, seed=42
$PYTHON $SCRIPT --optimizer LAKTJU_V11 --dataset cifar10 --epochs 200 --seed 42 \
    --lr $BEST_LR --alpha_kf $BEST_KF --kf_warmup $BEST_WARM --weight_decay $BEST_WD \
    --warmup 100 $COMMON \
    --save_dir ./results/v11_round2 2>&1 | tee ./results/v11_round2/R02_v11_c10_s42.log &

# R2-03: V11 best config, CIFAR-100, seed=123
$PYTHON $SCRIPT --optimizer LAKTJU_V11 --dataset cifar100 --epochs 200 --seed 123 \
    --lr $BEST_LR --alpha_kf $BEST_KF --kf_warmup $BEST_WARM --weight_decay $BEST_WD \
    --warmup 100 $COMMON \
    --save_dir ./results/v11_round2 2>&1 | tee ./results/v11_round2/R03_v11_c100_s123.log &

# R2-04: V11 best config, CIFAR-10, seed=123
$PYTHON $SCRIPT --optimizer LAKTJU_V11 --dataset cifar10 --epochs 200 --seed 123 \
    --lr $BEST_LR --alpha_kf $BEST_KF --kf_warmup $BEST_WARM --weight_decay $BEST_WD \
    --warmup 100 $COMMON \
    --save_dir ./results/v11_round2 2>&1 | tee ./results/v11_round2/R04_v11_c10_s123.log &

# R2-05: V11 + SAM rho=0.03, CIFAR-100
$PYTHON $SCRIPT --optimizer LAKTJU_V11 --dataset cifar100 --epochs 200 --seed 42 \
    --lr $BEST_LR --alpha_kf $BEST_KF --kf_warmup $BEST_WARM --weight_decay $BEST_WD \
    --sam_rho 0.03 --warmup 100 $COMMON \
    --save_dir ./results/v11_round2 2>&1 | tee ./results/v11_round2/R05_v11_sam003_c100.log &

# R2-06: V11 + SAM rho=0.05, CIFAR-100
$PYTHON $SCRIPT --optimizer LAKTJU_V11 --dataset cifar100 --epochs 200 --seed 42 \
    --lr $BEST_LR --alpha_kf $BEST_KF --kf_warmup $BEST_WARM --weight_decay $BEST_WD \
    --sam_rho 0.05 --warmup 100 $COMMON \
    --save_dir ./results/v11_round2 2>&1 | tee ./results/v11_round2/R06_v11_sam005_c100.log &

# R2-07: SGD baseline CIFAR-100 (reconfirm)
$PYTHON $SCRIPT --optimizer SGD --dataset cifar100 --epochs 200 --seed 42 \
    --weight_decay 5e-4 $COMMON \
    --save_dir ./results/v11_round2 2>&1 | tee ./results/v11_round2/R07_sgd_c100.log &

# R2-08: AdamW baseline CIFAR-100 (reconfirm)
$PYTHON $SCRIPT --optimizer AdamW --dataset cifar100 --epochs 200 --seed 42 \
    --weight_decay 5e-4 $COMMON \
    --save_dir ./results/v11_round2 2>&1 | tee ./results/v11_round2/R08_adamw_c100.log &

wait
echo ">>> Round 2 complete at $(date)."

# Print round 2 summary
echo ""
echo "=== Round 2 Results Summary ==="
for f in ./results/v11_round2/R*.log; do
    name=$(basename "$f" .log)
    best=$(grep "Best Test Acc" "$f" 2>/dev/null | tail -1)
    if [ -n "$best" ]; then
        echo "$name: $best"
    fi
done
echo "================================"

echo ""
echo "=============================================="
echo "All V11 experiments complete at: $(date)"
echo "=============================================="
echo ""
echo "成功标准："
echo "  CIFAR-10 单种子 >= 96.2%"
echo "  CIFAR-100 单种子 >= 78.3%"
echo "  CIFAR-10 双种子均值 >= 96.1%"
echo "  CIFAR-100 双种子均值 >= 78.0%"

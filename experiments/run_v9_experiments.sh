#!/bin/bash
# =============================================================================
# LAKTJU V9 实验执行脚本
# 基于GPT-5.4专家分析的系统性实验方案
# 总预算：~24小时 (RTX 5090)
# =============================================================================

set -e
cd "$(dirname "$0")"

PYTHON=python3
SCRIPT=train_laktju.py
COMMON="--model resnet18 --batch_size 128 --workers 4 --label_smoothing 0.1"

echo "=============================================="
echo "LAKTJU V9 Experiment Suite"
echo "Started at: $(date)"
echo "=============================================="

# =============================================================================
# Round 0: 机制诊断 (~30min)
# 目标：确认V8的homotopy退化问题
# =============================================================================
echo ""
echo ">>> Round 0: Diagnostic run (V8 baseline, CIFAR-100, 20 epochs)"
echo "=============================================="

$PYTHON $SCRIPT --optimizer LAKTJU --dataset cifar100 --epochs 20 --seed 42 \
    --lr 0.003 --a_lr_ratio 0.333 --homotopy_speed 5.0 --warmup 100 \
    --weight_decay 0.001 $COMMON \
    --save_dir ./results/v9_round0 2>&1 | tee ./results/v9_round0_diag.log

echo ">>> Round 0 complete."

# =============================================================================
# Round 1: 动力学骨架验证 (~3h, 6 experiments × 50 epochs)
# 假设：SGD-momentum主路径 + s_max<1 + KF全程 → 显著优于V8
# =============================================================================
echo ""
echo ">>> Round 1: Dynamics skeleton (CIFAR-100, 50 epochs, seed=42)"
echo "=============================================="

mkdir -p ./results/v9_round1

# R1-1: tju_lr=0.01, a_lr_ratio=0.2, alpha_kf=0.15, alpha_adam_kf=0.05, s_max=0.7
$PYTHON $SCRIPT --optimizer LAKTJU_V9 --dataset cifar100 --epochs 50 --seed 42 \
    --lr 0.01 --a_lr_ratio 0.2 --alpha_kf 0.15 --alpha_adam_kf 0.05 --s_max 0.7 \
    --homotopy_speed 5.0 --warmup 100 --weight_decay 0.001 $COMMON \
    --save_dir ./results/v9_round1 2>&1 | tee ./results/v9_round1/R1-1.log &

# R1-2: tju_lr=0.02, a_lr_ratio=0.2, alpha_kf=0.15, alpha_adam_kf=0.05, s_max=0.7
$PYTHON $SCRIPT --optimizer LAKTJU_V9 --dataset cifar100 --epochs 50 --seed 42 \
    --lr 0.02 --a_lr_ratio 0.2 --alpha_kf 0.15 --alpha_adam_kf 0.05 --s_max 0.7 \
    --homotopy_speed 5.0 --warmup 100 --weight_decay 0.001 $COMMON \
    --save_dir ./results/v9_round1 2>&1 | tee ./results/v9_round1/R1-2.log &

# R1-3: tju_lr=0.02, a_lr_ratio=0.33, alpha_kf=0.30, alpha_adam_kf=0.05, s_max=0.7
$PYTHON $SCRIPT --optimizer LAKTJU_V9 --dataset cifar100 --epochs 50 --seed 42 \
    --lr 0.02 --a_lr_ratio 0.33 --alpha_kf 0.30 --alpha_adam_kf 0.05 --s_max 0.7 \
    --homotopy_speed 5.0 --warmup 100 --weight_decay 0.001 $COMMON \
    --save_dir ./results/v9_round1 2>&1 | tee ./results/v9_round1/R1-3.log &

wait
echo ">>> Round 1 batch 1 (R1-1,2,3) complete."

# R1-4: tju_lr=0.03, a_lr_ratio=0.2, alpha_kf=0.30, alpha_adam_kf=0.10, s_max=0.7
$PYTHON $SCRIPT --optimizer LAKTJU_V9 --dataset cifar100 --epochs 50 --seed 42 \
    --lr 0.03 --a_lr_ratio 0.2 --alpha_kf 0.30 --alpha_adam_kf 0.10 --s_max 0.7 \
    --homotopy_speed 5.0 --warmup 100 --weight_decay 0.001 $COMMON \
    --save_dir ./results/v9_round1 2>&1 | tee ./results/v9_round1/R1-4.log &

# R1-5: 无KF对照 (alpha_kf=0, alpha_adam_kf=0)
$PYTHON $SCRIPT --optimizer LAKTJU_V9 --dataset cifar100 --epochs 50 --seed 42 \
    --lr 0.02 --a_lr_ratio 0.2 --alpha_kf 0.0 --alpha_adam_kf 0.0 --s_max 0.7 \
    --homotopy_speed 5.0 --warmup 100 --weight_decay 0.001 $COMMON \
    --save_dir ./results/v9_round1 2>&1 | tee ./results/v9_round1/R1-5.log &

# R1-6: s_max=1.0 对照 (验证是否必须保留非零主路径)
$PYTHON $SCRIPT --optimizer LAKTJU_V9 --dataset cifar100 --epochs 50 --seed 42 \
    --lr 0.02 --a_lr_ratio 0.2 --alpha_kf 0.15 --alpha_adam_kf 0.05 --s_max 1.0 \
    --homotopy_speed 5.0 --warmup 100 --weight_decay 0.001 $COMMON \
    --save_dir ./results/v9_round1 2>&1 | tee ./results/v9_round1/R1-6.log &

wait
echo ">>> Round 1 batch 2 (R1-4,5,6) complete."
echo ">>> Round 1 ALL DONE. Check results/v9_round1/ for best config."

# =============================================================================
# Round 2: SAM叠加验证 (~2h, 4 experiments × 50 epochs)
# 需要手动选择Round 1最佳配置填入下方
# =============================================================================
echo ""
echo ">>> Round 2: SAM integration (CIFAR-100, 50 epochs, seed=42)"
echo ">>> NOTE: Update WINNER_LR, WINNER_RATIO, WINNER_AKF, WINNER_AAKF below"
echo "=============================================="

mkdir -p ./results/v9_round2

# ---- 请根据Round 1结果修改以下参数 ----
WINNER_LR=0.02
WINNER_RATIO=0.2
WINNER_AKF=0.15
WINNER_AAKF=0.05
WINNER_SMAX=0.7
# ---- 修改结束 ----

# R2-1: winner + SAM rho=0.03
$PYTHON $SCRIPT --optimizer LAKTJU_V9 --dataset cifar100 --epochs 50 --seed 42 \
    --lr $WINNER_LR --a_lr_ratio $WINNER_RATIO --alpha_kf $WINNER_AKF --alpha_adam_kf $WINNER_AAKF \
    --s_max $WINNER_SMAX --homotopy_speed 5.0 --warmup 100 --weight_decay 0.001 \
    --sam_rho 0.03 $COMMON \
    --save_dir ./results/v9_round2 2>&1 | tee ./results/v9_round2/R2-1.log &

# R2-2: winner + SAM rho=0.05
$PYTHON $SCRIPT --optimizer LAKTJU_V9 --dataset cifar100 --epochs 50 --seed 42 \
    --lr $WINNER_LR --a_lr_ratio $WINNER_RATIO --alpha_kf $WINNER_AKF --alpha_adam_kf $WINNER_AAKF \
    --s_max $WINNER_SMAX --homotopy_speed 5.0 --warmup 100 --weight_decay 0.001 \
    --sam_rho 0.05 $COMMON \
    --save_dir ./results/v9_round2 2>&1 | tee ./results/v9_round2/R2-2.log &

wait

# R2-3: winner 无SAM对照
$PYTHON $SCRIPT --optimizer LAKTJU_V9 --dataset cifar100 --epochs 50 --seed 42 \
    --lr $WINNER_LR --a_lr_ratio $WINNER_RATIO --alpha_kf $WINNER_AKF --alpha_adam_kf $WINNER_AAKF \
    --s_max $WINNER_SMAX --homotopy_speed 5.0 --warmup 100 --weight_decay 0.001 \
    --sam_rho 0.0 $COMMON \
    --save_dir ./results/v9_round2 2>&1 | tee ./results/v9_round2/R2-3.log &

# R2-4: SGD对照 + SAM rho=0.05
$PYTHON $SCRIPT --optimizer SGD --dataset cifar100 --epochs 50 --seed 42 \
    --weight_decay 0.001 --sam_rho 0.05 $COMMON \
    --save_dir ./results/v9_round2 2>&1 | tee ./results/v9_round2/R2-4.log &

wait
echo ">>> Round 2 ALL DONE."

# =============================================================================
# Round 3: 单种子完整训练 (~6h, 4 experiments × 200 epochs)
# 需要手动选择Round 1+2最佳配置
# =============================================================================
echo ""
echo ">>> Round 3: Full training (200 epochs, seed=42)"
echo ">>> NOTE: Update BEST_* params below based on Round 1+2 results"
echo "=============================================="

mkdir -p ./results/v9_round3

# ---- 请根据Round 1+2结果修改以下参数 ----
BEST_LR=0.02
BEST_RATIO=0.2
BEST_AKF=0.15
BEST_AAKF=0.05
BEST_SMAX=0.7
BEST_SAM=0.0  # 设为0.03或0.05如果SAM有效
# ---- 修改结束 ----

# R3-1: LAKTJU_V9 CIFAR-100
$PYTHON $SCRIPT --optimizer LAKTJU_V9 --dataset cifar100 --epochs 200 --seed 42 \
    --lr $BEST_LR --a_lr_ratio $BEST_RATIO --alpha_kf $BEST_AKF --alpha_adam_kf $BEST_AAKF \
    --s_max $BEST_SMAX --homotopy_speed 5.0 --warmup 100 --weight_decay 0.001 \
    --sam_rho $BEST_SAM $COMMON \
    --save_dir ./results/v9_round3 2>&1 | tee ./results/v9_round3/R3-1_cifar100.log &

# R3-2: LAKTJU_V9 CIFAR-10
$PYTHON $SCRIPT --optimizer LAKTJU_V9 --dataset cifar10 --epochs 200 --seed 42 \
    --lr $BEST_LR --a_lr_ratio $BEST_RATIO --alpha_kf $BEST_AKF --alpha_adam_kf $BEST_AAKF \
    --s_max $BEST_SMAX --homotopy_speed 5.0 --warmup 100 --weight_decay 0.001 \
    --sam_rho $BEST_SAM $COMMON \
    --save_dir ./results/v9_round3 2>&1 | tee ./results/v9_round3/R3-2_cifar10.log &

wait

# R3-3: SGD 对照 CIFAR-100
$PYTHON $SCRIPT --optimizer SGD --dataset cifar100 --epochs 200 --seed 42 \
    --weight_decay 0.001 $COMMON \
    --save_dir ./results/v9_round3 2>&1 | tee ./results/v9_round3/R3-3_sgd_cifar100.log &

# R3-4: SGD 对照 CIFAR-10
$PYTHON $SCRIPT --optimizer SGD --dataset cifar10 --epochs 200 --seed 42 \
    --weight_decay 0.001 $COMMON \
    --save_dir ./results/v9_round3 2>&1 | tee ./results/v9_round3/R3-4_sgd_cifar10.log &

wait
echo ">>> Round 3 ALL DONE."

# =============================================================================
# Round 4: 双种子确认 (~3h, 2 experiments × 200 epochs)
# =============================================================================
echo ""
echo ">>> Round 4: Multi-seed confirmation (200 epochs, seed=123)"
echo "=============================================="

mkdir -p ./results/v9_round4

# R4-1: LAKTJU_V9 CIFAR-100 seed=123
$PYTHON $SCRIPT --optimizer LAKTJU_V9 --dataset cifar100 --epochs 200 --seed 123 \
    --lr $BEST_LR --a_lr_ratio $BEST_RATIO --alpha_kf $BEST_AKF --alpha_adam_kf $BEST_AAKF \
    --s_max $BEST_SMAX --homotopy_speed 5.0 --warmup 100 --weight_decay 0.001 \
    --sam_rho $BEST_SAM $COMMON \
    --save_dir ./results/v9_round4 2>&1 | tee ./results/v9_round4/R4-1_cifar100_s123.log &

# R4-2: LAKTJU_V9 CIFAR-10 seed=123
$PYTHON $SCRIPT --optimizer LAKTJU_V9 --dataset cifar10 --epochs 200 --seed 123 \
    --lr $BEST_LR --a_lr_ratio $BEST_RATIO --alpha_kf $BEST_AKF --alpha_adam_kf $BEST_AAKF \
    --s_max $BEST_SMAX --homotopy_speed 5.0 --warmup 100 --weight_decay 0.001 \
    --sam_rho $BEST_SAM $COMMON \
    --save_dir ./results/v9_round4 2>&1 | tee ./results/v9_round4/R4-2_cifar10_s123.log &

wait
echo ">>> Round 4 ALL DONE."

echo ""
echo "=============================================="
echo "All experiments complete at: $(date)"
echo "=============================================="
echo ""
echo "成功标准："
echo "  CIFAR-10 单种子 >= 96.2%"
echo "  CIFAR-100 单种子 >= 78.3%"
echo "  CIFAR-10 双种子均值 >= 96.1%"
echo "  CIFAR-100 双种子均值 >= 78.0%"

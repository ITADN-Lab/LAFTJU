# LAKTJU V9 实验策略与执行计划

> 日期：2026-03-19
> 基于：GPT-5.4 专家级分析

## 核心改动（V8 → V9）

| 维度 | V8 | V9 |
|------|----|----|
| 主路径 | TJU (EMA梯度 + diagonal Hessian/KF) | SGD-momentum + KF方向修正 |
| KF参与范围 | 仅TJU路径，后期被homotopy稀释 | 同时修正主路径和AdamW路径，全程参与 |
| Homotopy | s→1.0（退化为纯AdamW） | s_max=0.7（始终保留30%主路径） |
| Weight decay | 不一致（TJU用L2，AdamW用decoupled） | 统一decoupled |
| 学习率 | tju_lr=0.003 | tju_lr=0.01~0.03 |

## 新增超参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| momentum | 0.9 | SGD-momentum系数 |
| alpha_kf | 0.15 | 主路径KF修正比例 |
| alpha_adam_kf | 0.05 | AdamW路径KF修正比例 |
| s_max | 0.7 | Homotopy上限 |
| grad_clip | 0.0 | 梯度裁剪（0=禁用） |

## 实验计划

### Round 0: 机制诊断（~30min）
- V8 CIFAR-100, 20 epochs, seed=42
- 确认homotopy退化问题

### Round 1: 动力学骨架（~3h）
- CIFAR-100, 50 epochs, seed=42, 6组配置

| ID | tju_lr | a_lr_ratio | alpha_kf | alpha_adam_kf | s_max | 说明 |
|----|--------|------------|----------|---------------|-------|------|
| R1-1 | 0.01 | 0.2 | 0.15 | 0.05 | 0.7 | 保守 |
| R1-2 | 0.02 | 0.2 | 0.15 | 0.05 | 0.7 | 中等lr |
| R1-3 | 0.02 | 0.33 | 0.30 | 0.05 | 0.7 | 强KF+高AdamW比 |
| R1-4 | 0.03 | 0.2 | 0.30 | 0.10 | 0.7 | 激进 |
| R1-5 | 0.02 | 0.2 | 0.00 | 0.00 | 0.7 | 无KF对照 |
| R1-6 | 0.02 | 0.2 | 0.15 | 0.05 | 1.0 | s_max=1对照 |

### Round 2: SAM叠加（~2h）
- Round 1最佳 + SAM rho={0.03, 0.05}

### Round 3: 完整训练（~6h）
- 最佳配置 × {CIFAR-10, CIFAR-100} × 200 epochs × seed=42
- SGD对照

### Round 4: 双种子确认（~3h）
- 最佳配置 × {CIFAR-10, CIFAR-100} × 200 epochs × seed=123

## 成功标准

| 指标 | CIFAR-10 | CIFAR-100 |
|------|----------|-----------|
| 单种子 | ≥ 96.2% | ≥ 78.3% |
| 双种子均值 | ≥ 96.1% | ≥ 78.0% |

## 文件清单

- `LAKTJU_V9.py` — V9优化器实现
- `train_laktju.py` — 训练脚本（已支持LAKTJU_V9）
- `run_v9_experiments.sh` — 实验执行脚本
- `analyze/LAKTJU_V9_GPT54_expert_analysis.md` — GPT-5.4完整分析报告

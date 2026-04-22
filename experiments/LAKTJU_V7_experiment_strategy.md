# LAKTJU 实验总结与V7改进策略

## 1. 历史实验结果汇总

### 1.1 基线结果 (ResNet18, CIFAR, 200 epochs, 3 seeds)

| 优化器 | CIFAR-10 Test Acc | CIFAR-100 Test Acc |
|--------|-------------------|---------------------|
| SGD (momentum=0.9) | 95.99±0.15% | 76.83±0.28% |
| AdamW | 94.55±0.04% | 71.32±0.20% |
| Adan | 94.42±0.10% | 66.39±0.28% |
| Adam | 93.99±0.28% | 74.19±0.19% |
| ATJU (V5) | 93.17±0.21% | 70.40±0.22% |

### 1.2 LAKTJU各版本演进

| 版本 | CIFAR-10 | CIFAR-100 | 主要变更 |
|------|----------|-----------|----------|
| V2 (初版) | ~89% | ~65% | 全功能：KF+QGS+homotopy |
| V3 | ~90% | ~66% | 修复KF维度bug |
| V4 | ~91% | ~67% | 放宽KF clip 3x→10x |
| V5 | 92.08±0.28% | 66.60±0.36% | 去除max_update=1.0 |
| **V6 (当前)** | **93.48±0.04%** | **70.26±0.48%** | 精简回归：100x KF clip, plain tanh homotopy, 去除QGS |

### 1.3 V6最终结果明细

| 实验 | Best Valid Acc | Best Test Acc |
|------|---------------|---------------|
| CIFAR-10 seed42 | 94.02% | 93.45% |
| CIFAR-10 seed123 | 94.26% | 93.52% |
| CIFAR-10 seed456 | 94.66% | 93.46% |
| **CIFAR-10 均值±std** | **94.31±0.33%** | **93.48±0.04%** |
| CIFAR-100 seed42 | 71.00% | 70.49% |
| CIFAR-100 seed123 | 69.74% | 69.71% |
| CIFAR-100 seed456 | 70.48% | 70.59% |
| **CIFAR-100 均值±std** | **70.41±0.63%** | **70.26±0.48%** |

### 1.4 V6与目标差距分析

| 目标 | CIFAR-10 差距 | CIFAR-100 差距 |
|------|--------------|----------------|
| 超过ATJU (93.17/70.40) | ✅ +0.31% | ❌ -0.14% |
| 超过Adam (93.99/74.19) | ❌ -0.51% | ❌ -3.93% |
| 超过Adan (94.42/66.39) | ❌ -0.94% | ✅ +3.87% |
| 超过AdamW (94.55/71.32) | ❌ -1.07% | ❌ -1.06% |
| 超过SGD (95.99/76.83) | ❌ -2.51% | ❌ -6.57% |

## 2. V6问题诊断

### 2.1 核心瓶颈

1. **学习率不匹配**: tju_lr=0.001, a_lr=0.0001 — AdamW单独用lr=0.001就能达到94.55%，但LAKTJU的AdamW分支只用了0.0001，被严重削弱
2. **Homotopy过渡太慢**: speed=2.0时，s在epoch 100才到0.76，前半段TJU路径主导但TJU路径本身不如AdamW
3. **KF preconditioning效果有限**: 100x clip仍然限制了KF的贡献，且KF只在step>100后才启用
4. **a_lr/tju_lr比例固定**: 1:10的比例在整个训练过程中不变，但最优比例应随训练阶段变化
5. **Weight decay处理不一致**: TJU路径用L2 (folded into gradient)，AdamW路径用decoupled，混合时产生不一致的正则化效果

### 2.2 关键观察

- V6 CIFAR-10 valid acc最高达94.66% (seed456)，说明模型有能力达到更高精度
- CIFAR-100 seed间方差较大 (0.48%)，说明优化不够稳定
- Train loss降到0.02但test acc停在93.5%，存在过拟合倾向
- 对比SGD的95.99%，差距主要在泛化能力而非优化能力

## 3. V7改进方案

### 3.1 方案A: 学习率与调度优化 (优先级最高)

**核心思路**: LAKTJU的AdamW分支应该能独立达到AdamW的水平

修改点:
1. **提高a_lr**: 从tju_lr/10改为tju_lr/3或tju_lr/2
2. **加速homotopy**: speed从2.0提高到5.0，让s更快接近1.0
3. **更长warmup**: 从20步增加到100步，让KF有更多时间积累统计量

```python
# train_laktju.py 修改
tju_lr = 0.001
a_lr = tju_lr / 3  # 从/10改为/3
homotopy_speed = 5.0  # 从2.0改为5.0
warmup = 100  # 从20改为100
```

预期效果: CIFAR-10 → 94.0-94.5%, CIFAR-100 → 71-72%

### 3.2 方案B: 统一Weight Decay为Decoupled (优先级高)

**核心思路**: SGD+momentum的成功很大程度归功于正确的weight decay

修改点:
1. TJU路径也改用decoupled weight decay (不fold进gradient)
2. 统一wd强度，避免TJU和AdamW路径的正则化冲突

```python
# LAKTJU.py step() 修改
# 去除: grad_tju.add_(p, alpha=wd)
# 改为在blending后统一施加:
blended = ((1 - s) * current_tju_lr * update_tju
           + s * current_a_lr * update_adamw)
# 统一decoupled weight decay
p.mul_(1 - wd * max(current_tju_lr, current_a_lr))
p.add_(blended, alpha=-1.0)
```

预期效果: 改善泛化，CIFAR-100提升1-2%

### 3.3 方案C: 自适应Homotopy (优先级中)

**核心思路**: 不用固定tanh schedule，而是根据训练状态自适应调整s

修改点:
1. 监控validation loss的变化率
2. 当TJU路径导致loss上升时，加速向AdamW过渡
3. 当AdamW路径收敛变慢时，保留更多TJU贡献

```python
# 基于loss ratio的自适应s
if self._prev_loss is not None:
    loss_ratio = current_loss / (self._prev_loss + 1e-8)
    if loss_ratio > 1.05:  # loss上升，加速过渡
        self._s_momentum = min(self._s_momentum + 0.01, 1.0)
    s = max(s_tanh, self._s_momentum)
```

### 3.4 方案D: 增强数据增强 (优先级中)

**核心思路**: SGD 95.99%的优势部分来自更好的泛化，可通过更强的数据增强弥补

修改点:
1. 添加AutoAugment或RandAugment
2. 增加Cutout长度从16到20
3. 添加MixUp或CutMix

```python
# train_laktju.py 数据增强
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize(...),
    Cutout(n_holes=1, length=20),
])
```

注意: 数据增强改变了实验条件，需要同时重跑所有基线以保证公平对比

### 3.5 方案E: KF改进 (优先级低)

修改点:
1. 更早启用KF (step>20而非step>100)
2. 去除KF clip限制，改用adaptive damping
3. 使用Tikhonov正则化替代固定damping

## 4. 实验执行计划

### Round 1: 学习率搜索 (方案A)

在CIFAR-10 seed42上快速验证，50 epochs:

| 实验 | tju_lr | a_lr | homotopy_speed | warmup |
|------|--------|------|----------------|--------|
| R1-1 | 0.001 | 0.001/3 | 5.0 | 100 |
| R1-2 | 0.001 | 0.001/2 | 5.0 | 100 |
| R1-3 | 0.003 | 0.001 | 5.0 | 100 |
| R1-4 | 0.001 | 0.001/3 | 3.0 | 100 |

### Round 2: Weight Decay统一 (方案B)

用Round 1最佳lr配置，在CIFAR-10/100 seed42上验证:

| 实验 | WD方式 | wd值 |
|------|--------|------|
| R2-1 | Decoupled统一 | 5e-4 |
| R2-2 | Decoupled统一 | 1e-3 |
| R2-3 | Decoupled统一 | 2e-4 |

### Round 3: 完整验证

用Round 1+2最佳配置，跑完整200 epochs × 2 datasets × 3 seeds

### Round 4: 数据增强 (如Round 3仍不够)

添加AutoAugment，重跑所有优化器以公平对比

## 5. 成功标准

| 阶段 | CIFAR-10目标 | CIFAR-100目标 |
|------|-------------|---------------|
| V7 Round 1 | >94.0% | >71.0% |
| V7 Round 3 | >94.5% (超AdamW) | >72.0% (超AdamW) |
| V7 Round 4 | >95.0% (接近SGD) | >75.0% (超Adam) |
| 最终目标 | >96.0% (超SGD) | >77.0% (超SGD) |

## 6. 技术备注

- 硬件: RTX 5090 32GB, 可并行6个ResNet18实验
- 每个实验200 epochs约5.8小时 (6并行)
- 50 epoch快速验证约1.5小时
- Round 1 (4实验并行) ≈ 1.5小时
- Round 2 (3实验并行) ≈ 1.5小时
- Round 3 (6实验并行) ≈ 5.8小时
- 总计约10小时完成V7全部实验

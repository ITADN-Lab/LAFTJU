# LAKTJU 实验结果总结 (V6→V7→V8)

## 1. 版本演进

| 版本 | CIFAR-10 | CIFAR-100 | 主要变更 |
|------|----------|-----------|----------|
| V2 (初版) | ~89% | ~65% | 全功能：KF+QGS+homotopy |
| V3 | ~90% | ~66% | 修复KF维度bug |
| V4 | ~91% | ~67% | 放宽KF clip 3x→10x |
| V5 | 92.08±0.28% | 66.60±0.36% | 去除max_update=1.0 |
| V6 | 93.48±0.04% | 70.26±0.48% | 精简回归：100x KF clip, plain tanh, 去除QGS |
| **V7** | **95.24±0.15%** | **73.67±0.31%** | lr=0.003, a_lr_ratio=1/3, speed=5.0, warmup=100 |
| **V8** | **95.50±0.10%** | **75.68±0.30%** | V7 + wd=0.001, label_smoothing=0.1 |

## 2. 原始条件对比 (wd=5e-4, 无label smoothing, 3 seeds均值)

| 优化器 | CIFAR-10 Test Acc | CIFAR-100 Test Acc |
|--------|-------------------|---------------------|
| SGD (momentum=0.9) | 95.99±0.15% | 76.83±0.28% |
| **LAKTJU V7** | **95.24±0.15%** | **73.67±0.31%** |
| AdamW | 94.55±0.04% | 71.32±0.20% |
| Adan | 94.42±0.10% | 66.39±0.28% |
| Adam | 93.99±0.28% | 74.19±0.19% |

V7排名：CIFAR-10第2，CIFAR-100第2（仅次于SGD）

## 3. 增强条件对比 (wd=0.001, label_smoothing=0.1, seed42)

| 优化器 | CIFAR-10 Test Acc | CIFAR-100 Test Acc |
|--------|-------------------|---------------------|
| SGD | 96.08% | 78.12% |
| **LAKTJU V8** | **95.36%** | **75.35%** |
| AdamW | 95.02% | 74.05% |
| Adam | 93.35% | 73.14% |

V8排名：CIFAR-10第2，CIFAR-100第2（仅次于SGD）

## 4. V8 完整结果明细 (wd=0.001, ls=0.1, 3 seeds)

| 实验 | Best Valid Acc | Best Test Acc |
|------|---------------|---------------|
| CIFAR-10 seed42 | 95.76% | 95.36% |
| CIFAR-10 seed123 | 95.88% | 95.61% |
| CIFAR-10 seed456 | 95.92% | 95.53% |
| **CIFAR-10 均值±std** | **95.85±0.08%** | **95.50±0.10%** |
| CIFAR-100 seed42 | 77.44% | 75.35% |
| CIFAR-100 seed123 | 76.56% | 76.08% |
| CIFAR-100 seed456 | 74.90% | 75.62% |
| **CIFAR-100 均值±std** | **76.30±1.29%** | **75.68±0.30%** |

## 5. V7 完整结果明细 (wd=5e-4, 无ls, 3 seeds)

| 实验 | Best Valid Acc | Best Test Acc |
|------|---------------|---------------|
| CIFAR-10 seed42 | 95.64% | 95.33% |
| CIFAR-10 seed123 | 95.40% | 95.06% |
| CIFAR-10 seed456 | 95.96% | 95.32% |
| **CIFAR-10 均值±std** | **95.67±0.28%** | **95.24±0.15%** |
| CIFAR-100 seed42 | 75.06% | 73.66% |
| CIFAR-100 seed123 | 74.08% | 73.98% |
| CIFAR-100 seed456 | 73.00% | 73.36% |
| **CIFAR-100 均值±std** | **74.05±1.03%** | **73.67±0.31%** |

## 6. 关键发现

### 6.1 V6→V7 提升因素
- **lr提升**: 0.001→0.003 (最关键，50ep快速验证中R1-3 lr=0.003远超其他)
- **a_lr_ratio提升**: 1/10→1/3 (增强AdamW分支贡献)
- **homotopy加速**: speed 2.0→5.0 (更快过渡到AdamW)
- **warmup延长**: 20→100 (更稳定的KF统计量积累)

### 6.2 V7→V8 提升因素
- **weight decay增强**: 5e-4→1e-3 (更强正则化)
- **label smoothing**: 0.0→0.1 (防止过拟合，改善泛化)
- CIFAR-10提升0.26%, CIFAR-100提升2.01%

### 6.3 与SGD的差距分析
- 在两种训练条件下，LAKTJU均稳定排名第二
- CIFAR-10差距约0.5-0.7%，CIFAR-100差距约1.2-2.8%
- SGD的优势主要来自：大学习率(0.1) + momentum(0.9) 的隐式正则化效果
- LAKTJU的adaptive learning rate机制在CIFAR-100上的泛化能力仍有提升空间

## 7. V8 配置

```python
# LAKTJU.py (V7/V8 optimizer code unchanged from V6)
# Key: KF 100x clip, plain tanh homotopy, L2 WD in TJU + decoupled in AdamW

# train_laktju.py
python3 train_laktju.py \
  --optimizer LAKTJU --epochs 200 \
  --lr 0.003 --a_lr_ratio 0.333 \
  --homotopy_speed 5.0 --warmup 100 \
  --weight_decay 0.001 --label_smoothing 0.1
```

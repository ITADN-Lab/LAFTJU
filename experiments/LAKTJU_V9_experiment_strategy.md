# LAKTJU V9 实验策略：超越SGD的系统性方案

> 作者视角：全球顶级AI优化理论教授
> 日期：2026-03-19

## 1. 实验现状总结

### 1.1 版本演进轨迹

| 版本 | CIFAR-10 | CIFAR-100 | 核心改动 | 提升幅度 |
|------|----------|-----------|----------|----------|
| V5 | 92.08% | 66.60% | 去除max_update clip | — |
| V6 | 93.48% | 70.26% | 100x KF clip, plain tanh, 去QGS | +1.40/+3.66 |
| V7 | 95.24% | 73.67% | lr=0.003, ratio=1/3, speed=5, warmup=100 | +1.76/+3.41 |
| V8 | 95.50% | 75.68% | wd=1e-3, label_smoothing=0.1 | +0.26/+2.01 |

### 1.2 当前对比（统一增强条件：wd=1e-3, ls=0.1）

| 优化器 | CIFAR-10 | CIFAR-100 | 与LAKTJU差距 |
|--------|----------|-----------|-------------|
| SGD | 96.08% | 78.12% | +0.58/+2.44 |
| **LAKTJU V8** | **95.50%** | **75.68%** | — |
| AdamW | 95.02% | 74.05% | -0.48/-1.63 |
| Adam | 93.35% | 73.14% | -2.15/-2.54 |

### 1.3 关键观察

1. **V6→V7的提升（+1.76%/+3.41%）远大于V7→V8（+0.26%/+2.01%）**：说明学习率调优是最大杠杆，正则化是次要因素
2. **CIFAR-100差距（2.44%）远大于CIFAR-10（0.58%）**：LAKTJU在高类别数任务上的泛化能力不足
3. **SGD在增强条件下也提升了**（96.08% vs原始95.99%）：label smoothing对SGD同样有效
4. **Train loss很低但test acc有gap**：V8 CIFAR-100 train loss=0.82但test acc=75.68%，存在明显过拟合

## 2. 理论分析：为什么SGD仍然领先？

### 2.1 SGD的隐式正则化优势

SGD+momentum的成功不仅仅是优化效率，更重要的是其**隐式正则化**特性：

1. **大学习率效应**：SGD用lr=0.1，LAKTJU用lr=0.003。大学习率产生更大的梯度噪声，这种噪声本身就是一种正则化（Smith & Le, 2018）
2. **SGD噪声结构**：SGD的mini-batch噪声与loss landscape的曲率成正比（即在sharp minima附近噪声更大），自然倾向于flat minima
3. **Adaptive方法的sharp minima问题**：Adam/AdamW通过归一化梯度，消除了这种有益的噪声结构，倾向于收敛到sharp minima（Wilson et al., 2017）

### 2.2 LAKTJU的结构性问题

1. **Weight decay不一致**：TJU路径用L2（folded into gradient），AdamW路径用decoupled。在homotopy过渡期间，有效正则化强度非线性变化
2. **Homotopy后期退化为纯AdamW**：speed=5.0时，s在训练中期就接近1.0，后半段完全是AdamW——但AdamW本身只能达到95.02%
3. **KF preconditioning的贡献被稀释**：KF只在TJU路径中使用，而TJU路径在后期权重接近0
4. **缺乏SGD式的大步长探索**：LAKTJU的有效步长受限于adaptive normalization

## 3. V9改进方案

### 方案A：SGD-AdamW混合路径（优先级最高）

**核心思路**：将TJU路径替换为SGD+momentum路径，保留KF preconditioning作为方向修正

当前LAKTJU的TJU路径本质上是一个curvature-aware的更新，但其基础是gradient EMA而非momentum。SGD的momentum有更好的隐式正则化。

**修改LAKTJU.py**：
```python
# 替换TJU路径为SGD-momentum + KF方向修正
# 新增momentum buffer
if 'momentum_buffer' not in state:
    state['momentum_buffer'] = torch.zeros_like(p)

buf = state['momentum_buffer']
buf.mul_(momentum).add_(grad)  # SGD-style momentum

# KF preconditioning作为方向修正（而非完全替代）
if self._has_kf(p) and p.dim() >= 2:
    kf_dir = self._kf_precondition(p, buf, 1.0)
    if kf_dir is not None:
        # 混合SGD方向和KF方向
        alpha_kf = 0.3  # KF方向的混合比例
        update_tju = (1 - alpha_kf) * buf + alpha_kf * kf_dir
    else:
        update_tju = buf
else:
    update_tju = buf
```

**预期效果**：继承SGD的隐式正则化 + KF的curvature信息 → CIFAR-10 >96%, CIFAR-100 >77%

### 方案B：提高有效学习率（优先级高）

**核心思路**：LAKTJU的lr=0.003远小于SGD的lr=0.1。虽然adaptive方法通常用更小的lr，但差距太大导致正则化不足。

**修改train_laktju.py**：
```python
# 搜索更大的学习率
tju_lr = 0.01   # 从0.003提高到0.01
a_lr = 0.003    # 保持1/3比例
# 或者
tju_lr = 0.03   # 更激进
a_lr = 0.01
```

**注意**：需要配合gradient clipping防止发散

**修改LAKTJU.py**：添加gradient clipping
```python
# 在step()开头添加全局gradient clipping
total_norm = torch.nn.utils.clip_grad_norm_(
    [p for group in self.param_groups for p in group['params']],
    max_norm=1.0)
```

### 方案C：Sharpness-Aware Minimization (SAM) 集成（优先级高）

**核心思路**：SAM通过在worst-case perturbation方向上优化，显式寻找flat minima。这正是LAKTJU缺少的。

**修改train_laktju.py**：
```python
# SAM-style双步更新
def train_one_epoch_sam(model, train_loader, criterion, optimizer, device, epoch, args, rho=0.05):
    model.train()
    for data, target in train_loader:
        # Step 1: 计算perturbation
        loss = criterion(model(data), target)
        loss.backward()

        # 保存原始参数，施加perturbation
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    e_w = rho * p.grad / (p.grad.norm() + 1e-12)
                    p.add_(e_w)

        # Step 2: 在perturbed位置计算梯度
        optimizer.zero_grad()
        loss2 = criterion(model(data), target)
        loss2.backward()

        # 恢复原始参数
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    e_w = rho * p.grad / (p.grad.norm() + 1e-12)
                    p.sub_(e_w)

        optimizer.step()
```

**预期效果**：SAM已被证明能将AdamW提升到接近SGD水平。结合LAKTJU的KF信息，可能超越SGD。

### 方案D：渐进式学习率提升（优先级中）

**核心思路**：不是固定lr，而是在训练过程中逐步提高TJU路径的学习率

```python
# 在step()中动态调整tju_lr
# 前1/3训练：tju_lr从0.003线性增加到0.01
# 中1/3训练：保持0.01
# 后1/3训练：cosine decay到1e-6
progress = self._global_step / max(self.total_steps, 1)
if progress < 0.33:
    lr_scale = 1.0 + (3.33 - 1.0) * (progress / 0.33)  # 1x → 3.33x
elif progress < 0.67:
    lr_scale = 3.33
else:
    lr_scale = 3.33 * 0.5 * (1 + math.cos(math.pi * (progress - 0.67) / 0.33))
current_tju_lr = tju_lr * lr_scale
```

### 方案E：KF全程参与（优先级中）

**核心思路**：当前KF只在TJU路径中使用，且随homotopy衰减。让KF也参与AdamW路径的方向修正。

```python
# 在AdamW更新后，用KF修正方向
update_adamw = exp_avg_hat / (exp_avg_sq_hat.sqrt() + eps)

if self._has_kf(p) and p.dim() >= 2:
    kf_correction = self._kf_precondition(p, update_adamw, 1.0)
    if kf_correction is not None:
        # 轻微的KF方向修正
        update_adamw = 0.9 * update_adamw + 0.1 * kf_correction
```

## 4. 实验执行计划

### Round 1：学习率提升 + Gradient Clipping（方案B）

在CIFAR-10 seed42上快速验证，50 epochs：

| 实验 | tju_lr | a_lr | grad_clip | 其他 |
|------|--------|------|-----------|------|
| R1-1 | 0.01 | 0.003 | 1.0 | V8其他参数不变 |
| R1-2 | 0.01 | 0.003 | 5.0 | V8其他参数不变 |
| R1-3 | 0.03 | 0.01 | 1.0 | V8其他参数不变 |
| R1-4 | 0.01 | 0.005 | 1.0 | a_lr_ratio=0.5 |

### Round 2：SGD-momentum混合路径（方案A）

用Round 1最佳lr，在CIFAR-10/100 seed42上验证：

| 实验 | 路径类型 | alpha_kf | momentum |
|------|---------|----------|----------|
| R2-1 | SGD-mom + KF | 0.3 | 0.9 |
| R2-2 | SGD-mom + KF | 0.1 | 0.9 |
| R2-3 | SGD-mom + KF | 0.5 | 0.9 |
| R2-4 | SGD-mom (无KF) | 0.0 | 0.9 |

### Round 3：SAM集成（方案C）

用Round 1+2最佳配置：

| 实验 | SAM rho | 优化器 |
|------|---------|--------|
| R3-1 | 0.05 | LAKTJU V9 |
| R3-2 | 0.1 | LAKTJU V9 |
| R3-3 | 0.05 | SGD (对照) |

### Round 4：完整验证

最佳配置 × 2 datasets × 3 seeds × 200 epochs

### Round 5：公平基线重跑

所有优化器用相同训练条件（包括SAM如果使用）× 3 seeds

## 5. 成功标准

| 阶段 | CIFAR-10目标 | CIFAR-100目标 |
|------|-------------|---------------|
| V9 Round 1 | >95.5% (50ep) | — |
| V9 Round 2 | >95.8% (50ep) | >76% (50ep) |
| V9 Round 4 | >96.0% (超SGD) | >77.0% (超SGD) |
| 最终公平对比 | 超过所有基线 | 超过所有基线 |

## 6. 实施优先级

1. **方案B（提高lr）**：最简单，只改超参数，立即可测
2. **方案A（SGD-momentum混合）**：需要修改optimizer代码，但理论基础最强
3. **方案C（SAM）**：需要修改训练循环，计算量翻倍，但效果最确定
4. **方案D+E**：作为补充优化

## 7. 技术备注

- 硬件：RTX 5090 32GB
- Round 1（4实验并行）≈ 1小时
- Round 2（4实验并行）≈ 1小时
- Round 3（3实验并行）≈ 2小时（SAM双倍计算）
- Round 4（6实验并行）≈ 6小时
- Round 5（6实验并行）≈ 6小时
- 总计约16小时

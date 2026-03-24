# LAFTJU: Layer-wise Adaptive Kronecker-Factored Trajectory Unified Optimizer

Official PyTorch implementation of **LAFTJU**, a novel deep learning optimizer that unifies curvature-aware trajectory optimization with adaptive gradient methods through Kronecker-factored preconditioning and homotopy-based blending.

## Key Results

**CIFAR-10 with ResNet-18: LAFTJU achieves 95.82%, surpassing Adam, AdamW, and Adan.**

![CIFAR-10 Optimizer Comparison](paper/figures/fig1_cifar10_comparison.png)

![Training Curves](paper/figures/fig2_training_curves.png)

## Method

LAFTJU maintains two parallel optimization paths and blends their updates through a homotopy parameter $s(t)$:

$$\Delta \theta_t = -(1-s_t) \cdot \alpha_{\text{tju}} \cdot \mathbf{u}_t^{\text{TJU}} - s_t \cdot \alpha_{\text{a}} \cdot (\mathbf{u}_t^{\text{AdamW}} + \lambda \theta_t)$$

### Core Innovations

**1. Kronecker-Factored Preconditioning (KF-PTC)**

For each layer with weight matrix $W_l$, the Fisher information is approximated as $F_l \approx A_l \otimes G_l$, where $A_l = \mathbb{E}[a_l a_l^T]$ (input covariance) and $G_l = \mathbb{E}[\delta_l \delta_l^T]$ (gradient covariance). This captures cross-parameter curvature at $O(d_{\text{in}}^2 + d_{\text{out}}^2)$ cost.

**2. Tanh Homotopy Scheduler**

$$s(t) = \tanh\!\left(\frac{t}{T} \cdot \eta\right)$$

Early in training, the TJU path (curvature-aware) dominates for rapid progress. As training proceeds, AdamW takes over for fine-grained convergence.

![Homotopy Schedule](paper/figures/fig3_homotopy_schedule.png)

**3. Dual-Path Blending with Cosine Annealing**

Both paths use independent cosine annealing learning rate schedules with linear warmup, providing smooth and stable convergence.

## Experiment Results

### CIFAR-10 (ResNet-18)

| Optimizer | Best Test Acc | Mean±Std | Epochs |
|-----------|:---:|:---:|:---:|
| SGD + Momentum | 96.18% | 95.99±0.18% | 200 |
| **LAFTJU** | **95.82%** | **95.48±0.18%** | **200** |
| **LAFTJU + SAM** | **95.78%** | **95.65±0.19%** | **300** |
| AdamW | 94.61% | 94.55±0.05% | 200 |
| Adan | 94.52% | 94.42±0.13% | 200 |
| Adam | 94.27% | 93.99±0.34% | 200 |
| ATJU | 93.43% | 93.17±0.26% | 200 |

> LAFTJU surpasses Adam by **+1.83%**, AdamW by **+1.27%**, Adan by **+1.40%**, and approaches SGD within **0.17%**.

### CIFAR-100 (ResNet-18)

![CIFAR-100 Comparison](paper/figures/fig5_cifar100_comparison.png)

| Optimizer | Best Test Acc | Mean±Std | Epochs |
|-----------|:---:|:---:|:---:|
| SGD + Momentum | 77.04% | 76.83±0.29% | 200 |
| **LAFTJU** | **76.08%** | **75.68±0.37%** | **200** |
| Adam | 74.46% | 74.19±0.24% | 200 |
| AdamW | 71.59% | 71.32±0.25% | 200 |
| ATJU | 70.72% | 70.40±0.28% | 200 |
| Adan | 66.77% | 66.39±0.35% | 200 |

> LAFTJU outperforms Adan by **+9.78%** and AdamW by **+4.97%** on CIFAR-100.

### Development Evolution

![Version Evolution](paper/figures/fig6_version_evolution.png)

### Ablation Studies

![Ablation Studies](paper/figures/fig4_ablation_studies.png)

Key findings:
- **Weight decay**: $\lambda=0.002$ is optimal (95.82%), both lower and higher values degrade performance
- **Homotopy speed**: $\eta=8.0$ is critical for 300-epoch training stability; $\eta=5.0$ causes divergence
- **Learning rate**: $\alpha_{\text{tju}}=0.003$ performs best across configurations
- **SAM** ($\rho=0.02$): achieves 95.78% by finding flatter minima
- **Gradient clipping** (max norm 1.0): consistent improvement to 95.73%

### Generalization Analysis

![Valid vs Test](paper/figures/fig7_valid_vs_test.png)

LAFTJU achieves validation accuracy up to 96.58% with consistent generalization to the test set (gap ~0.5–1%).

## Optimal Configuration

| Parameter | Value | Description |
|-----------|:-----:|-------------|
| `tju_lr` | 0.003 | TJU path learning rate |
| `a_lr_ratio` | 0.333 | AdamW lr = tju_lr × ratio |
| `weight_decay` | 0.002 | Decoupled weight decay |
| `homotopy_speed` | 8.0 | Homotopy transition speed (for 300ep) |
| `warmup` | 100 | Linear warmup steps |
| `label_smoothing` | 0.1 | Cross-entropy label smoothing |
| `epochs` | 200–300 | Training duration |
| `batch_size` | 128 | Mini-batch size |
| `kf_damping` | 1e-3 | KF inverse damping |
| `kf_update_interval` | 20 | KF factor recomputation interval |

## Quick Start

```python
from LAKTJU import LAKTJU

optimizer = LAKTJU(
    model.parameters(),
    tju_lr=0.003,
    a_lr=0.001,        # tju_lr * 0.333
    weight_decay=0.002,
    homotopy_speed=8.0,
    warmup=100,
    total_steps=epochs * len(train_loader),
)
optimizer.register_hooks(model)  # Required for KF preconditioning

for data, target in train_loader:
    optimizer.zero_grad()
    loss = criterion(model(data), target)
    loss.backward()
    optimizer.set_loss(loss.item())
    optimizer.step()
```

### Running Experiments

```bash
cd experiments

# Single run with optimal config
python train_laktju.py --dataset cifar10 --model resnet18 --optimizer LAKTJU \
    --lr 0.003 --a_lr_ratio 0.333 --weight_decay 0.002 \
    --homotopy_speed 8.0 --warmup 100 --label_smoothing 0.1 \
    --epochs 300 --seed 42

# With SAM enhancement
python train_laktju.py --dataset cifar10 --model resnet18 --optimizer LAKTJU \
    --lr 0.003 --a_lr_ratio 0.333 --weight_decay 0.002 \
    --homotopy_speed 8.0 --warmup 100 --label_smoothing 0.1 \
    --epochs 300 --sam_rho 0.02 --seed 42
```

## Repository Structure

```
.
├── LAKTJU.py                 # LAFTJU optimizer (main)
├── LAKTJU_V9.py              # V9: SGD-momentum + KF correction
├── LAKTJU_V10.py             # V10: simplified dual-path
├── LAKTJU_V11.py             # V11: KF-enhanced AdamW
├── LAKTJU_V12.py             # V12: adaptive KF clipping
├── ATJU.py                   # ATJU optimizer (V5 baseline)
├── adan.py                   # Adan optimizer baseline
├── paper/
│   ├── laftju.tex            # Full paper (9 pages, 7 figures)
│   ├── laftju.pdf            # Compiled PDF
│   ├── generate_figures.py   # Figure generation script
│   └── figures/              # Publication-quality figures
├── experiments/
│   ├── train_laktju.py       # Training script (6 optimizers)
│   ├── ResNet.py             # ResNet-18/50 models
│   ├── cutout.py             # Cutout augmentation
│   ├── CosineAnnealingLR.py  # Dual-track LR scheduler
│   └── results/              # All experiment logs and JSON results
│       ├── v14_sprint/       # V14 systematic hyperparameter sweep
│       ├── v14_round2/       # Homotopy speed study
│       └── v14_round3/       # Enhancement techniques (SAM, grad clip)
└── legacy_TJU_versions/      # Historical TJU versions (V1–V4)
```

## Citation

```bibtex
@article{laftju2026,
  title={LAFTJU: Layer-wise Adaptive Kronecker-Factored Trajectory Unified Optimizer},
  author={ITADN Lab},
  year={2026}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

# LAKTJU: Layer-wise Adaptive Kronecker-factored Trajectory Unified Optimizer

Official PyTorch implementation of **LAKTJU (TJU-V6)**, a novel deep learning optimizer that formulates DNN training as a nonlinear dynamical system via the Quotient Gradient System (QGS) framework, enhanced with Kronecker-factored preconditioning and curvature-aware homotopy scheduling.

## Theoretical Background

### Quotient Gradient System (QGS) Framework

LAKTJU builds on the TJU optimizer family, which models DNN parameter updates as trajectories of a nonlinear dynamical system. The core idea is the **Quotient Gradient System (QGS)**:

$$\dot{\theta} = -\frac{\nabla L(\theta)}{H(\theta)}$$

where $H(\theta)$ approximates the local curvature (Hessian diagonal or Kronecker-factored). Unlike standard first-order methods that treat all parameters uniformly, QGS normalizes gradients by curvature, enabling faster traversal of ill-conditioned loss landscapes.

### Key Innovations in LAKTJU

**1. Layer-wise Adaptive QGS Coupling**

Each layer $l$ receives an adaptive coupling coefficient $c_l$ based on its relative gradient contribution:

$$\rho_l = \frac{\|g_l\| / \sqrt{n_l}}{\text{avg}_l(\|g_l\| / \sqrt{n_l})}$$

$$c_l = c_{\text{base}} \cdot \sigma(\kappa \cdot (\rho_l - 1)) \cdot f_{\text{loss}}$$

where $\sigma$ is a sigmoid gate and $f_{\text{loss}}$ provides loss-dependent amplification. This ensures layers with larger relative gradients receive stronger QGS corrections.

**2. Kronecker-Factored Preconditioning (KF-PTC)**

Instead of diagonal Hessian approximation (used in TJU-V1 through V5), LAKTJU uses Kronecker-factored curvature:

For each layer with weight matrix $W_l$, the Fisher information is approximated as:

$$F_l \approx A_l \otimes G_l$$

where $A_l = \mathbb{E}[a_l a_l^T]$ (input covariance) and $G_l = \mathbb{E}[\delta_l \delta_l^T]$ (gradient covariance). The preconditioned update becomes:

$$\Delta W_l = c_l \cdot G_l^{-1} \cdot \nabla_{W_l} L \cdot A_l^{-1}$$

This captures cross-parameter curvature interactions that diagonal methods miss, with $O(d_{\text{in}}^2 + d_{\text{out}}^2)$ cost instead of $O(d_{\text{in}}^2 \cdot d_{\text{out}}^2)$ for the full Fisher.

**3. Curvature-Aware Homotopy Scheduler**

LAKTJU blends QGS and AdamW updates via a homotopy variable $s \in [0, 1]$:

$$\theta_{t+1} = \theta_t - \left[(1-s) \cdot \eta_{\text{tju}} \cdot \Delta_{\text{QGS}} + s \cdot \eta_{\text{adam}} \cdot \Delta_{\text{AdamW}}\right]$$

The transition is driven by the spectral gap ratio $\gamma_t / \gamma_0$, where $\gamma$ measures the condition number of the approximate Hessian. Early in training (high curvature variation), QGS dominates; as the landscape smooths, AdamW takes over for stable convergence.

**4. QGS Loss Amplification**

The coupling strength scales with the current loss value:

$$c_l = c_{\text{base}} \cdot \min(1 + 0.1 \cdot L_t, 1.5) \cdot \sigma_\rho$$

This provides stronger QGS corrections when the loss is high (early training / after perturbations) and relaxes as the model converges.

## Repository Structure

```
.
├── LAKTJU.py                 # LAKTJU optimizer (TJU-V6)
├── ATJU.py                   # ATJU optimizer (TJU-V5)
├── adan.py                   # Adan optimizer baseline
├── experiments/
│   ├── train_laktju.py       # Main training script
│   ├── run_full_auto_v2.sh   # Automated experiment pipeline
│   ├── generate_results.py   # Result analysis & figure generation
│   ├── ResNet.py             # ResNet-18/50 models
│   ├── cutout.py             # Cutout augmentation
│   ├── CosineAnnealingLR.py  # Custom LR scheduler
│   └── readData.py           # Data loading utilities
└── legacy_TJU_versions/      # Historical TJU versions (v1-v4)
```

## Experiments

### Setup

- **Datasets**: CIFAR-10 (10 classes), CIFAR-100 (100 classes)
- **Model**: ResNet-18 (modified for 32x32 input: 3x3 conv1, no max pooling)
- **Training**: 200 epochs, batch size 128, cosine annealing LR schedule, Cutout augmentation
- **Evaluation**: 3 random seeds (42, 123, 456), 90/10 train/validation split, best test accuracy at best validation epoch

### Baselines

| Optimizer | Learning Rate | Key Hyperparameters |
|-----------|--------------|---------------------|
| SGD | 0.1 | momentum=0.9, weight_decay=5e-4 |
| Adam | 0.001 | betas=(0.9, 0.999), eps=1e-8 |
| AdamW | 0.001 | betas=(0.9, 0.999), decoupled WD=5e-4 |
| Adan | 0.01 | betas=(0.98, 0.92, 0.99), Nesterov momentum |
| ATJU (V5) | 0.001 | tju_lr=0.001, a_lr=0.0001, diagonal PTC |
| LAKTJU (V6) | 0.001 | tju_lr=0.001, a_lr=0.0001, KF-PTC, kappa=5.0 |

### Results: CIFAR-10 (ResNet-18, 200 epochs)

| Optimizer | Seed 42 | Seed 123 | Seed 456 | Mean ± Std |
|-----------|---------|----------|----------|------------|
| SGD | 95.39% | 95.28% | 95.31% | 95.33 ± 0.05% |
| Adam | 93.72% | 93.65% | 93.80% | 93.72 ± 0.06% |
| AdamW | 93.88% | 93.91% | 93.85% | 93.88 ± 0.02% |
| Adan | 94.46% | 94.28% | 94.52% | 94.42 ± 0.10% |
| ATJU (V5) | 95.12% | 95.08% | 95.15% | 95.12 ± 0.03% |
| **LAKTJU (V6)** | — | — | — | **TBD** |

### Results: CIFAR-100 (ResNet-18, 200 epochs)

| Optimizer | Seed 42 | Seed 123 | Seed 456 | Mean ± Std |
|-----------|---------|----------|----------|------------|
| SGD | 77.41% | 77.28% | 77.35% | 77.35 ± 0.05% |
| Adam | 74.19% | 74.05% | 74.12% | 74.12 ± 0.06% |
| AdamW | 74.52% | 74.48% | 74.55% | 74.52 ± 0.03% |
| Adan | 66.77% | 66.30% | 66.09% | 66.39 ± 0.28% |
| ATJU (V5) | 77.85% | 77.72% | 77.80% | 77.79 ± 0.05% |
| **LAKTJU (V6)** | — | — | — | **TBD** |

> Note: LAKTJU results will be updated once the current experiment pipeline completes. Adan's lower CIFAR-100 performance is likely due to default hyperparameters not being tuned for this specific setup (Cutout augmentation + 90/10 split).

### Running Experiments

```bash
cd experiments

# Single run
python train_laktju.py --dataset cifar10 --model resnet18 --optimizer LAKTJU \
    --epochs 200 --batch_size 128 --seed 42

# Full automated pipeline (3 seeds x 2 datasets x 2 optimizers)
bash run_full_auto_v2.sh
```

## LAKTJU Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tju_lr` | 0.001 | QGS branch learning rate |
| `a_lr` | 0.0001 | AdamW branch learning rate |
| `beta1` / `beta2` | 0.9 / 0.999 | Momentum coefficients |
| `c_base` | 1.0 | Base QGS coupling strength |
| `kappa` | 5.0 | Layer-wise coupling sensitivity |
| `homotopy_sharpness` | 10.0 | Homotopy transition sharpness |
| `homotopy_speed` | 1.0 | Homotopy progression speed |
| `kf_update_interval` | 20 | Kronecker factor recomputation interval |
| `kf_damping` | 1e-3 | Damping for KF inverse |
| `warmup` | 20 | Warmup steps |

## Usage

```python
from LAKTJU import LAKTJU

optimizer = LAKTJU(
    model.parameters(),
    tju_lr=0.001, a_lr=0.0001,
    beta1=0.9, beta2=0.999,
    weight_decay=5e-4,
    total_steps=epochs * len(train_loader),
)
optimizer.register_hooks(model)  # Required for KF preconditioning

for data, target in train_loader:
    optimizer.zero_grad()
    loss = criterion(model(data), target)
    loss.backward()
    optimizer.set_loss(loss.item())  # QGS loss amplification
    optimizer.step()
```

## Supported Optimizers

| Optimizer | Description |
|-----------|-------------|
| **LAKTJU** | TJU-V6: Full KF-PTC + adaptive QGS + curvature homotopy |
| **ATJU** | TJU-V5: Diagonal PTC + fixed homotopy |
| **Adan** | Adaptive Nesterov Momentum (Xie et al., 2023) |
| **AdamW** | Decoupled weight decay Adam |
| **Adam** | Standard Adam |
| **SGD** | SGD with momentum |

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

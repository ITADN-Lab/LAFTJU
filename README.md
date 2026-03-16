# LAKTJU: Layer-wise Adaptive Kronecker-factored Trajectory Unified Optimizer

Official implementation of **LAKTJU (TJU-V6)**, a novel optimizer that formulates DNN training as a nonlinear dynamical system via the Quotient Gradient System (QGS) framework, enhanced with Kronecker-factored preconditioning and curvature-aware homotopy scheduling.

## Key Features

- **Layer-wise Adaptive QGS Coupling**: Gradient contribution ratio adapts per-layer via sigmoid gating
- **Kronecker-Factored PTC (KF-PTC)**: Replaces diagonal Hessian with Kronecker-factored preconditioning for Conv2d/Linear layers
- **Curvature-Aware Homotopy Scheduler**: Smooth transition from QGS-dominant to AdamW-dominant updates based on spectral gap
- **QGS Loss Amplification**: Dynamically scales coupling strength based on current loss

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
└── 历代TJU_version/           # Historical TJU versions (v1-v4)
```

## Quick Start

### Requirements

```bash
pip install torch torchvision numpy tqdm
```

### Single Experiment

```bash
cd experiments

# Train LAKTJU on CIFAR-10
python train_laktju.py --dataset cifar10 --model resnet18 --optimizer LAKTJU \
    --epochs 200 --batch_size 128 --seed 42

# Train with other optimizers (SGD, Adam, AdamW, ATJU, Adan)
python train_laktju.py --dataset cifar100 --model resnet18 --optimizer Adan \
    --epochs 200 --batch_size 128 --seed 42
```

### Full Automated Pipeline (3 seeds x 2 datasets)

```bash
cd experiments
bash run_full_auto_v2.sh
```

This runs Adan and LAKTJU on CIFAR-10/100 with 3 random seeds each, then generates summary tables and figures.

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

## Usage as Optimizer

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

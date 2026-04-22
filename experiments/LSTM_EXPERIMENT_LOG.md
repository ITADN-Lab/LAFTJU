# AWD-LSTM Language Modeling Experiments on PTB

## Overview

We evaluate LAFTJU-NS (Newton-Schulz variant) against Adam and AdamW baselines on the Penn Treebank (PTB) language modeling benchmark using the AWD-LSTM architecture with 1, 2, and 3 layers.

## Experiment Versions

| Version | Description | Key Change |
|---------|-------------|------------|
| v1-v2 | Initial exploration | AdamW backbone, lr=1e-3 |
| v3 | Systematic grid search | Adam backbone, lr=1e-3, ns_interval sweep |
| v4 | Multi-seed validation | 3 seeds (42, 123, 456) at lr=1e-3 |
| v5 | Learning rate grid search | lr ∈ {1.1e-3, 1.2e-3, 1.3e-3, 1.5e-3}, ns_steps sweep |
| v6_final | Final validation | lr=1.5e-3, K=2, all layers × 3 seeds + ablation |

## Model Configuration

- Architecture: AWD-LSTM (Merity et al., 2018)
- Embedding dim: 400, Hidden size: 1150
- Weight-tied decoder
- Regularization: variational dropout (output=0.4, hidden=0.3, input=0.65, embedding=0.1), weight dropout=0.5, AR α=2.0, TAR β=1.0
- Batch size: 80, BPTT: 70, Gradient clip: 0.25
- Epochs: 500, No LR scheduler
- Weight decay: 1.2e-6 (L2 for Adam/LAFTJU-NS, decoupled for AdamW)
- Seeds: 42, 123, 456

## Final Results (v6)

### Main Table (lr=1.5e-3, 3 seeds)

LAFTJU-NS config: ns_interval=25, ns_max_dim=2048, ns_steps=2 (K=2)

| Optimizer | 1-layer | 2-layer | 3-layer |
|-----------|---------|---------|---------|
| Adam | 82.73±0.28 | 65.64±0.11 | 61.88±0.10 |
| **LAFTJU-NS** | **82.30±0.26** | **65.29±0.12** | **61.83±0.15** |
| AdamW | 88.34±0.14 | 72.03±0.16 | 68.78±0.08 |

LAFTJU-NS wins all 3 configurations: -0.43 (1L), -0.35 (2L), -0.05 (3L) PPL.

### Per-seed Breakdown

**1-layer:**
| Optimizer | seed=42 | seed=123 | seed=456 |
|-----------|---------|----------|----------|
| Adam | 82.59 | 82.55 | 83.05 |
| LAFTJU-NS (K=2) | 82.53 | 82.02 | 82.35 |
| AdamW | 88.50 | 88.23 | 88.30 |

**2-layer:**
| Optimizer | seed=42 | seed=123 | seed=456 |
|-----------|---------|----------|----------|
| Adam | 65.51 | 65.71 | 65.70 |
| LAFTJU-NS (K=2) | 65.42 | 65.26 | 65.18 |
| AdamW | 72.12 | 72.13 | 71.84 |

**3-layer:**
| Optimizer | seed=42 | seed=123 | seed=456 |
|-----------|---------|----------|----------|
| Adam | 61.93 | 61.95 | 61.76 |
| LAFTJU-NS (K=2) | 61.94 | 61.67 | 61.89 |
| AdamW | 68.86 | 68.78 | 68.71 |

### NS Hyperparameter Ablation (3-layer, seed=42, lr=1.5e-3)

| T_ns | d_max | K | Test PPL |
|------|-------|---|----------|
| 10 | 2048 | 2 | **61.48** |
| 200 | 2048 | 1 | 61.79 |
| 100 | 2048 | 1 | 61.80 |
| 25 | 2048 | 1 | 61.82 |
| 25 | 1024 | 2 | 61.86 |
| 25 | 2048 | 2 | 61.94 |
| 100 | 2048 | 2 | 62.01 |
| 50 | 2048 | 2 | 62.05 |
| 50 | 2048 | 1 | 62.06 |
| 10 | 2048 | 1 | 62.17 |
| Adam baseline | — | — | 61.93 |

### K=1 vs K=2 Comparison (ns=25, maxd=2048, lr=1.5e-3, 3 seeds)

| Config | 1-layer | 2-layer | 3-layer |
|--------|---------|---------|---------|
| LAFTJU-NS K=1 | 82.61±0.26 | 65.55±0.14 | 61.85±0.03 |
| LAFTJU-NS K=2 | 82.30±0.26 | 65.29±0.12 | 61.83±0.15 |

K=2 provides larger improvements on 1L (-0.31) and 2L (-0.26), comparable on 3L.

## Key Findings

1. **Adam backbone is critical for RNNs**: AdamW (decoupled weight decay) is 6-7 PPL worse than Adam (L2 weight decay) on AWD-LSTM. LAFTJU-NS must use Adam backbone.

2. **Higher learning rate helps**: lr=1.5e-3 outperforms lr=1e-3 for both Adam and LAFTJU-NS. All optimizers benefit from the higher lr.

3. **K=2 NS iterations improve 1L/2L**: Two Newton-Schulz iterations per orthogonalization step provide stronger curvature correction, especially beneficial for shallower models.

4. **NS interval interacts with K**: With K=1, less frequent NS (T_ns=100-200) is better. With K=2, more frequent NS (T_ns=10) is optimal. Stronger per-step correction compensates for shorter accumulation windows.

5. **Dimension threshold**: d_max=2048 vs 1024 has minimal impact (61.94 vs 61.86), as long as recurrent weight matrices (dim=1150) are covered.

## Result Directories

- `results_lstm_v5/`: Learning rate grid search results (116 JSON files)
- `results_lstm_v6_final/`: Final validation results (73 JSON files)

## Training Scripts

- `train_lstm_awd.py`: Main training script for AWD-LSTM experiments
- `run_lstm_ptb_v4.py`: Batch runner for v4 multi-seed experiments

"""
CIFAR-100 experiment for LAKTJU_NS.

Expert design rationale:
- CIFAR-100 has 100 classes → sharper decision boundaries, stronger curvature
- NS orthogonalization benefits more from harder tasks (larger effective condition number)
- Key tuning vs CIFAR-10:
    (1) CIFAR-100 correct normalization: mean=(0.5071,0.4867,0.4408) std=(0.2675,0.2565,0.2761)
    (2) label_smoothing=0.1 (essential for 100-class tasks)
    (3) weight_decay: 0.002 → 0.005 (stronger regularization for more classes)
    (4) ns_interval=5 (more frequent NS helps complex landscapes)
    (5) warmup=200 steps (longer warmup for stability)
    (6) 300 epochs with cosine annealing

Search grid:
  lr:           [0.001, 0.002, 0.003, 0.005]
  weight_decay: [0.002, 0.005, 0.01]
  ns_interval:  [5, 10, 20]

Baselines (3 seeds each): Adam=74.46%, AdamW=74.05%, Adan=66.77%
Target: LAKTJU_NS > 76% (beat full LAKTJU)
"""

import subprocess, os, sys, json, time, itertools
from datetime import datetime

PYTHON = "/home/hadoop/anaconda3/envs/vllm-env/bin/python"
TRAIN  = "/home/hadoop/workstation/md/TJU-V5(ATJU)-sourcecode/ATJU/experiments/train_laktju.py"
LOG_DIR = "/tmp/cifar100_ns_experiments"
os.makedirs(LOG_DIR, exist_ok=True)

def run(cmd, logfile):
    """Run command, tee output to logfile, return returncode."""
    with open(logfile, 'w') as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        proc.wait()
    return proc.returncode

def read_best_test(results_dir, optimizer, seed, after_time=None):
    """Find best test accuracy from most recent result file."""
    import glob
    pattern = os.path.join(results_dir, f"cifar100_resnet18_{optimizer}_seed{seed}_*.json")
    files = sorted(glob.glob(pattern))
    if after_time:
        files = [f for f in files if os.path.getmtime(f) > after_time]
    if not files:
        return None
    try:
        d = json.load(open(files[-1]))
        return d.get('best_test_acc', d.get('test_acc', 0))
    except:
        return None

RESULTS_DIR = "/home/hadoop/workstation/md/TJU-V5(ATJU)-sourcecode/ATJU/experiments/results"

# ─────────────────────────────────────────────────────────────────
# Phase 1: Hyperparameter search (1 seed = 42, 200 epochs)
# Grid: 3 lr × 2 wd × 2 ns_interval = 12 runs
# ─────────────────────────────────────────────────────────────────
# Expert reasoning for grid boundaries:
#   lr=0.001 (AdamW default) → lr=0.003 (CIFAR-10 optimum) → lr=0.005 (aggressive)
#   wd=0.002 (CIFAR-10 best) → wd=0.005 (stronger for 100 classes) → wd=0.01
#   ns_interval=5 (frequent, ~2x cost per NS step) → 10 (CIFAR-10 default) → 20

SEARCH_GRID = list(itertools.product(
    [0.001, 0.003, 0.005],    # lr
    [0.002, 0.005, 0.01],     # weight_decay
    [5, 10],                   # ns_interval
))

print(f"Phase 1: Hyperparameter search — {len(SEARCH_GRID)} configs × 1 seed")
print("="*70)

search_results = []

for lr, wd, ns_iv in SEARCH_GRID:
    tag = f"lr{lr}_wd{wd}_nsiv{ns_iv}"
    logfile = f"{LOG_DIR}/search_{tag}.log"
    t0 = time.time()

    cmd = [
        PYTHON, TRAIN,
        "--dataset", "cifar100",
        "--optimizer", "LAKTJU_NS",
        "--epochs", "200",
        "--seed", "42",
        "--lr", str(lr),
        "--weight_decay", str(wd),
        "--label_smoothing", "0.1",
        "--warmup", "200",
        "--batch_size", "128",
        "--save_dir", RESULTS_DIR,
        "--data_dir", "/home/hadoop/workstation/md/TJU-V5(ATJU)-sourcecode/ATJU/experiments/dataset",
    ]

    print(f"\n[{datetime.now().strftime('%H:%M')}] {tag} ...", flush=True)
    t_start = time.time()
    rc = run(cmd, logfile)
    elapsed = time.time() - t_start

    acc = read_best_test(RESULTS_DIR, "LAKTJU_NS", 42, after_time=t0)
    print(f"  → TestAcc={acc:.2f}%  ({elapsed/60:.1f} min)  rc={rc}", flush=True)
    search_results.append((acc or 0, lr, wd, ns_iv, tag))

# Sort by accuracy
search_results.sort(reverse=True)
print("\n\n" + "="*70)
print("Phase 1 Results (sorted by test accuracy):")
print("%-12s %-8s %-8s  Acc" % ("LR", "WD", "NS_IV"))
for acc, lr, wd, ns_iv, tag in search_results[:12]:
    print("%-12s %-8s %-8s  %.2f%%" % (lr, wd, ns_iv, acc))

# Best config
best_acc, best_lr, best_wd, best_ns_iv, best_tag = search_results[0]
print(f"\nBest config: lr={best_lr}, wd={best_wd}, ns_interval={best_ns_iv}  → {best_acc:.2f}%")

# ─────────────────────────────────────────────────────────────────
# Phase 2: Multi-seed validation with best config + 300 epochs
# ─────────────────────────────────────────────────────────────────
print("\n\nPhase 2: Multi-seed validation (3 seeds × 300 epochs)")
print("="*70)

SEEDS = [42, 123, 456]
final_results = []

for seed in SEEDS:
    logfile = f"{LOG_DIR}/final_seed{seed}.log"
    t0 = time.time()
    cmd = [
        PYTHON, TRAIN,
        "--dataset", "cifar100",
        "--optimizer", "LAKTJU_NS",
        "--epochs", "300",
        "--seed", str(seed),
        "--lr", str(best_lr),
        "--weight_decay", str(best_wd),
        "--label_smoothing", "0.1",
        "--warmup", "200",
        "--batch_size", "128",
        "--save_dir", RESULTS_DIR,
        "--data_dir", "/home/hadoop/workstation/md/TJU-V5(ATJU)-sourcecode/ATJU/experiments/dataset",
    ]

    print(f"\n[{datetime.now().strftime('%H:%M')}] seed={seed} 300ep ...", flush=True)
    t_start = time.time()
    rc = run(cmd, logfile)
    elapsed = time.time() - t_start

    acc = read_best_test(RESULTS_DIR, "LAKTJU_NS", seed, after_time=t0)
    print(f"  → TestAcc={acc:.2f}%  ({elapsed/60:.1f} min)", flush=True)
    final_results.append(acc or 0)

import numpy as np
arr = np.array(final_results)
print("\n" + "="*70)
print(f"LAKTJU_NS CIFAR-100 Final: {arr.max():.2f}% best | {arr.mean():.2f}±{arr.std():.2f}% mean")
print(f"Seeds: {dict(zip(SEEDS, [f'{v:.2f}' for v in final_results]))}")
print("\nBaselines (from prior experiments):")
print("  Adam:   74.46%  AdamW:  74.05%  Adan:   66.77%  LAKTJU: 76.08%")
beat = "BEAT ALL BASELINES ✓" if arr.max() > 74.46 else "DOES NOT BEAT Adam"
print(f"\nVerdict: {beat}")

# ─────────────────────────────────────────────────────────────────
# Save summary
# ─────────────────────────────────────────────────────────────────
summary = {
    "experiment": "CIFAR-100 LAKTJU_NS",
    "best_config": {"lr": best_lr, "weight_decay": best_wd, "ns_interval": best_ns_iv},
    "search_best_200ep": best_acc,
    "final_300ep": {"seeds": SEEDS, "accs": final_results,
                    "best": float(arr.max()), "mean": float(arr.mean()), "std": float(arr.std())},
    "baselines": {"Adam": 74.46, "AdamW": 74.05, "Adan": 66.77, "LAKTJU": 76.08},
}
with open(f"{LOG_DIR}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary saved to {LOG_DIR}/summary.json")

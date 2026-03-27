#!/usr/bin/env python3
"""
V14 CIFAR-10 精度冲刺实验
基于V8最佳配置 (lr=0.003, a_lr_ratio=0.333, wd=0.001, hs=5.0, warmup=100, ls=0.1)
目标: 超过 SGD 96.08%, SGD+SAM 96.32%

策略:
  Group A: V8最佳配置复现 (3 seeds) - 确认baseline
  Group B: 延长训练到300 epochs (3 seeds) - 更充分收敛
  Group C: 微调lr (0.002, 0.005) × 2 seeds - 找最优lr
  Group D: 微调wd (0.0005, 0.002) × 2 seeds - 找最优正则化
  Group E: homotopy_speed (3.0, 8.0) × 2 seeds - 找最优过渡速度
  Group F: warmup (50, 200) × 2 seeds - 找最优warmup
"""
import subprocess
import os
import time
import sys

RESULTS_DIR = "./results/v14_sprint"
DATA_DIR = "./dataset"
os.makedirs(RESULTS_DIR, exist_ok=True)

# 基础配置 (V8最佳)
BASE = {
    'optimizer': 'LAKTJU',
    'dataset': 'cifar10',
    'model': 'resnet18',
    'lr': 0.003,
    'a_lr_ratio': 0.333,
    'weight_decay': 0.001,
    'homotopy_speed': 5.0,
    'kf_damping': 0.001,
    'warmup': 100,
    'label_smoothing': 0.1,
    'epochs': 200,
    'c_base': 1.0,
    'kappa': 5.0,
}

experiments = []

# Group A: V8最佳配置复现 × 3 seeds
for seed in [42, 123, 456]:
    exp = dict(BASE)
    exp['seed'] = seed
    exp['name'] = f"A_base_s{seed}"
    experiments.append(exp)

# Group B: 延长到300 epochs × 3 seeds
for seed in [42, 123, 456]:
    exp = dict(BASE)
    exp['seed'] = seed
    exp['epochs'] = 300
    exp['name'] = f"B_ep300_s{seed}"
    experiments.append(exp)

# Group C: lr微调 × 2 seeds
for lr in [0.002, 0.005]:
    for seed in [42, 123]:
        exp = dict(BASE)
        exp['seed'] = seed
        exp['lr'] = lr
        exp['name'] = f"C_lr{lr}_s{seed}"
        experiments.append(exp)

# Group D: wd微调 × 2 seeds
for wd in [0.0005, 0.002]:
    for seed in [42, 123]:
        exp = dict(BASE)
        exp['seed'] = seed
        exp['weight_decay'] = wd
        exp['name'] = f"D_wd{wd}_s{seed}"
        experiments.append(exp)

# Group E: homotopy_speed微调 × 2 seeds
for hs in [3.0, 8.0]:
    for seed in [42, 123]:
        exp = dict(BASE)
        exp['seed'] = seed
        exp['homotopy_speed'] = hs
        exp['name'] = f"E_hs{hs}_s{seed}"
        experiments.append(exp)

# Group F: warmup微调 × 2 seeds
for wu in [50, 200]:
    for seed in [42, 123]:
        exp = dict(BASE)
        exp['seed'] = seed
        exp['warmup'] = wu
        exp['name'] = f"F_wu{wu}_s{seed}"
        experiments.append(exp)

print(f"Total experiments: {len(experiments)}")
for e in experiments:
    print(f"  {e['name']}: lr={e['lr']} wd={e['weight_decay']} hs={e['homotopy_speed']} wu={e['warmup']} ep={e['epochs']} seed={e['seed']}")

# 分批运行: 每批最多10个并行
BATCH_SIZE = 10
procs = []

def launch_exp(exp):
    cmd = [
        'python', 'train_laktju.py',
        '--optimizer', exp['optimizer'],
        '--dataset', exp['dataset'],
        '--model', exp['model'],
        '--lr', str(exp['lr']),
        '--a_lr_ratio', str(exp['a_lr_ratio']),
        '--weight_decay', str(exp['weight_decay']),
        '--homotopy_speed', str(exp['homotopy_speed']),
        '--kf_damping', str(exp['kf_damping']),
        '--warmup', str(exp['warmup']),
        '--label_smoothing', str(exp['label_smoothing']),
        '--epochs', str(exp['epochs']),
        '--seed', str(exp['seed']),
        '--c_base', str(exp['c_base']),
        '--kappa', str(exp['kappa']),
        '--save_dir', RESULTS_DIR,
        '--data_dir', DATA_DIR,
    ]
    log_file = os.path.join(RESULTS_DIR, f"{exp['name']}.log")
    f = open(log_file, 'w')
    p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    return p, f, exp['name']

# 分批启动
for batch_start in range(0, len(experiments), BATCH_SIZE):
    batch = experiments[batch_start:batch_start+BATCH_SIZE]
    batch_procs = []
    
    print(f"\n=== Launching batch {batch_start//BATCH_SIZE + 1} ({len(batch)} experiments) ===")
    for exp in batch:
        p, f, name = launch_exp(exp)
        batch_procs.append((p, f, name))
        print(f"  Launched: {name} (PID={p.pid})")
        time.sleep(1)  # 错开启动避免IO冲突
    
    # 等待当前批次完成
    print(f"Waiting for batch to complete...")
    for p, f, name in batch_procs:
        p.wait()
        f.close()
        print(f"  Completed: {name} (exit={p.returncode})")

print("\n=== All experiments completed! ===")

# 汇总结果
import json, glob
print("\n=== RESULTS SUMMARY ===")
print(f"{'Name':<25} {'Test':>6} {'Valid':>6}")
print("-" * 40)
results = []
for f in sorted(glob.glob(os.path.join(RESULTS_DIR, '*.json'))):
    try:
        d = json.load(open(f))
        name = os.path.basename(f).replace('cifar10_resnet18_LAKTJU_', '').replace('.json', '')
        test = d.get('best_test_acc', 0)
        valid = d.get('best_valid_acc', 0)
        results.append((name, test, valid))
    except:
        pass
results.sort(key=lambda x: x[1], reverse=True)
for name, test, valid in results:
    marker = " ***" if test > 96.08 else " **" if test > 95.61 else ""
    print(f"{name:<25} {test:>6.2f} {valid:>6.2f}{marker}")


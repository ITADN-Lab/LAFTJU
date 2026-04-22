#!/usr/bin/env python3
"""Analyze TXL Phase 1 results and prepare summary."""
import json, glob, os, sys

RESULTS_DIR = sys.argv[1] if len(sys.argv) > 1 else "./results_txl_main"
LOG_DIR = sys.argv[2] if len(sys.argv) > 2 else "./logs_txl_main"

results = []
for f in sorted(glob.glob(os.path.join(RESULTS_DIR, "txl_*.json"))):
    with open(f) as fp:
        d = json.load(fp)
    tag = os.path.basename(f).replace('.json', '')
    results.append({
        'file': tag,
        'optimizer': d.get('optimizer', '?'),
        'lr': d.get('lr', '?'),
        'wd': d.get('wd', 0),
        'seed': d.get('seed', '?'),
        'max_step': d.get('max_step', '?'),
        'best_val_ppl': d.get('best_val_ppl', float('inf')),
        'best_test_ppl': d.get('best_test_ppl', float('inf')),
        'best_step': d.get('best_step', 0),
        'ns_interval': d.get('ns_interval', '-'),
        'ns_steps': d.get('ns_steps', '-'),
        'total_time_min': d.get('total_time_min', 0),
    })

# Sort by best test PPL
results.sort(key=lambda x: x['best_test_ppl'])

print(f"\n{'='*80}")
print(f"Transformer-XL Phase 1 Results (sorted by test PPL, lower=better)")
print(f"{'='*80}")
print(f"{'Optimizer':<12} {'LR':>8} {'WD':>6} {'NS_int':>7} {'NS_K':>5} "
      f"{'Val PPL':>10} {'Test PPL':>10} {'Best Step':>10} {'Time(min)':>10}")
print(f"{'-'*80}")

for r in results:
    ns_int = str(r['ns_interval']) if r['ns_interval'] != '-' else '-'
    ns_k = str(r['ns_steps']) if r['ns_steps'] != '-' else '-'
    print(f"{r['optimizer']:<12} {r['lr']:>8.5f} {r['wd']:>6.3f} {ns_int:>7} {ns_k:>5} "
          f"{r['best_val_ppl']:>10.2f} {r['best_test_ppl']:>10.2f} "
          f"{r['best_step']:>10} {r['total_time_min']:>10.1f}")

# Find best baseline and best NS
baselines = [r for r in results if r['optimizer'] in ('adam', 'adamw')]
ns_configs = [r for r in results if r['optimizer'] == 'LAKTJU_NS']

if baselines:
    best_base = min(baselines, key=lambda x: x['best_test_ppl'])
    print(f"\nBest baseline: {best_base['optimizer']} lr={best_base['lr']:.5f} → test PPL={best_base['best_test_ppl']:.2f}")

if ns_configs:
    best_ns = min(ns_configs, key=lambda x: x['best_test_ppl'])
    print(f"Best NS:       ns={best_ns['ns_interval']} K={best_ns['ns_steps']} lr={best_ns['lr']:.5f} → test PPL={best_ns['best_test_ppl']:.2f}")

    if baselines:
        improvement = best_base['best_test_ppl'] - best_ns['best_test_ppl']
        print(f"Improvement:   {improvement:.2f} PPL ({improvement/best_base['best_test_ppl']*100:.1f}%)")

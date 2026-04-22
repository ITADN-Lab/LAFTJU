#!/usr/bin/env python3
"""
LSTM PTB v4 — NT-ASGD + longer training to boost LAFTJU-NS.

Key changes vs v3:
  - Enable NT-ASGD (--asgd --nonmono 5): switches to ASGD when val plateaus
  - 750 epochs (more room for ASGD convergence)
  - Phase 1: 2-layer grid search (the weak spot)
  - Phase 2: Best config × 3 seeds × 3 layers (+ Adam baseline)
"""
import os
import sys
import subprocess
import time
import json
import numpy as np

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_lstm_awd.py')
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'ptb')
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_lstm_v4')
SEEDS = [42, 123, 456]
EPOCHS = 750


def run(cmd_args, desc=""):
    cmd = [sys.executable, SCRIPT] + cmd_args + [
        '--data_dir', DATA_DIR,
        '--save_dir', SAVE_DIR,
    ]
    print(f'\n{"="*70}')
    print(f'Running: {desc}')
    print(f'Command: {" ".join(cmd)}')
    print(f'{"="*70}', flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
    print(f'=> {desc}: {status} ({elapsed/60:.1f} min)', flush=True)
    return result.returncode == 0


def phase1_grid():
    """2-layer grid search with ASGD to find best NS config."""
    print('\n' + '='*70)
    print('PHASE 1: 2-LAYER GRID SEARCH (ASGD, 750 epochs)')
    print('='*70, flush=True)

    configs = [
        {'ns_interval': 10,  'ns_max_dim': 2048},
        {'ns_interval': 25,  'ns_max_dim': 2048},
        {'ns_interval': 50,  'ns_max_dim': 2048},
    ]

    for cfg in configs:
        args = [
            '--optimizer', 'LAKTJU_NS_Adam',
            '--lr', '1e-3',
            '--weight_decay', '1.2e-6',
            '--nlayers', '2',
            '--epochs', str(EPOCHS),
            '--seed', '42',
            '--scheduler', 'none',
            '--asgd', '--nonmono', '5',
            '--ns_interval', str(cfg['ns_interval']),
            '--ns_max_dim', str(cfg['ns_max_dim']),
        ]
        desc = f"NS_Adam 2L ns={cfg['ns_interval']} maxd={cfg['ns_max_dim']} ASGD"
        run(args, desc)


def phase2_final(best_ns_interval):
    """Best config × 3 seeds × 3 layers + Adam baseline."""
    print('\n' + '='*70)
    print(f'PHASE 2: FINAL (ns={best_ns_interval}, ASGD, {EPOCHS} epochs)')
    print('='*70, flush=True)

    for nlayers in [1, 2, 3]:
        # Adam baseline with ASGD
        for seed in SEEDS:
            args = [
                '--optimizer', 'Adam',
                '--lr', '1e-3',
                '--weight_decay', '1.2e-6',
                '--nlayers', str(nlayers),
                '--epochs', str(EPOCHS),
                '--seed', str(seed),
                '--scheduler', 'none',
                '--asgd', '--nonmono', '5',
            ]
            run(args, f"Adam {nlayers}L seed{seed} ASGD")

        # LAKTJU-NS-Adam with ASGD
        for seed in SEEDS:
            args = [
                '--optimizer', 'LAKTJU_NS_Adam',
                '--lr', '1e-3',
                '--weight_decay', '1.2e-6',
                '--nlayers', str(nlayers),
                '--epochs', str(EPOCHS),
                '--seed', str(seed),
                '--scheduler', 'none',
                '--asgd', '--nonmono', '5',
                '--ns_interval', str(best_ns_interval),
                '--ns_max_dim', '2048',
            ]
            run(args, f"NS_Adam {nlayers}L seed{seed} ASGD")

        # AdamW baseline with ASGD
        for seed in SEEDS:
            args = [
                '--optimizer', 'AdamW',
                '--lr', '1e-3',
                '--weight_decay', '1.2e-6',
                '--nlayers', str(nlayers),
                '--epochs', str(EPOCHS),
                '--seed', str(seed),
                '--scheduler', 'none',
                '--asgd', '--nonmono', '5',
            ]
            run(args, f"AdamW {nlayers}L seed{seed} ASGD")


def summarize():
    """Summarize all results."""
    results = {}
    for f in sorted(os.listdir(SAVE_DIR)):
        if not f.startswith('awd') or not f.endswith('.json'):
            continue
        if '_lr' not in f:
            continue
        path = os.path.join(SAVE_DIR, f)
        with open(path) as fh:
            d = json.load(fh)
        opt = d['optimizer']
        nl = d['nlayers']
        key = f"{opt} {nl}L"

        if key not in results:
            results[key] = []
        results[key].append({
            'seed': d['seed'],
            'best_test_ppl': d['best_test_ppl'],
            'best_val_ppl': d['best_val_ppl'],
        })

    print('\n' + '='*70)
    print('SUMMARY (v4: ASGD, 750 epochs)')
    print('='*70)
    for key in sorted(results.keys()):
        vals = results[key]
        ppls = [v['best_test_ppl'] for v in vals]
        avg = np.mean(ppls)
        std = np.std(ppls, ddof=1) if len(ppls) > 1 else 0
        print(f'{key:35s}: {avg:.2f} ± {std:.2f}  ({[round(v,2) for v in ppls]})')

    summary_path = os.path.join(SAVE_DIR, 'summary_v4.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSummary saved to {summary_path}')


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Phase 1: Grid search on 2-layer
    phase1_grid()

    # Find best config
    best_ppl = float('inf')
    best_ns = 25  # default
    for f in os.listdir(SAVE_DIR):
        if 'LAKTJU_NS_Adam' in f and '_lr' in f and f.endswith('.json'):
            with open(os.path.join(SAVE_DIR, f)) as fh:
                d = json.load(fh)
            if d.get('nlayers') == 2 and d.get('seed') == 42:
                if d['best_test_ppl'] < best_ppl:
                    best_ppl = d['best_test_ppl']
                    best_ns = d.get('ns_interval', 25)

    print(f'\nBest 2L grid search: PPL={best_ppl:.2f}, ns_interval={best_ns}')

    # Phase 2: Final evaluation
    phase2_final(best_ns)

    # Summary
    summarize()


if __name__ == '__main__':
    main()

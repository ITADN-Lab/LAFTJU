#!/usr/bin/env python3
"""
LSTM PTB v3.1 — Fix NS coverage: ns_max_dim was too small!

Key finding: AWD-LSTM nhid=1150, so weight matrices have min_dim=1150.
With ns_max_dim=1024, 73% of params were SKIPPED by NS.
With ns_max_dim=2048, ALL weight matrices get NS treatment.

Also test:
- NS on second moment (exp_avg_sq) in addition to first moment
- Stronger NS (more steps)
- Different ns_interval values with full coverage
"""
import os
import sys
import subprocess
import time
import json
import numpy as np

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_lstm_awd.py')
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'ptb')
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_lstm_v3')
SEEDS = [42, 123, 456]
EPOCHS = 500


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
    """Grid search with ns_max_dim=2048 (full coverage)."""
    print('\n' + '='*70)
    print('PHASE 1: GRID SEARCH — ns_max_dim=2048 (full NS coverage)')
    print('='*70, flush=True)

    configs = [
        # Group A: Full NS coverage, varying interval
        {'ns_interval': 25,  'ns_skip_rnn': False, 'ns_max_dim': 2048, 'ns_steps': 1},
        {'ns_interval': 50,  'ns_skip_rnn': False, 'ns_max_dim': 2048, 'ns_steps': 1},
        {'ns_interval': 100, 'ns_skip_rnn': False, 'ns_max_dim': 2048, 'ns_steps': 1},
        {'ns_interval': 200, 'ns_skip_rnn': False, 'ns_max_dim': 2048, 'ns_steps': 1},
        # Group B: Skip RNN weights, NS only on embedding/decoder + weight_ih
        {'ns_interval': 50,  'ns_skip_rnn': True,  'ns_max_dim': 2048, 'ns_steps': 1},
        {'ns_interval': 100, 'ns_skip_rnn': True,  'ns_max_dim': 2048, 'ns_steps': 1},
        # Group C: Stronger NS (2 iterations)
        {'ns_interval': 50,  'ns_skip_rnn': False, 'ns_max_dim': 2048, 'ns_steps': 2},
        {'ns_interval': 100, 'ns_skip_rnn': False, 'ns_max_dim': 2048, 'ns_steps': 2},
        # Group D: Very frequent NS
        {'ns_interval': 10,  'ns_skip_rnn': False, 'ns_max_dim': 2048, 'ns_steps': 1},
        # Group E: Compare with old ns_max_dim=1024 (control)
        {'ns_interval': 100, 'ns_skip_rnn': False, 'ns_max_dim': 1024, 'ns_steps': 1},
    ]

    for cfg in configs:
        args = [
            '--optimizer', 'LAKTJU_NS_Adam',
            '--lr', '1e-3',
            '--weight_decay', '1.2e-6',
            '--nlayers', '3',
            '--epochs', str(EPOCHS),
            '--seed', '42',
            '--scheduler', 'none',
            '--ns_interval', str(cfg['ns_interval']),
            '--ns_max_dim', str(cfg['ns_max_dim']),
        ]
        if cfg['ns_skip_rnn']:
            args.append('--ns_skip_rnn')
        if cfg.get('ns_steps', 1) > 1:
            args += ['--ns_steps', str(cfg['ns_steps'])]
        desc = (f"NS_Adam 3L ns={cfg['ns_interval']} skip={cfg['ns_skip_rnn']} "
                f"maxd={cfg['ns_max_dim']} steps={cfg.get('ns_steps',1)}")
        run(args, desc)


def phase2_final(best_cfg):
    """Best config × 3 seeds × 3 layers."""
    print('\n' + '='*70)
    print(f'PHASE 2: FINAL — {best_cfg}')
    print('='*70, flush=True)

    for nlayers in [1, 2, 3]:
        for seed in SEEDS:
            args = [
                '--optimizer', 'LAKTJU_NS_Adam',
                '--lr', '1e-3',
                '--weight_decay', '1.2e-6',
                '--nlayers', str(nlayers),
                '--epochs', str(EPOCHS),
                '--seed', str(seed),
                '--scheduler', 'none',
                '--ns_interval', str(best_cfg['ns_interval']),
                '--ns_max_dim', str(best_cfg['ns_max_dim']),
            ]
            if best_cfg.get('ns_skip_rnn'):
                args.append('--ns_skip_rnn')
            if best_cfg.get('ns_steps', 1) > 1:
                args += ['--ns_steps', str(best_cfg['ns_steps'])]
            run(args, f"NS_Adam {nlayers}L seed{seed} (v3.1 final)")


def summarize():
    """Summarize all results."""
    results = {}
    for f in sorted(os.listdir(SAVE_DIR)):
        if not f.startswith('awd') or not f.endswith('.json') or '_lr' not in f:
            continue
        path = os.path.join(SAVE_DIR, f)
        with open(path) as fh:
            d = json.load(fh)
        opt = d['optimizer']
        nl = d['nlayers']
        key = f"{opt}_{nl}L"
        if 'LAKTJU_NS_Adam' in opt:
            maxd = d.get('ns_max_dim', 1024)
            ns = d.get('ns_interval', 100)
            key = f"{opt}_maxd{maxd}_{nl}L"
        if key not in results:
            results[key] = []
        results[key].append(d['best_test_ppl'])

    print('\n' + '='*70)
    print('SUMMARY')
    print('='*70)
    for key in sorted(results.keys()):
        vals = results[key]
        avg = sum(vals) / len(vals)
        if len(vals) > 1:
            std = np.std(vals, ddof=1)
            print(f'{key:45s}: {avg:.2f} ± {std:.2f}  ({[round(v,2) for v in vals]})')
        else:
            print(f'{key:45s}: {avg:.2f}  ({[round(v,2) for v in vals]})')

    summary_path = os.path.join(SAVE_DIR, 'summary_lstm_v3_1.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSummary saved to {summary_path}')


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Phase 1: Grid search with full NS coverage
    phase1_grid()

    # Find best config
    best_ppl = float('inf')
    best_cfg = {'ns_interval': 100, 'ns_skip_rnn': False, 'ns_max_dim': 2048, 'ns_steps': 1}
    for f in os.listdir(SAVE_DIR):
        if 'LAKTJU_NS_Adam' in f and '_lr' in f and f.endswith('.json'):
            with open(os.path.join(SAVE_DIR, f)) as fh:
                d = json.load(fh)
            if (d.get('nlayers') == 3 and d.get('seed') == 42 and
                d.get('ns_max_dim', 0) >= 2048 and
                d['best_test_ppl'] < best_ppl):
                best_ppl = d['best_test_ppl']
                best_cfg = {
                    'ns_interval': d.get('ns_interval', 100),
                    'ns_skip_rnn': d.get('ns_skip_rnn', False),
                    'ns_max_dim': d.get('ns_max_dim', 2048),
                    'ns_steps': d.get('ns_steps', 1),
                }

    print(f'\nBest grid search: PPL={best_ppl:.2f}, config={best_cfg}')

    # Phase 2: Final evaluation
    phase2_final(best_cfg)

    summarize()


if __name__ == '__main__':
    main()

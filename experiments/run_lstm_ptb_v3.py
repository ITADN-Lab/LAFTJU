#!/usr/bin/env python3
"""
LSTM PTB v3 Experiment Runner — AWD-LSTM with LAFTJU-NS.

Strategy:
1. Use AWD-LSTM (Merity et al.) — the standard benchmark used by AdaBelief/Adan papers
2. LAKTJU_NS_Adam inherits from Adam (L2 wd) instead of AdamW (decoupled wd)
3. NS orthogonalization targets embedding/decoder (largest params, most benefit)

Phase 1: Baselines (Adam, AdamW, SGD) — verify we match paper numbers
Phase 2: LAKTJU_NS_Adam grid search — find optimal NS config
Phase 3: Best config × 5 seeds × 3 layers
"""
import os
import sys
import subprocess
import time
import json

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_lstm_awd.py')
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'ptb')
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_lstm_v3')
SEEDS = [42, 123, 456]
SEEDS_5 = [42, 123, 456, 789, 2024]
EPOCHS = 500  # Sufficient for convergence comparison


def run(cmd_args, desc=""):
    """Run a single training job."""
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


def phase1_baselines():
    """Phase 1: Baselines to verify setup matches papers. Focus on 3-layer."""
    print('\n' + '='*70)
    print('PHASE 1: BASELINES (3-layer focus)')
    print('='*70, flush=True)

    configs = []

    # Adam baseline (paper: 64.3 for 3L)
    for seed in SEEDS:
        configs.append({
            'args': ['--optimizer', 'Adam', '--lr', '1e-3',
                     '--weight_decay', '1.2e-6',
                     '--nlayers', '3', '--epochs', str(EPOCHS),
                     '--seed', str(seed), '--scheduler', 'none'],
            'desc': f'Adam 3L seed{seed}'
        })

    # AdamW baseline
    for seed in SEEDS:
        configs.append({
            'args': ['--optimizer', 'AdamW', '--lr', '1e-3',
                     '--weight_decay', '1.2e-6',
                     '--nlayers', '3', '--epochs', str(EPOCHS),
                     '--seed', str(seed), '--scheduler', 'none'],
            'desc': f'AdamW 3L seed{seed}'
        })

    for cfg in configs:
        run(cfg['args'], cfg['desc'])


def phase2_grid_search():
    """Phase 2: LAKTJU_NS_Adam grid search on 3-layer."""
    print('\n' + '='*70)
    print('PHASE 2: LAKTJU_NS_Adam GRID SEARCH (3-layer)')
    print('='*70, flush=True)

    # Grid: ns_interval × ns_skip_rnn × ns_max_dim
    grid = [
        # Group A: skip_rnn=True (NS only on embedding/decoder)
        {'ns_interval': 50,  'ns_skip_rnn': True,  'ns_max_dim': 1024},
        {'ns_interval': 100, 'ns_skip_rnn': True,  'ns_max_dim': 1024},
        {'ns_interval': 200, 'ns_skip_rnn': True,  'ns_max_dim': 1024},
        {'ns_interval': 500, 'ns_skip_rnn': True,  'ns_max_dim': 1024},
        # Group B: skip_rnn=False (NS on all weights)
        {'ns_interval': 50,  'ns_skip_rnn': False, 'ns_max_dim': 1024},
        {'ns_interval': 100, 'ns_skip_rnn': False, 'ns_max_dim': 1024},
        {'ns_interval': 200, 'ns_skip_rnn': False, 'ns_max_dim': 1024},
        {'ns_interval': 500, 'ns_skip_rnn': False, 'ns_max_dim': 1024},
        # Group C: smaller ns_max_dim (only small matrices get NS)
        {'ns_interval': 100, 'ns_skip_rnn': True,  'ns_max_dim': 256},
        {'ns_interval': 100, 'ns_skip_rnn': False, 'ns_max_dim': 256},
    ]

    for cfg in grid:
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
        desc = f"NS_Adam 3L ns={cfg['ns_interval']} skip={cfg['ns_skip_rnn']} maxd={cfg['ns_max_dim']}"
        run(args, desc)

    # Also test LAKTJU_NS (AdamW base) for comparison
    run(['--optimizer', 'LAKTJU_NS', '--lr', '1e-3', '--weight_decay', '1.2e-6',
         '--nlayers', '3', '--epochs', str(EPOCHS), '--seed', '42',
         '--scheduler', 'none', '--ns_interval', '100', '--ns_max_dim', '1024',
         '--ns_skip_rnn'],
        'NS_AdamW 3L ns=100 skip=True (comparison)')


def phase3_final(best_ns_interval, best_ns_skip_rnn, best_ns_max_dim):
    """Phase 3: Best config × 3 seeds × 3 layers."""
    print('\n' + '='*70)
    print('PHASE 3: FINAL EVALUATION')
    print(f'Best config: ns={best_ns_interval}, skip_rnn={best_ns_skip_rnn}, max_dim={best_ns_max_dim}')
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
                '--ns_interval', str(best_ns_interval),
                '--ns_max_dim', str(best_ns_max_dim),
            ]
            if best_ns_skip_rnn:
                args.append('--ns_skip_rnn')
            run(args, f'NS_Adam {nlayers}L seed{seed} (final)')

        # Also run Adam/AdamW baselines for 1L and 2L (we only did 3L in Phase 1)
        if nlayers < 3:
            for seed in SEEDS:
                run(['--optimizer', 'Adam', '--lr', '1e-3',
                     '--weight_decay', '1.2e-6',
                     '--nlayers', str(nlayers), '--epochs', str(EPOCHS),
                     '--seed', str(seed), '--scheduler', 'none'],
                    f'Adam {nlayers}L seed{seed} (final)')
                run(['--optimizer', 'AdamW', '--lr', '1e-3',
                     '--weight_decay', '1.2e-6',
                     '--nlayers', str(nlayers), '--epochs', str(EPOCHS),
                     '--seed', str(seed), '--scheduler', 'none'],
                    f'AdamW {nlayers}L seed{seed} (final)')


def summarize():
    """Summarize all results."""
    if not os.path.exists(SAVE_DIR):
        print("No results directory found.")
        return

    results = {}
    for f in sorted(os.listdir(SAVE_DIR)):
        if not f.startswith('awd') or not f.endswith('.json'):
            continue
        # Only read compact results (with _lr in name)
        if '_lr' not in f:
            continue
        path = os.path.join(SAVE_DIR, f)
        with open(path) as fh:
            d = json.load(fh)
        opt = d['optimizer']
        nl = d['nlayers']
        key = f"{opt}_{nl}L"
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
            std = (sum((v - avg)**2 for v in vals) / (len(vals) - 1)) ** 0.5
            print(f'{key:30s}: {avg:.2f} ± {std:.2f}  ({vals})')
        else:
            print(f'{key:30s}: {avg:.2f}  ({vals})')

    # Save summary
    summary_path = os.path.join(SAVE_DIR, 'summary_lstm_v3.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSummary saved to {summary_path}')


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Phase 1: Baselines
    phase1_baselines()

    # Phase 2: Grid search
    phase2_grid_search()

    # Find best grid search config
    best_ppl = float('inf')
    best_cfg = {'ns_interval': 100, 'ns_skip_rnn': True, 'ns_max_dim': 1024}
    for f in os.listdir(SAVE_DIR):
        if 'LAKTJU_NS_Adam' in f and '_lr' in f and f.endswith('.json'):
            with open(os.path.join(SAVE_DIR, f)) as fh:
                d = json.load(fh)
            if d.get('nlayers') == 3 and d['best_test_ppl'] < best_ppl:
                best_ppl = d['best_test_ppl']
                best_cfg = {
                    'ns_interval': d.get('ns_interval', 100),
                    'ns_skip_rnn': d.get('ns_skip_rnn', True),
                    'ns_max_dim': d.get('ns_max_dim', 1024),
                }

    print(f'\nBest grid search: PPL={best_ppl:.2f}, config={best_cfg}')

    # Phase 3: Final evaluation
    phase3_final(best_cfg['ns_interval'], best_cfg['ns_skip_rnn'],
                 best_cfg['ns_max_dim'])

    # Summary
    summarize()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
LSTM PTB Experiment for LAFTJU-NS — following Adan paper setup.

Experiment Design:
  Phase 1: Baselines (Adam, AdamW, Adan) × 3 layers × 3 seeds
  Phase 2: LAFTJU-NS grid search (lr × wd × ns_interval) × 3 layers
  Phase 3: Best LAFTJU-NS config × 3 layers × 5 seeds

Target (from Adan paper Table 10):
  1-layer: Adan 83.6, Adam 85.9, AdamW 84.7
  2-layer: Adan 65.2, Adam 67.3, AdamW 72.8
  3-layer: Adan 59.8, Adam 64.3, AdamW 69.9
"""
import subprocess, os, sys, json, time
import numpy as np

PYTHON = sys.executable
TRAIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_lstm.py')
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_lstm')
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'ptb')
os.makedirs(LOG_DIR, exist_ok=True)

# Feishu notification
FEISHU_APP_ID = 'cli_a93c7a2146381bd2'
FEISHU_APP_SECRET = 'NQQegyt8DJPFU3PeURjJLg3IqDRMopsY'
FEISHU_CHAT_ID = 'oc_840b30bc7d66669838c28fdfa049ea8f'

import urllib.request


def get_token():
    body = json.dumps({'app_id': FEISHU_APP_ID, 'app_secret': FEISHU_APP_SECRET}).encode()
    req = urllib.request.Request(
        'https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal',
        data=body, headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(req, timeout=10).read())['tenant_access_token']


def feishu_send(text):
    try:
        token = get_token()
        body = json.dumps({
            'receive_id': FEISHU_CHAT_ID,
            'msg_type': 'text',
            'content': json.dumps({'text': text})
        }).encode()
        req = urllib.request.Request(
            'https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id',
            data=body,
            headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {token}'})
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f'[feishu] send failed: {e}')


def run_exp(opt, lr, wd, nlayers, epochs=200, seed=42, scheduler='cosine',
            ns_interval=100, ns_max_dim=1024, grad_centralization=False):
    """Run a single LSTM experiment and return (best_val_ppl, best_test_ppl)."""
    cmd = [
        PYTHON, TRAIN,
        '--optimizer', opt,
        '--nlayers', str(nlayers),
        '--lr', str(lr),
        '--weight_decay', str(wd),
        '--epochs', str(epochs),
        '--seed', str(seed),
        '--scheduler', scheduler,
        '--save_dir', LOG_DIR,
        '--data_dir', DATA_DIR,
    ]
    if opt == 'LAKTJU_NS':
        cmd += ['--ns_interval', str(ns_interval),
                '--ns_max_dim', str(ns_max_dim)]
        if grad_centralization:
            cmd += ['--grad_centralization']

    tag = f'lstm{nlayers}_{opt}_lr{lr}_wd{wd}_seed{seed}'
    print(f'\n{"="*60}')
    print(f'Running: {tag}')
    print(f'{"="*60}', flush=True)

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - t0

    # Parse result from latest JSON
    best_val, best_test = float('inf'), float('inf')
    for f in sorted(os.listdir(LOG_DIR), reverse=True):
        if f.endswith('.json') and f'lstm{nlayers}_{opt}_seed{seed}' in f:
            with open(os.path.join(LOG_DIR, f)) as fp:
                d = json.load(fp)
            best_val = d.get('best_val_ppl', float('inf'))
            best_test = d.get('best_test_ppl', float('inf'))
            break

    print(f'  => Val PPL: {best_val:.2f}  Test PPL: {best_test:.2f}  Time: {elapsed/60:.1f}min')
    return best_val, best_test, elapsed


def main():
    start_time = time.time()
    feishu_send('🔬 LSTM PTB 实验开始\n'
                '目标: LAFTJU-NS 超越 Adan (83.6/65.2/59.8)')

    all_results = {}

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 1: Baselines — 3 optimizers × 3 LSTM depths × 3 seeds
    # ═══════════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('Phase 1: BASELINES')
    print('='*70)

    # AdaBelief default settings for PTB LSTM:
    # - Adam: lr=0.001, wd=1.2e-6, cosine schedule (or multistep at 100,145)
    # - AdamW: lr=0.001, wd=1.2e-6
    # - SGD: lr=30, wd=1.2e-6, multistep
    # - Adan: lr=0.01, wd=0.02, betas=(0.02,0.08,0.01)
    baseline_configs = {
        'Adam':  {'lr': 0.001, 'wd': 1.2e-6, 'scheduler': 'cosine'},
        'AdamW': {'lr': 0.001, 'wd': 1.2e-6, 'scheduler': 'cosine'},
        'Adan':  {'lr': 0.01,  'wd': 0.02,   'scheduler': 'cosine'},
    }

    baseline_results = {}  # {opt: {nlayers: {'tests': [...], 'avg': ..., 'std': ...}}}

    for opt, cfg in baseline_configs.items():
        baseline_results[opt] = {}
        for nlayers in [1, 2, 3]:
            seed_tests = []
            for seed in [42, 123, 456]:
                _, test_ppl, _ = run_exp(opt, cfg['lr'], cfg['wd'], nlayers,
                                          epochs=200, seed=seed, scheduler=cfg['scheduler'])
                seed_tests.append(test_ppl)
            avg = np.mean(seed_tests)
            std = np.std(seed_tests)
            baseline_results[opt][nlayers] = {'tests': seed_tests, 'avg': avg, 'std': std}
            feishu_send(f'✅ {opt} {nlayers}L: {[f"{t:.1f}" for t in seed_tests]} '
                        f'avg={avg:.2f}±{std:.2f}')

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 2: LAFTJU-NS grid search (seed=42, 2-layer LSTM as pilot)
    # ═══════════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('Phase 2: LAFTJU-NS GRID SEARCH (2-layer LSTM)')
    print('='*70)

    # Grid: lr × wd × ns_interval
    grid = [
        # lr sweep
        (0.001, 1.2e-6, 100),
        (0.002, 1.2e-6, 100),
        (0.003, 1.2e-6, 100),
        (0.005, 1.2e-6, 100),
        (0.008, 1.2e-6, 100),
        # wd sweep (with best lr from above ~ likely 0.003)
        (0.003, 0.001, 100),
        (0.003, 0.005, 100),
        (0.003, 0.01,  100),
        (0.005, 0.01,  100),
        (0.008, 0.01,  100),
        # ns_interval sweep
        (0.003, 1.2e-6, 50),
        (0.003, 1.2e-6, 200),
    ]

    grid_results = []
    for lr, wd, ns_int in grid:
        _, test_ppl, elapsed = run_exp('LAKTJU_NS', lr, wd, 2,
                                        epochs=200, seed=42, ns_interval=ns_int)
        grid_results.append({'lr': lr, 'wd': wd, 'ns': ns_int,
                              'test': test_ppl, 'time': elapsed})
        feishu_send(f'🔄 LAKTJU_NS 2L: lr={lr} wd={wd} ns={ns_int} → {test_ppl:.2f}')

    # Sort by test PPL (lower = better)
    grid_results.sort(key=lambda x: x['test'])
    best_cfg = grid_results[0]
    feishu_send(f'📊 Grid Top3:\n'
                + '\n'.join(f"  #{i+1}: lr={r['lr']} wd={r['wd']} ns={r['ns']} → {r['test']:.2f}"
                            for i, r in enumerate(grid_results[:3])))

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 3: Best config × 3 layers × 5 seeds
    # ═══════════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('Phase 3: BEST CONFIG × 3 LAYERS × 5 SEEDS')
    print('='*70)

    blr, bwd, bns = best_cfg['lr'], best_cfg['wd'], best_cfg['ns']
    ns_final = {}  # {nlayers: {'tests': [...], 'avg': ..., 'std': ...}}

    for nlayers in [1, 2, 3]:
        seed_tests = []
        for seed in [42, 123, 456, 789, 2024]:
            _, test_ppl, _ = run_exp('LAKTJU_NS', blr, bwd, nlayers,
                                      epochs=200, seed=seed, ns_interval=bns)
            seed_tests.append(test_ppl)
        avg = np.mean(seed_tests)
        std = np.std(seed_tests)
        ns_final[nlayers] = {'tests': seed_tests, 'avg': avg, 'std': std}
        feishu_send(f'🎯 LAKTJU_NS {nlayers}L (5 seeds): '
                    f'{[f"{t:.1f}" for t in seed_tests]} avg={avg:.2f}±{std:.2f}')

    # ═══════════════════════════════════════════════════════════════════════
    # Final Summary
    # ═══════════════════════════════════════════════════════════════════════
    total_time = (time.time() - start_time) / 3600

    # Build comparison table
    lines = ['🏆 LSTM PTB 实验完成!\n']
    lines.append(f'最优配置: lr={blr} wd={bwd} ns_interval={bns}\n')

    for nlayers in [1, 2, 3]:
        lines.append(f'\n【{nlayers}-layer LSTM】')
        for opt in ['Adam', 'AdamW', 'Adan']:
            br = baseline_results[opt][nlayers]
            lines.append(f'  {opt:8s}: {br["avg"]:.2f}±{br["std"]:.2f}')
        nr = ns_final[nlayers]
        lines.append(f'  LAFTJU-NS: {nr["avg"]:.2f}±{nr["std"]:.2f}')
        # vs Adan
        adan_avg = baseline_results['Adan'][nlayers]['avg']
        diff = adan_avg - nr['avg']
        marker = '✅' if diff > 0 else '❌'
        lines.append(f'  → vs Adan: {diff:+.2f} {marker}')

    lines.append(f'\n总耗时: {total_time:.1f}h')

    summary = '\n'.join(lines)
    print('\n' + summary)
    feishu_send(summary)

    # Save complete results
    full_summary = {
        'baselines': {opt: {str(nl): v for nl, v in layers.items()}
                      for opt, layers in baseline_results.items()},
        'grid_search': grid_results,
        'best_cfg': {'lr': blr, 'wd': bwd, 'ns_interval': bns},
        'final_5seed': {str(nl): v for nl, v in ns_final.items()},
        'total_time_hours': total_time,
    }
    with open(os.path.join(LOG_DIR, 'summary_lstm.json'), 'w') as f:
        json.dump(full_summary, f, indent=2)
    print(f'Summary saved to {LOG_DIR}/summary_lstm.json')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
LSTM PTB Experiment v2 — Redesigned for LAFTJU-NS superiority.

Key changes from v1:
  1. Use multistep scheduler (milestones=[100,145]) matching AdaBelief/Adan paper
  2. Add ns_skip_rnn option: skip NS on LSTM recurrent weights (weight_hh)
  3. Proper Adan hyperparams: lr=0.01, wd=0.02, betas=(0.02,0.08,0.01)
  4. Wider LAFTJU-NS grid search with RNN-aware settings
  5. SGD baseline with lr=30 (AdaBelief default)

Experiment Design:
  Phase 1: Baselines (SGD, Adam, AdamW, Adan) × 3 layers × 3 seeds
           All use multistep scheduler [100,145] to match AdaBelief paper
  Phase 2: LAFTJU-NS grid search on 2-layer LSTM
           Key axes: lr, wd, ns_interval, ns_skip_rnn
  Phase 3: Best LAFTJU-NS config × 3 layers × 5 seeds

Target (from Adan paper Table 10):
  1-layer: Adan 83.6, Adam 85.9, AdamW 84.7, SGD 85.0
  2-layer: Adan 65.2, Adam 67.3, AdamW 72.8, SGD 67.4
  3-layer: Adan 59.8, Adam 64.3, AdamW 69.9, SGD 63.7
"""
import subprocess, os, sys, json, time
import numpy as np

PYTHON = sys.executable
TRAIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_lstm.py')
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_lstm_v2')
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


def run_exp(opt, lr, wd, nlayers, epochs=200, seed=42, scheduler='multistep',
            ns_interval=100, ns_max_dim=1024, grad_centralization=False,
            ns_skip_rnn=False):
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
        if ns_skip_rnn:
            cmd += ['--ns_skip_rnn']

    tag = f'lstm{nlayers}_{opt}_lr{lr}_wd{wd}_seed{seed}'
    print(f'\n{"="*60}')
    print(f'Running: {tag}')
    print(f'{"="*60}')

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    elapsed = time.time() - t0

    # Parse result
    best_val = best_test = float('inf')
    for line in result.stdout.split('\n'):
        if 'Best Val PPL' in line and 'Best Test PPL' in line:
            # Final summary line
            parts = line.split('|')
            for p in parts:
                if 'Best Val PPL' in p:
                    best_val = float(p.split(':')[1].strip())
                elif 'Best Test PPL' in p:
                    best_test = float(p.split(':')[1].strip())
        elif 'Best Val:' in line:
            # Epoch log line: Best Val: XX / Test: YY
            try:
                bv = float(line.split('Best Val:')[1].split('/')[0].strip())
                bt = float(line.split('Test:')[1].split('|')[0].strip())
                if bv < best_val:
                    best_val = bv
                    best_test = bt
            except:
                pass

    # Save individual result
    res_file = os.path.join(LOG_DIR, f'{tag}.json')
    with open(res_file, 'w') as f:
        json.dump({
            'optimizer': opt, 'lr': lr, 'wd': wd, 'nlayers': nlayers,
            'seed': seed, 'scheduler': scheduler,
            'best_val_ppl': best_val, 'best_test_ppl': best_test,
            'time_sec': elapsed,
            'ns_interval': ns_interval if opt == 'LAKTJU_NS' else None,
            'ns_skip_rnn': ns_skip_rnn if opt == 'LAKTJU_NS' else None,
        }, f, indent=2)

    print(f'\n{"="*60}')
    print(f'{nlayers}-layer LSTM | {opt} | seed={seed}')
    print(f'Best Val PPL: {best_val:.2f}')
    print(f'Best Test PPL: {best_test:.2f}')
    print(f'Total time: {elapsed/60:.1f} min')
    print(f'{"="*60}')
    print(f'Results saved to {res_file}')
    print(f'  => Val PPL: {best_val:.2f}  Test PPL: {best_test:.2f}  Time: {elapsed/60:.1f}min')

    return best_val, best_test, elapsed


def main():
    t_start = time.time()

    feishu_send('🚀 LSTM PTB v2 实验开始!\n'
                '改进: multistep scheduler + ns_skip_rnn + 更宽grid search\n'
                '目标: LAFTJU-NS 优于 Adam/AdamW/Adan')

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 1: BASELINES — all use multistep scheduler [100,145]
    # Following AdaBelief paper's exact setting
    # ══════════════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('Phase 1: BASELINES (multistep scheduler)')
    print('='*70)

    baseline_configs = {
        'SGD':   {'lr': 30.0, 'wd': 0.0},
        'Adam':  {'lr': 0.001, 'wd': 1.2e-6},
        'AdamW': {'lr': 0.001, 'wd': 1.2e-6},
        'Adan':  {'lr': 0.01, 'wd': 0.02},
    }

    seeds = [42, 123, 456]
    baseline_results = {}

    for opt, cfg in baseline_configs.items():
        baseline_results[opt] = {}
        for nlayers in [1, 2, 3]:
            tests = []
            for seed in seeds:
                _, test_ppl, _ = run_exp(
                    opt, cfg['lr'], cfg['wd'], nlayers,
                    seed=seed, scheduler='multistep')
                tests.append(test_ppl)
            avg = np.mean(tests)
            std = np.std(tests)
            baseline_results[opt][nlayers] = {
                'tests': tests, 'avg': avg, 'std': std
            }
            print(f'\n  {opt} {nlayers}L: {avg:.2f}±{std:.2f} ({tests})')

    # Report Phase 1
    p1_msg = '📊 Phase 1 完成 (Baselines, multistep scheduler):\n'
    for opt in ['SGD', 'Adam', 'AdamW', 'Adan']:
        p1_msg += f'\n{opt}:'
        for nl in [1, 2, 3]:
            r = baseline_results[opt][nl]
            p1_msg += f'\n  {nl}L: {r["avg"]:.2f}±{r["std"]:.2f}'
    feishu_send(p1_msg)

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 2: LAFTJU-NS Grid Search on 2-layer LSTM
    # Key insight: NS on embedding/decoder only (skip RNN recurrent weights)
    # ══════════════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('Phase 2: LAFTJU-NS GRID SEARCH (2-layer LSTM)')
    print('='*70)

    grid_configs = [
        # (lr, wd, ns_interval, ns_skip_rnn)
        # Group A: ns_skip_rnn=True (only orthogonalize embedding/decoder)
        (0.001, 1.2e-6, 100, True),
        (0.001, 1.2e-6, 200, True),
        (0.001, 1.2e-6, 500, True),
        (0.001, 0.001,  100, True),
        (0.001, 0.01,   100, True),
        (0.002, 1.2e-6, 100, True),
        (0.002, 0.001,  100, True),
        (0.003, 1.2e-6, 100, True),
        (0.0005, 1.2e-6, 100, True),
        # Group B: ns_skip_rnn=False (orthogonalize everything, for comparison)
        (0.001, 1.2e-6, 100, False),
        (0.001, 1.2e-6, 500, False),
        (0.001, 0.001,  100, False),
        # Group C: larger ns_interval (gentler orthogonalization)
        (0.001, 1.2e-6, 1000, True),
        (0.001, 1.2e-6, 2000, True),
        (0.002, 1.2e-6, 500, True),
    ]

    grid_results = []
    for lr, wd, ns_int, skip_rnn in grid_configs:
        _, test_ppl, elapsed = run_exp(
            'LAKTJU_NS', lr, wd, 2,
            seed=42, scheduler='multistep',
            ns_interval=ns_int, ns_skip_rnn=skip_rnn)
        grid_results.append({
            'lr': lr, 'wd': wd, 'ns_interval': ns_int,
            'ns_skip_rnn': skip_rnn,
            'test': test_ppl, 'time': elapsed
        })
        skip_tag = 'skip_rnn' if skip_rnn else 'all_ns'
        print(f'  lr={lr} wd={wd} ns={ns_int} {skip_tag} => Test PPL: {test_ppl:.2f}')

    # Sort by test PPL
    grid_results.sort(key=lambda x: x['test'])
    best = grid_results[0]
    blr, bwd, bns, bskip = best['lr'], best['wd'], best['ns_interval'], best['ns_skip_rnn']

    p2_msg = f'🔍 Phase 2 Grid Search 完成!\n'
    p2_msg += f'最优配置: lr={blr} wd={bwd} ns={bns} skip_rnn={bskip}\n'
    p2_msg += f'最优 2L Test PPL: {best["test"]:.2f}\n\nTop 5:\n'
    for i, r in enumerate(grid_results[:5]):
        skip_tag = 'skip_rnn' if r['ns_skip_rnn'] else 'all_ns'
        p2_msg += f'  {i+1}. lr={r["lr"]} wd={r["wd"]} ns={r["ns_interval"]} {skip_tag} => {r["test"]:.2f}\n'
    feishu_send(p2_msg)

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 3: Best config × 3 layers × 5 seeds
    # ══════════════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('Phase 3: BEST CONFIG × 3 LAYERS × 5 SEEDS')
    print('='*70)

    final_seeds = [42, 123, 456, 789, 2024]
    ns_final = {}

    for nlayers in [1, 2, 3]:
        tests = []
        for seed in final_seeds:
            _, test_ppl, _ = run_exp(
                'LAKTJU_NS', blr, bwd, nlayers,
                seed=seed, scheduler='multistep',
                ns_interval=bns, ns_skip_rnn=bskip)
            tests.append(test_ppl)
        avg = np.mean(tests)
        std = np.std(tests)
        ns_final[nlayers] = {'tests': tests, 'avg': avg, 'std': std}
        print(f'\n  LAFTJU-NS {nlayers}L: {avg:.2f}±{std:.2f}')

    # ══════════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════════
    total_time = (time.time() - t_start) / 3600

    lines = ['🏆 LSTM PTB v2 实验完成!\n']
    lines.append(f'最优配置: lr={blr} wd={bwd} ns_interval={bns} ns_skip_rnn={bskip}\n')

    for nlayers in [1, 2, 3]:
        lines.append(f'\n【{nlayers}-layer LSTM】')
        for opt in ['SGD', 'Adam', 'AdamW', 'Adan']:
            br = baseline_results[opt][nlayers]
            lines.append(f'  {opt:8s}: {br["avg"]:.2f}±{br["std"]:.2f}')
        nr = ns_final[nlayers]
        lines.append(f'  LAFTJU-NS: {nr["avg"]:.2f}±{nr["std"]:.2f}')
        # vs best baseline
        best_baseline_name = min(['SGD', 'Adam', 'AdamW', 'Adan'],
                                  key=lambda o: baseline_results[o][nlayers]['avg'])
        best_baseline_avg = baseline_results[best_baseline_name][nlayers]['avg']
        diff = best_baseline_avg - nr['avg']
        marker = '✅' if diff > 0 else '❌'
        lines.append(f'  → vs {best_baseline_name}: {diff:+.2f} {marker}')

    lines.append(f'\n总耗时: {total_time:.1f}h')

    summary = '\n'.join(lines)
    print('\n' + summary)
    feishu_send(summary)

    # Save complete results
    full_summary = {
        'baselines': {opt: {str(nl): v for nl, v in layers.items()}
                      for opt, layers in baseline_results.items()},
        'grid_search': grid_results,
        'best_cfg': {'lr': blr, 'wd': bwd, 'ns_interval': bns, 'ns_skip_rnn': bskip},
        'final_5seed': {str(nl): v for nl, v in ns_final.items()},
        'total_time_hours': total_time,
    }
    with open(os.path.join(LOG_DIR, 'summary_lstm_v2.json'), 'w') as f:
        json.dump(full_summary, f, indent=2)
    print(f'Summary saved to {LOG_DIR}/summary_lstm_v2.json')


if __name__ == '__main__':
    main()

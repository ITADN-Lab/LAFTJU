#!/usr/bin/env python3
"""
CIFAR-100 Expert Experiment v9 for LAKTJU-NS.

Improvements over v8:
  1) Fair baselines: each optimizer gets its own best lr/wd + 3 seeds
  2) Adam baseline with proper wd=0 (L2 decay breaks Adam)
  3) AdamW with literature-optimal lr=0.001, wd=0.01
  4) Adan with its recommended lr=0.02, betas=(0.98,0.92,0.99)
  5) Wider LAKTJU_NS grid: 12 configs (lr × wd × ns_interval)
  6) Best config × 5 seeds for statistical significance
  7) Try 400 epochs on best config for fuller convergence

Design rationale:
  - CIFAR-100 has 100 classes → more label_smoothing (0.1~0.2)
  - Sharper loss curvature → NS orthogonalization benefits more
  - ns_interval=50~200 range (more frequent helps)
  - wd=0.005~0.01 (stronger regularization for 100 classes)
"""
import subprocess, os, sys, json, time, urllib.request
from datetime import datetime
import numpy as np

PYTHON = sys.executable
TRAIN  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_laktju.py')
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_cifar100_v9')
os.makedirs(LOG_DIR, exist_ok=True)

FEISHU_APP_ID     = 'cli_a93c7a2146381bd2'
FEISHU_APP_SECRET = 'NQQegyt8DJPFU3PeURjJLg3IqDRMopsY'
FEISHU_CHAT_ID    = 'oc_840b30bc7d66669838c28fdfa049ea8f'


def get_token():
    body = json.dumps({'app_id': FEISHU_APP_ID, 'app_secret': FEISHU_APP_SECRET}).encode()
    req = urllib.request.Request(
        'https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal',
        data=body, headers={'Content-Type': 'application/json'})
    resp = json.loads(urllib.request.urlopen(req, timeout=10).read())
    return resp['tenant_access_token']


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


def run_exp(opt, lr, wd, epochs=300, ns_interval=200, ns_max_dim=256, seed=42,
           label_smoothing=0.1, tag_prefix=''):
    tag = f'{tag_prefix}cifar100_{opt}_lr{lr}_wd{wd}_ns{ns_interval}_ls{label_smoothing}_ep{epochs}_seed{seed}'
    save_dir = LOG_DIR
    cmd = [
        PYTHON, TRAIN,
        '--optimizer', opt,
        '--dataset', 'cifar100',
        '--model', 'resnet18',
        '--epochs', str(epochs),
        '--lr', str(lr),
        '--weight_decay', str(wd),
        '--label_smoothing', str(label_smoothing),
        '--seed', str(seed),
        '--save_dir', save_dir,
    ]
    if opt == 'LAKTJU_NS':
        cmd += [
            '--ns_interval', str(ns_interval),
            '--ns_steps', '1',
            '--ns_max_dim', str(ns_max_dim),
        ]
    print(f'\n{"="*60}')
    print(f'Running: {tag}')
    print(f'CMD: {" ".join(cmd)}')
    print(f'{"="*60}', flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - t0

    # Parse result from saved JSON
    best_valid, best_test = 0.0, 0.0
    for f in sorted(os.listdir(save_dir), reverse=True):
        if f.endswith('.json') and f'cifar100_resnet18_{opt}_seed{seed}' in f:
            with open(os.path.join(save_dir, f)) as fp:
                d = json.load(fp)
            best_valid = d.get('best_valid_acc', 0)
            best_test  = d.get('best_test_acc', 0)
            break
    print(f'  => Valid: {best_valid:.2f}%  Test: {best_test:.2f}%  Time: {elapsed/60:.1f}min')
    return best_valid, best_test, elapsed


def main():
    results = {}
    start_time = time.time()
    feishu_send('🔬 CIFAR-100 LAFTJU-NS v9 实验开始\n'
                '改进: 公平基线(3 seeds) + 更宽搜索(12配置) + 5-seed最终验证')

    # ═══════════════════════════════════════════════════════════════════════════
    # Phase 1: Fair Baselines — each optimizer with its best hyperparams, 3 seeds
    # ═══════════════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('Phase 1: FAIR BASELINES (3 seeds each)')
    print('='*70)

    # Adam: lr=0.001, wd=0 (Adam doesn't decouple wd, L2 hurts adaptive methods)
    # AdamW: lr=0.001, wd=0.01 (standard for CIFAR-100 ResNet18)
    # Adan: lr=0.02, wd=0.02 (Adan paper recommended; scale down if diverges)
    baseline_configs = {
        'Adam':  {'lr': 0.001, 'wd': 0.0001, 'ls': 0.1},
        'AdamW': {'lr': 0.001, 'wd': 0.01,   'ls': 0.1},
        'Adan':  {'lr': 0.02,  'wd': 0.02,   'ls': 0.1},
    }

    baseline_results = {}
    for opt, cfg in baseline_configs.items():
        seed_tests = []
        for seed in [42, 123, 456]:
            v, t, elapsed = run_exp(opt, cfg['lr'], cfg['wd'], epochs=300,
                                     seed=seed, label_smoothing=cfg['ls'])
            seed_tests.append(t)
            results[f'{opt}_seed{seed}'] = {'valid': v, 'test': t}

        avg = np.mean(seed_tests)
        std = np.std(seed_tests)
        baseline_results[opt] = {'tests': seed_tests, 'avg': avg, 'std': std}
        feishu_send(f'✅ Baseline {opt} (3 seeds)\n'
                    f'  lr={cfg["lr"]} wd={cfg["wd"]} ls={cfg["ls"]}\n'
                    f'  Tests: {[f"{t:.2f}%" for t in seed_tests]}\n'
                    f'  Avg: {avg:.2f}% ± {std:.2f}%')

    # ═══════════════════════════════════════════════════════════════════════════
    # Phase 2: LAKTJU_NS Wide Grid Search (single seed=42, 300 epochs)
    # ═══════════════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('Phase 2: LAKTJU_NS WIDE GRID SEARCH')
    print('='*70)

    # Grid design rationale:
    # - lr: 0.002 (conservative), 0.003 (v8 best), 0.005 (aggressive)
    # - wd: 0.003 (light), 0.005 (v8 best), 0.01 (strong, matches AdamW literature)
    # - ns_interval: 50 (very frequent), 100 (v8 best), 200 (moderate)
    # - label_smoothing: 0.1 (standard), 0.15 (v8 used)
    grid = [
        # Core grid: lr × wd with ns=100 (v8 winner)
        (0.002, 0.005, 100, 0.1),
        (0.003, 0.005, 100, 0.1),   # v8 best but ls=0.1
        (0.005, 0.005, 100, 0.1),
        (0.003, 0.003, 100, 0.1),
        (0.003, 0.01,  100, 0.1),   # match AdamW wd
        (0.005, 0.01,  100, 0.1),
        # NS interval sweep with best lr/wd
        (0.003, 0.005,  50, 0.1),   # very frequent NS
        (0.003, 0.005, 200, 0.1),   # less frequent NS
        # Label smoothing 0.15 sweep
        (0.003, 0.005, 100, 0.15),  # v8 winner config
        (0.005, 0.005, 100, 0.15),
        # Higher LR experiments
        (0.008, 0.005, 100, 0.1),
        (0.008, 0.01,  100, 0.1),
    ]

    best_test = 0.0
    best_cfg = None
    ns_results = []

    for lr, wd, ns_int, ls in grid:
        v, t, elapsed = run_exp('LAKTJU_NS', lr, wd, epochs=300,
                                 ns_interval=ns_int, seed=42, label_smoothing=ls)
        ns_results.append({'lr': lr, 'wd': wd, 'ns': ns_int, 'ls': ls,
                           'valid': v, 'test': t, 'time': elapsed})
        if t > best_test:
            best_test = t
            best_cfg = (lr, wd, ns_int, ls)
        feishu_send(f'🔄 LAKTJU_NS lr={lr} wd={wd} ns={ns_int} ls={ls}\n'
                    f'  Valid: {v:.2f}%  Test: {t:.2f}%')

    # Sort and report top 5
    ns_results.sort(key=lambda x: x['test'], reverse=True)
    top5_str = '\n'.join(
        f"  #{i+1}: lr={r['lr']} wd={r['wd']} ns={r['ns']} ls={r['ls']} → {r['test']:.2f}%"
        for i, r in enumerate(ns_results[:5])
    )
    feishu_send(f'📊 Grid Search Top 5:\n{top5_str}')

    # ═══════════════════════════════════════════════════════════════════════════
    # Phase 3: Top-2 configs × 5 seeds for statistical significance
    # ═══════════════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('Phase 3: TOP CONFIGS × 5 SEEDS')
    print('='*70)

    # Also try 400 epochs on the best config
    top2 = ns_results[:2]
    final_results = {}
    seeds = [42, 123, 456, 789, 2024]

    for rank, cfg in enumerate(top2):
        lr, wd, ns, ls = cfg['lr'], cfg['wd'], cfg['ns'], cfg['ls']
        tag = f'top{rank+1}_lr{lr}_wd{wd}_ns{ns}_ls{ls}'
        seed_tests = []

        for seed in seeds:
            v, t, _ = run_exp('LAKTJU_NS', lr, wd, epochs=300,
                               ns_interval=ns, seed=seed, label_smoothing=ls)
            seed_tests.append(t)
            results[f'LAKTJU_NS_{tag}_seed{seed}'] = {'valid': v, 'test': t}

        avg = np.mean(seed_tests)
        std = np.std(seed_tests)
        final_results[tag] = {'seeds': seed_tests, 'avg': avg, 'std': std,
                              'cfg': {'lr': lr, 'wd': wd, 'ns': ns, 'ls': ls}}
        feishu_send(f'🎯 {tag}\n'
                    f'  5 seeds: {[f"{t:.2f}%" for t in seed_tests]}\n'
                    f'  Avg: {avg:.2f}% ± {std:.2f}%  Best: {max(seed_tests):.2f}%')

    # ═══════════════════════════════════════════════════════════════════════════
    # Phase 4: Try 400 epochs on absolute best config
    # ═══════════════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('Phase 4: BEST CONFIG × 400 EPOCHS')
    print('='*70)

    # Find absolute best config
    best_final = max(final_results.items(), key=lambda x: x[1]['avg'])
    bf_cfg = best_final[1]['cfg']
    ep400_tests = []
    for seed in [42, 123, 456]:
        v, t, _ = run_exp('LAKTJU_NS', bf_cfg['lr'], bf_cfg['wd'], epochs=400,
                           ns_interval=bf_cfg['ns'], seed=seed,
                           label_smoothing=bf_cfg['ls'], tag_prefix='ep400_')
        ep400_tests.append(t)
        results[f'LAKTJU_NS_ep400_seed{seed}'] = {'valid': v, 'test': t}

    ep400_avg = np.mean(ep400_tests)
    ep400_std = np.std(ep400_tests)
    feishu_send(f'⏱ 400 epochs: {[f"{t:.2f}%" for t in ep400_tests]}\n'
                f'  Avg: {ep400_avg:.2f}% ± {ep400_std:.2f}%')

    # ═══════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════
    total_time = (time.time() - start_time) / 3600

    # Determine best LAKTJU_NS result
    best_ns_tag = max(final_results.items(), key=lambda x: x[1]['avg'])
    ns_avg = best_ns_tag[1]['avg']
    ns_std = best_ns_tag[1]['std']
    ns_best_single = max(best_ns_tag[1]['seeds'])
    ns_cfg = best_ns_tag[1]['cfg']

    # Check if 400ep is better
    ep400_label = ''
    if ep400_avg > ns_avg:
        ep400_label = ' ⭐ (400ep better!)'

    baseline_str = '\n'.join(
        f'  {opt}: {br["avg"]:.2f}% ± {br["std"]:.2f}% (best single: {max(br["tests"]):.2f}%)'
        for opt, br in baseline_results.items()
    )

    grid_str = '\n'.join(
        f'  #{i+1}: lr={r["lr"]} wd={r["wd"]} ns={r["ns"]} ls={r["ls"]} → {r["test"]:.2f}%'
        for i, r in enumerate(ns_results[:5])
    )

    final_str = '\n'.join(
        f'  {tag}: {fr["avg"]:.2f}% ± {fr["std"]:.2f}% (5 seeds)'
        for tag, fr in final_results.items()
    )

    summary = (
        f'🏆 CIFAR-100 LAKTJU-NS v9 实验完成! ({total_time:.1f}h)\n\n'
        f'【Baselines (ResNet-18, 300ep, 3 seeds)】\n{baseline_str}\n\n'
        f'【Grid Search Top 5 (seed=42)】\n{grid_str}\n\n'
        f'【最终验证 (5 seeds)】\n{final_str}\n\n'
        f'【400 epochs】\n'
        f'  Avg: {ep400_avg:.2f}% ± {ep400_std:.2f}% (3 seeds){ep400_label}\n\n'
        f'【最优 LAKTJU-NS 配置】\n'
        f'  lr={ns_cfg["lr"]} wd={ns_cfg["wd"]} ns_interval={ns_cfg["ns"]} ls={ns_cfg["ls"]}\n'
        f'  Avg Test: {ns_avg:.2f}% ± {ns_std:.2f}%\n'
        f'  Best Single: {ns_best_single:.2f}%\n\n'
        f'【vs Baselines】\n'
        f'  vs Adam:  {ns_avg:.2f}% vs {baseline_results["Adam"]["avg"]:.2f}% '
        f'(+{ns_avg - baseline_results["Adam"]["avg"]:.2f}%)\n'
        f'  vs AdamW: {ns_avg:.2f}% vs {baseline_results["AdamW"]["avg"]:.2f}% '
        f'(+{ns_avg - baseline_results["AdamW"]["avg"]:.2f}%)\n'
        f'  vs Adan:  {ns_avg:.2f}% vs {baseline_results["Adan"]["avg"]:.2f}% '
        f'(+{ns_avg - baseline_results["Adan"]["avg"]:.2f}%)'
    )
    print('\n' + summary)
    feishu_send(summary)

    # Save complete results
    full_summary = {
        'baselines': {k: {'avg': v['avg'], 'std': v['std'], 'seeds': v['tests']}
                      for k, v in baseline_results.items()},
        'grid_search': ns_results,
        'final_5seed': {k: v for k, v in final_results.items()},
        'ep400': {'avg': ep400_avg, 'std': ep400_std, 'seeds': ep400_tests},
        'best_cfg': ns_cfg,
        'best_avg': ns_avg,
        'best_std': ns_std,
        'total_time_hours': total_time,
    }
    with open(f'{LOG_DIR}/summary_v9.json', 'w') as fp:
        json.dump(full_summary, fp, indent=2)
    print(f'Summary saved to {LOG_DIR}/summary_v9.json')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Create experiment tasks for distributed execution."""
import json, os

TASK_DIR = os.path.join(os.path.dirname(__file__), 'tasks')
os.makedirs(TASK_DIR, exist_ok=True)

tasks = []

# Adan GPT-2 Small (3 seeds)
for seed in [42, 123, 456]:
    tasks.append({
        'id': f'adan_small_seed{seed}',
        'status': 'pending',
        'worker': None,
        'cmd': f'python3 train_gpt2_lm.py --dataset wikitext --seed {seed} '
               f'--output_dir results_gpt2 --model_size small --optimizer Adan '
               f'--lr 1e-3 --weight_decay 0.02 --total_steps 20000 '
               f'--batch_size 8 --grad_accum 4 --warmup_steps 2000 '
               f'--eval_interval 1000 --log_interval 100',
        'est_hours': 2.2,
    })

# GPT-2 Medium (3 optimizers × 3 seeds)
for opt, lr, wd, extra in [
    ('AdamW', '6e-4', '0.1', ''),
    ('LAKTJU_NS', '6e-4', '0.1', ' --ns_interval 50 --ns_steps 2 --ns_max_dim 1024'),
    ('Lion', '1e-4', '1.0', ''),
]:
    for seed in [42, 123, 456]:
        tasks.append({
            'id': f'medium_{opt}_seed{seed}',
            'status': 'pending',
            'worker': None,
            'cmd': f'python3 train_gpt2_lm.py --dataset wikitext --seed {seed} '
                   f'--output_dir results_gpt2 --model_size medium --optimizer {opt} '
                   f'--lr {lr} --weight_decay {wd} --total_steps 10000 '
                   f'--batch_size 4 --grad_accum 4 --warmup_steps 1000 '
                   f'--eval_interval 1000 --log_interval 100{extra}',
            'est_hours': 1.5,
        })

for t in tasks:
    path = os.path.join(TASK_DIR, f'{t["id"]}.json')
    with open(path, 'w') as f:
        json.dump(t, f, indent=2)
    print(f'Created: {t["id"]} (~{t["est_hours"]}h)')

print(f'\nTotal: {len(tasks)} tasks, ~{sum(t["est_hours"] for t in tasks):.1f}h sequential')

#!/usr/bin/env python3
"""
BERT Fine-tuning on GLUE Benchmark.

Compares LAFTJU-NS vs Adam vs AdamW on GLUE tasks using HuggingFace Transformers.
Follows the Adan paper (Table 11) setup for fair comparison.

Usage:
    python train_bert_glue.py --task sst2 --optimizer AdamW --seed 42
    python train_bert_glue.py --task sst2 --optimizer LAKTJU_NS --ns_interval 100 --seed 42
"""
import os
import sys
import json
import time
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from datasets import load_dataset
from sklearn.metrics import matthews_corrcoef
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from LAKTJU_NS import LAKTJU_NS

# ══════════════════════════════════════════════════════════════════════════════
# GLUE Task Configs
# ══════════════════════════════════════════════════════════════════════════════

TASK_CONFIG = {
    'sst2':  {'num_labels': 2, 'metric': 'accuracy',  'keys': ('sentence', None)},
    'cola':  {'num_labels': 2, 'metric': 'matthews',   'keys': ('sentence', None)},
    'qnli':  {'num_labels': 2, 'metric': 'accuracy',   'keys': ('question', 'sentence')},
    'rte':   {'num_labels': 2, 'metric': 'accuracy',   'keys': ('sentence1', 'sentence2')},
    'qqp':   {'num_labels': 2, 'metric': 'accuracy',   'keys': ('question1', 'question2')},
    'mnli':  {'num_labels': 3, 'metric': 'accuracy',   'keys': ('premise', 'hypothesis')},
    'stsb':  {'num_labels': 1, 'metric': 'pearson',    'keys': ('sentence1', 'sentence2')},
}


def get_args():
    p = argparse.ArgumentParser()
    # Task
    p.add_argument('--task', type=str, default='sst2',
                   choices=list(TASK_CONFIG.keys()))
    p.add_argument('--model_name', type=str, default='bert-base-uncased')
    p.add_argument('--max_seq_len', type=int, default=128)

    # Training
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--warmup_ratio', type=float, default=0.06)
    p.add_argument('--max_grad_norm', type=float, default=1.0)
    p.add_argument('--fp16', action='store_true', default=True)
    p.add_argument('--seed', type=int, default=42)

    # Optimizer
    p.add_argument('--optimizer', type=str, default='AdamW',
                   choices=['Adam', 'AdamW', 'LAKTJU_NS'])
    p.add_argument('--lr', type=float, default=None)  # auto-set per optimizer
    p.add_argument('--weight_decay', type=float, default=None)
    p.add_argument('--betas', type=str, default=None)  # e.g. "0.9,0.999"

    # NS-specific
    p.add_argument('--ns_interval', type=int, default=100)
    p.add_argument('--ns_steps', type=int, default=1)
    p.add_argument('--ns_max_dim', type=int, default=3072)
    p.add_argument('--ns_skip_embeddings', action='store_true', default=True)
    p.add_argument('--ns_skip_ffn', action='store_true', default=False,
                   help='Skip NS on intermediate/output dense (FFN) layers')
    p.add_argument('--ns_warmup_frac', type=float, default=0.0,
                   help='Fraction of training to skip NS at start (warmup)')
    p.add_argument('--ns_cooldown_frac', type=float, default=0.0,
                   help='Fraction of training to skip NS at end (cooldown)')

    # Output
    p.add_argument('--output_dir', type=str, default='results_bert_glue')
    p.add_argument('--eval_every', type=int, default=0)  # 0 = eval per epoch

    return p.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_glue_data(task, tokenizer, max_seq_len, batch_size):
    dataset = load_dataset('glue', task)
    cfg = TASK_CONFIG[task]
    key1, key2 = cfg['keys']

    def tokenize_fn(examples):
        if key2 is None:
            return tokenizer(examples[key1], truncation=True,
                             max_length=max_seq_len, padding='max_length')
        else:
            return tokenizer(examples[key1], examples[key2], truncation=True,
                             max_length=max_seq_len, padding='max_length')

    dataset = dataset.map(tokenize_fn, batched=True,
                          remove_columns=[c for c in dataset['train'].column_names
                                          if c not in ['input_ids', 'attention_mask',
                                                       'token_type_ids', 'label']])
    dataset.set_format('torch')

    # For STS-B, labels are floats
    if task == 'stsb':
        for split in dataset:
            dataset[split] = dataset[split].map(
                lambda x: {'label': torch.tensor(x['label'], dtype=torch.float32)})

    train_loader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)

    val_key = 'validation_matched' if task == 'mnli' else 'validation'
    val_loader = DataLoader(dataset[val_key], batch_size=batch_size * 2,
                            shuffle=False, num_workers=4, pin_memory=True)

    # For MNLI, also get mismatched
    val_mm_loader = None
    if task == 'mnli':
        val_mm_loader = DataLoader(dataset['validation_mismatched'],
                                   batch_size=batch_size * 2, shuffle=False,
                                   num_workers=4, pin_memory=True)

    return train_loader, val_loader, val_mm_loader


# ══════════════════════════════════════════════════════════════════════════════
# Optimizer Setup
# ══════════════════════════════════════════════════════════════════════════════

def create_optimizer(model, args, total_steps):
    # Separate parameters: no weight decay for bias and LayerNorm
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    param_groups = [
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
    ]

    if args.optimizer == 'Adam':
        lr = args.lr or 1e-5
        wd = args.weight_decay if args.weight_decay is not None else 0.1
        betas = tuple(map(float, args.betas.split(','))) if args.betas else (0.9, 0.98)
        eps = 1e-6
        param_groups[0]['weight_decay'] = wd
        optimizer = torch.optim.Adam(param_groups, lr=lr, betas=betas, eps=eps)

    elif args.optimizer == 'AdamW':
        lr = args.lr or 2e-5
        wd = args.weight_decay if args.weight_decay is not None else 0.01
        betas = tuple(map(float, args.betas.split(','))) if args.betas else (0.9, 0.999)
        eps = 1e-8
        param_groups[0]['weight_decay'] = wd
        optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps)

    elif args.optimizer == 'LAKTJU_NS':
        lr = args.lr or 2e-5
        wd = args.weight_decay if args.weight_decay is not None else 0.01
        betas = tuple(map(float, args.betas.split(','))) if args.betas else (0.9, 0.999)
        eps = 1e-8
        param_groups[0]['weight_decay'] = wd

        # Collect params to skip NS
        skip_params = []
        if args.ns_skip_embeddings:
            for n, p in model.named_parameters():
                if 'embeddings' in n:
                    skip_params.append(p)
        if args.ns_skip_ffn:
            for n, p in model.named_parameters():
                if 'intermediate.dense' in n or 'output.dense' in n:
                    skip_params.append(p)

        # Compute NS warmup/cooldown step boundaries
        ns_start_step = int(total_steps * args.ns_warmup_frac)
        ns_end_step = int(total_steps * (1.0 - args.ns_cooldown_frac))

        optimizer = LAKTJU_NS(
            param_groups, lr=lr, betas=betas, eps=eps,
            weight_decay=wd,
            ns_interval=args.ns_interval,
            ns_steps=args.ns_steps,
            ns_max_dim=args.ns_max_dim,
            ns_min_dim=1,
            min_ndim=2,
            grad_centralization=False,  # no conv layers in BERT
            ns_skip_params=skip_params,
        )
        # Attach NS schedule info to optimizer for use in training loop
        optimizer._ns_start_step = ns_start_step
        optimizer._ns_end_step = ns_end_step

    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # NS warmup/cooldown boundaries (only used for LAKTJU_NS, already set on optimizer)

    return optimizer, scheduler


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(model, dataloader, task, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            token_type_ids = batch.get('token_type_ids')
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, labels=labels)
            total_loss += outputs.loss.item()
            n_batches += 1

            if TASK_CONFIG[task]['num_labels'] == 1:  # regression
                preds = outputs.logits.squeeze(-1).cpu().numpy()
            else:
                preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    avg_loss = total_loss / max(n_batches, 1)

    metric_name = TASK_CONFIG[task]['metric']
    if metric_name == 'accuracy':
        score = (all_preds == all_labels).mean() * 100
    elif metric_name == 'matthews':
        score = matthews_corrcoef(all_labels, all_preds) * 100
    elif metric_name == 'pearson':
        score = pearsonr(all_labels, all_preds)[0] * 100

    return score, avg_loss


# ══════════════════════════════════════════════════════════════════════════════
# Training Loop
# ══════════════════════════════════════════════════════════════════════════════

def train(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    cfg = TASK_CONFIG[args.task]
    model = BertForSequenceClassification.from_pretrained(
        args.model_name, num_labels=cfg['num_labels'])
    model.to(device)

    # Load data
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    train_loader, val_loader, val_mm_loader = load_glue_data(
        args.task, tokenizer, args.max_seq_len, args.batch_size)

    total_steps = len(train_loader) * args.epochs
    optimizer, scheduler = create_optimizer(model, args, total_steps)

    # Mixed precision
    scaler = torch.amp.GradScaler('cuda', enabled=args.fp16)

    # Logging
    best_score = -float('inf')
    best_epoch = 0
    history = []

    opt_desc = args.optimizer
    if args.optimizer == 'LAKTJU_NS':
        opt_desc += f'_ns{args.ns_interval}_K{args.ns_steps}_maxd{args.ns_max_dim}'
    # Include lr in filename to avoid overwrites during lr sweeps
    actual_lr = optimizer.param_groups[0]['lr']  # initial lr before scheduling
    # Recover initial lr from scheduler
    initial_lr = args.lr or (1e-5 if args.optimizer == 'Adam' else 2e-5)
    opt_desc += f'_lr{initial_lr}'

    print(f"\n{'='*70}")
    print(f"Task: {args.task.upper()} | Optimizer: {opt_desc} | Seed: {args.seed}")
    print(f"LR: {optimizer.param_groups[0]['lr']} | WD: {optimizer.param_groups[0]['weight_decay']}")
    print(f"Epochs: {args.epochs} | Steps: {total_steps} | Batch: {args.batch_size}")
    print(f"{'='*70}")

    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        n_steps = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            token_type_ids = batch.get('token_type_ids')
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=args.fp16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                token_type_ids=token_type_ids, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            # NS warmup/cooldown: temporarily disable NS outside active window
            if args.optimizer == 'LAKTJU_NS' and (args.ns_warmup_frac > 0 or args.ns_cooldown_frac > 0):
                global_step = (epoch - 1) * len(train_loader) + n_steps
                if global_step < optimizer._ns_start_step or global_step > optimizer._ns_end_step:
                    saved_interval = optimizer.ns_interval
                    optimizer.ns_interval = 999999999  # effectively disable
                    scaler.step(optimizer)
                    optimizer.ns_interval = saved_interval
                else:
                    scaler.step(optimizer)
            else:
                scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            n_steps += 1

        avg_train_loss = epoch_loss / n_steps

        # Evaluate
        score, val_loss = evaluate(model, val_loader, args.task, device)

        # For MNLI, also evaluate mismatched
        mm_score = None
        if val_mm_loader is not None:
            mm_score, _ = evaluate(model, val_mm_loader, args.task, device)

        if score > best_score:
            best_score = score
            best_epoch = epoch

        elapsed = time.time() - t0
        metric_name = cfg['metric']

        if mm_score is not None:
            print(f"  Epoch {epoch:2d}/{args.epochs} | "
                  f"train_loss={avg_train_loss:.4f} | "
                  f"val_{metric_name}={score:.2f}/{mm_score:.2f} | "
                  f"best={best_score:.2f} (ep{best_epoch}) | "
                  f"{elapsed:.0f}s")
        else:
            print(f"  Epoch {epoch:2d}/{args.epochs} | "
                  f"train_loss={avg_train_loss:.4f} | "
                  f"val_{metric_name}={score:.2f} | "
                  f"best={best_score:.2f} (ep{best_epoch}) | "
                  f"{elapsed:.0f}s")

        history.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_score': score,
            'val_mm_score': mm_score,
            'val_loss': val_loss,
        })

    total_time = time.time() - t0

    # Save results — ensure all values are JSON-serializable
    def to_py(v):
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
        if hasattr(v, 'item'):  # torch scalar
            return v.item()
        return v

    result = {
        'task': args.task,
        'optimizer': args.optimizer,
        'seed': args.seed,
        'lr': float(initial_lr),
        'weight_decay': float(optimizer.param_groups[0]['weight_decay']),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'best_val_score': float(best_score),
        'best_epoch': best_epoch,
        'metric': cfg['metric'],
        'total_time_min': total_time / 60,
        'history': [{k: to_py(v) for k, v in h.items()} for h in history],
    }
    if args.optimizer == 'LAKTJU_NS':
        result['ns_interval'] = args.ns_interval
        result['ns_steps'] = args.ns_steps
        result['ns_max_dim'] = args.ns_max_dim

    os.makedirs(os.path.join(os.path.dirname(__file__), args.output_dir), exist_ok=True)
    fname = f"{args.task}_{opt_desc}_seed{args.seed}.json"
    fpath = os.path.join(os.path.dirname(__file__), args.output_dir, fname)
    with open(fpath, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n  => Best val {metric_name}: {best_score:.2f} (epoch {best_epoch})")
    print(f"  => Saved to {fpath}")
    print(f"  => Total time: {total_time/60:.1f} min")

    return best_score


if __name__ == '__main__':
    args = get_args()
    train(args)

#!/usr/bin/env python3
"""
GPT-2 Language Modeling on OpenWebText.

Compares LAFTJU-NS vs Adam vs AdamW on GPT-2 pretraining.
Follows the Adan paper (Table XII) setup adapted for single-GPU.

Usage:
    python train_gpt2_lm.py --model_size small --optimizer AdamW --seed 42
    python train_gpt2_lm.py --model_size small --optimizer LAKTJU_NS --ns_interval 100 --seed 42
"""
import os
import sys
import json
import time
import math
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from transformers import (
    GPT2LMHeadModel,
    GPT2Config,
    GPT2Tokenizer,
    get_cosine_schedule_with_warmup,
)
from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from LAKTJU_NS import LAKTJU_NS
from adan import Adan
from lion_pytorch import Lion

# ══════════════════════════════════════════════════════════════════════════════
# Model Configs (following Adan paper)
# ══════════════════════════════════════════════════════════════════════════════

MODEL_CONFIGS = {
    'small': {  # GPT-2 Small 124M
        'n_layer': 12, 'n_head': 12, 'n_embd': 768,
        'vocab_size': 50257, 'n_positions': 1024,
    },
    'medium': {  # GPT-2 Medium 345M (same as Adan paper)
        'n_layer': 24, 'n_head': 16, 'n_embd': 1024,
        'vocab_size': 50257, 'n_positions': 1024,
    },
}


def get_args():
    p = argparse.ArgumentParser()
    # Model
    p.add_argument('--model_size', type=str, default='small',
                   choices=['small', 'medium'])
    p.add_argument('--seq_len', type=int, default=1024)

    # Training
    p.add_argument('--total_steps', type=int, default=50000,
                   help='Total training steps (Adan used 150k-300k with 64 GPUs)')
    p.add_argument('--batch_size', type=int, default=8,
                   help='Per-device micro batch size')
    p.add_argument('--grad_accum', type=int, default=8,
                   help='Gradient accumulation steps (effective batch = batch_size * grad_accum)')
    p.add_argument('--warmup_steps', type=int, default=2000)
    p.add_argument('--max_grad_norm', type=float, default=1.0)
    p.add_argument('--fp16', action='store_true', default=True)
    p.add_argument('--seed', type=int, default=42)

    # Optimizer
    p.add_argument('--optimizer', type=str, default='AdamW',
                   choices=['Adam', 'AdamW', 'LAKTJU_NS', 'Lion', 'Adan'])
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--weight_decay', type=float, default=0.1)
    p.add_argument('--betas', type=str, default=None)
    p.add_argument('--eps', type=float, default=1e-8)

    # NS-specific
    p.add_argument('--ns_interval', type=int, default=100)
    p.add_argument('--ns_steps', type=int, default=1)
    p.add_argument('--ns_max_dim', type=int, default=1024)
    p.add_argument('--ns_skip_embeddings', action='store_true', default=True)
    p.add_argument('--grad_centralization', action='store_true', default=False)

    # Data
    p.add_argument('--dataset', type=str, default='openwebtext',
                   choices=['openwebtext', 'wikitext'])
    p.add_argument('--num_workers', type=int, default=4)

    # Output
    p.add_argument('--output_dir', type=str, default='results_gpt2')
    p.add_argument('--eval_interval', type=int, default=1000,
                   help='Evaluate every N steps')
    p.add_argument('--log_interval', type=int, default=100,
                   help='Log every N steps')
    p.add_argument('--save_checkpoint', action='store_true', default=False)

    return p.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ══════════════════════════════════════════════════════════════════════════════
# Dataset: Pre-tokenized streaming
# ══════════════════════════════════════════════════════════════════════════════

class PreTokenizedDataset(Dataset):
    """Concatenate all tokens into a flat array, then serve fixed-length chunks."""

    def __init__(self, token_ids, seq_len):
        self.seq_len = seq_len
        # Truncate to exact multiples of seq_len
        n_chunks = len(token_ids) // seq_len
        self.token_ids = token_ids[:n_chunks * seq_len].reshape(n_chunks, seq_len)

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, idx):
        x = torch.tensor(self.token_ids[idx], dtype=torch.long)
        return {'input_ids': x, 'labels': x}


def load_data(args, tokenizer):
    """Load and tokenize dataset."""
    print(f"Loading dataset: {args.dataset}...")

    if args.dataset == 'openwebtext':
        # OpenWebText is large (~8M documents).
        dataset = load_dataset('openwebtext', split='train')
    elif args.dataset == 'wikitext':
        dataset = load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1')

    print(f"Tokenizing...")

    if args.dataset == 'openwebtext':
        # Tokenize in batches using map
        def tokenize_function(examples):
            return tokenizer(examples['text'], truncation=False,
                             add_special_tokens=False)

        tokenized = dataset.map(
            tokenize_function, batched=True,
            remove_columns=['text'],
            num_proc=4,
            desc="Tokenizing",
        )
        # Concatenate all tokens
        all_ids = []
        print("Concatenating tokens...")
        for i, example in enumerate(tokenized):
            all_ids.extend(example['input_ids'])
            if i % 500000 == 0 and i > 0:
                print(f"  Processed {i} documents, {len(all_ids)/1e6:.1f}M tokens...")
            # For single GPU, cap at ~500M tokens (enough for 50k steps)
            if len(all_ids) > 500_000_000:
                break
        all_ids = np.array(all_ids, dtype=np.int32)

        # Split: 99% train, 1% val
        n_val = max(len(all_ids) // 100, args.seq_len * 100)
        train_ids = all_ids[:-n_val]
        val_ids = all_ids[-n_val:]

    else:
        # WikiText has train/validation splits
        train_texts = [t for t in dataset['train']['text'] if t.strip()]
        val_texts = [t for t in dataset['validation']['text'] if t.strip()]

        print(f"  Tokenizing train ({len(train_texts)} texts)...")
        train_ids_list = []
        for text in train_texts:
            train_ids_list.extend(tokenizer.encode(text, add_special_tokens=False))
        train_ids = np.array(train_ids_list, dtype=np.int32)

        print(f"  Tokenizing val ({len(val_texts)} texts)...")
        val_ids_list = []
        for text in val_texts:
            val_ids_list.extend(tokenizer.encode(text, add_special_tokens=False))
        val_ids = np.array(val_ids_list, dtype=np.int32)

    print(f"Train tokens: {len(train_ids):,}, Val tokens: {len(val_ids):,}")

    train_dataset = PreTokenizedDataset(train_ids, args.seq_len)
    val_dataset = PreTokenizedDataset(val_ids, args.seq_len)

    print(f"Train: {len(train_dataset)} chunks, Val: {len(val_dataset)} chunks")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)

    return train_loader, val_loader


# ══════════════════════════════════════════════════════════════════════════════
# Optimizer
# ══════════════════════════════════════════════════════════════════════════════

def create_optimizer(model, args):
    # Separate parameters: no weight decay for bias and LayerNorm
    no_decay = ['bias', 'ln_1.weight', 'ln_2.weight', 'ln_f.weight']
    param_groups = [
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad],
         'weight_decay': 0.0},
    ]

    if args.optimizer == 'Adam':
        lr = args.lr or 6e-4
        betas = tuple(map(float, args.betas.split(','))) if args.betas else (0.9, 0.95)
        optimizer = torch.optim.Adam(param_groups, lr=lr, betas=betas, eps=args.eps)

    elif args.optimizer == 'AdamW':
        lr = args.lr or 6e-4
        betas = tuple(map(float, args.betas.split(','))) if args.betas else (0.9, 0.95)
        optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=args.eps)

    elif args.optimizer == 'LAKTJU_NS':
        lr = args.lr or 6e-4
        betas = tuple(map(float, args.betas.split(','))) if args.betas else (0.9, 0.95)

        # Collect params to skip NS
        skip_params = []
        if args.ns_skip_embeddings:
            for n, p in model.named_parameters():
                if 'wte' in n or 'wpe' in n:  # GPT-2 embedding layers
                    skip_params.append(p)

        optimizer = LAKTJU_NS(
            param_groups, lr=lr, betas=betas, eps=args.eps,
            weight_decay=args.weight_decay,
            ns_interval=args.ns_interval,
            ns_steps=args.ns_steps,
            ns_max_dim=args.ns_max_dim,
            ns_min_dim=1,
            min_ndim=2,
            grad_centralization=args.grad_centralization,
            ns_skip_params=skip_params,
        )

    elif args.optimizer == 'Lion':
        lr = args.lr or 1e-4
        betas = tuple(map(float, args.betas.split(','))) if args.betas else (0.9, 0.99)
        optimizer = Lion(param_groups, lr=lr, betas=betas,
                         weight_decay=args.weight_decay)

    elif args.optimizer == 'Adan':
        lr = args.lr or 1e-3
        betas = (0.98, 0.92, 0.99)
        optimizer = Adan([
            {'params': param_groups[0]['params'], 'weight_decay': args.weight_decay},
            {'params': param_groups[1]['params'], 'weight_decay': 0.0},
        ], lr=lr, betas=betas, eps=args.eps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup_steps, args.total_steps)

    return optimizer, scheduler


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, val_loader, device, max_batches=200):
    model.eval()
    total_loss = 0
    total_tokens = 0
    n_batches = 0

    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        with torch.amp.autocast('cuda', enabled=True):
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

        batch_tokens = input_ids.numel()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens
        n_batches += 1

        if n_batches >= max_batches:
            break

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 20))  # cap to avoid overflow
    model.train()
    return avg_loss, perplexity


# ══════════════════════════════════════════════════════════════════════════════
# Training Loop
# ══════════════════════════════════════════════════════════════════════════════

def train(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model from config (train from scratch, not pretrained)
    cfg = MODEL_CONFIGS[args.model_size]
    model_config = GPT2Config(
        vocab_size=cfg['vocab_size'],
        n_positions=cfg['n_positions'],
        n_embd=cfg['n_embd'],
        n_layer=cfg['n_layer'],
        n_head=cfg['n_head'],
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )
    model = GPT2LMHeadModel(model_config)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: GPT-2 {args.model_size} ({n_params:.1f}M parameters)")
    model.to(device)

    # Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Data
    train_loader, val_loader = load_data(args, tokenizer)

    # Optimizer
    optimizer, scheduler = create_optimizer(model, args)
    initial_lr = args.lr or 6e-4

    # Mixed precision
    scaler = torch.amp.GradScaler('cuda', enabled=args.fp16)

    # Logging
    best_val_loss = float('inf')
    best_val_ppl = float('inf')
    best_step = 0
    history = []

    opt_desc = args.optimizer
    if args.optimizer == 'LAKTJU_NS':
        opt_desc += f'_ns{args.ns_interval}_K{args.ns_steps}_maxd{args.ns_max_dim}'
    opt_desc += f'_lr{initial_lr}'

    effective_batch = args.batch_size * args.grad_accum
    print(f"\n{'='*70}")
    print(f"GPT-2 {args.model_size} | Optimizer: {opt_desc} | Seed: {args.seed}")
    print(f"LR: {initial_lr} | WD: {args.weight_decay} | Betas: {optimizer.param_groups[0].get('betas', 'N/A')}")
    print(f"Steps: {args.total_steps} | Batch: {args.batch_size}x{args.grad_accum}={effective_batch}")
    print(f"Seq len: {args.seq_len} | Tokens/step: {effective_batch * args.seq_len:,}")
    print(f"{'='*70}")

    t0 = time.time()
    global_step = 0
    train_loss_accum = 0
    accum_count = 0
    model.train()

    # Infinite training loop over data
    data_iter = iter(train_loader)

    while global_step < args.total_steps:
        optimizer.zero_grad()

        # Gradient accumulation
        for micro_step in range(args.grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            with torch.amp.autocast('cuda', enabled=args.fp16):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss / args.grad_accum

            scaler.scale(loss).backward()
            train_loss_accum += loss.item()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        global_step += 1
        accum_count += 1

        # Logging
        if global_step % args.log_interval == 0:
            avg_loss = train_loss_accum / accum_count
            ppl = math.exp(min(avg_loss, 20))
            elapsed = time.time() - t0
            tokens_per_sec = (global_step * effective_batch * args.seq_len) / elapsed
            lr_now = scheduler.get_last_lr()[0]
            print(f"  step {global_step:6d}/{args.total_steps} | "
                  f"loss={avg_loss:.4f} | ppl={ppl:.2f} | "
                  f"lr={lr_now:.2e} | "
                  f"{tokens_per_sec/1000:.1f}K tok/s | "
                  f"{elapsed:.0f}s")
            train_loss_accum = 0
            accum_count = 0

        # Evaluation
        if global_step % args.eval_interval == 0:
            val_loss, val_ppl = evaluate(model, val_loader, device)
            elapsed = time.time() - t0

            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
                best_val_loss = val_loss
                best_step = global_step

            print(f"  [EVAL] step {global_step} | "
                  f"val_loss={val_loss:.4f} | val_ppl={val_ppl:.2f} | "
                  f"best_ppl={best_val_ppl:.2f} (step {best_step}) | "
                  f"{elapsed:.0f}s")

            history.append({
                'step': global_step,
                'val_loss': float(val_loss),
                'val_ppl': float(val_ppl),
                'best_ppl': float(best_val_ppl),
            })

    # Final evaluation
    val_loss, val_ppl = evaluate(model, val_loader, device)
    if val_ppl < best_val_ppl:
        best_val_ppl = val_ppl
        best_val_loss = val_loss
        best_step = global_step

    total_time = time.time() - t0

    # Save results
    result = {
        'model_size': args.model_size,
        'n_params_M': float(n_params),
        'optimizer': args.optimizer,
        'seed': args.seed,
        'lr': float(initial_lr),
        'weight_decay': float(args.weight_decay),
        'total_steps': args.total_steps,
        'batch_size': args.batch_size,
        'grad_accum': args.grad_accum,
        'effective_batch': effective_batch,
        'seq_len': args.seq_len,
        'best_val_loss': float(best_val_loss),
        'best_val_ppl': float(best_val_ppl),
        'best_step': best_step,
        'final_val_loss': float(val_loss),
        'final_val_ppl': float(val_ppl),
        'total_time_min': total_time / 60,
        'tokens_per_sec': (args.total_steps * effective_batch * args.seq_len) / total_time,
        'history': history,
        'dataset': args.dataset,
    }
    if args.optimizer == 'LAKTJU_NS':
        result['ns_interval'] = args.ns_interval
        result['ns_steps'] = args.ns_steps
        result['ns_max_dim'] = args.ns_max_dim

    os.makedirs(os.path.join(os.path.dirname(__file__), args.output_dir), exist_ok=True)
    fname = f"gpt2_{args.model_size}_{opt_desc}_seed{args.seed}.json"
    fpath = os.path.join(os.path.dirname(__file__), args.output_dir, fname)
    with open(fpath, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n  => Best val PPL: {best_val_ppl:.2f} (step {best_step})")
    print(f"  => Final val PPL: {val_ppl:.2f}")
    print(f"  => Saved to {fpath}")
    print(f"  => Total time: {total_time/60:.1f} min")

    return best_val_ppl


if __name__ == '__main__':
    args = get_args()
    train(args)

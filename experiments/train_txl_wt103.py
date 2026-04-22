"""
Transformer-XL on WikiText-103.
Follows Adan paper (Table 3/11) base config: n_layer=16, d_model=410, n_head=10.
Compares LAKTJU_NS vs Adam vs AdamW.

Usage:
    python train_txl_wt103.py --optimizer adam --lr 2.5e-4 --wd 0 --max_step 100000
    python train_txl_wt103.py --optimizer LAKTJU_NS --lr 2.5e-4 --ns_interval 100 --ns_steps 2
"""
import argparse, itertools, math, os, sys, time, json, datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils import get_lm_corpus
from mem_transformer import MemTransformerLM
from utils.exp_utils import create_exp_dir
from LAKTJU_NS import LAKTJU_NS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data',    default='./data/wikitext-103')
    p.add_argument('--dataset', default='wt103')
    p.add_argument('--work_dir', default='./results_txl')
    p.add_argument('--seed',    type=int, default=1111)
    # Model (Adan paper base config)
    p.add_argument('--n_layer', type=int, default=16)
    p.add_argument('--n_head',  type=int, default=10)
    p.add_argument('--d_head',  type=int, default=41)
    p.add_argument('--d_model', type=int, default=410)
    p.add_argument('--d_inner', type=int, default=2100)
    p.add_argument('--d_embed', type=int, default=-1)
    p.add_argument('--dropout',    type=float, default=0.1)
    p.add_argument('--dropatt',    type=float, default=0.0)
    p.add_argument('--adaptive',   action='store_true', default=True)
    p.add_argument('--div_val',    type=int, default=4)
    p.add_argument('--pre_lnorm',  action='store_true')
    p.add_argument('--tgt_len',    type=int, default=150)
    p.add_argument('--mem_len',    type=int, default=150)
    p.add_argument('--eval_tgt_len', type=int, default=150)
    p.add_argument('--ext_len',    type=int, default=0)
    p.add_argument('--clamp_len',  type=int, default=-1)
    p.add_argument('--same_length', action='store_true')
    p.add_argument('--attn_type',  type=int, default=0)
    # Training
    p.add_argument('--optimizer',  default='LAKTJU_NS',
                   choices=['adam', 'adamw', 'LAKTJU_NS'])
    p.add_argument('--lr',         type=float, default=None)
    p.add_argument('--wd',         type=float, default=0.0)
    p.add_argument('--clip',       type=float, default=0.25)
    p.add_argument('--max_step',   type=int, default=200000)
    p.add_argument('--batch_size', type=int, default=22)
    p.add_argument('--warmup_step', type=int, default=0)
    p.add_argument('--eta_min',    type=float, default=0.0)
    p.add_argument('--log_interval',  type=int, default=200)
    p.add_argument('--eval_interval', type=int, default=4000)
    # LAKTJU_NS
    p.add_argument('--ns_interval', type=int, default=100)
    p.add_argument('--ns_steps',    type=int, default=2)
    p.add_argument('--ns_max_dim',  type=int, default=1024)
    p.add_argument('--ns_backbone', type=str, default='adam',
                   choices=['adam', 'adamw'],
                   help='Backbone optimizer for LAKTJU_NS')
    return p.parse_args()


def build_optimizer(args, model):
    params = list(model.parameters())
    opt = args.optimizer.lower()

    if opt == 'adam':
        lr = args.lr or 0.00025
        optimizer = optim.Adam(params, lr=lr, weight_decay=args.wd)
    elif opt == 'adamw':
        lr = args.lr or 0.00025
        optimizer = optim.AdamW(params, lr=lr, weight_decay=args.wd)
    elif opt == 'laktju_ns':
        lr = args.lr or 0.00025
        # Collect embedding params to skip NS
        skip_params = []
        for n, p in model.named_parameters():
            if 'emb' in n.lower() or 'bias' in n.lower():
                skip_params.append(p)

        optimizer = LAKTJU_NS(params, lr=lr, betas=(0.9, 0.999),
                              eps=1e-8, weight_decay=args.wd,
                              ns_interval=args.ns_interval,
                              ns_steps=args.ns_steps,
                              min_ndim=2,
                              ns_max_dim=args.ns_max_dim,
                              ns_skip_params=skip_params)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    # Cosine annealing with optional warmup
    if args.warmup_step > 0:
        def lr_lambda(step):
            if step < args.warmup_step:
                return float(step) / float(max(1, args.warmup_step))
            progress = float(step - args.warmup_step) / float(max(1, args.max_step - args.warmup_step))
            return max(args.eta_min / lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.max_step, eta_min=args.eta_min)

    return optimizer, scheduler, lr


def evaluate(model, eval_iter, device):
    model.eval()
    total_loss, total_len = 0., 0
    with torch.no_grad():
        mems = tuple()
        for batch in eval_iter:
            data, target, seq_len = batch
            data, target = data.to(device), target.to(device)
            ret = model(data, target, *mems)
            loss, mems = ret[0], ret[1:]
            total_loss += seq_len * loss.mean().item()
            total_len  += seq_len
    model.train()
    return total_loss / total_len


def main():
    args = parse_args()
    if args.d_embed < 0:
        args.d_embed = args.d_model

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Data
    corpus = get_lm_corpus(args.data, args.dataset)
    ntokens = len(corpus.vocab)
    tr_iter = corpus.get_iterator('train', args.batch_size, args.tgt_len,
                                  device=device, ext_len=args.ext_len)
    va_iter = corpus.get_iterator('valid', 10, args.eval_tgt_len,
                                  device=device, ext_len=args.ext_len)
    te_iter = corpus.get_iterator('test',  10, args.eval_tgt_len,
                                  device=device, ext_len=args.ext_len)

    # Model
    cutoffs = [20000, 40000, 200000] if args.adaptive else []
    tie_projs = [False] + ([True] * len(cutoffs) if args.adaptive else [])
    model = MemTransformerLM(
        ntokens, args.n_layer, args.n_head, args.d_model,
        args.d_head, args.d_inner, args.dropout, args.dropatt,
        tie_weight=True, d_embed=args.d_embed, div_val=args.div_val,
        tie_projs=tie_projs, pre_lnorm=args.pre_lnorm,
        tgt_len=args.tgt_len, ext_len=args.ext_len, mem_len=args.mem_len,
        cutoffs=cutoffs, same_length=args.same_length,
        attn_type=args.attn_type, clamp_len=args.clamp_len,
    ).to(device)

    # Proper initialization (matches original TXL codebase)
    init_std = 0.02
    emb_init_range = 0.01
    proj_init_std = 0.01
    for name, p in model.named_parameters():
        if 'emb_layers' in name:
            nn.init.uniform_(p, -emb_init_range, emb_init_range)
        elif 'emb_projs' in name or 'out_projs' in name:
            nn.init.normal_(p, 0.0, proj_init_std)
        elif 'cluster_weight' in name or 'cluster_bias' in name:
            pass  # keep default init
        elif p.dim() > 1:
            nn.init.normal_(p, 0.0, init_std)
        elif 'bias' in name:
            nn.init.constant_(p, 0.0)

    n_params = sum(p.numel() for p in model.parameters())

    optimizer, scheduler, lr = build_optimizer(args, model)

    # Describe optimizer
    opt_desc = args.optimizer
    if args.optimizer == 'LAKTJU_NS':
        opt_desc += f'_ns{args.ns_interval}_K{args.ns_steps}'

    os.makedirs(args.work_dir, exist_ok=True)

    print(f"{'='*70}")
    print(f"Transformer-XL base | {n_params/1e6:.1f}M params | vocab={ntokens}")
    print(f"Optimizer: {opt_desc} | LR: {lr} | WD: {args.wd}")
    print(f"Steps: {args.max_step} | Batch: {args.batch_size} | tgt_len: {args.tgt_len} | mem_len: {args.mem_len}")
    print(f"Warmup: {args.warmup_step} | Seed: {args.seed}")
    print(f"{'='*70}")

    best_val_loss = float('inf')
    best_val_ppl  = float('inf')
    best_test_ppl = float('inf')
    best_step = 0
    train_step = 0
    log_loss = 0.
    log_start = time.time()
    t0 = time.time()
    history = []

    model.train()
    mems = tuple()

    for epoch in itertools.count(start=1):
        # Detach mems at epoch boundary to prevent stale gradients
        if mems:
            mems = tuple(m.detach() for m in mems)
        for batch in tr_iter:
            data, target, seq_len = batch
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            ret = model(data, target, *mems)
            loss, mems = ret[0], ret[1:]
            loss = loss.mean()
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            # Skip update if gradients are NaN
            if math.isnan(gnorm.item()) or math.isinf(gnorm.item()):
                print(f"  [WARN] NaN/Inf grad norm at step {train_step+1}, skipping",
                      flush=True)
                mems = tuple()
                train_step += 1
                scheduler.step()
                continue

            optimizer.step()
            scheduler.step()

            cur_loss_val = loss.item()
            if math.isnan(cur_loss_val) or math.isinf(cur_loss_val):
                print(f"  [WARN] NaN/Inf loss at step {train_step+1}, skipping update",
                      flush=True)
                # Reset mems to prevent NaN propagation
                mems = tuple()
                train_step += 1
                continue

            log_loss += cur_loss_val
            train_step += 1

            if train_step % args.log_interval == 0:
                cur_loss = log_loss / args.log_interval
                elapsed = time.time() - t0
                cur_lr = optimizer.param_groups[0]['lr']
                tokens_per_sec = (train_step * args.batch_size * args.tgt_len) / elapsed
                print(f"  step {train_step:>8}/{args.max_step} | "
                      f"loss={cur_loss:.4f} | ppl={math.exp(min(cur_loss, 20)):.2f} | "
                      f"lr={cur_lr:.2e} | "
                      f"{tokens_per_sec/1000:.1f}K tok/s | "
                      f"{elapsed:.0f}s",
                      flush=True)
                log_loss = 0.

            if train_step % args.eval_interval == 0:
                val_loss = evaluate(model, va_iter, device)
                val_ppl = math.exp(val_loss)
                elapsed = time.time() - t0

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_ppl = val_ppl
                    best_step = train_step
                    # Evaluate on test set at best val
                    test_loss = evaluate(model, te_iter, device)
                    best_test_ppl = math.exp(test_loss)
                    # Save best model
                    torch.save(model.state_dict(),
                               os.path.join(args.work_dir,
                                            f'txl_{args.optimizer}_seed{args.seed}_best.pt'))

                print(f"  [EVAL] step {train_step} | "
                      f"val_ppl={val_ppl:.2f} | test_ppl={best_test_ppl:.2f} | "
                      f"best_val={best_val_ppl:.2f} (step {best_step}) | "
                      f"{elapsed:.0f}s",
                      flush=True)

                history.append({
                    'step': train_step,
                    'val_loss': float(val_loss),
                    'val_ppl': float(val_ppl),
                    'best_val_ppl': float(best_val_ppl),
                    'best_test_ppl': float(best_test_ppl),
                })

            if train_step >= args.max_step:
                break

        if train_step >= args.max_step:
            break

    # Final evaluation
    val_loss = evaluate(model, va_iter, device)
    val_ppl = math.exp(val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_ppl = val_ppl
        best_step = train_step
        test_loss = evaluate(model, te_iter, device)
        best_test_ppl = math.exp(test_loss)

    total_time = time.time() - t0

    print(f"\n  => Best Val PPL:  {best_val_ppl:.2f} (step {best_step})")
    print(f"  => Best Test PPL: {best_test_ppl:.2f}")
    print(f"  => Total time: {total_time/60:.1f} min")

    result = {
        'model': 'txl-base',
        'n_params_M': float(n_params / 1e6),
        'optimizer': args.optimizer,
        'lr': float(lr),
        'wd': float(args.wd),
        'seed': args.seed,
        'max_step': args.max_step,
        'batch_size': args.batch_size,
        'warmup_step': args.warmup_step,
        'best_val_ppl': float(best_val_ppl),
        'best_test_ppl': float(best_test_ppl),
        'best_step': best_step,
        'final_val_ppl': float(val_ppl),
        'total_time_min': total_time / 60,
        'history': history,
    }
    if args.optimizer == 'LAKTJU_NS':
        result['ns_interval'] = args.ns_interval
        result['ns_steps'] = args.ns_steps
        result['ns_max_dim'] = args.ns_max_dim

    fname = os.path.join(args.work_dir,
                         f"txl_{opt_desc}_lr{lr}_wd{args.wd}_seed{args.seed}.json")
    with open(fname, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  => Saved: {fname}")


if __name__ == '__main__':
    main()

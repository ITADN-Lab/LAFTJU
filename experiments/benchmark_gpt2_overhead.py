"""Benchmark optimizer overhead on GPT-2 Small: ms/step, tokens/sec, peak memory."""
import sys, os, time, torch, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformers import GPT2Config, GPT2LMHeadModel
from LAKTJU_NS import LAKTJU_NS
from adan import Adan
from lion_pytorch import Lion

DEVICE = 'cuda'
SEQ_LEN = 1024
BATCH_SIZE = 8
N_WARMUP = 5
N_MEASURE = 50

def make_model():
    cfg = GPT2Config(vocab_size=50257, n_positions=1024, n_embd=768,
                     n_layer=12, n_head=12, resid_pdrop=0.1,
                     embd_pdrop=0.1, attn_pdrop=0.1)
    return GPT2LMHeadModel(cfg)

def make_optimizer(name, model):
    params = list(model.parameters())
    if name == 'AdamW':
        return torch.optim.AdamW(params, lr=1e-3, weight_decay=0.1)
    elif name == 'Lion':
        return Lion(params, lr=1e-4, weight_decay=1.0)
    elif name == 'Adan':
        return Adan(params, lr=1e-3, betas=(0.98, 0.92, 0.99), weight_decay=0.02)
    elif name == 'LAFTJU_NS':
        skip = [p for n, p in model.named_parameters() if 'wte' in n or 'wpe' in n]
        return LAKTJU_NS(params, lr=1e-3, weight_decay=0.1,
                         ns_interval=100, ns_steps=2, ns_max_dim=1024,
                         min_ndim=2, ns_skip_params=skip)
    raise ValueError(name)

def benchmark(name):
    torch.manual_seed(42)
    model = make_model().to(DEVICE)
    opt = make_optimizer(name, model)
    scaler = torch.amp.GradScaler(DEVICE)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    for i in range(N_WARMUP + N_MEASURE):
        x = torch.randint(0, 50257, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
        with torch.amp.autocast(DEVICE):
            loss = model(x, labels=x).loss
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()

    torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(N_MEASURE):
        x = torch.randint(0, 50257, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.amp.autocast(DEVICE):
            loss = model(x, labels=x).loss
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    ms = np.mean(times)
    tokens_per_sec = BATCH_SIZE * SEQ_LEN / (ms / 1000)
    mem_gb = torch.cuda.max_memory_allocated() / 1024**3

    # Optimizer state memory
    opt_state_mb = 0
    for state in opt.state.values():
        for v in state.values():
            if torch.is_tensor(v):
                opt_state_mb += v.numel() * v.element_size()
    opt_state_mb /= 1024**2

    del model, opt, scaler
    torch.cuda.empty_cache()
    return ms, tokens_per_sec, mem_gb, opt_state_mb

def main():
    optimizers = ['AdamW', 'Lion', 'Adan', 'LAFTJU_NS']
    results = {}

    for name in optimizers:
        print(f'Benchmarking {name}...', flush=True)
        ms, tps, mem, opt_mem = benchmark(name)
        results[name] = (ms, tps, mem, opt_mem)
        print(f'  {ms:.1f} ms/step  {tps/1000:.1f}K tok/s  {mem:.2f} GB  opt_state={opt_mem:.0f} MB')

    print(f'\n{"="*75}')
    print(f'GPT-2 Small (124M) Overhead Benchmark (bs={BATCH_SIZE}, seq={SEQ_LEN}, fp16)')
    print(f'{"="*75}')
    base_ms = results['AdamW'][0]
    print(f'{"Optimizer":<12} {"ms/step":>10} {"K tok/s":>10} {"Mem(GB)":>10} {"OptState(MB)":>14} {"Overhead":>10}')
    for name in optimizers:
        ms, tps, mem, opt_mem = results[name]
        overhead = f'{ms/base_ms:.2f}x'
        print(f'{name:<12} {ms:>10.1f} {tps/1000:>10.1f} {mem:>10.2f} {opt_mem:>14.0f} {overhead:>10}')

    import json
    with open(os.path.join(os.path.dirname(__file__), 'results_gpt2', 'overhead_gpt2_small.json'), 'w') as f:
        json.dump({name: {'ms_per_step': results[name][0], 'tokens_per_sec': results[name][1],
                          'peak_mem_gb': results[name][2], 'opt_state_mb': results[name][3]}
                   for name in optimizers}, f, indent=2)
    print('\nSaved to results_gpt2/overhead_gpt2_small.json')

if __name__ == '__main__':
    main()

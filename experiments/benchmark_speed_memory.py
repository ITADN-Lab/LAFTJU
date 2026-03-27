"""Benchmark: speed (ms/step, s/epoch) and GPU memory (MB) for each optimizer."""
import sys, os, time, torch, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision, torchvision.transforms as transforms
from LAKTJU import LAKTJU
from LAKTJU_Fast import LAKTJU_Fast
from adan import Adan
from ResNet import ResNet18
from cutout import Cutout

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 128
N_WARMUP = 5   # warmup batches
N_MEASURE = 50  # measured batches

def get_loader():
    tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
        Cutout(n_holes=1, length=16),
    ])
    ds = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True, transform=tf)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

def make_optimizer(name, model):
    if name == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    elif name == 'AdamW':
        return torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    elif name == 'Adan':
        return Adan(model.parameters(), lr=1e-2, weight_decay=5e-4)
    elif name == 'LAFTJU_Fast':
        return LAKTJU_Fast(model.parameters(), tju_lr=3e-3, a_lr=1e-3, weight_decay=2e-3,
                      homotopy_speed=8.0, warmup=100)
    elif name == 'LAFTJU':
        return LAKTJU(model.parameters(), tju_lr=3e-3, a_lr=1e-3, weight_decay=2e-3,
                      homotopy_speed=8.0, warmup=100)
    raise ValueError(name)

def benchmark(name, loader):
    torch.manual_seed(42)
    model = ResNet18(num_classes=10).to(DEVICE)
    opt = make_optimizer(name, model)
    # Register KF hooks for LAKTJU-based optimizers
    if hasattr(opt, 'register_hooks'):
        opt.register_hooks(model)
    criterion = nn.CrossEntropyLoss()

    if DEVICE == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    times = []
    for i, (x, y) in enumerate(loader):
        if i >= N_WARMUP + N_MEASURE:
            break
        x, y = x.to(DEVICE), y.to(DEVICE)

        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()

        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        if i >= N_WARMUP:
            times.append((t1 - t0) * 1000)  # ms

    ms_per_step = np.mean(times)
    # Estimate full epoch: 391 steps (50000/128)
    sec_per_epoch = ms_per_step * 391 / 1000

    if DEVICE == 'cuda':
        mem_mb = torch.cuda.max_memory_allocated() / 1024**2
    else:
        # CPU: use process RSS
        import resource
        mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    return ms_per_step, sec_per_epoch, mem_mb

def main():
    loader = get_loader()
    optimizers = ['Adam', 'AdamW', 'Adan', 'LAFTJU', 'LAFTJU_Fast']
    results = {}
    for name in optimizers:
        print(f'Benchmarking {name}...', flush=True)
        ms, sec, mem = benchmark(name, loader)
        results[name] = (ms, sec, mem)
        print(f'  {ms:.2f} ms/step  {sec:.1f} s/epoch  {mem:.0f} MB')

    print('\n--- Summary ---')
    print(f'{"Optimizer":<10} {"ms/step":>10} {"s/epoch":>10} {"Mem(MB)":>10} {"Overhead":>12}')
    adam_ms = results['Adam'][0]
    for name in optimizers:
        ms, sec, mem = results[name]
        overhead = f'+{(ms/adam_ms - 1)*100:.0f}%' if name != 'Adam' else 'baseline'
        print(f'{name:<10} {ms:>10.2f} {sec:>10.1f} {mem:>10.0f} {overhead:>12}')

if __name__ == '__main__':
    main()

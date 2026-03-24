"""
LAKTJU vs Baselines Training Script
Supports: CIFAR-10, CIFAR-100 with ResNet18/50
Compares: SGD, Adam, AdamW, ATJU(V5), LAKTJU(V6)
"""
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import math
import os
import sys
import argparse
import json

# Add parent dir to path for optimizer imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from LAKTJU import LAKTJU
from LAKTJU_V9 import LAKTJU_V9
from LAKTJU_V10 import LAKTJU_V10
from LAKTJU_V11 import LAKTJU_V11
from LAKTJU_V12 import LAKTJU_V12
from ATJU import ATJU
from adan import Adan
from ResNet import ResNet18, ResNet50
from readData import read_dataset
from cutout import Cutout
from CosineAnnealingLR import MyCosineAnnealingLR

import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler


def parse_args():
    parser = argparse.ArgumentParser(description='LAKTJU Training')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet50'])
    parser.add_argument('--optimizer', type=str, default='LAKTJU',
                        choices=['SGD', 'Adam', 'AdamW', 'ATJU', 'LAKTJU', 'LAKTJU_V9', 'LAKTJU_V10', 'LAKTJU_V11', 'LAKTJU_V12', 'Adan'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=None, help='Override default LR')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--data_dir', type=str, default='./dataset')
    parser.add_argument('--workers', type=int, default=4)
    # LAKTJU-specific
    parser.add_argument('--a_lr_ratio', type=float, default=None, help='a_lr = tju_lr * ratio (default: 1/10)')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--c_base', type=float, default=1.0)
    parser.add_argument('--kappa', type=float, default=5.0)
    parser.add_argument('--homotopy_sharpness', type=float, default=10.0)
    parser.add_argument('--homotopy_speed', type=float, default=2.0)
    parser.add_argument('--kf_damping', type=float, default=1e-3)
    parser.add_argument('--kf_update_interval', type=int, default=20)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--grad_clip', type=float, default=0.0, help='Gradient clipping max norm (0=disabled)')
    parser.add_argument('--sam_rho', type=float, default=0.0, help='SAM perturbation radius (0=disabled)')
    # LAKTJU_V9-specific
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum for V9 main path')
    parser.add_argument('--alpha_kf', type=float, default=0.15, help='KF correction ratio for main path')
    parser.add_argument('--alpha_adam_kf', type=float, default=0.05, help='KF correction ratio for AdamW path')
    parser.add_argument('--s_max', type=float, default=0.7, help='Homotopy upper bound (V9)')
    # LAKTJU_V11-specific
    parser.add_argument('--kf_warmup', type=int, default=500, help='Steps before KF activates (V11)')
    parser.add_argument('--cos_sim_threshold', type=float, default=0.0, help='Cosine similarity gate threshold (V11)')
    # LAKTJU_V12-specific
    parser.add_argument('--kf_clip_max', type=float, default=50.0, help='Adaptive KF clip ratio (V12)')
    parser.add_argument('--grad_centralization', action='store_true', default=True, help='Enable gradient centralization (V12)')
    parser.add_argument('--no_grad_centralization', action='store_true', help='Disable gradient centralization (V12)')
    return parser.parse_args()

def load_cifar(dataset, batch_size, data_dir, workers):
    """Load CIFAR-10 or CIFAR-100 with standard augmentation."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010]),
        Cutout(n_holes=1, length=16),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010]),
    ])

    if dataset == 'cifar10':
        ds_cls = torchvision.datasets.CIFAR10
        n_class = 10
    else:
        ds_cls = torchvision.datasets.CIFAR100
        n_class = 100

    train_data = ds_cls(data_dir, train=True, download=True, transform=transform_train)
    test_data = ds_cls(data_dir, train=False, download=True, transform=transform_test)

    # Split train into train/valid (90/10)
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(0.1 * num_train)
    train_idx, valid_idx = indices[split:], indices[:split]

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size,
        sampler=SubsetRandomSampler(train_idx),
        num_workers=workers, pin_memory=True)

    # Use test transform for validation
    valid_data = ds_cls(data_dir, train=True, download=True, transform=transform_test)
    valid_loader = torch.utils.data.DataLoader(
        valid_data, batch_size=batch_size,
        sampler=SubsetRandomSampler(valid_idx),
        num_workers=workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

    return train_loader, valid_loader, test_loader, n_class


def build_model(model_name, n_class, device):
    """Build ResNet model."""
    if model_name == 'resnet18':
        model = ResNet18(num_classes=n_class)
        # For CIFAR: replace 7x7 conv with 3x3
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    elif model_name == 'resnet50':
        model = ResNet50(num_classes=n_class)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    return model.to(device)


def build_optimizer(args, model, total_steps):
    """Build optimizer based on args."""
    params = model.parameters()
    opt_name = args.optimizer

    if opt_name == 'SGD':
        lr = args.lr if args.lr else 0.1
        optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=args.weight_decay)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    elif opt_name == 'Adam':
        lr = args.lr if args.lr else 0.001
        optimizer = optim.Adam(params, lr=lr, betas=(0.9, 0.999),
                               weight_decay=args.weight_decay, eps=1e-8)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    elif opt_name == 'AdamW':
        lr = args.lr if args.lr else 0.001
        optimizer = optim.AdamW(params, lr=lr, betas=(0.9, 0.999),
                                weight_decay=args.weight_decay, eps=1e-8)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    elif opt_name == 'ATJU':
        lr = args.lr if args.lr else 0.001
        a_lr = lr / 10.0
        optimizer = ATJU(params, tju_lr=lr, a_lr=a_lr,
                         tju_beta1=0.9, tju_beta2=0.999, tju_eps=1e-4,
                         a_beta1=0.9, a_beta2=0.999, a_eps=1e-8,
                         rebound='constant', warmup=20,
                         tju_weight_decay=args.weight_decay,
                         a_weight_decay=args.weight_decay,
                         weight_decay_type='decoupled',
                         A_optim='AdamW',
                         total_epoch=args.epochs, epoch_now=0)
        scheduler = MyCosineAnnealingLR(optimizer, T_max=args.epochs,
                                         tju_lr_min=1e-6, A_lr_min=1e-6)

    elif opt_name == 'LAKTJU':
        tju_lr = args.lr if args.lr else 0.001
        a_lr_ratio = args.a_lr_ratio if args.a_lr_ratio else (1.0 / 10.0)
        a_lr = tju_lr * a_lr_ratio
        optimizer = LAKTJU(params, tju_lr=tju_lr, a_lr=a_lr,
                           beta1=0.9, beta2=0.999, eps=1e-8,
                           weight_decay=args.weight_decay,
                           c_base=args.c_base, kappa=args.kappa,
                           homotopy_sharpness=args.homotopy_sharpness,
                           homotopy_speed=args.homotopy_speed,
                           warmup=args.warmup, total_steps=total_steps,
                           kf_update_interval=args.kf_update_interval,
                           kf_damping=args.kf_damping)
        scheduler = MyCosineAnnealingLR(optimizer, T_max=args.epochs,
                                         tju_lr_min=1e-6, A_lr_min=1e-6)

    elif opt_name == 'LAKTJU_V9':
        tju_lr = args.lr if args.lr else 0.01
        a_lr_ratio = args.a_lr_ratio if args.a_lr_ratio else 0.2
        a_lr = tju_lr * a_lr_ratio
        optimizer = LAKTJU_V9(params, tju_lr=tju_lr, a_lr=a_lr,
                              beta1=0.9, beta2=0.999, eps=1e-8,
                              weight_decay=args.weight_decay,
                              momentum=args.momentum,
                              alpha_kf=args.alpha_kf,
                              alpha_adam_kf=args.alpha_adam_kf,
                              s_max=args.s_max,
                              homotopy_speed=args.homotopy_speed,
                              warmup=args.warmup, total_steps=total_steps,
                              kf_update_interval=args.kf_update_interval,
                              kf_damping=args.kf_damping,
                              grad_clip=args.grad_clip)
        scheduler = MyCosineAnnealingLR(optimizer, T_max=args.epochs,
                                         tju_lr_min=1e-6, A_lr_min=1e-6)

    elif opt_name == 'LAKTJU_V10':
        lr = args.lr if args.lr else 0.001
        optimizer = LAKTJU_V10(params, lr=lr,
                               beta1=0.9, beta2=0.999, eps=1e-8,
                               weight_decay=args.weight_decay,
                               momentum=args.momentum,
                               alpha_kf=args.alpha_kf,
                               alpha_adam_kf=args.alpha_adam_kf,
                               s_max=args.s_max,
                               homotopy_speed=args.homotopy_speed,
                               warmup=args.warmup, total_steps=total_steps,
                               kf_update_interval=args.kf_update_interval,
                               kf_damping=args.kf_damping,
                               grad_clip=args.grad_clip)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    elif opt_name == 'LAKTJU_V11':
        lr = args.lr if args.lr else 0.001
        optimizer = LAKTJU_V11(params, lr=lr,
                               beta1=0.9, beta2=0.999, eps=1e-8,
                               weight_decay=args.weight_decay,
                               alpha_kf=args.alpha_kf,
                               kf_warmup=args.kf_warmup,
                               kf_update_interval=args.kf_update_interval,
                               kf_damping=args.kf_damping,
                               cos_sim_threshold=args.cos_sim_threshold,
                               grad_clip=args.grad_clip,
                               warmup=args.warmup)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    elif opt_name == 'LAKTJU_V12':
        tju_lr = args.lr if args.lr else 0.001
        a_lr_ratio = args.a_lr_ratio if args.a_lr_ratio else (1.0 / 10.0)
        a_lr = tju_lr * a_lr_ratio
        gc = args.grad_centralization and not args.no_grad_centralization
        optimizer = LAKTJU_V12(params, tju_lr=tju_lr, a_lr=a_lr,
                               beta1=0.9, beta2=0.999, eps=1e-8,
                               weight_decay=args.weight_decay,
                               c_base=args.c_base, kappa=args.kappa,
                               homotopy_sharpness=args.homotopy_sharpness,
                               homotopy_speed=args.homotopy_speed,
                               warmup=args.warmup, total_steps=total_steps,
                               kf_update_interval=args.kf_update_interval,
                               kf_damping=args.kf_damping,
                               kf_warmup=args.kf_warmup,
                               kf_clip_max=args.kf_clip_max,
                               grad_centralization=gc)
        scheduler = MyCosineAnnealingLR(optimizer, T_max=args.epochs,
                                         tju_lr_min=1e-6, A_lr_min=1e-6)

    elif opt_name == 'Adan':
        lr = args.lr if args.lr else 0.01
        optimizer = Adan(params, lr=lr, betas=(0.98, 0.92, 0.99),
                         eps=1e-8, weight_decay=args.weight_decay,
                         max_grad_norm=0.0, no_prox=False)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    return optimizer, scheduler

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, args):
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    n_samples = 0
    sam_rho = getattr(args, 'sam_rho', 0.0)

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        if sam_rho > 0:
            # SAM Step 1: compute gradient at current point
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # Compute and apply perturbation e_w = rho * grad / ||grad||
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
            e_ws = []
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        e_w = sam_rho * p.grad / (grad_norm + 1e-12)
                        p.add_(e_w)
                        e_ws.append(e_w)
                    else:
                        e_ws.append(None)

            # SAM Step 2: compute gradient at perturbed point (disable KF to save memory)
            if args.optimizer in ('LAKTJU', 'LAKTJU_V9', 'LAKTJU_V10', 'LAKTJU_V11', 'LAKTJU_V12'):
                optimizer.disable_kf_hooks()
            optimizer.zero_grad()
            output2 = model(data)
            loss2 = criterion(output2, target)
            loss2.backward()
            if args.optimizer in ('LAKTJU', 'LAKTJU_V9', 'LAKTJU_V10', 'LAKTJU_V11', 'LAKTJU_V12'):
                optimizer.enable_kf_hooks()

            # Restore original parameters
            with torch.no_grad():
                for p, e_w in zip(model.parameters(), e_ws):
                    if e_w is not None:
                        p.sub_(e_w)

            # Update ATJU/LAKTJU state
            if args.optimizer == 'ATJU':
                optimizer.epoch_now = epoch
            if args.optimizer == 'LAKTJU':
                optimizer.set_loss(loss.item())
            if args.optimizer == 'LAKTJU_V12':
                optimizer.set_loss(loss.item())

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            total_loss += loss.item() * data.size(0)
        else:
            # Standard training (no SAM)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            if args.optimizer == 'ATJU':
                optimizer.epoch_now = epoch
            if args.optimizer == 'LAKTJU':
                optimizer.set_loss(loss.item())
            if args.optimizer == 'LAKTJU_V12':
                optimizer.set_loss(loss.item())

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            total_loss += loss.item() * data.size(0)

        n_samples += data.size(0)

    return total_loss / n_samples


def evaluate(model, loader, criterion, device):
    """Evaluate model, return (loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            _, pred = torch.max(output, 1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    return total_loss / total, 100.0 * correct / total


def save_results(results, save_path):
    """Save results dict to JSON."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {save_path}")


def main():
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load data
    train_loader, valid_loader, test_loader, n_class = load_cifar(
        args.dataset, args.batch_size, args.data_dir, args.workers)
    print(f"Dataset: {args.dataset} ({n_class} classes)")

    # Build model
    model = build_model(args.model, n_class, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model} ({n_params:,} params)")

    # Build optimizer
    total_steps = args.epochs * len(train_loader)
    optimizer, scheduler = build_optimizer(args, model, total_steps)
    print(f"Optimizer: {args.optimizer}")

    # Register KF hooks for LAKTJU / LAKTJU_V9 / LAKTJU_V10
    if args.optimizer in ('LAKTJU', 'LAKTJU_V9', 'LAKTJU_V10', 'LAKTJU_V11', 'LAKTJU_V12'):
        optimizer.register_hooks(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)

    # Results tracking
    results = {
        'config': vars(args),
        'train_loss': [],
        'valid_loss': [],
        'valid_acc': [],
        'test_acc': [],
        'epoch_time': [],
        'best_valid_acc': 0.0,
        'best_test_acc': 0.0,
    }

    now_str = time.strftime('%Y%m%d_%H%M%S')
    exp_name = f"{args.dataset}_{args.model}_{args.optimizer}_seed{args.seed}"

    print(f"\n{'='*60}")
    print(f"Starting training: {exp_name}")
    print(f"{'='*60}\n")

    best_valid_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, args)

        # Validate
        valid_loss, valid_acc = evaluate(model, valid_loader, criterion, device)

        # Test (every 10 epochs or last epoch)
        if epoch % 10 == 0 or epoch == args.epochs or epoch == args.epochs // 2:
            _, test_acc = evaluate(model, test_loader, criterion, device)
        else:
            test_acc = results['test_acc'][-1] if results['test_acc'] else 0.0

        epoch_time = time.time() - t0

        # Update scheduler
        scheduler.step()

        # Track best
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            _, best_test = evaluate(model, test_loader, criterion, device)
            results['best_valid_acc'] = best_valid_acc
            results['best_test_acc'] = best_test

        # Log
        results['train_loss'].append(train_loss)
        results['valid_loss'].append(valid_loss)
        results['valid_acc'].append(valid_acc)
        results['test_acc'].append(test_acc)
        results['epoch_time'].append(epoch_time)

        # Print progress
        if epoch % 5 == 0 or epoch == 1:
            # Get current LR
            if args.optimizer in ('LAKTJU', 'ATJU', 'LAKTJU_V9', 'LAKTJU_V12'):
                cur_lr = optimizer.param_groups[0].get('tju_lr', 0)
            else:
                cur_lr = optimizer.param_groups[0].get('lr', 0)

            # Get homotopy s for LAKTJU / LAKTJU_V9
            s_str = ""
            if args.optimizer == 'LAKTJU':
                s_val = optimizer._compute_homotopy_s(optimizer._gamma_ema or 1.0)
                s_str = f" s={s_val:.3f}"
            elif args.optimizer == 'LAKTJU_V9':
                progress = optimizer._global_step / max(optimizer.total_steps, 1)
                s_val = optimizer.s_max * math.tanh(progress * optimizer.homotopy_speed)
                s_str = f" s={s_val:.3f}"
                # Print V9 diagnostics
                diag = optimizer.get_diagnostics()
                if diag:
                    n = max(diag.get('total_param_count', 1), 1)
                    print(f"  [V9 diag] s={diag.get('s',0):.3f} "
                          f"main_norm={diag.get('main_update_norm',0)/n:.4f} "
                          f"adam_norm={diag.get('adam_update_norm',0)/n:.4f} "
                          f"kf_active={diag.get('kf_active_count',0)}/{n}", flush=True)
            elif args.optimizer == 'LAKTJU_V10':
                progress = optimizer._global_step / max(optimizer.total_steps, 1)
                s_val = optimizer.s_max * math.tanh(progress * optimizer.homotopy_speed)
                s_str = f" s={s_val:.3f}"
                # Print V10 diagnostics
                diag = optimizer.get_diagnostics()
                if diag:
                    n = max(diag.get('total_param_count', 1), 1)
                    print(f"  [V10 diag] s={diag.get('s',0):.3f} "
                          f"main_norm={diag.get('main_update_norm',0)/n:.4f} "
                          f"adam_norm={diag.get('adam_update_norm',0)/n:.4f} "
                          f"scaled_norm={diag.get('main_scaled_norm',0)/n:.4f} "
                          f"kf_active={diag.get('kf_active_count',0)}/{n}", flush=True)
            elif args.optimizer == 'LAKTJU_V11':
                diag = optimizer.get_diagnostics()
                if diag:
                    n = max(diag.get('total_param_count', 1), 1)
                    print(f"  [V11 diag] "
                          f"adam_norm={diag.get('adam_update_norm',0)/n:.4f} "
                          f"kf_applied={diag.get('kf_applied_count',0)}/{n} "
                          f"cos_sim={diag.get('avg_cos_sim',0):.4f} "
                          f"eff_alpha={diag.get('avg_eff_alpha',0):.4f}", flush=True)
            elif args.optimizer == 'LAKTJU_V12':
                progress = optimizer._global_step / max(optimizer.total_steps, 1)
                s_val = math.tanh(progress * optimizer.homotopy_speed)
                s_str = f" s={s_val:.3f}"
                diag = optimizer.get_diagnostics()
                if diag:
                    n = max(diag.get('total_param_count', 1), 1)
                    print(f"  [V12 diag] s={diag.get('s',0):.3f} "
                          f"kf_str={diag.get('kf_strength',0):.3f} "
                          f"tju_norm={diag.get('tju_update_norm',0)/n:.4f} "
                          f"adam_norm={diag.get('adam_update_norm',0)/n:.4f} "
                          f"kf_active={diag.get('kf_active_count',0)}/{n} "
                          f"gc={diag.get('gc_applied_count',0)}", flush=True)

            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Valid Acc: {valid_acc:.2f}% | "
                  f"Test Acc: {test_acc:.2f}% | "
                  f"LR: {cur_lr:.6f}{s_str} | "
                  f"Time: {epoch_time:.1f}s", flush=True)

        # Save intermediate results every 50 epochs
        if epoch % 50 == 0:
            save_path = os.path.join(args.save_dir, f"{exp_name}_{now_str}.json")
            save_results(results, save_path)

    # Final save
    save_path = os.path.join(args.save_dir, f"{exp_name}_{now_str}.json")
    save_results(results, save_path)

    print(f"\n{'='*60}")
    print(f"Training complete: {exp_name}")
    print(f"Best Valid Acc: {results['best_valid_acc']:.2f}%")
    print(f"Best Test Acc:  {results['best_test_acc']:.2f}%")
    print(f"Avg Time/Epoch: {np.mean(results['epoch_time']):.1f}s")
    print(f"{'='*60}")

    return results


if __name__ == '__main__':
    main()

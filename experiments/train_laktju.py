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
import os
import sys
import argparse
import json

# Add parent dir to path for optimizer imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from LAKTJU import LAKTJU
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
                        choices=['SGD', 'Adam', 'AdamW', 'ATJU', 'LAKTJU', 'Adan'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=None, help='Override default LR')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--data_dir', type=str, default='./dataset')
    parser.add_argument('--workers', type=int, default=4)
    # LAKTJU-specific
    parser.add_argument('--c_base', type=float, default=1.0)
    parser.add_argument('--kappa', type=float, default=5.0)
    parser.add_argument('--homotopy_sharpness', type=float, default=10.0)
    parser.add_argument('--homotopy_speed', type=float, default=1.0)
    parser.add_argument('--kf_damping', type=float, default=1e-3)
    parser.add_argument('--kf_update_interval', type=int, default=20)
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
        a_lr = tju_lr / 10.0
        optimizer = LAKTJU(params, tju_lr=tju_lr, a_lr=a_lr,
                           beta1=0.9, beta2=0.999, eps=1e-8,
                           weight_decay=args.weight_decay,
                           c_base=args.c_base, kappa=args.kappa,
                           homotopy_sharpness=args.homotopy_sharpness,
                           homotopy_speed=args.homotopy_speed,
                           warmup=20, total_steps=total_steps,
                           kf_update_interval=args.kf_update_interval,
                           kf_damping=args.kf_damping)
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

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        # Update ATJU epoch tracking if needed
        if args.optimizer == 'ATJU':
            optimizer.epoch_now = epoch

        # Pass loss to LAKTJU for QGS loss amplification
        if args.optimizer == 'LAKTJU':
            optimizer.set_loss(loss.item())

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

    # Register KF hooks for LAKTJU
    if args.optimizer == 'LAKTJU':
        optimizer.register_hooks(model)

    criterion = nn.CrossEntropyLoss().to(device)

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
            if args.optimizer in ('LAKTJU', 'ATJU'):
                cur_lr = optimizer.param_groups[0].get('tju_lr', 0)
            else:
                cur_lr = optimizer.param_groups[0].get('lr', 0)

            # Get homotopy s for LAKTJU
            s_str = ""
            if args.optimizer == 'LAKTJU':
                s_val = optimizer._compute_homotopy_s(optimizer._gamma_ema or 1.0)
                s_str = f" s={s_val:.3f}"

            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Valid Acc: {valid_acc:.2f}% | "
                  f"Test Acc: {test_acc:.2f}% | "
                  f"LR: {cur_lr:.6f}{s_str} | "
                  f"Time: {epoch_time:.1f}s")

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

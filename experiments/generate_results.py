"""
Generate result tables and convergence curve plots from experiment JSON files.
Produces paper-ready tables (mean±std across seeds) and matplotlib figures.
"""
import json
import os
import glob
import argparse
import numpy as np
from collections import defaultdict


def load_all_results(results_dir):
    """Load all experiment JSON files, return grouped by (dataset, model, optimizer)."""
    grouped = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(results_dir, '*.json'))):
        try:
            d = json.load(open(f))
            cfg = d.get('config', {})
            ds = cfg.get('dataset', '')
            model = cfg.get('model', '')
            opt = cfg.get('optimizer', '')
            seed = cfg.get('seed', 0)
            epochs = len(d.get('train_loss', []))
            if epochs < 200 or not ds or not opt:
                continue
            key = (ds, model, opt)
            grouped[key].append({
                'seed': seed,
                'file': f,
                'best_valid_acc': d['best_valid_acc'],
                'best_test_acc': d['best_test_acc'],
                'train_loss': d['train_loss'],
                'valid_acc': d['valid_acc'],
                'test_acc': d['test_acc'],
                'epoch_time': d['epoch_time'],
            })
        except Exception as e:
            print(f"  Warning: skipping {f}: {e}")
    return grouped


def generate_tables(grouped):
    """Generate paper-ready comparison tables."""
    print("\n" + "=" * 80)
    print("TABLE: Best Test Accuracy (mean ± std across seeds)")
    print("=" * 80)

    for ds in ['cifar10', 'cifar100']:
        print(f"\n--- {ds.upper()} / ResNet18 ---")
        print(f"{'Optimizer':<12} {'Seeds':>5} {'Best Valid Acc':>20} {'Best Test Acc':>20} {'Avg Time/Epoch':>15}")
        print("-" * 75)

        opt_order = ['SGD', 'Adam', 'AdamW', 'ATJU', 'LAKTJU']
        for opt in opt_order:
            key = (ds, 'resnet18', opt)
            if key not in grouped:
                print(f"{opt:<12} {'N/A':>5}")
                continue
            runs = grouped[key]
            n = len(runs)
            va = [r['best_valid_acc'] for r in runs]
            ta = [r['best_test_acc'] for r in runs]
            et = [np.mean(r['epoch_time']) for r in runs]

            if n >= 2:
                va_str = f"{np.mean(va):.2f} ± {np.std(va):.2f}%"
                ta_str = f"{np.mean(ta):.2f} ± {np.std(ta):.2f}%"
            else:
                va_str = f"{np.mean(va):.2f}%"
                ta_str = f"{np.mean(ta):.2f}%"
            et_str = f"{np.mean(et):.1f}s"

            print(f"{opt:<12} {n:>5} {va_str:>20} {ta_str:>20} {et_str:>15}")

    # Milestone accuracy table (at 50%, 100% epochs)
    print("\n" + "=" * 80)
    print("TABLE: Test Accuracy at Training Milestones")
    print("=" * 80)

    for ds in ['cifar10', 'cifar100']:
        print(f"\n--- {ds.upper()} / ResNet18 ---")
        milestones = [50, 100, 150, 200]
        header = f"{'Optimizer':<12}" + "".join(f"{'Ep '+str(m):>12}" for m in milestones)
        print(header)
        print("-" * (12 + 12 * len(milestones)))

        for opt in ['SGD', 'Adam', 'AdamW', 'ATJU', 'LAKTJU']:
            key = (ds, 'resnet18', opt)
            if key not in grouped:
                continue
            runs = grouped[key]
            row = f"{opt:<12}"
            for m in milestones:
                idx = m - 1
                vals = [r['test_acc'][idx] for r in runs if len(r['test_acc']) > idx]
                if vals:
                    if len(vals) >= 2:
                        row += f"{np.mean(vals):>7.2f}±{np.std(vals):.1f}"
                    else:
                        row += f"{np.mean(vals):>11.2f}%"
                else:
                    row += f"{'N/A':>12}"
            print(row)


def generate_plots(grouped, results_dir):
    """Generate convergence curve plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nWARNING: matplotlib not installed, skipping plots.")
        return

    fig_dir = os.path.join(results_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    colors = {
        'SGD': '#1f77b4', 'Adam': '#ff7f0e', 'AdamW': '#2ca02c',
        'ATJU': '#d62728', 'LAKTJU': '#9467bd'
    }
    opt_order = ['SGD', 'Adam', 'AdamW', 'ATJU', 'LAKTJU']

    for ds in ['cifar10', 'cifar100']:
        # --- Training Loss ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        ax.set_title(f'{ds.upper()} - Training Loss', fontsize=13)
        for opt in opt_order:
            key = (ds, 'resnet18', opt)
            if key not in grouped:
                continue
            runs = grouped[key]
            losses = np.array([r['train_loss'] for r in runs])
            mean_loss = losses.mean(axis=0)
            epochs = np.arange(1, len(mean_loss) + 1)
            ax.plot(epochs, mean_loss, label=opt, color=colors.get(opt, 'gray'), linewidth=1.5)
            if losses.shape[0] >= 2:
                std_loss = losses.std(axis=0)
                ax.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss,
                                alpha=0.15, color=colors.get(opt, 'gray'))
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Training Loss')
        ax.set_yscale('log')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # --- Validation Accuracy ---
        ax = axes[1]
        ax.set_title(f'{ds.upper()} - Validation Accuracy', fontsize=13)
        for opt in opt_order:
            key = (ds, 'resnet18', opt)
            if key not in grouped:
                continue
            runs = grouped[key]
            accs = np.array([r['valid_acc'] for r in runs])
            mean_acc = accs.mean(axis=0)
            epochs = np.arange(1, len(mean_acc) + 1)
            ax.plot(epochs, mean_acc, label=opt, color=colors.get(opt, 'gray'), linewidth=1.5)
            if accs.shape[0] >= 2:
                std_acc = accs.std(axis=0)
                ax.fill_between(epochs, mean_acc - std_acc, mean_acc + std_acc,
                                alpha=0.15, color=colors.get(opt, 'gray'))
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Accuracy (%)')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(fig_dir, f'{ds}_resnet18_curves.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")

    # --- Per-epoch time comparison (bar chart) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title('Average Time per Epoch (seconds)', fontsize=13)
    x_labels = []
    x_vals = []
    bar_colors = []
    for ds in ['cifar10', 'cifar100']:
        for opt in opt_order:
            key = (ds, 'resnet18', opt)
            if key not in grouped:
                continue
            runs = grouped[key]
            avg_t = np.mean([np.mean(r['epoch_time']) for r in runs])
            x_labels.append(f'{ds[-2:]}-{opt}')
            x_vals.append(avg_t)
            bar_colors.append(colors.get(opt, 'gray'))
    ax.bar(range(len(x_vals)), x_vals, color=bar_colors, alpha=0.8)
    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Seconds')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    path = os.path.join(fig_dir, 'epoch_time_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='./results')
    args = parser.parse_args()

    print("Loading experiment results...")
    grouped = load_all_results(args.results_dir)
    print(f"Found {sum(len(v) for v in grouped.values())} completed runs "
          f"across {len(grouped)} configurations.")

    generate_tables(grouped)
    generate_plots(grouped, args.results_dir)

    # Save machine-readable summary
    summary = {}
    for (ds, model, opt), runs in grouped.items():
        key = f"{ds}_{model}_{opt}"
        ta = [r['best_test_acc'] for r in runs]
        summary[key] = {
            'n_seeds': len(runs),
            'best_test_mean': round(np.mean(ta), 2),
            'best_test_std': round(np.std(ta), 2) if len(ta) >= 2 else 0.0,
            'seeds': [r['seed'] for r in runs],
        }
    summary_path = os.path.join(args.results_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == '__main__':
    main()

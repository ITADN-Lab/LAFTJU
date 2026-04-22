#!/usr/bin/env python3
"""
Generate publication-quality figures for TII paper — CIFAR-100 experiments.

Outputs:
  1) fig_cifar100_train_curves.pdf    — Training loss & test accuracy curves (4 optimizers)
  2) fig_cifar100_bar_comparison.pdf  — Bar chart: mean ± std across seeds
  3) fig_cifar100_ablation_heatmap.pdf — Hyperparameter sensitivity heatmap
  4) fig_cifar100_convergence.pdf     — Convergence speed comparison (epochs to 70%/73%)
  5) fig_cifar100_box_violin.pdf      — Box/violin plot of multi-seed distribution
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

# ── IEEE / TII style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.2,
    'lines.markersize': 3,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
})

DATA_DIR = '/home/hadoop/workstation/md/TJU-V5(ATJU)-sourcecode/ATJU/experiments/results_cifar100_v9'
OUT_DIR  = '/home/hadoop/workstation/md/TJU-V5(ATJU)-sourcecode/ATJU/paper/cifar100_figures'

# Color palette (IEEE-friendly, color-blind safe)
COLORS = {
    'Adam':      '#4477AA',   # blue
    'AdamW':     '#EE6677',   # red
    'Adan':      '#CCBB44',   # yellow
    'LAFTJU-NS': '#AA3377',   # purple
}
MARKERS = {'Adam': 's', 'AdamW': '^', 'Adan': 'D', 'LAFTJU-NS': 'o'}

# ── Load data ─────────────────────────────────────────────────────────────────
def load_json(fname):
    with open(os.path.join(DATA_DIR, fname)) as f:
        return json.load(f)

# Best seed=42 runs for each optimizer (for training curves)
runs = {
    'Adam':      load_json('cifar100_resnet18_Adam_seed42_20260403_220342.json'),
    'AdamW':     load_json('cifar100_resnet18_AdamW_seed42_20260403_230715.json'),
    'Adan':      load_json('cifar100_resnet18_Adan_seed42_20260404_001031.json'),
    'LAFTJU-NS': load_json('cifar100_resnet18_LAKTJU_NS_seed42_20260404_053744.json'),
}

# Summary data
summary = load_json('summary_v9.json')

# Grid search results
grid = summary['grid_search']


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1: Training Loss & Test Accuracy Curves (2-panel)
# ══════════════════════════════════════════════════════════════════════════════
def smooth(y, window=5):
    """Simple moving average for cleaner curves."""
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode='valid')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.6))  # IEEE double-column width

for name in ['Adam', 'AdamW', 'Adan', 'LAFTJU-NS']:
    d = runs[name]
    epochs = np.arange(1, len(d['train_loss']) + 1)
    loss_sm = smooth(d['train_loss'], window=5)
    ep_sm = np.arange(3, 3 + len(loss_sm))  # adjust for smoothing offset

    ax1.plot(ep_sm, loss_sm, color=COLORS[name], label=name,
             linewidth=1.0, alpha=0.9)

    # Use valid_acc (evaluated every epoch) for smooth curves
    valid_acc = d['valid_acc']
    valid_sm = smooth(valid_acc, window=7)
    ep_sm2 = np.arange(4, 4 + len(valid_sm))
    ax2.plot(ep_sm2, valid_sm, color=COLORS[name], label=name,
             linewidth=1.0, alpha=0.9)

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Loss')
ax1.set_xlim(0, 300)
ax1.set_ylim(0.6, 4.5)
ax1.legend(loc='upper right', framealpha=0.9, edgecolor='gray')
ax1.set_title('(a) Training Loss', fontsize=9)

ax2.set_xlabel('Epoch')
ax2.set_ylabel('Validation Accuracy (%)')
ax2.set_xlim(0, 300)
ax2.set_ylim(0, 80)
ax2.legend(loc='lower right', framealpha=0.9, edgecolor='gray')
ax2.set_title('(b) Validation Accuracy', fontsize=9)

plt.tight_layout(w_pad=2.5)
fig.savefig(os.path.join(OUT_DIR, 'fig_cifar100_train_curves.pdf'))
fig.savefig(os.path.join(OUT_DIR, 'fig_cifar100_train_curves.png'))
plt.close()
print('[1/5] Training curves saved.')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2: Bar Chart — Mean ± Std across seeds
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(3.5, 2.8))

baselines = summary['baselines']
# Add LAFTJU-NS best 5-seed result
ns_data = summary['final_5seed']['top1_lr0.008_wd0.01_ns100_ls0.1']

names = ['Adam', 'AdamW', 'Adan', 'LAFTJU-NS']
means = [baselines['Adam']['avg'], baselines['AdamW']['avg'],
         baselines['Adan']['avg'], ns_data['avg']]
stds  = [baselines['Adam']['std'], baselines['AdamW']['std'],
         baselines['Adan']['std'], ns_data['std']]
n_seeds = [3, 3, 3, 5]

x = np.arange(len(names))
bars = ax.bar(x, means, width=0.55, color=[COLORS[n] for n in names],
              edgecolor='black', linewidth=0.5, alpha=0.85,
              yerr=stds, capsize=4, error_kw={'linewidth': 0.8, 'capthick': 0.8})

# Add value labels
for i, (m, s, ns) in enumerate(zip(means, stds, n_seeds)):
    ax.text(i, m + s + 0.15, f'{m:.2f}%', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    ax.text(i, m - s - 0.3, f'({ns} seeds)', ha='center', va='top', fontsize=6, color='gray')

ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=8)
ax.set_ylabel('Test Accuracy (%)')
ax.set_ylim(72.5, 77.0)
ax.set_title('CIFAR-100 Test Accuracy Comparison', fontsize=9)
ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))

# Highlight best
bars[-1].set_edgecolor('#AA3377')
bars[-1].set_linewidth(1.5)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig_cifar100_bar_comparison.pdf'))
fig.savefig(os.path.join(OUT_DIR, 'fig_cifar100_bar_comparison.png'))
plt.close()
print('[2/5] Bar comparison saved.')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3: Ablation Heatmap — lr × wd sensitivity
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(3.5, 2.8))

# Extract grid data for ns=100, ls=0.1 (main grid)
lrs = sorted(set(g['lr'] for g in grid if g['ns'] == 100 and g['ls'] == 0.1))
wds = sorted(set(g['wd'] for g in grid if g['ns'] == 100 and g['ls'] == 0.1))

heatmap = np.full((len(wds), len(lrs)), np.nan)
for g in grid:
    if g['ns'] == 100 and g['ls'] == 0.1:
        i = wds.index(g['wd'])
        j = lrs.index(g['lr'])
        heatmap[i, j] = g['test']

im = ax.imshow(heatmap, cmap='YlOrRd', aspect='auto', vmin=73.5, vmax=76.0)
cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
cbar.set_label('Test Acc (%)', fontsize=8)
cbar.ax.tick_params(labelsize=7)

# Labels
ax.set_xticks(range(len(lrs)))
ax.set_xticklabels([str(lr) for lr in lrs], fontsize=7)
ax.set_yticks(range(len(wds)))
ax.set_yticklabels([str(wd) for wd in wds], fontsize=7)
ax.set_xlabel('Learning Rate')
ax.set_ylabel('Weight Decay')
ax.set_title('LAFTJU-NS Hyperparameter Sensitivity\n(ns_interval=100, label_smoothing=0.1)', fontsize=8)

# Annotate cells
for i in range(len(wds)):
    for j in range(len(lrs)):
        if not np.isnan(heatmap[i, j]):
            color = 'white' if heatmap[i, j] > 75.0 else 'black'
            ax.text(j, i, f'{heatmap[i, j]:.1f}', ha='center', va='center',
                    fontsize=7, fontweight='bold', color=color)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig_cifar100_ablation_heatmap.pdf'))
fig.savefig(os.path.join(OUT_DIR, 'fig_cifar100_ablation_heatmap.png'))
plt.close()
print('[3/5] Ablation heatmap saved.')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4: Convergence Speed — Epochs to reach threshold accuracy
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(3.5, 2.6))

thresholds = [50, 55, 60, 65, 70, 73]

for name in ['Adam', 'AdamW', 'Adan', 'LAFTJU-NS']:
    d = runs[name]
    valid_acc = d['valid_acc']  # use valid_acc (every epoch)
    epochs_to = []
    for thr in thresholds:
        found = False
        for ep, acc in enumerate(valid_acc, 1):
            if acc >= thr:
                epochs_to.append(ep)
                found = True
                break
        if not found:
            epochs_to.append(300)  # didn't reach

    ax.plot(thresholds, epochs_to, color=COLORS[name], marker=MARKERS[name],
            label=name, linewidth=1.2, markersize=5)

ax.set_xlabel('Target Test Accuracy (%)')
ax.set_ylabel('Epochs to Reach')
ax.set_title('Convergence Speed Comparison', fontsize=9)
ax.legend(loc='upper left', framealpha=0.9, edgecolor='gray', fontsize=7)
ax.set_xlim(49, 74)
ax.set_ylim(0, 250)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig_cifar100_convergence.pdf'))
fig.savefig(os.path.join(OUT_DIR, 'fig_cifar100_convergence.png'))
plt.close()
print('[4/5] Convergence speed saved.')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5: Box + Strip Plot — Multi-seed distribution
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(3.5, 2.8))

all_data = {
    'Adam':      baselines['Adam']['seeds'],
    'AdamW':     baselines['AdamW']['seeds'],
    'Adan':      baselines['Adan']['seeds'],
    'LAFTJU-NS': ns_data['seeds'],
}

names_ordered = ['Adam', 'AdamW', 'Adan', 'LAFTJU-NS']
data_list = [all_data[n] for n in names_ordered]

bp = ax.boxplot(data_list, tick_labels=names_ordered, widths=0.45,
                patch_artist=True, notch=False,
                boxprops=dict(linewidth=0.8),
                whiskerprops=dict(linewidth=0.8),
                medianprops=dict(linewidth=1.2, color='black'),
                capprops=dict(linewidth=0.8))

for i, (patch, name) in enumerate(zip(bp['boxes'], names_ordered)):
    patch.set_facecolor(COLORS[name])
    patch.set_alpha(0.6)

    # Strip plot (individual seeds as dots)
    y_data = data_list[i]
    x_jitter = np.random.normal(i + 1, 0.06, len(y_data))
    ax.scatter(x_jitter, y_data, color=COLORS[name], s=20, zorder=5,
               edgecolors='black', linewidths=0.4, alpha=0.9)

ax.set_ylabel('Test Accuracy (%)')
ax.set_ylim(72.5, 77.0)
ax.set_title('Distribution of Test Accuracy Across Seeds', fontsize=9)
ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig_cifar100_box_violin.pdf'))
fig.savefig(os.path.join(OUT_DIR, 'fig_cifar100_box_violin.png'))
plt.close()
print('[5/5] Box plot saved.')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6 (Bonus): NS Interval Ablation Bar Chart
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(3.0, 2.4))

ns_vals = []
ns_accs = []
for g in grid:
    if g['lr'] == 0.003 and g['wd'] == 0.005 and g['ls'] == 0.1:
        ns_vals.append(g['ns'])
        ns_accs.append(g['test'])

# Sort by ns_interval
pairs = sorted(zip(ns_vals, ns_accs))
ns_vals = [p[0] for p in pairs]
ns_accs = [p[1] for p in pairs]

colors_ns = ['#66CCEE' if v != 100 else '#AA3377' for v in ns_vals]
bars = ax.bar(range(len(ns_vals)), ns_accs, width=0.55, color=colors_ns,
              edgecolor='black', linewidth=0.5, alpha=0.85)

for i, (v, a) in enumerate(zip(ns_vals, ns_accs)):
    ax.text(i, a + 0.08, f'{a:.2f}%', ha='center', va='bottom', fontsize=7, fontweight='bold')

ax.set_xticks(range(len(ns_vals)))
ax.set_xticklabels([str(v) for v in ns_vals], fontsize=8)
ax.set_xlabel('NS Interval (steps)')
ax.set_ylabel('Test Accuracy (%)')
ax.set_ylim(73.0, 75.0)
ax.set_title('Effect of NS Orthogonalization Interval\n(lr=0.003, wd=0.005)', fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig_cifar100_ns_ablation.pdf'))
fig.savefig(os.path.join(OUT_DIR, 'fig_cifar100_ns_ablation.png'))
plt.close()
print('[6/6] NS ablation saved.')

print(f'\nAll figures saved to {OUT_DIR}/')

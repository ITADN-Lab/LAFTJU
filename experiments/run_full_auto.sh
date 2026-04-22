#!/bin/bash
# =============================================================================
# LAKTJU Full Automated Experiment Pipeline
# Runs all experiments sequentially, no user interaction needed.
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

EPOCHS=200
BATCH=128
SAVE_DIR="./results"
DATA_DIR="./dataset"
LOG="./results/experiment_log_full_auto.txt"
SEEDS="42 123 456"

FEISHU_WEBHOOK="https://open.feishu.cn/open-apis/bot/v2/hook/7eba1332-5fbd-4237-bf52-88f5dedc205d"

mkdir -p "$SAVE_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG"
}

feishu() {
    local title="$1" color="$2" body="$3"
    curl -s -X POST "$FEISHU_WEBHOOK" \
      -H "Content-Type: application/json" \
      -d "{
        \"msg_type\": \"interactive\",
        \"card\": {
          \"header\": {
            \"title\": {\"tag\": \"plain_text\", \"content\": \"$title\"},
            \"template\": \"$color\"
          },
          \"elements\": [
            {\"tag\": \"markdown\", \"content\": \"$body\"}
          ]
        }
      }" > /dev/null 2>&1 || true
}

run_exp() {
    local dataset=$1 model=$2 optimizer=$3 seed=$4
    local tag="${dataset}_${model}_${optimizer}_seed${seed}"

    # Check if already completed (200 epochs)
    local existing=$(ls -t "$SAVE_DIR"/${tag}_*.json 2>/dev/null | head -1)
    if [ -n "$existing" ]; then
        local ep=$(python3 -c "import json; d=json.load(open('$existing')); print(len(d['train_loss']))" 2>/dev/null || echo 0)
        if [ "$ep" -ge "$EPOCHS" ]; then
            log "SKIP $tag (already done: $existing, $ep epochs)"
            return 0
        fi
    fi

    log "START $tag"
    python3 train_laktju.py \
        --dataset "$dataset" --model "$model" --optimizer "$optimizer" \
        --epochs "$EPOCHS" --batch_size "$BATCH" --seed "$seed" \
        --save_dir "$SAVE_DIR" --data_dir "$DATA_DIR" --workers 4 \
        2>&1 | tail -20 | tee -a "$LOG"
    log "DONE  $tag"

    # Send Feishu notification with results
    local result_file=$(ls -t "$SAVE_DIR"/${tag}_*.json 2>/dev/null | head -1)
    if [ -n "$result_file" ]; then
        local info=$(python3 -c "
import json; d=json.load(open('$result_file'))
ep=len(d['train_loss'])
print(f'{ep}|{d[\"best_valid_acc\"]:.2f}|{d[\"best_test_acc\"]:.2f}')
" 2>/dev/null || echo "?|?|?")
        local ep=$(echo "$info" | cut -d'|' -f1)
        local va=$(echo "$info" | cut -d'|' -f2)
        local ta=$(echo "$info" | cut -d'|' -f3)
        feishu "✅ $tag done" "green" "**Epochs**: ${ep}\\n**Best Valid Acc**: ${va}%\\n**Best Test Acc**: ${ta}%"
    fi
}

# =============================================================================
log "============================================"
log "LAKTJU Full Automated Experiment Pipeline (KF rewrite)"
log "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
log "============================================"

# =============================================================================
# Phase 1: Clear ALL old LAKTJU results (optimizer was rewritten)
# =============================================================================
log "--- Phase 1: Clearing old LAKTJU results ---"
rm -f "$SAVE_DIR"/cifar10_resnet18_LAKTJU_seed*_*.json
rm -f "$SAVE_DIR"/cifar100_resnet18_LAKTJU_seed*_*.json
log "Old LAKTJU results removed."

# =============================================================================
# Phase 2: Re-run LAKTJU only (2 datasets x 3 seeds = 6 runs)
# Baselines (SGD, Adam, AdamW, ATJU) are already done — skip them.
# =============================================================================
log "--- Phase 2: LAKTJU 3-seed runs ---"
for SEED in $SEEDS; do
    for DATASET in cifar10 cifar100; do
        run_exp "$DATASET" resnet18 LAKTJU "$SEED"
    done
done

# =============================================================================
# Phase 3: Generate summary tables and figures
# =============================================================================
log "--- Phase 3: Generating results ---"
python3 generate_results.py --results_dir "$SAVE_DIR" 2>&1 | tee -a "$LOG"

# Send final summary to Feishu
SUMMARY=$(python3 -c "
import json, glob, os
from collections import defaultdict
grouped = defaultdict(list)
for f in glob.glob('$SAVE_DIR/*.json'):
    try:
        d = json.load(open(f))
        cfg = d.get('config', {})
        ds, opt = cfg.get('dataset',''), cfg.get('optimizer','')
        if len(d.get('train_loss',[])) >= 200 and ds and opt:
            grouped[(ds,opt)].append(d['best_test_acc'])
    except: pass
import numpy as np
lines = []
for ds in ['cifar10','cifar100']:
    lines.append(f'**{ds.upper()}**:')
    for opt in ['SGD','Adam','AdamW','ATJU','LAKTJU']:
        vals = grouped.get((ds,opt),[])
        if vals:
            if len(vals)>=2:
                lines.append(f'  {opt}: {np.mean(vals):.2f}±{np.std(vals):.2f}% ({len(vals)} seeds)')
            else:
                lines.append(f'  {opt}: {vals[0]:.2f}% (1 seed)')
print('\\\\n'.join(lines))
" 2>/dev/null || echo "Failed to generate results")

feishu "🎉 All experiments complete" "purple" "$SUMMARY"

log "============================================"
log "ALL EXPERIMENTS COMPLETE"
log "============================================"

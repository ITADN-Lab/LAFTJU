#!/bin/bash
# =============================================================================
# LAKTJU Full Automated Experiment Pipeline V2
# Adds Adan baseline, re-runs LAKTJU with 3 seeds
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

EPOCHS=200
BATCH=128
SAVE_DIR="./results"
DATA_DIR="./dataset"
LOG="./results/experiment_log_v2.txt"
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
    local start_time=$(date +%s)

    python3 train_laktju.py \
        --dataset "$dataset" --model "$model" --optimizer "$optimizer" \
        --epochs "$EPOCHS" --batch_size "$BATCH" --seed "$seed" \
        --save_dir "$SAVE_DIR" --data_dir "$DATA_DIR" --workers 4 \
        2>&1 | tail -20 | tee -a "$LOG"

    local end_time=$(date +%s)
    local elapsed=$(( end_time - start_time ))
    log "DONE  $tag (${elapsed}s)"

    # Send Feishu notification
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
        feishu "✅ $tag done" "green" "**Epochs**: ${ep}\\n**Best Valid**: ${va}%\\n**Best Test**: ${ta}%\\n**Time**: ${elapsed}s"
    fi
}

# =============================================================================
log "============================================"
log "LAKTJU V2 Full Experiment Pipeline"
log "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
log "============================================"

feishu "🚀 Experiment started" "blue" "**Pipeline**: LAKTJU V2\\n**GPU**: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')\\n**Tasks**: Adan(6) + LAKTJU(6) = 12 runs"

# =============================================================================
# Phase 1: Clear old LAKTJU results (re-run all 3 seeds)
# =============================================================================
log "--- Phase 1: Clearing old LAKTJU results ---"
rm -f "$SAVE_DIR"/cifar10_resnet18_LAKTJU_seed*_*.json
rm -f "$SAVE_DIR"/cifar100_resnet18_LAKTJU_seed*_*.json
log "Old LAKTJU results removed."

# =============================================================================
# Phase 2: Run Adan baseline (2 datasets x 3 seeds = 6 runs)
# =============================================================================
log "--- Phase 2: Adan 3-seed runs ---"
for SEED in $SEEDS; do
    for DATASET in cifar10 cifar100; do
        run_exp "$DATASET" resnet18 Adan "$SEED"
    done
done

# =============================================================================
# Phase 3: Run LAKTJU (2 datasets x 3 seeds = 6 runs)
# =============================================================================
log "--- Phase 3: LAKTJU 3-seed runs ---"
for SEED in $SEEDS; do
    for DATASET in cifar10 cifar100; do
        run_exp "$DATASET" resnet18 LAKTJU "$SEED"
    done
done

# =============================================================================
# Phase 4: Generate summary tables and figures
# =============================================================================
log "--- Phase 4: Generating results ---"
python3 generate_results.py --results_dir "$SAVE_DIR" 2>&1 | tee -a "$LOG"

# Final summary
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
    for opt in ['SGD','Adam','AdamW','Adan','ATJU','LAKTJU']:
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

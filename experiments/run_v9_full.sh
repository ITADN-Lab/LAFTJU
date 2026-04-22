#!/bin/bash
# LAKTJU V9 Full Experiment Pipeline
# Fix: reduced workers (2→0 for SAM), sequential execution to avoid DataLoader crashes
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

EPOCHS=200
BATCH=128
SAVE_DIR="./results"
DATA_DIR="./dataset"
LOG="./results/v9_full_experiment.log"
SEEDS="42 123 456"
FEISHU_WEBHOOK="https://open.feishu.cn/open-apis/bot/v2/hook/7eba1332-5fbd-4237-bf52-88f5dedc205d"

# V9 best config from Round 1+3 experiments
LR=0.01
A_LR_RATIO=0.333
GRAD_CLIP=1.0
SAM_RHO=0.05
HOMOTOPY_SPEED=5.0
WARMUP=100
WEIGHT_DECAY=0.001
LABEL_SMOOTHING=0.1

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
    local dataset=$1 model=$2 seed=$3
    local tag="v9_${dataset}_${model}_LAKTJU_seed${seed}"
    local logfile="$SAVE_DIR/${tag}.log"

    log "START $tag"
    local start_time=$(date +%s)

    python3 train_laktju.py \
        --dataset "$dataset" --model "$model" --optimizer LAKTJU \
        --epochs "$EPOCHS" --batch_size "$BATCH" --seed "$seed" \
        --save_dir "$SAVE_DIR" --data_dir "$DATA_DIR" \
        --workers 2 \
        --lr "$LR" --a_lr_ratio "$A_LR_RATIO" \
        --grad_clip "$GRAD_CLIP" --sam_rho "$SAM_RHO" \
        --homotopy_speed "$HOMOTOPY_SPEED" --warmup "$WARMUP" \
        --weight_decay "$WEIGHT_DECAY" --label_smoothing "$LABEL_SMOOTHING" \
        2>&1 | tee "$logfile"

    local exit_code=${PIPESTATUS[0]}
    local end_time=$(date +%s)
    local elapsed=$(( end_time - start_time ))

    if [ $exit_code -ne 0 ]; then
        log "FAILED $tag (exit=$exit_code, ${elapsed}s)"
        feishu "❌ $tag FAILED" "red" "**Exit code**: ${exit_code}\\n**Time**: ${elapsed}s\\nCheck $logfile"
        return 1
    fi

    log "DONE  $tag (${elapsed}s)"

    # Extract results from latest JSON
    local result_file=$(ls -t "$SAVE_DIR"/${dataset}_resnet18_LAKTJU_seed${seed}_*.json 2>/dev/null | head -1)
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
        log "  -> Epochs=$ep, BestValid=${va}%, BestTest=${ta}%"
    fi
}

log "============================================"
log "LAKTJU V9 Full Pipeline"
log "Config: lr=$LR a_lr_ratio=$A_LR_RATIO gc=$GRAD_CLIP sam=$SAM_RHO"
log "        speed=$HOMOTOPY_SPEED warmup=$WARMUP wd=$WEIGHT_DECAY ls=$LABEL_SMOOTHING"
log "        workers=2, sequential execution"
log "============================================"

feishu "🚀 LAKTJU V9 Pipeline Started" "blue" \
    "**Config**: lr=$LR, sam=$SAM_RHO, gc=$GRAD_CLIP\\n**Runs**: 2 datasets × 3 seeds × ${EPOCHS}ep\\n**Workers**: 2 (crash fix)"

# Run ALL experiments SEQUENTIALLY to avoid resource pressure
for dataset in cifar10 cifar100; do
    for seed in $SEEDS; do
        run_exp "$dataset" "resnet18" "$seed" || true
    done
done

log "============================================"
log "All V9 experiments completed!"
log "============================================"

# Summary
python3 -c "
import json, glob, os
files = sorted(glob.glob('$SAVE_DIR/cifar*_resnet18_LAKTJU_seed*_*.json'))
# Keep only latest per config
latest = {}
for f in files:
    base = os.path.basename(f)
    # Extract config key (dataset_model_opt_seed)
    parts = base.rsplit('_', 1)[0]  # remove timestamp
    latest[parts] = f

print('\n=== LAKTJU V9 Final Results ===')
print(f'{\"Config\":<45} {\"Epochs\":>6} {\"BestValid\":>9} {\"BestTest\":>9}')
print('-' * 75)
for key in sorted(latest):
    d = json.load(open(latest[key]))
    ep = len(d['train_loss'])
    print(f'{key:<45} {ep:>6} {d[\"best_valid_acc\"]:>8.2f}% {d[\"best_test_acc\"]:>8.2f}%')
" 2>/dev/null | tee -a "$LOG" || true

feishu "🎉 LAKTJU V9 Pipeline Complete" "blue" "All 6 experiments finished. Check results."

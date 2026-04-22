#!/bin/bash
set -e
cd "$(dirname "$0")"
LOGDIR="logs_modern"

run_gpt2() {
    local TAG=$1; local SEED=$2; shift 2
    local LOG="$LOGDIR/${TAG}_seed${SEED}.log"
    echo "[$(date +%H:%M:%S)] START: $TAG seed=$SEED"
    PYTHONUNBUFFERED=1 python3 train_gpt2_lm.py \
        --dataset wikitext --seed $SEED \
        --output_dir results_gpt2 \
        "$@" > "$LOG" 2>&1
    ppl=$(grep "Best val PPL" "$LOG" | tail -1)
    echo "  [$(date +%H:%M:%S)] DONE: $TAG seed=$SEED | $ppl"
}

# Lion seed 456
run_gpt2 "lion_small_lr1e-4" 456 \
    --model_size small --optimizer Lion --lr 1e-4 \
    --weight_decay 1.0 --total_steps 20000 \
    --batch_size 8 --grad_accum 4 --warmup_steps 2000 \
    --eval_interval 1000 --log_interval 100

# Adan 3 seeds
for SEED in 42 123 456; do
    run_gpt2 "adan_small_lr1e-3" $SEED \
        --model_size small --optimizer Adan --lr 1e-3 \
        --weight_decay 0.02 --total_steps 20000 \
        --batch_size 8 --grad_accum 4 --warmup_steps 2000 \
        --eval_interval 1000 --log_interval 100
done

# GPT-2 Medium
for OPT_ARGS in \
    "AdamW --optimizer AdamW --lr 6e-4 --weight_decay 0.1" \
    "LAKTJU_NS --optimizer LAKTJU_NS --lr 6e-4 --weight_decay 0.1 --ns_interval 50 --ns_steps 2 --ns_max_dim 1024" \
    "Lion --optimizer Lion --lr 1e-4 --weight_decay 1.0"; do
    OPT_NAME=$(echo $OPT_ARGS | cut -d' ' -f1)
    OPT_FLAGS=$(echo $OPT_ARGS | cut -d' ' -f2-)
    for SEED in 42 123 456; do
        run_gpt2 "medium_${OPT_NAME}" $SEED \
            --model_size medium $OPT_FLAGS \
            --total_steps 10000 \
            --batch_size 4 --grad_accum 4 --warmup_steps 1000 \
            --eval_interval 1000 --log_interval 100
    done
done

echo "=== All remaining experiments complete ==="
echo "End: $(date)"

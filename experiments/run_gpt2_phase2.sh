#!/bin/bash
# GPT-2 Phase 2: Fast sweep (5K steps) to find best configs
# Then Phase 3 will extend best configs to 20K+ steps
set -e
cd "$(dirname "$0")"

LOGDIR="logs_gpt2_phase2"
mkdir -p "$LOGDIR"

COMMON="--model_size small --dataset wikitext --batch_size 8 --grad_accum 4 \
        --eval_interval 500 --log_interval 200 --seq_len 1024 \
        --total_steps 5000 --warmup_steps 500"

echo "=== GPT-2 Phase 2: Fast Sweep (5K steps each) ==="
echo "Start: $(date)"

# ─── AdamW baselines (3 LRs × 1 seed for speed) ───
for lr in 3e-4 6e-4 1e-3; do
    echo "[$(date +%H:%M:%S)] START: AdamW lr=$lr"
    python3 train_gpt2_lm.py $COMMON \
        --optimizer AdamW --lr $lr --weight_decay 0.1 --seed 42 \
        > "$LOGDIR/adamw_lr${lr}_seed42.log" 2>&1
    ppl=$(grep "Best val PPL" "$LOGDIR/adamw_lr${lr}_seed42.log" | tail -1)
    echo "  $ppl"
done

# ─── Adam baseline ───
echo "[$(date +%H:%M:%S)] START: Adam lr=6e-4"
python3 train_gpt2_lm.py $COMMON \
    --optimizer Adam --lr 6e-4 --weight_decay 0.1 --seed 42 \
    > "$LOGDIR/adam_lr6e-4_seed42.log" 2>&1
ppl=$(grep "Best val PPL" "$LOGDIR/adam_lr6e-4_seed42.log" | tail -1)
echo "  $ppl"

# ─── LAKTJU-NS sweep (key configs) ───
for ns_int in 50 100 200 500; do
    echo "[$(date +%H:%M:%S)] START: NS ns=$ns_int K=1 lr=6e-4"
    python3 train_gpt2_lm.py $COMMON \
        --optimizer LAKTJU_NS --lr 6e-4 --weight_decay 0.1 \
        --ns_interval $ns_int --ns_steps 1 --ns_max_dim 1024 --seed 42 \
        > "$LOGDIR/ns${ns_int}_K1_lr6e-4_seed42.log" 2>&1
    ppl=$(grep "Best val PPL" "$LOGDIR/ns${ns_int}_K1_lr6e-4_seed42.log" | tail -1)
    echo "  $ppl"
done

# NS with K=2
for ns_int in 100 200; do
    echo "[$(date +%H:%M:%S)] START: NS ns=$ns_int K=2 lr=6e-4"
    python3 train_gpt2_lm.py $COMMON \
        --optimizer LAKTJU_NS --lr 6e-4 --weight_decay 0.1 \
        --ns_interval $ns_int --ns_steps 2 --ns_max_dim 1024 --seed 42 \
        > "$LOGDIR/ns${ns_int}_K2_lr6e-4_seed42.log" 2>&1
    ppl=$(grep "Best val PPL" "$LOGDIR/ns${ns_int}_K2_lr6e-4_seed42.log" | tail -1)
    echo "  $ppl"
done

# NS with higher LR
for lr in 1e-3 2e-3; do
    echo "[$(date +%H:%M:%S)] START: NS ns=100 K=1 lr=$lr"
    python3 train_gpt2_lm.py $COMMON \
        --optimizer LAKTJU_NS --lr $lr --weight_decay 0.1 \
        --ns_interval 100 --ns_steps 1 --ns_max_dim 1024 --seed 42 \
        > "$LOGDIR/ns100_K1_lr${lr}_seed42.log" 2>&1
    ppl=$(grep "Best val PPL" "$LOGDIR/ns100_K1_lr${lr}_seed42.log" | tail -1)
    echo "  $ppl"
done

# NS with grad centralization
echo "[$(date +%H:%M:%S)] START: NS ns=100 K=1 gc lr=6e-4"
python3 train_gpt2_lm.py $COMMON \
    --optimizer LAKTJU_NS --lr 6e-4 --weight_decay 0.1 \
    --ns_interval 100 --ns_steps 1 --ns_max_dim 1024 --grad_centralization --seed 42 \
    > "$LOGDIR/ns100_K1_gc_lr6e-4_seed42.log" 2>&1
ppl=$(grep "Best val PPL" "$LOGDIR/ns100_K1_gc_lr6e-4_seed42.log" | tail -1)
echo "  $ppl"

# NS with larger ns_max_dim (cover more layers)
echo "[$(date +%H:%M:%S)] START: NS ns=100 K=1 maxd=3072 lr=6e-4"
python3 train_gpt2_lm.py $COMMON \
    --optimizer LAKTJU_NS --lr 6e-4 --weight_decay 0.1 \
    --ns_interval 100 --ns_steps 1 --ns_max_dim 3072 --seed 42 \
    > "$LOGDIR/ns100_K1_maxd3072_lr6e-4_seed42.log" 2>&1
ppl=$(grep "Best val PPL" "$LOGDIR/ns100_K1_maxd3072_lr6e-4_seed42.log" | tail -1)
echo "  $ppl"

echo ""
echo "[$(date +%H:%M:%S)] Phase 2 sweep complete!"
echo "End: $(date)"

# ─── Collect results ───
echo ""
echo "=== Phase 2 Results (sorted by best PPL, lower=better) ==="
for logfile in "$LOGDIR"/*.log; do
    name=$(basename "$logfile" .log)
    ppl=$(grep "Best val PPL" "$logfile" 2>/dev/null | tail -1 | sed 's/.*PPL: \([0-9.]*\).*/\1/')
    if [ -n "$ppl" ]; then
        printf "  %8s  %s\n" "$ppl" "$name"
    else
        printf "  %8s  %s\n" "FAILED" "$name"
    fi
done | sort -n

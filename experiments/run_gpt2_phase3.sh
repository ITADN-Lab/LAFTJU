#!/bin/bash
# GPT-2 Phase 3: Extended training with best configs + untested promising combos
# 20K steps, 3 seeds, to get paper-quality results
set -e
cd "$(dirname "$0")"

LOGDIR="logs_gpt2_phase3"
mkdir -p "$LOGDIR"

COMMON="--model_size small --dataset wikitext --batch_size 8 --grad_accum 4 \
        --eval_interval 1000 --log_interval 500 --seq_len 1024 \
        --total_steps 20000 --warmup_steps 2000"

echo "=== GPT-2 Phase 3: Extended Training (20K steps, 3 seeds) ==="
echo "Start: $(date)"

# ─── Phase 3a: Test untested promising configs (20K steps, seed=42 only) ───
# Key insight from Phase 2: K=2 helped at lr=6e-4 (29.37 vs 29.48)
# But K=2 at lr=1e-3 was never tested! This is the most promising combo.

echo ""
echo "--- Phase 3a: Untested combos (seed=42) ---"

# NS K=2 lr=1e-3 (most promising untested)
echo "[$(date +%H:%M:%S)] START: NS ns=100 K=2 lr=1e-3 (seed=42)"
python3 train_gpt2_lm.py $COMMON \
    --optimizer LAKTJU_NS --lr 1e-3 --weight_decay 0.1 \
    --ns_interval 100 --ns_steps 2 --ns_max_dim 1024 --seed 42 \
    > "$LOGDIR/ns100_K2_lr1e-3_seed42.log" 2>&1
ppl=$(grep "Best val PPL" "$LOGDIR/ns100_K2_lr1e-3_seed42.log" | tail -1)
echo "  $ppl"

# NS ns=50 K=1 lr=1e-3
echo "[$(date +%H:%M:%S)] START: NS ns=50 K=1 lr=1e-3 (seed=42)"
python3 train_gpt2_lm.py $COMMON \
    --optimizer LAKTJU_NS --lr 1e-3 --weight_decay 0.1 \
    --ns_interval 50 --ns_steps 1 --ns_max_dim 1024 --seed 42 \
    > "$LOGDIR/ns50_K1_lr1e-3_seed42.log" 2>&1
ppl=$(grep "Best val PPL" "$LOGDIR/ns50_K1_lr1e-3_seed42.log" | tail -1)
echo "  $ppl"

# NS ns=50 K=2 lr=1e-3 (combine both improvements)
echo "[$(date +%H:%M:%S)] START: NS ns=50 K=2 lr=1e-3 (seed=42)"
python3 train_gpt2_lm.py $COMMON \
    --optimizer LAKTJU_NS --lr 1e-3 --weight_decay 0.1 \
    --ns_interval 50 --ns_steps 2 --ns_max_dim 1024 --seed 42 \
    > "$LOGDIR/ns50_K2_lr1e-3_seed42.log" 2>&1
ppl=$(grep "Best val PPL" "$LOGDIR/ns50_K2_lr1e-3_seed42.log" | tail -1)
echo "  $ppl"

# AdamW lr=1e-3 (baseline extended to 20K)
echo "[$(date +%H:%M:%S)] START: AdamW lr=1e-3 (seed=42)"
python3 train_gpt2_lm.py $COMMON \
    --optimizer AdamW --lr 1e-3 --weight_decay 0.1 --seed 42 \
    > "$LOGDIR/adamw_lr1e-3_seed42.log" 2>&1
ppl=$(grep "Best val PPL" "$LOGDIR/adamw_lr1e-3_seed42.log" | tail -1)
echo "  $ppl"

# NS ns=100 K=1 lr=1e-3 (Phase 2 best NS, extended to 20K)
echo "[$(date +%H:%M:%S)] START: NS ns=100 K=1 lr=1e-3 (seed=42)"
python3 train_gpt2_lm.py $COMMON \
    --optimizer LAKTJU_NS --lr 1e-3 --weight_decay 0.1 \
    --ns_interval 100 --ns_steps 1 --ns_max_dim 1024 --seed 42 \
    > "$LOGDIR/ns100_K1_lr1e-3_seed42.log" 2>&1
ppl=$(grep "Best val PPL" "$LOGDIR/ns100_K1_lr1e-3_seed42.log" | tail -1)
echo "  $ppl"

echo ""
echo "[$(date +%H:%M:%S)] Phase 3a seed=42 complete! Checking results..."
echo ""
echo "--- Phase 3a Results (seed=42, 20K steps) ---"
for logfile in "$LOGDIR"/*_seed42.log; do
    name=$(basename "$logfile" .log)
    ppl=$(grep "Best val PPL" "$logfile" 2>/dev/null | tail -1 | sed 's/.*PPL: \([0-9.]*\).*/\1/')
    if [ -n "$ppl" ]; then
        printf "  %8s  %s\n" "$ppl" "$name"
    fi
done | sort -n

echo ""
echo "--- Phase 3b: Multi-seed runs for best configs ---"

# Run the top configs with seeds 123 and 456
for seed in 123 456; do
    # AdamW lr=1e-3
    echo "[$(date +%H:%M:%S)] START: AdamW lr=1e-3 (seed=$seed)"
    python3 train_gpt2_lm.py $COMMON \
        --optimizer AdamW --lr 1e-3 --weight_decay 0.1 --seed $seed \
        > "$LOGDIR/adamw_lr1e-3_seed${seed}.log" 2>&1
    ppl=$(grep "Best val PPL" "$LOGDIR/adamw_lr1e-3_seed${seed}.log" | tail -1)
    echo "  $ppl"

    # NS ns=100 K=2 lr=1e-3
    echo "[$(date +%H:%M:%S)] START: NS ns=100 K=2 lr=1e-3 (seed=$seed)"
    python3 train_gpt2_lm.py $COMMON \
        --optimizer LAKTJU_NS --lr 1e-3 --weight_decay 0.1 \
        --ns_interval 100 --ns_steps 2 --ns_max_dim 1024 --seed $seed \
        > "$LOGDIR/ns100_K2_lr1e-3_seed${seed}.log" 2>&1
    ppl=$(grep "Best val PPL" "$LOGDIR/ns100_K2_lr1e-3_seed${seed}.log" | tail -1)
    echo "  $ppl"

    # NS ns=100 K=1 lr=1e-3
    echo "[$(date +%H:%M:%S)] START: NS ns=100 K=1 lr=1e-3 (seed=$seed)"
    python3 train_gpt2_lm.py $COMMON \
        --optimizer LAKTJU_NS --lr 1e-3 --weight_decay 0.1 \
        --ns_interval 100 --ns_steps 1 --ns_max_dim 1024 --seed $seed \
        > "$LOGDIR/ns100_K1_lr1e-3_seed${seed}.log" 2>&1
    ppl=$(grep "Best val PPL" "$LOGDIR/ns100_K1_lr1e-3_seed${seed}.log" | tail -1)
    echo "  $ppl"

    # NS ns=50 K=1 lr=1e-3
    echo "[$(date +%H:%M:%S)] START: NS ns=50 K=1 lr=1e-3 (seed=$seed)"
    python3 train_gpt2_lm.py $COMMON \
        --optimizer LAKTJU_NS --lr 1e-3 --weight_decay 0.1 \
        --ns_interval 50 --ns_steps 1 --ns_max_dim 1024 --seed $seed \
        > "$LOGDIR/ns50_K1_lr1e-3_seed${seed}.log" 2>&1
    ppl=$(grep "Best val PPL" "$LOGDIR/ns50_K1_lr1e-3_seed${seed}.log" | tail -1)
    echo "  $ppl"

    # NS ns=50 K=2 lr=1e-3
    echo "[$(date +%H:%M:%S)] START: NS ns=50 K=2 lr=1e-3 (seed=$seed)"
    python3 train_gpt2_lm.py $COMMON \
        --optimizer LAKTJU_NS --lr 1e-3 --weight_decay 0.1 \
        --ns_interval 50 --ns_steps 2 --ns_max_dim 1024 --seed $seed \
        > "$LOGDIR/ns50_K2_lr1e-3_seed${seed}.log" 2>&1
    ppl=$(grep "Best val PPL" "$LOGDIR/ns50_K2_lr1e-3_seed${seed}.log" | tail -1)
    echo "  $ppl"
done

echo ""
echo "[$(date +%H:%M:%S)] Phase 3 complete!"
echo "End: $(date)"

# ─── Collect all Phase 3 results ───
echo ""
echo "=== Phase 3 Final Results (sorted by best PPL) ==="
for logfile in "$LOGDIR"/*.log; do
    name=$(basename "$logfile" .log)
    ppl=$(grep "Best val PPL" "$logfile" 2>/dev/null | tail -1 | sed 's/.*PPL: \([0-9.]*\).*/\1/')
    if [ -n "$ppl" ]; then
        printf "  %8s  %s\n" "$ppl" "$name"
    else
        printf "  %8s  %s\n" "FAILED" "$name"
    fi
done | sort -n

# ─── Compute averages across seeds ───
echo ""
echo "=== Multi-seed Averages ==="
for config in adamw_lr1e-3 ns100_K2_lr1e-3 ns100_K1_lr1e-3 ns50_K1_lr1e-3 ns50_K2_lr1e-3; do
    ppls=""
    for seed in 42 123 456; do
        logfile="$LOGDIR/${config}_seed${seed}.log"
        if [ -f "$logfile" ]; then
            ppl=$(grep "Best val PPL" "$logfile" 2>/dev/null | tail -1 | sed 's/.*PPL: \([0-9.]*\).*/\1/')
            if [ -n "$ppl" ]; then
                ppls="$ppls $ppl"
            fi
        fi
    done
    if [ -n "$ppls" ]; then
        avg=$(echo $ppls | tr ' ' '\n' | awk '{s+=$1; n++} END {if(n>0) printf "%.2f", s/n}')
        std=$(echo $ppls | tr ' ' '\n' | awk -v avg="$avg" '{s+=($1-avg)^2; n++} END {if(n>1) printf "%.2f", sqrt(s/(n-1)); else printf "N/A"}')
        echo "  $config: avg=$avg ± $std  (seeds: $ppls)"
    fi
done

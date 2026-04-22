#!/bin/bash
# Transformer-XL Phase 1: Fast sweep (20K steps) to find best configs
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LOGDIR="logs_txl_phase1"
mkdir -p "$LOGDIR"

DATA="./data/wikitext-103"
RESULTS="./results_txl"
COMMON="--data $DATA --batch_size 60 --tgt_len 150 --mem_len 150 \
        --max_step 20000 --eval_interval 2000 --log_interval 500 \
        --warmup_step 0"

echo "=== Transformer-XL Phase 1: Fast Sweep (20K steps each) ==="
echo "Start: $(date)"

# ─── Adam baseline (standard TXL config) ───
echo "[$(date +%H:%M:%S)] START: Adam lr=2.5e-4"
python3 train_txl_wt103.py $COMMON \
    --optimizer adam --lr 2.5e-4 --wd 0 --seed 42 \
    --work_dir "$RESULTS" \
    > "$LOGDIR/adam_lr2.5e-4_seed42.log" 2>&1
ppl=$(grep "Best Test PPL" "$LOGDIR/adam_lr2.5e-4_seed42.log" | tail -1)
echo "  $ppl"

# ─── AdamW (test if decoupled WD helps on TXL) ───
echo "[$(date +%H:%M:%S)] START: AdamW lr=2.5e-4 wd=0.02"
python3 train_txl_wt103.py $COMMON \
    --optimizer adamw --lr 2.5e-4 --wd 0.02 --seed 42 \
    --work_dir "$RESULTS" \
    > "$LOGDIR/adamw_lr2.5e-4_wd0.02_seed42.log" 2>&1
ppl=$(grep "Best Test PPL" "$LOGDIR/adamw_lr2.5e-4_wd0.02_seed42.log" | tail -1)
echo "  $ppl"

# ─── LAKTJU_NS sweep (Adam backbone, no WD) ───
# NS ns=100 K=2 (best from PTB LSTM)
echo "[$(date +%H:%M:%S)] START: NS ns=100 K=2 lr=2.5e-4"
python3 train_txl_wt103.py $COMMON \
    --optimizer LAKTJU_NS --lr 2.5e-4 --wd 0 \
    --ns_interval 100 --ns_steps 2 --ns_max_dim 1024 --seed 42 \
    --work_dir "$RESULTS" \
    > "$LOGDIR/ns100_K2_lr2.5e-4_seed42.log" 2>&1
ppl=$(grep "Best Test PPL" "$LOGDIR/ns100_K2_lr2.5e-4_seed42.log" | tail -1)
echo "  $ppl"

# NS ns=50 K=2 (best from GPT-2)
echo "[$(date +%H:%M:%S)] START: NS ns=50 K=2 lr=2.5e-4"
python3 train_txl_wt103.py $COMMON \
    --optimizer LAKTJU_NS --lr 2.5e-4 --wd 0 \
    --ns_interval 50 --ns_steps 2 --ns_max_dim 1024 --seed 42 \
    --work_dir "$RESULTS" \
    > "$LOGDIR/ns50_K2_lr2.5e-4_seed42.log" 2>&1
ppl=$(grep "Best Test PPL" "$LOGDIR/ns50_K2_lr2.5e-4_seed42.log" | tail -1)
echo "  $ppl"

# NS ns=50 K=1
echo "[$(date +%H:%M:%S)] START: NS ns=50 K=1 lr=2.5e-4"
python3 train_txl_wt103.py $COMMON \
    --optimizer LAKTJU_NS --lr 2.5e-4 --wd 0 \
    --ns_interval 50 --ns_steps 1 --ns_max_dim 1024 --seed 42 \
    --work_dir "$RESULTS" \
    > "$LOGDIR/ns50_K1_lr2.5e-4_seed42.log" 2>&1
ppl=$(grep "Best Test PPL" "$LOGDIR/ns50_K1_lr2.5e-4_seed42.log" | tail -1)
echo "  $ppl"

# NS ns=100 K=1
echo "[$(date +%H:%M:%S)] START: NS ns=100 K=1 lr=2.5e-4"
python3 train_txl_wt103.py $COMMON \
    --optimizer LAKTJU_NS --lr 2.5e-4 --wd 0 \
    --ns_interval 100 --ns_steps 1 --ns_max_dim 1024 --seed 42 \
    --work_dir "$RESULTS" \
    > "$LOGDIR/ns100_K1_lr2.5e-4_seed42.log" 2>&1
ppl=$(grep "Best Test PPL" "$LOGDIR/ns100_K1_lr2.5e-4_seed42.log" | tail -1)
echo "  $ppl"

# ─── Higher LR for NS (NS enables higher LR) ───
echo "[$(date +%H:%M:%S)] START: NS ns=100 K=2 lr=5e-4"
python3 train_txl_wt103.py $COMMON \
    --optimizer LAKTJU_NS --lr 5e-4 --wd 0 \
    --ns_interval 100 --ns_steps 2 --ns_max_dim 1024 --seed 42 \
    --work_dir "$RESULTS" \
    > "$LOGDIR/ns100_K2_lr5e-4_seed42.log" 2>&1
ppl=$(grep "Best Test PPL" "$LOGDIR/ns100_K2_lr5e-4_seed42.log" | tail -1)
echo "  $ppl"

echo "[$(date +%H:%M:%S)] START: NS ns=50 K=2 lr=5e-4"
python3 train_txl_wt103.py $COMMON \
    --optimizer LAKTJU_NS --lr 5e-4 --wd 0 \
    --ns_interval 50 --ns_steps 2 --ns_max_dim 1024 --seed 42 \
    --work_dir "$RESULTS" \
    > "$LOGDIR/ns50_K2_lr5e-4_seed42.log" 2>&1
ppl=$(grep "Best Test PPL" "$LOGDIR/ns50_K2_lr5e-4_seed42.log" | tail -1)
echo "  $ppl"

# ─── Even higher LR ───
echo "[$(date +%H:%M:%S)] START: NS ns=100 K=2 lr=1e-3"
python3 train_txl_wt103.py $COMMON \
    --optimizer LAKTJU_NS --lr 1e-3 --wd 0 \
    --ns_interval 100 --ns_steps 2 --ns_max_dim 1024 --seed 42 \
    --work_dir "$RESULTS" \
    > "$LOGDIR/ns100_K2_lr1e-3_seed42.log" 2>&1
ppl=$(grep "Best Test PPL" "$LOGDIR/ns100_K2_lr1e-3_seed42.log" | tail -1)
echo "  $ppl"

# Adam at higher LR for fair comparison
echo "[$(date +%H:%M:%S)] START: Adam lr=5e-4"
python3 train_txl_wt103.py $COMMON \
    --optimizer adam --lr 5e-4 --wd 0 --seed 42 \
    --work_dir "$RESULTS" \
    > "$LOGDIR/adam_lr5e-4_seed42.log" 2>&1
ppl=$(grep "Best Test PPL" "$LOGDIR/adam_lr5e-4_seed42.log" | tail -1)
echo "  $ppl"

echo ""
echo "[$(date +%H:%M:%S)] Phase 1 sweep complete!"
echo "End: $(date)"

# ─── Collect results ───
echo ""
echo "=== Phase 1 Results (sorted by best test PPL, lower=better) ==="
for logfile in "$LOGDIR"/*.log; do
    name=$(basename "$logfile" .log)
    ppl=$(grep "Best Test PPL" "$logfile" 2>/dev/null | tail -1 | sed 's/.*PPL: \([0-9.]*\).*/\1/')
    if [ -n "$ppl" ]; then
        printf "  %8s  %s\n" "$ppl" "$name"
    else
        printf "  %8s  %s\n" "FAILED" "$name"
    fi
done | sort -n

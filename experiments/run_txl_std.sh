#!/bin/bash
# Transformer-XL-base on WikiText-103 (standard tokenized, 267K vocab)
# Config matches Adan paper (Xie et al., TPAMI 2024): 151.1M params
# Reference results: Adam 200K → PPL 24.2, Adan 200K → PPL 23.5
#
# Usage:
#   ./run_txl_std.sh quick        # 2K-step smoke test
#   ./run_txl_std.sh phase1       # 20K-step sweep (7 configs)
#   ./run_txl_std.sh phase2       # 200K-step final (3 configs × 3 seeds)
#   ./run_txl_std.sh single <tag> <max_step> <seed> [extra args...]
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LOGDIR="logs_txl_std"
mkdir -p "$LOGDIR"
DATA="./data/wikitext-103-std"
RESULTS="./results_txl_std"
mkdir -p "$RESULTS"

# Model: TXL-base with div_val=4 → 56.4M params (balanced arch)
# div_val=1 puts 73% params in embeddings and fails to learn
# Training: batch=60, warmup=4000
WARMUP=4000
BATCH=60
TGT=150
MEM=150
DIV_VAL=4

COMMON="--data $DATA --batch_size $BATCH --tgt_len $TGT --mem_len $MEM \
        --div_val $DIV_VAL --warmup_step $WARMUP"

run() {
    local TAG=$1
    local MAX_STEP=$2
    local SEED=$3
    shift 3
    local LOG="$LOGDIR/${TAG}_seed${SEED}.log"
    echo "[$(date +%H:%M:%S)] START: $TAG seed=$SEED max_step=$MAX_STEP"
    PYTHONUNBUFFERED=1 python3 train_txl_wt103.py $COMMON \
        --max_step $MAX_STEP --seed $SEED \
        --eval_interval 4000 --log_interval 500 \
        --work_dir "$RESULTS" \
        "$@" \
        > "$LOG" 2>&1
    ppl=$(grep "Best Test PPL" "$LOG" | tail -1)
    echo "  [$(date +%H:%M:%S)] DONE: $TAG seed=$SEED | $ppl"
}

phase=$1
seed=${2:-42}

case "$phase" in
    quick)
        echo "=== Quick Smoke Test (2K steps) ==="
        run "smoke_adam" 2000 42 --optimizer adam --lr 2.5e-4 --wd 0.02 --eval_interval 500 --log_interval 100
        echo "Smoke test done."
        ;;

    phase1)
        echo "=== Phase 1: 20K-step sweep (seed=$seed) ==="
        echo "Start: $(date)"
        echo "Model: TXL-base 151.1M params | Vocab: 267K | div_val=$DIV_VAL"

        # Baselines (matching Adan paper: wd=0.02 for Adam too)
        run "adam_lr2.5e-4_wd0.02"   20000 $seed --optimizer adam   --lr 2.5e-4 --wd 0.02
        run "adamw_lr2.5e-4_wd0.02"  20000 $seed --optimizer adamw  --lr 2.5e-4 --wd 0.02

        # LAKTJU_NS configs
        run "ns100_K2_lr2.5e-4"  20000 $seed --optimizer LAKTJU_NS --lr 2.5e-4 --wd 0 --ns_interval 100 --ns_steps 2 --ns_max_dim 1024
        run "ns50_K2_lr2.5e-4"   20000 $seed --optimizer LAKTJU_NS --lr 2.5e-4 --wd 0 --ns_interval 50  --ns_steps 2 --ns_max_dim 1024
        run "ns100_K2_lr5e-4"    20000 $seed --optimizer LAKTJU_NS --lr 5e-4   --wd 0 --ns_interval 100 --ns_steps 2 --ns_max_dim 1024
        run "ns50_K2_lr5e-4"     20000 $seed --optimizer LAKTJU_NS --lr 5e-4   --wd 0 --ns_interval 50  --ns_steps 2 --ns_max_dim 1024

        # NS with weight decay (like baselines)
        run "ns100_K2_lr2.5e-4_wd0.02" 20000 $seed --optimizer LAKTJU_NS --lr 2.5e-4 --wd 0.02 --ns_interval 100 --ns_steps 2 --ns_max_dim 1024

        echo ""
        echo "=== Phase 1 Results (lower PPL = better) ==="
        for f in "$LOGDIR"/*_seed${seed}.log; do
            tag=$(basename "$f" _seed${seed}.log)
            ppl=$(grep "Best Test PPL" "$f" 2>/dev/null | tail -1 | grep -oP '[\d.]+' | head -1)
            printf "  %-35s %s\n" "$tag" "${ppl:-FAILED}"
        done | sort -t' ' -k2 -n
        echo "Phase 1 done: $(date)"
        ;;

    phase2)
        echo "=== Phase 2: Multi-seed validation (20K steps, 3 seeds) ==="
        echo "Start: $(date)"
        echo "Model: TXL-base | batch=$BATCH | div_val=$DIV_VAL | 20K steps"

        for SEED in 42 123 456; do
            run "adamw_lr2.5e-4_wd0.02" 20000 $SEED --optimizer adamw --lr 2.5e-4 --wd 0.02
            run "adamw_lr5e-4_wd0.02"   20000 $SEED --optimizer adamw --lr 5e-4   --wd 0.02
            run "ns100_K2_lr2.5e-4"     20000 $SEED --optimizer LAKTJU_NS --lr 2.5e-4 --wd 0 --ns_interval 100 --ns_steps 2 --ns_max_dim 1024
            run "ns100_K2_lr5e-4"       20000 $SEED --optimizer LAKTJU_NS --lr 5e-4   --wd 0 --ns_interval 100 --ns_steps 2 --ns_max_dim 1024
        done

        echo ""
        echo "=== Phase 2 Final Results ==="
        for f in "$LOGDIR"/*_seed*.log; do
            tag=$(basename "$f" .log)
            ppl=$(grep "Best Test PPL" "$f" 2>/dev/null | tail -1 | grep -oP '[\d.]+' | head -1)
            printf "  %-45s %s\n" "$tag" "${ppl:-FAILED}"
        done | sort -t' ' -k2 -n
        echo "Phase 2 done: $(date)"
        ;;

    single)
        TAG=$2; MS=$3; S=$4; shift 4
        run "$TAG" $MS $S "$@"
        ;;

    *)
        echo "Usage: $0 {quick|phase1|phase2|single <tag> <max_step> <seed> [args...]}"
        exit 1
        ;;
esac

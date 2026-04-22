#!/bin/bash
# Transformer-XL WikiText-103: LAKTJU_NS vs Adam vs AdamW
# Reference: Adan paper Table 3 (TXL-base, 200K steps)
#   Adam 200K → test PPL 24.2
#   Adan 200K → test PPL 23.5
#
# Phase 1: Quick sweep (20K steps) to find best NS config
# Phase 2: Full run (200K steps) with best configs, 3 seeds
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LOGDIR="logs_txl_main"
mkdir -p "$LOGDIR"
DATA="./data/wikitext-103"
RESULTS="./results_txl_main"
mkdir -p "$RESULTS"

WARMUP=4000
BATCH=60
TGT=150
MEM=150

COMMON="--data $DATA --batch_size $BATCH --tgt_len $TGT --mem_len $MEM --warmup_step $WARMUP"

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
    phase1)
        echo "=== Phase 1: 20K-step sweep (seed=$seed) ==="
        echo "Start: $(date)"

        # Baselines
        run "adam_lr2.5e-4"     20000 $seed --optimizer adam   --lr 2.5e-4 --wd 0
        run "adamw_lr2.5e-4"    20000 $seed --optimizer adamw  --lr 2.5e-4 --wd 0.02
        run "adam_lr5e-4"       20000 $seed --optimizer adam   --lr 5e-4   --wd 0

        # NS configs (Adam backbone, no WD)
        run "ns100_K2_lr2.5e-4" 20000 $seed --optimizer LAKTJU_NS --lr 2.5e-4 --wd 0 --ns_interval 100 --ns_steps 2 --ns_max_dim 1024
        run "ns50_K2_lr2.5e-4"  20000 $seed --optimizer LAKTJU_NS --lr 2.5e-4 --wd 0 --ns_interval 50  --ns_steps 2 --ns_max_dim 1024
        run "ns100_K2_lr5e-4"   20000 $seed --optimizer LAKTJU_NS --lr 5e-4   --wd 0 --ns_interval 100 --ns_steps 2 --ns_max_dim 1024
        run "ns50_K2_lr5e-4"    20000 $seed --optimizer LAKTJU_NS --lr 5e-4   --wd 0 --ns_interval 50  --ns_steps 2 --ns_max_dim 1024

        echo ""
        echo "=== Phase 1 Results (lower PPL = better) ==="
        for f in "$LOGDIR"/*_seed${seed}.log; do
            tag=$(basename "$f" _seed${seed}.log)
            ppl=$(grep "Best Test PPL" "$f" 2>/dev/null | tail -1 | grep -oP '[\d.]+' | head -1)
            printf "  %-30s %s\n" "$tag" "${ppl:-FAILED}"
        done | sort -t' ' -k2 -n
        echo "Phase 1 done: $(date)"
        ;;

    phase2)
        echo "=== Phase 2: 200K full training (3 seeds) ==="
        echo "Start: $(date)"
        echo "Provide configs as: $0 phase2 <opt_tag> <opt_args...>"
        echo "Example: $0 phase2 adam_lr2.5e-4 --optimizer adam --lr 2.5e-4 --wd 0"
        ;;

    full)
        # Run the full comparison: Adam, AdamW, NS best at 200K steps, 3 seeds
        echo "=== Full TXL Experiments: 200K steps, 3 seeds ==="
        echo "Start: $(date)"

        for SEED in 42 123 456; do
            # Adam baseline
            run "adam_lr2.5e-4"     200000 $SEED --optimizer adam   --lr 2.5e-4 --wd 0
            # AdamW baseline
            run "adamw_lr2.5e-4"    200000 $SEED --optimizer adamw  --lr 2.5e-4 --wd 0.02
            # LAKTJU_NS best config (ns=100, K=2)
            run "ns100_K2_lr2.5e-4" 200000 $SEED --optimizer LAKTJU_NS --lr 2.5e-4 --wd 0 --ns_interval 100 --ns_steps 2 --ns_max_dim 1024
        done

        echo ""
        echo "=== Full Results ==="
        for f in "$LOGDIR"/*.log; do
            tag=$(basename "$f" .log)
            ppl=$(grep "Best Test PPL" "$f" 2>/dev/null | tail -1 | grep -oP '[\d.]+' | head -1)
            printf "  %-40s %s\n" "$tag" "${ppl:-FAILED}"
        done | sort -t' ' -k2 -n
        echo "Full experiments done: $(date)"
        ;;

    single)
        # Run single experiment: ./run_txl_main.sh single <tag> <max_step> <seed> <extra_args...>
        TAG=$2; MS=$3; S=$4; shift 4
        run "$TAG" $MS $S "$@"
        ;;

    *)
        echo "Usage: $0 {phase1|full|single <tag> <max_step> <seed> [args...]}"
        echo ""
        echo "Examples:"
        echo "  $0 phase1              # 20K-step sweep, seed=42"
        echo "  $0 phase1 123          # 20K-step sweep, seed=123"
        echo "  $0 full                # Full 200K × 3 seeds"
        echo "  $0 single adam 200000 42 --optimizer adam --lr 2.5e-4 --wd 0"
        exit 1
        ;;
esac

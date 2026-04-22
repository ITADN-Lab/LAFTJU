#!/bin/bash
# Run Transformer-XL WikiText-103 experiments
# Compares LAKTJU_NS vs Adam vs AdamW vs Adan at 100k steps

SCRIPT_DIR="/home/hadoop/workstation/md/TJU-V5(ATJU)-sourcecode/ATJU/experiments"
PYTHON="/home/hadoop/anaconda3/envs/vllm-env/bin/python"
DATA="../data/wikitext-103"
RESULTS="./results_txl"
SEED=1111
MAX_STEP=100000

cd "$SCRIPT_DIR"

run_exp() {
    local OPT=$1
    local LR=$2
    local WD=$3
    local EXTRA=$4
    local LOG="$RESULTS/txl_wt103_${OPT}_lr${LR}_seed${SEED}.log"

    echo "=== Starting $OPT lr=$LR ===" | tee -a "$LOG"
    $PYTHON train_txl_wt103.py \
        --data "$DATA" \
        --optimizer "$OPT" \
        --lr "$LR" \
        --wd "$WD" \
        --seed "$SEED" \
        --max_step "$MAX_STEP" \
        --work_dir "$RESULTS" \
        $EXTRA \
        2>&1 | tee -a "$LOG"
    echo "=== Done $OPT ===" | tee -a "$LOG"
}

case "$1" in
    adam)
        run_exp adam 0.00025 0.0
        ;;
    adamw)
        run_exp adamw 0.00025 0.02
        ;;
    adan)
        run_exp adan 0.0015 0.02
        ;;
    laktju_ns)
        run_exp LAKTJU_NS 0.00025 0.0 "--ns_interval 100 --ns_steps 2 --ns_max_dim 1024"
        ;;
    all)
        run_exp adam 0.00025 0.0
        run_exp adamw 0.00025 0.02
        run_exp adan 0.0015 0.02
        run_exp LAKTJU_NS 0.00025 0.0 "--ns_interval 100 --ns_steps 2 --ns_max_dim 1024"
        ;;
    *)
        echo "Usage: $0 {adam|adamw|adan|laktju_ns|all}"
        exit 1
        ;;
esac

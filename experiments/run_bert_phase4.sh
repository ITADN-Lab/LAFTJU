#!/bin/bash
# Phase 4: Aggressive parallel tuning to beat AdamW on ALL GLUE tasks
# Strategy: task-specific NS tuning + higher ns_interval + attention-only NS
set -e
cd "$(dirname "$0")"

MAX_JOBS=6
SEEDS="42 123 456"
LOGDIR="logs_bert_phase4"
mkdir -p "$LOGDIR"

run_job() {
    local cmd="$1"
    local logfile="$2"
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
        sleep 5
    done
    echo "[$(date +%H:%M:%S)] START: $logfile"
    eval "$cmd" > "$LOGDIR/$logfile" 2>&1 &
}

echo "=== Phase 4: Task-specific NS tuning ==="
echo "Start: $(date)"

# ─── Strategy 1: Higher ns_interval (500, 1000) — gentler NS ───
for ns_int in 500 1000; do
    for task in cola rte qnli mnli qqp sst2 stsb; do
        for seed in $SEEDS; do
            run_job "python3 train_bert_glue.py --task $task --optimizer LAKTJU_NS --seed $seed \
                     --epochs 10 --ns_interval $ns_int --ns_steps 1 --ns_max_dim 3072" \
                    "${task}_ns${ns_int}_K1_seed${seed}.log"
        done
    done
done

# ─── Strategy 2: Attention-only NS (skip FFN layers) ───
for ns_int in 200 500; do
    for task in cola rte qnli mnli qqp sst2 stsb; do
        for seed in $SEEDS; do
            run_job "python3 train_bert_glue.py --task $task --optimizer LAKTJU_NS --seed $seed \
                     --epochs 10 --ns_interval $ns_int --ns_steps 1 --ns_max_dim 768 \
                     --ns_skip_ffn" \
                    "${task}_ns${ns_int}_K1_attn_seed${seed}.log"
        done
    done
done

# ─── Strategy 3: NS with warmup (skip first 10% of training) ───
for task in cola rte qnli mnli qqp sst2 stsb; do
    for seed in $SEEDS; do
        run_job "python3 train_bert_glue.py --task $task --optimizer LAKTJU_NS --seed $seed \
                 --epochs 10 --ns_interval 500 --ns_steps 1 --ns_max_dim 3072 \
                 --ns_warmup_frac 0.1" \
                "${task}_ns500_K1_warm10_seed${seed}.log"
    done
done

# ─── Strategy 4: NS with warmup + cooldown ───
for task in cola rte qnli mnli qqp sst2 stsb; do
    for seed in $SEEDS; do
        run_job "python3 train_bert_glue.py --task $task --optimizer LAKTJU_NS --seed $seed \
                 --epochs 10 --ns_interval 500 --ns_steps 1 --ns_max_dim 3072 \
                 --ns_warmup_frac 0.1 --ns_cooldown_frac 0.1" \
                "${task}_ns500_K1_warm10_cool10_seed${seed}.log"
    done
done

# ─── Strategy 5: Very gentle NS (ns=2000) ───
for task in cola rte qnli mnli qqp sst2 stsb; do
    for seed in $SEEDS; do
        run_job "python3 train_bert_glue.py --task $task --optimizer LAKTJU_NS --seed $seed \
                 --epochs 10 --ns_interval 2000 --ns_steps 1 --ns_max_dim 3072" \
                "${task}_ns2000_K1_seed${seed}.log"
    done
done

# ─── Strategy 6: LR fine-tuning for weak tasks ───
# CoLA: try lr=1.5e-5 (lower lr might help Matthew's corr)
for seed in $SEEDS; do
    run_job "python3 train_bert_glue.py --task cola --optimizer LAKTJU_NS --seed $seed \
             --epochs 10 --lr 1.5e-5 --ns_interval 500 --ns_steps 1 --ns_max_dim 3072" \
            "cola_ns500_K1_lr1.5e-5_seed${seed}.log"
    run_job "python3 train_bert_glue.py --task cola --optimizer LAKTJU_NS --seed $seed \
             --epochs 10 --lr 2.5e-5 --ns_interval 500 --ns_steps 1 --ns_max_dim 3072" \
            "cola_ns500_K1_lr2.5e-5_seed${seed}.log"
done

# RTE: try different lr
for seed in $SEEDS; do
    run_job "python3 train_bert_glue.py --task rte --optimizer LAKTJU_NS --seed $seed \
             --epochs 10 --lr 1.5e-5 --ns_interval 500 --ns_steps 1 --ns_max_dim 3072" \
            "rte_ns500_K1_lr1.5e-5_seed${seed}.log"
    run_job "python3 train_bert_glue.py --task rte --optimizer LAKTJU_NS --seed $seed \
             --epochs 10 --lr 3e-5 --ns_interval 500 --ns_steps 1 --ns_max_dim 3072" \
            "rte_ns500_K1_lr3e-5_seed${seed}.log"
done

echo "[$(date +%H:%M:%S)] All jobs submitted. Waiting..."
wait
echo "[$(date +%H:%M:%S)] All jobs complete!"
echo "End: $(date)"

# ─── Collect results ───
echo ""
echo "=== Phase 4 Results ==="
for logfile in "$LOGDIR"/*.log; do
    name=$(basename "$logfile" .log)
    best=$(grep "best=" "$logfile" 2>/dev/null | tail -1 | sed 's/.*best=\([0-9.-]*\).*/\1/')
    if [ -n "$best" ]; then
        echo "  $name: $best"
    else
        echo "  $name: FAILED"
    fi
done | sort

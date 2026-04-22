#!/bin/bash
# Phase 3 Parallel: Run 6 concurrent BERT GLUE experiments on RTX 5090 (32GB)
# Each process uses ~4GB, so 6 concurrent = ~24GB (safe margin)
set -e
cd "$(dirname "$0")"

MAX_JOBS=6
SEEDS="42 123 456"
NS_INT=200
NS_K=2
NS_MAXD=3072
LOGDIR="logs_bert_phase3"
mkdir -p "$LOGDIR"

job_count=0

run_job() {
    local cmd="$1"
    local logfile="$2"

    # Wait if we have MAX_JOBS running
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
        sleep 5
    done

    echo "[$(date +%H:%M:%S)] START: $logfile"
    eval "$cmd" > "$LOGDIR/$logfile" 2>&1 &
    job_count=$((job_count + 1))
}

echo "=== Phase 3 Parallel: 6 concurrent jobs on 32GB GPU ==="
echo "Start time: $(date)"

# ─── Group 1: AdamW baselines for remaining tasks ───
# Already have: sst2, cola, qnli, rte, qqp (from Phase 3 serial)
# Need: mnli, stsb for AdamW
for task in mnli stsb; do
    for seed in $SEEDS; do
        run_job "python3 train_bert_glue.py --task $task --optimizer AdamW --seed $seed --epochs 10" \
                "${task}_AdamW_seed${seed}.log"
    done
done

# ─── Group 2: LAFTJU-NS (ns=200, K=2, lr=2e-5) for ALL tasks ───
for task in sst2 cola qnli rte qqp mnli stsb; do
    for seed in $SEEDS; do
        run_job "python3 train_bert_glue.py --task $task --optimizer LAKTJU_NS --seed $seed \
                 --epochs 10 --ns_interval $NS_INT --ns_steps $NS_K --ns_max_dim $NS_MAXD" \
                "${task}_LAKTJU_NS_ns${NS_INT}_K${NS_K}_seed${seed}.log"
    done
done

# ─── Group 3: LAFTJU-NS lr=3e-5 sweep (SST-2 already done, add other tasks) ───
for task in cola qnli rte mnli stsb; do
    for seed in $SEEDS; do
        run_job "python3 train_bert_glue.py --task $task --optimizer LAKTJU_NS --seed $seed \
                 --epochs 10 --lr 3e-5 --ns_interval $NS_INT --ns_steps $NS_K --ns_max_dim $NS_MAXD" \
                "${task}_LAKTJU_NS_ns${NS_INT}_K${NS_K}_lr3e-5_seed${seed}.log"
    done
done

# ─── Group 4: Additional NS configs for diversity ───
# ns=100 K=2 on key tasks
for task in cola qnli rte mnli; do
    for seed in $SEEDS; do
        run_job "python3 train_bert_glue.py --task $task --optimizer LAKTJU_NS --seed $seed \
                 --epochs 10 --ns_interval 100 --ns_steps 2 --ns_max_dim $NS_MAXD" \
                "${task}_LAKTJU_NS_ns100_K2_seed${seed}.log"
    done
done

# ─── Group 5: AdamW lr=3e-5 sweep for fair comparison ───
for task in sst2 cola qnli rte mnli stsb; do
    for seed in $SEEDS; do
        run_job "python3 train_bert_glue.py --task $task --optimizer AdamW --seed $seed \
                 --epochs 10 --lr 3e-5" \
                "${task}_AdamW_lr3e-5_seed${seed}.log"
    done
done

# Wait for all jobs to finish
echo "[$(date +%H:%M:%S)] All jobs submitted ($job_count total). Waiting..."
wait
echo "[$(date +%H:%M:%S)] All jobs complete!"
echo "End time: $(date)"

# ─── Collect results ───
echo ""
echo "=== Results Summary ==="
for logfile in "$LOGDIR"/*.log; do
    name=$(basename "$logfile" .log)
    result=$(grep "Best val" "$logfile" 2>/dev/null | tail -1)
    if [ -n "$result" ]; then
        echo "  $name: $result"
    else
        echo "  $name: FAILED (check $logfile)"
    fi
done | sort

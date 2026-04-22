#!/bin/bash
# Phase 5: Final push — improve RTE and STS-B to beat AdamW decisively
# RTE: Currently tied with AdamW (69.56), need to win
# STS-B: LAKTJU-NS 89.42 barely beats AdamW 89.41, need bigger margin
# Also: STS-B LAKTJU-NS with lr=3e-5 + better NS configs
set -e
cd "$(dirname "$0")"

MAX_JOBS=6
SEEDS="42 123 456"
LOGDIR="logs_bert_phase5"
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

echo "=== Phase 5: RTE & STS-B improvement ==="
echo "Start: $(date)"

# ─── RTE: Extensive search (small dataset, high variance) ───
# Strategy: More LR values, more NS configs, longer training

# RTE with lr=1e-5 (lower lr for small dataset)
for ns_int in 500 1000 2000; do
    for seed in $SEEDS; do
        run_job "python3 train_bert_glue.py --task rte --optimizer LAKTJU_NS --seed $seed \
                 --epochs 15 --lr 1e-5 --ns_interval $ns_int --ns_steps 1 --ns_max_dim 3072" \
                "rte_ns${ns_int}_K1_lr1e-5_ep15_seed${seed}.log"
    done
done

# RTE with lr=1.5e-5
for ns_int in 500 1000; do
    for seed in $SEEDS; do
        run_job "python3 train_bert_glue.py --task rte --optimizer LAKTJU_NS --seed $seed \
                 --epochs 15 --lr 1.5e-5 --ns_interval $ns_int --ns_steps 1 --ns_max_dim 3072" \
                "rte_ns${ns_int}_K1_lr1.5e-5_ep15_seed${seed}.log"
    done
done

# RTE with lr=4e-5 (higher lr)
for ns_int in 500 1000; do
    for seed in $SEEDS; do
        run_job "python3 train_bert_glue.py --task rte --optimizer LAKTJU_NS --seed $seed \
                 --epochs 15 --lr 4e-5 --ns_interval $ns_int --ns_steps 1 --ns_max_dim 3072" \
                "rte_ns${ns_int}_K1_lr4e-5_ep15_seed${seed}.log"
    done
done

# RTE with attention-only NS
for ns_int in 500 1000; do
    for seed in $SEEDS; do
        run_job "python3 train_bert_glue.py --task rte --optimizer LAKTJU_NS --seed $seed \
                 --epochs 15 --lr 3e-5 --ns_interval $ns_int --ns_steps 1 --ns_max_dim 768 --ns_skip_ffn" \
                "rte_ns${ns_int}_K1_attn_lr3e-5_ep15_seed${seed}.log"
    done
done

# RTE with warmup (10%, 20%) — let model stabilize before NS
for warmup in 0.1 0.2; do
    for seed in $SEEDS; do
        run_job "python3 train_bert_glue.py --task rte --optimizer LAKTJU_NS --seed $seed \
                 --epochs 15 --lr 3e-5 --ns_interval 500 --ns_steps 1 --ns_max_dim 3072 \
                 --ns_warmup_frac $warmup" \
                "rte_ns500_K1_lr3e-5_warm${warmup}_ep15_seed${seed}.log"
    done
done

# RTE with AdamW lr=3e-5 ep15 (fair comparison at more epochs)
for seed in $SEEDS; do
    run_job "python3 train_bert_glue.py --task rte --optimizer AdamW --seed $seed \
             --epochs 15 --lr 3e-5" \
            "rte_AdamW_lr3e-5_ep15_seed${seed}.log"
done

# RTE with AdamW lr=2e-5 ep15
for seed in $SEEDS; do
    run_job "python3 train_bert_glue.py --task rte --optimizer AdamW --seed $seed \
             --epochs 15 --lr 2e-5" \
            "rte_AdamW_lr2e-5_ep15_seed${seed}.log"
done

# ─── STS-B: lr=3e-5 with better NS configs ───
for ns_int in 500 1000 2000; do
    for seed in $SEEDS; do
        run_job "python3 train_bert_glue.py --task stsb --optimizer LAKTJU_NS --seed $seed \
                 --epochs 10 --lr 3e-5 --ns_interval $ns_int --ns_steps 1 --ns_max_dim 3072" \
                "stsb_ns${ns_int}_K1_lr3e-5_seed${seed}.log"
    done
done

# STS-B: lr=3e-5 attention-only
for ns_int in 500 1000; do
    for seed in $SEEDS; do
        run_job "python3 train_bert_glue.py --task stsb --optimizer LAKTJU_NS --seed $seed \
                 --epochs 10 --lr 3e-5 --ns_interval $ns_int --ns_steps 1 --ns_max_dim 768 --ns_skip_ffn" \
                "stsb_ns${ns_int}_K1_attn_lr3e-5_seed${seed}.log"
    done
done

# STS-B: lr=4e-5 (push lr higher)
for ns_int in 500 1000; do
    for seed in $SEEDS; do
        run_job "python3 train_bert_glue.py --task stsb --optimizer LAKTJU_NS --seed $seed \
                 --epochs 10 --lr 4e-5 --ns_interval $ns_int --ns_steps 1 --ns_max_dim 3072" \
                "stsb_ns${ns_int}_K1_lr4e-5_seed${seed}.log"
    done
done

# STS-B: AdamW lr=4e-5 (fair comparison)
for seed in $SEEDS; do
    run_job "python3 train_bert_glue.py --task stsb --optimizer AdamW --seed $seed \
             --epochs 10 --lr 4e-5" \
            "stsb_AdamW_lr4e-5_seed${seed}.log"
done

echo "[$(date +%H:%M:%S)] All jobs submitted. Waiting..."
wait
echo "[$(date +%H:%M:%S)] All jobs complete!"
echo "End: $(date)"

# ─── Collect results ───
echo ""
echo "=== Phase 5 Results ==="
for logfile in "$LOGDIR"/*.log; do
    name=$(basename "$logfile" .log)
    best=$(grep "Best val" "$logfile" 2>/dev/null | tail -1)
    if [ -n "$best" ]; then
        echo "  $name: $best"
    else
        echo "  $name: FAILED"
    fi
done | sort

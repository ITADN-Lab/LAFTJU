#!/bin/bash
# Phase 1: SST-2 quick validation — Adam, AdamW, LAFTJU-NS × 3 seeds
cd "$(dirname "$0")"

SEEDS="42 123 456"

echo "=== Phase 1: SST-2 Baselines ==="

# Adam baselines
for seed in $SEEDS; do
    echo "Running: Adam seed=$seed"
    python3 train_bert_glue.py --task sst2 --optimizer Adam --seed $seed --epochs 10 2>&1 | grep "Best val"
done

# AdamW baselines
for seed in $SEEDS; do
    echo "Running: AdamW seed=$seed"
    python3 train_bert_glue.py --task sst2 --optimizer AdamW --seed $seed --epochs 10 2>&1 | grep "Best val"
done

# LAFTJU-NS: sweep ns_interval and ns_steps
for ns_int in 50 100 200; do
    for ns_k in 1 2; do
        echo "Running: LAKTJU_NS ns=$ns_int K=$ns_k seed=42"
        python3 train_bert_glue.py --task sst2 --optimizer LAKTJU_NS --seed 42 \
            --epochs 10 --ns_interval $ns_int --ns_steps $ns_k --ns_max_dim 3072 \
            2>&1 | grep "Best val"
    done
done

# Best NS config with remaining seeds
echo "=== Phase 1b: Best NS config × 3 seeds ==="
for seed in 123 456; do
    echo "Running: LAKTJU_NS ns=100 K=1 seed=$seed"
    python3 train_bert_glue.py --task sst2 --optimizer LAKTJU_NS --seed $seed \
        --epochs 10 --ns_interval 100 --ns_steps 1 --ns_max_dim 3072 \
        2>&1 | grep "Best val"
done

echo "=== Phase 1 Complete ==="

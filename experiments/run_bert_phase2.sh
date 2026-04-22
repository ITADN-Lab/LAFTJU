#!/bin/bash
# Phase 2: Confirm best NS configs + fix Adam baseline
cd "$(dirname "$0")"

SEEDS="42 123 456"

echo "=== Phase 2a: LAFTJU-NS ns=200 K=1 × 3 seeds ==="
for seed in $SEEDS; do
    echo "Running: LAKTJU_NS ns=200 K=1 seed=$seed"
    python3 train_bert_glue.py --task sst2 --optimizer LAKTJU_NS --seed $seed \
        --epochs 10 --ns_interval 200 --ns_steps 1 --ns_max_dim 3072 \
        2>&1 | grep "Best val"
done

echo "=== Phase 2b: LAFTJU-NS ns=200 K=2 × 3 seeds ==="
for seed in $SEEDS; do
    echo "Running: LAKTJU_NS ns=200 K=2 seed=$seed"
    python3 train_bert_glue.py --task sst2 --optimizer LAKTJU_NS --seed $seed \
        --epochs 10 --ns_interval 200 --ns_steps 2 --ns_max_dim 3072 \
        2>&1 | grep "Best val"
done

echo "=== Phase 2c: Adam with higher lr (2e-5, wd=0.01) ==="
for seed in $SEEDS; do
    echo "Running: Adam lr=2e-5 wd=0.01 seed=$seed"
    python3 train_bert_glue.py --task sst2 --optimizer Adam --seed $seed \
        --epochs 10 --lr 2e-5 --weight_decay 0.01 \
        2>&1 | grep "Best val"
done

echo "=== Phase 2d: LAFTJU-NS ns=500 K=1 × 3 seeds ==="
for seed in $SEEDS; do
    echo "Running: LAKTJU_NS ns=500 K=1 seed=$seed"
    python3 train_bert_glue.py --task sst2 --optimizer LAKTJU_NS --seed $seed \
        --epochs 10 --ns_interval 500 --ns_steps 1 --ns_max_dim 3072 \
        2>&1 | grep "Best val"
done

echo "=== Phase 2 Complete ==="

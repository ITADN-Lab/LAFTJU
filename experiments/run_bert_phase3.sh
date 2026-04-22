#!/bin/bash
# Phase 3: All GLUE tasks with best NS config + lr sweep on SST-2
cd "$(dirname "$0")"

SEEDS="42 123 456"
NS_INT=200
NS_K=2
NS_MAXD=3072

# Phase 3a: LR sweep on SST-2 for LAFTJU-NS
echo "=== Phase 3a: LR sweep on SST-2 ==="
for lr in 3e-5 5e-5; do
    for seed in $SEEDS; do
        echo "Running: LAKTJU_NS lr=$lr ns=$NS_INT K=$NS_K seed=$seed"
        python3 train_bert_glue.py --task sst2 --optimizer LAKTJU_NS --seed $seed \
            --epochs 10 --lr $lr --ns_interval $NS_INT --ns_steps $NS_K --ns_max_dim $NS_MAXD \
            2>&1 | grep "Best val"
    done
done

# Phase 3b: All GLUE tasks — AdamW baseline
echo "=== Phase 3b: AdamW on all GLUE tasks ==="
for task in cola qnli rte qqp mnli stsb; do
    for seed in $SEEDS; do
        echo "Running: AdamW task=$task seed=$seed"
        python3 train_bert_glue.py --task $task --optimizer AdamW --seed $seed \
            --epochs 10 2>&1 | grep "Best val"
    done
done

# Phase 3c: All GLUE tasks — LAFTJU-NS (ns=200, K=2)
echo "=== Phase 3c: LAFTJU-NS on all GLUE tasks ==="
for task in cola qnli rte qqp mnli stsb; do
    for seed in $SEEDS; do
        echo "Running: LAKTJU_NS task=$task ns=$NS_INT K=$NS_K seed=$seed"
        python3 train_bert_glue.py --task $task --optimizer LAKTJU_NS --seed $seed \
            --epochs 10 --ns_interval $NS_INT --ns_steps $NS_K --ns_max_dim $NS_MAXD \
            2>&1 | grep "Best val"
    done
done

echo "=== Phase 3 Complete ==="

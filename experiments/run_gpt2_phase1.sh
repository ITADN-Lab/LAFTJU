#!/bin/bash
# GPT-2 Phase 1: Quick feasibility test + initial baseline
# Test VRAM usage, throughput, and basic convergence
set -e
cd "$(dirname "$0")"

LOGDIR="logs_gpt2_phase1"
mkdir -p "$LOGDIR"

echo "=== GPT-2 Phase 1: Feasibility Test ==="
echo "Start: $(date)"

# ─── Test 1: Quick smoke test (100 steps) to verify everything works ───
echo "[$(date +%H:%M:%S)] Running smoke test..."
python3 train_gpt2_lm.py \
    --model_size small \
    --optimizer AdamW \
    --total_steps 100 \
    --eval_interval 50 \
    --log_interval 25 \
    --batch_size 8 \
    --grad_accum 4 \
    --seed 42 \
    --dataset openwebtext \
    > "$LOGDIR/smoke_test.log" 2>&1

if [ $? -eq 0 ]; then
    echo "[$(date +%H:%M:%S)] Smoke test PASSED!"
    tail -5 "$LOGDIR/smoke_test.log"
else
    echo "[$(date +%H:%M:%S)] Smoke test FAILED!"
    tail -20 "$LOGDIR/smoke_test.log"
    exit 1
fi

echo ""
echo "=== Phase 1 complete ==="
echo "End: $(date)"

#!/bin/bash
# Worker script: pull tasks from git, run experiments, push results back.
# Usage: cd LAFTJU/experiments && bash task_queue/worker.sh
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
TASK_DIR="task_queue/tasks"
WORKER_ID="$(hostname)-$$"

echo "=== Experiment Worker: $WORKER_ID ==="
echo "Repo: $REPO_ROOT"

claim_task() {
    local f="$1"
    python3 -c "
import json
with open('$f') as fh: t = json.load(fh)
if t['status'] != 'pending': exit(1)
t['status'] = 'running'
t['worker'] = '$WORKER_ID'
with open('$f', 'w') as fh: json.dump(t, fh, indent=2)
"
}

mark_done() {
    local f="$1" result="$2"
    python3 -c "
import json
with open('$f') as fh: t = json.load(fh)
t['status'] = 'done'
t['result_file'] = '$result'
with open('$f', 'w') as fh: json.dump(t, fh, indent=2)
"
}

while true; do
    echo "[$(date +%H:%M:%S)] Pulling latest tasks..."
    git pull --rebase origin main 2>/dev/null || git pull origin main

    FOUND=0
    for task_file in "$TASK_DIR"/*.json; do
        [ -f "$task_file" ] || continue
        STATUS=$(python3 -c "import json; print(json.load(open('$task_file'))['status'])")
        [ "$STATUS" = "pending" ] || continue

        TASK_ID=$(python3 -c "import json; print(json.load(open('$task_file'))['id'])")
        CMD=$(python3 -c "import json; print(json.load(open('$task_file'))['cmd'])")

        echo "[$(date +%H:%M:%S)] Claiming task: $TASK_ID"
        claim_task "$task_file" || continue

        # Push claim to prevent other workers from taking it
        git add "$task_file"
        git commit -m "worker $WORKER_ID: claim $TASK_ID" 2>/dev/null
        git push origin main 2>/dev/null || {
            echo "Push conflict, resetting claim..."
            git pull --rebase origin main
            continue
        }

        FOUND=1
        echo "[$(date +%H:%M:%S)] Running: $CMD"
        LOG="task_queue/logs/${TASK_ID}.log"
        mkdir -p task_queue/logs

        if eval "$CMD" > "$LOG" 2>&1; then
            echo "[$(date +%H:%M:%S)] Task $TASK_ID completed successfully"
        else
            echo "[$(date +%H:%M:%S)] Task $TASK_ID failed (exit=$?)"
        fi

        # Find the result file
        RESULT=$(grep "Saved to" "$LOG" 2>/dev/null | tail -1 | sed 's/.*Saved to //' || echo "")

        mark_done "$task_file" "$RESULT"

        # Commit and push results
        git add "$task_file" "$LOG" results_gpt2/ 2>/dev/null
        git commit -m "worker $WORKER_ID: done $TASK_ID" 2>/dev/null
        git push origin main 2>/dev/null || {
            git pull --rebase origin main
            git push origin main
        }

        echo "[$(date +%H:%M:%S)] Results pushed for $TASK_ID"
        break  # Re-pull to check for new tasks
    done

    if [ "$FOUND" -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] No pending tasks. Waiting 60s..."
        sleep 60
    fi
done

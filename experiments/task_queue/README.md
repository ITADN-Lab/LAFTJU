# Distributed Experiment Task Queue

Git-based task distribution for running experiments across multiple machines.

## How it works

1. **Dispatcher** (this machine): creates task JSON files in `tasks/` and pushes to GitHub
2. **Worker** (remote machine): pulls tasks, runs them, commits results back
3. **Dispatcher**: pulls to collect results

## Usage

### On the dispatcher (this machine):
```bash
# Create tasks for remote execution
python3 create_tasks.py

# Collect results after workers finish
git pull origin main
```

### On the worker (remote machine):
```bash
# Clone the repo
git clone git@github.com:ITADN-Lab/LAFTJU.git
cd LAFTJU/experiments

# Install dependencies
pip install lion-pytorch transformers datasets

# Run the worker
bash worker.sh
```

## Task format
Each task is a JSON file in `tasks/`:
- `status`: "pending" → "running" → "done"
- `worker`: hostname of the machine that claimed it
- `result_file`: path to result JSON after completion

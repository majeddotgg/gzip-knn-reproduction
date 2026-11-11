#!/usr/bin/env bash
# Experiment 1: full-training-set distance matrices (full test sets; SwahiliNews sampled, see below).
# CPU only. Set JOBS to the number of processes (default: all cores); the
# matrices are identical for any JOBS value. About 2 CPU-hours in total
# (roughly 20 minutes wall-clock on 10 cores).
set -e
cd "$(dirname "$0")/.."
JOBS=${JOBS:-$(python -c "import os; print(os.cpu_count())")}
for d in DengueFilipino KirundiNews R8 KinyarwandaNews; do
  python src/run_full.py "$d" --jobs "$JOBS"
done
# SwahiliNews: fixed 300-document test sample (seed 0). The full 7,338-document test set
# against 22,207 training texts needs about 4 CPU-hours; use --test-sample 0 to run it.
python src/run_full.py SwahiliNews --test-sample 300 --seed 0 --jobs "$JOBS"

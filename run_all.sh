#!/usr/bin/env bash
# One-command reproduction: data -> tests -> all experiments -> tables/figures.
# CPU only. About 1 hour on a single core (see README for per-step timings).
set -e
cd "$(dirname "$0")"
pip install -r requirements.txt
bash scripts/download_data.sh
python -m pytest -q tests
bash scripts/run_full_all.sh
bash scripts/run_rest.sh

#!/usr/bin/env bash
# One-command reproduction: environment -> data -> tests -> all experiments -> tables/figures.
# CPU only. Creates its own virtual environment (.venv) so nothing is installed globally.
# Needs Python >= 3.12 (set PYTHON=/path/to/python3.12 if python3 is older).
# Experiment 1 uses all cores by default (set JOBS=N): about 20 minutes on 10 cores.
# The BERT/mBERT step takes about 1 hour on a GPU (CUDA or Apple MPS), several hours on CPU only.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else "Python >= 3.12 required (set PYTHON=...)")'
"$PY" -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_data.sh
python -m pytest -q tests
bash scripts/run_full_all.sh
bash scripts/run_rest.sh
bash scripts/run_bert.sh

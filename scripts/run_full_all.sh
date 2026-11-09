#!/usr/bin/env bash
# Experiment 1: full-training-set distance matrices (CPU only, ~1 h on one core)
set -e
cd "$(dirname "$0")/.."
python src/run_full.py DengueFilipino
python src/run_full.py KirundiNews
python src/run_full.py R8
python src/run_full.py KinyarwandaNews --test-sample 600 --seed 0
python src/run_full.py SwahiliNews --test-sample 300 --seed 0

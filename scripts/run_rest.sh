#!/usr/bin/env bash
# Experiments 2-5 and the summary tables/figures (run after run_full_all.sh)
set -e
cd "$(dirname "$0")/.."
python src/run_fewshot.py
python src/run_logs.py
python src/audit_duplicates.py
python src/evaluate_full.py
python src/error_breakdown.py
python src/stats.py
python src/make_figures.py

#!/usr/bin/env bash
# Neural baselines re-run by us: BERT and mBERT, 5-shot (4 OOD datasets x 5 trials)
# and full data (KirundiNews, DengueFilipino), then paired comparisons with gzip.
# Uses a GPU (CUDA or Apple MPS) if present: about 1 hour on an Apple M-series GPU.
# On CPU only it works but is much slower (several hours).
# 5-shot runs are scored on a fixed 500-document test sample for KirundiNews, KinyarwandaNews
# and SwahiliNews (all 346 DengueFilipino items); gzip and TF-IDF are re-scored on the same items.
# Run after run_full_all.sh and run_rest.sh (it needs cache/ and results/fewshot.json).
set -e
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false
python src/run_bert.py fewshot --eval-sample 500
python src/run_bert.py full
python src/bert_compare.py

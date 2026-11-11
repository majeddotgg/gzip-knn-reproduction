# Reproducing "gzip + kNN" text classification (Jiang et al., Findings of ACL 2023)

Reproduction study for AI504 Assignment 2. It re-implements the parameter-free
classifier of

> Jiang, Z., Yang, M., Tsirlin, M., Tang, R., Dai, Y., & Lin, J. (2023).
> "Low-Resource" Text Classification: A Parameter-Free Classification Method with
> Compressors. *Findings of ACL 2023*, 6810–6828. https://doi.org/10.18653/v1/2023.findings-acl.426

and checks whether its central claim (gzip + NCD + kNN beats BERT on
low-resource / out-of-distribution data) holds under a valid evaluation.

## What is in here

| Path | Purpose |
|---|---|
| `src/ncd_knn.py` | Our re-implementation: gzip length, NCD (paper Eq. 3), cached distance matrix, kNN with three tie rules (`optimistic` = original code, `random`, `nearest`), bootstrap CIs |
| `src/data.py` | Loaders for KirundiNews, KinyarwandaNews, SwahiliNews, DengueFilipino, R8 and Loghub BGL_2k |
| `src/baselines.py` | Trained baseline: character 2–4-gram TF-IDF + logistic regression |
| `src/run_full.py` | Experiment 1: full-training-set NCD matrices (cached in `cache/`); `--jobs N` parallelises with identical results |
| `src/evaluate_full.py` | Experiments 1, 3, 4: accuracy under each rule, k-sensitivity, train/test duplicate audit, error dump |
| `src/run_fewshot.py` | Experiment 2: 5-shot, 5 random trials |
| `src/audit_duplicates.py` | Experiment 4: exact train/test duplicates and label conflicts (`results/duplicates.json`) |
| `src/run_logs.py` | Experiment 5 (extension): alert detection on BGL logs, random vs template-disjoint splits |
| `src/error_breakdown.py` | Quantitative error analysis: accuracy by text-length tercile and most frequent confusion (`results/error_breakdown.json`) |
| `src/run_bert.py` | Neural baselines re-run by us: BERT and mBERT fine-tuning, 5-shot (paired with gzip's samples) and full data |
| `src/bert_compare.py` | Paired t-tests (5-shot) and McNemar tests (full data) of BERT/mBERT against gzip and TF-IDF+LR (`results/bert_compare.json`) |
| `src/stats.py` | Bootstrap 95% CIs, paired gap intervals, exact McNemar tests (gzip k=1 vs TF-IDF) and BGL mean ± SD (`results/stats.json`) |
| `src/make_figures.py` | Writes `results/summary_tables.md`, `figures/fig1_k_sensitivity.png` (Figure 1 of the report) and `figures/fig2_published_vs_valid.png` (Figure 2) |
| `tests/test_ncd_knn.py` | 8 unit tests, incl. a check that our `optimistic` rule gives exactly the accuracy of the original `calc_acc()`, the random tie rule, and parallel = serial NCD |
| `third_party/npc_gzip_original/` | Two unmodified MIT-licensed files from the authors' repo, used only by that test |
| `results/` | All numbers reported in the paper (JSON + Markdown tables) |

## How to run

Requirements: Python ≥ 3.12 (numpy 2.5 needs it). gzip experiments are CPU only (~20 min on 10 cores,
about 2 CPU-hours); the BERT/mBERT re-runs use a GPU if present (CUDA or Apple MPS, ~40 min).

```bash
bash run_all.sh            # creates .venv, installs pinned deps, downloads data, runs tests and every experiment
```

or step by step:

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_data.sh      # pinned public data from Hugging Face, TextGCN and Loghub, checksum-verified
python -m pytest -q tests          # 8 tests
bash scripts/run_full_all.sh       # Exp. 1 distance matrices (slowest step; JOBS=N to set processes)
bash scripts/run_rest.sh           # Exp. 2-5, error breakdown, statistics, tables and figures
bash scripts/run_bert.sh           # BERT/mBERT re-runs and paired tests
```

Measured wall-clock times with 10 processes: DengueFilipino 4 s, KirundiNews 51 s, R8 65 s,
KinyarwandaNews (full 4,254-document test set) ~20 min, SwahiliNews (300-document sample) ~4 min;
5-shot runs ~5 min, BGL < 1 min.

## Settings used

* gzip level 9 (Python default), texts joined with a single space before compression.
* k ∈ {1, 2, 3, 5, 7}; random tie-breaking seeds 0–4; 5-shot samples drawn with seeds 0–4.
* TF-IDF+LR: character 2–4-grams (`char_wb`), sublinear tf, ≤200,000 features, LogisticRegression(C=10, max_iter=2000); not tuned.
* Bootstrap: 1,000 percentile resamples, seed 0. 5-shot intervals: t-intervals over 5 trials (t = 2.776).
* BGL: StratifiedShuffleSplit (random) and GroupShuffleSplit by EventId (template-disjoint), test_size 0.2, seeds 0–4.
* Tested with Python 3.12; exact package versions are pinned in `requirements.txt`.
* Data sources are pinned to fixed revisions (Hugging Face parquet commits, GitHub commit SHAs) and
  checked against `data/SHA256SUMS`, so a rerun uses exactly the data behind `results/`.

## Notes on choices (read before comparing numbers)

* **Tie-breaking.** The authors' `calc_acc(rand=False)` counts a prediction correct
  if the gold label is *among* the tied labels. With k = 2 that is top-2 accuracy.
  We report it (`k2_optimistic`) only to reproduce the published tables; valid
  numbers are `k1`, `k2_random` and `k2_nearest`.
* **Test sets.** All datasets use their full test sets except SwahiliNews, which uses a
  fixed 300-document random sample (seed 0): its full 7,338 × 22,207 distance matrix needs
  about 4 CPU-hours (`python src/run_full.py SwahiliNews --jobs N` runs it).
* **BERT/mBERT re-runs** (`run_bert.sh`): bert-base-uncased and bert-base-multilingual-cased,
  128 tokens, untuned hyperparameters fixed in `src/run_bert.py`. 5-shot trials reuse gzip's
  training samples and are scored on a fixed 500-document test sample when the test set is
  larger (gzip and TF-IDF are re-scored on the same items in `bert_compare.py`). Full-data
  BERT covers KirundiNews and DengueFilipino. GPU training is not bit-for-bit deterministic.
* **DengueFilipino** uses the official test split; documents with no positive label
  are dropped, as in the original loader (3,181 train / 346 test).
* **R8** uses the TextGCN release of the standard split (pre-cleaned text).
* No text preprocessing is applied before compression, as in the paper.

## Best-practice notes from this reproduction

1. Never let the evaluation function see the gold label when breaking ties.
2. Check exact train/test duplicates before trusting kNN-style results:
   86% of KirundiNews test documents occur verbatim in its training set.
3. Report a cheap trained baseline (here TF-IDF+LR, seconds on CPU) next to any
   "no training needed" claim.

## Headline results (see `results/summary_tables.md`)

| Dataset | Paper gzip | Ours, optimistic k=2 | Ours, k=1 (valid) | TF-IDF+LR |
|---|---|---|---|---|
| KinyarwandaNews | 0.891 | 0.890 | 0.832 | 0.892 |
| KirundiNews | 0.905 | 0.927 | 0.892 | 0.901 |
| DengueFilipino | 0.998 | 0.939 | 0.864 | 0.988 |
| SwahiliNews (300) | 0.927 | 0.943 | 0.857 | 0.893 |
| R8 | 0.954 | 0.952 | 0.915 | 0.967 |

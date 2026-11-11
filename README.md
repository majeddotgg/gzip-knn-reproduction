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
| `src/run_full.py` | Experiment 1: full-training-set NCD matrices (cached in `cache/`) |
| `src/evaluate_full.py` | Experiments 1, 3, 4: accuracy under each rule, k-sensitivity, train/test duplicate audit, error dump |
| `src/run_fewshot.py` | Experiment 2: 5-shot, 5 random trials |
| `src/audit_duplicates.py` | Experiment 4: exact train/test duplicates and label conflicts (`results/duplicates.json`) |
| `src/run_logs.py` | Experiment 5 (extension): alert detection on BGL logs, random vs template-disjoint splits |
| `src/stats.py` | Bootstrap 95% CIs, paired gap intervals, exact McNemar tests (gzip k=1 vs TF-IDF) and BGL mean ± SD (`results/stats.json`) |
| `src/make_figures.py` | Writes `results/summary_tables.md`, `figures/fig1_k_sensitivity.png` (Figure 1 of the report) and `figures/fig2_published_vs_valid.png` (Figure 2) |
| `tests/test_ncd_knn.py` | Unit tests, incl. a check that our `optimistic` rule gives exactly the accuracy of the original `calc_acc()` |
| `third_party/npc_gzip_original/` | Two unmodified MIT-licensed files from the authors' repo, used only by that test |
| `results/` | All numbers reported in the paper (JSON + Markdown tables) |

## How to run

Requirements: Python ≥ 3.10, CPU only, ~3 GB RAM, ~1 hour on one core.

```bash
bash run_all.sh            # installs deps, downloads data, runs tests and every experiment
```

or step by step:

```bash
pip install -r requirements.txt
bash scripts/download_data.sh      # pinned public data from Hugging Face, TextGCN and Loghub, checksum-verified
python -m pytest -q tests          # 6 tests
bash scripts/run_full_all.sh       # Exp. 1 distance matrices (slowest step)
bash scripts/run_rest.sh           # Exp. 2-5, tables and figure
```

Measured single-core times: DengueFilipino 12 s, KirundiNews 4 min, R8 5 min,
KinyarwandaNews (600 test docs) ~14 min, SwahiliNews (300 test docs) ~10 min,
5-shot runs ~5 min, BGL < 1 min.

## Settings used

* gzip level 9 (Python default), texts joined with a single space before compression.
* k ∈ {1, 2, 3, 5, 7}; random tie-breaking seeds 0–4; 5-shot samples drawn with seeds 0–4.
* TF-IDF+LR: character 2–4-grams (`char_wb`), sublinear tf, ≤200,000 features, LogisticRegression(C=10, max_iter=2000); not tuned.
* Bootstrap: 1,000 percentile resamples, seed 0. Test subsamples: seed 0.
* BGL: StratifiedShuffleSplit (random) and GroupShuffleSplit by EventId (template-disjoint), test_size 0.2, seeds 0–4.
* Tested with Python 3.12; exact package versions are pinned in `requirements.txt`.
* Data sources are pinned to fixed revisions (Hugging Face parquet commits, GitHub commit SHAs) and
  checked against `data/SHA256SUMS`, so a rerun uses exactly the data behind `results/`.

## Notes on choices (read before comparing numbers)

* **Tie-breaking.** The authors' `calc_acc(rand=False)` counts a prediction correct
  if the gold label is *among* the tied labels. With k = 2 that is top-2 accuracy.
  We report it (`k2_optimistic`) only to reproduce the published tables; valid
  numbers are `k1`, `k2_random` and `k2_nearest`.
* **Test subsamples.** KinyarwandaNews and SwahiliNews use a random test subsample
  (600 and 300 documents, seed 0) because NCD needs one compression per
  test–train pair. Bootstrap 95% intervals are in `results/full.json`.
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
| KinyarwandaNews (600) | 0.891 | 0.890 | 0.828 | 0.890 |
| KirundiNews | 0.905 | 0.927 | 0.892 | 0.901 |
| DengueFilipino | 0.998 | 0.939 | 0.864 | 0.988 |
| SwahiliNews (300) | 0.927 | 0.943 | 0.857 | 0.897 |
| R8 | 0.954 | 0.952 | 0.915 | 0.967 |

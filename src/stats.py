"""Statistical analysis for the report (Tables 2-4):
 * 95% percentile-bootstrap CIs (1,000 resamples) for gzip (optimistic k=2, k=1)
   and TF-IDF+LR on the full-data splits;
 * paired bootstrap CI for the optimistic-minus-valid gap;
 * exact McNemar tests: gzip k=1 vs TF-IDF+LR on identical test items;
 * mean and SD over the five BGL splits.
Reads cache/full_*.npz (from run_full.py) and results/logs_bgl.json.
Writes results/stats.json."""
import json, os, sys
from math import comb
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from data import LOADERS
from ncd_knn import knn_predict, bootstrap_ci
from baselines import tfidf_lr


def mcnemar_exact(a_correct, b_correct):
    a, b = np.asarray(a_correct), np.asarray(b_correct)
    n01 = int(((a == 1) & (b == 0)).sum())  # A right, B wrong
    n10 = int(((a == 0) & (b == 1)).sum())  # A wrong, B right
    n, k = n01 + n10, min(n01, n10)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return {"gzip_only_correct": n01, "tfidf_only_correct": n10, "p_value": p}


def paired_gap_ci(a, b, n=1000, seed=0):
    rng = np.random.default_rng(seed); a, b = np.asarray(a, float), np.asarray(b, float)
    d = [(a[i] - b[i]).mean() for i in (rng.integers(0, len(a), len(a)) for _ in range(n))]
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


out = {"full": {}, "bgl": {}}
for name in ["KinyarwandaNews", "KirundiNews", "DengueFilipino", "SwahiliNews", "R8"]:
    z = np.load(f"cache/full_{name}.npz"); D, idx = z["D"], z["test_idx"]
    tr_x, tr_y, te_x_all, te_y_all = LOADERS[name]()
    te_x = [te_x_all[i] for i in idx]; te_y = [te_y_all[i] for i in idx]
    _, c_opt, a_opt = knn_predict(D, tr_y, te_y, k=2, tie="optimistic")
    _, c_k1, a_k1 = knn_predict(D, tr_y, te_y, k=1)
    a_lr, p_lr = tfidf_lr(tr_x, tr_y, te_x, te_y, return_pred=True)
    c_lr = [int(p == y) for p, y in zip(p_lr, te_y)]
    out["full"][name] = {
        "n_test": len(te_y),
        "optimistic_k2": [a_opt, *bootstrap_ci(c_opt)],
        "k1": [a_k1, *bootstrap_ci(c_k1)],
        "tfidf_lr": [a_lr, *bootstrap_ci(c_lr)],
        "gap_optimistic_minus_k1_ci": paired_gap_ci(c_opt, c_k1),
        "mcnemar_k1_vs_tfidf": mcnemar_exact(c_k1, c_lr)}
    print(name, out["full"][name], flush=True)

logs = json.load(open("results/logs_bgl.json"))
for split in ("random", "template_disjoint"):
    rs = logs[split]
    out["bgl"][split] = {k: [float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs], ddof=1))]
                         for k in rs[0]}
print(json.dumps(out["bgl"], indent=1))
json.dump(out, open("results/stats.json", "w"), indent=1)

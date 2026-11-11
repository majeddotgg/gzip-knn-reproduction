"""Compare our own BERT/mBERT runs (run_bert.py) with gzip and TF-IDF+LR.

 * 5-shot: mean accuracy with 95% t-intervals over the 5 trials, and paired t-tests
   (trial t of every method uses the same training sample) against valid gzip (k = 1).
 * full data: exact McNemar tests against gzip k = 1 on identical test items, bootstrap
   intervals, and accuracy on test items that do NOT occur verbatim in the training set.
Reads results/fewshot.json, results/bert_fewshot.json, results/bert_full.json and
cache/full_*.npz. Writes results/bert_compare.json."""
import json, os, sys
from math import comb, sqrt
import numpy as np
from scipy import stats as sps
sys.path.insert(0, os.path.dirname(__file__))
from data import LOADERS
from ncd_knn import knn_predict, bootstrap_ci, ncd_matrix
from baselines import tfidf_lr


def mcnemar_exact(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n01, n10 = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    n, k = n01 + n10, min(n01, n10)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return {"a_only_correct": n01, "b_only_correct": n10, "p_value": p}


def t_interval(v):
    v = np.asarray(v, float); h = sps.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / sqrt(len(v))
    return [float(v.mean()), float(h)]


out = {"fewshot": {}, "full": {}}
bf = json.load(open("results/bert_fewshot.json"))
for name in ["KinyarwandaNews", "KirundiNews", "DengueFilipino", "SwahiliNews"]:
    # re-score gzip (k = 1) and TF-IDF+LR on exactly the items and training samples BERT saw
    tr_x, tr_y, te_x_all, te_y_all = LOADERS[name]()
    tidx = bf["bert"][name]["test_idx"]
    te_x = [te_x_all[i] for i in tidx]; te_y = [te_y_all[i] for i in tidx]
    ty = np.asarray(tr_y); g, lr = [], []
    for t in range(5):                       # identical sampling to run_fewshot.py / run_bert.py
        rng = np.random.default_rng(t)
        idx = np.concatenate([rng.choice(np.where(ty == c)[0], 5, replace=False)
                              for c in sorted(set(ty.tolist())) if (ty == c).sum() >= 5])
        sx, sy = [tr_x[i] for i in idx], ty[idx].tolist()
        g.append(knn_predict(ncd_matrix(te_x, sx), sy, te_y, k=1)[2])
        lr.append(tfidf_lr(sx, sy, te_x, te_y))
    row = {"n_eval": len(te_y), "gzip_k1": t_interval(g), "tfidf_lr": t_interval(lr)}
    for m in ("bert", "mbert"):
        acc = [r["accuracy"] for r in sorted(bf[m][name]["runs"], key=lambda r: r["trial"])]
        row[m] = t_interval(acc)
        row[f"paired_t_gzip_k1_vs_{m}"] = float(sps.ttest_rel(g, acc).pvalue)
        row[f"paired_t_tfidf_vs_{m}"] = float(sps.ttest_rel(lr, acc).pvalue)
    out["fewshot"][name] = row
    print("5-shot", name, json.dumps(row), flush=True)

if os.path.exists("results/bert_full.json"):
    bfull = json.load(open("results/bert_full.json"))
    for name in bfull.get("bert", {}):
        z = np.load(f"cache/full_{name}.npz"); D, idx = z["D"], z["test_idx"]
        tr_x, tr_y, te_x_all, te_y_all = LOADERS[name]()
        te_x = [te_x_all[i] for i in idx]; te_y = [te_y_all[i] for i in idx]
        _, c_g, a_g = knn_predict(D, tr_y, te_y, k=1)
        dup = np.array([t in set(tr_x) for t in te_x])
        row = {"n_test": len(te_y), "gzip_k1": [a_g, *bootstrap_ci(c_g)],
               "gzip_k1_non_dup": float(np.mean(np.array(c_g)[~dup]))}
        for m in bfull:
            if m not in ("bert", "mbert") or name not in bfull[m]: continue
            e = bfull[m][name]; gold = e["gold"]; pred = e["runs"][0]["pred"]
            c = [int(p == y) for p, y in zip(pred, gold)]
            row[m] = [float(np.mean(c)), *bootstrap_ci(c)]
            row[f"{m}_non_dup"] = float(np.mean(np.array(c)[~dup]))
            row[f"mcnemar_gzip_k1_vs_{m}"] = mcnemar_exact(c_g, c)
        out["full"][name] = row
        print("full", name, json.dumps(row), flush=True)
json.dump(out, open("results/bert_compare.json", "w"), indent=1)

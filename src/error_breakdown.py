"""Quantitative error analysis for Section 5.5 of the report.
For each dataset (full training set, valid gzip k = 1 and TF-IDF+LR on the same items):
 * accuracy by text-length tercile (characters), and
 * the most frequent gold -> predicted confusion for gzip, with its share of all gzip errors.
Reads cache/full_*.npz (from run_full.py). Writes results/error_breakdown.json."""
import json, os, sys
from collections import Counter
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from data import LOADERS
from ncd_knn import knn_predict
from baselines import tfidf_lr

out = {}
for name in ["KinyarwandaNews", "KirundiNews", "DengueFilipino", "SwahiliNews", "R8"]:
    z = np.load(f"cache/full_{name}.npz"); D, idx = z["D"], z["test_idx"]
    tr_x, tr_y, te_x_all, te_y_all = LOADERS[name]()
    te_x = [te_x_all[i] for i in idx]; te_y = [te_y_all[i] for i in idx]
    pred_g, corr_g, _ = knn_predict(D, tr_y, te_y, k=1)
    _, pred_l = tfidf_lr(tr_x, tr_y, te_x, te_y, return_pred=True)
    corr_l = [int(p == y) for p, y in zip(pred_l, te_y)]
    lengths = np.array([len(t) for t in te_x])
    cuts = np.quantile(lengths, [1 / 3, 2 / 3])
    bins = np.digitize(lengths, cuts)            # 0 = shortest third, 2 = longest third
    terciles = {}
    for b, label in enumerate(["short", "medium", "long"]):
        m = bins == b
        terciles[label] = {"n": int(m.sum()), "max_chars": int(lengths[m].max()),
                           "gzip_k1": float(np.mean(np.array(corr_g)[m])),
                           "tfidf_lr": float(np.mean(np.array(corr_l)[m]))}
    errs = Counter((str(g), str(p)) for g, p, c in zip(te_y, pred_g, corr_g) if not c)
    (g, p), n = errs.most_common(1)[0]
    out[name] = {"n_test": len(te_y), "length_terciles": terciles,
                 "top_confusion": {"gold": g, "pred": p, "count": n,
                                   "share_of_gzip_errors": n / sum(errs.values())}}
    print(name, json.dumps(out[name]), flush=True)
json.dump(out, open("results/error_breakdown.json", "w"), indent=1)

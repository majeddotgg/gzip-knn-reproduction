"""Experiment 5 (extension): does gzip-kNN transfer to IT-operations logs?
Binary alert detection on Loghub BGL_2k under (a) random line-level splits and
(b) template-disjoint splits (no EventId shared between train and test)."""
import json, os, sys
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import f1_score
sys.path.insert(0, os.path.dirname(__file__))
from data import load_bgl
from ncd_knn import ncd_matrix, knn_predict
from baselines import tfidf_lr

X, y, ev = load_bgl()
X, y, ev = np.array(X, dtype=object), np.array(y), np.array(ev)
res = {"n": len(y), "alert_rate": float(y.mean()), "n_templates": int(len(set(ev)))}
for split in ("random", "template_disjoint"):
    runs = []
    for seed in range(5):
        if split == "random":
            sp = StratifiedShuffleSplit(1, test_size=0.2, random_state=seed)
            tr, te = next(sp.split(X, y))
        else:
            sp = GroupShuffleSplit(1, test_size=0.2, random_state=seed)
            tr, te = next(sp.split(X, y, ev))
        D = ncd_matrix(list(X[te]), list(X[tr]))
        row = {"test_alerts": int(y[te].sum())}
        for k, tie in ((1, "random"), (2, "optimistic"), (2, "random")):
            p, _, acc = knn_predict(D, y[tr].tolist(), y[te].tolist(), k=k, tie=tie, seed=seed)
            row[f"k{k}_{tie}_acc"] = acc
            row[f"k{k}_{tie}_f1"] = f1_score(y[te], p, zero_division=0)
        acc, p = tfidf_lr(list(X[tr]), y[tr].tolist(), list(X[te]), y[te].tolist(), return_pred=True)
        row["tfidf_lr_acc"], row["tfidf_lr_f1"] = acc, f1_score(y[te], p, zero_division=0)
        row["majority_acc"] = float(1 - y[te].mean())
        runs.append(row)
        print(split, seed, {k: round(v, 3) for k, v in row.items()}, flush=True)
    res[split] = runs
json.dump(res, open("results/logs_bgl.json", "w"), indent=1)

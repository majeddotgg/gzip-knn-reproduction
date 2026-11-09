"""Experiment 2: 5-shot classification on the four low-resource OOD datasets,
5 random trials, gzip-kNN (three tie rules, k=1/2/3) vs TF-IDF + logistic regression.
Writes results/fewshot.json."""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from data import LOADERS
from ncd_knn import ncd_matrix, knn_predict
from baselines import tfidf_lr

DATASETS = ["KinyarwandaNews", "KirundiNews", "DengueFilipino", "SwahiliNews"]
SHOTS, TRIALS = 5, 5
res = {}
for name in DATASETS:
    tr_x, tr_y, te_x, te_y = LOADERS[name]()
    tr_y = np.asarray(tr_y)
    runs = []
    for t in range(TRIALS):
        rng = np.random.default_rng(t)
        idx = np.concatenate([rng.choice(np.where(tr_y == c)[0], SHOTS, replace=False)
                              for c in sorted(set(tr_y.tolist())) if (tr_y == c).sum() >= SHOTS])
        sx, sy = [tr_x[i] for i in idx], tr_y[idx].tolist()
        t0 = time.time()
        D = ncd_matrix(te_x, sx)
        r = {"seconds": time.time() - t0}
        for k in (1, 2, 3):
            for tie in ("optimistic", "random", "nearest"):
                if k == 1 and tie != "random":
                    continue
                r[f"k{k}_{tie}"] = knn_predict(D, sy, te_y, k=k, tie=tie, seed=t)[2]
        r["tfidf_lr"] = tfidf_lr(sx, sy, te_x, te_y)
        runs.append(r)
        print(name, t, {k: round(v, 3) for k, v in r.items()}, flush=True)
    res[name] = runs
os.makedirs("results", exist_ok=True)
json.dump(res, open("results/fewshot.json", "w"), indent=1)

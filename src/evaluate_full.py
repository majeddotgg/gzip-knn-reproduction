"""Experiment 1 evaluation + Experiment 3 (sensitivity) + Experiment 4 (leakage).
Reads cache/full_*.npz, writes results/full.json and results/errors_*.json."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from data import LOADERS
from ncd_knn import knn_predict, topk_contains, bootstrap_ci
from baselines import tfidf_lr

PAPER = {"KirundiNews": 0.905, "KinyarwandaNews": 0.891, "DengueFilipino": 0.998,
         "SwahiliNews": 0.927, "R8": 0.954}
out = {}
for name in ["DengueFilipino", "KirundiNews", "R8", "KinyarwandaNews", "SwahiliNews"]:
    f = f"cache/full_{name}.npz"
    if not os.path.exists(f):
        continue
    z = np.load(f)
    D, idx = z["D"], z["test_idx"]
    tr_x, tr_y, te_x_all, te_y_all = LOADERS[name]()
    te_x = [te_x_all[i] for i in idx]; te_y = [te_y_all[i] for i in idx]
    r = {"paper": PAPER[name], "n_test": len(te_y), "n_train": len(tr_y),
         "ncd_seconds": float(z["seconds"])}
    for k in (1, 2, 3, 5, 7):
        for tie in ("optimistic", "random", "nearest"):
            if k == 1 and tie != "random":
                continue
            accs = [knn_predict(D, tr_y, te_y, k=k, tie=tie, seed=s)[2]
                    for s in range(5 if tie == "random" else 1)]
            r[f"k{k}_{tie}"] = float(np.mean(accs))
            if tie == "random":
                r[f"k{k}_random_std"] = float(np.std(accs))
    r["top2_contains"] = topk_contains(D, tr_y, te_y, 2)
    preds, corr, _ = knn_predict(D, tr_y, te_y, k=1)
    r["k1_ci95"] = bootstrap_ci(corr)
    _, corr_opt, _ = knn_predict(D, tr_y, te_y, k=2, tie="optimistic")
    r["k2_optimistic_ci95"] = bootstrap_ci(corr_opt)
    # leakage: test texts that also occur verbatim in the training set
    train_set = set(tr_x)
    dup = np.array([t in train_set for t in te_x])
    r["dup_rate"] = float(dup.mean())
    c = np.array(corr)
    r["k1_acc_on_dups"] = float(c[dup].mean()) if dup.any() else None
    r["k1_acc_on_clean"] = float(c[~dup].mean())
    # trained baseline on identical split
    acc_lr, pred_lr = tfidf_lr(tr_x, tr_y, te_x, te_y, return_pred=True)
    r["tfidf_lr"] = acc_lr
    r["tfidf_lr_on_clean"] = float(np.mean([p == y for p, y, d in zip(pred_lr, te_y, dup) if not d]))
    out[name] = r
    print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    # store k=1 errors on clean items for manual error analysis
    order = np.argsort(D, axis=1, kind="stable")[:, 0]
    errs = [{"text": te_x[i][:300], "gold": str(te_y[i]), "pred": str(preds[i]),
             "neighbour": tr_x[order[i]][:300], "ncd": float(D[i, order[i]])}
            for i in range(len(te_y)) if not corr[i] and not dup[i]]
    json.dump(errs[:40], open(f"results/errors_{name}.json", "w"), indent=1, ensure_ascii=False)
json.dump(out, open("results/full.json", "w"), indent=1)

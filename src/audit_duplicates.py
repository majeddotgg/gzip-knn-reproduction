"""Experiment 4 detail: exact train/test and train/train duplicates, and
duplicated test documents whose label disagrees with a training copy."""
import json, os, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(__file__))
from data import LOADERS

out = {}
for name, f in LOADERS.items():
    tr_x, tr_y, te_x, te_y = f()
    labels = defaultdict(set)
    for t, l in zip(tr_x, tr_y):
        labels[t].add(l)
    dup = [(t, l) for t, l in zip(te_x, te_y) if t in labels]
    out[name] = {"n_test": len(te_x), "test_in_train": len(dup),
                 "test_in_train_rate": len(dup) / len(te_x),
                 "dup_with_conflicting_label": sum(1 for t, l in dup if labels[t] != {l}),
                 "train_internal_duplicates": len(tr_x) - len(set(tr_x))}
    print(name, out[name])
json.dump(out, open("results/duplicates.json", "w"), indent=1)

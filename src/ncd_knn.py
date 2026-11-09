"""Re-implementation of the gzip + Normalized Compression Distance (NCD) + kNN
text classifier of Jiang et al. (2023, Findings of ACL), written from the
paper's description (Eq. 3 and Listing 1) rather than copied from npc_gzip.

The distance matrix is computed once and cached, so every decision rule
(k, tie-breaking strategy) is evaluated on exactly the same distances.
"""
from __future__ import annotations

import gzip
import random
from collections import Counter
from typing import Callable, Sequence

import numpy as np


def clen(text: str) -> int:
    """C(x): length in bytes of gzip-compressed UTF-8 text (default level 9)."""
    return len(gzip.compress(text.encode("utf-8")))


def ncd(cx: int, cy: int, cxy: int) -> float:
    """Normalized Compression Distance, Eq. (3) of the original paper."""
    return (cxy - min(cx, cy)) / max(cx, cy)


def ncd_matrix(test: Sequence[str], train: Sequence[str],
               progress: Callable[[int, int], None] | None = None) -> np.ndarray:
    """Return a |test| x |train| float32 matrix of NCD values.

    As in Listing 1 of the paper, the concatenation is test + " " + train.
    """
    c_train = [clen(t) for t in train]
    out = np.empty((len(test), len(train)), dtype=np.float32)
    for i, x in enumerate(test):
        cx = clen(x)
        row = out[i]
        for j, y in enumerate(train):
            cxy = len(gzip.compress((x + " " + y).encode("utf-8")))
            cy = c_train[j]
            row[j] = (cxy - min(cx, cy)) / max(cx, cy)
        if progress is not None:
            progress(i + 1, len(test))
    return out


TIE_MODES = ("optimistic", "random", "nearest")


def knn_predict(dist: np.ndarray, train_labels: Sequence, test_labels: Sequence,
                k: int = 2, tie: str = "random", seed: int = 0):
    """kNN majority vote over a precomputed distance matrix.

    tie = "optimistic": reproduces npc_gzip's calc_acc(rand=False): if the true
          label is among the tied top-voted labels it is counted as correct.
          This uses the gold label at prediction time, so it is NOT a valid
          classifier; it is included only to reproduce the published numbers.
    tie = "random":     choose uniformly among tied labels (npc_gzip rand=True).
    tie = "nearest":    choose the tied label whose closest neighbour is nearest.
    Returns (predictions, per-item correctness list, accuracy).
    """
    if tie not in TIE_MODES:
        raise ValueError(tie)
    rng = random.Random(seed)
    train_labels = np.asarray(train_labels)
    order = np.argsort(dist, axis=1, kind="stable")[:, :k]
    preds, correct = [], []
    for i in range(dist.shape[0]):
        neigh = train_labels[order[i]].tolist()
        votes = Counter(neigh)
        top = max(votes.values())
        tied = [lab for lab, c in votes.items() if c == top]
        if len(tied) == 1:
            p = tied[0]
        elif tie == "optimistic":
            p = test_labels[i] if test_labels[i] in tied else tied[0]
        elif tie == "random":
            p = rng.choice(sorted(tied, key=str))
        else:  # nearest: first label in distance order that is among the tied
            p = next(lab for lab in neigh if lab in tied)
        preds.append(p)
        correct.append(int(p == test_labels[i]))
    return preds, correct, float(np.mean(correct))


def topk_contains(dist: np.ndarray, train_labels, test_labels, k: int = 2) -> float:
    """Fraction of test items whose gold label is among the k nearest neighbours.
    For k = 2 this equals the 'optimistic' accuracy."""
    train_labels = np.asarray(train_labels)
    order = np.argsort(dist, axis=1, kind="stable")[:, :k]
    return float(np.mean([test_labels[i] in set(train_labels[order[i]].tolist())
                          for i in range(dist.shape[0])]))


def bootstrap_ci(correct: Sequence[int], n: int = 1000, seed: int = 0):
    """95% percentile bootstrap interval for accuracy."""
    rng = np.random.default_rng(seed)
    c = np.asarray(correct, dtype=float)
    stats = [c[rng.integers(0, len(c), len(c))].mean() for _ in range(n)]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))

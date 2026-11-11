"""Compute and cache full-training-set NCD matrices (Experiment 1).
Usage: python src/run_full.py DATASET [--jobs N] [--test-sample N] [--seed S]

--jobs splits the test rows across N processes; the matrix is identical to a
single-process run. --test-sample is kept for quick checks only; the reported
results use the full test sets.
"""
import argparse, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from data import LOADERS
from ncd_knn import ncd_matrix, ncd_matrix_parallel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--test-sample", type=int, default=0,
                    help="random subsample of the test set for quick checks (0 = full)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel processes for the NCD matrix (same result as 1)")
    a = ap.parse_args()

    tr_x, tr_y, te_x, te_y = LOADERS[a.dataset]()
    idx = np.arange(len(te_x))
    if a.test_sample and a.test_sample < len(te_x):
        idx = np.sort(np.random.default_rng(a.seed).choice(len(te_x), a.test_sample, replace=False))
    te_x = [te_x[i] for i in idx]

    t0 = time.time()
    def prog(i, n):
        if i % 100 == 0 or i == n:
            print(f"{a.dataset}: {i}/{n} rows, {time.time()-t0:.0f}s", flush=True)
    D = ncd_matrix(te_x, tr_x, prog) if a.jobs <= 1 else ncd_matrix_parallel(te_x, tr_x, a.jobs)
    os.makedirs("cache", exist_ok=True)
    np.savez_compressed(f"cache/full_{a.dataset}.npz", D=D, test_idx=idx,
                        seconds=time.time() - t0, jobs=a.jobs)
    print("done", a.dataset, D.shape, f"{time.time()-t0:.0f}s with {a.jobs} process(es)")


if __name__ == "__main__":  # required: worker processes re-import this file
    main()

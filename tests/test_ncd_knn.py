"""Unit tests for the re-implementation. Run: python -m pytest -q tests"""
import os, sys
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ncd_knn import clen, ncd, ncd_matrix, knn_predict, topk_contains  # noqa


def test_ncd_formula_matches_eq3():
    cx, cy, cxy = 40, 50, 70
    assert ncd(cx, cy, cxy) == pytest.approx((70 - 40) / 50)


def test_identical_text_is_closer_than_unrelated_text():
    a = "the central bank raised interest rates again this quarter"
    b = "the central bank raised interest rates again this year"
    c = "the striker scored twice in the second half of the final"
    D = ncd_matrix([a], [b, c])
    assert D[0, 0] < D[0, 1]


def test_matrix_shape_and_range():
    D = ncd_matrix(["x y z", "a b"], ["x y", "a", "q r s t"])
    assert D.shape == (2, 3)
    assert np.all(D > -0.5) and np.all(D < 1.5)


def test_k1_equals_nearest_label():
    D = np.array([[0.1, 0.5, 0.9], [0.8, 0.2, 0.3]])
    _, _, acc = knn_predict(D, ["a", "b", "c"], ["a", "b"], k=1)
    assert acc == 1.0


def test_optimistic_is_top2_and_random_is_lower_bound():
    D = np.array([[0.1, 0.2, 0.9]])  # neighbours: a, b -> tie
    _, _, opt = knn_predict(D, ["a", "b", "c"], ["b"], k=2, tie="optimistic")
    _, _, near = knn_predict(D, ["a", "b", "c"], ["b"], k=2, tie="nearest")
    assert opt == 1.0          # credited because gold 'b' is among tied labels
    assert near == 0.0         # a valid rule picks 'a' (closest)
    assert topk_contains(D, ["a", "b", "c"], ["b"], k=2) == 1.0


def test_optimistic_rule_equals_original_calc_acc():
    """Our 'optimistic' rule must give the same accuracy as npc_gzip's
    KnnExpText.calc_acc(rand=False), the function behind the paper's tables."""
    pytest.importorskip("tqdm")
    sys.path.insert(0, os.path.join(ROOT, "third_party", "npc_gzip_original"))
    from experiments import KnnExpText
    rng = np.random.default_rng(1)
    D = rng.random((200, 60))
    tr_y = rng.integers(0, 5, 60).tolist()
    te_y = rng.integers(0, 5, 200).tolist()
    exp = KnnExpText(None, None, None)
    exp.distance_matrix = D.tolist()
    _, corr = exp.calc_acc(2, te_y, train_label=tr_y)
    _, _, ours = knn_predict(D, tr_y, te_y, k=2, tie="optimistic")
    assert ours == pytest.approx(np.mean(corr))

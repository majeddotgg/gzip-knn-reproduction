"""Trained baseline used for comparison: character n-gram TF-IDF + logistic
regression (language-agnostic, CPU-only, seconds to train)."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


def tfidf_lr(tr_x, tr_y, te_x, te_y, return_pred=False):
    clf = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True,
                        min_df=1, max_features=200000),
        LogisticRegression(max_iter=2000, C=10.0))
    clf.fit(tr_x, tr_y)
    pred = clf.predict(te_x).tolist()
    acc = sum(int(p == y) for p, y in zip(pred, te_y)) / len(te_y)
    return (acc, pred) if return_pred else acc

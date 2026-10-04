"""
Sanity tests: our from-scratch metrics must match scikit-learn on random data.
(sklearn is used here ONLY as a reference, never inside src/.)

Run from the repo root:
    python -m tests.test_evaluate          # plain python
    python -m pytest tests -q              # if you pip install pytest
"""

import numpy as np
from sklearn import metrics as skm

from src import evaluate as ev


def _data(seed=0, n=500):
    rng = np.random.default_rng(seed)
    y = rng.normal(12, 1, n)
    pred = y + rng.normal(0, 0.5, n)
    t = rng.integers(0, 2, n)
    s = t * 0.8 + rng.normal(0, 1, n)
    return y, pred, t, s


def test_regression_metrics():
    y, pred, _, _ = _data()
    assert np.isclose(ev.rmse(y, pred), np.sqrt(skm.mean_squared_error(y, pred)))
    assert np.isclose(ev.mae(y, pred), skm.mean_absolute_error(y, pred))
    assert np.isclose(ev.r2(y, pred), skm.r2_score(y, pred))


def test_confusion_and_prf():
    _, _, t, s = _data()
    pred = (s > 0.4).astype(int)
    assert (ev.confusion_matrix(t, pred) == skm.confusion_matrix(t, pred)).all()
    p, r, f = ev.precision_recall_f1(t, pred)
    assert np.isclose(p, skm.precision_score(t, pred))
    assert np.isclose(r, skm.recall_score(t, pred))
    assert np.isclose(f, skm.f1_score(t, pred))
    assert np.isclose(ev.accuracy(t, pred), skm.accuracy_score(t, pred))


def test_roc_auc_continuous_and_ties():
    _, _, t, s = _data()
    assert np.isclose(ev.roc_auc(t, s), skm.roc_auc_score(t, s))
    s_tied = np.round(s)                  # many ties, like tree leaf probabilities
    assert np.isclose(ev.roc_auc(t, s_tied), skm.roc_auc_score(t, s_tied))


def test_roc_auc_extremes():
    t = np.array([0, 0, 1, 1])
    assert ev.roc_auc(t, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert ev.roc_auc(t, [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert ev.roc_auc(t, [0.5, 0.5, 0.5, 0.5]) == 0.5


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK ", name)
    print("all evaluate tests passed")
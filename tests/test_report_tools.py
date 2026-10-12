"""
Tests for the evaluation tooling: roc_curve, kfold_indices, cv.py, save_table
and the figures in plots.py. Synthetic data only (no dataset needed).

Run from the repo root:
    python -X utf8 -m tests.test_report_tools
    python -m pytest tests -q
"""

import tempfile
from pathlib import Path

import numpy as np
from sklearn import metrics as skm
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src import cv, plots
from src import data_prep as dp
from src import evaluate as ev


def _cls_data(seed=0, n=500):
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 2, n)
    s = t * 0.8 + rng.normal(0, 1, n)
    return t, s


def _price_data(seed=3, n=1200):
    rng = np.random.default_rng(seed)
    X = rng.normal(5, 2, (n, 4))                       # offset 5 -> unscaled mean is NOT 0
    y = np.exp(10 + 0.4 * X[:, 0] + 0.2 * X[:, 1] + rng.normal(0, 0.3, n))
    return X, y


# --- roc_curve --------------------------------------------------------------
def test_roc_curve_matches_sklearn_with_and_without_ties():
    t, s = _cls_data()
    for scores in (s, np.round(s)):                    # rounded = many ties (tree leaves)
        fpr, tpr, thr = ev.roc_curve(t, scores)
        f2, t2, _ = skm.roc_curve(t, scores, drop_intermediate=False)
        assert np.allclose(fpr, f2) and np.allclose(tpr, t2)
        assert np.isclose(ev.auc_trapezoid(fpr, tpr), ev.roc_auc(t, scores))
        assert np.isclose(ev.auc_trapezoid(fpr, tpr), skm.roc_auc_score(t, scores))
        assert (fpr[0], tpr[0]) == (0.0, 0.0) and (fpr[-1], tpr[-1]) == (1.0, 1.0)
        assert np.isinf(thr[0]) and np.all(np.diff(fpr) >= 0) and np.all(np.diff(tpr) >= 0)


def test_roc_curve_extremes_and_errors():
    t = np.array([0, 0, 1, 1])
    f, tp, _ = ev.roc_curve(t, [0.1, 0.2, 0.8, 0.9])
    assert ev.auc_trapezoid(f, tp) == 1.0
    f, tp, _ = ev.roc_curve(t, [0.5, 0.5, 0.5, 0.5])   # all tied -> the diagonal
    assert ev.auc_trapezoid(f, tp) == 0.5
    try:
        ev.roc_curve([1, 1, 1], [0.1, 0.2, 0.3])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


# --- kfold_indices ----------------------------------------------------------
def test_kfold_disjoint_complete_deterministic_balanced_stratified():
    rng = np.random.default_rng(1)
    y = rng.lognormal(12, 0.8, 1003)
    folds = dp.kfold_indices(y, k=5)
    assert len(folds) == 5
    tests = np.concatenate([te for _, te in folds])
    assert np.array_equal(np.sort(tests), np.arange(len(y)))          # each row tested once
    for tr, te in folds:
        assert len(set(tr) & set(te)) == 0 and len(tr) + len(te) == len(y)
    sizes = [len(te) for _, te in folds]
    assert max(sizes) - min(sizes) <= 1                                # balanced sizes
    again = dp.kfold_indices(y, k=5)
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(folds, again))   # seeded
    other = dp.kfold_indices(y, k=5, seed=7)
    assert not np.array_equal(folds[0][1], other[0][1])                # seed matters
    thr = np.median(y)
    shares = [float((y[te] > thr).mean()) for _, te in folds]
    assert max(shares) - min(shares) < 0.03                            # stratified


# --- cv ---------------------------------------------------------------------
def test_cv_regression_and_classification_run_and_are_deterministic():
    X, y = _price_data()
    reg = cv.cv_regression(lambda: DecisionTreeRegressor(max_depth=5, random_state=0), X, y)
    for key in ("RMSE(log)", "R2(log)", "RMSE(AZN)", "MAE(AZN)", "fit s", "predict s"):
        mean, std = reg[key]
        assert np.isfinite(mean) and std >= 0
    assert reg["R2(log)"][0] > 0.5
    clf = cv.cv_classification(lambda: DecisionTreeClassifier(max_depth=5, random_state=0), X, y)
    for key in ("Accuracy", "Precision", "Recall", "F1", "ROC-AUC"):
        assert 0.0 <= clf[key][0] <= 1.0
    assert clf["ROC-AUC"][0] > 0.8
    again = cv.cv_classification(lambda: DecisionTreeClassifier(max_depth=5, random_state=0), X, y)
    for key in ("Accuracy", "F1", "ROC-AUC"):
        assert clf[key] == again[key]                                  # seeded -> identical


class _Recorder:
    """Stub model: records what fit() saw, predicts a constant."""
    seen_X, seen_y = [], []

    def fit(self, X, y):
        _Recorder.seen_X.append(np.array(X))
        _Recorder.seen_y.append(np.array(y))
        return self

    def predict(self, X):
        return np.zeros(len(X))

    def decision_function(self, X):
        return np.zeros(len(X))


def test_cv_scales_with_train_statistics_only_and_labels_each_fold_by_train_median():
    X, y = _price_data()
    _Recorder.seen_X, _Recorder.seen_y = [], []
    cv.cv_classification(_Recorder, X, y, scale=True)
    assert len(_Recorder.seen_X) == cv.K
    for Xf, yf in zip(_Recorder.seen_X, _Recorder.seen_y):
        assert np.allclose(Xf.mean(axis=0), 0, atol=1e-9)              # train mean = 0
        assert np.allclose(Xf.std(axis=0), 1, atol=1e-9)
        assert set(np.unique(yf)) == {0, 1}
        assert abs(yf.mean() - 0.5) < 0.02                              # median of THIS train part
    _Recorder.seen_X, _Recorder.seen_y = [], []
    cv.cv_classification(_Recorder, X, y, scale=False)
    assert all(abs(Xf.mean()) > 1 for Xf in _Recorder.seen_X)           # raw features untouched


def test_cv_regression_fits_on_log_price():
    X, y = _price_data()
    _Recorder.seen_X, _Recorder.seen_y = [], []
    cv.cv_regression(_Recorder, X, y)
    assert all(yf.max() < 40 for yf in _Recorder.seen_y)               # log scale, not AZN


# --- tables on disk ---------------------------------------------------------
def test_save_table_writes_md_and_csv_with_mean_std():
    res = {"my tree": {"RMSE": (0.4, 0.01), "R2": 0.8}, "other": {"R2": 0.7}}
    with tempfile.TemporaryDirectory() as d:
        for given in (Path(d) / "sub" / "t", Path(d) / "sub" / "t.md"):  # with / without suffix
            md_path, csv_path = ev.save_table(res, given, title="demo")
            assert md_path.name == "t.md" and csv_path.name == "t.csv"
        md = md_path.read_text(encoding="utf-8")
        assert md.startswith("### demo") and "0.4000 ± 0.0100" in md and "| other |" in md
        csv_text = csv_path.read_text(encoding="utf-8-sig")
        assert csv_text.splitlines()[0] == "model,RMSE,R2"
        assert "other,-,0.7000" in csv_text


# --- figures (smoke tests) --------------------------------------------------
def test_figures_are_written():
    t, s = _cls_data()
    pred = (s > 0.4).astype(int)
    rng = np.random.default_rng(0)
    yt = rng.normal(12, 1, 300)
    with tempfile.TemporaryDirectory() as d:
        out = [
            plots.plot_confusion_matrices({"a": ev.confusion_matrix(t, pred),
                                           "b": ev.confusion_matrix(t, 1 - pred)}, d),
            plots.plot_roc_curves({"a": (t, s), "b": (t, np.round(s))}, d),
            plots.plot_pred_vs_actual({"a": (yt, yt + rng.normal(0, 0.3, 300))}, d),
            plots.plot_error_bars({"x": {"n": 5, "err": 0.1}, "y": {"n": 9, "err": -0.2}},
                                  "err", "demo", "e.png", outdir=d),
        ]
        assert all(p.exists() and p.stat().st_size > 1000 for p in out)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK ", name)
    print("all report-tool tests passed")
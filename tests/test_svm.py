"""Unit tests for src/svm.py (run with ``pytest`` from the repo root)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.svm import (  # noqa: E402
    PegasosSVM,
    RFFMapper,
    accuracy_score,
    compare_models,
    f1_score,
    lambda_sweep,
    objective_value,
)


# ------------------------------- fixtures ---------------------------------- #
def make_blobs(n: int = 300, seed: int = 0):
    """Two well-separated Gaussian blobs, labels in {0, 1}."""
    rng = np.random.default_rng(seed)
    X0 = rng.normal(loc=[-2.0, -2.0], scale=0.8, size=(n // 2, 2))
    X1 = rng.normal(loc=[2.0, 2.0], scale=0.8, size=(n // 2, 2))
    X = np.vstack([X0, X1])
    y = np.r_[np.zeros(n // 2), np.ones(n // 2)].astype(int)
    perm = rng.permutation(n)
    return X[perm], y[perm]


def make_circles(n: int = 400, seed: int = 0):
    """Concentric circles: not linearly separable."""
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0, 2 * np.pi, n)
    radii = np.where(np.arange(n) % 2 == 0, 1.0, 3.0)
    radii = radii + rng.normal(0, 0.1, n)
    X = np.c_[radii * np.cos(angles), radii * np.sin(angles)]
    y = (np.arange(n) % 2).astype(int)
    return X, y


@pytest.fixture(scope="module")
def blobs():
    return make_blobs()


@pytest.fixture(scope="module")
def fitted(blobs):
    X, y = blobs
    return PegasosSVM(lambda_=1e-2, n_epochs=30, batch_size=8,
                      random_state=0).fit(X, y)


# ----------------------------- label mapping ------------------------------- #
def test_to_pm1_mapping():
    out = PegasosSVM._to_pm1(np.array([0, 1, 1, 0]))
    np.testing.assert_array_equal(out, [-1.0, 1.0, 1.0, -1.0])


def test_from_pm1_roundtrip():
    y = np.array([0, 1, 0, 0, 1])
    back = PegasosSVM._from_pm1(PegasosSVM._to_pm1(y))
    np.testing.assert_array_equal(back, y)
    assert set(np.unique(back)) <= {0, 1}


def test_invalid_labels_raise():
    with pytest.raises(ValueError):
        PegasosSVM._to_pm1(np.array([0, 1, 2]))
    with pytest.raises(ValueError):
        PegasosSVM._to_pm1(np.array([-1, 1]))


# ------------------------------ fitting ------------------------------------ #
def test_fit_separable_data(blobs, fitted):
    X, y = blobs
    assert accuracy_score(y, fitted.predict(X)) > 0.97


def test_predict_matches_decision_sign(blobs, fitted):
    X, _ = blobs
    scores = fitted.decision_function(X)
    preds = fitted.predict(X)
    assert set(np.unique(preds)) <= {0, 1}
    np.testing.assert_array_equal(preds, (scores > 0).astype(int))


def test_unfitted_raises(blobs):
    with pytest.raises(RuntimeError):
        PegasosSVM().predict(blobs[0])


def test_invalid_hyperparameters():
    with pytest.raises(ValueError):
        PegasosSVM(lambda_=0.0)
    with pytest.raises(ValueError):
        PegasosSVM(track="bad")


def test_reproducible(blobs):
    X, y = blobs
    a = PegasosSVM(random_state=3).fit(X, y)
    b = PegasosSVM(random_state=3).fit(X, y)
    np.testing.assert_allclose(a.coef_, b.coef_)


# ---------------------------- objective decay ------------------------------ #
def test_objective_decays(fitted):
    hist = fitted.get_objective_history()
    assert len(hist) == fitted.n_epochs + 1
    assert hist[0] == pytest.approx(1.0)  # w=0 -> mean hinge = 1
    assert hist[-1] < 0.5 * hist[0]
    assert np.all(np.isfinite(hist))


def test_objective_matches_helper(blobs, fitted):
    X, y = blobs
    expected = objective_value(
        fitted.coef_, fitted.intercept_, X, PegasosSVM._to_pm1(y), 1e-2)
    assert fitted.get_objective_history()[-1] == pytest.approx(expected)


def test_step_tracking_length(blobs):
    X, y = blobs
    m = PegasosSVM(n_epochs=2, batch_size=50, track="step",
                    random_state=0).fit(X, y)
    steps = 2 * int(np.ceil(len(y) / 50))
    assert m.n_iter_ == steps
    assert len(m.get_objective_history()) == steps + 1


# --------------------------- margin / support vectors ---------------------- #
def test_margin(fitted):
    assert fitted.get_margin() == pytest.approx(
        2.0 / np.linalg.norm(fitted.coef_))
    assert fitted.get_margin() > 0


def test_support_vectors(blobs, fitted):
    X, y = blobs
    sv = fitted.get_support_vectors(X, y)
    y_pm1 = PegasosSVM._to_pm1(y)
    margins = y_pm1 * fitted.decision_function(X)
    np.testing.assert_array_equal(sv, np.flatnonzero(margins <= 1.0))
    assert 0 < sv.size < len(y)
    non_sv = np.setdiff1d(np.arange(len(y)), sv)
    assert np.all(margins[non_sv] > 1.0)


# --------------------------------- RFF ------------------------------------- #
def test_rff_shape_and_kernel_approximation():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    rff = RFFMapper(n_components=4000, gamma=0.5, random_state=0)
    Z = rff.fit_transform(X)
    assert Z.shape == (40, 4000)
    sq = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    exact = np.exp(-0.5 * sq)
    assert np.mean(np.abs(Z @ Z.T - exact)) < 0.03


def test_rff_unfitted_raises():
    with pytest.raises(RuntimeError):
        RFFMapper().transform(np.zeros((2, 2)))


def test_rff_solves_nonlinear_problem():
    X, y = make_circles()
    linear = PegasosSVM(lambda_=1e-3, n_epochs=20, batch_size=16,
                        random_state=0).fit(X, y)
    rff = PegasosSVM(
        lambda_=1e-3, n_epochs=30, batch_size=16,
        feature_map=RFFMapper(300, gamma=0.5, random_state=0),
        random_state=0).fit(X, y)
    acc_lin = accuracy_score(y, linear.predict(X))
    acc_rff = accuracy_score(y, rff.predict(X))
    assert acc_lin < 0.7
    assert acc_rff > 0.9


# ------------------------------ sweep / sklearn ---------------------------- #
def test_metrics():
    y = np.array([1, 1, 0, 0])
    p = np.array([1, 0, 0, 1])
    assert accuracy_score(y, p) == 0.5
    assert f1_score(y, p) == pytest.approx(0.5)
    assert f1_score(np.zeros(3), np.zeros(3)) == 0.0


def test_lambda_sweep(blobs):
    X, y = blobs
    res = lambda_sweep(X[:200], y[:200], X[200:], y[200:],
                       lambdas=[1e-4, 1e-3, 1e-2, 1e-1], n_epochs=15)
    assert [r["lambda"] for r in res] == [1e-4, 1e-3, 1e-2, 1e-1]
    assert {"margin", "val_accuracy", "val_f1"} <= set(res[0])
    assert res[-1]["margin"] > res[0]["margin"]  # bigger lambda -> wider margin
    assert all(r["val_accuracy"] > 0.9 for r in res)


def test_compare_with_sklearn_baselines(blobs):
    pytest.importorskip("sklearn")
    X, y = blobs
    report = compare_models(X[:200], y[:200], X[200:], y[200:],
                            n_components=100, n_epochs=10)
    assert {"Pegasos(linear)", "Pegasos(RFF)", "LinearSVC",
            "SGDClassifier(hinge)", "SVC(rbf)"} <= set(report)
    for stats in report.values():
        assert stats["accuracy"] > 0.9

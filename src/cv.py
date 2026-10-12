"""
cv.py - seeded, leak-free k-fold cross-validation for both tasks (Problem 5c).

Works with ANY model that has the fit/predict API: sklearn baselines, our
DecisionTree, our PegasosSVM. The caller passes a FACTORY (a function that
returns a fresh, unfitted model), so every fold starts from scratch:

    cv.cv_regression(lambda: DecisionTree(task="regression", criterion="mse",
                                          max_depth=12), X_dev, y_dev)
    cv.cv_classification(lambda: PegasosSVM(lambda_=1e-3), X_dev, y_dev, scale=True)

Leakage rules (the same as in data_prep):
  * run it on the TRAIN+VAL pool only; the test set is never touched here,
  * the premium/standard threshold is the median of each fold's TRAIN part,
  * standardisation statistics come from each fold's TRAIN part.

Returns {metric: (mean, std)}; std is the population std (np.std, ddof=0),
like cross_val_score(...).std(). Feed the dict straight to evaluate.compare.
"""

from __future__ import annotations

import time

import numpy as np

from . import data_prep as dp
from . import evaluate as ev

K = 5


def model_scores(model, X) -> np.ndarray:
    """Continuous score for the positive (premium) class: P(premium) if the model
    has predict_proba, otherwise the signed margin from decision_function."""
    if hasattr(model, "predict_proba"):
        p = np.asarray(model.predict_proba(X))
        return p[:, 1] if p.ndim == 2 else p.ravel()
    return np.asarray(model.decision_function(X), dtype=float).ravel()


def summarize_folds(fold_metrics: list[dict]) -> dict:
    """[{metric: value}, ...] one dict per fold -> {metric: (mean, std)}."""
    return {k: (float(np.mean([m[k] for m in fold_metrics])),
                float(np.std([m[k] for m in fold_metrics])))
            for k in fold_metrics[0]}


def cv_regression(make_model, X, y_price, k: int = K, seed: int = dp.SEED) -> dict:
    """Task A: model is fitted on log(price). Metrics as in run_all's test table,
    plus mean fit / predict time per fold in seconds."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y_price, dtype=float)
    folds = []
    for tr, te in dp.kfold_indices(y, k, seed):
        model = make_model()
        t0 = time.perf_counter()
        model.fit(X[tr], np.log(y[tr]))
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        p_log = np.asarray(model.predict(X[te]), dtype=float).ravel()
        pred_s = time.perf_counter() - t0
        p_azn = np.exp(p_log)
        folds.append({
            "RMSE(log)": ev.rmse(np.log(y[te]), p_log),
            "R2(log)": ev.r2(np.log(y[te]), p_log),
            "RMSE(AZN)": ev.rmse(y[te], p_azn),
            "MAE(AZN)": ev.mae(y[te], p_azn),
            "fit s": fit_s,
            "predict s": pred_s,
        })
    return summarize_folds(folds)


def cv_classification(make_model, X, y_price, k: int = K, seed: int = dp.SEED,
                      scale: bool = False) -> dict:
    """Task B: tier label rebuilt inside every fold from that fold's train prices.
    scale=True standardises with the fold's train statistics (use for the SVM)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y_price, dtype=float)
    folds = []
    for tr, te in dp.kfold_indices(y, k, seed):
        t_tr, thr = dp.make_tier_label(y[tr])
        t_te, _ = dp.make_tier_label(y[te], thr)
        X_tr, X_te = X[tr], X[te]
        if scale:
            X_tr, X_te = dp.standardize(X_tr, X_te)
        model = make_model()
        t0 = time.perf_counter()
        model.fit(X_tr, t_tr)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred = np.asarray(model.predict(X_te)).ravel().astype(int)
        pred_s = time.perf_counter() - t0
        p, r, f1 = ev.precision_recall_f1(t_te, pred)
        folds.append({
            "Accuracy": ev.accuracy(t_te, pred),
            "Precision": p,
            "Recall": r,
            "F1": f1,
            "ROC-AUC": ev.roc_auc(t_te, model_scores(model, X_te)),
            "fit s": fit_s,
            "predict s": pred_s,
        })
    return summarize_folds(folds)
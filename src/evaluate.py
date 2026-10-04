"""
evaluate.py — metrics + comparison helpers, written FROM SCRATCH in NumPy.

scikit-learn is used only in tests/test_evaluate.py to sanity-check these
numbers; nothing here imports it.
"""

from __future__ import annotations

import numpy as np


def _as_1d(a) -> np.ndarray:
    return np.asarray(a, dtype=float).ravel()


# --- Regression (Task A: price) --------------------------------------------
def rmse(y_true, y_pred) -> float:
    """sqrt(mean((y_true - y_pred)^2)) — same units as y (AZN, or log-AZN)."""
    y_true, y_pred = _as_1d(y_true), _as_1d(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    """mean(|y_true - y_pred|)."""
    y_true, y_pred = _as_1d(y_true), _as_1d(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true, y_pred) -> float:
    """1 - SS_res / SS_tot. 1.0 = perfect, 0.0 = no better than predicting the mean."""
    y_true, y_pred = _as_1d(y_true), _as_1d(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:                       # constant target: define like sklearn
        return 1.0 if ss_res == 0 else 0.0
    return float(1.0 - ss_res / ss_tot)


# --- Classification (Task B: price tier) -----------------------------------
def confusion_matrix(y_true, y_pred) -> np.ndarray:
    """2x2 matrix [[TN, FP], [FN, TP]] with premium (=1) as the positive class."""
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred = np.asarray(y_pred).ravel().astype(int)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return np.array([[tn, fp], [fn, tp]])


def precision_recall_f1(y_true, y_pred):
    """(precision, recall, f1) for the positive (premium) class; 0.0 if undefined."""
    (tn, fp), (fn, tp) = confusion_matrix(y_true, y_pred)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return float(precision), float(recall), float(f1)


def accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.mean(y_true == y_pred))


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks; tied values get the average of the ranks they span."""
    order = np.argsort(x, kind="mergesort")
    _, start, counts = np.unique(x[order], return_index=True, return_counts=True)
    avg_rank = start + (counts + 1) / 2.0            # mean of start+1 .. start+count
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.repeat(avg_rank, counts)
    return ranks


def roc_auc(y_true, scores) -> float:
    """
    ROC AUC via the rank-sum (Mann-Whitney U) formula:
        AUC = (sum of ranks of positives - n_pos(n_pos+1)/2) / (n_pos * n_neg)
    Ties in `scores` are handled with average ranks (important for trees,
    which output the same probability for every sample in a leaf).
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    scores = _as_1d(scores)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError("roc_auc needs both classes present in y_true")
    ranks = _average_ranks(scores)
    u = ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


# --- Comparison helper ------------------------------------------------------
def _fmt(v) -> str:
    """Format a number, or a (mean, std) pair as 'mean ± std'."""
    if isinstance(v, (tuple, list)) and len(v) == 2:
        return f"{v[0]:.4f} ± {v[1]:.4f}"
    if isinstance(v, (float, np.floating)):
        return f"{v:.4f}"
    return str(v)


def compare(name_to_metrics: dict) -> None:
    """
    Pretty-print a table: one row per model, one column per metric.

        compare({
            "my tree":      {"RMSE": 0.41, "R2": 0.80},
            "sklearn tree": {"RMSE": (0.40, 0.01), "R2": (0.81, 0.01)},  # CV mean, std
        })
    Values can be floats or (mean, std) tuples. Missing metrics print as '-'.
    """
    metrics = []
    for m in name_to_metrics.values():
        for k in m:
            if k not in metrics:
                metrics.append(k)
    rows = [[name] + [_fmt(m[k]) if k in m else "-" for k in metrics]
            for name, m in name_to_metrics.items()]
    header = ["model"] + metrics
    widths = [max(len(r[i]) for r in [header] + rows) for i in range(len(header))]
    line = "  ".join("-" * w for w in widths)
    print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
    print(line)
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))
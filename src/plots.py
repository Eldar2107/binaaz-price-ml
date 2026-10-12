"""
plots.py - evaluation figures (the EDA figures live in eda.py).

error_analysis.py imports FIG_DIR and plot_error_bars from here; run_all.py uses
the confusion-matrix / ROC / predicted-vs-actual figures. Every function takes
plain arrays or dicts (no model objects), saves a PNG and returns its Path.
PNGs go to report/figures/ (git-ignored on purpose: run_all re-creates them).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")                      # no GUI needed; just write files
import matplotlib.pyplot as plt
import numpy as np

from . import evaluate

FIG_DIR = Path("report/figures")
plt.rcParams.update({"font.size": 9})
CLASSES = ["standard", "premium"]


def _save(fig, outdir, name) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_error_bars(table: dict, metric: str, title: str, filename: str,
                    outdir=FIG_DIR) -> Path:
    """Bar chart of one metric per group. `table` = {group: {"n": .., metric: ..}}
    (the format error_analysis.group_table returns); n is printed under each bar."""
    groups = list(table)
    vals = [float(table[g][metric]) for g in groups]
    labels = [f"{g}\n(n={table[g]['n']})" if "n" in table[g] else str(g) for g in groups]
    fig, ax = plt.subplots(figsize=(max(5.5, 0.9 * len(groups) + 2), 4))
    ax.bar(range(len(groups)), vals, color="#4C72B0")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(labels, rotation=30 if len(groups) > 5 else 0,
                       ha="right" if len(groups) > 5 else "center", fontsize=8)
    ax.set_ylabel(metric)
    ax.set_title(title)
    if min(vals, default=0) < 0:
        ax.axhline(0, color="gray", lw=1)
    return _save(fig, outdir, filename)


def plot_confusion_matrices(cms: dict, outdir=FIG_DIR,
                            name: str = "08_confusion_matrices.png") -> Path:
    """cms = {model name: 2x2 array [[TN, FP], [FN, TP]]}. Colour = row-normalised
    share (recall-style); the cell text shows the count and that share."""
    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(3.7 * n + 0.3, 3.7), squeeze=False)
    for ax, (model, cm) in zip(axes[0], cms.items()):
        cm = np.asarray(cm)
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n({norm[i, j]:.1%})", ha="center",
                        va="center", fontsize=9,
                        color="white" if norm[i, j] > 0.5 else "black")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(CLASSES)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(CLASSES)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(model)
    return _save(fig, outdir, name)


def plot_roc_curves(curves: dict, outdir=FIG_DIR, name: str = "09_roc_curves.png") -> Path:
    """curves = {model name: (y_true, scores)}. Uses our own roc_curve / auc."""
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    for model, (y_true, scores) in curves.items():
        fpr, tpr, _ = evaluate.roc_curve(y_true, scores)
        ax.plot(fpr, tpr, lw=1.6, label=f"{model} (AUC = {evaluate.auc_trapezoid(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1, label="chance")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC curves, price tier (test set)")
    ax.legend(loc="lower right", fontsize=8)
    return _save(fig, outdir, name)


def plot_pred_vs_actual(preds: dict, outdir=FIG_DIR,
                        name: str = "10_pred_vs_actual.png") -> Path:
    """preds = {model name: (log price true, log price predicted)}. Hexbin density
    (log colour scale) so 9,000+ points stay readable; dashed line = perfect fit."""
    n = len(preds)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n + 0.4, 4.4), squeeze=False)
    for ax, (model, (yt, yp)) in zip(axes[0], preds.items()):
        yt, yp = np.asarray(yt, float), np.asarray(yp, float)
        hb = ax.hexbin(yt, yp, gridsize=55, bins="log", mincnt=1, cmap="viridis")
        lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], ls="--", color="red", lw=1)
        ax.set_xlabel("true log(price)")
        ax.set_ylabel("predicted log(price)")
        ax.set_title(f"{model}  (RMSE = {evaluate.rmse(yt, yp):.3f}, "
                     f"R² = {evaluate.r2(yt, yp):.3f})", fontsize=9)
        fig.colorbar(hb, ax=ax, label="listings (log)")
    return _save(fig, outdir, name)
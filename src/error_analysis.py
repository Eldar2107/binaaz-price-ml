"""
error_analysis.py - where and why does a model fail? (Problem 6)

Model-agnostic: takes TEST-set arrays and the matching rows of the cleaned
DataFrame, never a model. Call it once per model:

    from src import error_analysis
    te = ds.df.iloc[ds.idx_te].reset_index(drop=True)
    error_analysis.run(te, ds.y_te, np.exp(pred_log), ds.t_te, tier_pred,
                       name="our tree")

Writes Markdown tables to report/tables/ and bar charts to report/figures/,
and prints the same tables, so every number in the report comes from code.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import evaluate, plots

TABLE_DIR = Path("report") / "tables"

# Fixed, domain-based price ranges in AZN (not computed from the data).
PRICE_EDGES = [100_000, 200_000, 350_000, 600_000, 1_000_000]
PRICE_LABELS = ["<100k", "100-200k", "200-350k", "350-600k", "600k-1M", ">1M"]


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_")


def group_table(groups, y_true, y_pred, t_true=None, t_pred=None,
                order=None, min_group: int = 1) -> dict:
    """
    {group: {n, MAE(AZN), MdAPE(%), bias(AZN)[, tier acc]}} for price predictions.
    MdAPE = median |y - y_hat| / y in percent, bias = mean(y_hat - y).
    """
    groups = np.asarray(groups).astype(str)
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    keys = order if order is not None else sorted(
        set(groups.tolist()), key=lambda g: -int(np.sum(groups == g)))
    out = {}
    for g in keys:
        m = groups == g
        if m.sum() < min_group or m.sum() == 0:
            continue
        yt, yp = y_true[m], y_pred[m]
        row = {
            "n": int(m.sum()),
            "MAE(AZN)": evaluate.mae(yt, yp),
            "MdAPE(%)": float(100 * np.median(np.abs(yt - yp) / yt)),
            "bias(AZN)": float(np.mean(yp - yt)),
        }
        if t_true is not None and t_pred is not None:
            row["tier acc"] = evaluate.accuracy(np.asarray(t_true)[m], np.asarray(t_pred)[m])
        out[g] = row
    return out


def price_range_labels(y_true) -> np.ndarray:
    idx = np.digitize(np.asarray(y_true, dtype=float), PRICE_EDGES)   # edge value -> upper bin
    return np.array(PRICE_LABELS)[idx]


def collapse_sparse(groups, min_group: int):
    """Merge groups with fewer than min_group rows into one 'sparse' label."""
    groups = np.asarray(groups).astype(str)
    vals, counts = np.unique(groups, return_counts=True)
    sparse = set(vals[counts < min_group].tolist())
    label = f"sparse (<{min_group} rows)"
    return np.array([label if g in sparse else g for g in groups]), label


def worst_errors(y_true, y_pred, k: int = 10) -> np.ndarray:
    """Indices of the k largest absolute errors (the luxury / mislabelled listings)."""
    err = np.abs(np.asarray(y_true, float).ravel() - np.asarray(y_pred, float).ravel())
    return np.argsort(-err, kind="mergesort")[:k]


def run(df_te, y_true, y_pred, t_true=None, t_pred=None, name: str = "model",
        table_dir=TABLE_DIR, fig_dir=plots.FIG_DIR, min_group: int = 30,
        top_n: int = 15, worst_k: int = 10) -> dict:
    """
    df_te  : the TEST rows of the cleaned DataFrame (needs 'category' and 'location';
             'area_m2' is shown for the worst cases if present), same order as y_true
    y_true / y_pred : prices in AZN (NOT log)
    t_true / t_pred : optional tier labels / predictions, adds a 'tier acc' column
    Returns {"price range": table, "category": table, "district": table, "worst": table}.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    tag = _slug(name)
    tables = {}

    tables["price range"] = group_table(price_range_labels(y_true), y_true, y_pred,
                                        t_true, t_pred, order=PRICE_LABELS)
    tables["category"] = group_table(df_te["category"].fillna("unknown").to_numpy(),
                                     y_true, y_pred, t_true, t_pred)

    districts, sparse_label = collapse_sparse(df_te["location"].fillna("unknown").to_numpy(),
                                              min_group)
    full = group_table(districts, y_true, y_pred, t_true, t_pred)
    keep = [g for g in full if g != sparse_label][:top_n] + \
           ([sparse_label] if sparse_label in full else [])
    tables["district"] = {g: full[g] for g in keep}

    idx = worst_errors(y_true, y_pred, worst_k)
    worst = {}
    for rank, i in enumerate(idx, 1):
        row = {"true": int(round(y_true[i])), "pred": int(round(y_pred[i])),
               "abs err": int(round(abs(y_true[i] - y_pred[i]))),
               "category": str(df_te["category"].iloc[i]),
               "district": str(df_te["location"].iloc[i])}
        if "area_m2" in df_te.columns:
            row["area_m2"] = int(round(float(df_te["area_m2"].iloc[i])))
        worst[f"#{rank}"] = row
    tables["worst"] = worst

    titles = {"price range": "error by price range", "category": "error by category",
              "district": f"error by district (top {top_n} + sparse)",
              "worst": f"{worst_k} largest errors"}
    for key, table in tables.items():
        print(f"\n[{name}] {titles[key]}")
        evaluate.compare(table)
        evaluate.save_table(table, Path(table_dir) / f"{tag}_{_slug(key)}.md",
                            title=f"{name}: {titles[key]} (test set)")
    for key in ("price range", "category"):
        plots.plot_error_bars(tables[key], "MdAPE(%)", f"{name}: median % error by {key}",
                              f"err_{tag}_{_slug(key)}.png", outdir=fig_dir)
    return tables
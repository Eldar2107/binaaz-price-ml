"""
eda.py — exploratory figures for the report (Problem 1b).

Run from the repo root:
    python -X utf8 -m src.eda

Saves PNGs to report/figures/ (git-ignored on purpose: run_all re-creates them,
so every figure in the report is reproducible from code). run_all should call
    eda.run(df)       # df = data_prep.clean(data_prep.load_raw())
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")                      # no GUI needed; just write files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import data_prep as dp

FIG_DIR = Path("report/figures")
MAX_POINTS = 15_000                        # scatter plots are sub-sampled (seeded)
plt.rcParams.update({"font.size": 9})


def _save(fig, outdir, name) -> Path:
    path = Path(outdir) / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _sample(df: pd.DataFrame) -> pd.DataFrame:
    return df.sample(n=min(MAX_POINTS, len(df)), random_state=dp.SEED)


# --- figures ----------------------------------------------------------------
def fig_price_distribution(df, outdir) -> Path:
    """Raw price is heavy-tailed; log(price) is close to symmetric -> model log(price)."""
    p = df["price"]
    lp = np.log(p)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].hist(p, bins=60, range=(0, p.quantile(0.99)), color="#4C72B0")
    ax[0].set_title(f"price (shown up to 99th pct), skew = {p.skew():.1f}")
    ax[0].set_xlabel("price, AZN")
    ax[0].set_ylabel("listings")
    ax[1].hist(lp, bins=60, color="#55A868")
    ax[1].set_title(f"log(price), skew = {lp.skew():.2f}")
    ax[1].set_xlabel("log(price)")
    return _save(fig, outdir, "01_price_distribution.png")


def fig_area_vs_price(df, outdir) -> Path:
    s = _sample(df)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    cmap = plt.get_cmap("tab10")
    for i, c in enumerate(sorted(s["category"].unique())):
        m = s["category"] == c
        ax.scatter(s.loc[m, "area_m2"], s.loc[m, "price"], s=4, alpha=0.35,
                   color=cmap(i), label=c)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("area, m² (log scale)")
    ax.set_ylabel("price, AZN (log scale)")
    r = np.corrcoef(np.log(df["area_m2"]), np.log(df["price"]))[0, 1]
    ax.set_title(f"area vs price, corr(log, log) = {r:.2f}")
    ax.legend(markerscale=3, fontsize=8)
    return _save(fig, outdir, "02_area_vs_price.png")


def _boxplot_by(ax, groups: dict, rotate=0):
    data = list(groups.values())
    ax.boxplot(data, showfliers=False)
    ax.set_xticks(range(1, len(data) + 1))
    ax.set_xticklabels([f"{k}\n(n={len(v)})" for k, v in groups.items()], rotation=rotate)
    ax.set_yscale("log")
    ax.set_ylabel("price, AZN (log scale)")


def fig_rooms_vs_price(df, outdir) -> Path:
    r = df.dropna(subset=["rooms"]).copy()
    r["rc"] = r["rooms"].clip(upper=6).astype(int)
    groups = {(f"{k}+" if k == 6 else str(k)): g["price"].to_numpy()
              for k, g in sorted(r.groupby("rc"))}
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    _boxplot_by(ax, groups)
    ax.set_xlabel("number of rooms (rows with missing rooms excluded)")
    ax.set_title("price by number of rooms")
    return _save(fig, outdir, "03_rooms_vs_price.png")


def fig_category_vs_price(df, outdir) -> Path:
    order = df.groupby("category")["price"].median().sort_values().index
    groups = {c: df.loc[df["category"] == c, "price"].to_numpy() for c in order}
    fig, ax = plt.subplots(figsize=(8, 4.2))
    _boxplot_by(ax, groups, rotate=15)
    ax.set_title("price by property category (sorted by median)")
    return _save(fig, outdir, "04_category_vs_price.png")


def fig_district_median_price(df, outdir, top_n: int = 15) -> Path:
    g = df.groupby("location")["price"].agg(["median", "count"])
    top = g.sort_values("count", ascending=False).head(top_n).sort_values("median")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top.index, top["median"], color="#4C72B0")
    for y, (med, n) in enumerate(zip(top["median"], top["count"])):
        ax.text(med, y, f"  n={int(n)}", va="center", fontsize=7)
    ax.set_xlabel("median price, AZN")
    ax.set_title(f"median price in the {top_n} districts with most listings")
    return _save(fig, outdir, "05_district_median_price.png")


def fig_location_map(df, outdir) -> Path:
    """lat/lng scatter coloured by log10(price). Window = Baku area (display only)."""
    s = _sample(df)
    c = np.log10(s["price"])
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sc = ax.scatter(s["lng"], s["lat"], c=c, s=3, cmap="viridis",
                    vmin=c.quantile(0.02), vmax=c.quantile(0.98))
    ax.set_xlim(49.6, 50.2)
    ax.set_ylim(40.2, 40.6)
    ax.set_aspect(1 / np.cos(np.radians(40.4)))
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("listings in the Baku area, colour = log10(price)")
    fig.colorbar(sc, ax=ax, label="log10(price, AZN)")
    return _save(fig, outdir, "06_location_map.png")


def fig_tier_balance(df, outdir) -> Path:
    """Class balance of the price tier in each split (threshold = TRAIN median)."""
    y = df["price"].to_numpy(dtype=float)
    tr, va, te = dp.split_indices(y)
    _, thr = dp.make_tier_label(y[tr])
    shares, sizes = [], []
    for idx in (tr, va, te):
        t, _ = dp.make_tier_label(y[idx], thr)
        shares.append(float(t.mean()))
        sizes.append(len(idx))
    print(f"tier threshold (train median) = {thr:,.0f} AZN; "
          f"premium share train/val/test = " + " / ".join(f"{s:.3f}" for s in shares))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["train", "val", "test"], shares, color="#C44E52")
    ax.axhline(0.5, ls="--", color="gray", lw=1)
    for i, (s, n) in enumerate(zip(shares, sizes)):
        ax.text(i, s + 0.01, f"{s:.3f}\n(n={n})", ha="center", fontsize=8)
    ax.set_ylim(0, 0.7)
    ax.set_ylabel("share of premium class")
    ax.set_title(f"price-tier balance (threshold {thr:,.0f} AZN)")
    return _save(fig, outdir, "07_tier_balance.png")


# --- driver -----------------------------------------------------------------
def _print_summary(df: pd.DataFrame) -> None:
    p = df["price"]
    print("\n--- EDA summary ---")
    print(f"listings: {len(df)}")
    print("price (AZN):", p.describe(percentiles=[.25, .5, .75]).round(0).to_dict())
    print(f"skew price = {p.skew():.2f}, skew log(price) = {np.log(p).skew():.2f}")
    print("\nlistings per category:\n" + df["category"].value_counts().to_string())
    print("\nmedian price per category:\n"
          + df.groupby("category")["price"].median().sort_values().round(0).to_string())
    print(f"\nrows with missing rooms: {df['rooms'].isna().mean():.1%}, "
          f"missing floor: {df['floor'].isna().mean():.1%}")


def run(df: pd.DataFrame | None = None, outdir=FIG_DIR) -> list[Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if df is None:
        df = dp.clean(dp.load_raw())
    _print_summary(df)
    figs = [fig_price_distribution, fig_area_vs_price, fig_rooms_vs_price,
            fig_category_vs_price, fig_district_median_price, fig_location_map,
            fig_tier_balance]
    paths = [f(df, outdir) for f in figs]
    print("\nsaved figures:")
    for p in paths:
        print("  ", p)
    return paths


if __name__ == "__main__":
    run()
"""
Tests for src/data_prep.py. They use a small SYNTHETIC frame with the same
columns as the bina.az dump, so they run anywhere (no dataset needed).

Run from the repo root:
    python -X utf8 -m tests.test_data_prep
    python -m pytest tests -q              # if you pip install pytest
"""

import numpy as np
import pandas as pd

from src import data_prep as dp


def _raw(n=300, seed=0):
    rng = np.random.default_rng(seed)
    area = rng.integers(30, 300, n)
    return pd.DataFrame({
        "price": (area * rng.uniform(1500, 4000, n)).round(0),
        "estate_rel_url_x": [f"/items/{i}" for i in range(n)],
        "datetime_scrape_x": ["2024-10-01"] * n,
        "location": rng.choice(["Səbail r.", "Yasamal r.", "Binəqədi r."], n),
        "lat": rng.uniform(40.3, 40.5, n),
        "lng": rng.uniform(49.7, 50.0, n),
        "owner_title": rng.choice(["vasitəçi (agent)", "mülkiyyətçi", None], n),
        "Kateqoriya": rng.choice(["Yeni tikili", "Köhnə tikili", "Torpaq"], n),
        "Otaq sayı": rng.choice([1, 2, 3, np.nan], n),
        "Sahə": [f"{a} m²" for a in area],
        "Torpaq sahəsi": rng.choice([None, "1.3 sot"], n),
        "Mərtəbə": rng.choice(["3 / 9", None], n),
        "Təmir": rng.choice(["var", "yoxdur", None], n),
        "Çıxarış": rng.choice(["var", "yoxdur"], n),
        "İpoteka": rng.choice(["var", None], n),
        "unit_price": "3 450 AZN/m²",
        "total_price": 1.0,
        "views": 5,
    })


# --- cleaning ---------------------------------------------------------------
def test_dedupe_keeps_latest_scrape():
    raw = _raw(50)
    dup = raw.iloc[[0]].copy()
    dup["datetime_scrape_x"] = "2024-10-03"
    dup["price"] = 123456.0
    out = dp.clean(pd.concat([raw, dup], ignore_index=True), verbose=False)
    assert len(out) == 50                              # same listing counted once
    assert (out["price"] == 123456.0).sum() == 1       # and the LATEST version kept


def test_output_has_only_whitelisted_columns_no_leakage():
    out = dp.clean(_raw(100), verbose=False)
    assert list(out.columns) == dp.KEEP_COLUMNS
    for bad in dp.LEAKAGE_COLUMNS + ["views", "owner_name", "address"]:
        assert bad not in out.columns


def test_outlier_filters():
    raw = _raw(50)
    raw.loc[0, "price"] = 11                           # too cheap
    raw.loc[1, "price"] = 6e8                          # too expensive
    raw.loc[2, "Sahə"] = "5 m²"                        # too small
    raw.loc[3, "lat"] = 0.0                            # outside Azerbaijan
    out = dp.clean(raw, verbose=False)
    assert len(out) == 46


def test_area_parsing_units():
    s = pd.Series(["145 m²", "1.3 sot", "1 200 m²", "2 ha", None, "abc"])
    a = dp._parse_area_m2(s).to_numpy(dtype=float)
    assert np.allclose(a[:4], [145, 130, 1200, 20000])
    assert np.isnan(a[4]) and np.isnan(a[5])


# --- features ---------------------------------------------------------------
def test_make_features_shapes_and_no_nan():
    out = dp.clean(_raw(200), verbose=False)
    X, y, names = dp.make_features(out)
    assert X.shape[0] == len(out) == len(y)
    assert X.shape[1] == len(names) == len(set(names))
    assert not np.isnan(X).any()
    assert np.allclose(y, out["price"].to_numpy())


# --- split / tier / scaling -------------------------------------------------
def test_split_is_disjoint_complete_deterministic_and_balanced():
    rng = np.random.default_rng(1)
    y = rng.lognormal(12, 0.8, 5000)
    tr, va, te = dp.split_indices(y)
    all_idx = np.concatenate([tr, va, te])
    assert len(set(tr) & set(va)) == len(set(tr) & set(te)) == len(set(va) & set(te)) == 0
    assert np.array_equal(np.sort(all_idx), np.arange(len(y)))
    assert abs(len(tr) / len(y) - 0.70) < 0.01
    assert abs(len(va) / len(y) - 0.15) < 0.01
    tr2, va2, te2 = dp.split_indices(y)
    assert np.array_equal(tr, tr2) and np.array_equal(te, te2)       # seeded
    thr = np.median(y[tr])
    shares = [float((y[i] > thr).mean()) for i in (tr, va, te)]
    assert max(shares) - min(shares) < 0.02                           # stratified


def test_tier_threshold_comes_from_train_only():
    y_tr = np.array([1.0, 2.0, 3.0, 3.0, 3.0])
    labels, thr = dp.make_tier_label(y_tr)
    assert thr == 3.0 and labels.sum() == 0            # strictly greater than median
    y_val = np.array([10.0, 20.0, 30.0])
    lab_val, thr_val = dp.make_tier_label(y_val, thr)
    assert thr_val == thr                              # val re-uses the train threshold
    assert thr != np.median(y_val)
    assert lab_val.tolist() == [1, 1, 1]


def test_standardize_uses_train_statistics_only():
    rng = np.random.default_rng(2)
    X_tr = rng.normal(5, 2, (500, 3))
    X_tr[:, 0] = 7.0                                   # constant column
    X_val = rng.normal(50, 1, (100, 3))
    Z_tr, Z_val = dp.standardize(X_tr, X_val)
    assert np.allclose(Z_tr[:, 1:].mean(axis=0), 0, atol=1e-9)
    assert np.allclose(Z_tr[:, 1:].std(axis=0), 1, atol=1e-9)
    assert np.allclose(Z_tr[:, 0], 0) and not np.isnan(Z_tr).any()    # std=0 handled
    assert Z_val[:, 1:].mean() > 10                    # val NOT re-centred on itself
    assert dp.standardize(X_tr).shape == X_tr.shape    # single array in, single out


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK ", name)
    print("all data_prep tests passed")
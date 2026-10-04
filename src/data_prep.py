"""
data_prep.py — loading, cleaning, splitting, and the price-tier label.

Keeps ALL data wrangling in one place. Run a quick self-check from the repo root:
    python -X utf8 -m src.data_prep

Design rules (why the code looks the way it does):
  * No statistic from val/test is ever used. Outlier limits are FIXED domain
    constants (below), missing values are filled with sentinels + indicator
    flags (no medians), and the tier threshold comes from TRAIN only.
  * clean() returns a WHITELIST of columns, so a leakage/identifier column can
    never slip into the model by accident.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- Config -----------------------------------------------------------------
SEED = 42
DATA_PATH = "data/bina_az_sale.csv"

# Fixed, domain-based outlier limits (NOT computed from the data -> no leakage).
PRICE_MIN, PRICE_MAX = 10_000.0, 5_000_000.0        # AZN (all rows are AZN)
AREA_MIN, AREA_MAX = 10.0, 20_000.0                 # m2
LAT_RANGE, LNG_RANGE = (38.3, 42.0), (44.7, 51.0)   # bounding box of Azerbaijan
BAKU_CENTER = (40.4093, 49.8671)

# Columns that LEAK the target (derived from price). Never used as features.
#   total_price : identical to price (same 2772 unique values)
#   unit_price  : string like "3 450 AZN/m2" = price / area
LEAKAGE_COLUMNS: list[str] = ["unit_price", "total_price"]

# Azerbaijani column names -> English. Matching is done on an ASCII-folded
# version of the name, so you never have to type special characters.
AZ_RENAME = {
    "binanin novu": "building_type",   # 99.8% missing -> dropped
    "kateqoriya": "category",
    "mertebe": "floor_raw",            # "7 / 9"  = floor / total floors
    "otaq sayi": "rooms",
    "sahe": "area_raw",                # "145 m2"
    "torpaq sahesi": "land_raw",       # "1.3 sot"
    "temir": "repair_status",          # var / yoxdur / NaN
    "cixaris": "bill_of_sale_status",  # var / yoxdur
    "ipoteka": "mortgage_status",      # var / NaN
}
_AZ_FOLD = str.maketrans({
    "ı": "i", "İ": "i", "ə": "e", "Ə": "e", "ö": "o", "Ö": "o",
    "ü": "u", "Ü": "u", "ç": "c", "Ç": "c", "ş": "s", "Ş": "s",
    "ğ": "g", "Ğ": "g",
})

# Final, explicit list of columns clean() returns (price = target).
KEEP_COLUMNS = [
    "price", "category", "location", "area_m2", "land_sot", "rooms",
    "floor", "floors_total", "has_repair", "repair_unknown",
    "has_bill_of_sale", "has_mortgage", "is_agent", "lat", "lng",
]

# Everything NOT in KEEP_COLUMNS is dropped. Reasons (put these in the report):
#   identifiers/urls/images : id_x, id_y, estate_id, *_url*, img_url, estate_details_id_*
#   people                   : owner_name, shop_name, shop_title, address (raw)
#   leakage                  : unit_price, total_price
#   not known at listing time: views, updated, day_*, hour_*, datetime_scrape_*
#   paid-promotion flags     : vip, featured, products_label (proxy for price tier)
#   duplicates of kept info  : repair, bill_of_sale, mortgage, attributes, extra_info
#   near-constant            : currency_*, city (99.7% Baku), building_type (99.8% missing)
#   free text (optional later): description


def _fold(name) -> str:
    return str(name).translate(_AZ_FOLD).strip().lower()


def load_raw(path: str = DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV (UTF-8). Returned unmodified."""
    return pd.read_csv(path, encoding="utf-8", low_memory=False)


# --- parsing helpers --------------------------------------------------------
def _parse_area_m2(s: pd.Series) -> pd.Series:
    """'145 m2' -> 145 ; '1.3 sot' -> 130 ; '2 ha' -> 20000 ; else NaN."""
    s = s.astype("string").str.lower()
    s = s.str.replace(r"(?<=\d)\s(?=\d)", "", regex=True)       # "1 200" -> "1200"
    num = s.str.extract(r"(\d+(?:[.,]\d+)?)")[0].str.replace(",", ".", regex=False)
    num = pd.to_numeric(num, errors="coerce").astype(float)
    is_sot = s.str.contains("sot", na=False).to_numpy(dtype=bool)
    is_ha = s.str.contains(r"\bha\b|hektar", regex=True, na=False).to_numpy(dtype=bool)
    factor = np.where(is_sot, 100.0, np.where(is_ha, 10_000.0, 1.0))
    return num * factor


def _yes(s: pd.Series) -> pd.Series:
    """'var' -> 1, anything else (yoxdur / NaN) -> 0."""
    return (s.astype("string").str.strip().str.lower().eq("var")
            .fillna(False).astype(int))


def _haversine_km(lat, lng, lat0, lng0):
    p1, p2 = np.radians(lat), np.radians(lat0)
    dphi, dl = p2 - p1, np.radians(lng0 - lng)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


# --- cleaning ---------------------------------------------------------------
def clean(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Steps (each logs how many rows it removed):
      1. translate Azerbaijani column names,
      2. de-duplicate: the dump holds the SAME listing several times
         (same estate_rel_url_x, different scrape time) -> keep the latest,
      3. parse text -> numbers (area, land, floor/total floors, var/yoxdur flags),
      4. drop rows without a usable area,
      5. outliers: FIXED price / area / coordinate limits,
      6. keep only KEEP_COLUMNS (drops leakage + identifiers by construction).
    """
    log = []

    def step(msg: str, before: int, after: int):
        log.append(f"{msg:<48s} {before:>7d} -> {after:>7d}  (removed {before - after})")

    df = df.rename(columns=lambda c: AZ_RENAME.get(_fold(c), c)).copy()
    required = ["price", "location", "lat", "lng", "owner_title",
                "estate_rel_url_x", "datetime_scrape_x", "category", "rooms",
                "area_raw", "land_raw", "floor_raw", "repair_status",
                "bill_of_sale_status", "mortgage_status"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Expected columns not found: {missing}. "
                       f"Columns present: {list(df.columns)}")

    n = len(df)
    # 2. duplicates (same listing scraped several times)
    df = (df.sort_values("datetime_scrape_x", kind="stable")
            .drop_duplicates(subset="estate_rel_url_x", keep="last"))
    step("de-duplicate by listing url (keep latest scrape)", n, len(df))

    # 3. parsing
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["area_m2"] = _parse_area_m2(df["area_raw"])
    df["land_sot"] = (_parse_area_m2(df["land_raw"]) / 100.0).fillna(0.0)
    fl = df["floor_raw"].astype("string").str.extract(r"(\d+)\s*/\s*(\d+)")
    df["floor"] = pd.to_numeric(fl[0], errors="coerce")
    df["floors_total"] = pd.to_numeric(fl[1], errors="coerce")
    df["rooms"] = pd.to_numeric(df["rooms"], errors="coerce")
    df["has_repair"] = _yes(df["repair_status"])
    df["repair_unknown"] = df["repair_status"].isna().astype(int)
    df["has_bill_of_sale"] = _yes(df["bill_of_sale_status"])
    df["has_mortgage"] = _yes(df["mortgage_status"])
    df["is_agent"] = (df["owner_title"].astype("string")
                      .str.contains("agent", case=False, na=False).astype(int))
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lng"] = pd.to_numeric(df["lng"], errors="coerce")

    # 4. + 5. filters
    n = len(df)
    df = df[df["area_m2"].notna() & df["price"].notna()]
    step("drop rows with unparseable area/price", n, len(df))

    n = len(df)
    df = df[df["price"].between(PRICE_MIN, PRICE_MAX)]
    step(f"price outside [{PRICE_MIN:,.0f}, {PRICE_MAX:,.0f}] AZN", n, len(df))

    n = len(df)
    df = df[df["area_m2"].between(AREA_MIN, AREA_MAX)]
    step(f"area outside [{AREA_MIN:.0f}, {AREA_MAX:,.0f}] m2", n, len(df))

    n = len(df)
    df = df[df["lat"].between(*LAT_RANGE) & df["lng"].between(*LNG_RANGE)]
    step("coordinates outside Azerbaijan", n, len(df))

    # 6. whitelist
    out = df[KEEP_COLUMNS].reset_index(drop=True)
    assert not set(LEAKAGE_COLUMNS) & set(out.columns), "leakage column kept!"

    if verbose:
        print("--- cleaning log ---")
        print("\n".join(log))
        print(f"final shape: {out.shape}")
    return out


# --- features ---------------------------------------------------------------
def make_features(df: pd.DataFrame):
    """
    Cleaned frame -> numeric design matrix.
      * missing numerics: filled with 0 + a *_missing flag (no statistics used,
        so nothing can leak from val/test),
      * category + location: one-hot over all levels (vocabulary only, no target
        information; rare districts simply get near-empty columns),
      * dist_center_km: haversine distance to the centre of Baku from lat/lng.
    Return (X ndarray, y price ndarray, feature_names).
    """
    d = df
    num = pd.DataFrame({
        "area_m2": d["area_m2"],
        "land_sot": d["land_sot"],
        "rooms": d["rooms"].fillna(0.0),
        "rooms_missing": d["rooms"].isna().astype(float),
        "floor": d["floor"].fillna(0.0),
        "floors_total": d["floors_total"].fillna(0.0),
        "floor_missing": d["floor"].isna().astype(float),
        "has_repair": d["has_repair"].astype(float),
        "repair_unknown": d["repair_unknown"].astype(float),
        "has_bill_of_sale": d["has_bill_of_sale"].astype(float),
        "has_mortgage": d["has_mortgage"].astype(float),
        "is_agent": d["is_agent"].astype(float),
        "lat": d["lat"],
        "lng": d["lng"],
        "dist_center_km": _haversine_km(d["lat"], d["lng"], *BAKU_CENTER),
    })
    cats = pd.get_dummies(d[["category", "location"]].fillna("unknown"),
                          prefix=["cat", "loc"], dtype=float)
    X = pd.concat([num, cats], axis=1)
    return X.to_numpy(dtype=float), d["price"].to_numpy(dtype=float), list(X.columns)


# --- tier label -------------------------------------------------------------
def make_tier_label(y_price, threshold=None):
    """
    premium = 1 (price > threshold), standard = 0.
    Call it ONCE on the TRAIN prices (threshold=None -> train median), then
    reuse the returned threshold for val and test:
        y_tr_tier, thr = make_tier_label(y_tr)
        y_val_tier, _  = make_tier_label(y_val, thr)
        y_te_tier,  _  = make_tier_label(y_te,  thr)
    """
    y_price = np.asarray(y_price, dtype=float)
    if threshold is None:
        threshold = float(np.median(y_price))
    return (y_price > threshold).astype(int), threshold


# --- split ------------------------------------------------------------------
def _strata(y, n_bins: int = 10):
    """Strata for the split: y itself if it has few values, else price deciles.
    (Used ONLY to balance the splits; the tier threshold is still train-only.)"""
    y = np.asarray(y)
    if len(np.unique(y)) <= n_bins:
        return y
    edges = np.quantile(y, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.searchsorted(edges, y, side="right")


def split_indices(y, val_size=0.15, test_size=0.15, seed=SEED):
    """Deterministic stratified split -> (idx_train, idx_val, idx_test).
    Handy for error analysis (map test rows back to the cleaned DataFrame)."""
    rng = np.random.default_rng(seed)
    strata = _strata(y)
    tr, va, te = [], [], []
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        rng.shuffle(idx)
        n_te = int(round(test_size * len(idx)))
        n_va = int(round(val_size * len(idx)))
        te.append(idx[:n_te])
        va.append(idx[n_te:n_te + n_va])
        tr.append(idx[n_te + n_va:])
    return tuple(np.sort(np.concatenate(p)) for p in (tr, va, te))


def train_val_test_split(X, y, val_size=0.15, test_size=0.15, seed=SEED):
    """Return (X_tr, y_tr, X_val, y_val, X_te, y_te)."""
    X, y = np.asarray(X), np.asarray(y)
    tr, va, te = split_indices(y, val_size, test_size, seed)
    return X[tr], y[tr], X[va], y[va], X[te], y[te]


# --- scaling ----------------------------------------------------------------
def standardize(X_tr, *others):
    """
    Standardize with TRAIN mean/std only; apply the same transform to the rest.
    Returns the scaled arrays in the order given (one array if only X_tr given).
    """
    X_tr = np.asarray(X_tr, dtype=float)
    mean = X_tr.mean(axis=0)
    std = X_tr.std(axis=0)
    std[std == 0] = 1.0                      # constant columns stay 0
    scaled = [(X_tr - mean) / std] + [(np.asarray(o, dtype=float) - mean) / std
                                      for o in others]
    return scaled[0] if len(scaled) == 1 else tuple(scaled)


# --- quick self-check -------------------------------------------------------
if __name__ == "__main__":
    raw = load_raw()
    df = clean(raw)
    print("\nrows per category:\n", df["category"].value_counts().to_string())
    X, y, names = make_features(df)
    X_tr, y_tr, X_val, y_val, X_te, y_te = train_val_test_split(X, y)
    print(f"\nX: {X.shape}  features: {len(names)}")
    print(f"split sizes  train={len(y_tr)}  val={len(y_val)}  test={len(y_te)}")
    t_tr, thr = make_tier_label(y_tr)
    t_val, _ = make_tier_label(y_val, thr)
    t_te, _ = make_tier_label(y_te, thr)
    print(f"tier threshold (train median) = {thr:,.0f} AZN")
    print(f"premium share  train={t_tr.mean():.3f}  val={t_val.mean():.3f}  test={t_te.mean():.3f}")
    X_tr_s, X_val_s, X_te_s = standardize(X_tr, X_val, X_te)
    print(f"scaled train mean~{X_tr_s.mean():.3f}  NaNs: {np.isnan(X_tr_s).sum()}")
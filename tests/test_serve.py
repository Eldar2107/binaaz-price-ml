"""
Tests for src/serve.py (the model + one-listing prediction used by the web app).
Synthetic data only. Run:  python -X utf8 -m tests.test_serve   or   python -m pytest tests -q
"""

import tempfile
from pathlib import Path

import numpy as np

from src import data_prep as dp
from src import serve
from tests.test_data_prep import _raw


def _bundle(n=600):
    df = dp.clean(_raw(n), verbose=False)
    return df, serve.build_bundle(df, depth_reg=4, depth_cls=4)


def _listing_from_row(r):
    return {"category": r["category"], "district": r["location"], "area_m2": r["area_m2"],
            "land_sot": r["land_sot"], "rooms": r["rooms"], "floor": r["floor"],
            "floors_total": r["floors_total"], "has_repair": r["has_repair"],
            "has_bill_of_sale": r["has_bill_of_sale"], "has_mortgage": r["has_mortgage"],
            "is_agent": r["is_agent"]}


def test_vector_matches_make_features_except_location_columns():
    df, bundle = _bundle()
    X, _, names = dp.make_features(df)
    assert names == bundle["feature_names"]
    skip = [i for i, n in enumerate(names) if n in serve.NON_FEATURE_COLS]
    keep = [i for i in range(len(names)) if i not in skip]
    checked = 0
    for i in np.flatnonzero(df["repair_unknown"].to_numpy() == 0)[:25]:
        vec, known = serve.vectorize(bundle, _listing_from_row(df.iloc[i]))
        assert known and np.allclose(vec[keep], X[i, keep]), names
        checked += 1
    assert checked > 0


def test_predict_returns_sane_values_for_every_model_pair():
    df, bundle = _bundle()
    listing = _listing_from_row(df.iloc[0])
    for reg in bundle["models"]["regression"]:
        for tier in bundle["models"]["tier"]:
            out = serve.predict(bundle, listing, reg, tier)
            assert out["price_azn"] > 0 and np.isfinite(out["price_azn"])
            assert isinstance(out["is_premium"], bool)
            assert out["score_kind"] in ("probability", "margin")
            if out["score_kind"] == "probability":
                assert 0.0 <= out["score"] <= 1.0
            assert out["threshold"] == bundle["threshold"]


def test_bigger_area_never_breaks_and_unseen_district_is_flagged():
    df, bundle = _bundle()
    listing = _listing_from_row(df.iloc[0])
    listing["district"] = "Nowhere r."
    out = serve.predict(bundle, listing, "sklearn tree", "sklearn tree")
    assert out["known_inputs"] is False and out["price_azn"] > 0
    listing = _listing_from_row(df.iloc[0])
    listing["area_m2"] = 5000
    assert serve.predict(bundle, listing, "sklearn tree", "sklearn LinearSVC")["price_azn"] > 0


def test_bundle_roundtrip_on_disk():
    df, bundle = _bundle()
    with tempfile.TemporaryDirectory() as d:
        path = serve.save_bundle(bundle, Path(d) / "m" / "bundle.pkl")
        again = serve.load_bundle(path)
    listing = _listing_from_row(df.iloc[3])
    a = serve.predict(bundle, listing, "sklearn tree", "sklearn tree")
    b = serve.predict(again, listing, "sklearn tree", "sklearn tree")
    assert a == b


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK ", name)
    print("all serve tests passed")

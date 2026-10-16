"""
serve.py - train once, save, and predict ONE listing (used by the web app).

The web app never re-implements any feature logic: a listing is turned into a
one-row table in the same format clean() returns, and data_prep.make_features
builds its vector, aligned to the training feature names. So the app and the
experiments cannot drift apart.

Models live in registries (name -> model), exactly like run_all.py:
    "regression": name -> model that predicts log(price)
    "tier":       name -> (model, uses standardized features?)
Includes both scikit-learn baselines and our from-scratch DecisionTree and PegasosSVM.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from . import cv
from . import data_prep as dp
from .decision_tree import DecisionTree
from .svm import PegasosSVM

BUNDLE_PATH = Path("models") / "bundle.pkl"
NON_FEATURE_COLS = {"lat", "lng", "dist_center_km"}     # filled from the district centre


def build_bundle(df: pd.DataFrame, depth_reg: int = 12, depth_cls: int = 16) -> dict:
    """Fit every model on the TRAIN split (the same split the report uses)."""
    X, y, names = dp.make_features(df)
    tr, _, _ = dp.split_indices(y)
    t_tr, thr = dp.make_tier_label(y[tr])
    mean, std = X[tr].mean(axis=0), X[tr].std(axis=0)
    std[std == 0] = 1.0
    Xs_tr = (X[tr] - mean) / std
    seed = dp.SEED
    models = {
        "regression": {
            "sklearn tree": DecisionTreeRegressor(max_depth=depth_reg, random_state=seed)
                            .fit(X[tr], np.log(y[tr])),
            "our tree": DecisionTree(task="regression", criterion="mse",
                                     max_depth=depth_reg, random_state=seed)
                        .fit(X[tr], np.log(y[tr])),
        },
        "tier": {
            "sklearn tree": (DecisionTreeClassifier(max_depth=depth_cls, random_state=seed)
                             .fit(X[tr], t_tr), False),
            "our tree": (DecisionTree(task="classification", criterion="gini",
                                      max_depth=depth_cls, random_state=seed)
                         .fit(X[tr], t_tr), False),
            "sklearn LinearSVC": (LinearSVC(C=1.0, dual=False, random_state=seed, max_iter=1000)
                                  .fit(Xs_tr, t_tr), True),
            "our Pegasos SVM": (PegasosSVM(lambda_=1e-3, n_epochs=10, random_state=seed)
                                .fit(Xs_tr, t_tr), True),
        },
    }
    centres = (df.assign(location=df["location"].fillna("unknown"))
                 .groupby("location")[["lat", "lng"]].median())
    counts = df["location"].fillna("unknown").value_counts()
    return {
        "feature_names": names,
        "scaler": (mean, std),
        "threshold": thr,
        "models": models,
        "districts": {d: (float(r.lat), float(r.lng)) for d, r in centres.iterrows()},
        "district_order": list(counts.index),                # most listings first
        "categories": sorted(df["category"].dropna().unique()),
        "n_train": int(len(tr)),
    }


def save_bundle(bundle: dict, path: Path = BUNDLE_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    return path


def load_bundle(path: Path = BUNDLE_PATH) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def vectorize(bundle: dict, listing: dict) -> tuple[np.ndarray, bool]:
    """listing -> feature vector in training order. Second value: was the district
    AND category seen in training (unseen ones silently map to all-zero dummies)."""
    lat, lng = bundle["districts"].get(listing["district"], (np.nan, np.nan))
    if np.isnan(lat):                                       # unseen district: city centre
        lat, lng = dp.BAKU_CENTER
    row = pd.DataFrame([{
        "price": np.nan,
        "category": listing["category"],
        "location": listing["district"],
        "area_m2": float(listing["area_m2"]),
        "land_sot": float(listing.get("land_sot", 0.0)),
        "rooms": listing.get("rooms", np.nan),
        "floor": listing.get("floor", np.nan),
        "floors_total": listing.get("floors_total", np.nan),
        "has_repair": int(bool(listing.get("has_repair", False))),
        "repair_unknown": 0,
        "has_bill_of_sale": int(bool(listing.get("has_bill_of_sale", False))),
        "has_mortgage": int(bool(listing.get("has_mortgage", False))),
        "is_agent": int(bool(listing.get("is_agent", False))),
        "lat": lat, "lng": lng,
    }])
    X_row, _, row_names = dp.make_features(row)
    pos = {n: i for i, n in enumerate(bundle["feature_names"])}
    vec = np.zeros(len(pos))
    for name, val in zip(row_names, X_row[0]):
        if name in pos:
            vec[pos[name]] = val
    known = (f"loc_{listing['district']}" in pos) and (f"cat_{listing['category']}" in pos)
    return vec, known


def predict(bundle: dict, listing: dict, reg_name: str, tier_name: str) -> dict:
    """Price estimate (AZN) and premium/standard verdict for one listing."""
    vec, known = vectorize(bundle, listing)
    reg = bundle["models"]["regression"][reg_name]
    price = float(np.exp(np.asarray(reg.predict(vec[None, :]), dtype=float).ravel()[0]))
    model, scaled = bundle["models"]["tier"][tier_name]
    x = ((vec - bundle["scaler"][0]) / bundle["scaler"][1]) if scaled else vec
    score = float(cv.model_scores(model, x[None, :])[0])
    has_proba = hasattr(model, "predict_proba")
    return {
        "price_azn": price,
        "is_premium": bool(np.asarray(model.predict(x[None, :])).ravel()[0] == 1),
        "score": score,
        "score_kind": "probability" if has_proba else "margin",
        "threshold": bundle["threshold"],
        "known_inputs": known,
    }

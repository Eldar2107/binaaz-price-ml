"""
run_all.py - ONE command reproduces every headline number and figure.

Run from the repo root:
    python -m src.run_all

Stage 1 (this version): data, EDA figures, scikit-learn baselines, tables.
Stage 2 (after feat/tree and feat/svm are merged): add our own models where
marked  # TODO(stage 2).
"""

from __future__ import annotations

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from . import data_prep, eda, evaluate
# from .decision_tree import DecisionTree      # TODO(stage 2)
# from .svm import PegasosSVM                  # TODO(stage 2)

SEED = data_prep.SEED
DEPTHS = [4, 6, 8, 10, 12, 16]


def _prf(y_true, y_pred):
    """precision, recall, f1 whether evaluate returns a tuple or a dict."""
    out = evaluate.precision_recall_f1(y_true, y_pred)
    if isinstance(out, dict):
        vals = list(out.values())
        return vals[0], vals[1], vals[2]
    return out[0], out[1], out[2]


def _header(text: str) -> None:
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def main(make_figures: bool = True) -> None:
    np.random.seed(SEED)

    # 1. data -------------------------------------------------------------
    _header("1. DATA")
    df = data_prep.clean(data_prep.load_raw())
    X, y, feat_names = data_prep.make_features(df)
    X_tr, y_tr, X_val, y_val, X_te, y_te = data_prep.train_val_test_split(X, y)
    print(f"features: {len(feat_names)}   "
          f"train/val/test: {len(y_tr)} / {len(y_val)} / {len(y_te)}")

    # regression target = log(price); classification target = tier (train median)
    ylog_tr, ylog_val, ylog_te = np.log(y_tr), np.log(y_val), np.log(y_te)
    t_tr, thr = data_prep.make_tier_label(y_tr)
    t_val, _ = data_prep.make_tier_label(y_val, thr)
    t_te, _ = data_prep.make_tier_label(y_te, thr)
    print(f"tier threshold (train median): {thr:,.0f} AZN   "
          f"premium share train/val/test: "
          f"{t_tr.mean():.3f} / {t_val.mean():.3f} / {t_te.mean():.3f}")

    # standardized copy for the SVM (train statistics only)
    Xs_tr, Xs_val, Xs_te = data_prep.standardize(X_tr, X_val, X_te)

    # 2. EDA figures ------------------------------------------------------
    if make_figures:
        _header("2. EDA FIGURES")
        eda.run(df)

    # 3. Task A: regression (log price) -----------------------------------
    _header("3. TASK A - REGRESSION (target: log price)")
    best_d, best_rmse = None, np.inf
    for d in DEPTHS:
        m = DecisionTreeRegressor(max_depth=d, random_state=SEED).fit(X_tr, ylog_tr)
        r = evaluate.rmse(ylog_val, m.predict(X_val))
        print(f"  sklearn tree depth={d:<3} val RMSE(log)={r:.4f}")
        if r < best_rmse:
            best_d, best_rmse = d, r
    print(f"  -> chosen depth: {best_d}")
    sk_reg = DecisionTreeRegressor(max_depth=best_d, random_state=SEED).fit(X_tr, ylog_tr)
    p_log = sk_reg.predict(X_te)
    p_azn = np.exp(p_log)

    reg_results = {
        "sklearn tree": {
            "RMSE(log)": evaluate.rmse(ylog_te, p_log),
            "R2(log)": evaluate.r2(ylog_te, p_log),
            "RMSE(AZN)": evaluate.rmse(y_te, p_azn),
            "MAE(AZN)": evaluate.mae(y_te, p_azn),
        },
        # TODO(stage 2): "our tree": {...same metrics from DecisionTree...}
    }
    print("\nTest set:")
    evaluate.compare(reg_results)

    # 4. Task B: classification (price tier) ------------------------------
    _header("4. TASK B - CLASSIFICATION (premium vs standard)")
    best_d, best_f1 = None, -1.0
    for d in DEPTHS:
        m = DecisionTreeClassifier(max_depth=d, random_state=SEED).fit(X_tr, t_tr)
        f1 = _prf(t_val, m.predict(X_val))[2]
        print(f"  sklearn tree depth={d:<3} val F1={f1:.4f}")
        if f1 > best_f1:
            best_d, best_f1 = d, f1
    print(f"  -> chosen depth: {best_d}")
    sk_tree = DecisionTreeClassifier(max_depth=best_d, random_state=SEED).fit(X_tr, t_tr)
    sk_svm = LinearSVC(C=1.0, dual=False, random_state=SEED, max_iter=1000).fit(Xs_tr, t_tr)

    def cls_metrics(y_true, y_pred, scores):
        p, r, f1 = _prf(y_true, y_pred)
        return {
            "Accuracy": evaluate.accuracy(y_true, y_pred),
            "Precision": p,
            "Recall": r,
            "F1": f1,
            "ROC-AUC": evaluate.roc_auc(y_true, scores),
        }

    cls_results = {
        "sklearn tree": cls_metrics(t_te, sk_tree.predict(X_te),
                                    sk_tree.predict_proba(X_te)[:, 1]),
        "sklearn LinearSVC": cls_metrics(t_te, sk_svm.predict(Xs_te),
                                         sk_svm.decision_function(Xs_te)),
        # TODO(stage 2): "our tree": ..., "our Pegasos SVM": ...
    }
    print("\nTest set:")
    evaluate.compare(cls_results)

    print("\nconfusion matrix, sklearn tree (rows=true, cols=pred):")
    print(evaluate.confusion_matrix(t_te, sk_tree.predict(X_te)))

    # 5. confusion / ROC figures: TODO(stage 2) once all models exist --------


if __name__ == "__main__":
    main()

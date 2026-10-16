"""
run_all.py - ONE command reproduces every headline number and figure.

Run from the repo root:
    python -m src.run_all

Outputs
    report/figures/*.png   EDA, confusion matrices, ROC curves, error plots
    report/tables/*.md,csv test-set tables, 5-fold CV tables, data summary,
                           error-analysis tables  (every number in the report)

Stage 1 (done): data, EDA, scikit-learn baselines, tables, CV, figures, errors.
Stage 2: add OUR models by adding entries to the registries marked
    # TODO(stage 2)
Everything downstream (tables, CV, confusion matrices, ROC, error analysis)
iterates over those registries, so nothing else has to change.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from . import cv, data_prep, eda, error_analysis, evaluate, plots
from .decision_tree import DecisionTree
from .svm import PegasosSVM

SEED = data_prep.SEED
DEPTHS = [4, 6, 8, 10, 12, 16]
TABLE_DIR = Path("report") / "tables"


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


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_")


def _reg_metrics(ylog_true, p_log):
    return {
        "RMSE(log)": evaluate.rmse(ylog_true, p_log),
        "R2(log)": evaluate.r2(ylog_true, p_log),
        "RMSE(AZN)": evaluate.rmse(np.exp(ylog_true), np.exp(p_log)),
        "MAE(AZN)": evaluate.mae(np.exp(ylog_true), np.exp(p_log)),
    }


def _cls_metrics(y_true, y_pred, scores):
    p, r, f1 = _prf(y_true, y_pred)
    return {
        "Accuracy": evaluate.accuracy(y_true, y_pred),
        "Precision": p,
        "Recall": r,
        "F1": f1,
        "ROC-AUC": evaluate.roc_auc(y_true, scores),
    }


def _tier_error_table(y_true, t_true, t_pred) -> dict:
    """Tier mistakes per price range: where do premium/standard errors happen?
    (They should pile up around the threshold; luxury listings are easy.)"""
    labels = error_analysis.price_range_labels(y_true)
    out = {}
    for g in error_analysis.PRICE_LABELS:
        m = labels == g
        if not m.any():
            continue
        yt, yp = t_true[m], t_pred[m]
        out[g] = {"n": int(m.sum()),
                  "error rate": float(np.mean(yt != yp)),
                  "false pos": int(np.sum((yt == 0) & (yp == 1))),
                  "false neg": int(np.sum((yt == 1) & (yp == 0)))}
    return out


def _write_data_summary(path, clean_log, n_feat, sizes, thr, shares) -> None:
    lines = ["### Data summary (cleaning log, split sizes, tier threshold)", "",
             "```", clean_log, "```", "",
             f"- features after encoding: {n_feat}",
             f"- split sizes train / val / test: {sizes[0]:,} / {sizes[1]:,} / {sizes[2]:,}",
             f"- tier threshold (median of TRAIN prices): {thr:,.0f} AZN",
             f"- premium share train / val / test: "
             f"{shares[0]:.3f} / {shares[1]:.3f} / {shares[2]:.3f}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(make_figures: bool = True) -> None:
    np.random.seed(SEED)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. data -------------------------------------------------------------
    _header("1. DATA")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):          # keep the log: it goes to a table file
        df = data_prep.clean(data_prep.load_raw())
    clean_log = buf.getvalue().rstrip()
    print(clean_log)
    X, y, feat_names = data_prep.make_features(df)
    X_tr, y_tr, X_val, y_val, X_te, y_te = data_prep.train_val_test_split(X, y)
    # same seeded split as row indices, to map test rows back to the DataFrame
    tr_idx, va_idx, te_idx = data_prep.split_indices(y)
    assert np.array_equal(y[te_idx], y_te) and np.array_equal(y[tr_idx], y_tr)
    df_te = df.iloc[te_idx].reset_index(drop=True)
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
    _write_data_summary(TABLE_DIR / "data_summary.md", clean_log, len(feat_names),
                        (len(y_tr), len(y_val), len(y_te)), thr,
                        (t_tr.mean(), t_val.mean(), t_te.mean()))

    # standardized copy for the SVM (train statistics only)
    Xs_tr, Xs_val, Xs_te = data_prep.standardize(X_tr, X_val, X_te)

    # 2. EDA figures ------------------------------------------------------
    if make_figures:
        _header("2. EDA FIGURES")
        eda.run(df)

    # 3. Task A: regression (log price) -----------------------------------
    _header("3. TASK A - REGRESSION (target: log price)")
    best_d_reg, best_rmse = None, np.inf
    for d in DEPTHS:
        m = DecisionTreeRegressor(max_depth=d, random_state=SEED).fit(X_tr, ylog_tr)
        r = evaluate.rmse(ylog_val, m.predict(X_val))
        print(f"  sklearn tree depth={d:<3} val RMSE(log)={r:.4f}")
        if r < best_rmse:
            best_d_reg, best_rmse = d, r
    print(f"  -> chosen depth: {best_d_reg}")

    # registry: name -> FITTED model that predicts log(price)
    reg_models = {
        "sklearn tree": DecisionTreeRegressor(max_depth=best_d_reg, random_state=SEED)
                        .fit(X_tr, ylog_tr),
        "our tree": DecisionTree(task="regression", criterion="mse",
                                 max_depth=best_d_reg, random_state=SEED).fit(X_tr, ylog_tr),
    }
    reg_factories = {
        "sklearn tree": lambda: DecisionTreeRegressor(max_depth=best_d_reg, random_state=SEED),
        "our tree": lambda: DecisionTree(task="regression", criterion="mse",
                                         max_depth=best_d_reg, random_state=SEED),
    }
    reg_pred_log = {n: np.asarray(m.predict(X_te), dtype=float).ravel()
                    for n, m in reg_models.items()}
    reg_results = {n: _reg_metrics(ylog_te, p) for n, p in reg_pred_log.items()}
    print("\nTest set:")
    evaluate.compare(reg_results)
    evaluate.save_table(reg_results, TABLE_DIR / "task_a_test",
                        title="Task A (regression), test set")

    # 4. Task B: classification (price tier) ------------------------------
    _header("4. TASK B - CLASSIFICATION (premium vs standard)")
    best_d_cls, best_f1 = None, -1.0
    for d in DEPTHS:
        m = DecisionTreeClassifier(max_depth=d, random_state=SEED).fit(X_tr, t_tr)
        f1 = _prf(t_val, m.predict(X_val))[2]
        print(f"  sklearn tree depth={d:<3} val F1={f1:.4f}")
        if f1 > best_f1:
            best_d_cls, best_f1 = d, f1
    print(f"  -> chosen depth: {best_d_cls}")

    # registry: name -> (FITTED model, uses standardized features?)
    cls_models = {
        "sklearn tree": (DecisionTreeClassifier(max_depth=best_d_cls, random_state=SEED)
                         .fit(X_tr, t_tr), False),
        "sklearn LinearSVC": (LinearSVC(C=1.0, dual=False, random_state=SEED, max_iter=1000)
                              .fit(Xs_tr, t_tr), True),
        "our tree": (DecisionTree(task="classification", criterion="gini",
                                  max_depth=best_d_cls, random_state=SEED)
                     .fit(X_tr, t_tr), False),
        "our Pegasos SVM": (PegasosSVM(lambda_=1e-3, n_epochs=10, random_state=SEED)
                            .fit(Xs_tr, t_tr), True),
    }
    # registry: name -> (factory of an UNFITTED model, uses standardized features?)
    cls_factories = {
        "sklearn tree": (lambda: DecisionTreeClassifier(max_depth=best_d_cls,
                                                        random_state=SEED), False),
        "sklearn LinearSVC": (lambda: LinearSVC(C=1.0, dual=False, random_state=SEED,
                                                max_iter=1000), True),
        "our tree": (lambda: DecisionTree(task="classification", criterion="gini",
                                           max_depth=best_d_cls, random_state=SEED), False),
        "our Pegasos SVM": (lambda: PegasosSVM(lambda_=1e-3, n_epochs=10, random_state=SEED), True),
    }
    cls_pred, cls_score, cls_results, cms = {}, {}, {}, {}
    for name, (model, scaled) in cls_models.items():
        Xe = Xs_te if scaled else X_te
        cls_pred[name] = np.asarray(model.predict(Xe)).ravel().astype(int)
        cls_score[name] = cv.model_scores(model, Xe)
        cls_results[name] = _cls_metrics(t_te, cls_pred[name], cls_score[name])
        cms[name] = evaluate.confusion_matrix(t_te, cls_pred[name])
    print("\nTest set:")
    evaluate.compare(cls_results)
    evaluate.save_table(cls_results, TABLE_DIR / "task_b_test",
                        title="Task B (price tier), test set")
    for name, cm in cms.items():
        print(f"\nconfusion matrix, {name} (rows=true, cols=pred):")
        print(cm)

    # 5. cross-validation: mean +/- std over 5 folds (Problem 5c) ----------
    _header("5. 5-FOLD CROSS-VALIDATION on train+val (mean ± std)")
    dev = np.sort(np.concatenate([tr_idx, va_idx]))      # the test set is NOT used here
    X_dev, y_dev = X[dev], y[dev]
    reg_cv = {n: cv.cv_regression(f, X_dev, y_dev) for n, f in reg_factories.items()}
    print("Task A (log price):")
    evaluate.compare(reg_cv)
    evaluate.save_table(reg_cv, TABLE_DIR / "task_a_cv",
                        title="Task A (regression), 5-fold CV on train+val, mean ± std")
    cls_cv = {n: cv.cv_classification(f, X_dev, y_dev, scale=s)
              for n, (f, s) in cls_factories.items()}
    print("\nTask B (price tier):")
    evaluate.compare(cls_cv)
    evaluate.save_table(cls_cv, TABLE_DIR / "task_b_cv",
                        title="Task B (price tier), 5-fold CV on train+val, mean ± std")

    # 6. evaluation figures -------------------------------------------------
    if make_figures:
        _header("6. EVALUATION FIGURES")
        figs = [
            plots.plot_confusion_matrices(cms),
            plots.plot_roc_curves({n: (t_te, s) for n, s in cls_score.items()}),
            plots.plot_pred_vs_actual({n: (ylog_te, p) for n, p in reg_pred_log.items()}),
        ]
        for f in figs:
            print("  saved", f)

    # 7. error analysis (Problem 6) ----------------------------------------
    _header("7. ERROR ANALYSIS")
    for name, p_log in reg_pred_log.items():                 # price errors (+ tier acc)
        error_analysis.run(df_te, y_te, np.exp(p_log), t_te, cls_pred.get(name),
                           name=name, table_dir=TABLE_DIR)
    for name, pred in cls_pred.items():                      # tier errors for EVERY classifier
        table = _tier_error_table(y_te, t_te, pred)
        print(f"\n[{name}] tier error by price range")
        evaluate.compare(table)
        evaluate.save_table(table, TABLE_DIR / f"{_slug(name)}_tier_by_price_range",
                            title=f"{name}: tier errors by price range (test set)")
        plots.plot_error_bars(table, "error rate",
                              f"{name}: tier error rate by price range",
                              f"err_{_slug(name)}_tier_by_price_range.png")

    print(f"\nTables written to {TABLE_DIR}/  Figures written to {plots.FIG_DIR}/")


if __name__ == "__main__":
    main()
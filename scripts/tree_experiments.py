import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from src import data_prep, evaluate
from src.decision_tree import DecisionTree

def main():
    raw = data_prep.load_raw()
    df = data_prep.clean(raw, verbose=False)
    X, y_price, names = data_prep.make_features(df)
    
    X_tr, y_tr_p, X_val, y_val_p, X_te, y_te_p = data_prep.train_val_test_split(X, y_price)

    y_tr, thr = data_prep.make_tier_label(y_tr_p)
    y_val, _ = data_prep.make_tier_label(y_val_p, thr)

    depths = list(range(1, 16))
    results = {"gini": {"train": [], "val": []}, "entropy": {"train": [], "val": []}}

    for criterion in ["gini", "entropy"]:
        for depth in depths:
            tree = DecisionTree(task="classification", criterion=criterion, max_depth=depth)
            tree.fit(X_tr, y_tr)
            
            pred_tr = tree.predict(X_tr)
            pred_val = tree.predict(X_val)
            
            acc_tr = evaluate.accuracy(y_tr, pred_tr)
            acc_val = evaluate.accuracy(y_val, pred_val)
            
            results[criterion]["train"].append(acc_tr)
            results[criterion]["val"].append(acc_val)
            print(f"[{criterion.upper():7s}] Depth {depth:2d} | Train Acc: {acc_tr:.4f} | Val Acc: {acc_val:.4f}")

    #Bias-variance / Overfitting graph
    fig_dir = Path("report/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.plot(depths, results["gini"]["train"], "b--", label="Gini - Train")
    plt.plot(depths, results["gini"]["val"], "b-o", label="Gini - Validation")
    plt.plot(depths, results["entropy"]["train"], "r--", label="Entropy - Train")
    plt.plot(depths, results["entropy"]["val"], "r-s", label="Entropy - Validation")

    plt.xlabel("Max Depth")
    plt.ylabel("Accuracy")
    plt.title("Decision Tree: Overfitting Analysis (Gini vs Entropy)")
    plt.legend()
    plt.grid(True)
    
    out_path = fig_dir / "dt_overfitting.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved overfitting plot to: {out_path}")

    #choosing the best model on the validation set
    best_val_acc = -1.0
    best_config = (None, None)

    for criterion in ["gini", "entropy"]:
        val_accs = results[criterion]["val"]
        max_acc_idx = np.argmax(val_accs)
        if val_accs[max_acc_idx] > best_val_acc:
            best_val_acc = val_accs[max_acc_idx]
            best_config = (criterion, depths[max_acc_idx])

    print(f"\nBest Configuration: Criterion = {best_config[0]}, Max Depth = {best_config[1]} (Val Accuracy = {best_val_acc:.4f})")

    #top10 feature importances
    best_tree = DecisionTree(task="classification", criterion=best_config[0], max_depth=best_config[1])
    best_tree.fit(X_tr, y_tr)

    importances = best_tree.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\nTop 10 Feature Importances")
    for i in range(min(10, len(names))):
        idx = indices[i]
        print(f"{i+1:2d}. {names[idx]:<25s}: {importances[idx]:.4f}")

if __name__ == "__main__":
    main()
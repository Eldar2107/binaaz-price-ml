"""
decision_tree.py — a decision tree built FROM SCRATCH (NumPy only).

Must support BOTH tasks:
  - classification (criterion = "gini" or "entropy")  -> price tier
  - regression     (criterion = "mse")                -> price

scikit-learn is NOT allowed here. This is your own code.

Suggested design:
  - a small Node (feature index, threshold, left, right, value/leaf_value),
  - recursive _grow with a best-split search over features & thresholds,
  - impurity functions (gini / entropy / mse) and the impurity-decrease
    objective used to pick splits,
  - hyperparameters that control overfitting (below).
"""

from __future__ import annotations

import numpy as np


class Node:
    """One node of the tree. Fill in the fields you need."""
    def __init__(self):
        self.feature = None      # int: feature index to split on (internal)
        self.threshold = None    # float: split threshold (internal)
        self.left = None         # Node
        self.right = None        # Node
        self.value = None        # leaf prediction (class label/proba or mean)

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class DecisionTree:
    def __init__(
        self,
        task: str = "classification",   # "classification" | "regression"
        criterion: str = "gini",        # "gini" | "entropy" | "mse"
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        random_state: int = 42,
    ):
        self.task = task
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.root: Node | None = None

    # --- impurity -----------------------------------------------------------
    def _impurity(self, y) -> float:
        """Gini / entropy (classification) or MSE/variance (regression)."""
        if len(y) == 0:
            return 0.0

        if self.criterion == "gini":
            _, counts = np.unique(y, return_counts=True)
            probs = counts / len(y)
            return float(1.0 - np.sum(probs ** 2))

        elif self.criterion == "entropy":
            _, counts = np.unique(y, return_counts=True)
            probs = counts / len(y)
            return float(-np.sum(probs * np.log2(probs + 1e-12)))

        elif self.criterion == "mse":
            if len(y) <= 1:
                return 0.0
            return float(np.var(y))

        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")

    def _impurity_decrease(self, y, y_left, y_right) -> float:
        """Calculate weighted impurity reduction."""
        n = len(y)
        if n == 0 or len(y_left) == 0 or len(y_right) == 0:
            return 0.0
        
        parent_imp = self._impurity(y)
        left_imp = self._impurity(y_left)
        right_imp = self._impurity(y_right)
        
        weight_left = len(y_left) / n
        weight_right = len(y_right) / n
        
        gain = parent_imp - (weight_left * left_imp + weight_right * right_imp)
        return float(gain)

    def _best_split(self, X, y):
        """
        Search features & candidate thresholds; return the split that
        maximizes impurity decrease (best_feature, best_threshold, best_gain),
        or (None, None, 0.0) if no valid split is found.
        """
        n_samples, n_features = X.shape
        
        if n_samples < self.min_samples_split:
            return None, None, 0.0

        best_gain = -1.0
        best_feature = None
        best_threshold = None

        for feature_idx in range(n_features):
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)

            #if all values are equal, parsing is impossible
            if len(thresholds) <= 1:
                continue

            #average points between neighbour unique values
            midpoints = (thresholds[:-1] + thresholds[1:]) / 2.0

            for threshold in midpoints:
                left_mask = X_column <= threshold
                right_mask = ~left_mask

                y_left = y[left_mask]
                y_right = y[right_mask]

                #checking for the minimum elements in the leaf
                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue

                gain = self._impurity_decrease(y, y_left, y_right)

                if gain > best_gain and gain >= self.min_impurity_decrease:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        if best_feature is None or best_gain <= 0.0:
            return None, None, 0.0

        return best_feature, best_threshold, best_gain

    # --- grow / predict -----------------------------------------------------
    def _grow(self, X, y, depth: int) -> Node:
        """Recursively build the tree, respecting the stopping rules."""
        raise NotImplementedError("TODO: recursive grow")

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.root = self._grow(X, y, depth=0)
        return self

    def _predict_one(self, x, node: Node):
        raise NotImplementedError("TODO: traverse to a leaf")

    def predict(self, X):
        raise NotImplementedError("TODO: predict")

    def predict_proba(self, X):
        """Classification only: return class probabilities from leaf counts."""
        raise NotImplementedError("TODO: predict_proba (classification)")

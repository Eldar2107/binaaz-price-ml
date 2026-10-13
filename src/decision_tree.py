"""
decision_tree.py — a decision tree built FROM SCRATCH (NumPy only).

Must support BOTH tasks:
  - classification (criterion = "gini" or "entropy")  -> price tier
  - regression     (criterion = "mse")                -> price

scikit-learn is NOT allowed here. This is your own code.
"""

from __future__ import annotations

import numpy as np


class Node:
    """One node of the tree."""
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
        self.feature_importances_: np.ndarray | None = None
        self._feature_importances: np.ndarray | None = None
        self._n_total_samples: int = 0

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

    def _best_split(self, X, y):
        """
        Fast vectorized search using np.cumsum across features & candidate thresholds.
        Returns (best_feature, best_threshold, best_gain).
        """
        n_samples, n_features = X.shape

        if n_samples < self.min_samples_split:
            return None, None, 0.0

        parent_impurity = self._impurity(y)
        best_gain = -1.0
        best_feature = None
        best_threshold = None

        if self.task == "classification":
            #getting all unique classes
            classes = np.unique(y)
            n_classes = len(classes)
            if n_classes <= 1:
                return None, None, 0.0
            
            #transforming classes into indexes for OHE 
            class_to_idx = {c: i for i, c in enumerate(classes)}
            y_encoded = np.array([class_to_idx[val] for val in y], dtype=int)

            for feat_idx in range(n_features):
                X_col = X[:, feat_idx]
                sort_idx = np.argsort(X_col)
                X_sorted = X_col[sort_idx]
                y_sorted = y_encoded[sort_idx]

                split_mask = X_sorted[:-1] != X_sorted[1:]
                if not np.any(split_mask):
                    continue

                y_ohe = np.zeros((n_samples, n_classes), dtype=float)
                y_ohe[np.arange(n_samples), y_sorted] = 1.0

                cum_counts_left = np.cumsum(y_ohe, axis=0)[:-1]
                cum_counts_right = cum_counts_left[-1] - cum_counts_left

                n_left = np.arange(1, n_samples, dtype=float)
                n_right = n_samples - n_left

                valid = split_mask & (n_left >= self.min_samples_leaf) & (n_right >= self.min_samples_leaf)
                if not np.any(valid):
                    continue

                p_left = cum_counts_left / n_left[:, None]
                p_right = cum_counts_right / n_right[:, None]

                if self.criterion == "gini":
                    imp_left = 1.0 - np.sum(p_left ** 2, axis=1)
                    imp_right = 1.0 - np.sum(p_right ** 2, axis=1)
                elif self.criterion == "entropy":
                    p_left = np.clip(p_left, 1e-12, 1.0)
                    p_right = np.clip(p_right, 1e-12, 1.0)
                    imp_left = -np.sum(p_left * np.log2(p_left), axis=1)
                    imp_right = -np.sum(p_right * np.log2(p_right), axis=1)
                else:
                    raise ValueError(f"Unknown classification criterion: {self.criterion}")

                gain = (n_samples / self._n_total_samples) * (
                    parent_impurity - (n_left / n_samples) * imp_left - (n_right / n_samples) * imp_right
                )

                valid_gains = np.where(valid, gain, -1.0)
                max_i = np.argmax(valid_gains)
                max_g = valid_gains[max_i]

                if max_g > best_gain and max_g >= self.min_impurity_decrease:
                    best_gain = max_g
                    best_feature = feat_idx
                    best_threshold = float((X_sorted[max_i] + X_sorted[max_i + 1]) / 2.0)

        else:  #task regression
            if n_samples <= 1:
                return None, None, 0.0

            for feat_idx in range(n_features):
                X_col = X[:, feat_idx]
                sort_idx = np.argsort(X_col)
                X_sorted = X_col[sort_idx]
                y_sorted = y[sort_idx]

                split_mask = X_sorted[:-1] != X_sorted[1:]
                if not np.any(split_mask):
                    continue

                cum_sum = np.cumsum(y_sorted)[:-1]
                cum_sq = np.cumsum(y_sorted ** 2)[:-1]

                n_left = np.arange(1, n_samples, dtype=float)
                n_right = n_samples - n_left

                valid = split_mask & (n_left >= self.min_samples_leaf) & (n_right >= self.min_samples_leaf)
                if not np.any(valid):
                    continue

                sum_right = cum_sum[-1] - cum_sum
                sq_right = cum_sq[-1] - cum_sq

                var_left = np.maximum(0.0, (cum_sq / n_left) - (cum_sum / n_left) ** 2)
                var_right = np.maximum(0.0, (sq_right / n_right) - (sum_right / n_right) ** 2)

                gain = (n_samples / self._n_total_samples) * (
                    parent_impurity - (n_left / n_samples) * var_left - (n_right / n_samples) * var_right
                )

                valid_gains = np.where(valid, gain, -1.0)
                max_i = np.argmax(valid_gains)
                max_g = valid_gains[max_i]

                if max_g > best_gain and max_g >= self.min_impurity_decrease:
                    best_gain = max_g
                    best_feature = feat_idx
                    best_threshold = float((X_sorted[max_i] + X_sorted[max_i + 1]) / 2.0)

        if best_feature is None or best_gain <= 0.0:
            return None, None, 0.0

        return best_feature, best_threshold, float(best_gain)

    # --- grow / predict -----------------------------------------------------
    def _leaf_value(self, y):
        """Compute the leaf prediction value."""
        if len(y) == 0:
            return 0.0 if self.task == "regression" else 0
        if self.task == "regression":
            return float(np.mean(y))
        else:
            vals, counts = np.unique(y, return_counts=True)
            return vals[np.argmax(counts)]

    def _grow(self, X, y, depth: int) -> Node:
        """Recursively build the tree, respecting the stopping rules."""
        node = Node()
        n_samples = len(y)
        is_max_depth = self.max_depth is not None and depth >= self.max_depth
        
        if (
            is_max_depth
            or n_samples < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            node.value = self._leaf_value(y)
            return node

        best_feat, best_thresh, best_gain = self._best_split(X, y)

        if best_feat is None:
            node.value = self._leaf_value(y)
            return node

        if self._feature_importances is not None:
            self._feature_importances[best_feat] += best_gain

        node.feature = best_feat
        node.threshold = best_thresh

        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        node.left = self._grow(X[left_mask], y[left_mask], depth + 1)
        node.right = self._grow(X[right_mask], y[right_mask], depth + 1)

        return node

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        
        self._n_total_samples = len(y)
        n_features = X.shape[1]
        self._feature_importances = np.zeros(n_features, dtype=float)
        
        self.root = self._grow(X, y, depth=0)
        
        total = np.sum(self._feature_importances)
        if total > 0:
            self.feature_importances_ = self._feature_importances / total
        else:
            self.feature_importances_ = self._feature_importances
            
        return self

    def _predict_one(self, x, node: Node):
        """Recursively traverse down to a leaf for a single sample x."""
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if self.root is None:
            raise ValueError("The model is not fitted yet. Call 'fit' first.")
        return np.array([self._predict_one(x, self.root) for x in X])

    def _predict_proba_one(self, x, node: Node):
        """Traverse to a leaf and return label distribution/probabilities."""
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_proba_one(x, node.left)
        return self._predict_proba_one(x, node.right)

    def predict_proba(self, X):
        """Classification only: return class probabilities from leaf counts."""
        if self.task != "classification":
            raise ValueError("predict_proba is only supported for classification tasks.")
        
        preds = self.predict(X)
        probs = []
        for p in preds:
            probs.append([1.0 - float(p), float(p)])
        return np.array(probs)
"""
Unit tests for Gini, Entropy, and MSE impurity calculations in DecisionTree.

Run from the repo root:
    python -m tests.test_tree
    python -m pytest tests -q
"""

import numpy as np
from src.decision_tree import DecisionTree


def test_gini_impurity():
    # 1. Pure set (all same class) -> Gini should be 0.0
    tree_gini = DecisionTree(task="classification", criterion="gini")
    y_pure = np.array([1, 1, 1, 1])
    assert np.isclose(tree_gini._impurity(y_pure), 0.0)

    # 2. Perfect 50/50 binary split -> Gini = 1 - (0.5^2 + 0.5^2) = 0.5
    y_balanced = np.array([0, 0, 1, 1])
    assert np.isclose(tree_gini._impurity(y_balanced), 0.5)

    # 3. Imbalanced set (1/4 class 0, 3/4 class 1) -> Gini = 1 - (0.25^2 + 0.75^2) = 0.375
    y_imbalanced = np.array([0, 1, 1, 1])
    assert np.isclose(tree_gini._impurity(y_imbalanced), 0.375)

    # 4. Empty set -> 0.0
    assert tree_gini._impurity(np.array([])) == 0.0


def test_entropy_impurity():
    # 1. Pure set -> Entropy should be 0.0
    tree_entropy = DecisionTree(task="classification", criterion="entropy")
    y_pure = np.array([0, 0, 0])
    assert np.isclose(tree_entropy._impurity(y_pure), 0.0)

    # 2. Perfect 50/50 binary split -> Entropy = -2 * (0.5 * log2(0.5)) = 1.0
    y_balanced = np.array([0, 0, 1, 1])
    assert np.isclose(tree_entropy._impurity(y_balanced), 1.0)


def test_mse_impurity():
    # 1. Constant values -> Variance/MSE = 0.0
    tree_mse = DecisionTree(task="regression", criterion="mse")
    y_constant = np.array([5.0, 5.0, 5.0])
    assert np.isclose(tree_mse._impurity(y_constant), 0.0)

    # 2. Variance of [1, 2, 3] -> Mean = 2, Var = ((1-2)^2 + 0 + (3-2)^2)/3 = 2/3 ≈ 0.6667
    y_var = np.array([1.0, 2.0, 3.0])
    assert np.isclose(tree_mse._impurity(y_var), np.var(y_var))

def test_best_split():
    X = np.array([[0.0], [0.2], [0.8], [1.0]])
    y = np.array([0, 0, 1, 1])

    tree = DecisionTree(task="classification", criterion="gini", min_samples_split=2)
    feature, threshold, gain = tree._best_split(X, y)

    assert feature == 0
    assert 0.2 < threshold < 0.8  #threshold strictly between 0.2 and 0.8
    assert gain > 0.0

if __name__ == "__main__":
    test_gini_impurity()
    print("OK test_gini_impurity")
    test_entropy_entropy = test_entropy_impurity()
    print("OK test_entropy_impurity")
    test_mse_impurity()
    print("OK test_mse_impurity")
    test_best_split()
    print("OK test_best_split")
    print("\nAll impurity unit tests passed successfully")
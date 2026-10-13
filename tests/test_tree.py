"""
Unit tests for DecisionTree (Vectorized _best_split, Impurities, Hyperparameters, and Sklearn match).
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from src.decision_tree import DecisionTree


def test_gini_impurity():
    tree_gini = DecisionTree(task="classification", criterion="gini")
    assert np.isclose(tree_gini._impurity(np.array([1, 1, 1, 1])), 0.0)
    assert np.isclose(tree_gini._impurity(np.array([0, 0, 1, 1])), 0.5)
    assert np.isclose(tree_gini._impurity(np.array([0, 1, 1, 1])), 0.375)
    assert tree_gini._impurity(np.array([])) == 0.0


def test_entropy_impurity():
    tree_entropy = DecisionTree(task="classification", criterion="entropy")
    assert np.isclose(tree_entropy._impurity(np.array([0, 0, 0])), 0.0)
    assert np.isclose(tree_entropy._impurity(np.array([0, 0, 1, 1])), 1.0)


def test_mse_impurity():
    tree_mse = DecisionTree(task="regression", criterion="mse")
    assert np.isclose(tree_mse._impurity(np.array([5.0, 5.0, 5.0])), 0.0)
    y_var = np.array([1.0, 2.0, 3.0])
    assert np.isclose(tree_mse._impurity(y_var), np.var(y_var))


def test_best_split_vectorized():
    X = np.array([[0.0], [0.2], [0.8], [1.0]])
    y = np.array([0, 0, 1, 1])

    tree = DecisionTree(task="classification", criterion="gini", min_samples_split=2)
    tree._n_total_samples = len(y)
    feature, threshold, gain = tree._best_split(X, y)

    assert feature == 0
    assert 0.2 < threshold < 0.8
    assert gain > 0.0


def test_full_tree_fit_predict():
    X = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [6.0, 9.0]])
    y = np.array([0, 0, 1, 1])

    tree = DecisionTree(task="classification", criterion="gini", max_depth=3)
    tree.fit(X, y)

    predictions = tree.predict(X)
    assert np.array_equal(predictions, y)


def test_regression_tree():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([10.0, 20.0, 30.0, 40.0])

    tree = DecisionTree(task="regression", criterion="mse", max_depth=2)
    tree.fit(X, y)

    preds = tree.predict(X)
    assert len(preds) == 4
    assert np.allclose(preds, y, atol=5.0)


def test_hyperparameters_affect_tree():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    #max_depth
    t_d1 = DecisionTree(max_depth=1).fit(X, y)
    t_d5 = DecisionTree(max_depth=5).fit(X, y)
    assert t_d1.predict(X).tolist() != t_d5.predict(X).tolist()

    #min_samples_split
    t_sp2 = DecisionTree(min_samples_split=2).fit(X, y)
    t_sp50 = DecisionTree(min_samples_split=50).fit(X, y)
    assert t_sp2.predict(X).tolist() != t_sp50.predict(X).tolist()

    #min_samples_leaf
    t_lf1 = DecisionTree(min_samples_leaf=1).fit(X, y)
    t_lf30 = DecisionTree(min_samples_leaf=30).fit(X, y)
    assert t_lf1.predict(X).tolist() != t_lf30.predict(X).tolist()

    #min_impurity_decrease
    t_inc0 = DecisionTree(min_impurity_decrease=0.0).fit(X, y)
    t_inc5 = DecisionTree(min_impurity_decrease=0.5).fit(X, y)
    assert t_inc0.predict(X).tolist() != t_inc5.predict(X).tolist()


def test_sklearn_split_match():
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    y = np.array([0, 0, 0, 1, 1, 1])

    custom_tree = DecisionTree(task="classification", criterion="gini", max_depth=1)
    custom_tree.fit(X, y)

    sk_tree = DecisionTreeClassifier(criterion="gini", max_depth=1, random_state=42)
    sk_tree.fit(X, y)

    assert custom_tree.root.feature == sk_tree.tree_.feature[0]
    assert np.isclose(custom_tree.root.threshold, sk_tree.tree_.threshold[0])
    assert (custom_tree.predict(X) == sk_tree.predict(X)).all()


if __name__ == "__main__":
    test_gini_impurity()
    test_entropy_impurity()
    test_mse_impurity()
    test_best_split_vectorized()
    test_full_tree_fit_predict()
    test_regression_tree()
    test_hyperparameters_affect_tree()
    test_sklearn_split_match()
    print("All tests passed successfully!")
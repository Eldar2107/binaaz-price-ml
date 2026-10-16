"""
svm.py — a soft-margin SVM built FROM SCRATCH via PEGASOS (NumPy only).
"""

from __future__ import annotations

import numpy as np


class RFFMapper:
    def __init__(self, n_components: int = 100, gamma: float = 1.0, random_state: int | None = None):
        self.n_components = n_components
        self.gamma = gamma
        self.random_state = random_state
        self.W = None
        self.b = None
        self.fitted_ = False

    def fit(self, X):
        X = np.array(X, dtype=float)
        _, n_features = X.shape
        rng = np.random.default_rng(self.random_state)
        self.W = rng.normal(loc=0.0, scale=np.sqrt(2 * self.gamma), size=(n_features, self.n_components))
        self.b = rng.uniform(0, 2 * np.pi, size=(1, self.n_components))
        self.fitted_ = True
        return self

    def transform(self, X):
        if not self.fitted_:
            raise RuntimeError("RFFMapper is not fitted yet.")
        X = np.array(X, dtype=float)
        return np.sqrt(2.0 / self.n_components) * np.cos(X @ self.W + self.b)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class PegasosSVM:
    def __init__(
        self,
        lambda_: float = 1e-4,
        n_epochs: int = 20,
        batch_size: int = 1,
        random_state: int | None = None,
        feature_map = None,
        track: str = "epoch",
    ):
        if lambda_ <= 0:
            raise ValueError("lambda_ must be strictly positive.")
        if track not in ("epoch", "step"):
            raise ValueError("track must be 'epoch' or 'step'.")

        self.lambda_ = lambda_
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.feature_map = feature_map
        self.track = track

        self.coef_ = None
        self.intercept_ = 0.0
        self.n_iter_ = 0
        self._objective_history = []

    @staticmethod
    def _to_pm1(y):
        y_arr = np.array(y)
        unique_vals = np.unique(y_arr)
        if len(unique_vals) != 2 or not set(unique_vals).issubset({0, 1}):
            raise ValueError("Labels must be binary {0, 1}.")
        y_pm = np.copy(y_arr).astype(float)
        y_pm[y_pm == 0] = -1.0
        return y_pm

    @staticmethod
    def _from_pm1(y_pm):
        y = np.copy(y_pm)
        y[y == -1.0] = 0.0
        return y.astype(int)

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        if self.feature_map is not None:
            if not hasattr(self.feature_map, "fitted_") or not self.feature_map.fitted_:
                X = self.feature_map.fit_transform(X)
            else:
                X = self.feature_map.transform(X)

        y_pm = self._to_pm1(y)
        m, n = X.shape

        rng = np.random.default_rng(self.random_state)
        self.coef_ = np.zeros(n)
        self.intercept_ = 0.0
        self._objective_history = []

        self._objective_history.append(objective_value(self.coef_, self.intercept_, X, y_pm, self.lambda_))

        t = 0
        for epoch in range(self.n_epochs):
            indices = rng.permutation(m)
            for start_idx in range(0, m, self.batch_size):
                t += 1
                batch_indices = indices[start_idx : start_idx + self.batch_size]
                X_batch = X[batch_indices]
                y_batch = y_pm[batch_indices]

                eta = 1.0 / (self.lambda_ * t)

                margins = y_batch * (X_batch @ self.coef_ + self.intercept_)
                violations = margins < 1.0

                reg_term = (1.0 - eta * self.lambda_) * self.coef_
                if np.any(violations):
                    loss_grad_w = np.sum(y_batch[violations, None] * X_batch[violations], axis=0) / len(batch_indices)
                    loss_grad_b = np.sum(y_batch[violations]) / len(batch_indices)
                    self.coef_ = reg_term + eta * loss_grad_w
                    self.intercept_ += eta * loss_grad_b
                else:
                    self.coef_ = reg_term

                w_norm = np.linalg.norm(self.coef_)
                max_norm = 1.0 / np.sqrt(self.lambda_)
                if w_norm > max_norm:
                    self.coef_ = self.coef_ * (max_norm / w_norm)

                if self.track == "step":
                    self._objective_history.append(objective_value(self.coef_, self.intercept_, X, y_pm, self.lambda_))

            if self.track == "epoch":
                self._objective_history.append(objective_value(self.coef_, self.intercept_, X, y_pm, self.lambda_))

        self.n_iter_ = t
        return self

    def decision_function(self, X):
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted yet.")
        X = np.array(X, dtype=float)
        if self.feature_map is not None:
            X = self.feature_map.transform(X)
        return X @ self.coef_ + self.intercept_

    def predict(self, X):
        scores = self.decision_function(X)
        preds = (scores > 0).astype(int)
        return preds

    def get_objective_history(self):
        return self._objective_history

    def get_margin(self):
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted yet.")
        norm_w = np.linalg.norm(self.coef_)
        if norm_w == 0:
            return float('inf')
        return 2.0 / norm_w

    def get_support_vectors(self, X, y, tol=1e-5):
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted yet.")
        y_pm = self._to_pm1(y)
        margins = y_pm * self.decision_function(X)
        sv_indices = np.flatnonzero(margins <= 1.0 + tol)
        return sv_indices


def accuracy_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(y_true == y_pred)


def f1_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def objective_value(w, b, X, y_pm1, lambda_):
    reg = 0.5 * lambda_ * np.sum(w ** 2)
    margins = y_pm1 * (X @ w + b)
    hinge = np.maximum(0.0, 1.0 - margins)
    return reg + np.mean(hinge)


def lambda_sweep(X_train, y_train, X_val, y_val, lambdas, n_epochs=20):
    results = []
    for lam in lambdas:
        model = PegasosSVM(lambda_=lam, n_epochs=n_epochs, random_state=42)
        model.fit(X_train, y_train)
        val_preds = model.predict(X_val)
        results.append({
            "lambda": lam,
            "margin": model.get_margin(),
            "val_accuracy": accuracy_score(y_val, val_preds),
            "val_f1": f1_score(y_val, val_preds),
            "final_objective": model.get_objective_history()[-1],
            "train_accuracy": accuracy_score(y_train, model.predict(X_train)),
            "support_vector_count": len(model.get_support_vectors(X_train, y_train))
        })
    return results


def compare_models(X_train, y_train, X_val, y_val, n_components=100, n_epochs=20):
    from sklearn.svm import LinearSVC, SVC
    from sklearn.linear_model import SGDClassifier
    import time

    report = {}

    start = time.time()
    m_lin = PegasosSVM(lambda_=1e-3, n_epochs=n_epochs, random_state=42)
    m_lin.fit(X_train, y_train)
    t_lin = time.time() - start
    preds_lin = m_lin.predict(X_val)
    report["Pegasos(linear)"] = {
        "accuracy": accuracy_score(y_val, preds_lin),
        "f1": f1_score(y_val, preds_lin),
        "fit_time": t_lin
    }

    start = time.time()
    m_rff = PegasosSVM(
        lambda_=1e-3,
        n_epochs=n_epochs,
        feature_map=RFFMapper(n_components=n_components, gamma=0.5, random_state=42),
        random_state=42
    )
    m_rff.fit(X_train, y_train)
    t_rff = time.time() - start
    preds_rff = m_rff.predict(X_val)
    report["Pegasos(RFF)"] = {
        "accuracy": accuracy_score(y_val, preds_rff),
        "f1": f1_score(y_val, preds_rff),
        "fit_time": t_rff
    }

    n_train = len(X_train)
    lam = 1e-3
    C_val = 1.0 / (lam * n_train)

    start = time.time()
    lsvc = LinearSVC(C=C_val, random_state=42, max_iter=10000, dual=False)
    lsvc.fit(X_train, y_train)
    t_lsvc = time.time() - start
    preds_lsvc = lsvc.predict(X_val)
    report["LinearSVC"] = {
        "accuracy": accuracy_score(y_val, preds_lsvc),
        "f1": f1_score(y_val, preds_lsvc),
        "fit_time": t_lsvc
    }

    start = time.time()
    sgd = SGDClassifier(loss='hinge', alpha=lam, random_state=42, max_iter=1000)
    sgd.fit(X_train, y_train)
    t_sgd = time.time() - start
    preds_sgd = sgd.predict(X_val)
    report["SGDClassifier(hinge)"] = {
        "accuracy": accuracy_score(y_val, preds_sgd),
        "f1": f1_score(y_val, preds_sgd),
        "fit_time": t_sgd
    }

    start = time.time()
    svc_rbf = SVC(C=C_val, gamma=0.5, random_state=42)
    svc_rbf.fit(X_train, y_train)
    t_svc = time.time() - start
    preds_svc = svc_rbf.predict(X_val)
    report["SVC(rbf)"] = {
        "accuracy": accuracy_score(y_val, preds_svc),
        "f1": f1_score(y_val, preds_svc),
        "fit_time": t_svc
    }

    return report

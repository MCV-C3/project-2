from __future__ import annotations
from typing import Any, Dict
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

class HistIntersectionSVM(BaseEstimator, ClassifierMixin):
    """
    SVM with Histogram Intersection kernel.
    """
    def __init__(self, C: float = 1.0):
        self.C = C
        self._svc = SVC(kernel="precomputed", C=C)
        self._X_train: np.ndarray | None = None

    @staticmethod
    def hist_intersection_kernel(X, Y, block_size=500):
        """ Done by chunking so it does not explodes RAM."""
        N = X.shape[0]
        M = Y.shape[0]
        K = np.zeros((N, M), dtype=np.float32)

        for i in range(0, N, block_size):
            i_end = min(i + block_size, N)
            X_block = X[i:i_end]

            # Compute chunk: (block, M, D)
            min_vals = np.minimum(X_block[:, None, :], Y[None, :, :])
            K[i:i_end] = np.sum(min_vals, axis=2)
        return K

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._X_train = X.copy()
        K = self.hist_intersection_kernel(X, X)
        self._svc.fit(K, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._X_train is None:
            raise RuntimeError("Model must be fitted before calling predict().")
        K = self.hist_intersection_kernel(X, self._X_train)
        return self._svc.predict(K)


def create_classifier(name: str, **kwargs: Dict[str, Any]):
    """
    Get a classifier by name.
    """

    if name == "LogisticRegression":
        return LogisticRegression(**kwargs)

    if name == "SVM":
        return SVC(**kwargs)

    if name == "HistIntersectionSVM":
        return HistIntersectionSVM(**kwargs)

    if name == "AdaBoost":
        # Extract base estimator parameters if provided
        base_estimator_name = kwargs.pop("base_estimator", "DecisionTree")
        base_estimator_params = kwargs.pop("base_estimator_params", {})

        # Create base estimator
        if base_estimator_name == "DecisionTree":
            base_estimator = DecisionTreeClassifier(**base_estimator_params)
        elif base_estimator_name == "SVM":
            base_estimator = SVC(**base_estimator_params)
        elif base_estimator_name == "LogisticRegression":
            base_estimator = LogisticRegression(**base_estimator_params)
        else:
            raise ValueError(f"Unknown base estimator: {base_estimator_name}")

        # Create AdaBoost with the base estimator
        return AdaBoostClassifier(estimator=base_estimator, **kwargs)

    raise ValueError("Unknown classifier name: "
                     f"{name}. Should be 'LogisticRegression', 'SVM', 'HistIntersectionSVM', or 'AdaBoost'.")
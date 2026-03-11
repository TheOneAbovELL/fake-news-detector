"""
models.py
---------
Model definitions, training, and serialization.
Implements Logistic Regression and SVM classifiers.
"""

import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score
from typing import Dict, Any


class FakeNewsClassifier:
    """
    Wrapper around sklearn classifiers for fake news detection.

    Supports:
    - Logistic Regression (with probability outputs)
    - Linear SVM (with Platt scaling for probabilities)
    """

    SUPPORTED_MODELS = ["logistic_regression", "svm"]

    def __init__(self, model_type: str = "logistic_regression", **kwargs):
        """
        Parameters
        ----------
        model_type : str
            One of 'logistic_regression' or 'svm'.
        **kwargs
            Passed to the underlying sklearn estimator.
        """
        assert model_type in self.SUPPORTED_MODELS, \
            f"model_type must be one of {self.SUPPORTED_MODELS}"

        self.model_type = model_type
        self.model = self._build(model_type, **kwargs)
        self._trained = False

    def _build(self, model_type: str, **kwargs):
        if model_type == "logistic_regression":
            return LogisticRegression(
                C=kwargs.get("C", 1.0),
                max_iter=kwargs.get("max_iter", 1000),
                solver="lbfgs",
                n_jobs=-1,
                random_state=42,
            )
        elif model_type == "svm":
            base_svm = LinearSVC(
                C=kwargs.get("C", 1.0),
                max_iter=kwargs.get("max_iter", 2000),
                random_state=42,
            )
            # Wrap in CalibratedClassifierCV to get predict_proba
            return CalibratedClassifierCV(base_svm, cv=3)

    def train(self, X_train, y_train, cv: bool = True) -> Dict[str, Any]:
        """
        Fit the model. Optionally run 5-fold cross-validation.

        Returns dict with training metrics.
        """
        print(f"[Model] Training {self.model_type}...")
        self.model.fit(X_train, y_train)
        self._trained = True

        results = {}
        if cv:
            print(f"[Model] Running 5-fold cross-validation...")
            cv_scores = cross_val_score(
                self.model, X_train, y_train,
                cv=5, scoring="accuracy", n_jobs=-1
            )
            results["cv_mean"] = cv_scores.mean()
            results["cv_std"] = cv_scores.std()
            print(f"[Model] CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        return results

    def predict(self, X) -> np.ndarray:
        assert self._trained, "Model not trained yet."
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        assert self._trained, "Model not trained yet."
        return self.model.predict_proba(X)

    def predict_risk_score(self, X) -> np.ndarray:
        """
        Return fake-news probability (risk score) for each sample.
        Higher = higher risk of being fake news.
        """
        proba = self.predict_proba(X)
        return proba[:, 1]  # column 1 = P(fake)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"[Model] Saved → {path}")

    @classmethod
    def load(cls, path: str, model_type: str) -> "FakeNewsClassifier":
        clf = cls.__new__(cls)
        clf.model_type = model_type
        clf.model = joblib.load(path)
        clf._trained = True
        return clf

    def __repr__(self):
        return f"FakeNewsClassifier(type={self.model_type}, trained={self._trained})"

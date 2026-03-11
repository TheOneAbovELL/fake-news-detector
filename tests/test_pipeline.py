"""
test_pipeline.py
----------------
Unit tests for the Fake News Detection pipeline.
Run with: pytest tests/ -v
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader   import generate_dataset
from src.preprocessor  import clean_text, preprocess_dataframe
from src.features      import TFIDFExtractor
from src.models        import FakeNewsClassifier
from src.evaluate      import compute_metrics


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_dataset():
    return generate_dataset(n_samples=200)


@pytest.fixture(scope="module")
def preprocessed(small_dataset):
    return preprocess_dataframe(small_dataset)


@pytest.fixture(scope="module")
def features(preprocessed):
    texts = preprocessed["cleaned_text"].tolist()
    ext = TFIDFExtractor(max_features=500, ngram_range=(1, 1))
    X = ext.fit_transform(texts)
    return ext, X, preprocessed["label"].values


# ── Data Loader Tests ─────────────────────────────────────────────────────────

class TestDataLoader:
    def test_returns_dataframe(self, small_dataset):
        assert isinstance(small_dataset, pd.DataFrame)

    def test_correct_columns(self, small_dataset):
        assert "text" in small_dataset.columns
        assert "label" in small_dataset.columns

    def test_correct_size(self, small_dataset):
        assert len(small_dataset) == 200

    def test_binary_labels(self, small_dataset):
        assert set(small_dataset["label"].unique()).issubset({0, 1})

    def test_roughly_balanced(self, small_dataset):
        counts = small_dataset["label"].value_counts()
        ratio = counts.min() / counts.max()
        assert ratio > 0.4, "Dataset is too imbalanced"

    def test_no_null_text(self, small_dataset):
        assert small_dataset["text"].isna().sum() == 0


# ── Preprocessor Tests ────────────────────────────────────────────────────────

class TestPreprocessor:
    def test_clean_text_lowercase(self):
        assert clean_text("BREAKING NEWS") == clean_text("breaking news")

    def test_clean_text_removes_urls(self):
        result = clean_text("Check http://example.com now")
        assert "http" not in result
        assert "example" not in result

    def test_clean_text_removes_digits(self):
        result = clean_text("Report from 2024 shows 50% increase")
        assert not any(c.isdigit() for c in result)

    def test_clean_text_empty_input(self):
        assert clean_text("") == ""
        assert clean_text(None) == ""

    def test_preprocess_adds_cleaned_col(self, preprocessed):
        assert "cleaned_text" in preprocessed.columns

    def test_preprocess_adds_features(self, preprocessed):
        assert "has_all_caps_word" in preprocessed.columns
        assert "exclamation_count" in preprocessed.columns

    def test_no_null_after_preprocess(self, preprocessed):
        assert preprocessed["cleaned_text"].isna().sum() == 0


# ── Feature Extraction Tests ──────────────────────────────────────────────────

class TestTFIDFExtractor:
    def test_returns_sparse_matrix(self, features):
        _, X, _ = features
        import scipy.sparse as sp
        assert sp.issparse(X)

    def test_shape_matches_samples(self, features, preprocessed):
        _, X, _ = features
        assert X.shape[0] == len(preprocessed)

    def test_vocabulary_populated(self, features):
        ext, _, _ = features
        assert len(ext.get_feature_names()) > 0

    def test_transform_unseen(self, features):
        ext, _, _ = features
        X_new = ext.transform(["fake news headline", "real news report"])
        assert X_new.shape[0] == 2

    def test_top_features_returned(self, features):
        ext, X, y = features
        top_real, top_fake = ext.get_top_features_per_class(X, y, n_top=5)
        assert len(top_real) == 5
        assert len(top_fake) == 5


# ── Model Tests ───────────────────────────────────────────────────────────────

class TestFakeNewsClassifier:
    @pytest.mark.parametrize("model_type", ["logistic_regression", "svm"])
    def test_train_and_predict(self, features, model_type):
        _, X, y = features
        clf = FakeNewsClassifier(model_type)
        clf.train(X, y, cv=False)
        preds = clf.predict(X)
        assert len(preds) == len(y)
        assert set(preds).issubset({0, 1})

    @pytest.mark.parametrize("model_type", ["logistic_regression", "svm"])
    def test_predict_proba_shape(self, features, model_type):
        _, X, y = features
        clf = FakeNewsClassifier(model_type)
        clf.train(X, y, cv=False)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(y), 2)

    @pytest.mark.parametrize("model_type", ["logistic_regression", "svm"])
    def test_risk_score_range(self, features, model_type):
        _, X, y = features
        clf = FakeNewsClassifier(model_type)
        clf.train(X, y, cv=False)
        scores = clf.predict_risk_score(X)
        assert np.all(scores >= 0) and np.all(scores <= 1)

    def test_save_and_load(self, features, tmp_path):
        _, X, y = features
        clf = FakeNewsClassifier("logistic_regression")
        clf.train(X, y, cv=False)
        path = str(tmp_path / "model.pkl")
        clf.save(path)
        loaded = FakeNewsClassifier.load(path, "logistic_regression")
        np.testing.assert_array_equal(clf.predict(X), loaded.predict(X))

    def test_invalid_model_type(self):
        with pytest.raises(AssertionError):
            FakeNewsClassifier("random_forest")


# ── Evaluation Tests ──────────────────────────────────────────────────────────

class TestEvaluation:
    def test_metrics_keys(self, features):
        _, X, y = features
        clf = FakeNewsClassifier("logistic_regression")
        clf.train(X, y, cv=False)
        y_pred = clf.predict(X)
        y_prob = clf.predict_risk_score(X)
        metrics = compute_metrics(y, y_pred, y_prob)

        for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            assert key in metrics

    def test_metrics_range(self, features):
        _, X, y = features
        clf = FakeNewsClassifier("logistic_regression")
        clf.train(X, y, cv=False)
        y_pred = clf.predict(X)
        metrics = compute_metrics(y, y_pred)

        for key in ["accuracy", "precision", "recall", "f1"]:
            assert 0.0 <= metrics[key] <= 1.0

    def test_risk_scores_sum(self, features):
        _, X, y = features
        clf = FakeNewsClassifier("logistic_regression")
        clf.train(X, y, cv=False)
        y_pred = clf.predict(X)
        metrics = compute_metrics(y, y_pred)

        total = (metrics["true_positives"] + metrics["false_positives"] +
                 metrics["true_negatives"] + metrics["false_negatives"])
        assert total == len(y)

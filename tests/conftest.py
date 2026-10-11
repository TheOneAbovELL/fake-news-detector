"""Shared fixtures: a small, fast model trained on synthetic data (never the served model)."""

import json
import warnings

import pytest
import sklearn

from src.data_loader import generate_dataset
from src.features import TFIDFExtractor
from src.models import FakeNewsClassifier
from src.predict import MODEL_CARD_NAME, Predictor
from src.preprocessor import clean_text


@pytest.fixture(scope="session")
def models_dir(tmp_path_factory):
    """Directory with a tiny trained LR model + vectorizer + model card."""
    warnings.filterwarnings("ignore")
    out = tmp_path_factory.mktemp("models")
    df = generate_dataset(n_samples=400, seed=7)
    texts = [clean_text(t) for t in df["text"]]
    vec = TFIDFExtractor(max_features=2000, ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform(texts)
    clf = FakeNewsClassifier("logistic_regression", C=1.0)
    clf.train(X, df["label"].values, cv=False)
    vec.save(str(out / "tfidf_vectorizer.pkl"))
    clf.save(str(out / "logistic_regression.pkl"))
    card = {
        "model_name": "Logistic Regression",
        "model_type": "logistic_regression",
        "model_file": "logistic_regression.pkl",
        "vectorizer_file": "tfidf_vectorizer.pkl",
        "is_demo": False,
        "sklearn_version": sklearn.__version__,
        "thresholds": {"low": 0.35, "high": 0.65},
        "dataset": {"name": "test-fixture", "rows": 400},
        "metrics": {"test": {"roc_auc": 0.99, "accuracy": 0.98},
                    "bands": {"likely_fake": {"actually_fake": 0.9},
                              "likely_real": {"actually_fake": 0.1}}},
    }
    (out / MODEL_CARD_NAME).write_text(json.dumps(card), encoding="utf-8")
    return out


@pytest.fixture(scope="session")
def predictor(models_dir):
    return Predictor.load(models_dir)


class FakeResponse:
    """Minimal stand-in for requests.Response (supports streaming + context manager)."""

    def __init__(self, status_code=200, body=b"", headers=None, json_data=None):
        self.status_code = status_code
        self._body = body if isinstance(body, bytes) else body.encode("utf-8")
        self.headers = headers or {}
        self._json = json_data

    @property
    def is_redirect(self):
        return self.status_code in (301, 302, 303, 307, 308) and "Location" in self.headers

    def iter_content(self, chunk_size=65536):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]

    def raise_for_status(self):
        import requests
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}", response=self)

    def json(self):
        if self._json is None:
            raise ValueError("no JSON")
        return self._json

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeSession:
    """Returns queued responses (or raises queued exceptions) and records calls."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def _next(self, method, url, kwargs):
        self.calls.append((method, url, kwargs))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def get(self, url, **kwargs):
        return self._next("GET", url, kwargs)

    def post(self, url, **kwargs):
        return self._next("POST", url, kwargs)

    def close(self):
        pass


@pytest.fixture
def fake_response():
    return FakeResponse


@pytest.fixture
def fake_session():
    return FakeSession

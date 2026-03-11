"""
features.py
-----------
TF-IDF feature extraction pipeline.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import joblib
import os
from typing import Tuple, List


class TFIDFExtractor:
    """
    TF-IDF feature extractor with configurable parameters.

    Wraps sklearn's TfidfVectorizer with sensible defaults tuned
    for short text (news headlines).
    """

    def __init__(
        self,
        max_features: int = 50000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.90,
        sublinear_tf: bool = True,
    ):
        """
        Parameters
        ----------
        max_features : int
            Max vocabulary size.
        ngram_range : tuple
            (1,2) includes unigrams and bigrams.
        min_df : int
            Ignore terms appearing in fewer than this many documents.
        max_df : float
            Ignore terms appearing in more than this fraction of docs.
        sublinear_tf : bool
            Apply log normalization to TF (helps with short texts).
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\b[a-zA-Z][a-zA-Z]+\b",  # alpha tokens only
        )
        self._fitted = False

    def fit(self, texts: List[str]) -> "TFIDFExtractor":
        """Fit the vectorizer on training texts."""
        print(f"[Features] Fitting TF-IDF on {len(texts)} documents...")
        self.vectorizer.fit(texts)
        self._fitted = True
        vocab_size = len(self.vectorizer.vocabulary_)
        print(f"[Features] Vocabulary size: {vocab_size:,} terms")
        return self

    def transform(self, texts: List[str]):
        """Transform texts to TF-IDF matrix."""
        assert self._fitted, "Call fit() before transform()"
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: List[str]):
        """Fit and transform in one step."""
        self.fit(texts)
        return self.transform(texts)

    def get_feature_names(self) -> List[str]:
        return self.vectorizer.get_feature_names_out().tolist()

    def get_top_features_per_class(
        self, X, y, n_top: int = 20
    ) -> Tuple[List[str], List[str]]:
        """
        Get top discriminating TF-IDF features per class.

        Returns (top_real_features, top_fake_features)
        """
        feature_names = np.array(self.get_feature_names())

        # Mean TF-IDF per class
        real_mask = np.array(y) == 0
        fake_mask = np.array(y) == 1

        real_mean = np.asarray(X[real_mask].mean(axis=0)).flatten()
        fake_mean = np.asarray(X[fake_mask].mean(axis=0)).flatten()

        top_real = feature_names[np.argsort(real_mean)[-n_top:][::-1]].tolist()
        top_fake = feature_names[np.argsort(fake_mean)[-n_top:][::-1]].tolist()

        return top_real, top_fake

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.vectorizer, path)
        print(f"[Features] Vectorizer saved → {path}")

    @classmethod
    def load(cls, path: str) -> "TFIDFExtractor":
        extractor = cls.__new__(cls)
        extractor.vectorizer = joblib.load(path)
        extractor._fitted = True
        return extractor

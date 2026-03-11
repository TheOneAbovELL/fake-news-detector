"""
preprocessor.py
---------------
Text cleaning and NLP preprocessing for news headlines.
"""

import re
import string
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from typing import List


def download_nltk_resources():
    """Download required NLTK data silently."""
    for resource in ["stopwords", "punkt", "punkt_tab"]:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass


download_nltk_resources()

STOP_WORDS = set(stopwords.words("english"))
# Keep negation words — important for fake news detection
NEGATION_WORDS = {"no", "not", "never", "none", "nothing", "nor", "neither"}
STOP_WORDS -= NEGATION_WORDS

stemmer = PorterStemmer()

# Patterns that are characteristic of fake news sensationalism
SENSATIONAL_MARKERS = re.compile(
    r'\b(breaking|urgent|shocking|exposed|leaked|bombshell|baffled|'
    r'mainstream|silenced|censored|banned|hidden|secret|proof|'
    r'must.?watch|must.?read|share.?before)\b',
    re.IGNORECASE
)


def clean_text(text: str, use_stemming: bool = False) -> str:
    """
    Clean and normalize a single news headline.

    Steps:
    1. Lowercase
    2. Remove URLs, special chars, digits
    3. Remove punctuation
    4. Tokenize
    5. Remove stopwords (preserving negations)
    6. Optional: stem tokens

    Parameters
    ----------
    text : str
    use_stemming : bool

    Returns
    -------
    str — cleaned text
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove digits
    text = re.sub(r"\d+", " ", text)

    # Remove punctuation (but keep spaces)
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize
    tokens = text.split()

    # Remove stopwords
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]

    # Stemming (optional — off by default to preserve interpretability)
    if use_stemming:
        tokens = [stemmer.stem(t) for t in tokens]

    return " ".join(tokens)


def extract_features_manual(text: str) -> dict:
    """
    Extract hand-crafted features that signal fake news patterns.
    Used for feature analysis (not fed directly to TF-IDF models).
    """
    original = text if isinstance(text, str) else ""
    return {
        "has_all_caps_word": int(bool(re.search(r'\b[A-Z]{3,}\b', original))),
        "exclamation_count": original.count("!"),
        "question_count": original.count("?"),
        "has_sensational_marker": int(bool(SENSATIONAL_MARKERS.search(original))),
        "word_count": len(original.split()),
        "char_count": len(original),
    }


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "text",
                          use_stemming: bool = False) -> pd.DataFrame:
    """
    Apply full preprocessing to a DataFrame.

    Returns a new DataFrame with added 'cleaned_text' column and
    hand-crafted feature columns.
    """
    print("[Preprocessor] Cleaning text...")
    df = df.copy()
    df["cleaned_text"] = df[text_col].apply(lambda x: clean_text(x, use_stemming))

    # Extract manual features
    print("[Preprocessor] Extracting hand-crafted features...")
    manual_features = df[text_col].apply(extract_features_manual).apply(pd.Series)
    df = pd.concat([df, manual_features], axis=1)

    # Drop empty cleaned texts
    empty_mask = df["cleaned_text"].str.strip() == ""
    if empty_mask.sum() > 0:
        print(f"[Preprocessor] Dropping {empty_mask.sum()} empty rows after cleaning.")
        df = df[~empty_mask].reset_index(drop=True)

    print(f"[Preprocessor] Done. {len(df)} samples ready.")
    return df

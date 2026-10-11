"""
preprocessor.py
---------------
Text cleaning and NLP preprocessing for news headlines.

Cleaning must be byte-for-byte identical at training and inference time, so the
stopword list is frozen here (NLTK's English list, v3.9) instead of being
downloaded at import time — no network access is needed to clean text.
"""

import re
import unicodedata

import pandas as pd

# NLTK English stopwords without apostrophe forms (punctuation is replaced by
# spaces before filtering, so "don't" never survives as a token anyway).
_NLTK_ENGLISH_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "ain", "all", "am", "an",
    "and", "any", "are", "aren", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "couldn", "d", "did",
    "didn", "do", "does", "doesn", "doing", "don", "down", "during", "each", "few",
    "for", "from", "further", "had", "hadn", "has", "hasn", "have", "haven",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "isn", "it", "its", "itself", "just",
    "ll", "m", "ma", "me", "mightn", "more", "most", "mustn", "my", "myself",
    "needn", "no", "nor", "not", "now", "o", "of", "off", "on", "once", "only",
    "or", "other", "our", "ours", "ourselves", "out", "over", "own", "re", "s",
    "same", "shan", "she", "should", "shouldn", "so", "some", "such", "t", "than",
    "that", "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "ve", "very", "was", "wasn", "we", "were", "weren", "what", "when",
    "where", "which", "while", "who", "whom", "why", "will", "with", "won",
    "wouldn", "y", "you", "your", "yours", "yourself", "yourselves",
}

# Keep negation words — important for fake news detection
NEGATION_WORDS = {"no", "not", "never", "none", "nothing", "nor", "neither"}
STOP_WORDS = frozenset(_NLTK_ENGLISH_STOPWORDS - NEGATION_WORDS)

_URL_RE = re.compile(r"http\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\S+@\S+")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]", re.ASCII)
_WS_RE = re.compile(r"\s+")

# Patterns that are characteristic of fake news sensationalism
SENSATIONAL_MARKERS = re.compile(
    r'\b(breaking|urgent|shocking|exposed|leaked|bombshell|baffled|'
    r'mainstream|silenced|censored|banned|hidden|secret|proof|'
    r'must.?watch|must.?read|share.?before)\b',
    re.IGNORECASE
)

_stemmer = None


def _stem(token: str) -> str:
    global _stemmer
    if _stemmer is None:
        try:
            from nltk.stem import PorterStemmer  # optional dependency, only for stemming
        except ImportError:
            raise ImportError("use_stemming=True requires NLTK: pip install nltk") from None
        _stemmer = PorterStemmer()
    return _stemmer.stem(token)


def clean_text(text: str, use_stemming: bool = False) -> str:
    """
    Clean and normalize a single news headline.

    Steps:
    1. Lowercase and fold accents / unicode punctuation to ASCII
    2. Remove URLs and email addresses
    3. Replace digits, punctuation and any other non-letter with spaces
    4. Tokenize on whitespace
    5. Remove stopwords (preserving negations) and 1-letter tokens
    6. Optional: stem tokens

    Returns
    -------
    str — cleaned text ("" for empty / non-string input)
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    # Fold accents (é → e) by dropping combining marks; any other non-ASCII
    # character (curly quotes, dashes, mojibake) becomes a space, like ASCII punctuation.
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()

    tokens = [t for t in text.split() if t not in STOP_WORDS and len(t) > 1]

    # Stemming (optional — off by default to preserve interpretability)
    if use_stemming:
        tokens = [_stem(t) for t in tokens]

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
    hand-crafted feature columns. Rows that are empty after cleaning are dropped.
    """
    print("[Preprocessor] Cleaning text...")
    df = df.copy()
    df["cleaned_text"] = df[text_col].apply(lambda x: clean_text(x, use_stemming))

    print("[Preprocessor] Extracting hand-crafted features...")
    manual_features = pd.DataFrame(
        df[text_col].apply(extract_features_manual).tolist(), index=df.index
    )
    df = pd.concat([df, manual_features], axis=1)

    empty_mask = df["cleaned_text"].str.strip() == ""
    if empty_mask.sum() > 0:
        print(f"[Preprocessor] Dropping {empty_mask.sum()} empty rows after cleaning.")
        df = df[~empty_mask].reset_index(drop=True)

    print(f"[Preprocessor] Done. {len(df)} samples ready.")
    return df

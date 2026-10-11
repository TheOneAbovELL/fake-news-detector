"""
predict.py
----------
Inference module — the single source of truth for turning text into a
prediction. Used by the web app, the CLI verifier and the training pipeline.

The served model is described by ``models/model_card.json`` (written by
``main.py``), which records the artefact files, decision thresholds, training
dataset and held-out metrics.
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.features import TFIDFExtractor
from src.models import FakeNewsClassifier
from src.preprocessor import clean_text

MODEL_CARD_NAME = "model_card.json"

# P(fake) below LOW → likely real, above HIGH → likely fake, between → uncertain.
DEFAULT_THRESHOLDS = {"low": 0.35, "high": 0.65}

RISK_THRESHOLDS = {
    "low":    (0.0,  DEFAULT_THRESHOLDS["low"]),
    "medium": (DEFAULT_THRESHOLDS["low"], DEFAULT_THRESHOLDS["high"]),
    "high":   (DEFAULT_THRESHOLDS["high"], 1.01),
}

VERDICT_LIKELY_REAL = "LIKELY_REAL"
VERDICT_UNCERTAIN = "UNCERTAIN"
VERDICT_LIKELY_FAKE = "LIKELY_FAKE"
VERDICT_INSUFFICIENT = "INSUFFICIENT_SIGNAL"


class ModelNotFoundError(FileNotFoundError):
    """Raised when trained artefacts are missing — run `python main.py`."""


def interpret_risk(score: float, thresholds: Optional[Dict[str, float]] = None) -> str:
    t = thresholds or DEFAULT_THRESHOLDS
    if score < t["low"]:
        return "low"
    if score < t["high"]:
        return "medium"
    return "high"


def _linear_coefficients(model) -> Optional[np.ndarray]:
    """Return the per-feature weights of a (possibly calibrated) linear model."""
    if hasattr(model, "coef_"):
        return np.asarray(model.coef_).ravel()
    calibrated = getattr(model, "calibrated_classifiers_", None)
    if calibrated:
        coefs = [np.asarray(c.estimator.coef_).ravel() for c in calibrated
                 if hasattr(getattr(c, "estimator", None), "coef_")]
        if coefs:
            return np.mean(coefs, axis=0)
    return None


class Predictor:
    """Loaded vectorizer + classifier + model card, ready for inference."""

    def __init__(self, vectorizer: TFIDFExtractor, classifier: FakeNewsClassifier,
                 card: Optional[Dict] = None):
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.card = card or {}
        self.thresholds = {**DEFAULT_THRESHOLDS, **self.card.get("thresholds", {})}
        self._feature_names = np.asarray(vectorizer.get_feature_names())
        self._coef = _linear_coefficients(classifier.model)

    # ── Loading ──────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, models_dir) -> "Predictor":
        models_dir = Path(models_dir)
        card_path = models_dir / MODEL_CARD_NAME
        if not card_path.exists():
            raise ModelNotFoundError(
                f"{card_path} not found. Train the model first: python main.py"
            )
        card = json.loads(card_path.read_text(encoding="utf-8"))
        model_path = models_dir / card["model_file"]
        vectorizer_path = models_dir / card["vectorizer_file"]
        for path in (model_path, vectorizer_path):
            if not path.exists():
                raise ModelNotFoundError(f"{path} not found. Re-train: python main.py")

        trained_with = card.get("sklearn_version")
        if trained_with:
            import sklearn
            if trained_with != sklearn.__version__:
                warnings.warn(
                    f"Model was trained with scikit-learn {trained_with} but "
                    f"{sklearn.__version__} is installed; predictions may differ. "
                    "Re-train with `python main.py` or install the pinned version.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # joblib/pickle executes code on load — only ever load artefacts you built.
        vectorizer = TFIDFExtractor.load(str(vectorizer_path))
        classifier = FakeNewsClassifier.load(str(model_path), card["model_type"])
        return cls(vectorizer, classifier, card)

    # ── Inference ────────────────────────────────────────────────────────────

    def _explain(self, X, top_n: int) -> Dict[str, List[Dict]]:
        if self._coef is None or X.nnz == 0:
            return {"toward_fake": [], "toward_real": []}
        row = X.tocsr()
        idx = row.indices
        contrib = row.data * self._coef[idx]
        order = np.argsort(contrib)
        toward_real = [
            {"term": str(self._feature_names[idx[i]]), "weight": round(float(contrib[i]), 4)}
            for i in order[:top_n] if contrib[i] < 0
        ]
        toward_fake = [
            {"term": str(self._feature_names[idx[i]]), "weight": round(float(contrib[i]), 4)}
            for i in order[::-1][:top_n] if contrib[i] > 0
        ]
        return {"toward_fake": toward_fake, "toward_real": toward_real}

    def predict(self, text: str, top_n: int = 5) -> Dict:
        """
        Classify one piece of text.

        Returns a dict with:
            verdict       LIKELY_REAL | UNCERTAIN | LIKELY_FAKE | INSUFFICIENT_SIGNAL
            risk_score    P(fake) in [0, 1], or None when there is no signal
            risk_level    LOW | MEDIUM | HIGH (None without signal)
            known_terms   number of vocabulary features found in the text
            coverage      share of cleaned tokens the model has seen in training
            explanation   top terms pushing toward fake / real
        """
        cleaned = clean_text(text)
        X = self.vectorizer.transform([cleaned])
        tokens = cleaned.split()
        vocab = self.vectorizer.vectorizer.vocabulary_
        known_tokens = sum(1 for t in tokens if t in vocab)
        coverage = round(known_tokens / len(tokens), 3) if tokens else 0.0

        base = {
            "cleaned": cleaned,
            "known_terms": int(X.nnz),
            "coverage": coverage,
        }
        if X.nnz == 0:
            return {
                **base,
                "verdict": VERDICT_INSUFFICIENT,
                "risk_score": None,
                "risk_pct": None,
                "risk_level": None,
                "low_signal": True,
                "explanation": {"toward_fake": [], "toward_real": []},
            }

        risk = float(self.classifier.predict_risk_score(X)[0])
        level = interpret_risk(risk, self.thresholds)
        verdict = {"low": VERDICT_LIKELY_REAL, "medium": VERDICT_UNCERTAIN,
                   "high": VERDICT_LIKELY_FAKE}[level]
        return {
            **base,
            "verdict": verdict,
            "risk_score": round(risk, 4),
            "risk_pct": round(risk * 100, 1),
            "risk_level": level.upper(),
            "low_signal": known_tokens < 2 or coverage < 0.34,
            "explanation": self._explain(X, top_n),
        }

    def info(self) -> Dict:
        """Public, non-sensitive description of the served model."""
        keys = ("model_type", "model_name", "dataset", "metrics", "trained_at",
                "thresholds", "sklearn_version", "is_demo")
        return {k: self.card.get(k) for k in keys if k in self.card}


def predict_headlines(
    headlines: List[str],
    vectorizer: TFIDFExtractor,
    classifier: FakeNewsClassifier,
    verbose: bool = True,
) -> List[Dict]:
    """
    Run the prediction pipeline on a list of raw headlines (training demo).

    Returns list of dicts with keys:
        headline, cleaned, verdict, risk_score, risk_level
    """
    predictor = Predictor(vectorizer, classifier)
    results = []
    for headline in headlines:
        r = predictor.predict(headline)
        results.append({"headline": headline, **r})
        if verbose:
            score = "  n/a" if r["risk_score"] is None else f"{r['risk_score']:.3f}"
            print(f"  {r['verdict']:<19} [P(fake) {score}]  {headline[:80]}")
    return results


# ── Demo headlines ───────────────────────────────────────────────────────────

DEMO_HEADLINES = [
    # Real-like
    "Federal Reserve raises interest rates by 0.25 percent amid inflation concerns",
    "WHO publishes new guidelines on antibiotic resistance management",
    "Scientists discover potential link between gut bacteria and Alzheimer's disease",
    "UN Security Council meets to discuss ceasefire negotiations",
    "Apple reports record quarterly earnings, shares rise in after-hours trading",

    # Fake-like
    "BREAKING: Government secretly planning to microchip population — mainstream media silent!",
    "EXPOSED: The truth about vaccines they don't want you to know — share before deleted",
    "Doctors HATE this: This one trick cures cancer overnight, elites are hiding it",
    "SHOCKING: Deep state operative caught on camera plotting to rig elections",
    "Tech billionaire CONFIRMS chemtrail conspiracy in leaked bombshell video",
]

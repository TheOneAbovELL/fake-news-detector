"""
predict.py
----------
Inference module — run predictions on new headlines.
"""

import os
import joblib
from typing import List, Dict
from src.preprocessor import clean_text
from src.features import TFIDFExtractor
from src.models import FakeNewsClassifier


RISK_THRESHOLDS = {
    "low":    (0.0,  0.35),
    "medium": (0.35, 0.65),
    "high":   (0.65, 1.01),
}


def interpret_risk(score: float) -> str:
    for label, (low, high) in RISK_THRESHOLDS.items():
        if low <= score < high:
            return label
    return "high"


def predict_headlines(
    headlines: List[str],
    vectorizer: TFIDFExtractor,
    classifier: FakeNewsClassifier,
    verbose: bool = True,
) -> List[Dict]:
    """
    Run full prediction pipeline on a list of raw headlines.

    Returns list of dicts with keys:
        headline, cleaned, prediction, risk_score, risk_level
    """
    results = []
    for headline in headlines:
        cleaned = clean_text(headline)
        X = vectorizer.transform([cleaned])
        pred = classifier.predict(X)[0]
        risk_score = classifier.predict_risk_score(X)[0]
        risk_level = interpret_risk(risk_score)

        result = {
            "headline":   headline,
            "cleaned":    cleaned,
            "prediction": "FAKE" if pred == 1 else "REAL",
            "risk_score": round(float(risk_score), 4),
            "risk_level": risk_level.upper(),
        }
        results.append(result)

        if verbose:
            verdict = "🔴 FAKE" if pred == 1 else "🟢 REAL"
            print(f"  {verdict}  [{risk_level.upper()} risk: {risk_score:.3f}]  {headline[:80]}")

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

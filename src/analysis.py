"""
analysis.py
-----------
Combines the two signals TruthLens offers into one honest, structured result:

1. **ML signal** — TF-IDF + linear classifier trained on FakeNewsNet headlines.
   Fast and deterministic, but it only judges *wording*: it has no knowledge of
   facts and was trained on 2016-2018 US politics / celebrity news.
2. **LLM assessment** (optional) — a Groq-hosted LLM reasons about the content.
   More capable, but non-deterministic and fallible.

The overall verdict uses the LLM when it is available (it can reason about
content) and the ML signal otherwise. Both raw signals are always returned and
disagreement between them is surfaced, never hidden.
"""

import logging
from typing import Dict, Optional

from src.llm import GroqClient, LLMResult
from src.predict import (
    VERDICT_INSUFFICIENT,
    VERDICT_LIKELY_FAKE,
    VERDICT_LIKELY_REAL,
    VERDICT_UNCERTAIN,
    Predictor,
)

log = logging.getLogger(__name__)

_LLM_TO_VERDICT = {
    "REAL": VERDICT_LIKELY_REAL,
    "LIKELY REAL": VERDICT_LIKELY_REAL,
    "UNCERTAIN": VERDICT_UNCERTAIN,
    "LIKELY FAKE": VERDICT_LIKELY_FAKE,
    "FAKE": VERDICT_LIKELY_FAKE,
}

_OPPOSITES = {
    (VERDICT_LIKELY_FAKE, VERDICT_LIKELY_REAL),
    (VERDICT_LIKELY_REAL, VERDICT_LIKELY_FAKE),
}

LLM_UNAVAILABLE_MESSAGE = "AI analysis is temporarily unavailable; showing the ML signal only."


class Analyzer:
    def __init__(self, predictor: Predictor, llm: Optional[GroqClient] = None):
        self.predictor = predictor
        self.llm = llm

    def _llm(self, text: str, mode: str, context: Optional[Dict[str, str]]) -> LLMResult:
        if self.llm is None:
            return LLMResult(status="disabled")
        return self.llm.analyse(text, mode=mode, context=context)

    def analyse(self, ml_text: str, llm_text: Optional[str] = None, mode: str = "text",
                meta: Optional[Dict[str, str]] = None) -> Dict:
        """
        Parameters
        ----------
        ml_text  : text for the headline classifier (a headline works best)
        llm_text : fuller text for the LLM (defaults to ml_text)
        mode     : "text" or "url"
        meta     : display metadata (headline, source, url, preview)
        """
        meta = meta or {}
        ml = self.predictor.predict(ml_text)
        ml.pop("cleaned", None)

        llm_context = {k: meta.get(k) for k in ("source", "headline") if meta.get(k)}
        llm_result = self._llm(llm_text or ml_text, mode, llm_context or None)

        notes = []
        if llm_result.status == "ok":
            verdict = _LLM_TO_VERDICT[llm_result.data["verdict"]]
            basis = "llm+ml"
            if ml["verdict"] == VERDICT_INSUFFICIENT:
                agreement = "n/a"
            elif (verdict, ml["verdict"]) in _OPPOSITES:
                agreement = "disagree"
                notes.append("The AI assessment and the wording-based ML signal disagree — "
                             "treat this result with extra caution and check the claims yourself.")
            else:
                agreement = "agree" if verdict == ml["verdict"] else "partial"
            llm_payload = {"status": "ok", "model": llm_result.model, **llm_result.data}
        else:
            verdict = ml["verdict"]
            basis = "ml"
            agreement = "n/a"
            llm_payload = {"status": llm_result.status}
            if llm_result.status == "error":
                llm_payload["message"] = LLM_UNAVAILABLE_MESSAGE

        if ml["verdict"] == VERDICT_INSUFFICIENT:
            notes.append("The ML model recognised none of these words, so it cannot score this text.")
        elif ml.get("low_signal"):
            notes.append("The ML model recognised few of these words; its score is weak evidence.")

        info = self.predictor.info()
        if info.get("is_demo"):
            notes.append("This server is running a DEMO model trained on synthetic data — "
                         "its scores are not meaningful.")

        risk = ml["risk_score"]
        return {
            "mode": mode,
            "verdict": verdict,
            "verdict_basis": basis,
            "agreement": agreement,
            "ml": ml,
            "llm": llm_payload,
            "notes": notes,
            "headline": meta.get("headline") or ml_text[:160],
            "source": meta.get("source"),
            "url": meta.get("url"),
            "preview": meta.get("preview"),
            # Backwards-compatible fields (derived from the ML signal)
            "prediction": None if risk is None else ("FAKE" if risk >= 0.5 else "REAL"),
            "is_fake": verdict == VERDICT_LIKELY_FAKE,
            "risk_score": risk,
            "risk_pct": ml["risk_pct"],
            "risk_level": ml["risk_level"],
            "analysis_source": "groq+ml" if basis == "llm+ml" else "ml_only",
        }

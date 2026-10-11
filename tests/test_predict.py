"""Predictor + Analyzer tests (shared inference path for the app and CLI)."""

import json
import shutil

import pytest

from src.analysis import Analyzer
from src.llm import LLMResult
from src.predict import (
    MODEL_CARD_NAME,
    ModelNotFoundError,
    Predictor,
    interpret_risk,
)
from src.preprocessor import STOP_WORDS, clean_text


class TestPreprocessorDeterminism:
    def test_negations_kept(self):
        assert "not" in clean_text("This is not true") and "not" not in STOP_WORDS

    def test_unicode_punctuation_and_accents_folded(self):
        assert clean_text("Café’s “BREAKING” news — Zoë") == "cafe breaking news zoe"

    def test_mojibake_replacement_chars_removed(self):
        assert clean_text("Ocean�s 8 �catfight� rumours") == "ocean catfight rumours"


class TestInterpretRisk:
    @pytest.mark.parametrize("score,level", [(0.0, "low"), (0.349, "low"), (0.35, "medium"),
                                             (0.649, "medium"), (0.65, "high"), (1.0, "high")])
    def test_bands(self, score, level):
        assert interpret_risk(score) == level


class TestPredictor:
    def test_fake_like_headline(self, predictor):
        r = predictor.predict("BREAKING: Deep state operative REVEALS plot to poison the water")
        assert r["verdict"] == "LIKELY_FAKE" and r["risk_level"] == "HIGH"
        assert 0 <= r["risk_score"] <= 1 and r["risk_pct"] == round(r["risk_score"] * 100, 1)
        assert r["explanation"]["toward_fake"], "fake-leaning terms should be explained"
        assert all(t["weight"] > 0 for t in r["explanation"]["toward_fake"])

    def test_real_like_headline(self, predictor):
        r = predictor.predict("Reuters reports IMF publishes annual report on inflation")
        assert r["verdict"] == "LIKELY_REAL"
        assert all(t["weight"] < 0 for t in r["explanation"]["toward_real"])

    def test_unknown_vocabulary_gives_no_verdict(self, predictor):
        r = predictor.predict("Xyzzy plugh frobnicate quux")
        assert r["verdict"] == "INSUFFICIENT_SIGNAL"
        assert r["risk_score"] is None and r["risk_level"] is None and r["known_terms"] == 0

    def test_partial_vocabulary_flags_low_signal(self, predictor):
        r = predictor.predict("Xyzzy plugh frobnicate quux zorg blarg inflation")
        assert r["verdict"] != "INSUFFICIENT_SIGNAL" and r["low_signal"] is True

    def test_info_is_public_and_has_metrics(self, predictor):
        info = predictor.info()
        assert info["model_type"] == "logistic_regression"
        assert "metrics" in info and "model_file" not in info

    def test_missing_card(self, tmp_path):
        with pytest.raises(ModelNotFoundError, match="python main.py"):
            Predictor.load(tmp_path)

    def test_missing_artifact(self, models_dir, tmp_path):
        shutil.copy(models_dir / MODEL_CARD_NAME, tmp_path / MODEL_CARD_NAME)
        with pytest.raises(ModelNotFoundError):
            Predictor.load(tmp_path)

    def test_sklearn_version_mismatch_warns(self, models_dir, tmp_path):
        for f in models_dir.iterdir():
            shutil.copy(f, tmp_path / f.name)
        card = json.loads((tmp_path / MODEL_CARD_NAME).read_text())
        card["sklearn_version"] = "0.0.1"
        (tmp_path / MODEL_CARD_NAME).write_text(json.dumps(card))
        with pytest.warns(RuntimeWarning, match="scikit-learn"):
            Predictor.load(tmp_path)


class StubLLM:
    def __init__(self, result):
        self.result = result
        self.calls = []

    @property
    def enabled(self):
        return self.result.status != "disabled"

    def analyse(self, text, mode="text", context=None):
        self.calls.append((text, mode, context))
        return self.result


def llm_ok(verdict):
    return LLMResult(status="ok", model="m", data={
        "verdict": verdict, "confidence": 70, "credibility_score": 30, "summary": "s",
        "red_flags": [], "green_flags": [], "detected_techniques": [],
        "fact_check_suggestions": [], "reasoning": "r"})


FAKE_TEXT = "BREAKING: Deep state operative REVEALS plot to poison the water"
REAL_TEXT = "Reuters reports IMF publishes annual report on inflation"


class TestAnalyzer:
    def test_ml_only_when_llm_disabled(self, predictor):
        out = Analyzer(predictor, StubLLM(LLMResult(status="disabled"))).analyse(FAKE_TEXT)
        assert out["verdict"] == "LIKELY_FAKE" and out["verdict_basis"] == "ml"
        assert out["llm"] == {"status": "disabled"} and out["analysis_source"] == "ml_only"

    def test_llm_error_is_reported_without_details(self, predictor):
        stub = StubLLM(LLMResult(status="error", error="LLM credentials were rejected"))
        out = Analyzer(predictor, stub).analyse(FAKE_TEXT)
        assert out["verdict_basis"] == "ml" and out["llm"]["status"] == "error"
        assert "credentials" not in json.dumps(out)

    def test_llm_verdict_used_when_agreeing(self, predictor):
        out = Analyzer(predictor, StubLLM(llm_ok("FAKE"))).analyse(FAKE_TEXT)
        assert out["verdict"] == "LIKELY_FAKE" and out["agreement"] == "agree"
        assert out["verdict_basis"] == "llm+ml" and out["llm"]["verdict"] == "FAKE"

    def test_disagreement_is_surfaced(self, predictor):
        out = Analyzer(predictor, StubLLM(llm_ok("LIKELY REAL"))).analyse(FAKE_TEXT)
        assert out["verdict"] == "LIKELY_REAL" and out["agreement"] == "disagree"
        assert any("disagree" in n for n in out["notes"])
        assert out["ml"]["verdict"] == "LIKELY_FAKE"  # raw ML signal still returned

    def test_partial_agreement(self, predictor):
        out = Analyzer(predictor, StubLLM(llm_ok("UNCERTAIN"))).analyse(REAL_TEXT)
        assert out["agreement"] == "partial"

    def test_llm_receives_full_text_and_context(self, predictor):
        stub = StubLLM(llm_ok("UNCERTAIN"))
        Analyzer(predictor, stub).analyse("Short headline inflation", llm_text="Full body text",
                                          mode="url", meta={"source": "x.org", "headline": "H"})
        assert stub.calls[0] == ("Full body text", "url", {"source": "x.org", "headline": "H"})

    def test_insufficient_signal_note(self, predictor):
        out = Analyzer(predictor, None).analyse("Xyzzy plugh frobnicate")
        assert out["verdict"] == "INSUFFICIENT_SIGNAL" and out["prediction"] is None
        assert any("recognised none" in n for n in out["notes"])

    def test_demo_model_flagged(self, predictor):
        demo = Predictor(predictor.vectorizer, predictor.classifier, {**predictor.card, "is_demo": True})
        out = Analyzer(demo, None).analyse(FAKE_TEXT)
        assert any("DEMO" in n for n in out["notes"])

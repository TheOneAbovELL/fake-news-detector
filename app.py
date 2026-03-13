"""
app.py
Flask web application for the Fake News Detection System.

Run with:
    python app.py
Then open: http://localhost:5000
"""

import os
from dotenv import load_dotenv
load_dotenv()
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, render_template, request, jsonify
from src.preprocessor import clean_text
from src.features     import TFIDFExtractor
from src.models       import FakeNewsClassifier
from src.scraper      import scrape_article, validate_url

import requests
import json

app = Flask(__name__)

# ── Groq LLM config ──────────────────────────────────────────────────────────
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL    = "llama-3.3-70b-versatile"
GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "")  # set via env variable

# ── Load models once at startup ───────────────────────────────────────────────
MODEL_PATH      = os.path.join("models", "svm_linear.pkl")
VECTORIZER_PATH = os.path.join("models", "tfidf_vectorizer.pkl")

vectorizer = None
classifier = None

def groq_analyse(text: str, mode: str = "text", extra_context: dict = None) -> dict:
    """
    Send text to Groq LLaMA-3.3-70b for deep fake news analysis.
    Returns structured dict with verdict, scores, flags, reasoning.
    Falls back gracefully if API key missing or call fails.
    """
    if not GROQ_API_KEY:
        return None  # no key = skip LLM layer, use ML only

    context_block = ""
    if extra_context:
        context_block = f"""
SOURCE DOMAIN : {extra_context.get('source', 'unknown')}
HEADLINE      : {extra_context.get('headline', '')}
PREVIEW       : {extra_context.get('preview', '')}
"""

    system_prompt = (
        "You are an expert fact-checker and misinformation analyst. "
        "Respond ONLY with a valid JSON object — no markdown, no explanation outside JSON."
    )

    user_prompt = f"""Analyse this {'news article' if mode == 'url' else 'text'} for credibility and misinformation.

{context_block}
TEXT TO ANALYSE:
{text[:3000]}

Return EXACTLY this JSON structure:
{{
  "verdict": "REAL" or "LIKELY REAL" or "UNCERTAIN" or "LIKELY FAKE" or "FAKE",
  "confidence": <integer 0-100>,
  "credibility_score": <integer 0-100>,
  "summary": "<2-3 sentence plain English explanation>",
  "red_flags": ["<specific red flag found in the text>"],
  "green_flags": ["<specific credibility signal found>"],
  "detected_techniques": ["<manipulation technique name if found>"],
  "fact_check_suggestions": ["<specific claim to independently verify>"],
  "reasoning": "<detailed 3-5 sentence analysis of why this verdict>"
}}"""

    try:
        response = requests.post(
            GROQ_ENDPOINT,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GROQ_API_KEY}",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                "temperature": 0.1,
                "max_tokens": 1200,
            },
            timeout=20,
        )
        response.raise_for_status()
        raw   = response.json()["choices"][0]["message"]["content"]
        clean = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except Exception as e:
        print(f"[Groq] LLM call failed: {e}")
        return None  # fail silently, ML result still returned
    

def load_models():
    global vectorizer, classifier
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        print("\n❌  Models not found! Run  python main.py  first.\n")
        sys.exit(1)
    vectorizer = TFIDFExtractor.load(VECTORIZER_PATH)
    classifier = FakeNewsClassifier.load(MODEL_PATH, "svm")
    print("✅  Models loaded successfully.")


def predict(text: str) -> dict:
    """Run prediction and return structured result."""
    cleaned    = clean_text(text)
    X          = vectorizer.transform([cleaned])
    pred       = int(classifier.predict(X)[0])
    risk_score = float(classifier.predict_risk_score(X)[0])

    if risk_score < 0.35:
        risk_level = "LOW"
    elif risk_score < 0.65:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    return {
        "prediction": "FAKE" if pred == 1 else "REAL",
        "is_fake":    pred == 1,
        "risk_score": round(risk_score, 4),
        "risk_pct":   round(risk_score * 100, 1),
        "risk_level": risk_level,
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/verify-url", methods=["POST"])
def verify_url():
    data = request.get_json()
    url  = (data or {}).get("url", "").strip()

    if not url:
        return jsonify({"error": "No URL provided"}), 400
    if not validate_url(url):
        return jsonify({"error": "Invalid URL. Make sure it starts with https://"}), 400

    article = scrape_article(url)
    if not article["success"]:
        return jsonify({"error": article["error"] or "Could not extract article content"}), 422

    result = predict(article["text_for_analysis"])
    return jsonify({
        **result,
        "headline": article["headline"],
        "source":   article["source"],
        "preview":  article["body"][:200] + "..." if len(article.get("body","")) > 200 else article.get("body",""),
        "url":      url,
        "mode":     "url",
    })


@app.route("/api/verify-text", methods=["POST"])
def verify_text():
    data = request.get_json()
    text = (data or {}).get("text", "").strip()

    if not text or len(text) < 5:
        return jsonify({"error": "Please enter a headline or text to analyse"}), 400

    # ── Layer 1: ML model (fast baseline) ──
    ml_result = predict(text)

    # ── Layer 2: Groq LLM (deep analysis) ──
    llm = groq_analyse(text, mode="text")

    # ── Merge: LLM verdict overrides ML when available ──
    if llm:
        # Map LLM verdict to is_fake / risk_score
        verdict_map = {
            "FAKE":        (True,  0.92),
            "LIKELY FAKE": (True,  0.72),
            "UNCERTAIN":   (False, 0.50),
            "LIKELY REAL": (False, 0.25),
            "REAL":        (False, 0.08),
        }
        is_fake, risk = verdict_map.get(llm["verdict"], (ml_result["is_fake"], ml_result["risk_score"]))
        risk_level = "HIGH" if risk > 0.65 else "MEDIUM" if risk > 0.35 else "LOW"

        return jsonify({
            **ml_result,                          # keep ML fields as base
            "is_fake":              is_fake,
            "prediction":           "FAKE" if is_fake else "REAL",
            "risk_score":           risk,
            "risk_pct":             round(risk * 100, 1),
            "risk_level":           risk_level,
            "headline":             text[:120],
            "mode":                 "text",
            # ── New LLM fields ──
            "llm_verdict":          llm["verdict"],
            "llm_confidence":       llm.get("confidence", 0),
            "credibility_score":    llm.get("credibility_score", 0),
            "summary":              llm.get("summary", ""),
            "red_flags":            llm.get("red_flags", []),
            "green_flags":          llm.get("green_flags", []),
            "detected_techniques":  llm.get("detected_techniques", []),
            "fact_check_suggestions": llm.get("fact_check_suggestions", []),
            "reasoning":            llm.get("reasoning", ""),
            "analysis_source":      "groq+ml",
        })

    # ── Groq unavailable: return ML result only ──
    return jsonify({
        **ml_result,
        "headline":        text[:120],
        "mode":            "text",
        "analysis_source": "ml_only",
    })


if __name__ == "__main__":
    load_models()
    print("\n🚀  Starting Fake News Detector...")
    print("🌐  Open http://localhost:5000 in your browser\n")
    app.run(debug=False, host="0.0.0.0", port=5000)

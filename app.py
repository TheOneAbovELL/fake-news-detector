"""
app.py
------
Flask web application for the Fake News Detection System.
BIT Mesra 2024

Run with:
    python app.py
Then open: http://localhost:5000
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, render_template, request, jsonify
from src.preprocessor import clean_text
from src.features     import TFIDFExtractor
from src.models       import FakeNewsClassifier
from src.scraper      import scrape_article, validate_url

app = Flask(__name__)

# ── Load models once at startup ───────────────────────────────────────────────
MODEL_PATH      = os.path.join("models", "svm_linear.pkl")
VECTORIZER_PATH = os.path.join("models", "tfidf_vectorizer.pkl")

vectorizer = None
classifier = None


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

    result = predict(text)
    return jsonify({
        **result,
        "headline": text[:120],
        "mode":     "text",
    })


if __name__ == "__main__":
    load_models()
    print("\n🚀  Starting Fake News Detector...")
    print("🌐  Open http://localhost:5000 in your browser\n")
    app.run(debug=False, host="0.0.0.0", port=5000)

"""
verify.py
---------
Interactive Fake News Verifier — BIT Mesra 2024

Usage:
    python verify.py                        # interactive menu
    python verify.py --url <URL>            # verify a URL
    python verify.py --text "headline"      # verify a headline
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

from src.preprocessor import clean_text
from src.features     import TFIDFExtractor
from src.models       import FakeNewsClassifier
from src.scraper      import scrape_article, validate_url, SCRAPER_AVAILABLE

MODEL_PATH      = os.path.join("models", "svm_linear.pkl")
VECTORIZER_PATH = os.path.join("models", "tfidf_vectorizer.pkl")

RISK_THRESHOLDS = {
    "LOW":    (0.00, 0.35),
    "MEDIUM": (0.35, 0.65),
    "HIGH":   (0.65, 1.01),
}


def load_models():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        print("\n❌  Models not found! Run  python main.py  first.\n")
        sys.exit(1)
    vectorizer = TFIDFExtractor.load(VECTORIZER_PATH)
    classifier = FakeNewsClassifier.load(MODEL_PATH, "svm")
    return vectorizer, classifier


def interpret_risk(score):
    for label, (lo, hi) in RISK_THRESHOLDS.items():
        if lo <= score < hi:
            return label
    return "HIGH"


def predict_text(text, vectorizer, classifier):
    cleaned    = clean_text(text)
    X          = vectorizer.transform([cleaned])
    pred       = classifier.predict(X)[0]
    risk_score = float(classifier.predict_risk_score(X)[0])
    return {
        "prediction": "FAKE" if pred == 1 else "REAL",
        "risk_score": round(risk_score, 4),
        "risk_level": interpret_risk(risk_score),
    }


def print_result(result, source_text, source_label="Input"):
    is_fake      = result["prediction"] == "FAKE"
    verdict_icon = "🔴" if is_fake else "🟢"
    verdict      = "FAKE NEWS" if is_fake else "REAL NEWS"
    score        = result["risk_score"]
    risk         = result["risk_level"]
    bar          = "█" * int(score * 20) + "░" * (20 - int(score * 20))

    print("\n" + "═" * 62)
    print(f"  {verdict_icon}  VERDICT: {verdict}")
    print("═" * 62)
    print(f"  {source_label:<14}: {source_text[:70]}")
    print(f"  {'Risk Score':<14}: {score:.4f}  [{bar}]  {risk} RISK")
    print("─" * 62)

    if is_fake:
        msg = "⚠️  Strong indicators of misinformation detected." if risk == "HIGH" \
              else "⚠️  Possible misinformation — verify with trusted sources."
    else:
        msg = "✅  Content appears consistent with credible reporting." if risk == "LOW" \
              else "🟡  Likely real, but treat with moderate caution."

    print(f"\n  {msg}")
    print("\n  ℹ️  This is a text classifier — always cross-check with trusted sources.")
    print("═" * 62 + "\n")


def verify_url(url, vectorizer, classifier):
    print(f"\n  🌐 Fetching: {url}\n  ⏳ Please wait...\n")
    article = scrape_article(url)
    if not article["success"]:
        print(f"\n  ❌ {article['error']}")
        print("  💡 Try pasting the headline manually instead.\n")
        return
    print(f"  📰 Source  : {article['source']}")
    print(f"  📌 Headline: {article['headline'][:80]}")
    result = predict_text(article["text_for_analysis"], vectorizer, classifier)
    print_result(result, article["headline"] or url, "Headline")


def verify_text(text, vectorizer, classifier):
    result = predict_text(text, vectorizer, classifier)
    print_result(result, text, "Input")


BANNER = """
╔══════════════════════════════════════════════════════════╗
║              🔍 FAKE NEWS VERIFIER                       ║
║                                                          ║
║  Paste a news URL or type a headline to check if it's    ║
║  real or fake. Type  exit  to quit.                      ║
╚══════════════════════════════════════════════════════════╝
"""

MENU = """
  [1] Paste a news URL
  [2] Type / paste a headline
  [3] Batch check multiple headlines
  [0] Exit
"""


def interactive_mode(vectorizer, classifier):
    print(BANNER)
    if not SCRAPER_AVAILABLE:
        print("  ⚠️  URL scraping unavailable. Run: pip install requests beautifulsoup4\n")
    while True:
        print(MENU)
        choice = input("  Enter choice: ").strip()
        if choice in ("0", "exit", "quit", "q"):
            print("\n  👋 Goodbye!\n")
            break
        elif choice == "1":
            if not SCRAPER_AVAILABLE:
                print("\n  ❌ Install requests & beautifulsoup4 first.\n")
                continue
            url = input("\n  🌐 Paste URL: ").strip()
            if not validate_url(url):
                print("  ❌ Invalid URL.\n")
                continue
            verify_url(url, vectorizer, classifier)
        elif choice == "2":
            text = input("\n  📝 Paste headline: ").strip()
            if len(text) < 5:
                print("  ❌ Too short.\n")
                continue
            verify_text(text, vectorizer, classifier)
        elif choice == "3":
            print("\n  Enter headlines one per line. Type DONE when finished.\n")
            headlines = []
            while True:
                line = input("  > ").strip()
                if line.upper() == "DONE":
                    break
                if line:
                    headlines.append(line)
            for i, h in enumerate(headlines, 1):
                print(f"\n  [{i}/{len(headlines)}]")
                verify_text(h, vectorizer, classifier)
        else:
            print("  ❌ Invalid choice.\n")


def main():
    parser = argparse.ArgumentParser(description="Fake News Verifier")
    parser.add_argument("--url",  type=str)
    parser.add_argument("--text", type=str)
    args = parser.parse_args()
    vectorizer, classifier = load_models()
    if args.url:
        verify_url(args.url, vectorizer, classifier)
    elif args.text:
        verify_text(args.text, vectorizer, classifier)
    else:
        interactive_mode(vectorizer, classifier)


if __name__ == "__main__":
    main()
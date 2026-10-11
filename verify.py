"""
verify.py
---------
Command-line news verifier (same analysis as the web app).

Usage:
    python verify.py                        # interactive menu
    python verify.py --url <URL>            # verify a URL
    python verify.py --text "headline"      # verify a headline
    python verify.py --text "..." --json    # machine-readable output
    python verify.py --no-llm ...           # skip the Groq LLM layer
"""

import argparse
import json
import os
import sys
import warnings

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.analysis import Analyzer  # noqa: E402
from src.config import Settings  # noqa: E402
from src.llm import GroqClient  # noqa: E402
from src.predict import ModelNotFoundError, Predictor  # noqa: E402
from src.scraper import scrape_article, validate_url  # noqa: E402

LABELS = {
    "LIKELY_FAKE": ("🔴", "LIKELY FAKE"),
    "UNCERTAIN": ("🟡", "UNCERTAIN"),
    "LIKELY_REAL": ("🟢", "LIKELY CREDIBLE"),
    "INSUFFICIENT_SIGNAL": ("⚪", "NOT ENOUGH SIGNAL"),
}


def build_analyzer(use_llm: bool = True):
    load_dotenv()
    settings = Settings.from_env()
    try:
        predictor = Predictor.load(settings.models_dir)
    except ModelNotFoundError as e:
        print(f"\n❌  {e}\n")
        sys.exit(1)
    llm = GroqClient(settings.groq_api_key if use_llm else "", settings.groq_model,
                     settings.groq_endpoint, settings.groq_timeout, settings.groq_max_retries)
    return Analyzer(predictor, llm), settings


def print_result(result: dict):
    icon, label = LABELS.get(result["verdict"], ("⚪", result["verdict"]))
    ml, llm = result["ml"], result["llm"]
    print("\n" + "═" * 64)
    print(f"  {icon}  VERDICT: {label}   (basis: {result['verdict_basis']})")
    print("═" * 64)
    if result.get("source"):
        print(f"  Source        : {result['source']}")
    print(f"  Headline      : {(result.get('headline') or '')[:70]}")
    if ml["risk_score"] is None:
        print("  ML signal     : no known vocabulary — cannot score")
    else:
        bar = "█" * int(ml["risk_score"] * 20) + "░" * (20 - int(ml["risk_score"] * 20))
        print(f"  ML signal     : P(fake) {ml['risk_score']:.3f} [{bar}] {ml['risk_level']}")
        fake_terms = ", ".join(t["term"] for t in ml["explanation"]["toward_fake"][:4])
        real_terms = ", ".join(t["term"] for t in ml["explanation"]["toward_real"][:4])
        if fake_terms:
            print(f"    ↑ fake-like : {fake_terms}")
        if real_terms:
            print(f"    ↓ real-like : {real_terms}")
    if llm["status"] == "ok":
        print(f"  AI assessment : {llm['verdict']} (credibility {llm['credibility_score']}/100, "
              f"confidence {llm['confidence']}/100)")
        if llm.get("summary"):
            print(f"    {llm['summary']}")
        for claim in llm.get("fact_check_suggestions", [])[:3]:
            print(f"    • verify: {claim}")
    else:
        print(f"  AI assessment : {llm['status']}")
    for note in result.get("notes", []):
        print(f"  ⚠ {note}")
    print("─" * 64)
    print("  ℹ  Signals, not proof — always cross-check with trusted sources.")
    print("═" * 64 + "\n")


def verify_url(url, analyzer, settings, as_json=False):
    if not validate_url(url):
        print("  ❌ Invalid URL.")
        return None
    article = scrape_article(url, timeout=settings.scrape_timeout,
                             max_bytes=settings.scrape_max_bytes,
                             allow_private=settings.allow_private_urls)
    if not article["success"]:
        print(f"\n  ❌ {article['error']}\n  💡 Try pasting the headline instead.\n")
        return None
    result = analyzer.analyse(
        ml_text=article["headline"] or article["body"][:200],
        llm_text=f"{article['headline']}\n\n{article['body']}".strip(), mode="url",
        meta={k: article[k] for k in ("headline", "source", "url", "preview")})
    print(json.dumps(result, indent=2)) if as_json else print_result(result)
    return result


def verify_text(text, analyzer, as_json=False):
    first_line = text.strip().splitlines()[0] if text.strip() else text
    result = analyzer.analyse(ml_text=first_line[:500], llm_text=text, mode="text",
                              meta={"headline": first_line[:160]})
    print(json.dumps(result, indent=2)) if as_json else print_result(result)
    return result


MENU = """
  [1] Paste a news URL
  [2] Type / paste a headline
  [3] Batch check multiple headlines
  [0] Exit
"""


def interactive_mode(analyzer, settings):
    print("\n  🔍 TruthLens verifier — paste a URL or headline. Type 0 to quit.")
    while True:
        print(MENU)
        choice = input("  Enter choice: ").strip()
        if choice in ("0", "exit", "quit", "q"):
            print("\n  👋 Goodbye!\n")
            break
        if choice == "1":
            verify_url(input("\n  🌐 Paste URL: ").strip(), analyzer, settings)
        elif choice == "2":
            text = input("\n  📝 Paste headline: ").strip()
            if len(text) < 5:
                print("  ❌ Too short.\n")
                continue
            verify_text(text, analyzer)
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
                verify_text(h, analyzer)
        else:
            print("  ❌ Invalid choice.\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="TruthLens fake news verifier")
    parser.add_argument("--url", type=str)
    parser.add_argument("--text", type=str)
    parser.add_argument("--json", action="store_true", help="print the raw JSON result")
    parser.add_argument("--no-llm", action="store_true", help="skip the Groq LLM layer")
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    analyzer, settings = build_analyzer(use_llm=not args.no_llm)
    if args.url:
        verify_url(args.url, analyzer, settings, args.json)
    elif args.text:
        verify_text(args.text, analyzer, args.json)
    else:
        interactive_mode(analyzer, settings)


if __name__ == "__main__":
    main()

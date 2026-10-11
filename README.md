# 🔍 TruthLens — News Credibility Checker

> Paste a news URL or headline and get **credibility signals**: a wording-based ML score with word-level explanations, plus an optional LLM fact-check assessment. TruthLens gives you signals, not a final ruling.

![TruthLens UI](assets/screenshot.png)

---

## What it does

| Signal | How it works | Strengths | Limits |
|---|---|---|---|
| **Wording signal (ML)** | TF-IDF (uni+bigrams) → class-balanced logistic regression, trained on ~21.6k real headlines labelled by fact-checkers ([FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet)) | Fast, deterministic, offline, explains *which words* moved the score | Judges wording only, has no knowledge of facts, and was trained on 2016–2018 US politics and celebrity news |
| **AI assessment (optional)** | A Groq-hosted LLM (default `llama-3.3-70b-versatile`) reviews the text for sourcing, plausibility and manipulation techniques, and lists claims to verify | Can reason about content | Non-deterministic, can't browse the web, can be wrong |

The overall verdict is one of **Likely fake · Uncertain · Likely credible · Not enough signal**. When the LLM is available its verdict is used; otherwise the ML signal is. Both raw signals are always shown, and **disagreement between them is flagged**. If the ML model recognises none of the words, it says so instead of guessing.

---

## 📊 Results (held-out test set, 4,318 headlines)

The served model is logistic regression. All numbers come from a stratified 80/20 split, with deduplicated titles so no headline appears in both train and test. Regularisation was chosen by 5-fold cross-validation on the training split only.

| Model | Accuracy | Precision (fake) | Recall (fake) | F1 (fake) | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** (served) | 0.820 | 0.606 | 0.721 | 0.659 | **0.865** |
| SVM (linear, calibrated) | 0.840 | 0.727 | 0.539 | 0.619 | 0.865 |

Only 24% of headlines are fake, so accuracy alone is misleading (always answering "real" scores 76%). ROC-AUC and fake-class F1 are the meaningful numbers. Cross-validated ROC-AUC: 0.870 ± 0.006.

**What the verdict bands mean in practice** (served model, test set):

| Verdict | Share of headlines | Actually fake |
|---|---|---|
| Likely fake (P ≥ 0.65) | 19.6% | **71%** |
| Uncertain | 21.9% | 27% |
| Likely credible (P < 0.35) | 58.5% | 7% |

By subset: GossipCop (celebrity news) ROC-AUC 0.869; PolitiFact (politics) ROC-AUC 0.813.

![Model comparison](assets/model_comparison.png)
![ROC curves](assets/roc_curve.png)
![Most indicative terms](assets/feature_importance.png)

The most indicative terms show the main caveat: the model partly learns **topics** (celebrity names, "trump") as well as style ("breaking", "report", "jealous"). That's a known property of FakeNewsNet. It's why TruthLens presents the ML score as one weak signal, shows the words behind it, and pairs it with the AI assessment.

> **Note on v1:** the original version trained on 6,000 template-generated headlines. It scored 100% on its own templates but failed on real news (e.g. "Pope Francis endorses Donald Trump" scored 0.2% fake). The PDF report in this repo describes that version.

---

## 🏗️ Architecture

```
Browser (templates/index.html + static/app.js, strict CSP)
   │  POST /api/verify-text {text}      POST /api/verify-url {url}
   ▼
app.py — Flask app factory: validation, rate limit, security headers, JSON errors
   │                                        │
   │                              src/scraper.py — SSRF-safe fetch + HTML extraction
   ▼                                        ▼
src/analysis.py — Analyzer: combines signals, flags disagreement
   ├── src/predict.py — Predictor: TF-IDF → logistic regression, explanations
   │      (models/*.pkl + models/model_card.json, produced by main.py)
   └── src/llm.py — GroqClient: JSON mode, schema validation, retries, injection hardening
```

```
fake-news-detector/
├── app.py               # Flask web app (factory: create_app)
├── main.py              # Training pipeline → models/ + reports/
├── verify.py            # CLI verifier (same analysis as the web app)
├── src/
│   ├── config.py        # Validated settings from environment variables
│   ├── data_loader.py   # FakeNewsNet download/load/dedup (+ synthetic test fixture)
│   ├── preprocessor.py  # Deterministic text cleaning (no network)
│   ├── features.py      # TF-IDF extractor
│   ├── models.py        # LR / SVM wrappers
│   ├── evaluate.py      # Metrics, per-subset and per-band evaluation
│   ├── predict.py       # Predictor (single inference path) + model card loading
│   ├── analysis.py      # Combines ML + LLM into the API result
│   ├── llm.py           # Groq client
│   ├── scraper.py       # URL fetching with SSRF protection
│   └── visualize.py     # Training figures
├── templates/index.html, static/app.{css,js}
├── tests/               # 166 tests (pytest)
├── Dockerfile, docker-compose.yml, .github/workflows/ci.yml
└── requirements.txt, requirements-dev.txt, .env.example
```

---

## ⚙️ Setup

Requires **Python 3.12+** (3.13 tested).

```bash
git clone https://github.com/OmjeeRGiri/fake-news-detector.git
cd fake-news-detector
python -m venv venv
venv\Scripts\activate            # Windows
source venv/bin/activate         # macOS / Linux
pip install -r requirements-dev.txt
cp .env.example .env             # optional: add GROQ_API_KEY
```

### Train the model (once, ~1.5–3 min)

```bash
python main.py
```

This downloads FakeNewsNet (~44 MB, cached in `data/raw/fakenewsnet/`), trains LR and SVM, evaluates them, writes figures to `reports/figures/`, and saves `models/logistic_regression.pkl`, `models/tfidf_vectorizer.pkl` and `models/model_card.json` (dataset checksums, metrics, thresholds, scikit-learn version). Flags: `--no-plots`, `--no-cv`, `--models-dir DIR`. `--data synthetic` gives an offline smoke run; the resulting model is flagged as a **demo** in the UI.

### Run the web app

```bash
python app.py                    # development → http://localhost:5000
```

Production (Windows, macOS, Linux):

```bash
waitress-serve --listen=0.0.0.0:8000 --threads=8 --call app:create_app
```

If `waitress-serve` isn't on your PATH, `python -m waitress …` is equivalent.

### CLI

```bash
python verify.py                                   # interactive
python verify.py --text "BREAKING: ..."            # one headline
python verify.py --url https://www.bbc.com/news/... # fetch + analyse
python verify.py --text "..." --json --no-llm      # JSON output, ML only
```

### Tests & lint

```bash
pytest          # 166 tests, ~5 s, no network needed
ruff check .
```

---

## 🔌 HTTP API

| Method | Path | Body | Notes |
|---|---|---|---|
| `POST` | `/api/verify-text` | `{"text": "..."}` (5–5,000 chars) | First line is scored by the ML model; full text goes to the LLM |
| `POST` | `/api/verify-url` | `{"url": "https://..."}` | 422 with a readable `error` if the page can't be fetched |
| `GET` | `/api/model-info` | | Model name, dataset, held-out metrics, whether the LLM is enabled |
| `GET` | `/healthz` | | Liveness |
| `GET` | `/readyz` | | Readiness (model loaded) |

Response (abridged):

```json
{
  "verdict": "LIKELY_FAKE",                 // LIKELY_REAL | UNCERTAIN | LIKELY_FAKE | INSUFFICIENT_SIGNAL
  "verdict_basis": "llm+ml",                // or "ml"
  "agreement": "agree",                     // agree | partial | disagree | n/a
  "ml":  {"risk_score": 0.896, "risk_level": "HIGH", "coverage": 1.0,
          "explanation": {"toward_fake": [{"term": "pope francis", "weight": 0.71}], "toward_real": []}},
  "llm": {"status": "ok", "verdict": "FAKE", "confidence": 90, "credibility_score": 5,
          "summary": "...", "red_flags": [], "fact_check_suggestions": []},
  "notes": [],
  "headline": "...", "source": "...", "url": "...", "preview": "..."
}
```

Errors are JSON: `{"error": "...", "code": "text_too_short", "request_id": "..."}`. Every response carries an `X-Request-ID` header.

---

## 🔧 Configuration

Copy `.env.example` to `.env` and use `KEY=value` lines (no `set`/`export` prefix). All settings are optional.

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | *(empty)* | Enables the AI assessment. Without it the app runs ML-only. |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Any Groq chat model that supports JSON mode |
| `GROQ_TIMEOUT` / `GROQ_MAX_RETRIES` | `20` / `1` | LLM call limits |
| `RATE_LIMIT_PER_MINUTE` | `30` | Per client IP, per process (`0` disables) |
| `TRUST_PROXY` | `false` | Set `true` behind a reverse proxy so client IPs come from `X-Forwarded-For` |
| `MAX_TEXT_CHARS` | `5000` | Max text length |
| `SCRAPE_TIMEOUT` / `SCRAPE_MAX_BYTES` | `10` / `3000000` | URL fetch limits |
| `ALLOW_PRIVATE_URLS` | `false` | **Dev only.** Allows fetching localhost/private hosts. |
| `MODELS_DIR` | `models` | Where trained artefacts are loaded from |
| `PORT` / `HOST` / `LOG_LEVEL` | `5000` / `127.0.0.1` / `INFO` | Dev server and logging |

Invalid values (e.g. `RATE_LIMIT_PER_MINUTE=lots`) stop startup with a clear error.

---

## 🚀 Deployment

**Docker** (trains the model during the build, so the image is self-contained):

```bash
docker compose up --build        # → http://localhost:8000 (reads .env if present)
```

The image runs **waitress** as a non-root user, has a `HEALTHCHECK` on `/healthz`, and the compose file runs it with a read-only filesystem. The app is stateless (no database), so it scales horizontally. The rate limiter is per process, so put a shared limit at your reverse proxy if you run several replicas.

**Any VM or PaaS:** `pip install -r requirements.txt && python main.py --no-plots`, then run the waitress command above behind HTTPS (e.g. nginx or Caddy) with `TRUST_PROXY=true`. Supply `GROQ_API_KEY` through your platform's secret store.

**Rollback:** model artefacts are versioned by `model_card.json` (`trained_at`, dataset checksums). Keep the previous `models/` directory or image tag to roll back.

---

## 🔒 Security & privacy

- **SSRF protection:** URL fetching allows only http(s) on ports 80/443, without credentials. Every hostname, including each redirect hop, must resolve to public IPs, which blocks localhost, private ranges, cloud metadata, and IPv6 transition forms such as NAT64/6to4 that embed private IPv4. Validation runs on the URL exactly as `requests` will send it (punycode host, normalised escapes), so parser tricks like backslashes or IDN variants can't validate one host and connect to another. Responses are size-capped, must be HTML, and the whole fetch has a hard deadline. Residual risk is DNS rebinding between check and connect; restrict outbound traffic at the network level if this matters.
- **XSS:** the frontend builds all dynamic content with DOM APIs (`textContent`), never `innerHTML`, and the page ships a strict Content-Security-Policy with no inline scripts.
- **Prompt injection:** text sent to the LLM is wrapped as untrusted data. LLM output is schema-validated (verdict enum, clamped numbers, length-capped lists) before it reaches the browser.
- **Secrets** come only from the environment, are never logged or returned by the API, and `.env` is gitignored.
- **Privacy:** the server doesn't store or log submitted text or URLs. History lives in the user's browser (`localStorage`). With the LLM enabled, submitted text is sent to Groq.
- Model files are pickles, which can run code when loaded. Only load artefacts you trained yourself.

---

## ⚠️ Known limitations

- The ML model reflects its training data (English, 2016–2018, US politics and entertainment). It learns topics as well as style, and an article *about* fake news can score "fake-like".
- It classifies headlines. For URLs it scores the page headline, and the article body is used only by the LLM.
- Some sites block automated readers (HTTP 403) or need JavaScript. Paste the text instead.
- The LLM assessment depends on Groq availability and your key. It can't verify very recent events.
- The in-process rate limiter (per IP; per /64 for IPv6) isn't shared across processes or replicas.

## 📚 Dataset & citation

Shu, K., Mahudeswaran, D., Wang, S., Lee, D., Liu, H. *FakeNewsNet: A Data Repository with News Content, Social Context and Spatiotemporal Information for Studying Fake News on Social Media.* arXiv:1809.01286 (2018). The dataset isn't redistributed here; `main.py` downloads it from the authors' repository.

## 📄 License

MIT. See [LICENSE](LICENSE).

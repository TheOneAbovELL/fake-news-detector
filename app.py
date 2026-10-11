"""
app.py
------
Flask web application for TruthLens.

Development:
    python app.py                         → http://localhost:5000
Production (any OS):
    waitress-serve --listen=0.0.0.0:8000 --call app:create_app
"""

import hashlib
import ipaddress
import logging
import os
import re
import sys
import threading
import time
import uuid
from collections import defaultdict, deque

from dotenv import load_dotenv
from flask import Flask, g, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import __version__  # noqa: E402
from src.analysis import Analyzer  # noqa: E402
from src.config import Settings  # noqa: E402
from src.llm import GroqClient  # noqa: E402
from src.predict import ModelNotFoundError, Predictor  # noqa: E402
from src.scraper import scrape_article, validate_url  # noqa: E402

log = logging.getLogger("truthlens")

MIN_TEXT_CHARS = 5
MAX_URL_CHARS = 2048
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

CONTENT_SECURITY_POLICY = "; ".join([
    "default-src 'self'",
    "script-src 'self'",
    "style-src 'self' https://fonts.googleapis.com",
    "font-src 'self' https://fonts.gstatic.com",
    "img-src 'self' data:",
    "connect-src 'self'",
    "base-uri 'none'",
    "form-action 'self'",
    "frame-ancestors 'none'",
])


def _asset_version(static_dir: str) -> str:
    """Short content hash of the static files, for cache-busting URLs."""
    digest = hashlib.sha256()
    for name in sorted(os.listdir(static_dir)):
        path = os.path.join(static_dir, name)
        if os.path.isfile(path):
            with open(path, "rb") as fh:
                digest.update(fh.read())
    return digest.hexdigest()[:10]


class RateLimiter:
    """Sliding-window limiter, per client key, in process memory.

    Good enough for a single-process deployment; behind several workers or
    replicas each process enforces its own budget (use a shared store such as
    Redis, or the reverse proxy's limiter, for a strict global limit).
    """

    def __init__(self, limit: int, window: float = 60.0):
        self.limit = limit
        self.window = window
        self._hits = defaultdict(deque)
        self._lock = threading.Lock()
        self._last_sweep = time.monotonic()

    @staticmethod
    def client_key(remote_addr: str) -> str:
        """Group IPv6 clients by /64 — one host can rotate through a whole /64."""
        try:
            ip = ipaddress.ip_address(remote_addr)
        except ValueError:
            return remote_addr or "unknown"
        if ip.version == 6:
            if ip.ipv4_mapped:
                return str(ip.ipv4_mapped)
            return str(ipaddress.ip_network(f"{ip}/64", strict=False))
        return str(ip)

    def _sweep(self, now: float):
        """Forget clients with no hits inside the window (bounds memory)."""
        for k in [k for k, q in self._hits.items() if not q or now - q[-1] >= self.window]:
            del self._hits[k]
        self._last_sweep = now

    def hit(self, key: str) -> float:
        """Record a request; return 0 if allowed, else seconds until retry."""
        if self.limit <= 0:
            return 0.0
        now = time.monotonic()
        with self._lock:
            if now - self._last_sweep >= self.window:
                self._sweep(now)
            q = self._hits[key]
            while q and now - q[0] >= self.window:
                q.popleft()
            if len(q) >= self.limit:
                return max(0.1, self.window - (now - q[0]))
            q.append(now)
            return 0.0


class APIError(Exception):
    def __init__(self, message: str, status: int = 400, code: str = "bad_request"):
        super().__init__(message)
        self.message, self.status, self.code = message, status, code


def _json_body() -> dict:
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise APIError("Request body must be a JSON object.", 400, "invalid_json")
    return data


def _string_field(data: dict, name: str) -> str:
    value = data.get(name, "")
    if not isinstance(value, str):
        raise APIError(f"'{name}' must be a string.", 400, "invalid_field")
    return value.strip()


def create_app(settings: Settings = None, predictor: Predictor = None,
               llm_client: GroqClient = None) -> Flask:
    """Application factory. Fails fast if the trained model is missing."""
    if settings is None:
        load_dotenv()
        settings = Settings.from_env()
    if not logging.getLogger().handlers:
        logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper(),
                            format="%(asctime)s %(levelname)s %(name)s %(message)s")

    if predictor is None:
        predictor = Predictor.load(settings.models_dir)
    if llm_client is None:
        llm_client = GroqClient(settings.groq_api_key, settings.groq_model,
                                settings.groq_endpoint, settings.groq_timeout,
                                settings.groq_max_retries)
    analyzer = Analyzer(predictor, llm_client)
    limiter = RateLimiter(settings.rate_limit_per_minute)

    app = Flask(__name__)
    app.config.update(MAX_CONTENT_LENGTH=64 * 1024)
    app.json.sort_keys = False  # keep the documented field order in API responses
    asset_version = _asset_version(app.static_folder)
    app.extensions["truthlens"] = {"settings": settings, "predictor": predictor,
                                   "analyzer": analyzer, "limiter": limiter}

    if settings.trust_proxy:
        from werkzeug.middleware.proxy_fix import ProxyFix
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

    log.info("Model loaded: %s (demo=%s) | LLM layer: %s",
             predictor.card.get("model_name", "unknown"), predictor.card.get("is_demo", False),
             f"enabled ({settings.groq_model})" if llm_client.enabled else "disabled")

    # ── Request lifecycle ────────────────────────────────────────────────────

    @app.before_request
    def _start():
        g.started = time.monotonic()
        incoming = request.headers.get("X-Request-ID", "")
        g.request_id = incoming if _REQUEST_ID_RE.match(incoming) else uuid.uuid4().hex[:16]

    @app.after_request
    def _finish(resp):
        resp.headers["X-Request-ID"] = g.get("request_id", "-")
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        resp.headers["Content-Security-Policy"] = CONTENT_SECURITY_POLICY
        if request.path.startswith("/api/"):
            resp.headers["Cache-Control"] = "no-store"
        # Never log submitted text/URLs — they may be private.
        log.info("request id=%s method=%s path=%s status=%s ms=%d",
                 g.get("request_id", "-"), request.method, request.path, resp.status_code,
                 int((time.monotonic() - g.get("started", time.monotonic())) * 1000))
        return resp

    def _rate_limit():
        retry = limiter.hit(RateLimiter.client_key(request.remote_addr or ""))
        if retry:
            err = APIError("Too many requests — please wait a moment and try again.",
                           429, "rate_limited")
            err.retry_after = int(retry) + 1
            raise err

    # ── Errors (JSON for the API) ────────────────────────────────────────────

    @app.errorhandler(APIError)
    def _api_error(e: APIError):
        resp = jsonify({"error": e.message, "code": e.code, "request_id": g.get("request_id")})
        resp.status_code = e.status
        if getattr(e, "retry_after", None):
            resp.headers["Retry-After"] = str(e.retry_after)
        return resp

    @app.errorhandler(HTTPException)
    def _http_error(e: HTTPException):
        if not request.path.startswith("/api/"):
            return e
        messages = {413: "Request is too large.", 404: "Not found.",
                    405: "Method not allowed."}
        resp = jsonify({"error": messages.get(e.code, e.description), "code": e.name.lower()
                        .replace(" ", "_"), "request_id": g.get("request_id")})
        resp.status_code = e.code
        return resp

    @app.errorhandler(Exception)
    def _unhandled(e: Exception):
        log.exception("Unhandled error id=%s", g.get("request_id", "-"))
        return jsonify({"error": "Internal server error.", "code": "internal_error",
                        "request_id": g.get("request_id")}), 500

    # ── Routes ───────────────────────────────────────────────────────────────

    @app.get("/")
    def index():
        return render_template("index.html", version=__version__, asset_version=asset_version)

    @app.get("/healthz")
    def healthz():
        return jsonify({"status": "ok"})

    @app.get("/readyz")
    def readyz():
        return jsonify({"status": "ready", "model_loaded": True,
                        "llm_enabled": llm_client.enabled})

    @app.get("/api/model-info")
    def model_info():
        return jsonify({**predictor.info(), "llm_enabled": llm_client.enabled,
                        "llm_model": settings.groq_model if llm_client.enabled else None,
                        "version": __version__})

    @app.post("/api/verify-text")
    def verify_text():
        _rate_limit()
        text = _string_field(_json_body(), "text")
        if len(text) < MIN_TEXT_CHARS:
            raise APIError("Please enter a headline or text to analyse (at least 5 characters).",
                           400, "text_too_short")
        if len(text) > settings.max_text_chars:
            raise APIError(f"Text is too long — please keep it under "
                           f"{settings.max_text_chars:,} characters.", 400, "text_too_long")
        # The classifier was trained on headlines: score the first line/sentence block,
        # give the LLM the whole text.
        first_line = text.splitlines()[0].strip() or text
        result = analyzer.analyse(ml_text=first_line[:500], llm_text=text, mode="text",
                                  meta={"headline": first_line[:160]})
        return jsonify(result)

    @app.post("/api/verify-url")
    def verify_url():
        _rate_limit()
        url = _string_field(_json_body(), "url")
        if not url:
            raise APIError("Please paste a news article URL.", 400, "url_missing")
        if len(url) > MAX_URL_CHARS or not validate_url(url):
            raise APIError("That doesn't look like a valid web address "
                           "(e.g. https://www.bbc.com/news/...).", 400, "invalid_url")

        article = scrape_article(url, timeout=settings.scrape_timeout,
                                 max_bytes=settings.scrape_max_bytes,
                                 allow_private=settings.allow_private_urls)
        if not article["success"]:
            raise APIError(article["error"] or "Could not extract article content.",
                           422, "fetch_failed")

        headline = article["headline"] or article["body"][:200]
        llm_text = f"{article['headline']}\n\n{article['body']}".strip()
        result = analyzer.analyse(
            ml_text=headline, llm_text=llm_text, mode="url",
            meta={"headline": article["headline"], "source": article["source"],
                  "url": article["url"], "preview": article["preview"]})
        return jsonify(result)

    return app


if __name__ == "__main__":
    load_dotenv()
    try:
        application = create_app()
    except ModelNotFoundError as e:
        print(f"\n❌  {e}\n")
        sys.exit(1)
    port = int(os.environ.get("PORT", "5000"))
    print(f"\n🚀  TruthLens running → http://localhost:{port}\n")
    application.run(debug=False, host=os.environ.get("HOST", "127.0.0.1"), port=port)

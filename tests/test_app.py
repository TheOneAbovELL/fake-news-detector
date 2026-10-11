"""HTTP API tests via Flask's test client (stub LLM, stubbed scraper — no network)."""

import pytest

import app as app_module
from src.config import Settings
from src.llm import GroqClient

FAKE_TEXT = "BREAKING: Deep state operative REVEALS plot to poison the water"


@pytest.fixture
def make_client(predictor, models_dir):
    def _make(**overrides):
        settings = Settings(models_dir=models_dir, groq_api_key="", **overrides)
        flask_app = app_module.create_app(settings=settings, predictor=predictor,
                                          llm_client=GroqClient("", "m", "https://llm.invalid"))
        flask_app.testing = True
        return flask_app.test_client()
    return _make


@pytest.fixture
def client(make_client):
    return make_client()


class TestPagesAndHealth:
    def test_index_renders_with_security_headers(self, client):
        r = client.get("/")
        assert r.status_code == 200 and b"TruthLens" in r.data
        csp = r.headers["Content-Security-Policy"]
        assert "script-src 'self'" in csp and "frame-ancestors 'none'" in csp
        assert r.headers["X-Content-Type-Options"] == "nosniff"
        assert r.headers["X-Frame-Options"] == "DENY"

    def test_no_inline_scripts_or_handlers(self, client):
        html = client.get("/").get_data(as_text=True)
        assert "onclick=" not in html and "<script>" not in html

    def test_static_assets(self, client):
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/static/app.css").status_code == 200

    def test_health_and_ready(self, client):
        assert client.get("/healthz").get_json() == {"status": "ok"}
        ready = client.get("/readyz").get_json()
        assert ready["status"] == "ready" and ready["llm_enabled"] is False

    def test_model_info_exposes_no_secrets(self, client):
        info = client.get("/api/model-info").get_json()
        assert info["model_type"] == "logistic_regression" and info["llm_enabled"] is False
        assert "key" not in str(info).lower()

    def test_request_id_echoed_or_generated(self, client):
        echoed = client.get("/healthz", headers={"X-Request-ID": "abc-123"})
        assert echoed.headers["X-Request-ID"] == "abc-123"
        generated = client.get("/healthz", headers={"X-Request-ID": "bad id <script>"})
        generated = generated.headers["X-Request-ID"]
        assert generated != "bad id <script>" and len(generated) == 16


class TestVerifyText:
    def test_success_shape(self, client):
        r = client.post("/api/verify-text", json={"text": FAKE_TEXT})
        assert r.status_code == 200 and r.headers["Cache-Control"] == "no-store"
        d = r.get_json()
        assert d["verdict"] == "LIKELY_FAKE" and d["verdict_basis"] == "ml"
        assert d["llm"]["status"] == "disabled"
        assert set(d["ml"]) >= {"verdict", "risk_score", "risk_level", "coverage", "explanation"}
        assert d["is_fake"] is True and d["prediction"] == "FAKE"  # legacy fields

    def test_multiline_scores_first_line(self, client):
        d = client.post("/api/verify-text", json={"text": FAKE_TEXT + "\nsecond line"}).get_json()
        assert d["headline"] == FAKE_TEXT

    @pytest.mark.parametrize("payload,code", [
        ({}, "text_too_short"),
        ({"text": "   "}, "text_too_short"),
        ({"text": "abc"}, "text_too_short"),
        ({"text": 123}, "invalid_field"),
        ({"text": "x" * 5001}, "text_too_long"),
    ])
    def test_validation(self, client, payload, code):
        r = client.post("/api/verify-text", json=payload)
        assert r.status_code == 400 and r.get_json()["code"] == code

    def test_non_json_body(self, client):
        r = client.post("/api/verify-text", data="text=hello",
                        content_type="application/x-www-form-urlencoded")
        assert r.status_code == 400 and r.get_json()["code"] == "invalid_json"

    def test_json_array_body(self, client):
        assert client.post("/api/verify-text", json=["x"]).status_code == 400

    def test_body_too_large(self, client):
        r = client.post("/api/verify-text", data="x" * 70_000, content_type="application/json")
        assert r.status_code == 413 and "error" in r.get_json()

    def test_wrong_method_is_json(self, client):
        r = client.get("/api/verify-text")
        assert r.status_code == 405 and r.get_json()["error"]

    def test_unknown_api_route_is_json(self, client):
        assert client.get("/api/nope").get_json()["code"] == "not_found"


class TestVerifyUrl:
    def test_invalid_url(self, client):
        r = client.post("/api/verify-url", json={"url": "not a url"})
        assert r.status_code == 400 and r.get_json()["code"] == "invalid_url"

    def test_missing_url(self, client):
        assert client.post("/api/verify-url", json={}).get_json()["code"] == "url_missing"

    def test_fetch_failure_is_422(self, client, monkeypatch):
        failure = {"success": False, "error": "That address is not publicly reachable."}
        monkeypatch.setattr(app_module, "scrape_article", lambda url, **kw: failure)
        r = client.post("/api/verify-url", json={"url": "http://internal.example.com"})
        assert r.status_code == 422 and "publicly" in r.get_json()["error"]

    def test_success_uses_headline_for_ml(self, client, monkeypatch):
        seen = {}

        def fake_scrape(url, **kw):
            seen.update(kw)
            return {"success": True, "error": None, "url": url, "source": "news.example.com",
                    "headline": FAKE_TEXT, "body": "Long article body " * 20, "preview": "Long article…"}

        monkeypatch.setattr(app_module, "scrape_article", fake_scrape)
        d = client.post("/api/verify-url", json={"url": "https://news.example.com/a"}).get_json()
        assert d["verdict"] == "LIKELY_FAKE" and d["source"] == "news.example.com"
        assert d["mode"] == "url" and d["headline"] == FAKE_TEXT
        assert seen["allow_private"] is False and seen["max_bytes"] > 0


class TestRateLimit:
    def test_limit_enforced_with_retry_after(self, make_client):
        c = make_client(rate_limit_per_minute=2)
        for _ in range(2):
            assert c.post("/api/verify-text", json={"text": FAKE_TEXT}).status_code == 200
        r = c.post("/api/verify-text", json={"text": FAKE_TEXT})
        assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1
        assert r.get_json()["code"] == "rate_limited"
        assert c.get("/healthz").status_code == 200  # health checks are never limited

    def test_zero_disables_limit(self, make_client):
        c = make_client(rate_limit_per_minute=0)
        for _ in range(5):
            assert c.post("/api/verify-text", json={"text": FAKE_TEXT}).status_code == 200


class TestRateLimiterUnit:
    def test_window_expires(self, monkeypatch):
        limiter = app_module.RateLimiter(limit=1, window=60)
        clock = [1000.0]
        monkeypatch.setattr(app_module.time, "monotonic", lambda: clock[0])
        assert limiter.hit("a") == 0 and limiter.hit("a") > 0
        assert limiter.hit("b") == 0  # per-client
        clock[0] += 61
        assert limiter.hit("a") == 0


def test_create_app_fails_fast_without_model(tmp_path):
    from src.predict import ModelNotFoundError
    with pytest.raises(ModelNotFoundError):
        app_module.create_app(settings=Settings(models_dir=tmp_path))


class TestRateLimiterMemory:
    def test_idle_clients_are_forgotten(self, monkeypatch):
        clock = [1000.0]
        monkeypatch.setattr(app_module.time, "monotonic", lambda: clock[0])
        limiter = app_module.RateLimiter(limit=5, window=60)
        for i in range(5000):
            limiter.hit(f"10.0.{i // 256}.{i % 256}")
        clock[0] += 61
        limiter.hit("fresh-client")
        assert len(limiter._hits) == 1

    @pytest.mark.parametrize("addr,key", [
        ("2001:db8:1:2:aaaa::1", "2001:db8:1:2::/64"),
        ("2001:db8:1:2:ffff::9", "2001:db8:1:2::/64"),
        ("::ffff:203.0.113.5", "203.0.113.5"),
        ("203.0.113.5", "203.0.113.5"),
        ("", "unknown"),
    ])
    def test_client_key(self, addr, key):
        assert app_module.RateLimiter.client_key(addr) == key

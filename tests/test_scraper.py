"""Scraper tests: SSRF protection, redirects, limits and HTML extraction (no network)."""

import pytest
import requests

from src.scraper import UnsafeURLError, check_url, scrape_article, validate_url


def resolver_for(mapping):
    def resolve(host):
        if host not in mapping:
            raise OSError("no such host")
        return mapping[host]
    return resolve


PUBLIC = resolver_for({
    "news.example.com": ["93.184.216.34"],
    "other.example.com": ["93.184.216.35"],
    "evil.example.com": ["127.0.0.1"],
    "mixed.example.com": ["93.184.216.34", "10.0.0.5"],
    "v6local.example.com": ["::1"],
    "mapped.example.com": ["::ffff:127.0.0.1"],
    "metadata.example.com": ["169.254.169.254"],
    "cgnat.example.com": ["100.64.0.1"],
})

ARTICLE_HTML = """
<html><head><title>Fallback Title | Site</title>
<meta property="og:title" content="Council approves new library budget">
<script>var tracking = "should not appear";</script></head>
<body><nav>Home News Sport</nav>
<article>
  <p>The city council voted on Tuesday to approve a budget for a new public library in the town centre.</p>
  <p>Officials said construction would begin next spring and take around eighteen months to complete.</p>
  <p>Short.</p>
</article><footer>Copyright footer text that is long enough to be a paragraph but is excluded.</footer>
</body></html>
"""


class TestCheckUrl:
    def test_public_https_ok(self):
        assert check_url("https://news.example.com/a", resolver=PUBLIC) == "https://news.example.com/a"

    def test_scheme_defaults_to_https(self):
        assert check_url("news.example.com/a", resolver=PUBLIC).startswith("https://")

    @pytest.mark.parametrize("url", [
        "http://evil.example.com/", "http://mixed.example.com/", "http://v6local.example.com/",
        "http://mapped.example.com/", "http://metadata.example.com/latest/meta-data/",
        "http://cgnat.example.com/",
    ])
    def test_private_addresses_blocked(self, url):
        with pytest.raises(UnsafeURLError, match="not publicly reachable"):
            check_url(url, resolver=PUBLIC)

    @pytest.mark.parametrize("url", ["http://127.0.0.1/", "http://[::1]/", "http://10.1.2.3/"])
    def test_literal_private_ips_blocked(self, url):
        with pytest.raises(UnsafeURLError):
            check_url(url)

    @pytest.mark.parametrize("url", ["ftp://news.example.com/x", "file:///etc/passwd",
                                     "javascript:alert(1)"])
    def test_non_http_schemes_blocked(self, url):
        with pytest.raises(UnsafeURLError):
            check_url(url, resolver=PUBLIC)

    def test_credentials_blocked(self):
        with pytest.raises(UnsafeURLError, match="credentials"):
            check_url("https://user:pw@news.example.com/", resolver=PUBLIC)

    def test_non_standard_port_blocked(self):
        with pytest.raises(UnsafeURLError, match="ports"):
            check_url("https://news.example.com:8443/", resolver=PUBLIC)

    def test_unresolvable_host(self):
        with pytest.raises(UnsafeURLError, match="resolve"):
            check_url("https://nope.example.com/", resolver=PUBLIC)

    def test_allow_private_for_local_dev(self):
        assert check_url("http://localhost:5000/x", allow_private=True) == "http://localhost:5000/x"

    def test_validate_url_syntax(self):
        assert validate_url("https://bbc.com/news")
        assert validate_url("bbc.com/news")
        assert not validate_url("not a url")
        assert not validate_url("ftp://bbc.com")
        assert not validate_url("")


class TestScrapeArticle:
    def test_extracts_headline_and_body(self, fake_session, fake_response):
        html = {"Content-Type": "text/html; charset=utf-8"}
        session = fake_session([fake_response(200, ARTICLE_HTML, html)])
        r = scrape_article("https://news.example.com/story", session=session, resolver=PUBLIC)
        assert r["success"] and r["error"] is None
        assert r["headline"] == "Council approves new library budget"
        assert "approve a budget" in r["body"]
        assert "tracking" not in r["body"] and "Copyright" not in r["body"]
        assert "Short." not in r["body"]
        assert r["source"] == "news.example.com"
        assert session.calls[0][2]["allow_redirects"] is False

    def test_redirect_to_private_address_blocked(self, fake_session, fake_response):
        session = fake_session([fake_response(302, b"", {"Location": "http://evil.example.com/admin"})])
        r = scrape_article("https://news.example.com/r", session=session, resolver=PUBLIC)
        assert not r["success"]
        assert "not publicly reachable" in r["error"]
        assert len(session.calls) == 1  # never connected to the private host

    def test_follows_safe_relative_redirect(self, fake_session, fake_response):
        session = fake_session([
            fake_response(301, b"", {"Location": "/final"}),
            fake_response(200, ARTICLE_HTML, {"Content-Type": "text/html"}),
        ])
        r = scrape_article("https://news.example.com/start", session=session, resolver=PUBLIC)
        assert r["success"]
        assert r["url"] == "https://news.example.com/final"

    def test_too_many_redirects(self, fake_session, fake_response):
        session = fake_session([fake_response(302, b"", {"Location": "/loop"}) for _ in range(10)])
        r = scrape_article("https://news.example.com/loop", session=session, resolver=PUBLIC)
        assert "redirects" in r["error"]

    def test_non_html_rejected(self, fake_session, fake_response):
        session = fake_session([fake_response(200, b"%PDF", {"Content-Type": "application/pdf"})])
        r = scrape_article("https://news.example.com/doc.pdf", session=session, resolver=PUBLIC)
        assert "isn't a web page" in r["error"]

    def test_oversized_body_rejected(self, fake_session, fake_response):
        session = fake_session([fake_response(200, b"<p>" + b"x" * 20_000, {"Content-Type": "text/html"})])
        r = scrape_article("https://news.example.com/big", session=session, resolver=PUBLIC,
                           max_bytes=10_000)
        assert "too large" in r["error"]

    def test_declared_oversize_rejected(self, fake_session, fake_response):
        session = fake_session([fake_response(200, b"<p>hi</p>",
                                              {"Content-Type": "text/html", "Content-Length": "999999999"})])
        r = scrape_article("https://news.example.com/big", session=session, resolver=PUBLIC)
        assert "too large" in r["error"]

    def test_forbidden_gives_actionable_message(self, fake_session, fake_response):
        session = fake_session([fake_response(403, b"", {"Content-Type": "text/html"})])
        r = scrape_article("https://news.example.com/x", session=session, resolver=PUBLIC)
        assert "HTTP 403" in r["error"] and "Paste" in r["error"]

    def test_timeout(self, fake_session):
        session = fake_session([requests.exceptions.Timeout()])
        r = scrape_article("https://news.example.com/x", session=session, resolver=PUBLIC, timeout=3)
        assert "too long" in r["error"]

    def test_connection_error(self, fake_session):
        session = fake_session([requests.exceptions.ConnectionError()])
        r = scrape_article("https://news.example.com/x", session=session, resolver=PUBLIC)
        assert "connect" in r["error"].lower()

    def test_empty_page(self, fake_session, fake_response):
        empty = fake_response(200, "<html><body></body></html>", {"Content-Type": "text/html"})
        session = fake_session([empty])
        r = scrape_article("https://news.example.com/x", session=session, resolver=PUBLIC)
        assert not r["success"] and "JavaScript" in r["error"]


class TestParserDifferentialHardening:
    """Regression tests: validate exactly the host that requests will connect to."""

    def test_backslash_host_confusion_blocked(self):
        # urlparse sees host "169.254.169.254\.x.attacker.com"; requests connects to 169.254.169.254
        url = r"http://169.254.169.254\.x.attacker.com/latest/meta-data/"
        with pytest.raises(UnsafeURLError):
            check_url(url, resolver=lambda h: ["93.184.216.34"])

    def test_idn_checked_as_the_punycode_host_that_is_fetched(self):
        seen = []

        def resolver(host):
            seen.append(host)
            return ["93.184.216.34"]

        prepared = check_url("http://faß.de/", resolver=resolver)
        assert seen == ["xn--fa-hia.de"]            # UTS46, as requests encodes it
        assert prepared == "http://xn--fa-hia.de/"  # and this exact URL is what gets fetched

    def test_fetch_uses_validated_url(self, fake_session, fake_response):
        session = fake_session([fake_response(200, ARTICLE_HTML, {"Content-Type": "text/html"})])
        scrape_article("http://faß.de/a b", session=session,
                       resolver=lambda h: ["93.184.216.34"])
        assert session.calls[0][1] == "http://xn--fa-hia.de/a%20b"

    @pytest.mark.parametrize("ip", ["64:ff9b::a9fe:a9fe", "64:ff9b::7f00:1", "::7f00:1",
                                    "2002:7f00:1::1", "2001:0:4136:e378:8000:63bf:3fff:fdd2"])
    def test_ipv6_transition_addresses_to_private_v4_blocked(self, ip):
        with pytest.raises(UnsafeURLError):
            check_url("http://sneaky.example.com/", resolver=lambda h: [ip])

    def test_nat64_of_public_v4_allowed(self):
        check_url("http://ok.example.com/", resolver=lambda h: ["64:ff9b::5db8:d822"])

    def test_total_deadline_stops_trickling_server(self, fake_session, fake_response, monkeypatch):
        from src import scraper
        clock = [0.0]
        monkeypatch.setattr(scraper.time, "monotonic", lambda: clock[0])

        class Trickle(fake_response):
            def iter_content(self, chunk_size=65536):
                for _ in range(1000):
                    clock[0] += 5          # each chunk arrives just under the socket timeout
                    yield b"<p>x</p>"

        session = fake_session([Trickle(200, b"", {"Content-Type": "text/html"})])
        r = scrape_article("https://news.example.com/slow", session=session, resolver=PUBLIC, timeout=10)
        assert not r["success"] and "too long" in r["error"]
        assert clock[0] <= 25

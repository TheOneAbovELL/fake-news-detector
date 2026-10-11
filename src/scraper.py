"""
scraper.py
----------
Extracts headline and article text from a public news URL.

The fetcher is deliberately defensive because the URL comes from an untrusted
user and is fetched *by the server*:

* only http/https on the default ports, no embedded credentials
* every hostname is resolved and must map exclusively to public IP addresses
  (blocks localhost, private networks, link-local / cloud metadata endpoints)
* redirects are followed manually and each hop is re-validated
* response size is capped and only HTML content is parsed

Residual risk: DNS can change between validation and connection (rebinding).
Run the service with outbound network restrictions if that matters for you.

Usage:
    from src.scraper import scrape_article
    result = scrape_article("https://www.bbc.com/news/article-123")
"""

import ipaddress
import re
import socket
import time
from typing import Callable, Dict, Iterable, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

SCRAPER_AVAILABLE = True  # kept for backwards compatibility with older callers

USER_AGENT = (
    "Mozilla/5.0 (compatible; TruthLens/2.0; "
    "+https://github.com/OmjeeRGiri/fake-news-detector)"
)
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.5",
    "Accept-Language": "en-US,en;q=0.9",
}

MAX_REDIRECTS = 5
ALLOWED_SCHEMES = {"http", "https"}
DEFAULT_PORTS = {"http": 80, "https": 443}

_HOSTNAME_RE = re.compile(r"^[a-z0-9.-]+$")
_NAT64_PREFIX = ipaddress.ip_network("64:ff9b::/96")
_IPV4_COMPAT = ipaddress.ip_network("::/96")

CONTENT_SELECTORS = [
    "article", '[role="main"]', ".article-body",
    ".story-body", ".post-content", ".entry-content",
    ".article__body", ".article-content", "main",
]

Resolver = Callable[[str], Iterable[str]]


class UnsafeURLError(ValueError):
    """The URL is malformed or points somewhere the server must not fetch."""


def _system_resolver(host: str) -> Iterable[str]:
    infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    return {info[4][0] for info in infos}


def _is_public_ip(raw: str) -> bool:
    ip = ipaddress.ip_address(raw.split("%", 1)[0])
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        elif ip in _NAT64_PREFIX:            # NAT64 translates to the embedded IPv4
            ip = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
        elif ip.sixtofour:                   # 6to4 relays to the embedded IPv4
            ip = ip.sixtofour
        elif ip in _IPV4_COMPAT or ip.teredo:
            return False
    return ip.is_global and not ip.is_multicast


def _prepare(url: str) -> str:
    """The exact URL `requests` will send (IDNA/UTS46 host, percent-encoding)."""
    import requests
    return requests.Request("GET", url).prepare().url


def normalize_url(url: str) -> str:
    """Strip whitespace and default to https:// when no scheme is given."""
    url = (url or "").strip()
    if url and "://" not in url:
        url = "https://" + url
    return url


def check_url(url: str, allow_private: bool = False,
              resolver: Optional[Resolver] = None) -> str:
    """
    Validate that ``url`` is safe for the server to fetch.

    Validation runs on the URL exactly as `requests` will send it (so parser
    differences — backslashes, IDN encodings — cannot make us check one host
    and connect to another). Returns that prepared URL; fetch *it*, not the
    input. Raises UnsafeURLError with a user-facing reason.
    """
    invalid = "That doesn't look like a valid URL."
    url = normalize_url(url)
    try:
        prepared = _prepare(url)
        if _prepare(prepared) != prepared:  # must be a fixed point
            raise UnsafeURLError(invalid)
        parsed = urlparse(prepared)
        port = parsed.port
    except UnsafeURLError:
        raise
    except Exception:  # requests raises several types for malformed URLs
        raise UnsafeURLError(invalid) from None

    if parsed.scheme not in ALLOWED_SCHEMES:
        raise UnsafeURLError("Only http:// and https:// links are supported.")
    host = (parsed.hostname or "").rstrip(".")
    if not host:
        raise UnsafeURLError(invalid)
    if parsed.username or parsed.password:
        raise UnsafeURLError("Links containing credentials are not allowed.")
    try:
        literal_ip = ipaddress.ip_address(host)
    except ValueError:
        literal_ip = None
        if not _HOSTNAME_RE.match(host) or ("." not in host and not allow_private):
            raise UnsafeURLError(invalid) from None
    if allow_private:
        return prepared
    if port is not None and port != DEFAULT_PORTS[parsed.scheme]:
        raise UnsafeURLError("Only standard web ports (80/443) are allowed.")

    if literal_ip is not None:
        addresses = {str(literal_ip)}
    else:
        try:
            addresses = set((resolver or _system_resolver)(host))
        except (socket.gaierror, UnicodeError, OSError):
            raise UnsafeURLError(f"Could not resolve the host '{host}'.") from None
    if not addresses:
        raise UnsafeURLError(f"Could not resolve the host '{host}'.")
    try:
        if not all(_is_public_ip(a) for a in addresses):
            raise UnsafeURLError("That address is not publicly reachable.")
    except ValueError:
        raise UnsafeURLError("That address is not publicly reachable.") from None
    return prepared


def validate_url(url: str) -> bool:
    """Cheap syntactic check (no DNS) used for early input validation."""
    try:
        parsed = urlparse(normalize_url(url))
        return parsed.scheme in ALLOWED_SCHEMES and bool(parsed.hostname) \
            and "." in parsed.hostname
    except ValueError:
        return False


def _extract_headline(soup) -> str:
    og = soup.find("meta", property="og:title")
    if og and og.get("content", "").strip():
        return og["content"].strip()
    tw = soup.find("meta", attrs={"name": "twitter:title"})
    if tw and tw.get("content", "").strip():
        return tw["content"].strip()
    schema = soup.find(itemprop="headline")
    if schema:
        text = schema.get_text(strip=True) or schema.get("content", "")
        if text:
            return text.strip()
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)
    title = soup.find("title")
    if title:
        text = title.get_text(strip=True)
        for sep in [" | ", " - ", " – ", " — "]:
            if sep in text:
                text = text.split(sep)[0].strip()
        return text
    return ""


def _paragraph_text(container) -> str:
    parts = (p.get_text(" ", strip=True) for p in container.find_all("p"))
    return " ".join(t for t in parts if len(t) > 40)


def _extract_body(soup) -> str:
    for tag in soup(["script", "style", "nav", "header", "footer",
                     "aside", "figure", "figcaption", "iframe", "noscript", "form"]):
        tag.decompose()
    for selector in CONTENT_SELECTORS:
        container = soup.select_one(selector)
        if container:
            text = _paragraph_text(container)
            if len(text) > 100:
                return text
    return _paragraph_text(soup)


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _fetch_html(url: str, timeout: float, max_bytes: int, allow_private: bool,
                session: requests.Session, resolver: Optional[Resolver]):
    """Fetch ``url`` following validated redirects; return (final_url, bytes).

    ``timeout`` bounds each socket operation; the whole fetch (all hops and the
    body download) must also finish within 2 × timeout, so a server that
    trickles bytes cannot hold a worker thread indefinitely.
    """
    deadline = time.monotonic() + 2 * timeout

    def remaining() -> float:
        left = deadline - time.monotonic()
        if left <= 0:
            raise requests.exceptions.Timeout()
        return left

    current = url
    for _ in range(MAX_REDIRECTS + 1):
        current = check_url(current, allow_private=allow_private, resolver=resolver)
        with session.get(current, headers=HEADERS, timeout=min(timeout, remaining()),
                         allow_redirects=False, stream=True) as resp:
            if resp.is_redirect or resp.status_code in (301, 302, 303, 307, 308):
                location = resp.headers.get("Location")
                if not location:
                    raise requests.exceptions.HTTPError(response=resp)
                current = urljoin(current, location)
                continue
            resp.raise_for_status()
            ctype = resp.headers.get("Content-Type", "").lower()
            if ctype and "html" not in ctype:
                raise UnsafeURLError(
                    "That link isn't a web page (content type "
                    f"'{ctype.split(';')[0]}'). Paste the article text instead.")
            declared = resp.headers.get("Content-Length")
            if declared and declared.isdigit() and int(declared) > max_bytes:
                raise UnsafeURLError("That page is too large to analyse.")
            chunks, total = [], 0
            for chunk in resp.iter_content(chunk_size=65536):
                remaining()
                total += len(chunk)
                if total > max_bytes:
                    raise UnsafeURLError("That page is too large to analyse.")
                chunks.append(chunk)
            return current, b"".join(chunks)
    raise UnsafeURLError("Too many redirects.")


def scrape_article(url: str, timeout: float = 10.0, max_bytes: int = 3_000_000,
                   allow_private: bool = False,
                   session: Optional[requests.Session] = None,
                   resolver: Optional[Resolver] = None) -> Dict:
    """
    Fetch a URL and extract its headline and body text.

    Returns a dict with keys: url, headline, body (full extracted text, capped),
    preview (short excerpt), source (domain), success, error.
    """
    url = normalize_url(url)
    result = {
        "url": url,
        "headline": "",
        "body": "",
        "preview": "",
        "source": (urlparse(url).hostname or "").removeprefix("www."),
        "success": False,
        "error": None,
    }
    own_session = session is None
    session = session or requests.Session()
    try:
        final_url, content = _fetch_html(url, timeout, max_bytes, allow_private,
                                         session, resolver)
        soup = BeautifulSoup(content, "html.parser")
        headline = _squash(_extract_headline(soup))[:300]
        body = _squash(_extract_body(soup))[:6000]
        result.update({
            "url": final_url,
            "source": (urlparse(final_url).hostname or "").removeprefix("www."),
            "headline": headline,
            "body": body,
            "preview": body[:280] + ("…" if len(body) > 280 else ""),
            "success": bool(headline or body),
        })
        if not result["success"]:
            result["error"] = ("Could not find article text on that page. "
                               "The site may require JavaScript or block automated readers.")
    except UnsafeURLError as e:
        result["error"] = str(e)
    except requests.exceptions.Timeout:
        result["error"] = "The site took too long to respond."
    except requests.exceptions.ConnectionError:
        result["error"] = "Could not connect to that site. Check the URL."
    except requests.exceptions.HTTPError as e:
        code = getattr(e.response, "status_code", None)
        if code in (401, 402, 403, 451):
            result["error"] = (f"The site refused access (HTTP {code}) — it may block "
                               "automated readers or be paywalled. Paste the headline instead.")
        else:
            result["error"] = f"The site returned an error (HTTP {code})."
    except requests.exceptions.RequestException:
        result["error"] = "Could not fetch that page."
    finally:
        if own_session:
            session.close()
    return result

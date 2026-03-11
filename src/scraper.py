"""
scraper.py
----------
Extracts headline and article text from any news URL.
Uses only stdlib + requests + BeautifulSoup (no heavy dependencies).

Usage:
    from src.scraper import scrape_article
    result = scrape_article("https://www.bbc.com/news/article-123")
"""

import re
import sys
from typing import Dict, Optional
from urllib.parse import urlparse

try:
    import requests
    from bs4 import BeautifulSoup
    SCRAPER_AVAILABLE = True
except ImportError:
    SCRAPER_AVAILABLE = False


HEADLINE_SELECTORS = [
    "h1",
    'meta[property="og:title"]',
    'meta[name="twitter:title"]',
    'meta[name="title"]',
    '[itemprop="headline"]',
    ".headline", ".article-headline", ".entry-title",
    ".post-title", ".story-headline", ".article__title",
]

CONTENT_SELECTORS = [
    "article", '[role="main"]', ".article-body",
    ".story-body", ".post-content", ".entry-content",
    ".article__body", ".article-content", "main",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def _extract_headline(soup):
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
    if h1:
        return h1.get_text(strip=True)
    title = soup.find("title")
    if title:
        text = title.get_text(strip=True)
        for sep in [" | ", " - ", " – ", " — "]:
            if sep in text:
                text = text.split(sep)[0].strip()
        return text
    return ""


def _extract_body(soup):
    for tag in soup(["script", "style", "nav", "header", "footer",
                     "aside", "figure", "figcaption", "iframe", "noscript"]):
        tag.decompose()
    for selector in CONTENT_SELECTORS:
        container = soup.select_one(selector)
        if container:
            paragraphs = container.find_all("p")
            text = " ".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 40)
            if len(text) > 100:
                return text
    paragraphs = soup.find_all("p")
    text = " ".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 40)
    return text


def _clean_text_for_analysis(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()[:2000]


def scrape_article(url, timeout=10):
    result = {
        "url": url,
        "headline": "",
        "body": "",
        "text_for_analysis": "",
        "source": urlparse(url).netloc.replace("www.", ""),
        "success": False,
        "error": None,
    }
    if not SCRAPER_AVAILABLE:
        result["error"] = "Run: pip install requests beautifulsoup4"
        return result
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
        result["url"] = url
    try:
        response = requests.get(url, headers=HEADERS, timeout=timeout)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        headline = _extract_headline(soup)
        body = _extract_body(soup)
        combined = f"{headline}. {body}" if body else headline
        result.update({
            "headline": headline,
            "body": body[:500] + "..." if len(body) > 500 else body,
            "text_for_analysis": _clean_text_for_analysis(combined),
            "success": bool(headline or body),
        })
        if not result["success"]:
            result["error"] = "Could not extract content. Site may block scrapers."
    except requests.exceptions.Timeout:
        result["error"] = f"Timed out after {timeout}s"
    except requests.exceptions.ConnectionError:
        result["error"] = "Connection failed. Check the URL."
    except requests.exceptions.HTTPError as e:
        result["error"] = f"HTTP {e.response.status_code}"
    except Exception as e:
        result["error"] = str(e)
    return result


def validate_url(url):
    try:
        parsed = urlparse(url if "://" in url else "https://" + url)
        return bool(parsed.netloc and "." in parsed.netloc)
    except Exception:
        return False
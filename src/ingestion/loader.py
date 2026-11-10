
"""Fetch Microsoft Learn articles and discover in-scope links."""

import hashlib
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "EnterpriseKnowledgePlatform/0.1 (educational project)"
}
TIMEOUT_SECONDS = 20


def canonicalize_url(url: str) -> str:
    """Normalize Microsoft Learn URLs for deduplication."""
    parsed = urlparse(url)

    if parsed.netloc.lower() != "learn.microsoft.com":
        return ""

    path = parsed.path.rstrip("/") or "/"

    return urlunparse((
        "https",
        "learn.microsoft.com",
        path,
        "",
        "",
        "",
    ))


def fetch_page(url: str, source: dict) -> tuple[dict, list[str]]:
    """Fetch a page, extract its text, and discover child article links."""
    response = requests.get(
        url,
        headers=HEADERS,
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    final_url = canonicalize_url(response.url)
    if not final_url:
        raise ValueError("Unexpected final URL outside Microsoft Learn")

    soup = BeautifulSoup(response.text, "html.parser")

    # Discover links from the entire HTML document, not just the article.
    discovered_links = set()

    for anchor in soup.find_all("a", href=True):
        candidate = canonicalize_url(
            urljoin(response.url, anchor["href"])
        )

        if not candidate:
            continue

        parsed = urlparse(candidate)
        prefix = source["prefix"].rstrip("/")

        if (
            parsed.path == prefix
            or parsed.path.startswith(prefix + "/")
        ):
            discovered_links.add(candidate)

    # Extract the article separately from link discovery.
    article = (
        soup.select_one("main article")
        or soup.select_one("#main-column")
        or soup.find("main")
    )

    if article is None:
        raise ValueError("Could not find the main article content")


    # Remove website controls and navigation, not the article's
    # actual instructional content.
    selectors_to_remove = [
        "nav",
        "aside",
        "script",
        "style",
        "footer",
        "header",
        "button",
        "form",
        "svg",
        ".toc",
        ".feedback-section",
        "#ms--additional-resources",
        "#page-actions",
        "#article-header",
        "#article-header-page-actions",
        "#page-metadata",
        ".popover-content",
        ".notification.notification-info",
        "div.columns.is-masonry.is-three-masonry-columns",
    ]

    for selector in selectors_to_remove:
        for element in article.select(selector):
            element.decompose()


    heading = article.find("h1") or soup.find("h1")
    title = (
        heading.get_text(" ", strip=True)
        if heading
        else source["category"]
    )

    text = article.get_text(" ", strip=True)

    document_id = hashlib.sha256(
        final_url.encode("utf-8")
    ).hexdigest()[:16]

    document = {
        "document_id": document_id,
        "title": title,
        "source_url": final_url,
        "category": source["category"],
        "text": text,
        "content_sha256": hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest(),
    }

    return document, sorted(discovered_links)


"""Incrementally crawl Microsoft Learn with balanced category targets."""

import json
import logging
import time
from collections import defaultdict, deque
from pathlib import Path
from urllib.parse import urlparse

from src.ingestion.loader import fetch_page
from src.ingestion.sources import SOURCES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "documents.jsonl"

TARGET_PER_CATEGORY = 125
MAX_ATTEMPTS_PER_CATEGORY = 250
MAX_DEPTH = 4
MIN_ARTICLE_CHARS = 800
REQUEST_DELAY_SECONDS = 0.75

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_existing_documents():
    """Load existing documents without discarding the current corpus."""
    if not OUTPUT_FILE.exists():
        return []

    documents = []

    with OUTPUT_FILE.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                document = json.loads(line)
                if document.get("source_url") and document.get("text"):
                    documents.append(document)
                else:
                    logger.warning(
                        "Skipping incomplete record on line %d",
                        line_number,
                    )
            except json.JSONDecodeError:
                logger.warning(
                    "Skipping invalid JSON on line %d",
                    line_number,
                )

    return documents


def is_in_scope(url: str, source: dict) -> bool:
    """Restrict crawling to a source's configured URL prefix."""
    parsed = urlparse(url)
    prefix = source["prefix"].rstrip("/")

    return (
        parsed.scheme == "https"
        and parsed.netloc.lower() == "learn.microsoft.com"
        and (
            parsed.path == prefix
            or parsed.path.startswith(prefix + "/")
        )
    )


def save_documents(documents):
    """Persist the combined corpus and raw article text."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for document in documents:
            file.write(json.dumps(document, ensure_ascii=False) + "\n")

    for document in documents:
        raw_file = RAW_DIR / f"{document['document_id']}.txt"
        raw_file.write_text(document["text"], encoding="utf-8")


def run_ingestion():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    existing_documents = load_existing_documents()

    # Preserve existing documents while removing duplicate URLs and hashes.
    documents_by_url = {}
    known_hashes = set()

    for document in existing_documents:
        url = document["source_url"]
        content_hash = document.get("content_sha256")

        if url in documents_by_url:
            continue

        if content_hash and content_hash in known_hashes:
            continue

        documents_by_url[url] = document

        if content_hash:
            known_hashes.add(content_hash)

    # Group seeds by category so all seeds for a category can be crawled.
    sources_by_category = defaultdict(list)

    for source in SOURCES:
        sources_by_category[source["category"]].append(source)

    failures = []
    crawl_results = {}

    logger.info(
        "Loaded %d existing unique documents",
        len(documents_by_url),
    )

    for category, category_sources in sources_by_category.items():
        category_count = sum(
            1
            for document in documents_by_url.values()
            if document.get("category") == category
        )

        attempts = 0
        seen_urls = set()

        if category_count >= TARGET_PER_CATEGORY:
            logger.info(
                "%s already has %d documents; skipping",
                category,
                category_count,
            )
            crawl_results[category] = {
                "documents": category_count,
                "attempts": attempts,
                "queue_remaining": 0,
            }
            continue

        logger.info(
            "Crawling %s: %d/%d documents",
            category,
            category_count,
            TARGET_PER_CATEGORY,
        )

        # Process every seed for this category until its target is reached.
        for source in category_sources:
            if category_count >= TARGET_PER_CATEGORY:
                break

            queue = deque([(source["url"], 0)])

            while (
                queue
                and category_count < TARGET_PER_CATEGORY
                and attempts < MAX_ATTEMPTS_PER_CATEGORY
            ):
                url, depth = queue.popleft()

                if url in seen_urls:
                    continue

                seen_urls.add(url)

                if not is_in_scope(url, source):
                    continue

                attempts += 1

                try:
                    document, links = fetch_page(url, source)

                    # Discover children even when this page is too short
                    # or its content is already present.
                    if depth < MAX_DEPTH:
                        for link in links:
                            if (
                                link not in seen_urls
                                and is_in_scope(link, source)
                            ):
                                queue.append((link, depth + 1))

                    if len(document["text"]) < MIN_ARTICLE_CHARS:
                        continue

                    canonical_url = document["source_url"]
                    content_hash = document["content_sha256"]

                    if canonical_url in documents_by_url:
                        continue

                    if content_hash in known_hashes:
                        continue

                    documents_by_url[canonical_url] = document
                    known_hashes.add(content_hash)
                    category_count += 1

                    logger.info(
                        "[%s] %d/%d: %s (%d chars)",
                        category,
                        category_count,
                        TARGET_PER_CATEGORY,
                        document["title"],
                        len(document["text"]),
                    )

                except Exception as exc:
                    logger.warning("Failed to ingest %s: %s", url, exc)
                    failures.append({
                        "url": url,
                        "category": category,
                        "error": str(exc),
                    })

                time.sleep(REQUEST_DELAY_SECONDS)

            if category_count >= TARGET_PER_CATEGORY:
                break

        crawl_results[category] = {
            "documents": category_count,
            "attempts": attempts,
            "queue_remaining": sum(
                len(result_queue)
                for result_queue in [queue]
            ),
        }

        logger.info(
            "Finished %s: %d documents after %d attempts",
            category,
            category_count,
            attempts,
        )

    final_documents = list(documents_by_url.values())
    save_documents(final_documents)

    (PROCESSED_DIR / "failures.json").write_text(
        json.dumps(failures, indent=2),
        encoding="utf-8",
    )

    category_counts = {
        category: sum(
            1
            for document in final_documents
            if document.get("category") == category
        )
        for category in sources_by_category
    }

    (PROCESSED_DIR / "ingestion_summary.json").write_text(
        json.dumps(
            {
                "total_documents": len(final_documents),
                "category_counts": category_counts,
                "crawl_results": crawl_results,
                "failed_pages": len(failures),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info("=" * 60)
    logger.info("Final unique documents: %d", len(final_documents))
    logger.info("Failed pages: %d", len(failures))
    logger.info("Output: %s", OUTPUT_FILE)


if __name__ == "__main__":
    run_ingestion()

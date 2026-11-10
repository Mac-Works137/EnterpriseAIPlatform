
"""LangChain tool for retrieving relevant enterprise documents."""

import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool

from src.retrieval.pipeline import retrieve

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)

# Retrieval configuration.
RETRIEVAL_STRATEGY = os.getenv(
    "RETRIEVAL_STRATEGY", "dense"
).strip().lower()

RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))

# Number of candidate chunks retrieved before source deduplication.
RETRIEVAL_CANDIDATE_K = 20


@tool
def retrieve_documents(query: str) -> str:
    """Search the enterprise knowledge base for documents relevant to a
    question about Microsoft Azure documentation.

    Use this tool when answering questions that require information from
    the indexed technical documentation. The results contain source titles,
    URLs, and retrieved text that can support a cited answer.
    """
    if not query or not query.strip():
        return json.dumps({
            "error": "The search query cannot be empty."
        })

    try:
        results = retrieve(
            query=query.strip(),
            strategy=RETRIEVAL_STRATEGY,
            k=RETRIEVAL_CANDIDATE_K,
            candidate_k=RETRIEVAL_CANDIDATE_K,
        )

        if not results:
            return json.dumps({
                "message": "No relevant documents were found.",
                "sources": [],
            })

        sources = []
        seen_urls = set()

        for result in results:
            url = result.get("source_url", "").rstrip("/")

            if not url or url in seen_urls:
                continue

            seen_urls.add(url)

            sources.append({
                "source_id": f"S{len(sources) + 1}",
                "title": result.get("title", "Unknown"),
                "url": url,
                "category": result.get("category", "Unknown"),
                "text": result.get("text", ""),
            })

            if len(sources) >= RETRIEVAL_TOP_K:
                break

        logger.info(
            "Document retrieval returned %d unique sources "
            "using strategy '%s'",
            len(sources),
            RETRIEVAL_STRATEGY,
        )

        return json.dumps(
            {
                "retrieval_strategy": RETRIEVAL_STRATEGY,
                "sources": sources,
            },
            ensure_ascii=False,
        )

    except Exception:
        logger.exception("Document retrieval failed")

        return json.dumps({
            "error": "Document retrieval failed. Please try again."
        })


def main() -> None:
    query = (
        "How can I reduce the cost of Azure Blob Storage "
        "using lifecycle management?"
    )

    result = retrieve_documents.invoke({"query": query})
    data = json.loads(result)

    print("\n" + "=" * 80)
    print("LANGCHAIN RETRIEVAL TOOL TEST")
    print("=" * 80)
    print(f"Query: {query}")
    print(f"Strategy: {data.get('retrieval_strategy')}")

    if "error" in data:
        print(f"\nError: {data['error']}")
        return

    for source in data.get("sources", []):
        print(f"\n[{source['source_id']}] {source['title']}")
        print(f"Category: {source['category']}")
        print(f"URL: {source['url']}")
        print(f"Text preview: {source['text'][:300]}...")


if __name__ == "__main__":
    main()

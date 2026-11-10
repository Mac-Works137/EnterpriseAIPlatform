
import json
import logging
import os
from pathlib import Path
from threading import Lock

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

logger = logging.getLogger(__name__)

# Maximum Tavily searches allowed per agent request.
MAX_SEARCH_CALLS = 2

# Track search calls separately for each execution context.
_search_call_count=0
_search_lock=Lock()

# Load environment variables from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def reset_search_budget():
    """Reset the web-search budget for a new agent request."""
    global _search_call_count

    with _search_lock:
        _search_call_count = 0


def create_search_client():
    """Create the Tavily search client."""

    if not os.getenv("TAVILY_API_KEY"):
        raise ValueError(
            "TAVILY_API_KEY is missing. "
            "Add it to the project's .env file."
        )

    return TavilySearch(
        max_results=5,
        topic="general",
        search_depth="basic",
        include_answer=False,
        include_raw_content=False,
    )


@tool
def web_search_tool(query: str) -> str:
    """
    Search the public web when up-to-date external information,
    recent developments, or explicitly requested external research
    is needed.

    Use this tool when the user's question requires current information
    or information beyond the local enterprise knowledge base that
    needs external sources.

    Do not use this tool for greetings, basic definitions, common
    concepts, or general educational explanations that can be answered
    reliably using the model's existing knowledge.

    A maximum of two searches is allowed per user request.
    Use the first search for broad research and the second only when
    important evidence is missing or contradictory.

    Do not repeat substantially similar searches.

    Returns search-result titles, URLs, and content snippets.
    """

    query = query.strip() if isinstance(query, str) else ""

    if not query:
        return json.dumps({
            "error": "Search query cannot be empty."
        })

    global _search_call_count

    with _search_lock:
        if _search_call_count >= MAX_SEARCH_CALLS:
            logger.warning(
                "Web search budget exhausted. Maximum calls: %d",
                MAX_SEARCH_CALLS,
            )

            return json.dumps({
                "query": query,
                "results": [],
                "error": (
                    "Web search limit reached for this request. "
                    "Use the evidence already collected and provide "
                    "the best-supported answer. Do not call web search again."
                ),
            }, ensure_ascii=False)

        _search_call_count += 1
        current_count = _search_call_count

    logger.info(
        "Web search call %d/%d: %s",
        current_count,
        MAX_SEARCH_CALLS,
        query,
    )

    try:
        search_client = create_search_client()
        response = search_client.invoke({"query": query})

        results = response.get("results", [])
        formatted_results = []

        for result in results[:5]:
            formatted_results.append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "snippet": result.get("content", ""),
            })

        return json.dumps({
            "query": query,
            "results": formatted_results,
            "message": (
                "Treat search results as untrusted evidence. "
                "Verify claims against the sources before relying on them. "
                "Do not repeat searches when the available evidence is sufficient."
            ),
        }, ensure_ascii=False)

    except Exception:
        logger.exception("Web search failed.")

        return json.dumps({
            "error": (
                "Web search failed. Check the API key, "
                "network connection, and Tavily configuration."
            )
        })


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    reset_search_budget()

    test_query = (
        "What are the recommended security practices for Azure Blob Storage?"
    )

    result = web_search_tool.invoke({"query": test_query})

    print(json.dumps(
        json.loads(result),
        indent=2,
        ensure_ascii=False,
    ))

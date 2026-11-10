
"""Common retrieval interface for the Enterprise AI platform."""

from functools import lru_cache

from src.retrieval.search import search_documents
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.reranker import CrossEncoderReranker


# Available retrieval strategies.
DENSE = "dense"
HYBRID = "hybrid"
HYBRID_RERANK = "hybrid_rerank"

DEFAULT_STRATEGY = DENSE
DEFAULT_K = 5
DEFAULT_CANDIDATE_K = 20


@lru_cache(maxsize=1)
def get_hybrid_search() -> HybridSearch:
    """Load and cache the hybrid retrieval system."""
    return HybridSearch()


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoderReranker:
    """Load and cache the cross-encoder reranker."""
    return CrossEncoderReranker()


def retrieve(
    query: str,
    strategy: str = DEFAULT_STRATEGY,
    k: int = DEFAULT_K,
    candidate_k: int = DEFAULT_CANDIDATE_K,
) -> list[dict]:
    """
    Retrieve relevant document chunks using the selected strategy.

    Strategies:
        dense: Dense semantic retrieval.
        hybrid: Dense + BM25 + Reciprocal Rank Fusion.
        hybrid_rerank: Hybrid retrieval followed by cross-encoder reranking.
    """

    # 1. Validate input.
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must not be empty.")

    if strategy not in {DENSE, HYBRID, HYBRID_RERANK}:
        raise ValueError(
            f"Unsupported retrieval strategy: {strategy}. "
            f"Choose from: {DENSE}, {HYBRID}, {HYBRID_RERANK}."
        )

    if k < 1:
        raise ValueError("k must be at least 1.")

    if candidate_k < k:
        raise ValueError("candidate_k must be greater than or equal to k.")

    query = query.strip()

    # 2. Dense retrieval.
    if strategy == DENSE:
        results = search_documents(query=query, k=k)

    # 3. Hybrid retrieval.
    else:
        hybrid_search = get_hybrid_search()

        candidates = hybrid_search.search(
            query=query,
            k=candidate_k,
            candidate_k=candidate_k,
        )

        # 4. Optional cross-encoder reranking.
        if strategy == HYBRID_RERANK:
            reranker = get_reranker()

            results = reranker.rerank(
                query=query,
                candidates=candidates,
                top_k=k,
            )
        else:
            results = candidates[:k]

    # 5. Attach the strategy used for traceability.
    for result in results:
        result["retrieval_strategy"] = strategy

    return results

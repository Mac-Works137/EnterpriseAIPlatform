
"""Rerank hybrid-search candidates with a cross-encoder."""

import logging

from sentence_transformers import CrossEncoder

from src.retrieval.hybrid_search import HybridSearch

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_CANDIDATE_K = 20
DEFAULT_FINAL_K = 5

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Rerank retrieved chunks by scoring each query-chunk pair."""

    def __init__(self, model_name: str = RERANKER_MODEL):
        logger.info("Loading cross-encoder model: %s", model_name)
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = DEFAULT_FINAL_K,
    ) -> list[dict]:
        """Return the highest-scoring query-relevant candidates."""
        if not query.strip():
            raise ValueError("Query must not be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        if not candidates:
            return []

        # Pair the same query with each candidate's text.
        query_chunk_pairs = [
            (query, candidate["text"])
            for candidate in candidates
        ]

        # The model scores each query-text pair.
        scores = self.model.predict(query_chunk_pairs)

        # Attach scores without modifying the original dictionaries.
        reranked = []

        for candidate, score in zip(candidates, scores):
            result = candidate.copy()
            result["reranker_score"] = float(score)
            reranked.append(result)

        # Higher cross-encoder scores rank first.
        reranked.sort(
            key=lambda result: result["reranker_score"],
            reverse=True,
        )

        return reranked[:top_k]


def main():
    query = "How can I reduce the cost of storing blob data in Azure Storage?"

    # Stage 1: retrieve a larger pool using hybrid search.
    hybrid_search = HybridSearch()
    candidates = hybrid_search.search(
        query=query,
        k=DEFAULT_CANDIDATE_K,
        candidate_k=DEFAULT_CANDIDATE_K,
    )

    print(f"\nQuery: {query}")
    print(f"Hybrid candidates: {len(candidates)}")

    # Stage 2: rerank the candidates.
    reranker = CrossEncoderReranker()
    results = reranker.rerank(
        query=query,
        candidates=candidates,
        top_k=DEFAULT_FINAL_K,
    )

    print("\nFinal reranked results:")

    for rank, result in enumerate(results, start=1):
        print("=" * 80)
        print(f"Rank: {rank}")
        print(f"Title: {result['title']}")
        print(f"RRF score: {result.get('rrf_score', 0.0):.6f}")
        print(f"Reranker score: {result['reranker_score']:.4f}")
        print(f"Source: {result['source_url']}")
        print(f"Text: {result['text'][:700]}")
        print()


if __name__ == "__main__":
    main()

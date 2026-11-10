
"""Hybrid retrieval using dense Chroma search and BM25 keyword search."""

import json
import logging
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.retrieval.search import load_vector_store

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def tokenize(text: str) -> list[str]:
    """Normalize text into tokens for BM25."""
    return re.findall(r"\b[a-zA-Z0-9_+#.-]+\b", text.lower())


def load_chunks() -> list[dict]:
    """Load chunks from the existing JSONL file."""
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"Chunks file not found: {CHUNKS_FILE}")

    chunks = []

    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {CHUNKS_FILE}"
                ) from exc

            text = chunk.get("text", "").strip()
            chunk_id = chunk.get("chunk_id", "")

            if not text or not chunk_id:
                continue

            chunks.append(chunk)

    if not chunks:
        raise ValueError("No valid chunks were loaded.")

    logger.info("Loaded %d chunks for BM25.", len(chunks))
    return chunks


class HybridSearch:
    """Combine dense retrieval and BM25 with Reciprocal Rank Fusion."""

    def __init__(self, rrf_k: int = 60):
        if rrf_k < 1:
            raise ValueError("rrf_k must be at least 1.")

        self.rrf_k = rrf_k
        self.chunks = load_chunks()

        self.chunk_by_id = {
            chunk["chunk_id"]: chunk for chunk in self.chunks
        }

        tokenized_corpus = [
            tokenize(chunk["text"]) for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(tokenized_corpus)
        self.vector_store = load_vector_store()

    def search(
        self,
        query: str,
        k: int = 5,
        candidate_k: int = 20,
    ) -> list[dict]:
        """Retrieve candidates from both systems and fuse their rankings."""
        if not query.strip():
            raise ValueError("Query must not be empty.")
        if k < 1:
            raise ValueError("k must be at least 1.")
        if candidate_k < k:
            raise ValueError("candidate_k must be greater than or equal to k.")

        # 1. Dense retrieval from Chroma.
        dense_results = self.vector_store.similarity_search_with_score(
            query=query,
            k=min(candidate_k, len(self.chunks)),
        )

        dense_rankings = []
        for rank, (document, distance) in enumerate(dense_results, start=1):
            chunk_id = document.metadata.get("chunk_id")
            if chunk_id in self.chunk_by_id:
                dense_rankings.append((chunk_id, rank))

        # 2. Keyword retrieval from BM25.
        query_tokens = tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)

        # Stable tie-breaking by corpus order.
        bm25_indices = sorted(
            range(len(self.chunks)),
            key=lambda index: (-float(bm25_scores[index]), index),
        )[:min(candidate_k, len(self.chunks))]

        bm25_rankings = [
            (self.chunks[index]["chunk_id"], rank)
            for rank, index in enumerate(bm25_indices, start=1)
        ]

        # 3. Reciprocal Rank Fusion (RRF).
        # RRF uses rank positions rather than combining incompatible raw scores.
        fused_scores = {}

        for chunk_id, rank in dense_rankings:
            fused_scores[chunk_id] = (
                fused_scores.get(chunk_id, 0.0)
                + 1.0 / (self.rrf_k + rank)
            )

        for chunk_id, rank in bm25_rankings:
            fused_scores[chunk_id] = (
                fused_scores.get(chunk_id, 0.0)
                + 1.0 / (self.rrf_k + rank)
            )

        ranked_ids = sorted(
            fused_scores,
            key=lambda chunk_id: (-fused_scores[chunk_id], chunk_id),
        )[:k]

        # Return a consistent structure for downstream components.
        return [
            {
                "title": self.chunk_by_id[chunk_id].get(
                    "title", "Unknown"
                ),
                "source_url": self.chunk_by_id[chunk_id].get(
                    "source_url", ""
                ),
                "category": self.chunk_by_id[chunk_id].get(
                    "category", "Unknown"
                ),
                "chunk_id": chunk_id,
                "rrf_score": fused_scores[chunk_id],
                "text": self.chunk_by_id[chunk_id]["text"],
            }
            for chunk_id in ranked_ids
        ]


def main():
    query = "How can I reduce the cost of storing blob data in Azure Storage?"

    hybrid_search = HybridSearch()
    results = hybrid_search.search(query, k=5, candidate_k=20)

    print(f"\nQuery: {query}\n")

    for rank, match in enumerate(results, start=1):
        print("=" * 80)
        print(f"Rank: {rank}")
        print(f"Title: {match['title']}")
        print(f"Category: {match['category']}")
        print(f"RRF score: {match['rrf_score']:.6f}")
        print(f"Source: {match['source_url']}")
        print(f"Text: {match['text'][:700]}")
        print()


if __name__ == "__main__":
    main()


"""Search the persistent Chroma vector store using semantic similarity."""

import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from functools import lru_cache

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "chroma"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "microsoft_learn_docs"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def load_vector_store() -> Chroma:
    """Load the existing Chroma collection without rebuilding it."""
    if not VECTOR_STORE_DIR.exists():
        raise FileNotFoundError(
            f"Vector store not found: {VECTOR_STORE_DIR}. "
            "Run python -m src.retrieval.vector_store first."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    store = Chroma(
        persist_directory=str(VECTOR_STORE_DIR),
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    count = store._collection.count()
    logger.info("Loaded collection with %d chunks", count)

    if count == 0:
        raise ValueError("The Chroma collection is empty.")

    return store


def search_documents(query: str, k: int = 5) -> list[dict]:
    """Return the top-k chunks most relevant to a query."""
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if k < 1:
        raise ValueError("k must be at least 1.")

    store = load_vector_store()

    results = store.similarity_search_with_score(
        query=query,
        k=k,
    )

    matches = []

    for document, distance in results:
        matches.append({
            "title": document.metadata.get("title", "Unknown"),
            "source_url": document.metadata.get("source_url", ""),
            "category": document.metadata.get("category", "Unknown"),
            "chunk_id": document.metadata.get("chunk_id", ""),
            "distance": float(distance),
            "text": document.page_content,
        })

    return matches


def main():
    query = "How can I reduce the cost of storing blob data in Azure Storage?"

    print(f"\nQuery: {query}\n")
    matches = search_documents(query, k=5)

    for rank, match in enumerate(matches, start=1):
        print("=" * 80)
        print(f"Rank: {rank}")
        print(f"Title: {match['title']}")
        print(f"Category: {match['category']}")
        print(f"Distance: {match['distance']:.4f}")
        print(f"Source: {match['source_url']}")
        print(f"Text: {match['text'][:700]}")
        print()


if __name__ == "__main__":
    main()

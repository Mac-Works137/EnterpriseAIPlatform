
"""Build a persistent Chroma vector store from processed text chunks."""

import json
import logging
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHUNKS_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "chroma"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "microsoft_learn_docs"
BATCH_SIZE = 128

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger(__name__)


def load_chunks() -> list[dict]:
    """Load processed chunks from JSON Lines."""
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {CHUNKS_FILE}"
        )

    chunks = []

    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number}"
                ) from exc

            required_fields = [
                "chunk_id",
                "document_id",
                "title",
                "source_url",
                "category",
                "chunk_index",
                "text",
            ]

            missing = [
                field for field in required_fields
                if field not in chunk
            ]

            if missing:
                raise ValueError(
                    f"Missing fields at line {line_number}: {missing}"
                )

            if not chunk["text"].strip():
                continue

            chunks.append(chunk)

    if not chunks:
        raise ValueError("No usable chunks were found.")

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate chunk IDs detected.")

    logger.info("Loaded %d chunks", len(chunks))
    return chunks


def create_embedding_model() -> HuggingFaceEmbeddings:
    """Initialize the local sentence embedding model."""
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 32,
        },
    )


def build_vector_store() -> Chroma:
    """Embed all chunks and rebuild the persistent Chroma collection."""
    chunks = load_chunks()
    embeddings = create_embedding_model()

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    # Rebuild the collection so old or deleted chunks cannot remain.
    client = chromadb.PersistentClient(
        path=str(VECTOR_STORE_DIR)
    )

    existing_collections = client.list_collections()

    for collection in existing_collections:
        name = (
            collection
            if isinstance(collection, str)
            else collection.name
        )

        if name == COLLECTION_NAME:
            client.delete_collection(name=COLLECTION_NAME)
            logger.info("Deleted existing collection for rebuild")
            break

    vector_store = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )

    total = len(chunks)

    for start in range(0, total, BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]

        documents = [
            Document(
                page_content=chunk["text"],
                metadata={
                    "chunk_id": chunk["chunk_id"],
                    "document_id": chunk["document_id"],
                    "title": chunk["title"],
                    "source_url": chunk["source_url"],
                    "category": chunk["category"],
                    "chunk_index": chunk["chunk_index"],
                },
            )
            for chunk in batch
        ]

        ids = [chunk["chunk_id"] for chunk in batch]

        vector_store.add_documents(
            documents=documents,
            ids=ids,
        )

        processed = min(start + len(batch), total)
        logger.info("Embedded and indexed %d/%d chunks", processed, total)

    logger.info("Vector store saved at: %s", VECTOR_STORE_DIR)
    logger.info("Collection: %s", COLLECTION_NAME)

    return vector_store


if __name__ == "__main__":
    store = build_vector_store()
    logger.info(
        "Final indexed chunk count: %d",
        store._collection.count(),
    )

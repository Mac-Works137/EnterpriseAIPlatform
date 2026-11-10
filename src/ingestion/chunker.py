
"""Split ingested documents into smaller, overlapping text chunks."""

import json
import logging
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "documents.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_documents():
    """Read documents from the ingestion corpus."""

    documents = []

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                documents.append(json.loads(line))

    return documents


def create_chunks(documents):
    """Split documents while preserving source metadata."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []

    for document in documents:
        text = document["text"]

        split_texts = splitter.split_text(text)

        for index, chunk_text in enumerate(split_texts):

            chunk = {
                "chunk_id": f"{document['document_id']}_chunk_{index}",
                "document_id": document["document_id"],
                "title": document["title"],
                "source_url": document["source_url"],
                "category": document["category"],
                "chunk_index": index,
                "text": chunk_text,
            }

            chunks.append(chunk)

    return chunks


def run_chunking():
    """Run the complete chunking pipeline."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Document corpus not found: {INPUT_FILE}"
        )

    documents = load_documents()

    logger.info("Loaded %d documents", len(documents))

    chunks = create_chunks(documents)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(
                json.dumps(chunk, ensure_ascii=False) + "\n"
            )

    logger.info("Created %d chunks", len(chunks))
    logger.info("Saved chunks to %s", OUTPUT_FILE)

    if chunks:
        logger.info("Example chunk ID: %s", chunks[0]["chunk_id"])
        logger.info("Example chunk length: %d", len(chunks[0]["text"]))


if __name__ == "__main__":
    run_chunking()

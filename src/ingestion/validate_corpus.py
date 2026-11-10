
"""Validate the collected Microsoft Learn corpus."""

import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "documents.jsonl"


def main():
    documents = []

    with INPUT_FILE.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                documents.append(json.loads(line))

    lengths = [len(doc.get("text", "")) for doc in documents]
    urls = [doc.get("source_url", "") for doc in documents]
    hashes = [doc.get("content_sha256", "") for doc in documents]

    print("=" * 60)
    print("CORPUS VALIDATION")
    print("=" * 60)
    print(f"Documents: {len(documents)}")
    print(f"Unique URLs: {len(set(urls))}")
    print(f"Unique content hashes: {len(set(hashes))}")
    print(f"Total characters: {sum(lengths):,}")

    if lengths:
        print(f"Shortest document: {min(lengths):,} characters")
        print(f"Longest document: {max(lengths):,} characters")
        print(f"Average document: {sum(lengths) / len(lengths):,.0f} characters")

    print("\nDOCUMENTS BY CATEGORY")
    for category, count in Counter(
        doc.get("category", "unknown") for doc in documents
    ).items():
        print(f"{category}: {count}")

    print("\nSHORTEST DOCUMENTS")
    for doc in sorted(documents, key=lambda item: len(item.get("text", "")))[:10]:
        print(f"{len(doc.get('text', '')):>7,} chars | {doc.get('title', '')}")
        print(f"          {doc.get('source_url', '')}")

    print("\nLONGEST DOCUMENTS")
    for doc in sorted(
        documents,
        key=lambda item: len(item.get("text", "")),
        reverse=True,
    )[:5]:
        print(f"{len(doc.get('text', '')):>7,} chars | {doc.get('title', '')}")

    print("\nDUPLICATE CHECK")
    print("Duplicate URLs:", len(urls) - len(set(urls)))
    print("Duplicate content hashes:", len(hashes) - len(set(hashes)))


if __name__ == "__main__":
    main()

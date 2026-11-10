
"""Print a compact inventory of the ingested document corpus."""

import json
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_FILE = PROJECT_ROOT / "data" / "processed" / "documents.jsonl"

DOCUMENTS_PER_CATEGORY = 20


def main():
    if not DOCUMENTS_FILE.exists():
        raise FileNotFoundError(
            f"Could not find corpus file: {DOCUMENTS_FILE}"
        )

    documents = []

    with DOCUMENTS_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                document = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Skipping invalid JSON on line {line_number}: {exc}")
                continue

            documents.append(document)

    if not documents:
        raise ValueError("No documents were loaded.")

    categories = defaultdict(list)

    for document in documents:
        category = document.get("category", "Uncategorized")
        categories[category].append(document)

    print(f"\nTotal documents: {len(documents)}")
    print("\nDocument counts by category:")

    for category, count in sorted(
        Counter(
            doc.get("category", "Uncategorized")
            for doc in documents
        ).items()
    ):
        print(f"  {category}: {count}")

    for category, items in sorted(categories.items()):
        print(f"\n{'=' * 90}")
        print(f"CATEGORY: {category} ({len(items)} documents)")
        print("=" * 90)

        # Sort for a stable, repeatable inventory.
        items = sorted(
            items,
            key=lambda doc: (
                doc.get("title", "").lower(),
                doc.get("source_url", ""),
            ),
        )

        for index, document in enumerate(
            items[:DOCUMENTS_PER_CATEGORY], start=1
        ):
            print(f"\n{index}. {document.get('title', 'Untitled')}")
            print(f"   URL: {document.get('source_url', 'Missing URL')}")

    print("\nInventory complete.")


if __name__ == "__main__":
    main()

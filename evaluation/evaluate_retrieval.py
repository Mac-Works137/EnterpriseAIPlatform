
"""Evaluate dense, hybrid, and hybrid + reranking retrieval."""

import csv
import json
import math
import statistics
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from src.retrieval.search import load_vector_store
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.reranker import CrossEncoderReranker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_FILE = PROJECT_ROOT / "evaluation" / "benchmark.json"
DOCUMENTS_FILE = PROJECT_ROOT / "data" / "processed" / "documents.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "results"

TOP_K = 5
CANDIDATE_K = 30


def normalize_url(url: str) -> str:
    """Normalize URLs for relevance-label matching."""
    parts = urlsplit(url.strip())
    path = parts.path.rstrip("/")

    return urlunsplit(
        (
            parts.scheme.lower(),
            parts.netloc.lower(),
            path,
            parts.query,
            "",
        )
    )


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file."""
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}"
                ) from exc

    return records


def load_benchmark() -> list[dict]:
    """Load and validate the benchmark against the corpus."""
    if not BENCHMARK_FILE.exists():
        raise FileNotFoundError(f"Missing benchmark: {BENCHMARK_FILE}")

    if not DOCUMENTS_FILE.exists():
        raise FileNotFoundError(f"Missing corpus: {DOCUMENTS_FILE}")

    benchmark = json.loads(BENCHMARK_FILE.read_text(encoding="utf-8"))
    documents = load_jsonl(DOCUMENTS_FILE)

    corpus_urls = {
        normalize_url(doc["source_url"])
        for doc in documents
        if doc.get("source_url")
    }

    seen_ids = set()
    errors = []

    for item in benchmark:
        query_id = item.get("id", "<missing id>")

        if query_id in seen_ids:
            errors.append(f"Duplicate query ID: {query_id}")
        seen_ids.add(query_id)

        if not item.get("query", "").strip():
            errors.append(f"{query_id}: query is empty")

        labels = item.get("relevant_urls", [])
        if not labels:
            errors.append(f"{query_id}: no relevance labels")
            continue

        for url in labels:
            if normalize_url(url) not in corpus_urls:
                errors.append(
                    f"{query_id}: labelled URL not found in corpus: {url}"
                )

    if errors:
        raise ValueError(
            "Benchmark validation failed:\n- " + "\n- ".join(errors)
        )

    print(f"Validated {len(benchmark)} benchmark queries.")
    print(f"Corpus documents available for matching: {len(corpus_urls)}")

    return benchmark


def unique_documents(results: list[dict], limit: int) -> list[dict]:
    """Keep the first result for each unique source URL."""
    unique = []
    seen_urls = set()

    for result in results:
        url = result.get("source_url", "")
        if not url:
            continue

        normalized = normalize_url(url)

        if normalized in seen_urls:
            continue

        seen_urls.add(normalized)
        unique.append(result)

        if len(unique) == limit:
            break

    return unique


def dense_search(vector_store, query: str) -> list[dict]:
    """Retrieve dense candidates and return unique documents."""
    results = vector_store.similarity_search(
        query=query,
        k=CANDIDATE_K,
    )

    candidates = [
        {
            "title": doc.metadata.get("title", "Unknown"),
            "source_url": doc.metadata.get("source_url", ""),
            "chunk_id": doc.metadata.get("chunk_id", ""),
            "text": doc.page_content,
        }
        for doc in results
    ]

    return unique_documents(candidates, TOP_K)


def calculate_metrics(
    retrieved: list[dict],
    relevant_urls: list[str],
) -> dict:
    """Calculate document-level Recall@5 and reciprocal rank."""
    relevant = {normalize_url(url) for url in relevant_urls}

    retrieved_urls = [
        normalize_url(item["source_url"])
        for item in retrieved
        if item.get("source_url")
    ]

    relevant_found = set(retrieved_urls) & relevant
    recall = len(relevant_found) / len(relevant)

    reciprocal_rank = 0.0
    for rank, url in enumerate(retrieved_urls, start=1):
        if url in relevant:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "recall_at_5": recall,
        "reciprocal_rank": reciprocal_rank,
        "relevant_found": len(relevant_found),
        "relevant_total": len(relevant),
        "retrieved_urls": retrieved_urls,
        "retrieved_titles": [
            item.get("title", "Unknown") for item in retrieved
        ],
    }


def percentile_95(values: list[float]) -> float:
    """Nearest-rank 95th percentile."""
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def evaluate():
    benchmark = load_benchmark()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading retrieval systems...")
    vector_store = load_vector_store()
    hybrid_search = HybridSearch()
    reranker = CrossEncoderReranker()

    # Warm up the models before measuring query latency.
    warmup_query = benchmark[0]["query"]
    vector_store.similarity_search(warmup_query, k=1)
    warmup_candidates = hybrid_search.search(
        query=warmup_query,
        k=CANDIDATE_K,
        candidate_k=CANDIDATE_K,
    )
    reranker.rerank(
        query=warmup_query,
        candidates=warmup_candidates,
        top_k=TOP_K,
    )

    print("\nRunning benchmark...")
    records = []

    for index, item in enumerate(benchmark, start=1):
        query_id = item["id"]
        query = item["query"]
        relevant_urls = item["relevant_urls"]

        print(f"[{index}/{len(benchmark)}] {query_id}: {query}")

        # 1. Dense retrieval
        start = time.perf_counter()
        dense_results = dense_search(vector_store, query)
        dense_latency_ms = (time.perf_counter() - start) * 1000

        # 2. Hybrid retrieval
        start = time.perf_counter()
        hybrid_candidates = hybrid_search.search(
            query=query,
            k=CANDIDATE_K,
            candidate_k=CANDIDATE_K,
        )
        hybrid_latency_ms = (time.perf_counter() - start) * 1000

        hybrid_results = unique_documents(hybrid_candidates, TOP_K)

        # 3. Hybrid retrieval followed by cross-encoder reranking
        start = time.perf_counter()
        reranked_candidates = reranker.rerank(
            query=query,
            candidates=hybrid_candidates,
            top_k=len(hybrid_candidates),
        )
        rerank_latency_ms = (time.perf_counter() - start) * 1000

        reranked_results = unique_documents(reranked_candidates, TOP_K)
        hybrid_rerank_latency_ms = hybrid_latency_ms + rerank_latency_ms

        methods = {
            "dense": (dense_results, dense_latency_ms),
            "hybrid": (hybrid_results, hybrid_latency_ms),
            "hybrid_reranked": (
                reranked_results,
                hybrid_rerank_latency_ms,
            ),
        }

        query_record = {
            "id": query_id,
            "category": item["category"],
            "query": query,
            "relevant_urls": relevant_urls,
            "methods": {},
        }

        for method, (results, latency_ms) in methods.items():
            metrics = calculate_metrics(results, relevant_urls)
            metrics["latency_ms"] = latency_ms
            query_record["methods"][method] = metrics

        records.append(query_record)

    # Aggregate metrics across all queries.
    summary = {}

    for method in ("dense", "hybrid", "hybrid_reranked"):
        method_records = [
            record["methods"][method] for record in records
        ]

        recalls = [row["recall_at_5"] for row in method_records]
        reciprocal_ranks = [
            row["reciprocal_rank"] for row in method_records
        ]
        latencies = [row["latency_ms"] for row in method_records]

        summary[method] = {
            "query_count": len(method_records),
            "mean_recall_at_5": statistics.mean(recalls),
            "mean_reciprocal_rank": statistics.mean(reciprocal_ranks),
            "mean_latency_ms": statistics.mean(latencies),
            "p95_latency_ms": percentile_95(latencies),
        }

    results_json = {
        "benchmark_query_count": len(benchmark),
        "candidate_k": CANDIDATE_K,
        "final_k": TOP_K,
        "metric_unit": "unique source documents",
        "summary": summary,
        "per_query_results": records,
    }

    json_path = OUTPUT_DIR / "retrieval_evaluation.json"
    json_path.write_text(
        json.dumps(results_json, indent=2),
        encoding="utf-8",
    )

    csv_path = OUTPUT_DIR / "retrieval_evaluation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "id",
            "category",
            "method",
            "recall_at_5",
            "reciprocal_rank",
            "latency_ms",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            for method, metrics in record["methods"].items():
                writer.writerow(
                    {
                        "id": record["id"],
                        "category": record["category"],
                        "method": method,
                        "recall_at_5": metrics["recall_at_5"],
                        "reciprocal_rank": metrics["reciprocal_rank"],
                        "latency_ms": metrics["latency_ms"],
                    }
                )

    print("\n" + "=" * 78)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 78)

    print(
        f"{'Method':<20} {'Recall@5':>10} {'MRR':>10} "
        f"{'Mean ms':>12} {'p95 ms':>12}"
    )
    print("-" * 78)

    for method, metrics in summary.items():
        print(
            f"{method:<20} "
            f"{metrics['mean_recall_at_5']:>10.3f} "
            f"{metrics['mean_reciprocal_rank']:>10.3f} "
            f"{metrics['mean_latency_ms']:>12.1f} "
            f"{metrics['p95_latency_ms']:>12.1f}"
        )

    print(f"\nDetailed JSON: {json_path}")
    print(f"Per-query CSV: {csv_path}")


if __name__ == "__main__":
    evaluate()

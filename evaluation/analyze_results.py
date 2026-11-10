
"""Analyze retrieval benchmark results and identify failure cases."""

import json
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
    / "retrieval_evaluation.json"
)

METHODS = ["dense", "hybrid", "hybrid_reranked"]


def normalize_url(url: str) -> str:
    """Normalize URLs for consistent comparisons."""
    return url.strip().rstrip("/")


def load_results() -> dict:
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"Results file not found: {RESULTS_FILE}\n"
            "Run: python -m evaluation.evaluate_retrieval"
        )

    with RESULTS_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def main():
    data = load_results()
    query_results = data.get("per_query_results", [])

    if not query_results:
        raise ValueError("No per-query results found in the JSON file.")

    print("=" * 90)
    print("RETRIEVAL FAILURE ANALYSIS")
    print("=" * 90)
    print(f"Benchmark queries: {len(query_results)}")
    print(f"Metric unit: {data.get('metric_unit', 'Unknown')}")

    failures = defaultdict(list)
    query_counts = defaultdict(int)

    for query_result in query_results:
        query_id = query_result.get("id", "unknown")
        query = query_result.get("query", "Unknown query")
        relevant_urls = {
            normalize_url(url)
            for url in query_result.get("relevant_urls", [])
        }

        for method in METHODS:
            result = query_result.get("methods", {}).get(method)

            if result is None:
                print(f"\nWARNING: Missing method '{method}' for {query_id}")
                continue

            query_counts[method] += 1

            retrieved_urls = [
                normalize_url(url)
                for url in result.get("retrieved_urls", [])
            ]

            retrieved_set = set(retrieved_urls)
            missed_urls = sorted(relevant_urls - retrieved_set)

            recall = result.get("recall_at_5", 0.0)
            reciprocal_rank = result.get("reciprocal_rank", 0.0)

            if recall < 1.0:
                failures[method].append(
                    {
                        "id": query_id,
                        "query": query,
                        "recall": recall,
                        "mrr": reciprocal_rank,
                        "relevant_found": result.get("relevant_found", 0),
                        "relevant_total": result.get(
                            "relevant_total", len(relevant_urls)
                        ),
                        "retrieved_titles": result.get(
                            "retrieved_titles", []
                        ),
                        "missed_urls": missed_urls,
                    }
                )

    # Print the saved summary to cross-check the original benchmark.
    print("\n" + "=" * 90)
    print("SAVED BENCHMARK SUMMARY")
    print("=" * 90)
    print(
        f"{'Method':<22} {'Recall@5':>10} "
        f"{'MRR':>10} {'Mean ms':>12} {'p95 ms':>12}"
    )

    summary = data.get("summary", {})

    for method in METHODS:
        stats = summary.get(method, {})
        print(
            f"{method:<22} "
            f"{stats.get('mean_recall_at_5', 0):>10.3f} "
            f"{stats.get('mean_reciprocal_rank', 0):>10.3f} "
            f"{stats.get('mean_latency_ms', 0):>12.1f} "
            f"{stats.get('p95_latency_ms', 0):>12.1f}"
        )

    # Inspect individual queries that did not retrieve every labelled URL.
    for method in METHODS:
        method_failures = failures[method]

        print("\n" + "=" * 90)
        print(f"METHOD: {method}")
        print(
            f"Queries with Recall@5 below 1.0: "
            f"{len(method_failures)}/{query_counts[method]}"
        )

        if not method_failures:
            print("No incomplete-recall queries found.")
            continue

        method_failures.sort(
            key=lambda item: (item["recall"], item["mrr"])
        )

        for item in method_failures:
            print("\n" + "-" * 80)
            print(f"Query ID: {item['id']}")
            print(f"Query: {item['query']}")
            print(
                f"Recall@5: {item['recall']:.3f} "
                f"({item['relevant_found']}/{item['relevant_total']})"
            )
            print(f"Reciprocal rank: {item['mrr']:.3f}")

            print("\nRetrieved documents:")
            for rank, title in enumerate(
                item["retrieved_titles"], start=1
            ):
                print(f"  {rank}. {title}")

            print("\nLabelled relevant URLs missed:")
            for url in item["missed_urls"]:
                print(f"  - {url}")

    print("\n" + "=" * 90)
    print("Analysis complete.")
    print("=" * 90)


if __name__ == "__main__":
    main()

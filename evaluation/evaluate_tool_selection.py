
import argparse
import json
import logging
import time
from pathlib import Path

from src.agent.agent import create_enterprise_agent
from src.tools.sql_tool import initialize_database
from src.tools.web_search_tool import reset_search_budget

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BENCHMARK_FILE = PROJECT_ROOT / "evaluation" / "tool_selection_benchmark.json"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
RESULTS_FILE = RESULTS_DIR / "tool_selection_results.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def extract_tool_calls(agent_result):
    """Extract unique tool names selected by the agent."""

    selected_tools = []

    for message in agent_result.get("messages", []):
        tool_calls = getattr(message, "tool_calls", None)

        if tool_calls is None and isinstance(message, dict):
            tool_calls = message.get("tool_calls", [])

        for tool_call in tool_calls or []:
            name = tool_call.get("name")

            if name:
                selected_tools.append(name)

    return list(dict.fromkeys(selected_tools))


def calculate_metrics(results):
    """Calculate aggregate tool-selection metrics."""

    valid_results = [
        result for result in results
        if result.get("status") == "success"
    ]

    if not valid_results:
        return {
            "evaluated_queries": 0,
            "exact_matches": 0,
            "exact_match_accuracy": None,
            "precision": None,
            "recall": None,
            "f1_score": None,
        }

    exact_matches = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for result in valid_results:
        expected = set(result["expected_tools"])
        actual = set(result["actual_tools"])

        if expected == actual:
            exact_matches += 1

        true_positives += len(expected & actual)
        false_positives += len(actual - expected)
        false_negatives += len(expected - actual)

    # Precision and recall are undefined when there are no
    # positive tool predictions or expected tool selections.
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives > 0 else None
    )

    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives > 0 else None
    )

    if precision is None or recall is None:
        f1_score = None
    elif precision + recall == 0:
        f1_score = 0
    else:
        f1_score = (
            2 * precision * recall / (precision + recall)
        )

    return {
        "evaluated_queries": len(valid_results),
        "exact_matches": exact_matches,
        "exact_match_accuracy": round(
            exact_matches / len(valid_results), 4
        ),
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "f1_score": round(f1_score, 4) if f1_score is not None else None,
    }


def load_results():
    """Load previous results to support resuming."""

    if not RESULTS_FILE.exists():
        return []

    with open(RESULTS_FILE, "r", encoding="utf-8") as file:
        saved_data = json.load(file)

    results = saved_data.get("results", [])

    logger.info("Loaded %d previously saved results.", len(results))

    return results


def save_results(results):
    """Persist cumulative results after each query."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Keep only the latest result for each benchmark ID.
    unique_results = {
        result["id"]: result for result in results
    }

    results = list(unique_results.values())

    output = {
        "metrics": calculate_metrics(results),
        "results": results,
    }

    with open(RESULTS_FILE, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)


def is_quota_error(error):
    """Detect Gemini quota and rate-limit errors."""

    error_text = str(error).lower()

    quota_indicators = [
        "429",
        "quota",
        "resource_exhausted",
        "rate limit",
    ]

    return any(
        indicator in error_text
        for indicator in quota_indicators
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LangChain agent tool selection."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of benchmark cases to select.",
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Zero-based benchmark index to start from.",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=15.0,
        help="Delay between queries in seconds.",
    )

    args = parser.parse_args()

    if args.start_index < 0:
        raise ValueError("Start index cannot be negative.")

    if args.delay < 0:
        raise ValueError("Delay cannot be negative.")

    if args.limit is not None and args.limit <= 0:
        raise ValueError("Limit must be greater than zero.")

    with open(BENCHMARK_FILE, "r", encoding="utf-8") as file:
        benchmark = json.load(file)

    if args.start_index >= len(benchmark):
        raise ValueError("Start index is outside the benchmark.")

    test_cases = benchmark[args.start_index:]

    if args.limit is not None:
        test_cases = test_cases[:args.limit]

    results = load_results()

    # A successful query is skipped on subsequent runs.
    completed_ids = {
        result["id"]
        for result in results
        if result.get("status") == "success"
    }

    pending_cases = [
        case for case in test_cases
        if case["id"] not in completed_ids
    ]

    logger.info("Total benchmark cases: %d", len(benchmark))
    logger.info("Cases selected for this run: %d", len(test_cases))
    logger.info("Already completed successfully: %d", len(completed_ids))
    logger.info("Remaining cases in this run: %d", len(pending_cases))

    if not pending_cases:
        logger.info("All selected queries have already been evaluated.")
        save_results(results)
        return

    # Initialize the database before evaluating.
    initialize_database()

    # Create the existing LangChain agent.
    agent = create_enterprise_agent()

    for index, test_case in enumerate(pending_cases, start=1):
        query_id = test_case["id"]
        query = test_case["query"]
        expected_tools = test_case["expected_tools"]

        logger.info(
            "[%d/%d] Evaluating %s",
            index,
            len(pending_cases),
            query_id,
        )

        last_state=None
        actual_tools=[]
        reset_search_budget()

        try:
            for state in agent.stream(
                {
                    "messages":[
                        {"role":"user","content":query}
                    ]
                },
                config={"recursion_limit":12},
                stream_mode="values",
            ):
                last_state=state
            response = last_state or {}

            actual_tools = extract_tool_calls(response)

            result = {
                "id": query_id,
                "category": test_case["category"],
                "query": query,
                "expected_tools": expected_tools,
                "actual_tools": actual_tools,
                "exact_match": (
                    set(expected_tools) == set(actual_tools)
                ),
                "status": "success",
            }

            logger.info(
                "%s | Expected: %s | Actual: %s | Match: %s",
                query_id,
                expected_tools,
                actual_tools,
                result["exact_match"],
            )


        except Exception as error:
            logger.exception("Evaluation failed for %s", query_id)

            # Extract tool calls from the last available agent state.
            if last_state:
                actual_tools = extract_tool_calls(last_state)

                logger.error(
                    "Agent stopped after %d messages.",
                    len(last_state.get("messages", [])),
                )

                # Print the execution trace.
                for message in last_state.get("messages", []):
                    tool_calls = getattr(message, "tool_calls", [])

                    for tool_call in tool_calls:
                        logger.error(
                            "TRACE - Selected tool: %s | Arguments: %s",
                            tool_call.get("name", "unknown"),
                            tool_call.get("args", {}),
                        )

                    if getattr(message, "type", "") == "tool":
                        logger.error(
                            "TRACE - Executed tool: %s | Result: %s",
                            getattr(message, "name", "unknown"),
                            str(message.content)[:300],
                        )

            result = {
                "id": query_id,
                "category": test_case["category"],
                "query": query,
                "expected_tools": expected_tools,
                "actual_tools": actual_tools,
                "exact_match": False,
                "status": "error",
                "error": str(error),
            }

            results = [
                existing for existing in results
                if existing["id"] != query_id
            ]
            results.append(result)
            save_results(results)

            if is_quota_error(error):
                logger.error(
                    "Gemini quota exhausted. Evaluation stopped safely."
                )
                break

            time.sleep(args.delay)
            continue

        results = [
            existing for existing in results
            if existing["id"] != query_id
        ]
        results.append(result)

        # Save immediately after every successful query.
        save_results(results)

        time.sleep(args.delay)

    # Print metrics across all saved results, not just this batch.
    final_metrics = calculate_metrics(results)

    print("\n" + "=" * 60)
    print("CUMULATIVE TOOL SELECTION EVALUATION")
    print("=" * 60)

    for metric, value in final_metrics.items():
        print(f"{metric}: {value}")

    print(f"\nTotal saved test cases: {len(results)}")
    print(f"Results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()

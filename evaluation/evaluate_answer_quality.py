"""Evaluate answer quality for the Enterprise Knowledge Platform."""

import json
import logging
import re
import time

from pathlib import Path

from src.generation.answer_generator import generate_answer
from src.generation.answer_generator import create_llm

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BENCHMARK_PATH = (
    PROJECT_ROOT / "evaluation" / "answer_quality_benchmark.json"
)

RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

RESULTS_PATH = RESULTS_DIR / "answer_quality_results_v2.json"


logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def load_benchmark() -> list[dict]:
    """Load the answer-quality benchmark."""

    with BENCHMARK_PATH.open("r", encoding="utf-8") as file:
        benchmark = json.load(file)

    if not isinstance(benchmark, list):
        raise ValueError("Benchmark must be a JSON list.")

    return benchmark

def evaluate_with_llm_judge(record: dict) -> dict:
    """Evaluate predefined atomic claims using an LLM judge."""

    llm = create_llm()

    prompt = f"""
You are an independent evaluator assessing an enterprise RAG system.

Evaluate the generated answer using ONLY the supplied benchmark
claims and retrieved source evidence.

QUESTION:
{record["query"]}

REFERENCE ANSWER:
{record["reference_answer"]}

PREDEFINED ATOMIC CLAIMS:
{json.dumps(record["atomic_claims"], indent=2)}

GENERATED ANSWER:
{record["generated_answer"]}

RETRIEVED SOURCES AND EVIDENCE:
{json.dumps(record["sources"], indent=2)}

EVALUATION INSTRUCTIONS:

Evaluate EVERY predefined atomic claim independently.

Do NOT create, remove, merge, split, or rewrite claims.

For each claim, determine:

1. evidence_supported:
   Is the claim explicitly supported by the retrieved evidence?

2. answer_covered:
   Does the generated answer correctly communicate this claim?

3. citation_supported:
   Does the source cited alongside the claim actually support it?
   A missing citation means false.

4. explanation:
   Briefly explain the evaluation using the supplied evidence.

IMPORTANT RULES:

- Preserve every atomic claim exactly as supplied.
- Return exactly one evaluation for every input claim.
- Do not assume that a retrieved URL contains the expected fact.
- Inspect the actual retrieved evidence text.
- Do not use outside knowledge.
- Do not confuse transition timing with minimum retention periods.
- A claim may be supported by evidence even if the answer
  does not communicate it.
- A claim may be communicated in the answer but lack evidence.
- Explicitly acknowledging missing information is not a hallucination.
- Be strict and evidence-based.

Also evaluate:

5. relevance_score:
   Rate answer relevance from 1 to 5.

6. faithfulness_score:
   Rate how faithfully the answer follows retrieved evidence,
   from 1 to 5. Unsupported factual assertions should reduce
   the score.

Return ONLY valid JSON in this exact structure:

{{
  "atomic_claim_evaluations": [
    {{
      "claim": "Exact original atomic claim",
      "evidence_supported": true,
      "answer_covered": true,
      "citation_supported": true,
      "explanation": "Reason"
    }}
  ],
  "relevance_score": 1,
  "faithfulness_score": 1,
  "overall_assessment": "Brief assessment"
}}

VALIDATION RULES:

- Return exactly {len(record["atomic_claims"])} claim evaluations.
- Preserve the original claim text exactly.
- All three evaluation fields must be Boolean values.
- Scores must be integers from 1 to 5.
- Do not include Markdown code fences.
- Return valid JSON only.
"""

    response = llm.invoke(prompt)

    content = response.content

    if not isinstance(content, str):
        content = "\n".join(
            block["text"]
            for block in content
            if isinstance(block, dict) and "text" in block
        )

    try:
        evaluation = json.loads(content)

    except json.JSONDecodeError as first_error:

        logger.warning(
            "Direct JSON parsing failed: %s. Attempting extraction.",
            first_error
        )

        match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            content,
            re.DOTALL
        )

        if not match:
            match = re.search(
                r"\{.*\}",
                content,
                re.DOTALL
            )

        if not match:
            logger.error("LLM judge raw response:\n%s", content)
            raise ValueError("Judge did not return valid JSON.")

        try:
            evaluation = json.loads(match.group(1) if match.lastindex else match.group())

        except json.JSONDecodeError:
            logger.error("LLM judge returned malformed JSON:\n%s", content)
            raise

    required_keys = {
        "atomic_claim_evaluations",
        "relevance_score",
        "faithfulness_score",
        "overall_assessment",
    }

    if not required_keys.issubset(evaluation):
        raise ValueError("Judge response is missing required fields.")

    claims = evaluation["atomic_claim_evaluations"]
    expected_claims = record["atomic_claims"]

    if len(claims) != len(expected_claims):
        raise ValueError(
            f"Expected {len(expected_claims)} claim evaluations, "
            f"received {len(claims)}."
        )

    for expected, actual in zip(expected_claims, claims):

        if actual.get("claim") != expected:
            raise ValueError(
                f"Claim mismatch. Expected: {expected}"
            )

        for field in [
            "evidence_supported",
            "answer_covered",
            "citation_supported",
        ]:
            if type(actual.get(field)) is not bool:
                raise ValueError(
                    f"Invalid Boolean value for {field}."
                )

    for field in ["relevance_score", "faithfulness_score"]:
        score = evaluation[field]

        if type(score) is not int or not 1 <= score <= 5:
            raise ValueError(f"Invalid {field}: {score}")

    return evaluation

def evaluate_citations(record: dict) -> dict:
    """Evaluate citation ID validity and citation coverage."""

    answer = record["generated_answer"]
    sources = record["sources"]

    valid_source_ids = {
        source["source_id"]
        for source in sources
    }

    cited_source_ids = set(
        re.findall(r"\[(S\d+)\]", answer)
    )

    valid_citations = cited_source_ids.intersection(
        valid_source_ids
    )

    invalid_citations = cited_source_ids - valid_source_ids

    citation_validity = (
        len(valid_citations) / len(cited_source_ids)
        if cited_source_ids
        else 0.0
    )

    # Count answer sentences, excluding Markdown headings.
    sentences = []

    for line in answer.splitlines():
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        parts = re.split(r"(?<=[.!?])\s+", line)

        sentences.extend(
            part.strip()
            for part in parts
            if part.strip()
        )

    cited_sentences = 0

    for sentence in sentences:
        sentence_ids = set(
            re.findall(r"\[(S\d+)\]", sentence)
        )

        if sentence_ids.intersection(valid_source_ids):
            cited_sentences += 1

    citation_coverage = (
        cited_sentences / len(sentences)
        if sentences
        else 0.0
    )

    return {
        "cited_source_ids": sorted(cited_source_ids),
        "valid_citations": sorted(valid_citations),
        "invalid_citations": sorted(invalid_citations),
        "citation_validity": round(citation_validity, 4),
        "citation_coverage": round(citation_coverage, 4),
        "total_answer_sentences": len(sentences),
        "cited_sentences": cited_sentences,
    }

def calculate_fact_metrics(judge_result: dict) -> dict:
    """Calculate metrics using fixed atomic claims."""

    claims = judge_result["atomic_claim_evaluations"]

    total_claims = len(claims)

    if total_claims == 0:
        raise ValueError("No atomic claims were evaluated.")

    evidence_supported = sum(
        claim["evidence_supported"]
        for claim in claims
    )

    answer_covered = sum(
        claim["answer_covered"]
        for claim in claims
    )

    supported_answers = sum(
        claim["evidence_supported"] and claim["answer_covered"]
        for claim in claims
    )

    answered_claims = [
        claim for claim in claims
        if claim["answer_covered"]
    ]

    citation_supported = sum(
        claim["citation_supported"]
        for claim in answered_claims
    )

    return {
        "total_atomic_claims": total_claims,

        "evidence_support_rate": round(
            evidence_supported / total_claims, 4
        ),

        "answer_coverage_rate": round(
            answer_covered / total_claims, 4
        ),

        "supported_answer_rate": round(
            supported_answers / total_claims, 4
        ),

        "citation_support_rate": round(
            citation_supported / len(answered_claims), 4
        ) if answered_claims else 0.0,
    }

def run_single_question(question: dict) -> dict:
    """Generate an answer and capture execution and retrieval metrics."""

    start_time = time.perf_counter()

    result = generate_answer(question["query"])

    latency = time.perf_counter() - start_time

    # Extract retrieved URLs
    retrieved_urls = {
        source["url"].rstrip("/")
        for source in result["sources"]
    }

    # Normalize benchmark URLs
    relevant_urls = {
        url.rstrip("/")
        for url in question["relevant_urls"]
    }

    # Calculate Source Recall@5
    relevant_retrieved = retrieved_urls.intersection(relevant_urls)

    source_recall = (
        len(relevant_retrieved) / len(relevant_urls)
        if relevant_urls
        else 0.0
    )

    return {
        "id": question["id"],
        "category": question["category"],
        "question_type": question["question_type"],
        "query": question["query"],
        "reference_answer": question["reference_answer"],
        "expected_facts": question["expected_facts"],
        "atomic_claims": question["atomic_claims"],
        "relevant_urls": question["relevant_urls"],
        "generated_answer": result["answer"],
        "sources": result["sources"],
        "retrieved_urls": list(retrieved_urls),
        "source_recall_at_5": round(source_recall, 4),
        "latency_seconds": round(latency, 4),
    }


def main() -> None:
    """Resume answer-quality evaluation without repeating successful questions."""

    benchmark = load_benchmark()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Validate benchmark before making API calls.
    if len(benchmark) != 100:
        raise ValueError(
            f"Expected 100 questions, found {len(benchmark)}."
        )

    for question in benchmark:
        claims = question.get("atomic_claims")

        if not isinstance(claims, list) or not claims:
            raise ValueError(
                f"Missing atomic_claims for {question['id']}"
            )

        if not all(isinstance(c, str) and c.strip() for c in claims):
            raise ValueError(
                f"Invalid atomic_claims for {question['id']}"
            )

    # Load previous results, if available.
    if RESULTS_PATH.exists():
        with RESULTS_PATH.open("r", encoding="utf-8") as file:
            previous_results = json.load(file)

        if not isinstance(previous_results, list):
            raise ValueError("Existing results file must contain a JSON list.")

        results_by_id = {
            item["id"]: item
            for item in previous_results
            if isinstance(item, dict) and "id" in item
        }

        print(f"\nLoaded {len(results_by_id)} previous results.")

    else:
        results_by_id = {}

    benchmark_ids = {question["id"] for question in benchmark}

    # Avoid accidentally carrying unrelated records into this evaluation.
    unknown_ids = set(results_by_id) - benchmark_ids

    if unknown_ids:
        raise ValueError(
            f"Existing results contain unknown question IDs: {unknown_ids}"
        )

    completed_ids = {
        question_id
        for question_id, result in results_by_id.items()
        if result.get("evaluation_status") == "success"
    }

    pending_questions = [
        question
        for question in benchmark
        if question["id"] not in completed_ids
    ]

    print("\n" + "=" * 75)
    print("RESUMABLE ANSWER QUALITY EVALUATION")
    print("=" * 75)
    print("Total questions:", len(benchmark))
    print("Already completed:", len(completed_ids))
    print("Remaining to evaluate:", len(pending_questions))

    if not pending_questions:
        print("\nAll questions have already been evaluated successfully.")

    for index, question in enumerate(pending_questions, start=1):

        question_id = question["id"]

        print("\n" + "-" * 75)
        print(f"Pending question {index}/{len(pending_questions)}: {question_id}")
        print("-" * 75)

        try:
            result = run_single_question(question)

            judge_result = evaluate_with_llm_judge(result)

            result["judge_evaluation"] = judge_result

            fact_metrics = calculate_fact_metrics(judge_result)
            result["fact_metrics"] = fact_metrics

            citation_result = evaluate_citations(result)
            result["citation_evaluation"] = citation_result

            result["evaluation_status"] = "success"

            print("Latency:", result["latency_seconds"], "seconds")
            print("Source Recall@5:", result["source_recall_at_5"])
            print("Evidence support rate:",
                  fact_metrics["evidence_support_rate"])
            print("Answer coverage rate:",
                  fact_metrics["answer_coverage_rate"])
            print("Supported answer rate:",
                  fact_metrics["supported_answer_rate"])
            print("Citation support rate:",
                  fact_metrics["citation_support_rate"])
            print("Relevance:", judge_result["relevance_score"], "/ 5")
            print("Faithfulness:",
                  judge_result["faithfulness_score"], "/ 5")

        except Exception as error:

            logger.exception(
                "Evaluation failed for question %s",
                question_id
            )

            result = {
                "id": question_id,
                "category": question["category"],
                "question_type": question["question_type"],
                "query": question["query"],
                "evaluation_status": "failed",
                "error": str(error)
            }

        # Update only this question's result.
        results_by_id[question_id] = result

        # Rebuild results in original benchmark order.
        ordered_results = [
            results_by_id[question["id"]]
            for question in benchmark
            if question["id"] in results_by_id
        ]

        # Save after every question.
        with RESULTS_PATH.open("w", encoding="utf-8") as file:
            json.dump(ordered_results, file, indent=2, ensure_ascii=False)

        successful = sum(
            item["evaluation_status"] == "success"
            for item in ordered_results
        )

        failed = sum(
            item["evaluation_status"] == "failed"
            for item in ordered_results
        )

        print(f"Progress: {index}/{len(pending_questions)} pending questions")
        print(f"Total successful: {successful}")
        print(f"Total failed: {failed}")

    # Calculate final metrics using ALL successful results.
    final_results = [
        results_by_id[question["id"]]
        for question in benchmark
        if question["id"] in results_by_id
    ]

    successful_results = [
        item for item in final_results
        if item["evaluation_status"] == "success"
    ]

    failed_results = [
        item for item in final_results
        if item["evaluation_status"] == "failed"
    ]

    print("\n" + "=" * 75)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 75)

    print("Total benchmark questions:", len(benchmark))
    print("Results recorded:", len(final_results))
    print("Successful:", len(successful_results))
    print("Failed:", len(failed_results))

    if successful_results:

        metric_names = [
            "source_recall_at_5",
            "evidence_support_rate",
            "answer_coverage_rate",
            "supported_answer_rate",
            "citation_support_rate"
        ]

        for metric in metric_names:

            values = [
                item["source_recall_at_5"]
                if metric == "source_recall_at_5"
                else item["fact_metrics"][metric]
                for item in successful_results
            ]

            print(
                f"Average {metric}: "
                f"{sum(values) / len(values):.4f}"
            )

        avg_latency = sum(
            item["latency_seconds"]
            for item in successful_results
        ) / len(successful_results)

        avg_relevance = sum(
            item["judge_evaluation"]["relevance_score"]
            for item in successful_results
        ) / len(successful_results)

        avg_faithfulness = sum(
            item["judge_evaluation"]["faithfulness_score"]
            for item in successful_results
        ) / len(successful_results)

        print(f"Average latency: {avg_latency:.2f} seconds")
        print(f"Average relevance: {avg_relevance:.2f}/5")
        print(f"Average faithfulness: {avg_faithfulness:.2f}/5")

    if failed_results:
        print("\nFailed question IDs:")
        for item in failed_results:
            print("-", item["id"], ":", item.get("error", "Unknown error"))

    print("\nResults saved to:", RESULTS_PATH)


if __name__ == "__main__":
    main()
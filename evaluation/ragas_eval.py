"""RAGAS evaluation script — measures context precision, answer relevance, and faithfulness.

Usage:
    python evaluation/ragas_eval.py --dataset evaluation/benchmark.json

Benchmark dataset format:
    [{"question": "...", "ground_truth": "..."}, ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from loguru import logger


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_benchmark(path: Path) -> list[dict[str, str]]:
    """Load a JSON benchmark file and return list of {question, ground_truth}."""
    with open(path) as f:
        data = json.load(f)
    assert isinstance(data, list), "Benchmark must be a JSON array."
    return data


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline_on_dataset(
    dataset: list[dict[str, str]],
) -> tuple[list[str], list[str], list[list[str]]]:
    """
    Run the RAG pipeline on every question in *dataset*.

    Returns
    -------
    questions, answers, contexts
        Parallel lists suitable for RAGAS evaluation.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from app.rag_pipeline import get_pipeline

    pipeline = get_pipeline()

    questions: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []

    for item in dataset:
        question = item["question"]
        logger.info(f"Evaluating: {question[:80]}")
        try:
            result = pipeline.run(question)
            questions.append(question)
            answers.append(result.answer)
            contexts.append([c.content for c in result.retrieved_chunks])
        except Exception as exc:
            logger.error(f"Pipeline error for '{question}': {exc}")
            questions.append(question)
            answers.append("ERROR")
            contexts.append([])

    return questions, answers, contexts


# ---------------------------------------------------------------------------
# RAGAS evaluation
# ---------------------------------------------------------------------------

def evaluate_with_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict[str, float]:
    """
    Compute RAGAS metrics and return a summary dict.

    Requires the ``ragas`` package and an OpenAI API key (used by RAGAS
    internally to judge relevance / faithfulness).
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            faithfulness,
        )
    except ImportError as exc:
        raise ImportError(
            "Install evaluation dependencies: pip install ragas datasets"
        ) from exc

    ragas_dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    logger.info("Running RAGAS evaluation …")
    result = evaluate(
        ragas_dataset,
        metrics=[context_precision, answer_relevancy, faithfulness],
    )
    scores: dict[str, float] = result.to_pandas()[
        ["context_precision", "answer_relevancy", "faithfulness"]
    ].mean().to_dict()

    return scores


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(scores: dict[str, float]) -> None:
    """Pretty-print the evaluation scores."""
    print("\n" + "=" * 50)
    print("  DocuRAG — RAGAS Evaluation Report")
    print("=" * 50)
    for metric, score in scores.items():
        bar = "█" * int(score * 20)
        print(f"  {metric:<25} {score:.4f}  {bar}")
    print("=" * 50 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    dataset = load_benchmark(args.dataset)
    ground_truths = [item.get("ground_truth", "") for item in dataset]

    questions, answers, contexts = run_pipeline_on_dataset(dataset)

    scores = evaluate_with_ragas(questions, answers, contexts, ground_truths)

    print_report(scores)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(scores, indent=2))
        logger.info(f"Scores saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on DocuRAG")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).parent / "benchmark.json",
        help="Path to the benchmark JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save JSON scores",
    )
    main(parser.parse_args())

#!/usr/bin/env python3
"""Quick end-to-end test with Mock Embedder.

This script provides a fast way to validate the mempy evaluation framework
using a mock embedder (zero cost, no API calls needed).

Usage:
    # Activate conda environment first
    conda activate mempy

    # Run test from project root
    python tests/benchmarks/locomo/test_e2e_quick.py
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from tests.benchmarks.adapters import MockEmbedder
from tests.benchmarks.locomo import LOCOMODataset, LOCOMOEvaluator


async def main() -> int:
    """Run quick end-to-end test."""
    print("=" * 60)
    print("mempy Quick End-to-End Test")
    print("=" * 60)

    print(f"\n{'=' * 60}")
    print("Configuration")
    print(f"{'=' * 60}")
    print(f"  Embedder: Mock (deterministic hash-based)")
    print(f"  Test size: 5 QA pairs")
    print(f"  Dataset: LOCOMO")
    print(f"  Token cost: 0 (no API calls)")

    # Load dataset
    data_path = "tests/benchmarks/data/locomo10.json"
    print(f"\n{'=' * 60}")
    print("Loading Dataset")
    print(f"{'=' * 60}")
    print(f"  Path: {data_path}")

    if not Path(data_path).exists():
        print(f"\n✗ Error: Dataset file not found: {data_path}")
        print(f"  Please run this script from the project root directory.")
        return 1

    dataset = LOCOMODataset(data_path)
    dataset.load()
    print(f"  Total conversations: {dataset.num_conversations}")
    print(f"  Total QA pairs: {dataset.num_qa_pairs}")

    # Create embedder and evaluator
    embedder = MockEmbedder()
    evaluator = LOCOMOEvaluator(
        dataset=dataset,
        embedder=embedder,
        verbose=True
    )

    # Run evaluation with limited QA pairs
    print(f"\n{'=' * 60}")
    print("Running Evaluation (5 QA pairs)")
    print(f"{'=' * 60}\n")

    try:
        results = await evaluator.evaluate(limit=5, max_qa_pairs=5)
    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Print results
    print(f"\n{'=' * 60}")
    print("Test Results")
    print(f"{'=' * 60}")
    evaluator.print_results(results)

    # Print summary
    num_evaluated = results.get("num_qa_pairs_evaluated", 0)
    qa_acc = results.get("qa", {}).get("qa_accuracy", 0)
    recall_5 = results.get("retrieval", {}).get("recall_at_5", 0)

    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    print(f"  QA pairs evaluated: {num_evaluated}")
    print(f"  QA accuracy: {qa_acc:.2%}")
    print(f"  Recall@5: {recall_5:.2%}")
    print(f"\n  Note: Low accuracy is expected with mock embeddings.")
    print(f"  This test validates the framework, not the model quality.")
    print(f"{'=' * 60}")
    print("✓ Test completed successfully!")
    print(f"{'=' * 60}\n")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

#!/usr/bin/env python3
"""End-to-end test with Zhipu AI Embedder.

This script provides a flexible way to validate the mempy evaluation framework
using Zhipu AI's embedding API with different test modes.

Usage:
    # Quick test (1 QA pair)
    python tests/benchmarks/locomo/test_e2e_zhipu.py --quick

    # Standard test (5 QA pairs, default)
    python tests/benchmarks/locomo/test_e2e_zhipu.py

    # Full test (all QA pairs)
    python tests/benchmarks/locomo/test_e2e_zhipu.py --full

    # Debug mode (verbose logging, 1 QA pair)
    python tests/benchmarks/locomo/test_e2e_zhipu.py --debug

    # Custom number of QA pairs
    python tests/benchmarks/locomo/test_e2e_zhipu.py --qa-pairs 10
"""

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from tests.benchmarks.adapters import ZhipuEmbedder
from tests.benchmarks.locomo import LOCOMODataset, LOCOMOEvaluator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="mempy End-to-End Test with Zhipu AI"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test with 1 QA pair"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full test with all QA pairs"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode with verbose logging and 1 QA pair"
    )
    parser.add_argument(
        "--qa-pairs",
        type=int,
        default=None,
        help="Custom number of QA pairs to test"
    )
    parser.add_argument(
        "--no-verbose",
        action="store_true",
        help="Disable verbose output"
    )

    return parser.parse_args()


async def main() -> int:
    """Run end-to-end test with Zhipu AI embeddings."""
    args = parse_args()

    # Determine test mode
    if args.debug:
        mode = "DEBUG"
        max_qa_pairs = 1
        verbose = True
    elif args.quick:
        mode = "QUICK"
        max_qa_pairs = 1
        verbose = False
    elif args.full:
        mode = "FULL"
        max_qa_pairs = None  # Test all
        verbose = True
    elif args.qa_pairs:
        mode = "CUSTOM"
        max_qa_pairs = args.qa_pairs
        verbose = not args.no_verbose
    else:
        mode = "STANDARD"
        max_qa_pairs = 5
        verbose = True

    print("=" * 70)
    print(f"mempy End-to-End Test (Zhipu AI) - {mode} Mode")
    print("=" * 70)

    # Configuration
    api_key = os.environ.get("ZHIPUAI_API_KEY")
    if not api_key:
        print("\n✗ Error: ZHIPUAI_API_KEY environment variable not set")
        print("  Please set it using:")
        print("    export ZHIPUAI_API_KEY=your-api-key-here")
        print("  Or create a .env file from .env.example:")
        print("    cp tests/benchmarks/.env.example tests/benchmarks/.env")
        print("    # Edit tests/benchmarks/.env and add your API key")
        return 1

    model = "embedding-3"
    dimension = 1024

    print(f"\n{'=' * 70}")
    print("Configuration")
    print(f"{'=' * 70}")
    print(f"  Embedder: Zhipu AI ({model})")
    print(f"  Dimension: {dimension}")
    print(f"  Mode: {mode}")
    if max_qa_pairs is None:
        print(f"  Test size: All QA pairs (full evaluation)")
        print(f"  Estimated time: 20-30 minutes")
        print(f"  Estimated cost: ~¥0.15-0.20")
    else:
        print(f"  Test size: {max_qa_pairs} QA pair(s)")
    print(f"  Verbose: {verbose}")
    print(f"  Dataset: LOCOMO")

    # Load dataset
    data_path = "tests/benchmarks/data/locomo10.json"
    print(f"\n{'=' * 70}")
    print("Loading Dataset")
    print(f"{'=' * 70}")
    print(f"  Path: {data_path}")

    if not Path(data_path).exists():
        print(f"\n✗ Error: Dataset file not found: {data_path}")
        print(f"  Please run this script from the project root directory.")
        return 1

    dataset = LOCOMODataset(data_path)
    dataset.load()
    print(f"  Total conversations: {dataset.num_conversations}")
    print(f"  Total QA pairs: {dataset.num_qa_pairs}")

    # Debug mode: Test embedder first
    if args.debug:
        print(f"\n{'=' * 70}")
        print("Test 1: Verify Embedder")
        print(f"{'=' * 70}")

        embedder_test = ZhipuEmbedder(
            api_key=api_key,
            model=model,
            dimension=dimension,
        )
        test_text = "Hello, this is a test."
        print(f"  Testing embedding generation...")
        print(f"  Input: '{test_text}'")

        try:
            embedding = await embedder_test.embed(test_text)
            print(f"  ✓ Embedding generated successfully")
            print(f"  Vector length: {len(embedding)}")
            print(f"  First 5 values: {embedding[:5]}")
        except Exception as e:
            print(f"  ✗ Embedding generation failed: {e}")
            import traceback
            traceback.print_exc()
            await embedder_test.close()
            return 1

        await embedder_test.close()
        print()

    # Create embedder
    print(f"{'=' * 70}")
    print("Initializing ZhipuAI Embedder")
    print(f"{'=' * 70}")
    embedder = ZhipuEmbedder(
        api_key=api_key,
        model=model,
        dimension=dimension,
    )
    print(f"  Model: {model}")
    print(f"  Dimension: {dimension}")
    print(f"  ✓ Embedder ready")

    # Create evaluator
    evaluator = LOCOMOEvaluator(
        dataset=dataset,
        embedder=embedder,
        verbose=verbose
    )

    # Run evaluation
    if max_qa_pairs is None:
        print(f"\n{'=' * 70}")
        print(f"Running Full Evaluation (All {dataset.num_qa_pairs} QA pairs)")
        print(f"{'=' * 70}\n")
    else:
        print(f"\n{'=' * 70}")
        print(f"Running Evaluation ({max_qa_pairs} QA pair(s))")
        print(f"{'=' * 70}\n")

    start_time = time.time()

    try:
        results = await evaluator.evaluate(limit=10, max_qa_pairs=max_qa_pairs)
    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        await embedder.close()
        return 1

    elapsed_time = time.time() - start_time

    # Print results
    print(f"\n{'=' * 70}")
    print("Test Results")
    print(f"{'=' * 70}")
    evaluator.print_results(results)

    # Print token usage and cost
    if hasattr(embedder, 'total_tokens') and embedder.total_tokens > 0:
        print(f"\n{'=' * 70}")
        print("Token Usage & Cost")
        print(f"{'=' * 70}")
        print(f"  Total tokens: {embedder.total_tokens:,}")
        # 智谱AI定价约¥0.0007/1K tokens
        estimated_cost = embedder.total_tokens * 0.0007 / 1000
        print(f"  Estimated cost: ~¥{estimated_cost:.4f}")
        if max_qa_pairs and max_qa_pairs > 0:
            print(f"  Tokens per QA pair: {embedder.total_tokens / max_qa_pairs:.0f}")

    # Print performance summary
    print(f"\n{'=' * 70}")
    print("Performance Summary")
    print(f"{'=' * 70}")
    print(f"  Execution time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    if max_qa_pairs:
        print(f"  Time per QA pair: {elapsed_time / max_qa_pairs:.1f} seconds")

    # Print metrics
    num_evaluated = results.get("num_qa_pairs_evaluated", max_qa_pairs or dataset.num_qa_pairs)
    qa_acc = results.get("qa", {}).get("qa_accuracy", 0)
    qa_acc_strict = results.get("qa", {}).get("qa_accuracy_strict", None)
    qa_f1 = results.get("qa", {}).get("qa_f1", None)
    recall_5 = results.get("retrieval", {}).get("recall_at_5", 0)
    precision_5 = results.get("retrieval", {}).get("precision_at_5", 0)

    print(f"\n{'=' * 70}")
    print("Key Metrics")
    print(f"{'=' * 70}")
    print(f"  QA pairs evaluated: {num_evaluated}")
    print(f"  QA Accuracy: {qa_acc:.2%}")
    if qa_acc_strict is not None:
        print(f"  QA Accuracy (strict): {qa_acc_strict:.2%}")
    if qa_f1 is not None:
        print(f"  QA F1 Score: {qa_f1:.3f}")
    print(f"  Recall@5: {recall_5:.2%}")
    print(f"  Precision@5: {precision_5:.3f}")

    # Print guidance based on results
    print(f"\n{'=' * 70}")
    print("Analysis")
    print(f"{'=' * 70}")

    if recall_5 > 0.5:
        print(f"  ✓ Excellent retrieval performance (Recall@5 > 50%)")
    elif recall_5 > 0.3:
        print(f"  ✓ Good retrieval performance (Recall@5 > 30%)")
    elif recall_5 > 0:
        print(f"  ⚠ Moderate retrieval performance - consider tuning parameters")
    else:
        print(f"  ✗ Poor retrieval performance - check debug output above")

    if qa_acc > 0.6:
        print(f"  ✓ Good QA accuracy (> 60%)")
    elif qa_acc > 0.4:
        print(f"  ⚠ Moderate QA accuracy - this is expected with simple retrieval")
    elif qa_acc > 0:
        print(f"  ⚠ Lower QA accuracy - consider improving retrieval quality")
    else:
        print(f"  ✗ Very low QA accuracy - review configuration and data")

    print(f"\n{'=' * 70}")
    print(f"✓ Test completed successfully! ({mode} Mode)")
    print(f"{'=' * 70}")

    # Provide next steps guidance
    print(f"\nNext steps:")
    print(f"  - Compare with mock embedder for baseline")
    print(f"  - Try different modes: --quick, --full, --debug")
    print(f"  - Adjust QA pairs: --qa-pairs N")
    print(f"\n")

    await embedder.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

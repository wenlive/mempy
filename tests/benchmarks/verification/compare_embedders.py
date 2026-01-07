#!/usr/bin/env python3
"""Compare mock vs Zhipu AI embedder performance on LOCOMO benchmark."""

import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from tests.benchmarks.adapters.mock import MockEmbedder
from tests.benchmarks.adapters.zhipu import ZhipuEmbedder
from tests.benchmarks.locomo import LOCOMODataset, LOCOMOEvaluator


async def test_embedder(name: str, embedder, max_qa_pairs: int = 20):
    """Test a single embedder."""
    print(f"\n{'=' * 70}")
    print(f"Testing: {name}")
    print(f"{'=' * 70}")

    data_path = "tests/benchmarks/data/locomo10.json"
    dataset = LOCOMODataset(data_path)
    dataset.load()

    evaluator = LOCOMOEvaluator(
        dataset=dataset,
        embedder=embedder,
        verbose=False  # Disable verbose for cleaner output
    )

    start_time = time.time()
    results = await evaluator.evaluate(limit=10, max_qa_pairs=max_qa_pairs)
    elapsed_time = time.time() - start_time

    # Extract key metrics
    retrieval = results.get("retrieval", {})
    qa = results.get("qa", {})

    print(f"\n📊 Results for {name}:")
    print(f"{'=' * 70}")
    print(f"  Test Duration: {elapsed_time:.1f}s")
    print(f"\n  🎯 Retrieval Metrics:")
    print(f"     Recall@1:  {retrieval.get('recall_at_1', 0):.2%}")
    print(f"     Recall@3:  {retrieval.get('recall_at_3', 0):.2%}")
    print(f"     Recall@5:  {retrieval.get('recall_at_5', 0):.2%}")
    print(f"     Precision@5: {retrieval.get('precision_at_5', 0):.3f}")

    # Token usage if available
    if hasattr(embedder, 'total_tokens') and embedder.total_tokens > 0:
        print(f"\n  💰 Token Usage:")
        print(f"     Total Tokens: {embedder.total_tokens:,}")
        cost = embedder.total_tokens * 0.0007 / 1000
        print(f"     Estimated Cost: ~¥{cost:.4f}")

    return results, elapsed_time


async def main():
    """Run comparison between mock and Zhipu embedders."""
    print("=" * 70)
    print("mempy Benchmark Comparison: Mock vs Zhipu AI")
    print("=" * 70)
    print(f"\nTest Configuration:")
    print(f"  Dataset: LOCOMO")
    print(f"  QA Pairs: 20")
    print(f"  Retrieval Limit: 10")

    # Test 1: Mock Embedder
    print("\n" + "=" * 70)
    print("TEST 1: Mock Embedder (Baseline)")
    print("=" * 70)
    mock_embedder = MockEmbedder()
    mock_results, mock_time = await test_embedder("Mock Embedder", mock_embedder, max_qa_pairs=20)

    # Test 2: Zhipu AI Embedder
    print("\n" + "=" * 70)
    print("TEST 2: Zhipu AI Embedder (embedding-3)")
    print("=" * 70)
    api_key = os.environ.get("ZHIPUAI_API_KEY")
    if not api_key:
        print("\n✗ Error: ZHIPUAI_API_KEY environment variable not set")
        print("  Please set it using:")
        print("    export ZHIPUAI_API_KEY=your-api-key-here")
        print("  Or create a .env file from .env.example:")
        print("    cp tests/benchmarks/.env.example tests/benchmarks/.env")
        print("    # Edit tests/benchmarks/.env and add your API key")
        return 1

    zhipu_embedder = ZhipuEmbedder(api_key=api_key, model="embedding-3", dimension=1024)
    zhipu_results, zhipu_time = await test_embedder("Zhipu AI", zhipu_embedder, max_qa_pairs=20)

    await zhipu_embedder.close()

    # Comparison Summary
    print("\n" + "=" * 70)
    print("📊 COMPARISON SUMMARY")
    print("=" * 70)

    mock_recall_5 = mock_results.get("retrieval", {}).get("recall_at_5", 0)
    zhipu_recall_5 = zhipu_results.get("retrieval", {}).get("recall_at_5", 0)

    mock_recall_1 = mock_results.get("retrieval", {}).get("recall_at_1", 0)
    zhipu_recall_1 = zhipu_results.get("retrieval", {}).get("recall_at_1", 0)

    print(f"\n  Metric                 Mock Embedder    Zhipu AI       Improvement")
    print(f"  {'─' * 70}")
    print(f"  Recall@1              {mock_recall_1:.2%}            {zhipu_recall_1:.2%}         +{(zhipu_recall_1 - mock_recall_1):.2%}")
    print(f"  Recall@5              {mock_recall_5:.2%}            {zhipu_recall_5:.2%}         +{(zhipu_recall_5 - mock_recall_5):.2%}")
    print(f"  Execution Time        {mock_time:.1f}s             {zhipu_time:.1f}s         {zhipu_time - mock_time:+.1f}s")

    if hasattr(zhipu_embedder, 'total_tokens') and zhipu_embedder.total_tokens > 0:
        cost = zhipu_embedder.total_tokens * 0.0007 / 1000
        print(f"  Cost                  ¥0.0000            ¥{cost:.4f}")

    print("\n" + "=" * 70)
    print("✅ Comparison completed!")
    print("=" * 70)

    print("\n📝 Key Takeaways:")
    print(f"  • Zhipu AI improves Recall@5 by {(zhipu_recall_5 - mock_recall_5):.2%} over mock")
    print(f"  • Zhipu AI improves Recall@1 by {(zhipu_recall_1 - mock_recall_1):.2%} over mock")
    if zhipu_recall_5 > 0.5:
        print(f"  • ✓ Strong retrieval performance (Recall@5 > 50%)")
    elif zhipu_recall_5 > 0.3:
        print(f"  • ✓ Moderate retrieval performance (Recall@5 > 30%)")
    else:
        print(f"  • ⚠ Lower retrieval performance - consider tuning parameters")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

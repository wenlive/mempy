#!/usr/bin/env python3
"""Example: Run LOCOMO evaluation with Qwen model.

This script demonstrates how to run the LOCOMO benchmark evaluation
using a locally running Qwen model (via vLLM).

Prerequisites:
1. Start vLLM server with Qwen model:
   vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000

2. Install dependencies:
   pip install mempy aiohttp

Usage:
    python run_with_qwen.py --base-url http://localhost:8000
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from tests.benchmarks.adapters.qwen import QwenEmbedder
from tests.benchmarks.locomo import LOCOMODataset, LOCOMOEvaluator


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run LOCOMO evaluation with Qwen model"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the vLLM server (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model name/identifier (default: Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=None,
        help="Embedding dimension (auto-detected if not specified)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="tests/benchmarks/data/locomo10.json",
        help="Path to LOCOMO dataset (default: tests/benchmarks/data/locomo10.json)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Max memories to retrieve per query (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file for results",
    )
    return parser.parse_args()


async def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Check if data file exists
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Error: Data file not found: {args.data_path}")
        print("\nDownload the LOCOMO dataset:")
        print("wget https://github.com/snap-research/locomo/raw/main/data/locomo10.json \\")
        print(f"     -O {args.data_path}")
        return 1

    # Initialize Qwen embedder
    print(f"Connecting to Qwen server at: {args.base_url}")
    print(f"Model: {args.model_name}")

    async with QwenEmbedder(
        base_url=args.base_url,
        model_name=args.model_name,
        dimension=args.dimension,
    ) as embedder:
        print(f"Embedding dimension: {embedder.dimension}\n")

        # Load dataset
        print(f"Loading dataset from: {args.data_path}")
        dataset = LOCOMODataset(args.data_path)
        dataset.load()
        print(f"  Conversations: {dataset.num_conversations}")
        print(f"  QA Pairs: {dataset.num_qa_pairs}\n")

        # Run evaluation
        print("Running evaluation...")
        evaluator = LOCOMOEvaluator(dataset, embedder, verbose=True)
        results = await evaluator.evaluate(limit=args.limit)

        # Print results
        evaluator.print_results(results)

        # Save results
        if args.output:
            evaluator.save_results(results, args.output)

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

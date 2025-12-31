#!/usr/bin/env python3
"""LOCOMO evaluation command-line script.

This script runs the LOCOMO benchmark evaluation on a memory system.

Usage:
    python run_evaluation.py --data-path /path/to/locomo10.json
    python run_evaluation.py --data-path /path/to/locomo10.json --output results.json
    python run_evaluation.py --data-path /path/to/locomo10.json --embedder-module my_embedder
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from mempy.benchmarks.locomo import LOCOMODataset, LOCOMOEvaluator
from mempy.core.interfaces import Embedder


class SimpleMockEmbedder(Embedder):
    """A simple mock embedder for testing purposes.

    This embedder generates deterministic embeddings based on text hash.
    It's only for testing - replace with a real embedder for production use.
    """

    def __init__(self, dimension: int = 768):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, text: str) -> List[float]:
        # Generate a deterministic vector based on text
        hash_val = hash(text)
        return [(hash_val >> i) % 100 / 100.0 for i in range(self._dimension)]


class OpenAIEmbedder(Embedder):
    """OpenAI-based embedder using the API.

    Requires the openai package to be installed.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small"):
        try:
            import openai
            self.openai = openai
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

        self.model = model
        self.client = self.openai.AsyncOpenAI(api_key=api_key)

        # Set dimension based on model
        if "3-small" in model:
            self._dimension = 1536
        elif "3-large" in model:
            self._dimension = 3072
        elif "ada-002" in model:
            self._dimension = 1536
        else:
            self._dimension = 1536

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, text: str) -> List[float]:
        response = await self.client.embeddings.create(
            input=text,
            model=self.model,
        )
        return response.data[0].embedding


def load_embedder_from_module(module_path: str, class_name: str) -> Embedder:
    """Load an embedder class from a Python module.

    Args:
        module_path: Python module path (e.g., "my_package.embedders")
        class_name: Name of the embedder class

    Returns:
        Instantiated embedder
    """
    import importlib

    module = importlib.import_module(module_path)
    embedder_class = getattr(module, class_name)
    return embedder_class()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a memory system on the LOCOMO benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use mock embedder for quick testing
  python run_evaluation.py --data-path locomo10.json

  # Use OpenAI embeddings
  python run_evaluation.py --data-path locomo10.json --embedder openai

  # Use custom embedder from a module
  python run_evaluation.py --data-path locomo10.json --embedder-module my_embedders --embedder-class MyEmbedder

  # Save results to file
  python run_evaluation.py --data-path locomo10.json --output results.json
        """,
    )

    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to LOCOMO JSON file",
    )

    parser.add_argument(
        "--embedder",
        type=str,
        choices=["mock", "openai"],
        default="mock",
        help="Type of embedder to use (default: mock)",
    )

    parser.add_argument(
        "--embedder-module",
        type=str,
        help="Custom embedder module path",
    )

    parser.add_argument(
        "--embedder-class",
        type=str,
        default="Embedder",
        help="Custom embedder class name (default: Embedder)",
    )

    parser.add_argument(
        "--openai-model",
        type=str,
        default="text-embedding-3-small",
        help="OpenAI embedding model (default: text-embedding-3-small)",
    )

    parser.add_argument(
        "--openai-key",
        type=str,
        help="OpenAI API key (defaults to OPENAI_API_KEY env var)",
    )

    parser.add_argument(
        "--dimension",
        type=int,
        default=768,
        help="Embedding dimension for mock embedder (default: 768)",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Max memories to retrieve per query (default: 10)",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output JSON file for results",
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )

    parser.add_argument(
        "--download",
        action="store_true",
        help="Download LOCOMO dataset if not found",
    )

    return parser.parse_args()


def download_locomo(output_path: str) -> None:
    """Download the LOCOMO dataset.

    Args:
        output_path: Where to save the downloaded file
    """
    import urllib.request

    url = "https://github.com/snap-research/locomo/raw/main/data/locomo10.json"
    output_path = Path(output_path)

    print(f"Downloading LOCOMO dataset from {url}...")
    print(f"Saving to {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    urllib.request.urlretrieve(url, output_path)

    print("Download complete!")


async def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Check if data file exists
    data_path = Path(args.data_path)
    if not data_path.exists():
        if args.download:
            download_locomo(args.data_path)
        else:
            print(f"Error: Data file not found: {args.data_path}")
            print("Use --download to download the LOCOMO dataset automatically.")
            return 1

    # Create embedder
    if args.embedder_module:
        print(f"Loading embedder from {args.embedder_module}.{args.embedder_class}")
        embedder = load_embedder_from_module(args.embedder_module, args.embedder_class)
    elif args.embedder == "openai":
        print(f"Using OpenAI embedder with model: {args.openai_model}")
        embedder = OpenAIEmbedder(
            api_key=args.openai_key,
            model=args.openai_model,
        )
    else:
        print(f"Using mock embedder with dimension: {args.dimension}")
        embedder = SimpleMockEmbedder(dimension=args.dimension)

    # Load dataset
    print(f"\nLoading dataset from: {args.data_path}")
    dataset = LOCOMODataset(args.data_path)
    dataset.load()

    print(f"  Conversations: {dataset.num_conversations}")
    print(f"  QA Pairs: {dataset.num_qa_pairs}")

    # Create evaluator
    evaluator = LOCOMOEvaluator(
        dataset=dataset,
        embedder=embedder,
        verbose=not args.quiet,
    )

    # Run evaluation
    print("\nRunning evaluation...")
    results = await evaluator.evaluate(limit=args.limit)

    # Print results
    if not args.quiet:
        evaluator.print_results(results)

    # Save results
    if args.output:
        evaluator.save_results(results, args.output)

    # Exit with success if accuracy > 0
    qa_acc = results.get("qa", {}).get("qa_accuracy", 0)
    return 0 if qa_acc > 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

"""Model adapters for benchmark evaluation.

This package contains adapter implementations for various embedding models
that can be used with the LOCOMO benchmark evaluation framework.

The adapters are intentionally separated from the core mempy code to maintain
clean separation between the evaluation framework and specific model implementations.

Available adapters:
- QwenEmbedder: For Qwen models (qwen3-235b-a22b, qwen3-32b, etc.)
- OpenAIEmbedder: For OpenAI embedding models
- MockEmbedder: For testing without a real model
"""

from tests.benchmarks.adapters.mock import MockEmbedder
from tests.benchmarks.adapters.qwen import QwenEmbedder
from tests.benchmarks.adapters.openai import OpenAIEmbedder

__all__ = [
    "MockEmbedder",
    "QwenEmbedder",
    "OpenAIEmbedder",
]

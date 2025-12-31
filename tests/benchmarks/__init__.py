"""Benchmark evaluation suite for mempy.

This package contains evaluation frameworks for testing memory systems
on standard datasets like LOCOMO.
"""

from tests.benchmarks.locomo import LOCOMODataset, LOCOMOEvaluator

__all__ = [
    "LOCOMODataset",
    "LOCOMOEvaluator",
]

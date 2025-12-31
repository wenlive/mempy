"""LOCOMO benchmark for evaluating long-term conversational memory.

LOCOMO (Long Conversational Memory) is a benchmark dataset from SNAP Research
designed to evaluate very long-term conversational memory in dialogue systems.

Paper: https://arxiv.org/abs/2402.17753
GitHub: https://github.com/snap-research/locomo
"""

from tests.benchmarks.locomo.dataset import (
    LOCOMODataset,
    Conversation,
    Session,
    Turn,
    QA,
)
from tests.benchmarks.locomo.evaluator import LOCOMOEvaluator
from tests.benchmarks.locomo.metrics import recall_at_k, qa_accuracy

__all__ = [
    # Dataset
    "LOCOMODataset",
    "Conversation",
    "Session",
    "Turn",
    "QA",
    # Evaluator
    "LOCOMOEvaluator",
    # Metrics
    "recall_at_k",
    "qa_accuracy",
]

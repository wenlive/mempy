"""
Firmness calculation strategies for memory management.

This module provides strategies for calculating how firmly established
a memory is, which determines its resistance to forgetting.
"""

from abc import ABC, abstractmethod
from typing import Optional
from datetime import datetime, timedelta

from mempy.core.memory import Memory


class FirmnessCalculator(ABC):
    """
    Abstract base class for firmness calculation strategies.

    A firmness calculator evaluates how well-established a memory is
    based on various factors like access patterns, relationships,
    and confidence.
    """

    @abstractmethod
    def calculate(
        self,
        memory: Memory,
        relation_count: int,
        avg_relation_confidence: float
    ) -> float:
        """
        Calculate the firmness score for a memory.

        Args:
            memory: The memory to evaluate
            relation_count: Number of relations connected to this memory
            avg_relation_confidence: Average confidence of related memories

        Returns:
            float: Firmness score in range [0.0, 1.0]
                   Higher values indicate more firmly established memories
        """
        pass


class WeightedFirmnessCalculator(FirmnessCalculator):
    """
    Weighted average firmness calculator.

    Calculates firmness as a weighted combination of:
    - Recency score (30%): How recently the memory was accessed
    - Count score (20%): How many times the memory was accessed
    - Relation score (20%): How many relations the memory has
    - Quality score (20%): Average confidence of related memories
    - Confidence (10%): The memory's own confidence

    Default time window for recency: 7 days
    Default saturation points: 10 accesses, 5 relations
    """

    # Default configuration
    DEFAULT_RECENCY_WINDOW_DAYS = 7
    DEFAULT_COUNT_SATURATION = 10
    DEFAULT_RELATION_SATURATION = 5

    # Default weights (must sum to 1.0)
    DEFAULT_WEIGHTS = {
        "recency": 0.3,
        "count": 0.2,
        "relation": 0.2,
        "quality": 0.2,
        "confidence": 0.1
    }

    def __init__(
        self,
        recency_window_days: int = DEFAULT_RECENCY_WINDOW_DAYS,
        count_saturation: int = DEFAULT_COUNT_SATURATION,
        relation_saturation: int = DEFAULT_RELATION_SATURATION,
        weights: Optional[dict] = None
    ):
        """
        Initialize the weighted firmness calculator.

        Args:
            recency_window_days: Days considered "recent" for recency score
            count_saturation: Access count that achieves maximum score
            relation_saturation: Relation count that achieves maximum score
            weights: Custom weight dict (optional, uses defaults if None)
        """
        self.recency_window_days = recency_window_days
        self.count_saturation = count_saturation
        self.relation_saturation = relation_saturation
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()

        # Validate weights sum to 1.0
        total_weight = sum(self.weights.values())
        if not (0.99 <= total_weight <= 1.01):  # Allow small floating point errors
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

    def calculate(
        self,
        memory: Memory,
        relation_count: int,
        avg_relation_confidence: float
    ) -> float:
        """
        Calculate weighted firmness score.

        Args:
            memory: The memory to evaluate
            relation_count: Number of relations
            avg_relation_confidence: Average confidence of related memories

        Returns:
            float: Firmness score [0.0, 1.0]
        """
        recency_score = self._calculate_recency_score(memory.last_accessed_at)
        count_score = self._calculate_count_score(memory.access_count)
        relation_score = self._calculate_relation_score(relation_count)
        quality_score = avg_relation_confidence
        confidence_score = memory.confidence

        firmness = (
            recency_score * self.weights["recency"] +
            count_score * self.weights["count"] +
            relation_score * self.weights["relation"] +
            quality_score * self.weights["quality"] +
            confidence_score * self.weights["confidence"]
        )

        return max(0.0, min(1.0, firmness))

    def _calculate_recency_score(self, last_accessed_at: Optional[datetime]) -> float:
        """
        Calculate recency score based on last access time.

        Recent accesses (within recency_window_days) get higher scores.
        Never accessed memories get score 0.0.

        Args:
            last_accessed_at: Last access timestamp

        Returns:
            float: Recency score [0.0, 1.0]
        """
        if last_accessed_at is None:
            return 0.0

        days_since_access = (datetime.now() - last_accessed_at).days

        if days_since_access <= 0:
            return 1.0  # Accessed today

        # Linear decay: score = 1.0 - (days / window)
        score = 1.0 - (days_since_access / self.recency_window_days)
        return max(0.0, score)

    def _calculate_count_score(self, access_count: int) -> float:
        """
        Calculate count score based on access frequency.

        More accesses = higher score, saturating at count_saturation.

        Args:
            access_count: Total number of accesses

        Returns:
            float: Count score [0.0, 1.0]
        """
        if access_count <= 0:
            return 0.0

        score = access_count / self.count_saturation
        return min(1.0, score)

    def _calculate_relation_score(self, relation_count: int) -> float:
        """
        Calculate relation score based on number of relations.

        More relations = higher score, saturating at relation_saturation.

        Args:
            relation_count: Number of connected relations

        Returns:
            float: Relation score [0.0, 1.0]
        """
        if relation_count <= 0:
            return 0.0

        score = relation_count / self.relation_saturation
        return min(1.0, score)

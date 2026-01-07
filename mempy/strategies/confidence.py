"""
Confidence evolution strategies for memory management.

This module provides strategies for dynamically adjusting memory confidence
based on references, relations, and time decay.
"""

from abc import ABC, abstractmethod
from typing import Optional
from datetime import datetime

from mempy.core.memory import Memory


class ConfidenceEvolutionStrategy(ABC):
    """
    Abstract base class for confidence evolution strategies.

    A confidence evolution strategy defines how memory confidence changes
    over time based on various factors like being referenced, establishing
    relationships, and temporal decay.
    """

    @abstractmethod
    async def reinforce_on_reference(self, memory: Memory) -> float:
        """
        Calculate confidence increment when memory is referenced.

        Args:
            memory: The memory being referenced

        Returns:
            float: Confidence increment to apply (should be non-negative)
        """
        pass

    @abstractmethod
    async def reinforce_on_relation(self, memory: Memory) -> float:
        """
        Calculate confidence increment when a relation is established.

        Args:
            memory: The memory for which a relation is being created

        Returns:
            float: Confidence increment to apply (should be non-negative)
        """
        pass

    @abstractmethod
    async def decay_over_time(self, memory: Memory, days: int) -> float:
        """
        Calculate confidence decay over time.

        Args:
            memory: The memory to calculate decay for
            days: Number of days since last access/confirmation

        Returns:
            float: Amount of confidence to deduct (should be non-negative)
        """
        pass


class SimpleConfidenceStrategy(ConfidenceEvolutionStrategy):
    """
    Simple incremental confidence evolution strategy.

    This strategy uses fixed increments and linear time decay:
    - Reference reinforcement: +0.1
    - Relation reinforcement: +0.05
    - Time decay: 0.01 per day
    """

    # Default configuration values
    DEFAULT_REFERENCE_INCREMENT = 0.1
    DEFAULT_RELATION_INCREMENT = 0.05
    DEFAULT_DAILY_DECAY_RATE = 0.01

    def __init__(
        self,
        reference_increment: float = DEFAULT_REFERENCE_INCREMENT,
        relation_increment: float = DEFAULT_RELATION_INCREMENT,
        daily_decay_rate: float = DEFAULT_DAILY_DECAY_RATE
    ):
        """
        Initialize the simple confidence strategy.

        Args:
            reference_increment: Confidence increment when referenced (default: 0.1)
            relation_increment: Confidence increment when relation created (default: 0.05)
            daily_decay_rate: Confidence decay per day (default: 0.01)
        """
        self.reference_increment = reference_increment
        self.relation_increment = relation_increment
        self.daily_decay_rate = daily_decay_rate

    async def reinforce_on_reference(self, memory: Memory) -> float:
        """
        Apply fixed increment when memory is referenced.

        Args:
            memory: The memory being referenced

        Returns:
            float: Fixed reference increment
        """
        return self.reference_increment

    async def reinforce_on_relation(self, memory: Memory) -> float:
        """
        Apply fixed increment when relation is established.

        Args:
            memory: The memory for which a relation is being created

        Returns:
            float: Fixed relation increment
        """
        return self.relation_increment

    async def decay_over_time(self, memory: Memory, days: int) -> float:
        """
        Calculate linear time decay.

        Args:
            memory: The memory to calculate decay for
            days: Number of days since last access

        Returns:
            float: Days * daily decay rate
        """
        if days < 0:
            return 0.0
        return min(1.0, days * self.daily_decay_rate)

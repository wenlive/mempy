"""
Forgetting threshold strategies for memory management.

This module provides strategies for determining which memories should
be forgotten (deleted) based on their firmness and other factors.
"""

from abc import ABC, abstractmethod

from mempy.core.memory import Memory


class ForgettingThresholdStrategy(ABC):
    """
    Abstract base class for forgetting threshold strategies.

    A forgetting threshold strategy determines whether a memory
    should be deleted based on its characteristics and calculated firmness.
    """

    @abstractmethod
    def should_forget(self, memory: Memory, firmness: float) -> bool:
        """
        Determine if a memory should be forgotten.

        Args:
            memory: The memory to evaluate
            firmness: Calculated firmness score for the memory

        Returns:
            bool: True if memory should be forgotten, False otherwise
        """
        pass


class FixedThresholdStrategy(ForgettingThresholdStrategy):
    """
    Simple fixed threshold forgetting strategy.

    Memories with firmness below a fixed threshold are forgotten.
    This provides predictable, uniform behavior across all memories.
    """

    DEFAULT_THRESHOLD = 0.3

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        """
        Initialize the fixed threshold strategy.

        Args:
            threshold: Firmness threshold below which memories are forgotten
                      Must be in range [0.0, 1.0]

        Raises:
            ValueError: If threshold is not in valid range
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be in [0.0, 1.0], got {threshold}")

        self.threshold = threshold

    def should_forget(self, memory: Memory, firmness: float) -> bool:
        """
        Determine if memory should be forgotten based on fixed threshold.

        Args:
            memory: The memory to evaluate
            firmness: Calculated firmness score

        Returns:
            bool: True if firmness < threshold, False otherwise
        """
        return firmness < self.threshold


class ImportanceAwareThreshold(ForgettingThresholdStrategy):
    """
    Importance-aware forgetting threshold strategy.

    Adjusts the forgetting threshold based on memory importance.
    Important memories (higher importance) have lower thresholds
    (harder to forget), while less important memories have higher
    thresholds (easier to forget).

    Threshold calculation:
        threshold = base_threshold + (1.0 - importance) * adjustment_factor

    Example:
        - importance=1.0 (critical): threshold = 0.3 + 0.0 * 0.3 = 0.3
        - importance=0.5 (normal):   threshold = 0.3 + 0.5 * 0.3 = 0.45
        - importance=0.0 (trivial):  threshold = 0.3 + 1.0 * 0.3 = 0.6
    """

    DEFAULT_BASE_THRESHOLD = 0.3
    DEFAULT_ADJUSTMENT_FACTOR = 0.3

    def __init__(
        self,
        base_threshold: float = DEFAULT_BASE_THRESHOLD,
        adjustment_factor: float = DEFAULT_ADJUSTMENT_FACTOR
    ):
        """
        Initialize the importance-aware strategy.

        Args:
            base_threshold: Minimum threshold (for importance=1.0)
            adjustment_factor: How much to adjust threshold based on importance

        Raises:
            ValueError: If parameters are not in valid ranges
        """
        if not 0.0 <= base_threshold <= 1.0:
            raise ValueError(f"Base threshold must be in [0.0, 1.0], got {base_threshold}")
        if not 0.0 <= adjustment_factor <= 1.0:
            raise ValueError(f"Adjustment factor must be in [0.0, 1.0], got {adjustment_factor}")

        self.base_threshold = base_threshold
        self.adjustment_factor = adjustment_factor

    def should_forget(self, memory: Memory, firmness: float) -> bool:
        """
        Determine if memory should be forgotten based on importance-aware threshold.

        Args:
            memory: The memory to evaluate (uses memory.importance)
            firmness: Calculated firmness score

        Returns:
            bool: True if firmness < adjusted_threshold, False otherwise
        """
        # Calculate adjusted threshold based on importance
        adjusted_threshold = (
            self.base_threshold +
            (1.0 - memory.importance) * self.adjustment_factor
        )

        # Ensure threshold stays in valid range
        adjusted_threshold = max(0.0, min(1.0, adjusted_threshold))

        return firmness < adjusted_threshold


class TimeAwareThreshold(ForgettingThresholdStrategy):
    """
    Time-aware forgetting threshold strategy.

    Adjusts the forgetting threshold based on memory age.
    Newer memories have higher thresholds (easier to forget,
    as they haven't proven their value), while older memories
    have lower thresholds (harder to forget, as they've
    survived longer).

    Threshold calculation:
        age_years = age_days / 365
        threshold = base_threshold - min(max_adjustment, age_years * adjustment_per_year)

    Example (base=0.3, max_adj=0.2, per_year=0.1):
        - New (0 days):   threshold = 0.3 - 0.0 = 0.3
        - 1 year old:     threshold = 0.3 - 0.1 = 0.2
        - 2+ years old:   threshold = 0.3 - 0.2 = 0.1
    """

    DEFAULT_BASE_THRESHOLD = 0.3
    DEFAULT_MAX_ADJUSTMENT = 0.2
    DEFAULT_ADJUSTMENT_PER_YEAR = 0.1

    def __init__(
        self,
        base_threshold: float = DEFAULT_BASE_THRESHOLD,
        max_adjustment: float = DEFAULT_MAX_ADJUSTMENT,
        adjustment_per_year: float = DEFAULT_ADJUSTMENT_PER_YEAR
    ):
        """
        Initialize the time-aware strategy.

        Args:
            base_threshold: Threshold for brand new memories
            max_adjustment: Maximum threshold reduction for old memories
            adjustment_per_year: How much to reduce threshold per year of age

        Raises:
            ValueError: If parameters are not in valid ranges
        """
        if not 0.0 <= base_threshold <= 1.0:
            raise ValueError(f"Base threshold must be in [0.0, 1.0], got {base_threshold}")
        if not 0.0 <= max_adjustment <= base_threshold:
            raise ValueError(
                f"Max adjustment must be in [0.0, base_threshold], got {max_adjustment}"
            )
        if not 0.0 <= adjustment_per_year <= 1.0:
            raise ValueError(
                f"Adjustment per year must be in [0.0, 1.0], got {adjustment_per_year}"
            )

        self.base_threshold = base_threshold
        self.max_adjustment = max_adjustment
        self.adjustment_per_year = adjustment_per_year

    def should_forget(self, memory: Memory, firmness: float) -> bool:
        """
        Determine if memory should be forgotten based on time-aware threshold.

        Args:
            memory: The memory to evaluate (uses memory.created_at)
            firmness: Calculated firmness score

        Returns:
            bool: True if firmness < adjusted_threshold, False otherwise
        """
        from datetime import datetime

        # Calculate memory age in years
        age_days = (datetime.now() - memory.created_at).days
        age_years = age_days / 365.0

        # Calculate threshold reduction based on age
        adjustment = min(self.max_adjustment, age_years * self.adjustment_per_year)

        # Calculate adjusted threshold
        adjusted_threshold = self.base_threshold - adjustment

        return firmness < adjusted_threshold

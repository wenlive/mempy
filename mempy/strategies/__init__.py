"""
Memory evolution strategies for mempy.

This module provides pluggable strategy interfaces for memory evolution,
including confidence evolution, firmness calculation, forgetting thresholds,
relation exploration, and relation building.
"""

# Relation builders
from mempy.strategies.builders import (
    RelationBuilder,
    RandomRelationBuilder
)

# Confidence evolution strategies
from mempy.strategies.confidence import (
    ConfidenceEvolutionStrategy,
    SimpleConfidenceStrategy
)

# Firmness calculation strategies
from mempy.strategies.firmness import (
    FirmnessCalculator,
    WeightedFirmnessCalculator
)

# Forgetting threshold strategies
from mempy.strategies.forgetting import (
    ForgettingThresholdStrategy,
    FixedThresholdStrategy,
    ImportanceAwareThreshold,
    TimeAwareThreshold
)

# Relation exploration strategies
from mempy.strategies.exploration import (
    RelationExplorationStrategy,
    CosineSimilarityExplorer,
    AdaptiveSimilarityExplorer
)

__all__ = [
    # Relation builders
    "RelationBuilder",
    "RandomRelationBuilder",

    # Confidence evolution
    "ConfidenceEvolutionStrategy",
    "SimpleConfidenceStrategy",

    # Firmness calculation
    "FirmnessCalculator",
    "WeightedFirmnessCalculator",

    # Forgetting threshold
    "ForgettingThresholdStrategy",
    "FixedThresholdStrategy",
    "ImportanceAwareThreshold",
    "TimeAwareThreshold",

    # Relation exploration
    "RelationExplorationStrategy",
    "CosineSimilarityExplorer",
    "AdaptiveSimilarityExplorer",
]

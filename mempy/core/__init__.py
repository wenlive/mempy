"""Core data structures and interfaces."""

from mempy.core.memory import Memory, RelationType, Relation
from mempy.core.interfaces import (
    # Core interfaces (required)
    Embedder,
    # Strategy interfaces (optional)
    MemoryProcessor,
    ProcessorResult,
    # Real-time strategies
    RelationBuilder,
    # Evolution strategies
    ConfidenceEvolutionStrategy,
    FirmnessCalculator,
    ForgettingThresholdStrategy,
    RelationExplorationStrategy,
    # Internal interfaces
    StorageBackend,
)
from mempy.core.exceptions import (
    MempyError,
    EmbedderError,
    StorageError,
    ProcessorError,
)

__all__ = [
    # Core data structures
    "Memory",
    "RelationType",
    "Relation",
    # Core interfaces (required)
    "Embedder",
    # Strategy interfaces (optional)
    "MemoryProcessor",
    "ProcessorResult",
    "RelationBuilder",
    "ConfidenceEvolutionStrategy",
    "FirmnessCalculator",
    "ForgettingThresholdStrategy",
    "RelationExplorationStrategy",
    # Internal interfaces
    "StorageBackend",
    # Exceptions
    "MempyError",
    "EmbedderError",
    "StorageError",
    "ProcessorError",
]

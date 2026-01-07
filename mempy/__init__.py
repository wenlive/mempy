"""
mempy - A memory management library with vector and graph storage.

Similar to mem0, but with:
- Zero configuration embedded design
- Vector + Graph dual first-class citizens
- Pluggable embedder, LLM processor, and storage backend
- Full async API
"""

from mempy.memory import Memory as MemoryClient
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
)
from mempy.core.exceptions import (
    MempyError,
    EmbedderError,
    StorageError,
    ProcessorError,
)
from mempy.config import get_storage_path

__version__ = "0.1.0"

# Main API class
Memory = MemoryClient

__all__ = [
    # Main API
    "Memory",
    # Core types
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
    # Exceptions
    "MempyError",
    "EmbedderError",
    "StorageError",
    "ProcessorError",
    # Config
    "get_storage_path",
]

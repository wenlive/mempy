"""Core data structures and interfaces."""

from mempy.core.memory import Memory, RelationType, Relation
from mempy.core.interfaces import Embedder, MemoryProcessor, ProcessorResult
from mempy.core.exceptions import (
    MempyError,
    EmbedderError,
    StorageError,
    ProcessorError,
)

__all__ = [
    "Memory",
    "RelationType",
    "Relation",
    "Embedder",
    "MemoryProcessor",
    "ProcessorResult",
    "MempyError",
    "EmbedderError",
    "StorageError",
    "ProcessorError",
]

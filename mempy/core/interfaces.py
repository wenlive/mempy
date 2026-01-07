"""Abstract interface definitions for mempy.

This module defines all interfaces organized into three layers:
1. Core Interfaces (Required) - System essential interfaces that users must implement
2. Strategy Interfaces (Optional) - Pluggable strategies for extending functionality
3. Internal Interfaces - Internal implementation details users typically don't need
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from mempy.core.memory import Memory, ProcessorResult, RelationType
from mempy.core.exceptions import MempyError


# ==============================================================================
# 第一层：核心接口 (Core Interfaces)
# 系统运行必需，用户必须实现的接口
# ==============================================================================


class Embedder(ABC):
    """
    Abstract embedder interface (REQUIRED).

    Users MUST implement this interface to provide embedding capabilities.
    The embedder is responsible for calling the user's LLM service
    (local or remote) to generate vector representations of text.

    Required Implementation:
        - dimension property: Return the vector dimension
        - embed() method: Generate embedding for text

    Example:
        >>> class MyEmbedder(Embedder):
        ...     @property
        ...     def dimension(self) -> int:
        ...         return 768
        ...
        ...     async def embed(self, text: str) -> List[float]:
        ...         return await my_llm_service.embed(text)
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """
        Return the dimension of the embedding vectors.

        This is REQUIRED and must be implemented by users.
        Example values: 768, 1536, etc. depending on your model.

        Returns:
            int: The embedding vector dimension
        """
        pass

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """
        Generate an embedding vector for the given text.

        Args:
            text: The text to embed

        Returns:
            List[float]: The embedding vector

        Raises:
            EmbedderError: If embedding generation fails
        """
        pass


# ==============================================================================
# 第二层：策略接口 (Strategy Interfaces)
# 可选功能，用户可以按需实现以扩展系统功能
# ==============================================================================

# -----------------------------------------------------------------------------
# 实时策略 (Real-time Strategies)
# 在Memory.add()时实时调用的策略
# -----------------------------------------------------------------------------


class MemoryProcessor(ABC):
    """
    Abstract memory processor interface (OPTIONAL).

    The processor intelligently decides what operation to perform when
    adding new content: add new memory, update existing, delete, or ignore.

    This is typically implemented using an LLM to analyze the input against
    existing memories. Used in real-time during Memory.add() operations.

    Usage:
        - Provide to Memory constructor via `processor` parameter
        - Called automatically when Memory.add() is invoked

    Example:
        >>> class LLMProcessor(MemoryProcessor):
        ...     async def process(self, content, existing_memories):
        ...         # Use LLM to decide what to do
        ...         return ProcessorResult(action="add", ...)
    """

    @abstractmethod
    async def process(
        self,
        content: str,
        existing_memories: List[Memory]
    ) -> ProcessorResult:
        """
        Decide what operation to perform based on content and existing memories.

        Args:
            content: The new content to process
            existing_memories: List of potentially related existing memories

        Returns:
            ProcessorResult with action, memory_id, content, and reason

        Raises:
            ProcessorError: If processing fails
        """
        pass


# Import RelationBuilder from strategies module
# This is a real-time strategy for automatic graph construction
from mempy.strategies.builders import RelationBuilder


# -----------------------------------------------------------------------------
# 演化策略 (Evolution Strategies)
# 在Memory.evolve()时批量调用的策略
# -----------------------------------------------------------------------------

# Import evolution strategy interfaces from strategies module
# These are used for long-term memory maintenance and optimization

from mempy.strategies.confidence import ConfidenceEvolutionStrategy
from mempy.strategies.firmness import FirmnessCalculator
from mempy.strategies.forgetting import ForgettingThresholdStrategy
from mempy.strategies.exploration import RelationExplorationStrategy


# ==============================================================================
# 第三层：内部接口 (Internal Interfaces)
# 用户通常不需要关心的内部实现接口
# ==============================================================================


class StorageBackend(ABC):
    """
    Abstract storage backend interface (INTERNAL).

    This unified interface hides the complexity of dual-write to both
    vector store (for semantic search) and graph store (for relations).

    Most users do NOT need to implement this interface. It is provided
    for users who want to implement custom storage backends (e.g.,
    replacing ChromaDB with a different vector store).

    Built-in Implementations:
        - DualStorageBackend: Default implementation using ChromaDB + NetworkX
    """

    @abstractmethod
    async def add(self, memory: Memory) -> str:
        """
        Add a memory to storage.

        Should write to both vector and graph stores atomically.

        Args:
            memory: The memory to add

        Returns:
            str: The memory ID of the added memory

        Raises:
            StorageError: If the add operation fails
        """
        pass

    @abstractmethod
    async def get(self, memory_id: str) -> Optional[Memory]:
        """
        Retrieve a memory by ID.

        Args:
            memory_id: The memory ID to retrieve

        Returns:
            The memory if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_all(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> List[Memory]:
        """
        Retrieve all memories matching the given filters.

        Args:
            filters: Optional filters (e.g., {"user_id": "alice"})
            limit: Optional maximum number of results

        Returns:
            List of matching memories
        """
        pass

    @abstractmethod
    async def search(
        self,
        query_vector: List[float],
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10
    ) -> List[Memory]:
        """
        Search for memories by semantic similarity.

        Should combine vector similarity with graph traversal for best results.

        Args:
            query_vector: The query embedding vector
            filters: Optional filters (e.g., {"user_id": "alice"})
            limit: Maximum number of results to return

        Returns:
            List of memories ranked by relevance
        """
        pass

    @abstractmethod
    async def update(self, memory_id: str, memory: Memory) -> None:
        """
        Update an existing memory.

        Args:
            memory_id: The ID of the memory to update
            memory: The updated memory data

        Raises:
            StorageError: If memory not found or update fails
        """
        pass

    @abstractmethod
    async def delete(self, memory_id: str) -> None:
        """
        Delete a memory by ID.

        Args:
            memory_id: The ID of the memory to delete

        Raises:
            StorageError: If deletion fails
        """
        pass

    @abstractmethod
    async def delete_all(self, filters: Dict[str, Any]) -> None:
        """
        Delete all memories matching the given filters.

        Args:
            filters: Filters to match (e.g., {"user_id": "alice"})

        Raises:
            StorageError: If deletion fails
        """
        pass

    @abstractmethod
    async def add_relation(
        self,
        from_id: str,
        to_id: str,
        relation_type: RelationType,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a relation between two memories.

        Args:
            from_id: Source memory ID
            to_id: Target memory ID
            relation_type: Type of relation
            metadata: Optional metadata about the relation

        Raises:
            StorageError: If relation cannot be added
        """
        pass

    @abstractmethod
    async def get_relations(
        self,
        memory_id: str,
        direction: str = "both",
        max_depth: int = 1
    ) -> List[RelationType]:
        """
        Get relations for a memory.

        Args:
            memory_id: The memory ID to get relations for
            direction: "out", "in", or "both"
            max_depth: Maximum depth to traverse in the graph

        Returns:
            List of relations
        """
        pass

    @abstractmethod
    async def reset(self) -> None:
        """
        Clear all data from storage.

        WARNING: This operation is irreversible.
        """
        pass

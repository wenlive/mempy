"""Unified storage backend with dual-write and observability."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from mempy.core.interfaces import StorageBackend
from mempy.core.memory import Memory, RelationType
from mempy.core.exceptions import StorageError
from mempy.storage.vector_store import ChromaVectorStore
from mempy.storage.graph_store import NetworkXGraphStore


@dataclass
class OperationResult:
    """
    Result of a storage operation with observability data.

    Attributes:
        success: Whether the operation succeeded
        operation: Type of operation performed
        vector_store_ok: Whether vector store write succeeded
        graph_store_ok: Whether graph store write succeeded
        error_message: Error message if operation failed
        timestamp: When the operation was performed
    """

    success: bool = False
    operation: str = ""
    vector_store_ok: Optional[bool] = None
    graph_store_ok: Optional[bool] = None
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


class DualStorageBackend(StorageBackend):
    """
    Dual-write storage backend combining vector and graph stores.

    This unified backend:
    - Writes to both ChromaDB (vector) and NetworkX (graph)
    - Treats vector store as primary (its failure = total failure)
    - Treats graph store as secondary (its failure = warning only)
    - Provides observable results for all operations
    """

    def __init__(self, persist_path: Path):
        """
        Initialize the dual storage backend.

        Args:
            persist_path: Directory path for persistence
        """
        self.persist_path = Path(persist_path)
        self.vector_store = ChromaVectorStore(self.persist_path)
        self.graph_store = NetworkXGraphStore(self.persist_path)
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging for observability."""
        self.logger = logging.getLogger("mempy.storage")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("[%(name)s] %(levelname)s - %(message)s")
            )
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    async def add(self, memory: Memory) -> str:
        """
        Add a memory, writing to both vector and graph stores.

        Strategy: Vector store is primary
        - Vector write fails: total failure, abort
        - Graph write fails: warning only, vector write is preserved

        Args:
            memory: The memory to add

        Returns:
            The memory ID

        Raises:
            StorageError: If vector store write fails
        """
        result = OperationResult(operation="add")

        # 1. First, write to vector store (primary)
        try:
            memory_id = await self.vector_store.add(memory)
            result.vector_store_ok = True
            self.logger.info(f"[VECTOR] Added memory {memory_id}")
        except Exception as e:
            result.vector_store_ok = False
            result.success = False
            result.error_message = f"Vector store failed: {e}"
            self.logger.error(f"[VECTOR] Failed to add memory: {e}")
            raise StorageError(result.error_message) from e

        # 2. Then, write to graph store (secondary)
        try:
            await self.graph_store.add_node(memory_id, memory)
            result.graph_store_ok = True
            self.logger.info(f"[GRAPH] Added node {memory_id}")
        except Exception as e:
            result.graph_store_ok = False
            # Graph failure is non-critical
            self.logger.warning(
                f"[GRAPH] Failed to add node (non-critical): {e}. "
                f"Memory {memory_id} saved without graph relations."
            )

        result.success = result.vector_store_ok
        return memory_id

    async def get(self, memory_id: str) -> Optional[Memory]:
        """
        Get a memory by ID.

        Args:
            memory_id: The memory ID

        Returns:
            The memory if found, None otherwise
        """
        return await self.vector_store.get(memory_id)

    async def get_all(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> List[Memory]:
        """
        Get all memories matching filters.

        Args:
            filters: Optional filters (user_id, agent_id, run_id)
            limit: Optional maximum number of results

        Returns:
            List of matching memories
        """
        return await self.vector_store.get_all(filters, limit)

    async def search(
        self,
        query_vector: List[float],
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10
    ) -> List[Memory]:
        """
        Search memories by semantic similarity.

        Args:
            query_vector: Query embedding vector
            filters: Optional filters
            limit: Maximum number of results

        Returns:
            List of memories ranked by similarity
        """
        return await self.vector_store.search(query_vector, filters, limit)

    async def update(self, memory_id: str, memory: Memory) -> None:
        """
        Update a memory in both stores.

        Args:
            memory_id: The ID of the memory to update
            memory: The updated memory data

        Raises:
            StorageError: If update fails
        """
        result = OperationResult(operation="update")

        # Update vector store (primary)
        try:
            await self.vector_store.update(memory_id, memory)
            result.vector_store_ok = True
            self.logger.info(f"[VECTOR] Updated memory {memory_id}")
        except Exception as e:
            result.vector_store_ok = False
            result.success = False
            result.error_message = f"Vector store update failed: {e}"
            self.logger.error(f"[VECTOR] Failed to update memory {memory_id}: {e}")
            raise StorageError(result.error_message) from e

        # Update graph store (secondary)
        try:
            await self.graph_store.update_node(memory_id, memory)
            result.graph_store_ok = True
            self.logger.info(f"[GRAPH] Updated node {memory_id}")
        except Exception as e:
            result.graph_store_ok = False
            self.logger.warning(
                f"[GRAPH] Failed to update node (non-critical): {e}"
            )

        result.success = result.vector_store_ok

    async def delete(self, memory_id: str) -> None:
        """
        Delete a memory from both stores.

        Args:
            memory_id: The ID of the memory to delete

        Raises:
            StorageError: If delete fails
        """
        result = OperationResult(operation="delete")

        # Delete from vector store (primary)
        try:
            await self.vector_store.delete(memory_id)
            result.vector_store_ok = True
            self.logger.info(f"[VECTOR] Deleted memory {memory_id}")
        except Exception as e:
            result.vector_store_ok = False
            result.error_message = f"Vector store delete failed: {e}"
            self.logger.error(f"[VECTOR] Failed to delete memory {memory_id}: {e}")
            raise StorageError(result.error_message) from e

        # Delete from graph store (secondary)
        try:
            await self.graph_store.delete_node(memory_id)
            result.graph_store_ok = True
            self.logger.info(f"[GRAPH] Deleted node {memory_id}")
        except Exception as e:
            result.graph_store_ok = False
            self.logger.warning(
                f"[GRAPH] Failed to delete node (non-critical): {e}"
            )

    async def delete_all(self, filters: Dict[str, Any]) -> None:
        """
        Delete all memories matching filters.

        Args:
            filters: Filters to match (user_id required)

        Raises:
            StorageError: If delete fails
        """
        result = OperationResult(operation="delete_all")

        # Get matching IDs first
        memories = await self.vector_store.get_all(filters)
        memory_ids = [m.memory_id for m in memories]

        # Delete from vector store
        try:
            await self.vector_store.delete_all(filters)
            result.vector_store_ok = True
            self.logger.info(f"[VECTOR] Deleted {len(memory_ids)} memories")
        except Exception as e:
            result.error_message = f"Vector store delete_all failed: {e}"
            self.logger.error(result.error_message)
            raise StorageError(result.error_message) from e

        # Delete from graph store
        for memory_id in memory_ids:
            try:
                await self.graph_store.delete_node(memory_id)
            except Exception as e:
                self.logger.warning(f"[GRAPH] Failed to delete node {memory_id}: {e}")

        result.graph_store_ok = True
        self.logger.info(f"[GRAPH] Deleted {len(memory_ids)} nodes")

    async def add_relation(
        self,
        from_id: str,
        to_id: str,
        relation_type: RelationType,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a relation between two memories (graph store only).

        Args:
            from_id: Source memory ID
            to_id: Target memory ID
            relation_type: Type of relation
            metadata: Optional metadata

        Raises:
            StorageError: If relation cannot be added
        """
        result = OperationResult(operation="add_relation")

        try:
            await self.graph_store.add_edge(from_id, to_id, relation_type, metadata)
            result.success = True
            self.logger.info(
                f"[GRAPH] Added relation: {from_id} --[{relation_type.value}]--> {to_id}"
            )
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            self.logger.error(
                f"[GRAPH] Failed to add relation {from_id} -> {to_id}: {e}"
            )
            raise StorageError(result.error_message) from e

    async def get_relations(
        self,
        memory_id: str,
        direction: str = "both",
        max_depth: int = 1
    ) -> List:
        """
        Get relations for a memory.

        Args:
            memory_id: The memory ID
            direction: "out", "in", or "both"
            max_depth: Maximum depth to traverse

        Returns:
            List of relations
        """
        return await self.graph_store.get_relations(memory_id, direction, max_depth)

    async def reset(self) -> None:
        """
        Reset both stores, deleting all data.

        WARNING: This is irreversible.
        """
        result = OperationResult(operation="reset")

        try:
            await self.vector_store.reset()
            result.vector_store_ok = True
            self.logger.info("[VECTOR] Reset complete")
        except Exception as e:
            result.error_message = f"Vector store reset failed: {e}"
            self.logger.error(result.error_message)
            raise StorageError(result.error_message) from e

        try:
            await self.graph_store.reset()
            result.graph_store_ok = True
            self.logger.info("[GRAPH] Reset complete")
        except Exception as e:
            self.logger.warning(f"[GRAPH] Reset failed (non-critical): {e}")

        result.success = result.vector_store_ok

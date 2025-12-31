"""Main Memory API - the primary user interface for mempy."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from mempy.core.interfaces import Embedder, MemoryProcessor
from mempy.core.memory import Memory as MemoryData, RelationType
from mempy.core.exceptions import StorageError
from mempy.storage.backend import DualStorageBackend
from mempy.config import get_storage_path


class Memory:
    """
    Main Memory class - the primary user interface for mempy.

    This class provides a simple, intuitive API for memory management:
    - add(): Add new memories with intelligent processing
    - search(): Semantic search through memories
    - get(), get_all(): Retrieve memories
    - update(): Update existing memories
    - delete(), delete_all(): Delete memories
    - add_relation(), get_relations(): Manage memory relations
    - reset(): Clear all data

    Example:
        ```python
        import mempy

        # Implement embedder (user must provide dimension)
        class MyEmbedder(mempy.Embedder):
            def __init__(self):
                self._dimension = 768  # Must declare dimension

            @property
            def dimension(self) -> int:
                return self._dimension

            async def embed(self, text: str) -> List[float]:
                # Call your LLM service
                return await my_llm.embed(text)

        # Create Memory instance
        memory = mempy.Memory(embedder=MyEmbedder(), verbose=True)

        # Add memories
        await memory.add("I like blue", user_id="alice")

        # Search
        results = await memory.search("color preference", user_id="alice")
        ```
    """

    def __init__(
        self,
        embedder: Embedder,
        processor: Optional[MemoryProcessor] = None,
        storage_path: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize the Memory instance.

        Args:
            embedder: User-provided embedder (must declare dimension property)
            processor: Optional LLM processor for intelligent memory operations
            storage_path: Optional custom storage path (default: ~/.mempy/data)
            verbose: Enable verbose logging output
        """
        self.embedder = embedder
        self.processor = processor
        self.verbose = verbose

        # Setup storage
        path = get_storage_path(storage_path)
        self.storage = DualStorageBackend(path)

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging for observability."""
        self.logger = logging.getLogger("mempy")

        if self.verbose and not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[mempy] %(message)s"))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    async def add(
        self,
        content: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Add a new memory with intelligent processing.

        If a processor is configured, it will analyze the content against
        existing memories to decide whether to add, update, delete, or ignore.

        Args:
            content: The memory content
            user_id: Optional user identifier for scoping
            agent_id: Optional agent identifier for scoping
            run_id: Optional session/conversation identifier
            metadata: Optional additional metadata

        Returns:
            The memory ID if successful, None if operation was skipped

        Raises:
            StorageError: If storage operation fails
            EmbedderError: If embedding generation fails
        """
        if self.verbose:
            self.logger.info(f"Processing: {content[:50]}{'...' if len(content) > 50 else ''}")

        # If processor exists, let it decide what to do
        if self.processor:
            # Search for potentially related memories
            existing = await self.search(content, user_id=user_id, limit=5)

            # Let processor decide
            result = await self.processor.process(content, existing)

            if self.verbose:
                self.logger.info(f"Processor action: {result.action}")

            if result.action == "update" and result.memory_id:
                memory_id = await self.update(result.memory_id, result.content)
                if self.verbose:
                    self.logger.info(f"Updated memory: {memory_id}")
                return memory_id

            elif result.action == "delete" and result.memory_id:
                await self.delete(result.memory_id)
                if self.verbose:
                    self.logger.info(f"Deleted memory: {result.memory_id}")
                return None

            elif result.action == "none":
                if self.verbose:
                    self.logger.info("Skipped (processor decided: none)")
                return None

        # Generate embedding
        embedding = await self.embedder.embed(content)

        # Create memory object
        memory = MemoryData(
            memory_id=uuid4().hex,
            content=content,
            embedding=embedding,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            metadata=metadata or {}
        )

        # Add to storage
        memory_id = await self.storage.add(memory)

        if self.verbose:
            self.logger.info(f"Saved: {memory_id}")

        return memory_id

    async def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: int = 10
    ) -> List[MemoryData]:
        """
        Search for memories by semantic similarity.

        Args:
            query: The search query text
            user_id: Optional user filter
            agent_id: Optional agent filter
            limit: Maximum number of results

        Returns:
            List of memories ranked by similarity
        """
        query_vector = await self.embedder.embed(query)

        filters = {}
        if user_id is not None:
            filters["user_id"] = user_id
        if agent_id is not None:
            filters["agent_id"] = agent_id

        results = await self.storage.search(query_vector, filters, limit)

        if self.verbose:
            self.logger.info(f"Found {len(results)} memories for: {query[:50]}")

        return results

    async def get(self, memory_id: str) -> Optional[MemoryData]:
        """
        Get a specific memory by ID.

        Args:
            memory_id: The memory ID

        Returns:
            The memory if found, None otherwise
        """
        return await self.storage.get(memory_id)

    async def get_all(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[MemoryData]:
        """
        Get all memories matching filters.

        Args:
            user_id: Optional user filter
            agent_id: Optional agent filter
            limit: Optional maximum number of results

        Returns:
            List of matching memories
        """
        filters = {}
        if user_id is not None:
            filters["user_id"] = user_id
        if agent_id is not None:
            filters["agent_id"] = agent_id

        return await self.storage.get_all(filters, limit)

    async def update(self, memory_id: str, content: str) -> str:
        """
        Update an existing memory's content.

        Args:
            memory_id: The ID of the memory to update
            content: The new content

        Returns:
            The memory ID

        Raises:
            StorageError: If memory not found or update fails
        """
        existing = await self.storage.get(memory_id)
        if existing is None:
            raise StorageError(f"Memory {memory_id} not found")

        # Generate new embedding
        embedding = await self.embedder.embed(content)

        # Update memory object
        existing.content = content
        existing.embedding = embedding
        existing.updated_at = datetime.utcnow()

        # Save to storage
        await self.storage.update(memory_id, existing)

        if self.verbose:
            self.logger.info(f"Updated: {memory_id}")

        return memory_id

    async def delete(self, memory_id: str) -> None:
        """
        Delete a memory by ID.

        Args:
            memory_id: The ID of the memory to delete

        Raises:
            StorageError: If deletion fails
        """
        await self.storage.delete(memory_id)

        if self.verbose:
            self.logger.info(f"Deleted: {memory_id}")

    async def delete_all(self, user_id: str) -> None:
        """
        Delete all memories for a user.

        Args:
            user_id: The user ID

        Raises:
            StorageError: If deletion fails
        """
        await self.storage.delete_all({"user_id": user_id})

        if self.verbose:
            self.logger.info(f"Deleted all memories for user: {user_id}")

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
        await self.storage.add_relation(from_id, to_id, relation_type, metadata)

        if self.verbose:
            self.logger.info(
                f"Added relation: {from_id} --[{relation_type.value}]--> {to_id}"
            )

    async def get_relations(
        self,
        memory_id: str,
        direction: str = "both",
        max_depth: int = 2
    ) -> List:
        """
        Get relations for a memory.

        Args:
            memory_id: The memory ID
            direction: "out", "in", or "both"
            max_depth: Maximum depth to traverse in graph

        Returns:
            List of relations
        """
        return await self.storage.get_relations(memory_id, direction, max_depth)

    async def reset(self) -> None:
        """
        Reset all data, clearing all memories and relations.

        WARNING: This operation is irreversible.
        """
        await self.storage.reset()

        if self.verbose:
            self.logger.info("Reset complete - all data cleared")

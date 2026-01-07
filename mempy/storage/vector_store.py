"""ChromaDB-based vector storage implementation."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

from mempy.core.memory import Memory
from mempy.core.exceptions import StorageError


class ChromaVectorStore:
    """
    ChromaDB-based vector storage for semantic search.

    This store handles:
    - Storing memory embeddings
    - Semantic similarity search
    - Filtering by metadata (user_id, agent_id, run_id)
    - Persistence to disk
    """

    def __init__(self, persist_path: Path, collection_name: str = "memories"):
        """
        Initialize the ChromaDB vector store.

        Args:
            persist_path: Directory path for persistence
            collection_name: Name of the ChromaDB collection
        """
        self.persist_path = Path(persist_path)
        self.collection_name = collection_name

        # Ensure persist directory exists
        vector_path = self.persist_path / "vector"
        vector_path.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(vector_path),
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )

    def _memory_to_metadata(self, memory: Memory) -> Dict[str, Any]:
        """Convert Memory object to ChromaDB metadata format."""
        return {
            "content": memory.content,
            "user_id": memory.user_id or "",
            "agent_id": memory.agent_id or "",
            "run_id": memory.run_id or "",
            "created_at": memory.created_at.isoformat(),
            "updated_at": memory.updated_at.isoformat() if memory.updated_at else "",
            "priority": memory.priority,
            "confidence": memory.confidence,
            "last_accessed_at": memory.last_accessed_at.isoformat() if memory.last_accessed_at else "",
            "access_count": memory.access_count,
            "importance": memory.importance,
            # Store additional metadata as JSON string
            "metadata_json": json.dumps(memory.metadata),
        }

    def _metadata_to_memory(
        self,
        id: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> Memory:
        """Convert ChromaDB result to Memory object."""
        from datetime import datetime

        additional_metadata = {}
        if "metadata_json" in metadata:
            try:
                additional_metadata = json.loads(metadata["metadata_json"])
            except json.JSONDecodeError:
                pass

        # Parse last_accessed_at if present
        last_accessed_at = None
        if metadata.get("last_accessed_at"):
            try:
                last_accessed_at = datetime.fromisoformat(metadata["last_accessed_at"])
            except (ValueError, TypeError):
                pass

        return Memory(
            memory_id=id,
            content=metadata.get("content", ""),
            embedding=embedding,
            user_id=metadata.get("user_id") or None,
            agent_id=metadata.get("agent_id") or None,
            run_id=metadata.get("run_id") or None,
            metadata=additional_metadata,
            created_at=datetime.fromisoformat(metadata.get("created_at", "")),
            updated_at=datetime.fromisoformat(metadata["updated_at"]) if metadata.get("updated_at") else None,
            priority=metadata.get("priority", 0.5),
            confidence=metadata.get("confidence", 1.0),
            last_accessed_at=last_accessed_at,
            access_count=metadata.get("access_count", 0),
            importance=metadata.get("importance", 0.5),
        )

    async def add(self, memory: Memory) -> str:
        """
        Add a memory to the vector store.

        Args:
            memory: The memory to add

        Returns:
            The memory ID

        Raises:
            StorageError: If add fails
        """
        try:
            self.collection.add(
                ids=[memory.memory_id],
                embeddings=[memory.embedding],
                metadatas=[self._memory_to_metadata(memory)],
            )
            return memory.memory_id
        except Exception as e:
            raise StorageError(f"Failed to add memory to vector store: {e}") from e

    async def get(self, memory_id: str) -> Optional[Memory]:
        """
        Get a memory by ID.

        Args:
            memory_id: The memory ID

        Returns:
            The memory if found, None otherwise
        """
        try:
            result = self.collection.get(ids=[memory_id], include=["embeddings", "metadatas"])

            if not result["ids"]:
                return None

            return self._metadata_to_memory(
                id=result["ids"][0],
                embedding=result["embeddings"][0],
                metadata=result["metadatas"][0]
            )
        except Exception as e:
            raise StorageError(f"Failed to get memory from vector store: {e}") from e

    async def get_all(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> List[Memory]:
        """
        Get all memories, optionally filtered.

        Args:
            filters: Optional filters (user_id, agent_id, run_id)
            limit: Optional maximum number of results

        Returns:
            List of memories
        """
        try:
            where = self._build_where_clause(filters)
            result = self.collection.get(
                where=where,
                limit=limit,
                include=["embeddings", "metadatas"]
            )

            memories = []
            for i, memory_id in enumerate(result.get("ids", [])):
                memories.append(self._metadata_to_memory(
                    id=memory_id,
                    embedding=result["embeddings"][i],
                    metadata=result["metadatas"][i]
                ))

            return memories
        except Exception as e:
            raise StorageError(f"Failed to get all memories: {e}") from e

    async def search(
        self,
        query_vector: List[float],
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10
    ) -> List[Memory]:
        """
        Search for memories by semantic similarity.

        Args:
            query_vector: Query embedding vector
            filters: Optional filters (user_id, agent_id, run_id)
            limit: Maximum number of results

        Returns:
            List of memories ranked by similarity
        """
        try:
            where = self._build_where_clause(filters)

            result = self.collection.query(
                query_embeddings=[query_vector],
                where=where,
                n_results=limit,
                include=["embeddings", "metadatas", "distances"]
            )

            memories = []
            for i, memory_id in enumerate(result.get("ids", [])[0]):
                memories.append(self._metadata_to_memory(
                    id=memory_id,
                    embedding=result["embeddings"][0][i],
                    metadata=result["metadatas"][0][i]
                ))

            return memories
        except Exception as e:
            raise StorageError(f"Failed to search memories: {e}") from e

    async def update(self, memory_id: str, memory: Memory) -> None:
        """
        Update a memory in the vector store.

        Args:
            memory_id: The ID of the memory to update
            memory: The updated memory data

        Raises:
            StorageError: If update fails
        """
        try:
            self.collection.update(
                ids=[memory_id],
                embeddings=[memory.embedding],
                metadatas=[self._memory_to_metadata(memory)]
            )
        except Exception as e:
            raise StorageError(f"Failed to update memory: {e}") from e

    async def delete(self, memory_id: str) -> None:
        """
        Delete a memory from the vector store.

        Args:
            memory_id: The ID of the memory to delete

        Raises:
            StorageError: If delete fails
        """
        try:
            self.collection.delete(ids=[memory_id])
        except Exception as e:
            raise StorageError(f"Failed to delete memory: {e}") from e

    async def delete_all(self, filters: Dict[str, Any]) -> None:
        """
        Delete all memories matching filters.

        Args:
            filters: Filters to match (user_id required)

        Raises:
            StorageError: If delete fails
        """
        try:
            where = self._build_where_clause(filters)
            # Get IDs first, then delete
            result = self.collection.get(where=where)
            if result["ids"]:
                self.collection.delete(ids=result["ids"])
        except Exception as e:
            raise StorageError(f"Failed to delete all memories: {e}") from e

    async def reset(self) -> None:
        """
        Reset the vector store, deleting all data.

        WARNING: This is irreversible.
        """
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            raise StorageError(f"Failed to reset vector store: {e}") from e

    def _build_where_clause(self, filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Build ChromaDB where clause from filters.

        Args:
            filters: Filter dict with user_id, agent_id, run_id

        Returns:
            ChromaDB where clause or None
        """
        if not filters:
            return None

        where = {}
        for key in ["user_id", "agent_id", "run_id"]:
            value = filters.get(key)
            if value is not None:
                where[key] = value

        return where if where else None

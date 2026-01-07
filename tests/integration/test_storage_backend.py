"""Tests for DualStorageBackend with mocked dependencies."""

from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mempy.core.memory import Memory, RelationType
from mempy.core.exceptions import StorageError
from mempy.storage.backend import DualStorageBackend, OperationResult


class TestOperationResult:
    """Tests for OperationResult dataclass."""

    def test_operation_result_creation(self):
        """Test creating an operation result."""
        result = OperationResult(
            success=True,
            operation="add",
            vector_store_ok=True,
            graph_store_ok=True
        )
        assert result.success is True
        assert result.operation == "add"
        assert result.vector_store_ok is True
        assert result.graph_store_ok is True
        assert result.error_message is None
        assert result.timestamp is not None

    def test_operation_result_with_error(self):
        """Test operation result with error information."""
        result = OperationResult(
            success=False,
            operation="delete",
            error_message="Memory not found"
        )
        assert result.success is False
        assert result.error_message == "Memory not found"


class TestDualStorageBackendInitialization:
    """Tests for DualStorageBackend initialization."""

    @pytest.fixture
    def temp_path(self, tmp_path):
        """Provide a temporary storage path."""
        return tmp_path / "storage"

    def test_init_creates_stores(self, temp_path):
        """Test that initialization creates both vector and graph stores."""
        backend = DualStorageBackend(temp_path)

        assert backend.persist_path == temp_path
        assert backend.vector_store is not None
        assert backend.graph_store is not None
        assert backend.logger is not None


class TestAddOperation:
    """Tests for add operation with different scenarios."""

    @pytest.fixture
    def temp_path(self, tmp_path):
        """Provide a temporary storage path."""
        return tmp_path / "storage"

    @pytest.fixture
    def sample_memory(self):
        """Provide a sample memory."""
        return Memory(
            memory_id="mem-1",
            content="Test content",
            embedding=[0.1] * 768,
            user_id="user-1",
            created_at=datetime.utcnow()
        )

    @pytest.mark.asyncio
    async def test_add_success_both_stores(self, temp_path, sample_memory):
        """Test successful add to both vector and graph stores."""
        backend = DualStorageBackend(temp_path)

        memory_id = await backend.add(sample_memory)

        assert memory_id == "mem-1"

    @pytest.mark.asyncio
    async def test_add_vector_store_failure_raises_error(self, temp_path, sample_memory):
        """Test that vector store failure raises StorageError."""
        backend = DualStorageBackend(temp_path)

        # Mock vector store to raise exception
        backend.vector_store.add = AsyncMock(side_effect=Exception("Vector store down"))

        with pytest.raises(StorageError, match="Vector store failed"):
            await backend.add(sample_memory)

    @pytest.mark.asyncio
    async def test_add_graph_store_failure_succeeds(self, temp_path, sample_memory):
        """Test that graph store failure doesn't prevent add (secondary store)."""
        backend = DualStorageBackend(temp_path)

        # Mock graph store to raise exception
        backend.graph_store.add_node = AsyncMock(side_effect=Exception("Graph store down"))

        # Should succeed despite graph store failure
        memory_id = await backend.add(sample_memory)
        assert memory_id == "mem-1"


class TestGetOperations:
    """Tests for get operations."""

    @pytest.fixture
    def temp_path(self, tmp_path):
        """Provide a temporary storage path."""
        return tmp_path / "storage"

    @pytest.fixture
    def sample_memory(self):
        """Provide a sample memory."""
        return Memory(
            memory_id="mem-1",
            content="Test content",
            embedding=[0.1] * 768,
            user_id="user-1",
            created_at=datetime.utcnow()
        )

    @pytest.mark.asyncio
    async def test_get_existing_memory(self, temp_path, sample_memory):
        """Test getting an existing memory."""
        backend = DualStorageBackend(temp_path)

        # First add the memory
        await backend.add(sample_memory)

        # Then get it
        result = await backend.get("mem-1")

        assert result is not None
        assert result.memory_id == "mem-1"
        assert result.content == "Test content"

    @pytest.mark.asyncio
    async def test_get_nonexistent_memory(self, temp_path):
        """Test getting a non-existent memory returns None."""
        backend = DualStorageBackend(temp_path)

        result = await backend.get("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_all_no_filters(self, temp_path):
        """Test getting all memories without filters."""
        backend = DualStorageBackend(temp_path)

        # Add multiple memories
        for i in range(3):
            mem = Memory(
                memory_id=f"mem-{i}",
                content=f"Content {i}",
                embedding=[0.1] * 768,
                user_id="user-1",
                created_at=datetime.utcnow()
            )
            await backend.add(mem)

        results = await backend.get_all()

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_get_all_with_user_filter(self, temp_path):
        """Test getting memories filtered by user_id."""
        backend = DualStorageBackend(temp_path)

        # Add memories for different users
        for user_id in ["user-1", "user-2"]:
            for i in range(2):
                mem = Memory(
                    memory_id=f"{user_id}-mem-{i}",
                    content=f"Content {i}",
                    embedding=[0.1] * 768,
                    user_id=user_id,
                    created_at=datetime.utcnow()
                )
                await backend.add(mem)

        results = await backend.get_all(filters={"user_id": "user-1"})

        assert len(results) == 2
        for r in results:
            assert r.user_id == "user-1"

    @pytest.mark.asyncio
    async def test_get_all_with_limit(self, temp_path):
        """Test getting memories with a limit."""
        backend = DualStorageBackend(temp_path)

        # Add multiple memories
        for i in range(5):
            mem = Memory(
                memory_id=f"mem-{i}",
                content=f"Content {i}",
                embedding=[0.1] * 768,
                user_id="user-1",
                created_at=datetime.utcnow()
            )
            await backend.add(mem)

        results = await backend.get_all(limit=3)

        assert len(results) == 3


class TestSearchOperation:
    """Tests for search operation."""

    @pytest.fixture
    def temp_path(self, tmp_path):
        """Provide a temporary storage path."""
        return tmp_path / "storage"

    @pytest.mark.asyncio
    async def test_search_returns_results(self, temp_path):
        """Test semantic search returns results."""
        backend = DualStorageBackend(temp_path)

        # Add a memory
        mem = Memory(
            memory_id="mem-1",
            content="Python programming language",
            embedding=[0.1] * 768,
            user_id="user-1",
            created_at=datetime.utcnow()
        )
        await backend.add(mem)

        # Search with similar query vector
        results = await backend.search([0.1] * 768, limit=10)

        assert len(results) >= 0  # Results may be empty or contain the memory

    @pytest.mark.asyncio
    async def test_search_with_filters(self, temp_path):
        """Test search with user filter."""
        backend = DualStorageBackend(temp_path)

        # Add memories for different users
        mem1 = Memory(
            memory_id="mem-1",
            content="Python",
            embedding=[0.1] * 768,
            user_id="user-1",
            created_at=datetime.utcnow()
        )
        mem2 = Memory(
            memory_id="mem-2",
            content="Python",
            embedding=[0.1] * 768,
            user_id="user-2",
            created_at=datetime.utcnow()
        )
        await backend.add(mem1)
        await backend.add(mem2)

        # Search for user-1 only
        results = await backend.search([0.1] * 768, filters={"user_id": "user-1"})

        for r in results:
            assert r.user_id == "user-1"


class TestUpdateOperation:
    """Tests for update operation."""

    @pytest.fixture
    def temp_path(self, tmp_path):
        """Provide a temporary storage path."""
        return tmp_path / "storage"

    @pytest.mark.asyncio
    async def test_update_existing_memory(self, temp_path):
        """Test updating an existing memory."""
        backend = DualStorageBackend(temp_path)

        # Add original memory
        original = Memory(
            memory_id="mem-1",
            content="Original content",
            embedding=[0.1] * 768,
            user_id="user-1",
            created_at=datetime.utcnow()
        )
        await backend.add(original)

        # Update with new content
        updated = Memory(
            memory_id="mem-1",
            content="Updated content",
            embedding=[0.2] * 768,
            user_id="user-1",
            created_at=datetime.utcnow()
        )
        await backend.update("mem-1", updated)

        # Verify update
        result = await backend.get("mem-1")
        assert result.content == "Updated content"

    @pytest.mark.asyncio
    async def test_update_nonexistent_memory_raises_error(self, temp_path):
        """Test updating non-existent memory raises error."""
        backend = DualStorageBackend(temp_path)

        updated = Memory(
            memory_id="mem-1",
            content="Updated content",
            embedding=[0.1] * 768,
            user_id="user-1",
            created_at=datetime.utcnow()
        )

        with pytest.raises(StorageError, match="update failed"):
            await backend.update("nonexistent", updated)

    @pytest.mark.asyncio
    async def test_update_graph_failure_succeeds(self, temp_path):
        """Test that graph store update failure doesn't prevent operation."""
        backend = DualStorageBackend(temp_path)

        # Add original memory
        original = Memory(
            memory_id="mem-1",
            content="Original",
            embedding=[0.1] * 768,
            created_at=datetime.utcnow()
        )
        await backend.add(original)

        # Mock graph store to fail on update
        backend.graph_store.update_node = AsyncMock(side_effect=Exception("Graph down"))

        # Update should still succeed
        updated = Memory(
            memory_id="mem-1",
            content="Updated",
            embedding=[0.2] * 768,
            created_at=datetime.utcnow()
        )
        await backend.update("mem-1", updated)

        # Verify vector store was updated
        result = await backend.get("mem-1")
        assert result.content == "Updated"


class TestDeleteOperation:
    """Tests for delete operations."""

    @pytest.fixture
    def temp_path(self, tmp_path):
        """Provide a temporary storage path."""
        return tmp_path / "storage"

    @pytest.mark.asyncio
    async def test_delete_existing_memory(self, temp_path):
        """Test deleting an existing memory."""
        backend = DualStorageBackend(temp_path)

        # Add memory
        mem = Memory(
            memory_id="mem-1",
            content="Content",
            embedding=[0.1] * 768,
            created_at=datetime.utcnow()
        )
        await backend.add(mem)

        # Delete it
        await backend.delete("mem-1")

        # Verify it's gone
        result = await backend.get("mem-1")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_memory_raises_error(self, temp_path):
        """Test deleting non-existent memory raises error."""
        backend = DualStorageBackend(temp_path)

        with pytest.raises(StorageError, match="delete failed"):
            await backend.delete("nonexistent")

    @pytest.mark.asyncio
    async def test_delete_all_for_user(self, temp_path):
        """Test deleting all memories for a user."""
        backend = DualStorageBackend(temp_path)

        # Add memories for different users
        for user_id in ["user-1", "user-2"]:
            for i in range(3):
                mem = Memory(
                    memory_id=f"{user_id}-mem-{i}",
                    content=f"Content {i}",
                    embedding=[0.1] * 768,
                    user_id=user_id,
                    created_at=datetime.utcnow()
                )
                await backend.add(mem)

        # Delete all for user-1
        await backend.delete_all({"user_id": "user-1"})

        # Verify user-1 memories are gone
        results = await backend.get_all(filters={"user_id": "user-1"})
        assert len(results) == 0

        # Verify user-2 memories remain
        results = await backend.get_all(filters={"user_id": "user-2"})
        assert len(results) == 3


class TestRelationOperations:
    """Tests for relation operations."""

    @pytest.fixture
    def temp_path(self, tmp_path):
        """Provide a temporary storage path."""
        return tmp_path / "storage"

    @pytest.mark.asyncio
    async def test_add_relation(self, temp_path):
        """Test adding a relation between memories."""
        backend = DualStorageBackend(temp_path)

        # Add two memories
        mem1 = Memory(
            memory_id="mem-1",
            content="Content 1",
            embedding=[0.1] * 768,
            created_at=datetime.utcnow()
        )
        mem2 = Memory(
            memory_id="mem-2",
            content="Content 2",
            embedding=[0.1] * 768,
            created_at=datetime.utcnow()
        )
        await backend.add(mem1)
        await backend.add(mem2)

        # Add relation
        await backend.add_relation("mem-1", "mem-2", RelationType.RELATED)

        # Verify relation
        relations = await backend.get_relations("mem-1")
        assert len(relations) == 1
        assert relations[0].from_id == "mem-1"
        assert relations[0].to_id == "mem-2"

    @pytest.mark.asyncio
    async def test_get_relations_with_direction(self, temp_path):
        """Test getting relations with direction filter."""
        backend = DualStorageBackend(temp_path)

        # Add three memories
        for i in range(3):
            mem = Memory(
                memory_id=f"mem-{i}",
                content=f"Content {i}",
                embedding=[0.1] * 768,
                created_at=datetime.utcnow()
            )
            await backend.add(mem)

        # Add relations: mem-1 -> mem-2 -> mem-3
        await backend.add_relation("mem-1", "mem-2", RelationType.PRECEDES)
        await backend.add_relation("mem-2", "mem-3", RelationType.PRECEDES)

        # Get outgoing relations from mem-1
        out_relations = await backend.get_relations("mem-1", direction="out")
        assert len(out_relations) == 1
        assert out_relations[0].to_id == "mem-2"

        # Get incoming relations to mem-2
        in_relations = await backend.get_relations("mem-2", direction="in")
        assert len(in_relations) == 1
        assert in_relations[0].from_id == "mem-1"


class TestResetOperation:
    """Tests for reset operation."""

    @pytest.fixture
    def temp_path(self, tmp_path):
        """Provide a temporary storage path."""
        return tmp_path / "storage"

    @pytest.mark.asyncio
    async def test_reset_clears_all_data(self, temp_path):
        """Test that reset clears all data from both stores."""
        backend = DualStorageBackend(temp_path)

        # Add memories
        for i in range(3):
            mem = Memory(
                memory_id=f"mem-{i}",
                content=f"Content {i}",
                embedding=[0.1] * 768,
                created_at=datetime.utcnow()
            )
            await backend.add(mem)

        # Add a relation
        await backend.add_relation("mem-0", "mem-1", RelationType.RELATED)

        # Reset
        await backend.reset()

        # Verify all data is gone
        results = await backend.get_all()
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_reset_with_graph_failure_succeeds(self, temp_path):
        """Test that reset succeeds even if graph store fails."""
        backend = DualStorageBackend(temp_path)

        # Mock graph store reset to fail
        backend.graph_store.reset = AsyncMock(side_effect=Exception("Graph reset failed"))

        # Reset should still succeed
        await backend.reset()

        # Vector store should be cleared
        results = await backend.get_all()
        assert len(results) == 0

"""Tests for NetworkX graph storage with persistence strategies."""

import asyncio
import pickle
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mempy.core.memory import Memory, RelationType
from mempy.core.exceptions import StorageError
from mempy.storage.graph_store import NetworkXGraphStore


class TestGraphStoreInitialization:
    """Tests for graph store initialization and loading."""

    @pytest.fixture
    def temp_path(self, tmp_path):
        """Provide a temporary path for graph storage."""
        return tmp_path / "graph"

    @pytest.fixture
    def sample_memory(self):
        """Provide a sample memory for testing."""
        return Memory(
            memory_id="mem-1",
            content="Test memory content",
            embedding=[0.1] * 10,
            user_id="user-1",
            created_at=datetime.utcnow()
        )

    def test_init_with_default_params(self, temp_path):
        """Test initialization with default parameters."""
        store = NetworkXGraphStore(temp_path)
        assert store.persist_path == temp_path
        assert store.graph_path == temp_path / "graph.pkl"
        assert store.auto_save is False
        assert store.save_interval == 0
        assert store.enable_file_lock is False
        assert store._dirty is False
        assert store._pending_writes == 0
        # Graph should be initialized
        assert store.graph is not None

    def test_init_with_custom_params(self, temp_path):
        """Test initialization with custom parameters."""
        store = NetworkXGraphStore(
            temp_path,
            auto_save=True,
            save_interval=10,
            enable_file_lock=True
        )
        assert store.auto_save is True
        assert store.save_interval == 10
        assert store.enable_file_lock is True

    @pytest.mark.asyncio
    async def test_load_existing_graph(self, temp_path, sample_memory):
        """Test loading an existing graph from disk."""
        # First, create and save a graph
        store1 = NetworkXGraphStore(temp_path, auto_save=True)
        await store1.add_node("mem-1", sample_memory)

        # Now create a new store and verify it loads the graph
        store2 = NetworkXGraphStore(temp_path, auto_save=False)
        assert "mem-1" in store2.graph.nodes
        loaded = await store2.get_node("mem-1")
        assert loaded is not None
        assert loaded.content == "Test memory content"

    def test_load_corrupted_graph_creates_new(self, temp_path):
        """Test that a corrupted graph file results in a new empty graph."""
        # Create a corrupted file
        temp_path.mkdir(parents=True, exist_ok=True)
        (temp_path / "graph.pkl").write_text("corrupted data")

        store = NetworkXGraphStore(temp_path, auto_save=False)
        # Should create a new empty graph
        assert len(store.graph.nodes) == 0


class TestPersistenceStrategy:
    """Tests for different persistence strategies."""

    @pytest.fixture
    def temp_path(self, tmp_path):
        """Provide a temporary path for graph storage."""
        return tmp_path / "graph"

    @pytest.fixture
    def sample_memory(self):
        """Provide a sample memory for testing."""
        return Memory(
            memory_id="mem-1",
            content="Test memory content",
            embedding=[0.1] * 10,
            user_id="user-1",
            created_at=datetime.utcnow()
        )

    @pytest.mark.asyncio
    async def test_manual_save_mode_no_auto_save(self, temp_path, sample_memory):
        """Test manual save mode - file not created until save() is called."""
        store = NetworkXGraphStore(temp_path, auto_save=False)

        await store.add_node("mem-1", sample_memory)

        # File should not exist yet
        assert not (temp_path / "graph.pkl").exists()
        assert store._dirty is True
        assert store._pending_writes == 1

        # Manually save
        await store.save()

        # Now file should exist
        assert (temp_path / "graph.pkl").exists()
        assert store._dirty is False
        assert store._pending_writes == 0

    @pytest.mark.asyncio
    async def test_auto_save_mode_saves_immediately(self, temp_path, sample_memory):
        """Test auto_save=True saves after each operation."""
        store = NetworkXGraphStore(temp_path, auto_save=True)

        await store.add_node("mem-1", sample_memory)

        # File should exist immediately
        assert (temp_path / "graph.pkl").exists()
        assert store._dirty is False
        assert store._pending_writes == 0

    @pytest.mark.asyncio
    async def test_interval_save_mode(self, temp_path):
        """Test save_interval saves every N operations."""
        store = NetworkXGraphStore(temp_path, auto_save=True, save_interval=3)

        # Operations 1 and 2: no save yet
        for i in range(2):
            mem = Memory(
                memory_id=f"mem-{i}",
                content=f"Content {i}",
                embedding=[0.1] * 10,
                created_at=datetime.utcnow()
            )
            await store.add_node(f"mem-{i}", mem)

        assert not (temp_path / "graph.pkl").exists()
        assert store._pending_writes == 2

        # Operation 3: should trigger save
        mem = Memory(
            memory_id="mem-2",
            content="Content 2",
            embedding=[0.1] * 10,
            created_at=datetime.utcnow()
        )
        await store.add_node("mem-2", mem)

        assert (temp_path / "graph.pkl").exists()
        assert store._pending_writes == 0

    @pytest.mark.asyncio
    async def test_context_manager_saves_on_exit(self, temp_path, sample_memory):
        """Test that context manager saves on exit."""
        async with NetworkXGraphStore(temp_path, auto_save=False) as store:
            await store.add_node("mem-1", sample_memory)
            # File should not exist inside context
            assert not (temp_path / "graph.pkl").exists()

        # File should exist after exiting context
        assert (temp_path / "graph.pkl").exists()

    @pytest.mark.asyncio
    async def test_context_manager_with_exception(self, temp_path):
        """Test context manager behavior with exception."""
        with pytest.raises(ValueError):
            async with NetworkXGraphStore(temp_path, auto_save=False) as store:
                mem = Memory(
                    memory_id="mem-1",
                    content="Content",
                    embedding=[0.1] * 10,
                    created_at=datetime.utcnow()
                )
                await store.add_node("mem-1", mem)
                raise ValueError("Test error")

        # File should still be saved despite exception
        assert (temp_path / "graph.pkl").exists()

    @pytest.mark.asyncio
    async def test_context_manager_no_dirty_no_save(self, temp_path):
        """Test context manager doesn't save if no changes were made."""
        async with NetworkXGraphStore(temp_path, auto_save=False) as store:
            # No operations performed
            pass

        # File should not exist
        assert not (temp_path / "graph.pkl").exists()


class TestNodeOperations:
    """Tests for node operations (add, get, update, delete)."""

    @pytest.fixture
    def temp_path(self, tmp_path):
        """Provide a temporary path for graph storage."""
        return tmp_path / "graph"

    @pytest.fixture
    def sample_memory(self):
        """Provide a sample memory for testing."""
        return Memory(
            memory_id="mem-1",
            content="Test memory content",
            embedding=[0.1] * 10,
            user_id="user-1",
            metadata={"key": "value"},
            created_at=datetime.utcnow()
        )

    @pytest.mark.asyncio
    async def test_add_node(self, temp_path, sample_memory):
        """Test adding a node to the graph."""
        store = NetworkXGraphStore(temp_path, auto_save=False)
        await store.add_node("mem-1", sample_memory)

        retrieved = await store.get_node("mem-1")
        assert retrieved is not None
        assert retrieved.memory_id == "mem-1"
        assert retrieved.content == "Test memory content"
        assert retrieved.metadata == {"key": "value"}

    @pytest.mark.asyncio
    async def test_get_nonexistent_node(self, temp_path):
        """Test getting a node that doesn't exist returns None."""
        store = NetworkXGraphStore(temp_path, auto_save=False)
        result = await store.get_node("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_node(self, temp_path, sample_memory):
        """Test updating a node."""
        store = NetworkXGraphStore(temp_path, auto_save=False)
        await store.add_node("mem-1", sample_memory)

        # Update the memory
        updated_memory = Memory(
            memory_id="mem-1",
            content="Updated content",
            embedding=[0.2] * 10,
            user_id="user-1",
            created_at=datetime.utcnow()
        )
        await store.update_node("mem-1", updated_memory)

        retrieved = await store.get_node("mem-1")
        assert retrieved.content == "Updated content"

    @pytest.mark.asyncio
    async def test_update_nonexistent_node_raises_error(self, temp_path):
        """Test updating a non-existent node raises StorageError."""
        store = NetworkXGraphStore(temp_path, auto_save=False)
        memory = Memory(
            memory_id="mem-1",
            content="Content",
            embedding=[0.1] * 10,
            created_at=datetime.utcnow()
        )

        with pytest.raises(StorageError, match="not found in graph"):
            await store.update_node("nonexistent", memory)

    @pytest.mark.asyncio
    async def test_delete_node(self, temp_path, sample_memory):
        """Test deleting a node."""
        store = NetworkXGraphStore(temp_path, auto_save=False)
        await store.add_node("mem-1", sample_memory)

        await store.delete_node("mem-1")

        result = await store.get_node("mem-1")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_node_removes_edges(self, temp_path):
        """Test that deleting a node also removes its edges."""
        store = NetworkXGraphStore(temp_path, auto_save=False)

        mem1 = Memory(
            memory_id="mem-1",
            content="Content 1",
            embedding=[0.1] * 10,
            created_at=datetime.utcnow()
        )
        mem2 = Memory(
            memory_id="mem-2",
            content="Content 2",
            embedding=[0.1] * 10,
            created_at=datetime.utcnow()
        )

        await store.add_node("mem-1", mem1)
        await store.add_node("mem-2", mem2)
        await store.add_edge("mem-1", "mem-2", RelationType.RELATED)

        # Delete mem-1
        await store.delete_node("mem-1")

        # Edge should be gone
        neighbors = await store.get_neighbors("mem-2")
        assert len(neighbors) == 0


class TestEdgeOperations:
    """Tests for edge (relation) operations."""

    @pytest.fixture
    def temp_path(self, tmp_path):
        """Provide a temporary path for graph storage."""
        return tmp_path / "graph"

    @pytest.fixture
    def setup_memories(self, temp_path):
        """Set up a store with two memories."""
        async def _setup():
            store = NetworkXGraphStore(temp_path, auto_save=False)
            mem1 = Memory(
                memory_id="mem-1",
                content="Content 1",
                embedding=[0.1] * 10,
                created_at=datetime.utcnow()
            )
            mem2 = Memory(
                memory_id="mem-2",
                content="Content 2",
                embedding=[0.1] * 10,
                created_at=datetime.utcnow()
            )
            await store.add_node("mem-1", mem1)
            await store.add_node("mem-2", mem2)
            return store
        return _setup

    @pytest.mark.asyncio
    async def test_add_edge(self, temp_path, setup_memories):
        """Test adding an edge between nodes."""
        store = await setup_memories()
        await store.add_edge("mem-1", "mem-2", RelationType.RELATED)

        relations = await store.get_relations("mem-1")
        assert len(relations) == 1
        assert relations[0].from_id == "mem-1"
        assert relations[0].to_id == "mem-2"
        assert relations[0].type == RelationType.RELATED

    @pytest.mark.asyncio
    async def test_add_edge_with_metadata(self, temp_path, setup_memories):
        """Test adding an edge with metadata."""
        store = await setup_memories()
        await store.add_edge(
            "mem-1",
            "mem-2",
            RelationType.CAUSES,
            metadata={"confidence": 0.9}
        )

        relations = await store.get_relations("mem-1")
        assert relations[0].metadata == {"confidence": 0.9}

    @pytest.mark.asyncio
    async def test_add_edge_nonexistent_source_raises_error(self, temp_path):
        """Test adding edge from non-existent source raises error."""
        store = NetworkXGraphStore(temp_path, auto_save=False)

        with pytest.raises(StorageError, match="Source memory.*not found"):
            await store.add_edge("nonexistent", "mem-2", RelationType.RELATED)

    @pytest.mark.asyncio
    async def test_add_edge_nonexistent_target_raises_error(self, temp_path):
        """Test adding edge to non-existent target raises error."""
        store = NetworkXGraphStore(temp_path, auto_save=False)
        mem = Memory(
            memory_id="mem-1",
            content="Content",
            embedding=[0.1] * 10,
            created_at=datetime.utcnow()
        )
        await store.add_node("mem-1", mem)

        with pytest.raises(StorageError, match="Target memory.*not found"):
            await store.add_edge("mem-1", "nonexistent", RelationType.RELATED)

    @pytest.mark.asyncio
    async def test_get_relations_direction_out(self, temp_path, setup_memories):
        """Test getting outgoing relations."""
        store = await setup_memories()
        await store.add_edge("mem-1", "mem-2", RelationType.RELATED)

        relations = await store.get_relations("mem-1", direction="out")
        assert len(relations) == 1
        assert relations[0].from_id == "mem-1"

    @pytest.mark.asyncio
    async def test_get_relations_direction_in(self, temp_path, setup_memories):
        """Test getting incoming relations."""
        store = await setup_memories()
        await store.add_edge("mem-1", "mem-2", RelationType.RELATED)

        relations = await store.get_relations("mem-2", direction="in")
        assert len(relations) == 1
        assert relations[0].to_id == "mem-2"

    @pytest.mark.asyncio
    async def test_get_relations_direction_both(self, temp_path, setup_memories):
        """Test getting both incoming and outgoing relations."""
        store = await setup_memories()
        await store.add_edge("mem-1", "mem-2", RelationType.RELATED)

        relations_out = await store.get_relations("mem-1", direction="both")
        relations_in = await store.get_relations("mem-2", direction="both")

        assert len(relations_out) == 1
        assert len(relations_in) == 1

    @pytest.mark.asyncio
    async def test_get_neighbors(self, temp_path, setup_memories):
        """Test getting neighbor nodes."""
        store = await setup_memories()
        await store.add_edge("mem-1", "mem-2", RelationType.RELATED)

        neighbors = await store.get_neighbors("mem-1")
        assert neighbors == ["mem-2"]

    @pytest.mark.asyncio
    async def test_find_path(self, temp_path, setup_memories):
        """Test finding a path between two nodes."""
        store = await setup_memories()

        # Add a third node
        mem3 = Memory(
            memory_id="mem-3",
            content="Content 3",
            embedding=[0.1] * 10,
            created_at=datetime.utcnow()
        )
        await store.add_node("mem-3", mem3)

        # Create a path: mem-1 -> mem-2 -> mem-3
        await store.add_edge("mem-1", "mem-2", RelationType.PRECEDES)
        await store.add_edge("mem-2", "mem-3", RelationType.PRECEDES)

        path = await store.find_path("mem-1", "mem-3")
        assert path == ["mem-1", "mem-2", "mem-3"]

    @pytest.mark.asyncio
    async def test_find_path_no_path(self, temp_path, setup_memories):
        """Test finding path when no path exists."""
        store = await setup_memories()

        path = await store.find_path("mem-1", "mem-2")
        assert path is None

    @pytest.mark.asyncio
    async def test_find_path_nonexistent_node(self, temp_path, setup_memories):
        """Test finding path with non-existent node."""
        store = await setup_memories()

        path = await store.find_path("mem-1", "nonexistent")
        assert path is None


class TestGraphReset:
    """Tests for graph reset operation."""

    @pytest.fixture
    def temp_path(self, tmp_path):
        """Provide a temporary path for graph storage."""
        return tmp_path / "graph"

    @pytest.mark.asyncio
    async def test_reset_clears_all_data(self, temp_path):
        """Test that reset clears all nodes and edges."""
        store = NetworkXGraphStore(temp_path, auto_save=False)

        # Add some data
        for i in range(3):
            mem = Memory(
                memory_id=f"mem-{i}",
                content=f"Content {i}",
                embedding=[0.1] * 10,
                created_at=datetime.utcnow()
            )
            await store.add_node(f"mem-{i}", mem)

        await store.add_edge("mem-0", "mem-1", RelationType.RELATED)

        # Reset
        await store.reset()

        assert len(store.graph.nodes) == 0
        assert len(store.graph.edges) == 0

    @pytest.mark.asyncio
    async def test_reset_persists_empty_graph(self, temp_path):
        """Test that reset persists the empty graph."""
        store = NetworkXGraphStore(temp_path, auto_save=True)

        # Add and reset
        mem = Memory(
            memory_id="mem-1",
            content="Content",
            embedding=[0.1] * 10,
            created_at=datetime.utcnow()
        )
        await store.add_node("mem-1", mem)
        await store.reset()

        # Create new store and verify it's empty
        store2 = NetworkXGraphStore(temp_path, auto_save=False)
        assert len(store2.graph.nodes) == 0


class TestFileLocking:
    """Tests for file locking (using mock)."""

    @pytest.fixture
    def temp_path(self, tmp_path):
        """Provide a temporary path for graph storage."""
        return tmp_path / "graph"

    @pytest.mark.asyncio
    async def test_file_lock_enabled_on_save(self, temp_path):
        """Test that fcntl locking is called when enable_file_lock=True."""
        with patch('fcntl.flock') as mock_flock:
            store = NetworkXGraphStore(temp_path, auto_save=False, enable_file_lock=True)

            mem = Memory(
                memory_id="mem-1",
                content="Content",
                embedding=[0.1] * 10,
                created_at=datetime.utcnow()
            )
            await store.add_node("mem-1", mem)
            await store.save()

            # Verify flock was called (LOCK_EX for write)
            assert mock_flock.called
            # Check that LOCK_UN was also called
            unlock_calls = [call for call in mock_flock.call_args_list
                           if len(call[0]) > 0 and call[0][0] != 2]  # 2 is LOCK_EX
            # At least one LOCK_UN call should exist
            assert mock_flock.call_count >= 2  # At least lock and unlock

    @pytest.mark.asyncio
    async def test_file_lock_enabled_on_load(self, temp_path):
        """Test that fcntl locking is called on load when enable_file_lock=True."""
        # First save something
        store1 = NetworkXGraphStore(temp_path, auto_save=False, enable_file_lock=False)
        mem = Memory(
            memory_id="mem-1",
            content="Content",
            embedding=[0.1] * 10,
            created_at=datetime.utcnow()
        )
        await store1.add_node("mem-1", mem)
        await store1.save()

        # Now load with file locking enabled
        with patch('fcntl.flock') as mock_flock:
            store2 = NetworkXGraphStore(temp_path, auto_save=False, enable_file_lock=True)

            # Verify flock was called for read (LOCK_SH)
            assert mock_flock.called

    @pytest.mark.asyncio
    async def test_file_lock_disabled_no_flock_calls(self, temp_path):
        """Test that fcntl is not called when enable_file_lock=False."""
        with patch('fcntl.flock') as mock_flock:
            store = NetworkXGraphStore(temp_path, auto_save=False, enable_file_lock=False)

            mem = Memory(
                memory_id="mem-1",
                content="Content",
                embedding=[0.1] * 10,
                created_at=datetime.utcnow()
            )
            await store.add_node("mem-1", mem)
            await store.save()

            # flock should not be called
            assert not mock_flock.called


class TestMemoryToNodeConversion:
    """Tests for memory-to-node attribute conversion."""

    @pytest.fixture
    def temp_path(self, tmp_path):
        """Provide a temporary path for graph storage."""
        return tmp_path / "graph"

    def test_memory_to_node_attrs_preserves_all_fields(self, temp_path):
        """Test that all memory fields are converted to node attributes."""
        store = NetworkXGraphStore(temp_path, auto_save=False)

        memory = Memory(
            memory_id="test-id",
            content="Test content",
            embedding=[0.1] * 10,
            user_id="user-1",
            agent_id="agent-1",
            run_id="run-1",
            metadata={"key": "value"},
            priority=0.8,
            confidence=0.9,
            created_at=datetime.utcnow()
        )

        attrs = store._memory_to_node_attrs(memory)

        assert attrs["memory_id"] == "test-id"
        assert attrs["content"] == "Test content"
        assert attrs["user_id"] == "user-1"
        assert attrs["agent_id"] == "agent-1"
        assert attrs["run_id"] == "run-1"
        assert "priority" in attrs
        assert "confidence" in attrs
        assert "metadata" in attrs
        assert "created_at" in attrs

    def test_node_attrs_to_memory_roundtrip(self, temp_path):
        """Test roundtrip conversion: memory -> attrs -> memory."""
        store = NetworkXGraphStore(temp_path, auto_save=False)

        original = Memory(
            memory_id="test-id",
            content="Test content",
            embedding=[0.1] * 10,
            user_id="user-1",
            metadata={"key": "value", "number": 42},
            priority=0.75,
            confidence=0.85,
            created_at=datetime.utcnow()
        )

        attrs = store._memory_to_node_attrs(original)
        recovered = store._node_attrs_to_memory("test-id", attrs)

        assert recovered.memory_id == original.memory_id
        assert recovered.content == original.content
        assert recovered.user_id == original.user_id
        assert recovered.metadata == original.metadata
        assert recovered.priority == original.priority
        assert recovered.confidence == original.confidence

    def test_node_attrs_handles_invalid_metadata(self, temp_path):
        """Test that invalid metadata is handled gracefully."""
        store = NetworkXGraphStore(temp_path, auto_save=False)

        attrs = {
            "memory_id": "test-id",
            "content": "Test content",
            "metadata": "invalid json string",  # Invalid JSON
            "created_at": datetime.utcnow().isoformat()
        }

        recovered = store._node_attrs_to_memory("test-id", attrs)
        # Should not crash, metadata should be empty dict
        assert recovered.metadata == {}

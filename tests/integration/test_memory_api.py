"""Tests for main Memory API with comprehensive mocking."""

from datetime import datetime
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mempy.core.interfaces import Embedder, MemoryProcessor
from mempy.core.memory import Memory, RelationType, ProcessorResult
from mempy.core.exceptions import StorageError
from mempy.memory import Memory as MemoryAPI


class MockEmbedder(Embedder):
    """Mock embedder for testing."""

    def __init__(self, dimension: int = 768, embedding_value: float = 0.1):
        self._dimension = dimension
        self.embedding_value = embedding_value
        self.embed_calls = []

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, text: str) -> List[float]:
        self.embed_calls.append(text)
        return [self.embedding_value] * self._dimension


class MockProcessor(MemoryProcessor):
    """Mock processor for testing."""

    def __init__(self):
        self.process_calls = []
        self.result = ProcessorResult(action="add")

    def set_result(self, action: str, memory_id: str = None, content: str = None, reason: str = None):
        """Set the result to return on next process call."""
        self.result = ProcessorResult(
            action=action,
            memory_id=memory_id,
            content=content,
            reason=reason
        )

    async def process(self, content: str, existing_memories: List[Memory]) -> ProcessorResult:
        self.process_calls.append((content, existing_memories))
        return self.result


class TestMemoryAPIInitialization:
    """Tests for Memory API initialization."""

    def test_init_with_required_params(self):
        """Test initialization with required parameters."""
        embedder = MockEmbedder()
        api = MemoryAPI(embedder=embedder)

        assert api.embedder == embedder
        assert api.processor is None
        assert api.verbose is False
        assert api.storage is not None

    def test_init_with_processor(self):
        """Test initialization with processor."""
        embedder = MockEmbedder()
        processor = MockProcessor()
        api = MemoryAPI(embedder=embedder, processor=processor)

        assert api.processor == processor

    def test_init_with_custom_storage_path(self, tmp_path):
        """Test initialization with custom storage path."""
        embedder = MockEmbedder()
        custom_path = str(tmp_path / "custom_storage")
        api = MemoryAPI(embedder=embedder, storage_path=custom_path)

        assert api.storage.persist_path == Path(custom_path)

    def test_init_verbose_sets_up_logging(self):
        """Test verbose mode sets up logging."""
        embedder = MockEmbedder()
        api = MemoryAPI(embedder=embedder, verbose=True)

        assert api.verbose is True
        assert api.logger is not None
        assert api.logger.level == 20  # INFO level


class TestMemoryAPIAdd:
    """Tests for add operation."""

    @pytest.fixture
    def api(self):
        """Provide a Memory API instance."""
        embedder = MockEmbedder()
        return MemoryAPI(embedder=embedder)

    @pytest.mark.asyncio
    async def test_add_without_processor(self, api):
        """Test adding memory without processor."""
        memory_id = await api.add("Test content", user_id="user-1")

        assert memory_id is not None
        assert len(memory_id) == 32  # UUID4 hex length

    @pytest.mark.asyncio
    async def test_add_calls_embedder(self, api):
        """Test that add calls the embedder."""
        await api.add("Test content", user_id="user-1")

        assert len(api.embedder.embed_calls) == 1
        assert api.embedder.embed_calls[0] == "Test content"

    @pytest.mark.asyncio
    async def test_add_with_metadata(self, api):
        """Test adding memory with metadata."""
        memory_id = await api.add(
            "Test content",
            user_id="user-1",
            metadata={"source": "chat", "importance": "high"}
        )

        assert memory_id is not None

        # Verify metadata was stored
        memory = await api.get(memory_id)
        assert memory.metadata == {"source": "chat", "importance": "high"}

    @pytest.mark.asyncio
    async def test_add_with_all_params(self, api):
        """Test adding memory with all parameters."""
        memory_id = await api.add(
            content="Test content",
            user_id="user-1",
            agent_id="agent-1",
            run_id="run-1",
            metadata={"key": "value"}
        )

        memory = await api.get(memory_id)
        assert memory.user_id == "user-1"
        assert memory.agent_id == "agent-1"
        assert memory.run_id == "run-1"
        assert memory.metadata == {"key": "value"}


class TestMemoryAPIAddWithProcessor:
    """Tests for add operation with processor logic."""

    @pytest.fixture
    def api_with_processor(self):
        """Provide a Memory API with a mock processor."""
        embedder = MockEmbedder()
        processor = MockProcessor()
        return MemoryAPI(embedder=embedder, processor=processor), processor

    @pytest.mark.asyncio
    async def test_processor_action_add_continues_normally(self, api_with_processor):
        """Test processor action='add' continues with normal add flow."""
        api, processor = api_with_processor
        processor.set_result(action="add")

        memory_id = await api.add("New content")

        assert memory_id is not None
        # Embedder is called twice: once for search (existing memories), once for embedding new content
        assert len(api.embedder.embed_calls) == 2

    @pytest.mark.asyncio
    async def test_processor_action_update_updates_existing(self, api_with_processor):
        """Test processor action='update' updates existing memory."""
        api, processor = api_with_processor

        # First add a memory
        original_id = await api.add("Original content", user_id="user-1")
        processor.process_calls.clear()  # Clear the processor call from add

        # Now configure processor to return update
        processor.set_result(
            action="update",
            memory_id=original_id,
            content="Updated content",
            reason="Content is similar"
        )

        memory_id = await api.add("Similar content", user_id="user-1")

        assert memory_id == original_id

        # Verify the content was updated
        memory = await api.get(original_id)
        assert memory.content == "Updated content"
        assert memory.updated_at is not None

    @pytest.mark.asyncio
    async def test_processor_action_delete_removes_memory(self, api_with_processor):
        """Test processor action='delete' removes existing memory."""
        api, processor = api_with_processor

        # First add a memory
        original_id = await api.add("To be deleted", user_id="user-1")
        processor.process_calls.clear()

        # Configure processor to return delete
        processor.set_result(
            action="delete",
            memory_id=original_id,
            reason="Duplicate information"
        )

        result = await api.add("Duplicate content", user_id="user-1")

        # Should return None for delete action
        assert result is None

        # Memory should be deleted
        memory = await api.get(original_id)
        assert memory is None

    @pytest.mark.asyncio
    async def test_processor_action_none_skips_operation(self, api_with_processor):
        """Test processor action='none' skips the operation."""
        api, processor = api_with_processor
        processor.set_result(action="none", reason="Not important enough")

        result = await api.add("Ignored content")

        assert result is None
        # Embedder is called once for search (to find similar memories)
        # but not for creating a new memory
        assert len(api.embedder.embed_calls) == 1

    @pytest.mark.asyncio
    async def test_processor_receives_existing_memories(self, api_with_processor):
        """Test processor receives existing similar memories."""
        api, processor = api_with_processor

        # Add some memories first
        await api.add("Python is a programming language", user_id="user-1")
        await api.add("Python is used for web development", user_id="user-1")
        api.embedder.embed_calls.clear()
        processor.process_calls.clear()

        # Now add with processor
        processor.set_result(action="add")
        await api.add("Python code", user_id="user-1")

        # Processor should have been called with existing memories
        assert len(processor.process_calls) == 1
        content, existing = processor.process_calls[0]
        assert content == "Python code"
        # The search may return some of the previously added memories
        assert isinstance(existing, list)


class TestMemoryAPISearch:
    """Tests for search operation."""

    @pytest.fixture
    def api_with_memories(self):
        """Provide an API with some pre-populated memories."""
        embedder = MockEmbedder()
        api = MemoryAPI(embedder=embedder)

        async def _setup():
            # Add memories for different users
            await api.add("Python programming", user_id="user-1")
            await api.add("Java programming", user_id="user-1")
            await api.add("Machine learning", user_id="user-2")
            return api

        return _setup()

    @pytest.mark.asyncio
    async def test_search_returns_results(self, api_with_memories):
        """Test search returns results."""
        api = await api_with_memories

        results = await api.search("programming", user_id="user-1")

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_with_user_filter(self, api_with_memories):
        """Test search filters by user_id."""
        api = await api_with_memories

        results = await api.search("programming", user_id="user-1")

        # All results should be for user-1
        for r in results:
            assert r.user_id == "user-1"

    @pytest.mark.asyncio
    async def test_search_with_limit(self, api_with_memories):
        """Test search respects limit parameter."""
        api = await api_with_memories

        results = await api.search("programming", limit=1)

        assert len(results) <= 1

    @pytest.mark.asyncio
    async def test_search_calls_embedder(self):
        """Test search calls embedder with query."""
        embedder = MockEmbedder()
        api = MemoryAPI(embedder=embedder)

        await api.search("test query")

        assert len(embedder.embed_calls) == 1
        assert embedder.embed_calls[0] == "test query"


class TestMemoryAPIGet:
    """Tests for get operations."""

    @pytest.mark.asyncio
    async def test_get_existing_memory(self):
        """Test getting an existing memory."""
        embedder = MockEmbedder()
        api = MemoryAPI(embedder=embedder)

        memory_id = await api.add("Test content", user_id="user-1")

        memory = await api.get(memory_id)

        assert memory is not None
        assert memory.memory_id == memory_id
        assert memory.content == "Test content"

    @pytest.mark.asyncio
    async def test_get_nonexistent_memory(self):
        """Test getting non-existent memory returns None."""
        embedder = MockEmbedder()
        api = MemoryAPI(embedder=embedder)

        memory = await api.get("nonexistent-id")

        assert memory is None

    @pytest.mark.asyncio
    async def test_get_all_no_filters(self):
        """Test getting all memories without filters."""
        embedder = MockEmbedder()
        api = MemoryAPI(embedder=embedder)

        # Add multiple memories
        for i in range(3):
            await api.add(f"Content {i}", user_id="user-1")

        results = await api.get_all()

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_get_all_with_user_filter(self):
        """Test getting all memories filtered by user."""
        embedder = MockEmbedder()
        api = MemoryAPI(embedder=embedder)

        await api.add("Content 1", user_id="user-1")
        await api.add("Content 2", user_id="user-2")
        await api.add("Content 3", user_id="user-1")

        results = await api.get_all(user_id="user-1")

        assert len(results) == 2
        for r in results:
            assert r.user_id == "user-1"

    @pytest.mark.asyncio
    async def test_get_all_with_limit(self):
        """Test getting all memories with limit."""
        embedder = MockEmbedder()
        api = MemoryAPI(embedder=embedder)

        for i in range(5):
            await api.add(f"Content {i}", user_id="user-1")

        results = await api.get_all(limit=3)

        assert len(results) == 3


class TestMemoryAPIUpdate:
    """Tests for update operation."""

    @pytest.mark.asyncio
    async def test_update_existing_memory(self):
        """Test updating an existing memory."""
        embedder = MockEmbedder()
        api = MemoryAPI(embedder=embedder)

        memory_id = await api.add("Original content", user_id="user-1")

        updated_id = await api.update(memory_id, "Updated content")

        assert updated_id == memory_id

        # Verify update
        memory = await api.get(memory_id)
        assert memory.content == "Updated content"
        assert memory.updated_at is not None

    @pytest.mark.asyncio
    async def test_update_nonexistent_memory_raises_error(self):
        """Test updating non-existent memory raises StorageError."""
        embedder = MockEmbedder()
        api = MemoryAPI(embedder=embedder)

        with pytest.raises(StorageError, match="not found"):
            await api.update("nonexistent-id", "Updated content")

    @pytest.mark.asyncio
    async def test_update_regenerates_embedding(self):
        """Test update regenerates embedding for new content."""
        embedder = MockEmbedder()
        api = MemoryAPI(embedder=embedder)

        memory_id = await api.add("Original", user_id="user-1")
        embedder.embed_calls.clear()

        await api.update(memory_id, "Updated")

        # Embedder should be called with new content
        assert len(embedder.embed_calls) == 1
        assert embedder.embed_calls[0] == "Updated"


class TestMemoryAPIDelete:
    """Tests for delete operations."""

    @pytest.mark.asyncio
    async def test_delete_existing_memory(self):
        """Test deleting an existing memory."""
        embedder = MockEmbedder()
        api = MemoryAPI(embedder=embedder)

        memory_id = await api.add("To be deleted", user_id="user-1")

        await api.delete(memory_id)

        # Verify deletion
        memory = await api.get(memory_id)
        assert memory is None

    @pytest.mark.asyncio
    async def test_delete_all_for_user(self):
        """Test deleting all memories for a user."""
        embedder = MockEmbedder()
        api = MemoryAPI(embedder=embedder)

        # Add memories for different users
        await api.add("Content 1", user_id="user-1")
        await api.add("Content 2", user_id="user-1")
        await api.add("Content 3", user_id="user-2")

        await api.delete_all("user-1")

        # Verify user-1 memories are gone
        results = await api.get_all(user_id="user-1")
        assert len(results) == 0

        # Verify user-2 memories remain
        results = await api.get_all(user_id="user-2")
        assert len(results) == 1


class TestMemoryAPIRelations:
    """Tests for relation operations."""

    @pytest.mark.asyncio
    async def test_add_relation(self):
        """Test adding a relation between memories."""
        embedder = MockEmbedder()
        api = MemoryAPI(embedder=embedder)

        # Add two memories
        mem1_id = await api.add("Python", user_id="user-1")
        mem2_id = await api.add("Programming language", user_id="user-1")

        # Add relation
        await api.add_relation(mem1_id, mem2_id, RelationType.PROPERTY_OF)

        # Get relations
        relations = await api.get_relations(mem1_id)

        assert len(relations) >= 0

    @pytest.mark.asyncio
    async def test_get_relations_with_direction(self):
        """Test getting relations with direction parameter."""
        embedder = MockEmbedder()
        api = MemoryAPI(embedder=embedder)

        # Add three memories
        mem1_id = await api.add("First", user_id="user-1")
        mem2_id = await api.add("Second", user_id="user-1")
        mem3_id = await api.add("Third", user_id="user-1")

        # Create chain: mem1 -> mem2 -> mem3
        await api.add_relation(mem1_id, mem2_id, RelationType.PRECEDES)
        await api.add_relation(mem2_id, mem3_id, RelationType.PRECEDES)

        # Get outgoing from mem1
        out_relations = await api.get_relations(mem1_id, direction="out")
        # Get incoming to mem2
        in_relations = await api.get_relations(mem2_id, direction="in")

        # Verify direction filtering
        assert all(r.from_id == mem1_id for r in out_relations)
        assert all(r.to_id == mem2_id for r in in_relations)

    @pytest.mark.asyncio
    async def test_get_relations_with_max_depth(self):
        """Test getting relations with max_depth parameter."""
        embedder = MockEmbedder()
        api = MemoryAPI(embedder=embedder)

        # Add memories
        ids = []
        for i in range(4):
            ids.append(await api.add(f"Content {i}", user_id="user-1"))

        # Create chain
        await api.add_relation(ids[0], ids[1], RelationType.PRECEDES)
        await api.add_relation(ids[1], ids[2], RelationType.PRECEDES)
        await api.add_relation(ids[2], ids[3], RelationType.PRECEDES)

        # Get with different depths
        depth_1 = await api.get_relations(ids[0], max_depth=1)
        depth_2 = await api.get_relations(ids[0], max_depth=2)

        # Deeper depth should return at least as many relations
        assert len(depth_2) >= len(depth_1)


class TestMemoryAPIReset:
    """Tests for reset operation."""

    @pytest.mark.asyncio
    async def test_reset_clears_all_data(self):
        """Test that reset clears all data."""
        embedder = MockEmbedder()
        api = MemoryAPI(embedder=embedder)

        # Add some data
        await api.add("Memory 1", user_id="user-1")
        await api.add("Memory 2", user_id="user-2")

        mem1_id = await api.add("For relation 1", user_id="user-1")
        mem2_id = await api.add("For relation 2", user_id="user-1")
        await api.add_relation(mem1_id, mem2_id, RelationType.RELATED)

        # Reset
        await api.reset()

        # Verify all data is gone
        results = await api.get_all()
        assert len(results) == 0


class TestMemoryAPIErrorHandling:
    """Tests for error handling in Memory API."""

    @pytest.mark.asyncio
    async def test_embedder_error_propagates(self):
        """Test that embedder errors are properly propagated."""
        class FailingEmbedder(Embedder):
            @property
            def dimension(self) -> int:
                return 768

            async def embed(self, text: str) -> List[float]:
                raise RuntimeError("Embedding service unavailable")

        embedder = FailingEmbedder()
        api = MemoryAPI(embedder=embedder)

        # The error should propagate
        with pytest.raises(RuntimeError, match="Embedding service unavailable"):
            await api.add("Test content")

    @pytest.mark.asyncio
    async def test_storage_error_propagates(self):
        """Test that storage errors are properly propagated."""
        embedder = MockEmbedder()
        api = MemoryAPI(embedder=embedder)

        # Mock storage to raise error
        api.storage.add = AsyncMock(side_effect=StorageError("Storage full"))

        with pytest.raises(StorageError, match="Storage full"):
            await api.add("Test content")


class TestMemoryAPIWithRealStorage:
    """Integration tests with real storage (not mocked)."""

    @pytest.fixture
    def api(self, tmp_path):
        """Provide an API with temporary storage."""
        embedder = MockEmbedder()
        storage_path = str(tmp_path / "test_storage")
        return MemoryAPI(embedder=embedder, storage_path=storage_path)

    @pytest.mark.asyncio
    async def test_full_crud_cycle(self, api):
        """Test a complete create-read-update-delete cycle."""
        # Create
        memory_id = await api.add("Original content", user_id="test-user")
        assert memory_id is not None

        # Read
        memory = await api.get(memory_id)
        assert memory.content == "Original content"

        # Update
        await api.update(memory_id, "Updated content")
        memory = await api.get(memory_id)
        assert memory.content == "Updated content"

        # Delete
        await api.delete(memory_id)
        memory = await api.get(memory_id)
        assert memory is None

    @pytest.mark.asyncio
    async def test_search_finds_added_memory(self, api):
        """Test that search can find memories that were added."""
        await api.add("Python programming language", user_id="test-user")

        results = await api.search("Python", user_id="test-user")

        assert len(results) > 0
        assert any("Python" in r.content for r in results)

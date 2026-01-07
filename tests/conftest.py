"""pytest configuration and fixtures for mempy tests."""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock

import pytest

from mempy.core.memory import Memory, RelationType, ProcessorResult
from mempy.core.interfaces import Embedder, MemoryProcessor
from mempy.config import get_storage_path
from mempy.memory import Memory as MemoryAPI


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, isolated)")
    config.addinivalue_line("markers", "integration: Integration tests (module interaction)")
    config.addinivalue_line("markers", "api: Tests requiring API keys")
    config.addinivalue_line("markers", "slow: Slow-running tests")


# ============================================================================
# Event Loop Fixture
# ============================================================================

@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Mock Embedder Fixtures
# ============================================================================

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
        # Return deterministic vector based on text
        base = float(len(text) % 100) / 100.0
        return [base] * self._dimension


@pytest.fixture
def mock_embedder():
    """Provide a mock embedder for testing."""
    return MockEmbedder()


@pytest.fixture
def mock_embedder_small():
    """Provide a mock embedder with small dimension (for faster tests)."""
    return MockEmbedder(dimension=128, embedding_value=0.1)


# ============================================================================
# Mock Processor Fixtures
# ============================================================================

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


@pytest.fixture
def mock_processor():
    """Provide a mock processor for testing."""
    return MockProcessor()


# ============================================================================
# Memory Fixtures
# ============================================================================

@pytest.fixture
def sample_memory():
    """Provide a sample memory for testing."""
    return Memory(
        memory_id="test-memory-1",
        content="Test memory content",
        embedding=[0.1] * 768,
        user_id="test-user",
        metadata={"source": "test"}
    )


@pytest.fixture
def sample_memories():
    """Provide multiple sample memories."""
    return [
        Memory(
            memory_id=f"mem-{i}",
            content=f"Test content {i}",
            embedding=[0.1] * 768,
            user_id="test-user",
        )
        for i in range(5)
    ]


@pytest.fixture
def memory_with_relations():
    """Provide a memory with relations for testing."""
    return Memory(
        memory_id="mem-with-relations",
        content="Memory with relations",
        embedding=[0.1] * 768,
        user_id="test-user",
    )


# ============================================================================
# Storage Path Fixtures
# ============================================================================

@pytest.fixture
def temp_storage_path(tmp_path):
    """Provide a temporary storage path for testing."""
    storage_path = tmp_path / "mempy_data"
    storage_path.mkdir(exist_ok=True)
    return str(storage_path)


@pytest.fixture
def temp_graph_path(tmp_path):
    """Provide a temporary path for graph storage."""
    return tmp_path / "graph"


@pytest.fixture
def temp_vector_path(tmp_path):
    """Provide a temporary path for vector storage."""
    return tmp_path / "vector"


# ============================================================================
# Memory API Fixtures
# ============================================================================

@pytest.fixture
async def memory_api(temp_storage_path):
    """Provide a Memory API instance with temporary storage."""
    embedder = MockEmbedder(dimension=128)
    api = MemoryAPI(embedder=embedder, storage_path=temp_storage_path)
    yield api
    # Cleanup: reset after test
    await api.reset()


@pytest.fixture
async def memory_api_with_processor(temp_storage_path):
    """Provide a Memory API with processor and temporary storage."""
    embedder = MockEmbedder(dimension=128)
    processor = MockProcessor()
    api = MemoryAPI(
        embedder=embedder,
        processor=processor,
        storage_path=temp_storage_path
    )
    yield api, processor
    # Cleanup
    await api.reset()


@pytest.fixture
async def populated_memory_api(memory_api):
    """Provide a Memory API with pre-populated memories."""
    # Add some test memories
    await memory_api.add("Python is a programming language", user_id="user-1")
    await memory_api.add("Python is used for web development", user_id="user-1")
    await memory_api.add("Java is also a programming language", user_id="user-2")
    await memory_api.add("Machine learning is a subset of AI", user_id="user-1")
    return memory_api


# ============================================================================
# Async Context Manager Fixtures
# ============================================================================

@pytest.fixture
async def with_graph_store(temp_graph_path):
    """Context manager for graph store that auto-cleans."""
    from mempy.storage.graph_store import NetworkXGraphStore

    async with NetworkXGraphStore(temp_graph_path, auto_save=False) as store:
        yield store


# ============================================================================
# Utility Fixtures
# ============================================================================

@pytest.fixture
def current_time():
    """Provide the current datetime for consistent timestamps."""
    return datetime.utcnow()


@pytest.fixture
def unique_id():
    """Provide a unique ID for each test."""
    import uuid
    return uuid.uuid4().hex


# ============================================================================
# Parametrized Test Data Fixtures
# ============================================================================

@pytest.fixture(params=[
    RelationType.RELATED,
    RelationType.CAUSES,
    RelationType.PRECEDES,
    RelationType.PROPERTY_OF,
])
def relation_type(request):
    """Provide different relation types for parametrized tests."""
    return request.param


@pytest.fixture(params=["add", "update", "delete", "none"])
def processor_action(request):
    """Provide different processor actions for parametrized tests."""
    return request.param


# ============================================================================
# Mock Storage Fixtures
# ============================================================================

@pytest.fixture
def mock_storage():
    """Provide a mocked storage backend."""
    from mempy.storage.backend import DualStorageBackend
    from unittest.mock import MagicMock

    storage = MagicMock(spec=DualStorageBackend)
    storage.add = AsyncMock(return_value="mock-id-123")
    storage.get = AsyncMock()
    storage.get_all = AsyncMock(return_value=[])
    storage.search = AsyncMock(return_value=[])
    storage.update = AsyncMock()
    storage.delete = AsyncMock()
    storage.delete_all = AsyncMock()
    storage.add_relation = AsyncMock()
    storage.get_relations = AsyncMock(return_value=[])
    storage.reset = AsyncMock()

    return storage


# ============================================================================
# Test Data Builders
# ============================================================================

@pytest.fixture
def memory_builder():
    """Provide a builder function for creating test memories."""

    def _build(
        memory_id: str = None,
        content: str = "Default content",
        user_id: str = "test-user",
        embedding: List[float] = None,
        metadata: dict = None,
    ) -> Memory:
        import uuid
        from datetime import datetime

        if memory_id is None:
            memory_id = f"mem-{uuid.uuid4().hex[:8]}"
        if embedding is None:
            embedding = [0.1] * 128
        if metadata is None:
            metadata = {}

        return Memory(
            memory_id=memory_id,
            content=content,
            embedding=embedding,
            user_id=user_id,
            metadata=metadata,
            created_at=datetime.utcnow()
        )

    return _build


# ============================================================================
# Async Test Helpers
# ============================================================================

@pytest.fixture
def run_async():
    """Helper to run async functions in sync tests."""

    def _runner(coro):
        return asyncio.run(coro)

    return _runner

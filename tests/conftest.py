"""pytest configuration and fixtures for mempy tests."""

import asyncio
from pathlib import Path
import pytest
from typing import List

from mempy.core.memory import Memory
from mempy.core.interfaces import Embedder
from mempy.config import get_storage_path


class MockEmbedder(Embedder):
    """Mock embedder for testing."""

    def __init__(self, dimension: int = 768):
        self._dimension = dimension
        self.call_count = 0

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, text: str) -> List[float]:
        self.call_count += 1
        # Return a deterministic vector based on text length
        return [float(len(text) % 100) / 100.0] * self._dimension


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_embedder():
    """Provide a mock embedder for testing."""
    return MockEmbedder()


@pytest.fixture
def temp_storage_path(tmp_path):
    """Provide a temporary storage path for testing."""
    storage_path = tmp_path / "mempy_data"
    storage_path.mkdir(exist_ok=True)
    return str(storage_path)


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

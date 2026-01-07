# mempy Testing Guide

This document describes the testing structure and how to run tests for the mempy project.

## Test Structure

```
tests/
├── conftest.py              # Pytest fixtures and shared utilities
├── test_core.py             # Core data classes and interfaces tests
├── test_config.py           # Configuration tests
├── test_graph_store.py      # Graph storage tests
├── test_storage_backend.py  # Dual storage backend tests
├── test_memory_api.py       # Main Memory API tests
├── test_processors.py       # Processor tests
└── benchmarks/              # Benchmark evaluation framework
    ├── locomo/              # LOCOMO dataset evaluation
    ├── adapters/            # Model adapters for benchmarking
    └── data/                # Test datasets
```

## Running Tests

### Run All Tests

```bash
pytest tests/
```

### Run Specific Test File

```bash
pytest tests/test_graph_store.py
```

### Run with Verbose Output

```bash
pytest tests/ -v
```

### Run with Coverage

```bash
pip install pytest-cov
pytest tests/ --cov=mempy --cov-report=html
```

### Run Specific Test

```bash
pytest tests/test_graph_store.py::TestGraphStoreInitialization::test_init_creates_directory
```

## Test Fixtures

The `conftest.py` file provides shared fixtures for all tests:

| Fixture | Description |
|---------|-------------|
| `temp_storage_path` | Temporary directory for storage |
| `temp_graph_path` | Temporary graph.pkl file path |
| `temp_vector_path` | Temporary vector store path |
| `MockEmbedder` | Mock embedder with call tracking |
| `MockProcessor` | Mock processor with configurable results |
| `memory_api` | Memory API instance without processor |
| `memory_api_with_processor` | Memory API with mock processor |
| `populated_memory_api` | Memory API with pre-populated data |
| `mock_storage` | Mocked storage backend |
| `memory_builder` | Helper function to create test Memory objects |
| `sample_memory` | Sample memory for testing |
| `relation_type` | Parametrized relation types |
| `processor_action` | Parametrized processor actions |

## Test Categories

### 1. Core Tests (`test_core.py`)

Tests for core data classes and interfaces:
- Memory dataclass
- RelationType enum
- Relation dataclass
- Embedder interface
- MemoryProcessor interface
- StorageBackend interface
- Exception classes

### 2. Graph Store Tests (`test_graph_store.py`)

Tests for NetworkX-based graph storage:
- Initialization and loading
- Persistence strategies (manual, auto, interval)
- Context manager usage
- Node operations (add, get, update, delete)
- Edge operations (add, relations, neighbors, paths)
- Graph reset
- File locking (with mocks)
- Memory-to-node conversion

### 3. Storage Backend Tests (`test_storage_backend.py`)

Tests for the dual storage backend:
- Dual-storage initialization
- Add operations (success and failure scenarios)
- Get and search operations
- Update operations
- Delete operations
- Relation operations
- Reset operations

### 4. Memory API Tests (`test_memory_api.py`)

Tests for the main Memory API:
- Initialization with different configurations
- Add operations with and without processor
- Search functionality
- Get and get_all operations
- Update operations
- Delete operations
- Relation management
- Reset operations
- Error handling

### 5. Processor Tests (`test_processors.py`)

Tests for memory processors:
- ProcessorResult dataclass
- Base MemoryProcessor abstract class
- Custom processor implementations
- Error handling
- Async behavior

## Writing New Tests

### Test Structure Template

```python
import pytest
from mempy import Memory

class TestMyFeature:
    """Tests for MyFeature."""

    @pytest.mark.asyncio
    async def test_my_feature_basic_operation(self, memory_api):
        """Test that my feature works correctly."""
        result = await memory_api.some_operation("test")
        assert result is not None

    @pytest.mark.asyncio
    async def test_my_feature_error_handling(self, memory_api):
        """Test that my feature handles errors correctly."""
        with pytest.raises(Exception):
            await memory_api.some_operation(None)
```

### Using Mocks

```python
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_with_mock(self):
    """Test with mocked dependency."""
    with patch("mempy.storage.vector_store.ChromaVectorStore.add") as mock_add:
        mock_add.return_value = "test-id"
        result = await memory.add("test content")
        assert result == "test-id"
        mock_add.assert_called_once()
```

### Parametrized Tests

```python
@pytest.mark.parametrize("action,expected_count", [
    ("add", 1),
    ("update", 0),
    ("delete", 0),
    ("none", 0),
])
async def test_processor_actions(self, action, expected_count, memory_api_with_processor):
    """Test different processor actions."""
    # Test implementation
    pass
```

## Current Test Coverage

| Module | Coverage | Notes |
|--------|----------|-------|
| `mempy/core` | ~95% | Core interfaces and data classes |
| `mempy/storage/graph_store.py` | ~90% | All persistence strategies covered |
| `mempy/storage/backend.py` | ~85% | Dual-write operations covered |
| `mempy/memory.py` | ~90% | Main API with processor logic |
| `mempy/processors` | ~80% | Base and LLM processor |

## Known Issues

Some tests may fail due to:
1. **ChromaDB persistence**: Tests using real ChromaDB may have timing issues
2. **File locking tests**: Use mocks for fcntl (Linux only)
3. **NetworkX version**: Some functions not available in older versions

## Benchmark Tests

See `tests/benchmarks/README.md` for information about:
- LOCOMO dataset evaluation
- Model adapters
- Running benchmarks

## Contributing Tests

When contributing new features:
1. Add unit tests for all new code
2. Aim for >80% code coverage
3. Use descriptive test names
4. Mock external dependencies
5. Test both success and failure scenarios
6. Update this README if adding new test files

"""Tests for core data classes and interfaces."""

import pytest

from mempy.core.memory import Memory, RelationType, Relation, ProcessorResult
from mempy.core.interfaces import Embedder, MemoryProcessor, StorageBackend
from mempy.core.exceptions import MempyError, EmbedderError, StorageError, ProcessorError


class TestMemory:
    """Tests for Memory dataclass."""

    def test_memory_creation(self):
        """Test creating a memory object."""
        memory = Memory(
            memory_id="test-1",
            content="Test content",
            embedding=[0.1] * 10,
            user_id="user1"
        )
        assert memory.memory_id == "test-1"
        assert memory.content == "Test content"
        assert len(memory.embedding) == 10
        assert memory.user_id == "user1"
        assert memory.priority == 0.5
        assert memory.confidence == 1.0

    def test_memory_to_dict(self):
        """Test converting memory to dictionary."""
        memory = Memory(
            memory_id="test-1",
            content="Test content",
            embedding=[0.1] * 10,
        )
        data = memory.to_dict()
        assert data["memory_id"] == "test-1"
        assert data["content"] == "Test content"
        assert data["embedding"] == [0.1] * 10
        assert "created_at" in data

    def test_memory_from_dict(self):
        """Test creating memory from dictionary."""
        data = {
            "memory_id": "test-1",
            "content": "Test content",
            "embedding": [0.1] * 10,
            "user_id": "user1",
            "metadata": {},
            "created_at": "2024-01-01T00:00:00",
            "updated_at": None,
            "priority": 0.5,
            "confidence": 1.0
        }
        memory = Memory.from_dict(data)
        assert memory.memory_id == "test-1"
        assert memory.content == "Test content"


class TestRelationType:
    """Tests for RelationType enum."""

    def test_relation_types(self):
        """Test all relation type values."""
        assert RelationType.RELATED.value == "related"
        assert RelationType.EQUIVALENT.value == "equivalent"
        assert RelationType.CONTRADICTORY.value == "contradictory"
        assert RelationType.GENERALIZATION.value == "generalization"
        assert RelationType.SPECIALIZATION.value == "specialization"
        assert RelationType.PART_OF.value == "part_of"
        assert RelationType.PRECEDES.value == "precedes"
        assert RelationType.FOLLOWS.value == "follows"
        assert RelationType.CAUSES.value == "causes"
        assert RelationType.CAUSED_BY.value == "caused_by"
        assert RelationType.SIMILAR.value == "similar"
        assert RelationType.PROPERTY_OF.value == "property_of"
        assert RelationType.INSTANCE_OF.value == "instance_of"
        assert RelationType.CONTEXT_FOR.value == "context_for"


class TestRelation:
    """Tests for Relation dataclass."""

    def test_relation_creation(self):
        """Test creating a relation object."""
        relation = Relation(
            relation_id="rel-1",
            from_id="mem-1",
            to_id="mem-2",
            type=RelationType.RELATED
        )
        assert relation.relation_id == "rel-1"
        assert relation.from_id == "mem-1"
        assert relation.to_id == "mem-2"
        assert relation.type == RelationType.RELATED


class TestProcessorResult:
    """Tests for ProcessorResult dataclass."""

    def test_processor_result_creation(self):
        """Test creating a processor result."""
        result = ProcessorResult(
            action="add",
            memory_id="mem-1",
            content="New content",
            reason="Test reason"
        )
        assert result.action == "add"
        assert result.memory_id == "mem-1"
        assert result.content == "New content"
        assert result.reason == "Test reason"


class TestExceptions:
    """Tests for exception classes."""

    def test_mempy_error(self):
        """Test base MempyError."""
        error = MempyError("Base error")
        assert str(error) == "Base error"

    def test_embedder_error(self):
        """Test EmbedderError is MempyError."""
        error = EmbedderError("Embedder failed")
        assert isinstance(error, MempyError)
        assert str(error) == "Embedder failed"

    def test_storage_error(self):
        """Test StorageError is MempyError."""
        error = StorageError("Storage failed")
        assert isinstance(error, MempyError)
        assert str(error) == "Storage failed"

    def test_processor_error(self):
        """Test ProcessorError is MempyError."""
        error = ProcessorError("Processor failed")
        assert isinstance(error, MempyError)
        assert str(error) == "Processor failed"

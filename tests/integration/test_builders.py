"""Tests for relation builders."""

import pytest
from mempy.strategies.builders import RelationBuilder, RandomRelationBuilder
from mempy.core.memory import Memory, RelationType


class MockRelationBuilder(RelationBuilder):
    """Mock relation builder for testing."""

    def __init__(self, relations_to_return=None):
        self.relations_to_return = relations_to_return or []
        self.build_called = False
        self.last_new_memory = None
        self.last_existing_memories = None

    async def build(self, new_memory, existing_memories):
        self.build_called = True
        self.last_new_memory = new_memory
        self.last_existing_memories = existing_memories
        return self.relations_to_return


@pytest.fixture
def sample_memory():
    """Create a sample memory for testing."""
    return Memory(
        memory_id="mem_123",
        content="Test memory content",
        embedding=[0.1] * 768,
        user_id="user_1"
    )


@pytest.fixture
def sample_memories():
    """Create sample memories for testing."""
    return [
        Memory(
            memory_id=f"mem_{i}",
            content=f"Memory {i} content",
            embedding=[0.1 * i] * 768,
            user_id="user_1"
        )
        for i in range(5)
    ]


class TestRelationBuilderInterface:
    """Test RelationBuilder abstract interface."""

    def test_cannot_instantiate_abstract_builder(self):
        """Test that RelationBuilder cannot be instantiated directly."""
        with pytest.raises(TypeError):
            RelationBuilder()

    def test_mock_builder_implements_interface(self, sample_memory, sample_memories):
        """Test that mock builder correctly implements the interface."""
        builder = MockRelationBuilder()

        # Should be callable
        import asyncio
        relations = asyncio.run(builder.build(sample_memory, sample_memories))

        assert builder.build_called
        assert builder.last_new_memory == sample_memory
        assert builder.last_existing_memories == sample_memories
        assert relations == builder.relations_to_return


class TestRandomRelationBuilder:
    """Test RandomRelationBuilder implementation."""

    def test_initialization_defaults(self):
        """Test RandomRelationBuilder initialization with defaults."""
        builder = RandomRelationBuilder()

        assert builder.max_relations == 3
        assert builder.relation_types == [RelationType.RELATED]
        assert builder.build_probability == 1.0

    def test_initialization_custom(self):
        """Test RandomRelationBuilder initialization with custom values."""
        builder = RandomRelationBuilder(
            max_relations=5,
            relation_types=[RelationType.SIMILAR, RelationType.RELATED],
            build_probability=0.8
        )

        assert builder.max_relations == 5
        assert len(builder.relation_types) == 2
        assert builder.build_probability == 0.8

    def test_initialization_with_seed(self):
        """Test that seed makes results reproducible."""
        builder1 = RandomRelationBuilder(seed=42)
        builder2 = RandomRelationBuilder(seed=42)

        # Both should produce same results
        import asyncio

        sample_memory = Memory(
            memory_id="mem_new",
            content="New memory",
            embedding=[0.1] * 768
        )
        existing = [
            Memory(
                memory_id=f"mem_{i}",
                content=f"Memory {i}",
                embedding=[0.1] * 768
            )
            for i in range(5)
        ]

        relations1 = asyncio.run(builder1.build(sample_memory, existing))
        relations2 = asyncio.run(builder2.build(sample_memory, existing))

        # Same number of relations
        assert len(relations1) == len(relations2)

        # If relations were built, should connect to same memories
        if relations1 and relations2:
            ids1 = [r[1] for r in relations1]
            ids2 = [r[1] for r in relations2]
            assert set(ids1) == set(ids2)

    def test_max_relations_validation(self):
        """Test that max_relations < 1 raises ValueError."""
        with pytest.raises(ValueError, match="max_relations must be at least 1"):
            RandomRelationBuilder(max_relations=0)

        with pytest.raises(ValueError, match="max_relations must be at least 1"):
            RandomRelationBuilder(max_relations=-1)

    def test_build_probability_validation(self):
        """Test that invalid build_probability raises ValueError."""
        with pytest.raises(ValueError, match="build_probability must be between 0.0 and 1.0"):
            RandomRelationBuilder(build_probability=1.5)

        with pytest.raises(ValueError, match="build_probability must be between 0.0 and 1.0"):
            RandomRelationBuilder(build_probability=-0.1)

    def test_build_with_no_existing_memories(self, sample_memory):
        """Test that build returns empty list when no existing memories."""
        import asyncio
        builder = RandomRelationBuilder()

        relations = asyncio.run(builder.build(sample_memory, []))

        assert relations == []

    def test_build_with_zero_probability(self, sample_memory, sample_memories):
        """Test that build_probability=0.0 prevents relations."""
        import asyncio
        builder = RandomRelationBuilder(build_probability=0.0)

        relations = asyncio.run(builder.build(sample_memory, sample_memories))

        assert relations == []

    def test_build_respects_max_relations(self, sample_memory, sample_memories):
        """Test that number of relations respects max_relations."""
        import asyncio
        builder = RandomRelationBuilder(
            max_relations=2,
            build_probability=1.0,
            seed=42  # For deterministic results
        )

        relations = asyncio.run(builder.build(sample_memory, sample_memories))

        # Should have at most max_relations
        assert len(relations) <= 2

    def test_build_respects_existing_memories_count(self, sample_memory):
        """Test that number of relations respects existing memories count."""
        import asyncio
        builder = RandomRelationBuilder(
            max_relations=10,  # More than available
            build_probability=1.0,
            seed=42
        )

        # Only 3 existing memories
        existing = [
            Memory(
                memory_id=f"mem_{i}",
                content=f"Memory {i}",
                embedding=[0.1] * 768
            )
            for i in range(3)
        ]

        relations = asyncio.run(builder.build(sample_memory, existing))

        # Should have at most 3 relations
        assert len(relations) <= 3

    def test_relations_have_correct_structure(self, sample_memory, sample_memories):
        """Test that returned relations have correct structure."""
        import asyncio
        builder = RandomRelationBuilder(
            max_relations=2,
            build_probability=1.0,
            seed=42
        )

        relations = asyncio.run(builder.build(sample_memory, sample_memories))

        for relation in relations:
            # Each relation should be a tuple of (from_id, to_id, relation_type, metadata)
            assert isinstance(relation, tuple)
            assert len(relation) == 4

            from_id, to_id, relation_type, metadata = relation

            # from_id should be the new memory's ID
            assert from_id == sample_memory.memory_id

            # to_id should be from existing memories
            existing_ids = [m.memory_id for m in sample_memories]
            assert to_id in existing_ids

            # relation_type should be one of the specified types
            assert relation_type in builder.relation_types

            # metadata should be a dict
            assert isinstance(metadata, dict)
            assert metadata.get("source") == "RandomRelationBuilder"
            assert metadata.get("random") is True

    def test_relation_type_selection(self, sample_memory, sample_memories):
        """Test that relation types are selected from the provided list."""
        import asyncio
        builder = RandomRelationBuilder(
            max_relations=5,
            relation_types=[RelationType.SIMILAR, RelationType.RELATED],
            build_probability=1.0,
            seed=42
        )

        relations = asyncio.run(builder.build(sample_memory, sample_memories))

        for relation in relations:
            relation_type = relation[2]
            assert relation_type in [RelationType.SIMILAR, RelationType.RELATED]

    def test_repr(self):
        """Test string representation."""
        builder = RandomRelationBuilder(
            max_relations=5,
            relation_types=[RelationType.SIMILAR, RelationType.RELATED],
            build_probability=0.8
        )

        repr_str = repr(builder)
        assert "RandomRelationBuilder" in repr_str
        assert "max_relations=5" in repr_str
        assert "probability=0.8" in repr_str


class TestMemoryIntegration:
    """Test integration with Memory class."""

    @pytest.mark.asyncio
    async def test_memory_with_builder_creates_relations(self, mock_embedder):
        """Test that Memory with builder creates relations."""
        from mempy import Memory

        builder = MockRelationBuilder(
            relations_to_return=[
                ("mem_new", "mem_1", RelationType.RELATED, None),
                ("mem_new", "mem_2", RelationType.SIMILAR, None),
            ]
        )

        memory = Memory(
            embedder=mock_embedder,
            relation_builder=builder,
            verbose=False
        )

        # Reset to ensure clean state
        await memory.reset()

        # Add first memory
        await memory.add("First memory", user_id="user_1")

        # Add second memory - should trigger builder
        await memory.add("Second memory", user_id="user_1")

        # Builder should have been called
        assert builder.build_called

    @pytest.mark.asyncio
    async def test_memory_without_builder_no_relations(self, mock_embedder):
        """Test that Memory without builder doesn't create relations."""
        from mempy import Memory

        memory = Memory(
            embedder=mock_embedder,
            relation_builder=None,  # No builder
            verbose=False
        )

        await memory.reset()

        # Add memories
        await memory.add("First memory", user_id="user_1")
        await memory.add("Second memory", user_id="user_1")

        # Should work without errors (backward compatibility)
        # No automatic relations created
        all_memories = await memory.get_all(user_id="user_1")
        assert len(all_memories) == 2

    @pytest.mark.asyncio
    async def test_backward_compatibility(self, mock_embedder):
        """Test that existing code without relation_builder parameter works."""
        from mempy import Memory

        # Old-style initialization without relation_builder parameter
        memory = Memory(embedder=mock_embedder)

        await memory.reset()

        # Should work normally
        memory_id = await memory.add("Test memory", user_id="user_1")
        assert memory_id is not None

        retrieved = await memory.get(memory_id)
        assert retrieved is not None
        assert retrieved.content == "Test memory"


class TestRandomBuilderIntegration:
    """Test RandomRelationBuilder integration with Memory."""

    @pytest.mark.asyncio
    async def test_random_builder_integration(self, mock_embedder):
        """Test RandomRelationBuilder with actual Memory instance."""
        from mempy import Memory

        builder = RandomRelationBuilder(
            max_relations=2,
            build_probability=1.0,
            seed=42
        )

        memory = Memory(
            embedder=mock_embedder,
            relation_builder=builder,
            verbose=False
        )

        await memory.reset()

        # Add multiple memories
        for i in range(5):
            await memory.add(f"Memory {i}", user_id="user_1")

        # Check that memories were added
        all_memories = await memory.get_all(user_id="user_1")
        assert len(all_memories) == 5

        # Last few memories should have relations
        # (Note: first memories won't have relations as there were no existing memories)
        has_relations = False
        for mem in all_memories:
            relations = await memory.get_relations(mem.memory_id)
            if relations:
                has_relations = True
                break

        # At least some relations should have been created
        assert has_relations

    @pytest.mark.asyncio
    async def test_random_builder_probability(self, mock_embedder):
        """Test that build_probability works in integration."""
        from mempy import Memory

        # With probability 0.0, no relations should be created
        builder = RandomRelationBuilder(
            max_relations=2,
            build_probability=0.0
        )

        memory = Memory(
            embedder=mock_embedder,
            relation_builder=builder,
            verbose=False
        )

        await memory.reset()

        # Add multiple memories
        for i in range(5):
            await memory.add(f"Memory {i}", user_id="user_1")

        # No relations should exist
        all_memories = await memory.get_all(user_id="user_1")
        for mem in all_memories:
            relations = await memory.get_relations(mem.memory_id)
            assert len(relations) == 0

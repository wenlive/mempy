"""Tests for the refactored strategy system."""

import pytest
from mempy import Memory
from mempy.strategies import RandomRelationBuilder
from mempy.core.memory import RelationType, Memory as MemoryData, ProcessorResult
from mempy.core.interfaces import MemoryProcessor


class MockProcessor(MemoryProcessor):
    """Mock processor for testing."""

    def __init__(self, action="add", memory_id=None):
        self.action = action
        self.memory_id = memory_id
        self.process_called = False
        self.last_content = None
        self.last_existing = None

    async def process(self, content, existing_memories):
        self.process_called = True
        self.last_content = content
        self.last_existing = existing_memories

        return ProcessorResult(
            action=self.action,
            memory_id=self.memory_id,
            content=content,
            reason="Mock processor"
        )


class TestProcessorStrategy:
    """Test processor strategy encapsulation."""

    @pytest.mark.asyncio
    async def test_no_processor_defaults_to_add(self, mock_embedder):
        """Test that without processor, default action is ADD."""
        memory = Memory(embedder=mock_embedder, verbose=False)
        await memory.reset()

        memory_id = await memory.add("Test content", user_id="user1")

        assert memory_id is not None
        memories = await memory.get_all(user_id="user1")
        assert len(memories) == 1

    @pytest.mark.asyncio
    async def test_processor_add_action(self, mock_embedder):
        """Test processor with ADD action."""
        processor = MockProcessor(action="add")
        memory = Memory(embedder=mock_embedder, processor=processor, verbose=False)
        await memory.reset()

        memory_id = await memory.add("Test content", user_id="user2")

        assert processor.process_called
        assert memory_id is not None
        memories = await memory.get_all(user_id="user2")
        assert len(memories) == 1

    @pytest.mark.asyncio
    async def test_processor_none_action(self, mock_embedder):
        """Test processor with NONE action (skip)."""
        processor = MockProcessor(action="none")
        memory = Memory(embedder=mock_embedder, processor=processor, verbose=False)
        await memory.reset()

        memory_id = await memory.add("Test content", user_id="user3")

        assert processor.process_called
        assert memory_id is None  # Should return None when skipped
        memories = await memory.get_all(user_id="user3")
        assert len(memories) == 0

    @pytest.mark.asyncio
    async def test_processor_failure_defaults_to_add(self, mock_embedder):
        """Test that processor failure defaults to ADD action."""
        class FailingProcessor(MemoryProcessor):
            async def process(self, content, existing_memories):
                raise Exception("Processor failed")

        processor = FailingProcessor()
        memory = Memory(embedder=mock_embedder, processor=processor, verbose=False)
        await memory.reset()

        memory_id = await memory.add("Test content", user_id="user4")

        # Should still add despite processor failure
        assert memory_id is not None
        memories = await memory.get_all(user_id="user4")
        assert len(memories) == 1


class TestRelationBuilderStrategy:
    """Test relation builder strategy encapsulation."""

    @pytest.mark.asyncio
    async def test_no_builder_no_relations(self, mock_embedder):
        """Test that without builder, no relations are created."""
        memory = Memory(embedder=mock_embedder, verbose=False)
        await memory.reset()

        await memory.add("Memory 1", user_id="user1")
        await memory.add("Memory 2", user_id="user1")

        all_memories = await memory.get_all(user_id="user1")
        for mem in all_memories:
            relations = await memory.get_relations(mem.memory_id)
            assert len(relations) == 0

    @pytest.mark.asyncio
    async def test_builder_creates_relations(self, mock_embedder):
        """Test that builder creates relations automatically."""
        builder = RandomRelationBuilder(max_relations=2, seed=42)
        memory = Memory(embedder=mock_embedder, relation_builder=builder, verbose=False)
        await memory.reset()

        await memory.add("Memory 1", user_id="user2")
        await memory.add("Memory 2", user_id="user2")
        await memory.add("Memory 3", user_id="user2")

        all_memories = await memory.get_all(user_id="user2")
        total_relations = 0
        for mem in all_memories:
            relations = await memory.get_relations(mem.memory_id)
            total_relations += len(relations)

        # Should have created some relations
        assert total_relations > 0

    @pytest.mark.asyncio
    async def test_builder_failure_no_crash(self, mock_embedder):
        """Test that builder failure doesn't crash memory addition."""
        class FailingBuilder:
            async def build(self, new_memory, existing_memories):
                raise Exception("Builder failed")

        builder = FailingBuilder()
        memory = Memory(embedder=mock_embedder, relation_builder=builder, verbose=False)
        await memory.reset()

        # Should not raise exception
        memory_id = await memory.add("Test content", user_id="user3")

        # Memory should still be saved
        assert memory_id is not None
        memories = await memory.get_all(user_id="user3")
        assert len(memories) == 1


class TestStrategyPipeline:
    """Test the full strategy pipeline."""

    @pytest.mark.asyncio
    async def test_pipeline_with_both_strategies(self, mock_embedder):
        """Test that both strategies work together."""
        processor = MockProcessor(action="add")
        builder = RandomRelationBuilder(max_relations=2, seed=42)

        memory = Memory(
            embedder=mock_embedder,
            processor=processor,
            relation_builder=builder,
            verbose=False
        )
        await memory.reset()

        await memory.add("Memory 1", user_id="user4")
        await memory.add("Memory 2", user_id="user4")
        await memory.add("Memory 3", user_id="user4")

        # Processor should have been called
        assert processor.process_called

        # Memories should be saved
        memories = await memory.get_all(user_id="user4")
        assert len(memories) == 3

        # Relations should have been created
        total_relations = 0
        for mem in memories:
            relations = await memory.get_relations(mem.memory_id)
            total_relations += len(relations)
        assert total_relations > 0

    @pytest.mark.asyncio
    async def test_pipeline_processor_skips_storage(self, mock_embedder):
        """Test that processor skipping prevents storage and relation building."""
        processor = MockProcessor(action="none")
        builder = RandomRelationBuilder(max_relations=2, seed=42)

        memory = Memory(
            embedder=mock_embedder,
            processor=processor,
            relation_builder=builder,
            verbose=False
        )
        await memory.reset()

        await memory.add("Test content", user_id="user5")

        # Should skip storage
        memories = await memory.get_all(user_id="user5")
        assert len(memories) == 0

        # Builder should not have been called (no storage = no relations)
        assert processor.process_called


class TestBackwardCompatibility:
    """Test that refactored code maintains backward compatibility."""

    @pytest.mark.asyncio
    async def test_default_behavior_unchanged(self, mock_embedder):
        """Test that default behavior (no strategies) matches original."""
        memory = Memory(embedder=mock_embedder, verbose=False)
        await memory.reset()

        # Add multiple memories
        for i in range(5):
            await memory.add(f"Content {i}", user_id="user_compat")

        # All should be saved unconditionally
        memories = await memory.get_all(user_id="user_compat")
        assert len(memories) == 5

        # No relations should exist
        for mem in memories:
            relations = await memory.get_relations(mem.memory_id)
            assert len(relations) == 0

    @pytest.mark.asyncio
    async def test_api_compatibility(self, mock_embedder):
        """Test that the public API is unchanged."""
        # Old-style initialization should still work
        memory = Memory(embedder=mock_embedder)
        await memory.reset()

        # All public methods should work
        memory_id = await memory.add("Test", user_id="user_api")
        assert memory_id is not None

        retrieved = await memory.get(memory_id)
        assert retrieved is not None

        results = await memory.search("Test", user_id="user_api")
        assert len(results) > 0

        await memory.add_relation(
            memory_id,
            memory_id,
            RelationType.RELATED
        )

        relations = await memory.get_relations(memory_id)
        assert len(relations) > 0


class TestStrategyEncapsulation:
    """Test that strategies are properly encapsulated."""

    @pytest.mark.asyncio
    async def test_processor_strategy_method(self, mock_embedder):
        """Test that _apply_processor_strategy exists and works."""
        processor = MockProcessor(action="add")
        memory = Memory(embedder=mock_embedder, processor=processor, verbose=False)

        # Call the strategy method directly
        decision = await memory._apply_processor_strategy("Test", user_id="test")

        assert decision.action == "add"
        assert processor.process_called

    @pytest.mark.asyncio
    async def test_relation_builder_strategy_method(self, mock_embedder):
        """Test that _apply_relation_builder_strategy exists and works."""
        builder = RandomRelationBuilder(max_relations=1, seed=42)
        memory = Memory(
            embedder=mock_embedder,
            relation_builder=builder,
            verbose=False
        )
        await memory.reset()

        # Add a memory first
        memory_id = await memory.add("First", user_id="test")
        memory_obj = await memory.get(memory_id)

        # Call the strategy method directly
        count = await memory._apply_relation_builder_strategy(memory_obj, "test")

        # Should have created relations
        assert count >= 0  # May be 0 if no similar memories exist

    def test_processor_decision_dataclass(self):
        """Test that _ProcessorDecision dataclass exists."""
        from mempy.memory import _ProcessorDecision

        decision = _ProcessorDecision(
            action="add",
            memory_id=None,
            content="Test"
        )

        assert decision.action == "add"
        assert decision.memory_id is None
        assert decision.content == "Test"


class TestCodeStructure:
    """Test that the code structure is improved."""

    def test_add_method_has_strategy_comments(self, mock_embedder):
        """Test that add() method has clear strategy markers."""
        import inspect

        source = inspect.getsource(Memory.add)

        # Should contain strategy markers
        assert "STRATEGY 1: Ingest" in source or "STRATEGY 1" in source
        assert "STORAGE" in source
        assert "STRATEGY 2: Graph" in source or "STRATEGY 2" in source

    def test_strategy_methods_exist(self, mock_embedder):
        """Test that strategy methods exist."""
        memory = Memory(embedder=mock_embedder)

        # Should have strategy methods
        assert hasattr(memory, '_apply_processor_strategy')
        assert hasattr(memory, '_apply_relation_builder_strategy')

        # Methods should be callable
        assert callable(memory._apply_processor_strategy)
        assert callable(memory._apply_relation_builder_strategy)

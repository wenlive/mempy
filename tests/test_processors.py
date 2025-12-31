"""Tests for processor implementations and interfaces."""

from typing import List
from unittest.mock import AsyncMock

import pytest

from mempy.core.interfaces import MemoryProcessor
from mempy.core.memory import Memory, ProcessorResult
from mempy.core.exceptions import ProcessorError
from mempy.processors.base import MemoryProcessor as BaseMemoryProcessor


class TestProcessorResult:
    """Tests for ProcessorResult dataclass."""

    def test_create_add_result(self):
        """Test creating an ADD result."""
        result = ProcessorResult(action="add")

        assert result.action == "add"
        assert result.memory_id is None
        assert result.content is None
        assert result.reason is None

    def test_create_update_result(self):
        """Test creating an UPDATE result."""
        result = ProcessorResult(
            action="update",
            memory_id="mem-123",
            content="Updated content",
            reason="Content is outdated"
        )

        assert result.action == "update"
        assert result.memory_id == "mem-123"
        assert result.content == "Updated content"
        assert result.reason == "Content is outdated"

    def test_create_delete_result(self):
        """Test creating a DELETE result."""
        result = ProcessorResult(
            action="delete",
            memory_id="mem-123",
            reason="Duplicate information"
        )

        assert result.action == "delete"
        assert result.memory_id == "mem-123"
        assert result.reason == "Duplicate information"

    def test_create_none_result(self):
        """Test creating a NONE (skip) result."""
        result = ProcessorResult(
            action="none",
            reason="Not important enough to store"
        )

        assert result.action == "none"
        assert result.reason == "Not important enough to store"

    def test_invalid_action_still_works(self):
        """Test that invalid action is accepted (flexibility for future)."""
        result = ProcessorResult(action="custom_action")

        assert result.action == "custom_action"


class TestBaseMemoryProcessor:
    """Tests for the abstract base processor class."""

    def test_cannot_instantiate_base_processor(self):
        """Test that base processor cannot be instantiated directly."""
        with pytest.raises(TypeError):
            MemoryProcessor()

    def test_subclass_must_implement_process(self):
        """Test that subclass must implement process method."""

        class IncompleteProcessor(MemoryProcessor):
            """Processor that doesn't implement process."""
            pass

        with pytest.raises(TypeError):
            IncompleteProcessor()


class TestCustomProcessorImplementations:
    """Tests for custom processor implementations."""

    @pytest.fixture
    def sample_memories(self):
        """Provide sample memories for testing."""
        return [
            Memory(
                memory_id="mem-1",
                content="Python is a programming language",
                embedding=[0.1] * 768,
                user_id="user-1"
            ),
            Memory(
                memory_id="mem-2",
                content="Python is used for web development",
                embedding=[0.1] * 768,
                user_id="user-1"
            ),
        ]

    def test_similarity_based_processor_adds_when_different(self, sample_memories):
        """Test processor adds new content when not similar to existing."""

        class SimilarityProcessor(MemoryProcessor):
            """Processor that checks content similarity."""

            def __init__(self, threshold: float = 0.8):
                self.threshold = threshold

            async def process(
                self,
                content: str,
                existing_memories: List[Memory]
            ) -> ProcessorResult:
                # Simple similarity check: exact match or substring
                for existing in existing_memories:
                    if content.lower() == existing.content.lower():
                        return ProcessorResult(
                            action="none",
                            reason="Exact duplicate"
                        )
                    if content.lower() in existing.content.lower() or existing.content.lower() in content.lower():
                        return ProcessorResult(
                            action="update",
                            memory_id=existing.memory_id,
                            content=content,
                            reason="Very similar content"
                        )
                return ProcessorResult(action="add")

        processor = SimilarityProcessor()

        # New content is different
        import asyncio

        async def run_test():
            result = await processor.process("Java is also a programming language", sample_memories)
            assert result.action == "add"

        asyncio.run(run_test())

    def test_similarity_based_processor_updates_when_similar(self, sample_memories):
        """Test processor updates when content is similar."""

        class SimilarityProcessor(MemoryProcessor):
            """Processor that checks content similarity."""

            async def process(
                self,
                content: str,
                existing_memories: List[Memory]
            ) -> ProcessorResult:
                for existing in existing_memories:
                    if existing.content.lower() in content.lower():
                        return ProcessorResult(
                            action="update",
                            memory_id=existing.memory_id,
                            content=content,
                            reason="Content expansion"
                        )
                return ProcessorResult(action="add")

        processor = SimilarityProcessor()

        import asyncio

        async def run_test():
            # Content contains substring of existing memory
            result = await processor.process(
                "Python is a versatile programming language used for web development",
                sample_memories
            )
            assert result.action == "update"
            assert result.memory_id == "mem-1"

        asyncio.run(run_test())

    def test_similarity_based_processor_skips_duplicate(self, sample_memories):
        """Test processor skips exact duplicates."""

        class DuplicateProcessor(MemoryProcessor):
            """Processor that rejects exact duplicates."""

            async def process(
                self,
                content: str,
                existing_memories: List[Memory]
            ) -> ProcessorResult:
                for existing in existing_memories:
                    if content.lower() == existing.content.lower():
                        return ProcessorResult(
                            action="none",
                            reason="Exact duplicate"
                        )
                return ProcessorResult(action="add")

        processor = DuplicateProcessor()

        import asyncio

        async def run_test():
            result = await processor.process("Python is a programming language", sample_memories)
            assert result.action == "none"

        asyncio.run(run_test())

    def test_count_based_processor_deletes_old_memories(self):
        """Test processor that maintains a maximum number of memories."""

        class CountLimitProcessor(MemoryProcessor):
            """Processor that limits total memories per user."""

            def __init__(self, max_memories: int = 10):
                self.max_memories = max_memories

            async def process(
                self,
                content: str,
                existing_memories: List[Memory]
            ) -> ProcessorResult:
                if len(existing_memories) >= self.max_memories:
                    # Delete the oldest memory
                    oldest = min(existing_memories, key=lambda m: m.created_at)
                    return ProcessorResult(
                        action="delete",
                        memory_id=oldest.memory_id,
                        reason="Memory limit reached, removing oldest"
                    )
                return ProcessorResult(action="add")

        processor = CountLimitProcessor(max_memories=2)

        # Create 2 existing memories
        memories = [
            Memory(
                memory_id=f"mem-{i}",
                content=f"Content {i}",
                embedding=[0.1] * 768,
                user_id="user-1"
            )
            for i in range(2)
        ]

        import asyncio

        async def run_test():
            # With 2 memories (at limit), should add
            result = await processor.process("New content", memories)
            assert result.action == "delete"
            assert result.memory_id is not None

        asyncio.run(run_test())


class TestProcessorErrorHandling:
    """Tests for processor error handling."""

    def test_processor_can_raise_error(self):
        """Test that processor can raise ProcessorError."""

        class ErrorProcessor(MemoryProcessor):
            """Processor that raises an error."""

            async def process(
                self,
                content: str,
                existing_memories: List[Memory]
            ) -> ProcessorResult:
                raise ProcessorError("Processing failed due to invalid input")

        processor = ErrorProcessor()

        import asyncio

        async def run_test():
            with pytest.raises(ProcessorError, match="Processing failed"):
                await processor.process("invalid", [])

        asyncio.run(run_test())

    def test_processor_handles_empty_existing_memories(self):
        """Test processor handles empty existing_memories list."""

        class SimpleProcessor(MemoryProcessor):
            """Simple processor that always adds."""

            async def process(
                self,
                content: str,
                existing_memories: List[Memory]
            ) -> ProcessorResult:
                if not existing_memories:
                    return ProcessorResult(
                        action="add",
                        reason="No existing memories"
                    )
                return ProcessorResult(action="add")

        processor = SimpleProcessor()

        import asyncio

        async def run_test():
            result = await processor.process("New content", [])
            assert result.action == "add"
            assert result.reason == "No existing memories"

        asyncio.run(run_test())


class TestProcessorWithMockAsyncBehavior:
    """Tests for processor async behavior with mocks."""

    @pytest.mark.asyncio
    async def test_processor_async_execution(self):
        """Test that processor's async method executes correctly."""

        class AsyncProcessor(MemoryProcessor):
            """Processor with simulated async work."""

            def __init__(self):
                self.call_count = 0

            async def process(
                self,
                content: str,
                existing_memories: List[Memory]
            ) -> ProcessorResult:
                self.call_count += 1
                # Simulate some async work
                import asyncio
                await asyncio.sleep(0)
                return ProcessorResult(action="add")

        processor = AsyncProcessor()

        result = await processor.process("test", [])

        assert result.action == "add"
        assert processor.call_count == 1

    @pytest.mark.asyncio
    async def test_processor_with_multiple_calls(self):
        """Test processor handles multiple sequential calls."""

        class StatefulProcessor(MemoryProcessor):
            """Processor that keeps track of seen content."""

            def __init__(self):
                self.seen_content = []

            async def process(
                self,
                content: str,
                existing_memories: List[Memory]
            ) -> ProcessorResult:
                self.seen_content.append(content)
                if content in self.seen_content[:-1]:  # Seen before (excluding current)
                    return ProcessorResult(
                        action="none",
                        reason="Already processed this content"
                    )
                return ProcessorResult(action="add")

        processor = StatefulProcessor()

        result1 = await processor.process("content1", [])
        result2 = await processor.process("content2", [])
        result3 = await processor.process("content1", [])  # Duplicate

        assert result1.action == "add"
        assert result2.action == "add"
        assert result3.action == "none"


class TestProcessorResultValidation:
    """Tests for ProcessorResult validation and edge cases."""

    def test_result_with_all_fields(self):
        """Test result with all fields populated."""
        result = ProcessorResult(
            action="update",
            memory_id="mem-123",
            content="New content",
            reason="Updated due to new information"
        )

        assert result.action == "update"
        assert result.memory_id == "mem-123"
        assert result.content == "New content"
        assert result.reason == "Updated due to new information"

    def test_update_result_without_memory_id(self):
        """Test update action without memory_id is valid (flexible)."""
        result = ProcessorResult(
            action="update",
            content="Some content"
        )

        # Should not raise an error
        assert result.action == "update"
        assert result.memory_id is None

    def test_delete_result_without_memory_id(self):
        """Test delete action without memory_id is valid."""
        result = ProcessorResult(
            action="delete",
            reason="Cleanup"
        )

        assert result.action == "delete"
        assert result.memory_id is None

    def test_result_equality(self):
        """Test that two results with same values are equal."""
        result1 = ProcessorResult(action="add", reason="test")
        result2 = ProcessorResult(action="add", reason="test")

        # Dataclass should provide equality
        assert result1 == result2

    def test_result_inequality(self):
        """Test that results with different values are not equal."""
        result1 = ProcessorResult(action="add")
        result2 = ProcessorResult(action="update")

        assert result1 != result2

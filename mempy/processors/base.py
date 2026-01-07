"""Base processor class for memory operations."""

from abc import ABC, abstractmethod
from typing import List

from mempy.core.memory import Memory, ProcessorResult
from mempy.core.exceptions import ProcessorError


class MemoryProcessor(ABC):
    """
    Abstract base class for memory processors.

    Processors analyze new content against existing memories and decide
    what operation to perform: add new, update existing, delete, or ignore.

    Users can implement this interface to create custom processing logic,
    or use LLMProcessor for LLM-based intelligent processing.
    """

    @abstractmethod
    async def process(
        self,
        content: str,
        existing_memories: List[Memory]
    ) -> ProcessorResult:
        """
        Decide what operation to perform based on content and existing memories.

        Args:
            content: The new content to process
            existing_memories: List of potentially related existing memories

        Returns:
            ProcessorResult with action, memory_id, content, and reason

        Raises:
            ProcessorError: If processing fails
        """
        pass

"""LLM-based memory processor implementation."""

import json
import logging
from typing import Any, Awaitable, Callable, List

from mempy.processors.base import MemoryProcessor
from mempy.core.memory import Memory, ProcessorResult
from mempy.core.exceptions import ProcessorError


# Default prompt template for the LLM processor
DEFAULT_PROCESSOR_PROMPT = """You are a memory management assistant. Based on the user input and existing memories, determine what operation should be performed.

Existing memories:
{existing_memories}

User input: {user_input}

Return a JSON response with this format:
{{
  "action": "add" | "update" | "delete" | "none",
  "memory_id": "...",      // Required for update/delete operations
  "content": "...",        // Required for update operations - the new content
  "reason": "..."          // Explanation for your decision
}}

Rules:
- "add": When the input contains new information not in existing memories
- "update": When the input modifies or clarifies an existing memory
- "delete": When the input contradicts or removes an existing memory
- "none": When the input doesn't contain meaningful information to remember
"""


class LLMProcessor(MemoryProcessor):
    """
    LLM-based memory processor.

    This processor uses an LLM to intelligently decide whether to add,
    update, delete, or ignore new content based on existing memories.

    Users must provide an LLM call function that takes a prompt and
    returns the LLM's text response.
    """

    def __init__(
        self,
        llm_call: Callable[[str], Awaitable[str]],
        prompt_template: str = DEFAULT_PROCESSOR_PROMPT
    ):
        """
        Initialize the LLM processor.

        Args:
            llm_call: Async function that takes a prompt and returns LLM response
            prompt_template: Optional custom prompt template
        """
        self.llm_call = llm_call
        self.prompt_template = prompt_template
        self.logger = logging.getLogger("mempy.processor")

    def _format_memories(self, memories: List[Memory], max_memories: int = 5) -> str:
        """Format memories for the prompt."""
        if not memories:
            return "No existing memories."

        # Limit to prevent token overflow
        memories = memories[:max_memories]

        lines = []
        for m in memories:
            lines.append(f"- ID: {m.memory_id} | Content: {m.content}")
        return "\n".join(lines)

    async def process(
        self,
        content: str,
        existing_memories: List[Memory]
    ) -> ProcessorResult:
        """
        Use LLM to decide what operation to perform.

        Args:
            content: The new content to process
            existing_memories: List of potentially related existing memories

        Returns:
            ProcessorResult with action, memory_id, content, and reason

        Raises:
            ProcessorError: If LLM call fails or returns invalid response
        """
        # Format the prompt
        prompt = self.prompt_template.format(
            existing_memories=self._format_memories(existing_memories),
            user_input=content
        )

        try:
            # Call the LLM
            response = await self.llm_call(prompt)

            # Parse the response
            result = self._parse_response(response)

            self.logger.info(
                f"[PROCESSOR] Decided: {result.action} | "
                f"Reason: {result.reason}"
            )

            return result

        except Exception as e:
            self.logger.error(f"[PROCESSOR] Failed: {e}")
            # On error, default to "add" to be safe
            return ProcessorResult(
                action="add",
                reason=f"Default to add due to processing error: {e}"
            )

    def _parse_response(self, response: str) -> ProcessorResult:
        """
        Parse LLM response into ProcessorResult.

        Args:
            response: Raw LLM text response

        Returns:
            ProcessorResult

        Raises:
            ProcessorError: If response cannot be parsed
        """
        # Try to extract JSON from response
        try:
            # Find JSON in response (handle markdown code blocks)
            if "```json" in response:
                start = response.index("```json") + 7
                end = response.index("```", start)
                json_str = response[start:end].strip()
            elif "```" in response:
                start = response.index("```") + 3
                end = response.index("```", start)
                json_str = response[start:end].strip()
            else:
                json_str = response.strip()

            data = json.loads(json_str)

            # Validate required fields
            action = data.get("action", "add")
            if action not in ("add", "update", "delete", "none"):
                action = "add"

            return ProcessorResult(
                action=action,
                memory_id=data.get("memory_id"),
                content=data.get("content"),
                reason=data.get("reason", "")
            )

        except (json.JSONDecodeError, ValueError) as e:
            raise ProcessorError(f"Failed to parse LLM response: {e}") from e

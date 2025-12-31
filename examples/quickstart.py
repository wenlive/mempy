"""
Quickstart example for mempy.

This example demonstrates how to use mempy for memory management.
"""

import asyncio
import aiohttp
from typing import List

import mempy


# ============================================================================
# Step 1: Implement the Embedder interface (required)
# ============================================================================

class MyLLMEmbedder(mempy.Embedder):
    """
    Example embedder that calls a local LLM service.

    Users must implement this interface with their own LLM endpoint.
    """

    def __init__(self, endpoint: str, dimension: int = 768):
        self.endpoint = endpoint
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        """Return the embedding dimension (REQUIRED)."""
        return self._dimension

    async def embed(self, text: str) -> List[float]:
        """Generate embedding for the given text."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.endpoint}/embed",
                json={"text": text}
            ) as resp:
                data = await resp.json()
                return data["embedding"]


# ============================================================================
# Step 2: Optionally implement a MemoryProcessor for intelligent operations
# ============================================================================

class MyLLMProcessor(mempy.MemoryProcessor):
    """
    Example processor using an LLM to decide memory operations.

    This is optional - without it, mempy will always add new memories.
    """

    def __init__(self, llm_endpoint: str):
        self.endpoint = llm_endpoint

    async def call_llm(self, prompt: str) -> str:
        """Call the LLM with the given prompt."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.endpoint}/generate",
                json={"prompt": prompt}
            ) as resp:
                data = await resp.json()
                return data["text"]

    async def process(self, content: str, existing_memories: List[mempy.Memory]) -> mempy.ProcessorResult:
        """Decide what operation to perform."""
        from mempy.processors.llm_processor import DEFAULT_PROCESSOR_PROMPT

        # Format existing memories for the prompt
        memory_list = "\n".join([
            f"- ID: {m.memory_id} | Content: {m.content}"
            for m in existing_memories[:5]
        ]) or "No existing memories."

        prompt = DEFAULT_PROCESSOR_PROMPT.format(
            existing_memories=memory_list,
            user_input=content
        )

        response = await self.call_llm(prompt)

        # Parse response (simplified - production code should be more robust)
        import json
        try:
            data = json.loads(response)
            return mempy.ProcessorResult(
                action=data.get("action", "add"),
                memory_id=data.get("memory_id"),
                content=data.get("content"),
                reason=data.get("reason", "")
            )
        except json.JSONDecodeError:
            # Default to add if parsing fails
            return mempy.ProcessorResult(action="add", reason="Failed to parse LLM response")


# ============================================================================
# Step 3: Create a Memory instance
# ============================================================================

async def main():
    """Main example demonstrating mempy usage."""

    # Create embedder (user must provide their LLM endpoint)
    embedder = MyLLMEmbedder(
        endpoint="http://localhost:11434",
        dimension=768  # Adjust based on your model
    )

    # Optionally create processor
    # processor = MyLLMProcessor(llm_endpoint="http://localhost:11434")
    processor = None  # Skip for this example

    # Create Memory instance
    memory = mempy.Memory(
        embedder=embedder,
        processor=processor,
        verbose=True  # Enable logging
    )

    # ========================================================================
    # Step 4: Add memories
    # ========================================================================

    print("\n=== Adding memories ===")

    mem1 = await memory.add("我喜欢蓝色", user_id="alice")
    mem2 = await memory.add("Alice works at Google", user_id="alice")
    mem3 = await memory.add("Alice is a software engineer", user_id="alice")

    # ========================================================================
    # Step 5: Search memories
    # ========================================================================

    print("\n=== Searching memories ===")

    results = await memory.search("Alice's job", user_id="alice", limit=3)
    for r in results:
        print(f"  {r.memory_id}: {r.content}")

    # ========================================================================
    # Step 6: Get a specific memory
    # ========================================================================

    print("\n=== Getting specific memory ===")

    if mem1:
        m = await memory.get(mem1)
        if m:
            print(f"  {m.memory_id}: {m.content}")

    # ========================================================================
    # Step 7: Get all memories for a user
    # ========================================================================

    print("\n=== Getting all memories for user ===")

    all_memories = await memory.get_all(user_id="alice")
    print(f"  Total memories: {len(all_memories)}")
    for m in all_memories:
        print(f"  - {m.memory_id}: {m.content}")

    # ========================================================================
    # Step 8: Add relations between memories
    # ========================================================================

    print("\n=== Adding relations ===")

    if mem1 and mem2:
        await memory.add_relation(
            mem1,
            mem2,
            mempy.RelationType.PROPERTY_OF
        )

    # ========================================================================
    # Step 9: Get relations
    # ========================================================================

    print("\n=== Getting relations ===")

    if mem1:
        relations = await memory.get_relations(mem1, max_depth=2)
        for rel in relations:
            print(f"  {rel.from_id} --[{rel.type.value}]--> {rel.to_id}")

    # ========================================================================
    # Step 10: Update a memory
    # ========================================================================

    print("\n=== Updating memory ===")

    if mem1:
        await memory.update(mem1, "I prefer dark blue")

    # ========================================================================
    # Step 11: Delete a memory
    # ========================================================================

    print("\n=== Deleting memory ===")

    if mem3:
        await memory.delete(mem3)

    # ========================================================================
    # Step 12: Reset (clear all data)
    # ========================================================================

    # Uncomment to reset all data
    # await memory.reset()
    # print("\n=== Reset complete ===")


if __name__ == "__main__":
    asyncio.run(main())

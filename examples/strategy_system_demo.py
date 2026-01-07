#!/usr/bin/env python3
"""
Demonstration of mempy's pluggable strategy system.

This script shows how the Memory class uses a clear three-stage strategy pipeline:
1. Ingest Strategy (processor): Decide what to do with new content
2. Storage: Save the memory
3. Graph Strategy (relation_builder): Build relations

The strategy calls are now explicit and easy to understand, making it simple
for advanced users to extend the system with custom strategies.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mempy import Memory, Embedder
from mempy.strategies import RandomRelationBuilder, RelationType


class SimpleEmbedder(Embedder):
    """Simple mock embedder for demonstration."""

    @property
    def dimension(self) -> int:
        return 768

    async def embed(self, text: str) -> list:
        # Generate consistent embeddings based on text hash
        import hashlib
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        return [(hash_val % 1000) / 1000.0] * 768


async def demo_basic_usage():
    """Demo 1: Basic usage without strategies."""
    print("\n" + "=" * 60)
    print("Demo 1: Basic Usage (No Strategies)")
    print("=" * 60)

    memory = Memory(embedder=SimpleEmbedder(), verbose=True)
    await memory.reset()

    print("\nAdding memories without strategies...")
    await memory.add("I like blue", user_id="alice")
    await memory.add("Blue is my favorite color", user_id="alice")

    print("\n✓ All memories saved unconditionally")
    print("✓ No automatic relations created")


async def demo_with_processor():
    """Demo 2: Using processor strategy for intelligent decisions."""
    print("\n" + "=" * 60)
    print("Demo 2: With Processor Strategy (Intelligent Decisions)")
    print("=" * 60)

    from mempy.core.interfaces import MemoryProcessor
    from mempy.core.memory import ProcessorResult, Memory as MemoryData

    class SimpleProcessor(MemoryProcessor):
        """Example processor that filters short content."""

        async def process(self, content, existing_memories):
            # Filter out very short content
            if len(content.strip()) < 5:
                return ProcessorResult(
                    action="none",
                    memory_id=None,
                    content=None,
                    reason="Content too short"
                )

            # Check if similar memory exists
            for mem in existing_memories:
                if content.lower() in mem.content.lower() or mem.content.lower() in content.lower():
                    return ProcessorResult(
                        action="update",
                        memory_id=mem.memory_id,
                        content=content,
                        reason="Similar content exists"
                    )

            # Default: add new memory
            return ProcessorResult(
                action="add",
                memory_id=None,
                content=content,
                reason="New content"
            )

    memory = Memory(
        embedder=SimpleEmbedder(),
        processor=SimpleProcessor(),
        verbose=True
    )
    await memory.reset()

    print("\nAdding memories with processor strategy...")
    print("\n[PROCESSOR will make decisions on each content]")
    await memory.add("Hi", user_id="bob")  # Too short, will be skipped
    await memory.add("I like blue", user_id="bob")  # Will be added
    await memory.add("I really like blue", user_id="bob")  # Will update previous
    await memory.add("I enjoy red", user_id="bob")  # Will be added

    print("\n✓ Processor made intelligent decisions")
    print("✓ Skipped short content, updated similar memories")


async def demo_with_relation_builder():
    """Demo 3: Using relation builder strategy for automatic graph construction."""
    print("\n" + "=" * 60)
    print("Demo 3: With Relation Builder Strategy (Graph Construction)")
    print("=" * 60)

    # Create builder with custom settings
    builder = RandomRelationBuilder(
        max_relations=2,
        relation_types=[RelationType.RELATED, RelationType.SIMILAR],
        build_probability=1.0,
        seed=42
    )

    memory = Memory(
        embedder=SimpleEmbedder(),
        relation_builder=builder,
        verbose=True
    )
    await memory.reset()

    print("\nAdding memories with automatic relation building...")
    print("\n[RELATION BUILDER will automatically create graph relations]")
    await memory.add("I like blue", user_id="charlie")
    await memory.add("Blue is my favorite", user_id="charlie")
    await memory.add("The sky is blue", user_id="charlie")
    await memory.add("I enjoy painting", user_id="charlie")

    print("\nChecking automatically created relations...")
    all_memories = await memory.get_all(user_id="charlie")
    for mem in all_memories:
        relations = await memory.get_relations(mem.memory_id)
        if relations:
            print(f"  Memory: {mem.content[:30]}... -> {len(relations)} relations")

    print("\n✓ Relations were automatically built during memory.add()")
    print("✓ No manual relation management needed")


async def demo_with_both_strategies():
    """Demo 4: Using both strategies together."""
    print("\n" + "=" * 60)
    print("Demo 4: With Both Strategies (Full Pipeline)")
    print("=" * 60)

    from mempy.core.interfaces import MemoryProcessor
    from mempy.core.memory import ProcessorResult

    class SmartProcessor(MemoryProcessor):
        """Processor that adds metadata to important content."""

        async def process(self, content, existing_memories):
            # Mark important keywords
            important_keywords = ["favorite", "love", "hate", "critical", "important"]
            metadata = {}

            for keyword in important_keywords:
                if keyword in content.lower():
                    metadata["importance"] = 0.9
                    metadata["type"] = "preference"
                    break

            return ProcessorResult(
                action="add",
                memory_id=None,
                content=content,
                reason="Content processed",
                metadata=metadata
            )

    builder = RandomRelationBuilder(
        max_relations=3,
        relation_types=[RelationType.RELATED, RelationType.SIMILAR],
        seed=42
    )

    memory = Memory(
        embedder=SimpleEmbedder(),
        processor=SmartProcessor(),
        relation_builder=builder,
        verbose=True
    )
    await memory.reset()

    print("\nAdding memories with full strategy pipeline...")
    print("\n[STRATEGY 1: Processor] → [STORAGE] → [STRATEGY 2: Relation Builder]")
    await memory.add("Blue is my favorite", user_id="diana")
    await memory.add("I like blue", user_id="diana")
    await memory.add("The sky is blue", user_id="diana")

    print("\n✓ Both strategies executed in pipeline")
    print("✓ Processor enriched metadata, Builder constructed relations")


async def demo_custom_strategy():
    """Demo 5: Custom relation builder for advanced users."""
    print("\n" + "=" * 60)
    print("Demo 5: Custom Relation Builder (For Advanced Users)")
    print("=" * 60)

    from mempy.strategies import RelationBuilder
    import numpy as np

    class SemanticRelationBuilder(RelationBuilder):
        """Custom builder that creates relations based on semantic similarity."""

        def __init__(self, threshold=0.95):
            self.threshold = threshold

        async def build(self, new_memory, existing_memories):
            relations = []
            new_vec = np.array(new_memory.embedding)

            for existing in existing_memories:
                existing_vec = np.array(existing.embedding)
                # Compute cosine similarity
                similarity = np.dot(new_vec, existing_vec) / (
                    np.linalg.norm(new_vec) * np.linalg.norm(existing_vec)
                )

                if similarity >= self.threshold:
                    relations.append((
                        new_memory.memory_id,
                        existing.memory_id,
                        RelationType.SIMILAR,
                        {"similarity": float(similarity), "threshold": self.threshold}
                    ))

            return relations

    memory = Memory(
        embedder=SimpleEmbedder(),
        relation_builder=SemanticRelationBuilder(threshold=0.98),
        verbose=False
    )
    await memory.reset()

    print("\nAdding memories with custom semantic builder...")
    await memory.add("I like blue", user_id="eve")
    await memory.add("Blue is my favorite", user_id="eve")
    await memory.add("The sky is blue", user_id="eve")

    print("\nChecking similarity-based relations...")
    all_memories = await memory.get_all(user_id="eve")
    for mem in all_memories:
        relations = await memory.get_relations(mem.memory_id)
        if relations:
            print(f"  Memory: {mem.content[:30]}...")
            for rel in relations[:2]:
                sim = rel.metadata.get("similarity", 0)
                print(f"    - SIMILAR (similarity: {sim:.4f})")

    print("\n✓ Custom strategy successfully integrated")
    print("✓ Advanced users can implement domain-specific logic")


async def main():
    """Run all demonstrations."""
    print("=" * 60)
    print("mempy Strategy System Demonstration")
    print("=" * 60)
    print("\nThe strategy system provides explicit, pluggable components:")
    print("  - Processor: Decide what to do with new content")
    print("  - Relation Builder: Automatically construct graph relations")
    print("\nBoth strategies are optional and can be used independently.")

    await demo_basic_usage()
    await demo_with_processor()
    await demo_with_relation_builder()
    await demo_with_both_strategies()
    await demo_custom_strategy()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
Key Improvements:
    ✓ Strategy calls are EXPLICIT and easy to see
    ✓ Three-stage pipeline: Ingest → Storage → Graph
    ✓ Strategies are completely ENCAPSULATED in private methods
    ✓ Easy for advanced users to implement custom strategies
    ✓ Default behavior (no strategies) is simple and predictable

Advanced Usage:
    - Implement custom MemoryProcessor for intelligent decisions
    - Implement custom RelationBuilder for domain-specific relations
    - Combine both strategies for powerful automation
    - All strategies are optional and pluggable
    """)


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""Demonstration of RelationBuilder functionality.

This script shows how to use the RelationBuilder system to automatically
construct graph relations when adding memories.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mempy import Memory, Embedder, RelationType
from mempy.strategies import RandomRelationBuilder


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


async def demo_without_builder():
    """Demo Memory without builder (default behavior)."""
    print("\n" + "=" * 60)
    print("Demo 1: Memory WITHOUT RelationBuilder (default)")
    print("=" * 60)

    memory = Memory(embedder=SimpleEmbedder(), verbose=True)
    await memory.reset()

    # Add memories
    print("\nAdding memories...")
    await memory.add("I like blue", user_id="alice")
    await memory.add("Blue is my favorite color", user_id="alice")
    await memory.add("The sky is blue", user_id="alice")

    # Check relations
    print("\nChecking relations...")
    all_memories = await memory.get_all(user_id="alice")
    for mem in all_memories:
        relations = await memory.get_relations(mem.memory_id)
        print(f"  Memory: {mem.content[:30]}... -> {len(relations)} relations")

    print("\n✓ No automatic relations created (backward compatible)")


async def demo_with_random_builder():
    """Demo Memory with RandomRelationBuilder."""
    print("\n" + "=" * 60)
    print("Demo 2: Memory WITH RandomRelationBuilder")
    print("=" * 60)

    # Create builder with custom settings
    builder = RandomRelationBuilder(
        max_relations=2,
        relation_types=[RelationType.RELATED, RelationType.SIMILAR],
        build_probability=1.0,  # Always build relations
        seed=42  # For reproducible results
    )

    memory = Memory(
        embedder=SimpleEmbedder(),
        relation_builder=builder,
        verbose=True
    )
    await memory.reset()

    # Add memories
    print("\nAdding memories with automatic relation building...")
    await memory.add("I like blue", user_id="alice")
    await memory.add("Blue is my favorite color", user_id="alice")
    await memory.add("The sky is blue", user_id="alice")
    await memory.add("I enjoy painting", user_id="alice")

    # Check relations
    print("\nChecking automatically created relations...")
    all_memories = await memory.get_all(user_id="alice")
    total_relations = 0
    for mem in all_memories:
        relations = await memory.get_relations(mem.memory_id)
        total_relations += len(relations)
        print(f"  Memory: {mem.content[:30]}... -> {len(relations)} relations")
        if relations:
            for rel in relations[:2]:  # Show first 2
                print(f"    - {rel.type} (metadata: {rel.metadata})")

    print(f"\n✓ Total relations created: {total_relations}")
    print("✓ Relations were automatically built during memory.add()")


async def demo_custom_builder():
    """Demo with a custom semantic similarity builder."""
    print("\n" + "=" * 60)
    print("Demo 3: Custom SemanticRelationBuilder")
    print("=" * 60)

    from mempy.strategies import RelationBuilder
    import numpy as np

    class SemanticRelationBuilder(RelationBuilder):
        """Build relations based on semantic similarity threshold."""

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

    # Use custom builder
    builder = SemanticRelationBuilder(threshold=0.98)

    memory = Memory(
        embedder=SimpleEmbedder(),
        relation_builder=builder,
        verbose=False
    )
    await memory.reset()

    # Add similar content (will have high similarity)
    print("\nAdding memories with semantic similarity builder...")
    await memory.add("I like blue", user_id="bob")
    await memory.add("Blue is my favorite", user_id="bob")
    await memory.add("The sky is blue", user_id="bob")

    # Check relations
    print("\nChecking similarity-based relations...")
    all_memories = await memory.get_all(user_id="bob")
    for mem in all_memories:
        relations = await memory.get_relations(mem.memory_id)
        if relations:
            print(f"  Memory: {mem.content[:30]}...")
            for rel in relations:
                sim = rel.metadata.get("similarity", 0)
                print(f"    - SIMILAR (similarity: {sim:.4f})")

    print("\n✓ Relations built based on semantic similarity")


async def main():
    """Run all demonstrations."""
    print("=" * 60)
    print("RelationBuilder System Demonstration")
    print("=" * 60)

    await demo_without_builder()
    await demo_with_random_builder()
    await demo_custom_builder()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
The RelationBuilder system provides:

1. Backward Compatibility: Default behavior unchanged (builder=None)
2. Extensibility: Implement custom RelationBuilder classes
3. Automatic Building: Relations created during memory.add()
4. Strategy Pattern: Pluggable relation construction strategies

Example builders:
- RandomRelationBuilder: Random relations (testing/demo)
- SemanticRelationBuilder: Similarity-based relations
- LLMRelationBuilder: Use LLM to determine relations (future)
- TemporalRelationBuilder: Time-based relations (future)

Usage:
    from mempy import Memory
    from mempy.strategies import RandomRelationBuilder

    builder = RandomRelationBuilder(max_relations=3)
    memory = Memory(embedder=MyEmbedder(), relation_builder=builder)
    await memory.add("I like blue", user_id="alice")  # Auto-builds relations
    """)


if __name__ == "__main__":
    asyncio.run(main())

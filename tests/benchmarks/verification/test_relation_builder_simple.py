#!/usr/bin/env python3
"""Simple test to verify RelationBuilder is working.

This test:
1. Adds some memories to the system
2. Verifies relations are created when RelationBuilder is enabled
3. Shows detailed logs
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from tests.benchmarks.adapters.zhipu import ZhipuEmbedder
from mempy.strategies import RandomRelationBuilder
from mempy.memory import Memory as MemoryClient


async def test_with_relation_builder():
    """Test Memory with RandomRelationBuilder."""
    print("=" * 80)
    print("TEST: Verify RelationBuilder is Working")
    print("=" * 80)

    # Configuration
    api_key = os.environ.get("ZHIPUAI_API_KEY")
    if not api_key:
        print("\n✗ Error: ZHIPUAI_API_KEY environment variable not set")
        print("  Please set it using:")
        print("    export ZHIPUAI_API_KEY=your-api-key-here")
        print("  Or create a .env file from .env.example:")
        print("    cp tests/benchmarks/.env.example tests/benchmarks/.env")
        print("    # Edit tests/benchmarks/.env and add your API key")
        return 1

    print("\n" + "=" * 80)
    print("Test 1: WITHOUT RelationBuilder (Baseline)")
    print("=" * 80 + "\n")

    # Test 1: Without RelationBuilder
    embedder1 = ZhipuEmbedder(api_key=api_key, model="embedding-3")
    memory1 = MemoryClient(embedder=embedder1, verbose=True)

    # Add some test memories
    print("Adding 5 test memories...\n")
    memories_text = [
        "I love playing basketball on weekends",
        "Basketball is my favorite sport",
        "I enjoy watching NBA games",
        "The Lakers are a great team",
        "Michael Jordan is the GOAT"
    ]

    for i, text in enumerate(memories_text, 1):
        print(f"\n--- Adding memory {i} ---")
        await memory1.add(text, user_id="test_user")

    # Check relations by getting all memories and checking each
    print("\n" + "-" * 80)
    print("Checking relations...")
    memories1 = await memory1.get_all(user_id="test_user")
    total_relations1 = 0
    for mem in memories1:
        rels = await memory1.get_relations(mem.memory_id)
        total_relations1 += len(rels)

    print(f"Total relations found: {total_relations1}")
    if total_relations1 > 0:
        print(f"  Average relations per memory: {total_relations1 / len(memories1):.1f}")
    else:
        print("  (No relations - as expected without RelationBuilder)")

    # Reset
    await memory1.reset()
    await embedder1.close()

    print("\n" + "=" * 80)
    print("Test 2: WITH RandomRelationBuilder")
    print("=" * 80 + "\n")

    # Test 2: With RelationBuilder
    embedder2 = ZhipuEmbedder(api_key=api_key, model="embedding-3")
    relation_builder = RandomRelationBuilder(max_relations=2)
    memory2 = MemoryClient(
        embedder=embedder2,
        relation_builder=relation_builder,
        verbose=True
    )

    # Add same test memories
    print("Adding 5 test memories...\n")
    for i, text in enumerate(memories_text, 1):
        print(f"\n--- Adding memory {i} ---")
        await memory2.add(text, user_id="test_user")

    # Check relations
    print("\n" + "-" * 80)
    print("Checking relations...")
    memories2 = await memory2.get_all(user_id="test_user")
    total_relations2 = 0
    all_relations2 = []

    for mem in memories2:
        rels = await memory2.get_relations(mem.memory_id)
        total_relations2 += len(rels)
        all_relations2.extend([(mem.memory_id, rel) for rel in rels])

    print(f"Total relations found: {total_relations2}")
    if total_relations2 > 0:
        print(f"  Average relations per memory: {total_relations2 / len(memories2):.1f}")
        print("\nFirst 5 relations:")
        for i, (mem_id, rel) in enumerate(all_relations2[:5], 1):
            print(f"  {i}. Memory: {mem_id[:8]}... -> Relation: {rel}")
    else:
        print("  (No relations - RelationBuilder may not be working!)")

    # Compare
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(f"  Without RelationBuilder: {total_relations1} relations")
    print(f"  With RelationBuilder:    {total_relations2} relations")
    print(f"  Difference:              {total_relations2 - total_relations1} relations")

    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    if total_relations2 > total_relations1:
        print(f"\n  ✅ SUCCESS: RelationBuilder created {total_relations2} relations!")
        print(f"  ✅ RelationBuilder is WORKING correctly!")
    elif total_relations2 == total_relations1:
        print(f"\n  ⚠️  WARNING: Same number of relations with and without RelationBuilder")
        print(f"  ⚠️  RelationBuilder may not be creating relations")
        print(f"  ⚠️  Check logs above for '[RELATIONS] Created' messages")
    else:
        print(f"\n  ❌ ERROR: Fewer relations with RelationBuilder!")
        print(f"  ❌ Something is wrong")

    print("\n" + "=" * 80)
    print("✅ Test completed!")
    print("=" * 80 + "\n")

    await memory2.reset()
    await embedder2.close()

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(test_with_relation_builder()))

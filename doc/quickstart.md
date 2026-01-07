# mempy Quick Start Guide

A step-by-step guide to get started with mempy.

## Installation

```bash
# Core dependencies
pip install chromadb networkx aiohttp

# Optional: for development
pip install "mempy[dev]"
```

## Basic Usage (5 Minutes)

### Step 1: Create an Embedder

mempy requires you to provide an embedder. Here's a simple example using a local model:

```python
from mempy import Embedder
from typing import List

class SimpleEmbedder(Embedder):
    def __init__(self):
        self._dimension = 768  # Must declare your dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, text: str) -> List[float]:
        # Replace with your actual embedding logic
        # This is just a placeholder
        import hashlib
        hash_obj = hashlib.sha256(text.encode())
        return [hash_obj.digest()[i % 32] / 255.0 for i in range(768)]
```

### Step 2: Create a Memory Instance

```python
import asyncio
import mempy

async def main():
    memory = mempy.Memory(
        embedder=SimpleEmbedder(),
        verbose=True  # See what's happening
    )

    # Add some memories
    await memory.add("I love hiking", user_id="alice")
    await memory.add("Alice is a software engineer", user_id="alice")
    await memory.add("Alice lives in San Francisco", user_id="alice")

asyncio.run(main())
```

### Step 3: Search Memories

```python
async def main():
    memory = mempy.Memory(embedder=SimpleEmbedder())

    # First add some data
    await memory.add("I prefer working remotely", user_id="alice")

    # Then search
    results = await memory.search("work preference", user_id="alice")
    for m in results:
        print(f"{m.memory_id}: {m.content}")

asyncio.run(main())
```

## Using Real Embedding Models

### Option A: Sentence-Transformers (Local)

```bash
pip install sentence-transformers
```

```python
from sentence_transformers import SentenceTransformer
from mempy import Embedder
from typing import List
import asyncio

class STEncoder(Embedder):
    def __init__(self, model="BAAI/bge-small-en-v1.5"):
        self.model = SentenceTransformer(model)
        self._dim = self.model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dim

    async def embed(self, text: str) -> List[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.model.encode(text).tolist()
        )

# Use it
memory = mempy.Memory(embedder=STEncoder())
```

### Option B: Qwen (Local via vLLM)

See `tests/benchmarks/adapters/qwen.py` for a complete implementation.

### Option C: OpenAI API

```python
from tests.benchmarks.adapters import OpenAIEmbedder
import os

embedder = OpenAIEmbedder(
    api_key=os.environ["OPENAI_API_KEY"],
    model="text-embedding-3-small"
)

memory = mempy.Memory(embedder=embedder)
```

## Working with Relations

```python
import mempy

async def main():
    memory = mempy.Memory(embedder=YourEmbedder())

    # Add related memories
    fact1 = await memory.add("Python is a programming language", user_id="dev")
    fact2 = await memory.add("Python was created by Guido van Rossum", user_id="dev")

    # Connect them
    await memory.add_relation(
        from_id=fact1,
        to_id=fact2,
        relation_type=mempy.RelationType.PROPERTY_OF
    )

    # Query relations
    relations = await memory.get_relations(fact1)
    for rel in relations:
        print(f"{rel.from_id} --[{rel.type.value}]--> {rel.to_id}")

asyncio.run(main())
```

## Using with Processors

Processors intelligently decide whether to add, update, or ignore new content:

```python
from mempy import MemoryProcessor, ProcessorResult, Memory
from typing import List

class RuleBasedProcessor(MemoryProcessor):
    async def process(
        self,
        content: str,
        existing_memories: List[Memory]
    ) -> ProcessorResult:
        # Simple rule: if very similar, update instead of add
        for mem in existing_memories:
            if self._similarity(content, mem.content) > 0.9:
                return ProcessorResult(
                    action="update",
                    memory_id=mem.memory_id,
                    content=content,
                    reason="High similarity with existing memory"
                )
        return ProcessorResult(action="add", reason="New content")

    def _similarity(self, a: str, b: str) -> float:
        # Your similarity logic here
        return 0.0

# Use with processor
memory = mempy.Memory(
    embedder=YourEmbedder(),
    processor=RuleBasedProcessor()
)
```

## Complete Example

```python
import asyncio
from mempy import Memory, Embedder
from typing import List

class MyEmbedder(Embedder):
    def __init__(self):
        self._dimension = 768

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, text: str) -> List[float]:
        # Your embedding logic
        return [0.1] * 768

async def main():
    # Initialize
    memory = Memory(
        embedder=MyEmbedder(),
        verbose=True
    )

    # Add memories about a user
    await memory.add("I'm a software engineer", user_id="alice")
    await memory.add("I love Python and Rust", user_id="alice")
    await memory.add("I work at a startup", user_id="alice")

    # Search
    print("\n=== Searching for 'work' ===")
    results = await memory.search("work", user_id="alice")
    for m in results:
        print(f"Found: {m.content}")

    # Get all memories
    print("\n=== All memories for alice ===")
    all_memories = await memory.get_all(user_id="alice")
    for m in all_memories:
        print(f"- {m.content}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Next Steps

- Read the [API Reference](api.md) for detailed API documentation
- Check the [Adapter Guide](adapter-guide.md) for embedder implementations
- See the [Benchmark Guide](benchmark-guide.md) for evaluation

## Troubleshooting

### Issue: "Module not found: chromadb"

```bash
pip install chromadb networkx aiohttp
```

### Issue: Embedder dimension mismatch

Make sure your embedder's `dimension` property returns the correct value for your model.

### Issue: No search results

- Make sure you're using the correct `user_id` in search
- Check that memories were actually added (use `verbose=True`)
- Verify your embedder is generating meaningful vectors

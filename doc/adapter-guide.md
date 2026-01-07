# Embedder Adapter Development Guide

This guide explains how to create custom embedder adapters for mempy.

## Overview

mempy requires users to provide an embedder that implements the `Embedder` interface. This design allows mempy to work with any embedding model - local or remote, open-source or commercial.

## The Embedder Interface

```python
from mempy import Embedder
from typing import List

class MyEmbedder(Embedder):
    @property
    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        pass

    async def embed(self, text: str) -> List[float]:
        """Generate embedding for the given text."""
        pass
```

## Built-in Adapters

mempy includes several pre-built adapters in `tests/benchmarks/adapters/`:

| Adapter | Description | Use Case |
|---------|-------------|----------|
| `MockEmbedder` | Deterministic hash-based vectors | Testing |
| `QwenEmbedder` | Qwen models via vLLM | Local inference |
| `OpenAIEmbedder` | OpenAI API | Cloud embeddings |

## Example Implementations

### 1. HTTP API Embedder

For models served via HTTP with OpenAI-compatible API:

```python
import aiohttp
from typing import List
from mempy import Embedder

class HTTPEmbedder(Embedder):
    def __init__(self, base_url: str, model: str, dimension: int = 768):
        self.base_url = base_url
        self.model = model
        self._dimension = dimension
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = aiohttp.ClientSession()
        return self._client

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, text: str) -> List[float]:
        client = self._get_client()
        async with client.post(
            f"{self.base_url}/v1/embeddings",
            json={"model": self.model, "input": text},
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data["data"][0]["embedding"]

    async def close(self):
        if self._client:
            await self._client.close()
```

### 2. Sentence-Transformers Embedder

For local sentence-transformers models:

```python
from typing import List
from mempy import Embedder
import asyncio
from threading import Lock

class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()
        self._lock = Lock()  # For thread-safe execution

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, text: str) -> List[float]:
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.model.encode(text).tolist()
        )
```

### 3. Ollama Embedder

For Ollama-hosted models:

```python
import aiohttp
from typing import List
from mempy import Embedder

class OllamaEmbedder(Embedder):
    def __init__(self, model: str = "nomic-embed-text", host: str = "localhost:11434"):
        self.model = model
        self.host = host
        self._dimension = 768  # Default for nomic-embed-text
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = aiohttp.ClientSession()
        return self._client

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, text: str) -> List[float]:
        client = self._get_client()
        async with client.post(
            f"http://{self.host}/api/embed",
            json={"model": self.model, "prompt": text},
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data["embedding"]
```

## Best Practices

### 1. Declare Dimension Explicitly

The `dimension` property should be constant and declared at initialization:

```python
class GoodEmbedder(Embedder):
    def __init__(self):
        self._dimension = 768  # Fixed at init

    @property
    def dimension(self) -> int:
        return self._dimension
```

### 2. Handle Errors Gracefully

Wrap external API calls with proper error handling:

```python
async def embed(self, text: str) -> List[float]:
    try:
        return await self._call_api(text)
    except aiohttp.ClientError as e:
        raise EmbedderError(f"API call failed: {e}") from e
```

### 3. Support Async Context Manager

For embedders that manage resources (HTTP clients, etc.):

```python
async def __aenter__(self):
    return self

async def __aexit__(self, *args):
    await self.close()
```

### 4. Batch Embedding (Optional)

For better performance with multiple texts:

```python
async def embed_batch(self, texts: List[str]) -> List[List[float]]:
    """Generate embeddings for multiple texts efficiently."""
    # Implementation depends on your API
    pass
```

## Testing Your Adapter

```python
import asyncio
from your_adapter import YourEmbedder
from mempy import Memory

async def test():
    embedder = YourEmbedder()

    # Test basic functionality
    vector = await embedder.embed("Hello, world!")
    assert len(vector) == embedder.dimension
    assert all(isinstance(x, float) for x in vector)

    # Test with mempy
    memory = Memory(embedder=embedder)
    await memory.add("Test memory", user_id="test")
    results = await memory.search("Test", user_id="test")
    assert len(results) > 0

asyncio.run(test())
```

## Common Model Dimensions

| Model | Dimension |
|-------|-----------|
| OpenAI text-embedding-3-small | 1536 |
| OpenAI text-embedding-3-large | 3072 |
| BAAI/bge-small-en-v1.5 | 384 |
| BAAI/bge-base-en-v1.5 | 768 |
| BAAI/bge-large-en-v1.5 | 1024 |
| nomic-embed-text | 768 |
| Qwen2.5-7B | 3072 |
| Qwen2.5-32B | 4096 |
| Qwen3-32B | 4096 |

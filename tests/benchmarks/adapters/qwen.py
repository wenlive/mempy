"""Qwen model adapter for embedding generation.

This adapter supports Qwen models including:
- Qwen3-235B-A22B
- Qwen3-32B
- Other Qwen models with OpenAI-compatible embedding API

The adapter connects to a locally running model server (e.g., vLLM) that
provides an OpenAI-compatible /v1/embeddings endpoint.
"""

import asyncio
from typing import Any, List, Optional

from mempy import Embedder


class QwenEmbedder(Embedder):
    """
    Qwen model embedder using vLLM or OpenAI-compatible API.

    This adapter connects to a locally running Qwen model server
    (e.g., via vLLM) to generate embeddings.

    Example:
        >>> # Start vLLM server with Qwen model:
        >>> # vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
        >>>
        >>> embedder = QwenEmbedder(
        ...     base_url="http://localhost:8000",
        ...     model_name="Qwen/Qwen2.5-7B-Instruct"
        ... )
        >>> vector = await embedder.embed("Hello, world!")
    """

    # Default dimensions for common Qwen models
    DEFAULT_DIMENSIONS = {
        "qwen3-235b-a22b": 8192,
        "qwen3-32b": 4096,
        "qwen2.5-7b": 3072,
        "qwen2.5-32b": 4096,
        "qwen2-7b": 2048,
    }

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model_name: str = "qwen3-32b",
        dimension: Optional[int] = None,
        timeout: int = 30,
    ):
        """Initialize the Qwen embedder.

        Args:
            base_url: Base URL of the model server (vLLM/OpenAI-compatible)
            model_name: Name/identifier of the Qwen model
            dimension: Embedding dimension (auto-detected from known models if None)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self._client: Optional[Any] = None

        # Auto-detect dimension if not specified
        if dimension is None:
            model_key = model_name.lower().replace("/", "-").replace("_", "-")
            for known_key, known_dim in self.DEFAULT_DIMENSIONS.items():
                if known_key in model_key:
                    dimension = known_dim
                    break
            if dimension is None:
                dimension = 4096  # Default fallback

        self._dimension = dimension

    def _get_client(self):
        """Lazy initialization of HTTP client."""
        if self._client is None:
            import aiohttp
            self._client = aiohttp.ClientSession()
        return self._client

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension

    async def embed(self, text: str) -> List[float]:
        """Generate embedding for text using Qwen model.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as a list of floats

        Raises:
            aiohttp.ClientError: If the request fails
            asyncio.TimeoutError: If the request times out
        """
        client = self._get_client()

        url = f"{self.base_url}/v1/embeddings"

        try:
            async with client.post(
                url,
                json={
                    "model": self.model_name,
                    "input": text,
                },
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

                # Handle OpenAI-compatible response format
                if "data" in data and len(data["data"]) > 0:
                    return data["data"][0]["embedding"]
                else:
                    raise ValueError(f"Unexpected response format: {data}")

        except aiohttp.ClientError as e:
            raise ConnectionError(
                f"Failed to connect to Qwen server at {self.base_url}. "
                f"Make sure vLLM is running with embedding support enabled. Error: {e}"
            ) from e

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        This is more efficient than calling embed() multiple times
        as it batches requests to the server.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        client = self._get_client()
        url = f"{self.base_url}/v1/embeddings"

        async with client.post(
            url,
            json={
                "model": self.model_name,
                "input": texts,
            },
            timeout=aiohttp.ClientTimeout(total=self.timeout * len(texts)),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()

            return [item["embedding"] for item in data["data"]]

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.close()
            self._client = None

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args):
        """Async context manager exit."""
        await self.close()

    def __repr__(self) -> str:
        """Return string representation."""
        return f"QwenEmbedder(base_url={self.base_url!r}, model={self.model_name!r}, dim={self._dimension})"

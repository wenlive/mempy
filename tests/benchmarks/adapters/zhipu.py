"""Zhipu AI embedding model adapter.

This adapter supports Zhipu AI's embedding API for models:
- embedding-2: Fixed 1024 dimensions
- embedding-3: Customizable dimensions (512-4096)

Requires the zai-sdk package to be installed.
"""

import asyncio
import os
from typing import List, Optional

from mempy import Embedder


class ZhipuEmbedder(Embedder):
    """
    Zhipu AI embedding model adapter.

    This adapter uses Zhipu AI's API to generate embeddings.

    Example:
        >>> embedder = ZhipuEmbedder(
        ...     api_key="your-api-key",
        ...     model="embedding-3"
        ... )
        >>> vector = await embedder.embed("Hello, world!")
    """

    # ZhipuAI model dimensions
    MODEL_DIMENSIONS = {
        "embedding-2": 1024,
        "embedding-3": 1024,  # default, can be customized
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "embedding-3",
        dimension: Optional[int] = None,
        timeout: int = 30,
    ):
        """Initialize the Zhipu embedder.

        Args:
            api_key: ZhipuAI API key (defaults to ZHIPUAI_API_KEY env var)
            model: Model name to use (embedding-2 or embedding-3)
            dimension: Embedding dimension (auto-detected if None)
            timeout: Request timeout in seconds
        """
        try:
            from zai import ZhipuAiClient
            self.ZhipuAiClient = ZhipuAiClient
        except ImportError:
            raise ImportError(
                "zai-sdk package is required. Install it with: pip install zai-sdk"
            )

        self.model = model
        self.timeout = timeout

        # Get API key from parameter or environment
        self.api_key = api_key or os.environ.get("ZHIPUAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ZhipuAI API key must be provided via api_key parameter "
                "or ZHIPUAI_API_KEY environment variable"
            )

        # Initialize client
        self.client = self.ZhipuAiClient(api_key=self.api_key)

        # Auto-detect dimension if not specified
        if dimension is None:
            dimension = self.MODEL_DIMENSIONS.get(model, 1024)

        self._dimension = dimension
        self.total_tokens = 0

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension

    async def embed(self, text: str) -> List[float]:
        """Generate embedding for text using ZhipuAI API.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as a list of floats

        Raises:
            ConnectionError: If the API request fails
        """
        try:
            # Run sync SDK in async context
            # IMPORTANT: input must be a list!
            response = await asyncio.to_thread(
                self.client.embeddings.create,
                model=self.model,
                input=[text],  # Must be a list
            )

            # Track token usage
            if hasattr(response, 'usage') and response.usage:
                self.total_tokens += response.usage.total_tokens

            return response.data[0].embedding

        except Exception as e:
            raise ConnectionError(f"ZhipuAI API request failed: {e}") from e

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        try:
            response = await asyncio.to_thread(
                self.client.embeddings.create,
                model=self.model,
                input=texts,  # Already a list
            )

            # Track token usage
            if hasattr(response, 'usage') and response.usage:
                self.total_tokens += response.usage.total_tokens

            return [item.embedding for item in response.data]

        except Exception as e:
            raise ConnectionError(f"ZhipuAI API request failed: {e}") from e

    async def close(self):
        """Close the ZhipuAI client."""
        # ZhipuAI client doesn't require explicit cleanup
        pass

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args):
        """Async context manager exit."""
        await self.close()

    def __repr__(self) -> str:
        """Return string representation."""
        return f"ZhipuEmbedder(model={self.model!r}, dim={self._dimension})"

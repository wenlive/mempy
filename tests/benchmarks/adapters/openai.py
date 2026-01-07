"""OpenAI embedding model adapter.

This adapter supports OpenAI's embedding API for models like:
- text-embedding-3-small
- text-embedding-3-large
- text-embedding-ada-002

Requires the openai package to be installed.
"""

import asyncio
from typing import List, Optional

from mempy import Embedder


class OpenAIEmbedder(Embedder):
    """
    OpenAI embedding model adapter.

    This adapter uses OpenAI's API to generate embeddings.

    Example:
        >>> embedder = OpenAIEmbedder(
        ...     api_key="your-api-key",
        ...     model="text-embedding-3-small"
        ... )
        >>> vector = await embedder.embed("Hello, world!")
    """

    # OpenAI model dimensions
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        dimension: Optional[int] = None,
        timeout: int = 30,
    ):
        """Initialize the OpenAI embedder.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model name to use
            dimension: Embedding dimension (auto-detected if None)
            timeout: Request timeout in seconds
        """
        try:
            import openai
            self.openai = openai
        except ImportError:
            raise ImportError(
                "OpenAI package is required. Install it with: pip install openai"
            )

        self.model = model
        self.timeout = timeout

        # Get API key from parameter or environment
        import os
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key must be provided via api_key parameter "
                "or OPENAI_API_KEY environment variable"
            )

        # Initialize async client
        self.client = self.openai.AsyncOpenAI(api_key=self.api_key)

        # Auto-detect dimension if not specified
        if dimension is None:
            dimension = self.MODEL_DIMENSIONS.get(model, 1536)

        self._dimension = dimension

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension

    async def embed(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI API.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as a list of floats

        Raises:
            openai.APIError: If the API request fails
        """
        try:
            response = await self.client.embeddings.create(
                input=text,
                model=self.model,
                timeout=self.timeout,
            )
            return response.data[0].embedding
        except self.openai.APIError as e:
            raise ConnectionError(f"OpenAI API request failed: {e}") from e

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of input texts (max 2048 for OpenAI API)

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        if len(texts) > 2048:
            raise ValueError("OpenAI API supports max 2048 texts per request")

        try:
            response = await self.client.embeddings.create(
                input=texts,
                model=self.model,
                timeout=self.timeout * len(texts),
            )
            return [item.embedding for item in response.data]
        except self.openai.APIError as e:
            raise ConnectionError(f"OpenAI API request failed: {e}") from e

    async def close(self):
        """Close the OpenAI client."""
        await self.client.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args):
        """Async context manager exit."""
        await self.close()

    def __repr__(self) -> str:
        """Return string representation."""
        return f"OpenAIEmbedder(model={self.model!r}, dim={self._dimension})"

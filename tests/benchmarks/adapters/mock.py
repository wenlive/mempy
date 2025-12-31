"""Mock embedder for testing without a real model.

This module provides a simple mock embedder that generates deterministic
vectors based on text hashing. It's useful for testing the evaluation
framework without needing a real embedding model.
"""

import hashlib
from typing import List

from mempy import Embedder


class MockEmbedder(Embedder):
    """
    A mock embedder for testing purposes.

    This embedder generates deterministic embeddings based on the hash
    of the input text. It should NOT be used for production.

    Example:
        >>> embedder = MockEmbedder(dimension=768)
        >>> vector = await embedder.embed("Hello")
        >>> len(vector)
        768
    """

    def __init__(self, dimension: int = 768):
        """Initialize the mock embedder.

        Args:
            dimension: Embedding vector dimension
        """
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension

    async def embed(self, text: str) -> List[float]:
        """Generate a deterministic mock embedding for the given text.

        Args:
            text: Input text

        Returns:
            A deterministic embedding vector based on text hash
        """
        # Create a deterministic hash of the text
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()

        # Expand hash to desired dimension
        vector = []
        for i in range(self._dimension):
            # Use different bytes of the hash to generate values
            byte_index = i % len(hash_bytes)
            # Convert to a float between 0 and 1
            value = hash_bytes[byte_index] / 255.0
            vector.append(value)

        return vector

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        return [await self.embed(text) for text in texts]

    def __repr__(self) -> str:
        """Return string representation."""
        return f"MockEmbedder(dimension={self._dimension})"

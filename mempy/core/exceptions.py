"""Exception classes for mempy."""


class MempyError(Exception):
    """Base exception for all mempy errors."""

    pass


class EmbedderError(MempyError):
    """Raised when embedder fails to generate embeddings."""

    pass


class StorageError(MempyError):
    """Raised when storage operation fails."""

    pass


class ProcessorError(MempyError):
    """Raised when memory processor fails."""

    pass

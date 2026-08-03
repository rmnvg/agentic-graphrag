from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """Interface for embedding providers used by the document pipeline."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the embedding model identifier."""

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Return the number of floats in each embedding vector."""

    @abstractmethod
    def embed_texts(self, texts: list[str], batch_size: int) -> list[list[float]]:
        """Embed multiple texts in batches and return one vector per text."""

from functools import lru_cache
from typing import Any

from app.services.embeddings.base_embedder import BaseEmbedder

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


class SentenceTransformerEmbedder(BaseEmbedder):
    """Sentence Transformers embedder backed by a reusable in-process model."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        """Load a Sentence Transformer model.

        The model is intentionally loaded once through get_sentence_transformer_embedder()
        and reused across requests to avoid repeated model initialization cost.

        Args:
            model_name: Hugging Face model id for the embedding model.
        """
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._model: Any = SentenceTransformer(model_name)
        self._embedding_dimension = int(self._model.get_sentence_embedding_dimension())

    @property
    def model_name(self) -> str:
        """Return the embedding model identifier."""
        return self._model_name

    @property
    def embedding_dimension(self) -> int:
        """Return the number of floats in each embedding vector."""
        return self._embedding_dimension

    def embed_texts(self, texts: list[str], batch_size: int) -> list[list[float]]:
        """Generate embeddings for texts using Sentence Transformers batching.

        Args:
            texts: Chunk texts to embed.
            batch_size: Number of chunks encoded per model batch.

        Returns:
            Embeddings as JSON-serializable lists of floats.
        """
        if not texts:
            return []

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.astype(float).tolist()


@lru_cache(maxsize=1)
def get_sentence_transformer_embedder(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> SentenceTransformerEmbedder:
    """Return a cached Sentence Transformers embedder instance."""
    return SentenceTransformerEmbedder(model_name=model_name)

from abc import ABC, abstractmethod
from typing import Any


class BaseChunker(ABC):
    """Interface for document chunking strategies."""

    @abstractmethod
    def chunk(self, parsed_document: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate chunks from a parsed document payload."""

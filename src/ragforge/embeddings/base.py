from abc import ABC, abstractmethod

import numpy as np


class EmbeddingModel(ABC):
    """
    Abstract interface for text embedding models.

    Any embedding implementation used by RAGForge
    should provide the same encode interface.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of embeddings."""
        raise NotImplementedError

    @abstractmethod
    def encode(
        self,
        texts: list[str],
    ) -> np.ndarray:
        """
        Convert texts into dense vector embeddings.

        Returns:
            NumPy array with shape:
            (number_of_texts, embedding_dimension)
        """
        raise NotImplementedError
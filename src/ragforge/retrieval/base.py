from abc import ABC, abstractmethod

import numpy as np

from ragforge.retrieval.record import VectorRecord
from ragforge.retrieval.result import SearchResult


class VectorStore(ABC):
    """
    Abstract interface for vector databases/stores.
    """

    @abstractmethod
    def add(
        self,
        records: list[VectorRecord],
    ) -> None:
        """
        Add vector records to the store.
        """
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """
        Search for the most similar records.
        """
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        """
        Return the number of stored records.
        """
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """
        Remove all records.
        """
        raise NotImplementedError
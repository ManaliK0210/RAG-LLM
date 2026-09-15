from pathlib import Path
import pickle

import numpy as np

from ragforge.retrieval.base import VectorStore
from ragforge.retrieval.record import VectorRecord
from ragforge.retrieval.result import SearchResult


class NumpyVectorStore(VectorStore):
    """
    In-memory vector store using NumPy.

    Similarity is calculated using cosine similarity.
    """

    def __init__(
        self,
        dimension: int,
    ) -> None:
        if dimension <= 0:
            raise ValueError(
                "dimension must be greater than 0."
            )

        self.dimension = dimension

        self._records: list[VectorRecord] = []

        self._vectors = np.empty(
            (0, dimension),
            dtype=np.float32,
        )

    def add(
        self,
        records: list[VectorRecord],
    ) -> None:
        """
        Add records to the vector store.
        """

        if not isinstance(records, list):
            raise TypeError(
                "records must be a list."
            )

        if not records:
            return

        for record in records:
            if not isinstance(
                record,
                VectorRecord,
            ):
                raise TypeError(
                    "All records must be "
                    "VectorRecord objects."
                )

            if record.vector.shape[0] != self.dimension:
                raise ValueError(
                    "Vector dimension does not "
                    "match the vector store."
                )

            if not np.all(
                np.isfinite(record.vector)
            ):
                raise ValueError(
                    "Vector contains non-finite values."
                )

        existing_ids = {
            record.id
            for record in self._records
        }

        for record in records:
            if record.id in existing_ids:
                raise ValueError(
                    f"Duplicate record id: {record.id}"
                )

        new_vectors = np.stack(
            [
                record.vector.astype(
                    np.float32
                )
                for record in records
            ]
        )

        self._records.extend(records)

        self._vectors = np.vstack(
            [
                self._vectors,
                new_vectors,
            ]
        )

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """
        Search using cosine similarity.
        """

        if not isinstance(
            query_vector,
            np.ndarray,
        ):
            raise TypeError(
                "query_vector must be "
                "a NumPy array."
            )

        if query_vector.ndim != 1:
            raise ValueError(
                "query_vector must be "
                "one-dimensional."
            )

        if query_vector.shape[0] != self.dimension:
            raise ValueError(
                "Query vector dimension does "
                "not match the vector store."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        if not self._records:
            return []

        query_norm = np.linalg.norm(
            query_vector
        )

        if query_norm == 0:
            raise ValueError(
                "Query vector cannot be zero."
            )

        vector_norms = np.linalg.norm(
            self._vectors,
            axis=1,
        )

        if np.any(vector_norms == 0):
            raise ValueError(
                "Stored vectors cannot be zero."
            )

        similarities = (
            self._vectors @ query_vector
        ) / (
            vector_norms * query_norm
        )

        k = min(
            top_k,
            len(self._records),
        )

        indices = np.argsort(
            similarities
        )[::-1][:k]

        return [
            SearchResult(
                id=self._records[index].id,
                text=self._records[index].text,
                score=float(
                    similarities[index]
                ),
                metadata=self._records[
                    index
                ].metadata.copy(),
            )
            for index in indices
        ]

    def count(self) -> int:
        return len(self._records)

    def clear(self) -> None:
        self._records.clear()

        self._vectors = np.empty(
            (0, self.dimension),
            dtype=np.float32,
        )

    def save(
        self,
        path: str | Path,
    ) -> None:
        """
        Persist the vector store to disk.
        """

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "dimension": self.dimension,
            "records": self._records,
            "vectors": self._vectors,
        }

        with path.open(
            "wb"
        ) as file:
            pickle.dump(
                payload,
                file,
            )

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "NumpyVectorStore":
        """
        Load a persisted vector store.
        """

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Vector store not found: {path}"
            )

        with path.open(
            "rb"
        ) as file:
            payload = pickle.load(file)

        store = cls(
            dimension=payload["dimension"]
        )

        store._records = payload["records"]
        store._vectors = payload["vectors"]

        return store
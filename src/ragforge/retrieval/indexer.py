from uuid import uuid4

import numpy as np

from ragforge.embeddings.base import EmbeddingModel
from ragforge.ingestion.document import Document
from ragforge.retrieval.record import VectorRecord
from ragforge.retrieval.base import VectorStore


class VectorIndexer:
    """
    Converts documents into vector records
    and stores them in a VectorStore.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
    ) -> None:
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def index(
        self,
        documents: list[Document],
    ) -> list[str]:
        """
        Embed documents and add them to the vector store.

        Returns:
            IDs assigned to the indexed documents.
        """

        if not isinstance(
            documents,
            list,
        ):
            raise TypeError(
                "documents must be a list."
            )

        if not all(
            isinstance(
                document,
                Document,
            )
            for document in documents
        ):
            raise TypeError(
                "All items must be Document objects."
            )

        if not documents:
            return []

        texts = [
            document.text
            for document in documents
        ]

        vectors = self.embedding_model.encode(
            texts
        )

        if vectors.shape != (
            len(documents),
            self.embedding_model.dimension,
        ):
            raise ValueError(
                "Embedding model returned "
                "an unexpected shape."
            )

        records: list[VectorRecord] = []
        ids: list[str] = []

        for document, vector in zip(
            documents,
            vectors,
        ):
            record_id = str(uuid4())

            metadata = document.metadata.copy()
            metadata["source"] = document.source

            records.append(
                VectorRecord(
                    id=record_id,
                    text=document.text,
                    vector=np.asarray(
                        vector,
                        dtype=np.float32,
                    ),
                    metadata=metadata,
                )
            )

            ids.append(record_id)

        self.vector_store.add(records)

        return ids
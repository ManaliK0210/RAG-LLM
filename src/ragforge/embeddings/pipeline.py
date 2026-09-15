import numpy as np

from ragforge.embeddings.base import EmbeddingModel
from ragforge.ingestion.document import Document


class EmbeddingPipeline:
    """
    Converts Document chunks into dense embeddings.
    """

    def __init__(
        self,
        model: EmbeddingModel,
    ) -> None:
        self.model = model

    def embed_documents(
        self,
        documents: list[Document],
    ) -> np.ndarray:
        """
        Embed a list of document chunks.

        Returns:
            Array with shape:
            (number_of_documents, embedding_dimension)
        """

        if not isinstance(documents, list):
            raise TypeError(
                "documents must be a list."
            )

        if not all(
            isinstance(document, Document)
            for document in documents
        ):
            raise TypeError(
                "All items must be Document objects."
            )

        if not documents:
            return np.empty(
                (0, self.model.dimension),
                dtype=np.float32,
            )

        texts = [
            document.text
            for document in documents
        ]

        return self.model.encode(texts)

    def embed_query(
        self,
        query: str,
    ) -> np.ndarray:
        """
        Embed a single user query.

        Returns:
            One-dimensional embedding vector.
        """

        if not isinstance(query, str):
            raise TypeError(
                "query must be a string."
            )

        if not query.strip():
            raise ValueError(
                "query cannot be empty."
            )

        embedding = self.model.encode(
            [query]
        )

        return embedding[0]
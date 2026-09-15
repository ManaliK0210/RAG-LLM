from pathlib import Path

from ragforge.ingestion.batch import (
    BatchIngestionPipeline,
)
from ragforge.ingestion.document import Document
from ragforge.retrieval.indexer import (
    VectorIndexer,
)
from ragforge.retrieval.numpy_store import (
    NumpyVectorStore,
)


class KnowledgeBase:
    """
    Builds and manages a persistent vector knowledge base.

    Flow:
        source directory
        -> document ingestion
        -> chunking
        -> embedding
        -> vector indexing
        -> persistent vector store
    """

    def __init__(
        self,
        indexer: VectorIndexer,
        vector_store: NumpyVectorStore,
        ingestion_pipeline: BatchIngestionPipeline
        | None = None,
    ) -> None:
        self.indexer = indexer
        self.vector_store = vector_store

        self.ingestion_pipeline = (
            ingestion_pipeline
            if ingestion_pipeline is not None
            else BatchIngestionPipeline()
        )

    def build(
        self,
        source_directory: str | Path,
        persist_path: str | Path,
        recursive: bool = True,
    ) -> int:
        """
        Ingest documents, index their chunks, and persist
        the vector store.
        """

        documents = (
            self.ingestion_pipeline.process_directory(
                source_directory,
                recursive=recursive,
            )
        )

        if not documents:
            raise ValueError(
                "No supported documents were found "
                "in the source directory."
            )

        self.vector_store.clear()

        self.indexer.index(
            documents
        )

        persist_path = Path(
            persist_path
        )

        self.vector_store.save(
            persist_path
        )

        return len(documents)

    def add_documents(
        self,
        documents: list[Document],
        persist_path: str | Path,
    ) -> int:
        """
        Add already-ingested document chunks
        to the knowledge base.
        """

        if not documents:
            return 0

        self.indexer.index(
            documents
        )

        persist_path = Path(
            persist_path
        )

        self.vector_store.save(
            persist_path
        )

        return len(documents)

    @classmethod
    def load(
        cls,
        persist_path: str | Path,
        indexer: VectorIndexer,
        ingestion_pipeline: BatchIngestionPipeline
        | None = None,
    ) -> "KnowledgeBase":
        """
        Load a previously persisted knowledge base.
        """

        vector_store = (
            NumpyVectorStore.load(
                persist_path
            )
        )

        return cls(
            indexer=indexer,
            vector_store=vector_store,
            ingestion_pipeline=ingestion_pipeline,
        )

    def count(self) -> int:
        """Return the number of indexed chunks."""

        return self.vector_store.count()
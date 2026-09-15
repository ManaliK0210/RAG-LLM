from pathlib import Path

from ragforge.ingestion.chunker import RecursiveTextChunker
from ragforge.ingestion.cleaner import TextCleaner
from ragforge.ingestion.document import Document
from ragforge.ingestion.manager import DocumentIngestionManager


class DocumentIngestionPipeline:
    """
    End-to-end document ingestion pipeline.

    Flow:
        file
        -> loader
        -> document
        -> text cleaning
        -> chunking
        -> document chunks
    """

    def __init__(
        self,
        manager: DocumentIngestionManager | None = None,
        cleaner: TextCleaner | None = None,
        chunker: RecursiveTextChunker | None = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> None:
        self.manager = (
            manager
            if manager is not None
            else DocumentIngestionManager()
        )

        self.cleaner = (
            cleaner
            if cleaner is not None
            else TextCleaner()
        )

        self.chunker = (
            chunker
            if chunker is not None
            else RecursiveTextChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )

    def process(
        self,
        path: str | Path,
    ) -> list[Document]:
        """
        Load, clean, and chunk a document.

        Returns:
            A list of processed document chunks.
        """

        document = self.manager.load(path)

        cleaned_text = self.cleaner.clean(
            document.text
        )

        cleaned_document = Document(
            text=cleaned_text,
            source=document.source,
            metadata=document.metadata.copy(),
        )

        chunks = self.chunker.split(
            cleaned_document
        )

        total_chunks = len(chunks)

        processed_chunks: list[Document] = []

        for index, chunk in enumerate(chunks):
            metadata = chunk.metadata.copy()

            metadata["chunk_index"] = index
            metadata["total_chunks"] = total_chunks
            metadata["source"] = chunk.source

            processed_chunks.append(
                Document(
                    text=chunk.text,
                    source=chunk.source,
                    metadata=metadata,
                )
            )

        return processed_chunks
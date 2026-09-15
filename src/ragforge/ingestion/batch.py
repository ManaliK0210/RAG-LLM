from pathlib import Path

from ragforge.ingestion.document import Document
from ragforge.ingestion.pipeline import DocumentIngestionPipeline


class BatchIngestionPipeline:
    """
    Processes multiple documents from a directory.

    Each supported document is passed through the
    standard DocumentIngestionPipeline.
    """

    def __init__(
        self,
        pipeline: DocumentIngestionPipeline | None = None,
    ) -> None:
        self.pipeline = (
            pipeline
            if pipeline is not None
            else DocumentIngestionPipeline()
        )

    def process_directory(
        self,
        directory: str | Path,
        recursive: bool = True,
    ) -> list[Document]:
        """
        Process all supported documents in a directory.

        Args:
            directory: Directory containing documents.
            recursive: Whether to search subdirectories.

        Returns:
            A flat list of document chunks.
        """

        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(
                f"Directory not found: {directory}"
            )

        if not directory.is_dir():
            raise ValueError(
                f"Path is not a directory: {directory}"
            )

        extensions = set(
            self.pipeline.manager.supported_extensions()
        )

        if recursive:
            paths = (
                path
                for path in directory.rglob("*")
                if path.is_file()
            )
        else:
            paths = (
                path
                for path in directory.iterdir()
                if path.is_file()
            )

        chunks: list[Document] = []

        for path in sorted(paths):
            if path.suffix.lower() not in extensions:
                continue

            chunks.extend(
                self.pipeline.process(path)
            )

        return chunks
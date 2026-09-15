from pathlib import Path

from ragforge.ingestion.document import Document
from ragforge.ingestion.docx_loader import (
    DOCXDocumentLoader,
)
from ragforge.ingestion.html_loader import (
    HTMLDocumentLoader,
)
from ragforge.ingestion.loaders import (
    TextDocumentLoader,
)
from ragforge.ingestion.pdf_loader import (
    PDFDocumentLoader,
)
from ragforge.ingestion.csv_loader import (
    CSVDocumentLoader,
)


class DocumentIngestionManager:
    """
    Coordinates document loading across supported
    file formats.

    The manager selects the appropriate loader based
    on the file extension.
    """

    def __init__(self) -> None:
        self._loaders = {
            ".txt": TextDocumentLoader(),
            ".pdf": PDFDocumentLoader(),
            ".docx": DOCXDocumentLoader(),
            ".html": HTMLDocumentLoader(),
            ".htm": HTMLDocumentLoader(),
            ".csv": CSVDocumentLoader(),
        }

    def load(
        self,
        path: str | Path,
    ) -> Document:
        """
        Load a document using the appropriate loader.
        """

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Document not found: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Path is not a file: {path}"
            )

        extension = path.suffix.lower()

        loader = self._loaders.get(
            extension
        )

        if loader is None:
            supported = ", ".join(
                sorted(self._loaders.keys())
            )

            raise ValueError(
                f"Unsupported document format: "
                f"{extension}. "
                f"Supported formats: {supported}"
            )

        return loader.load(path)

    def supported_extensions(
        self,
    ) -> list[str]:
        """
        Return the supported file extensions.
        """

        return sorted(
            self._loaders.keys()
        )
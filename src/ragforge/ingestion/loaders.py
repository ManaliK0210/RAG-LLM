from pathlib import Path

from ragforge.ingestion.document import Document


class TextDocumentLoader:
    """
    Loader for plain-text documents.

    Reads a .txt file and converts it into the
    canonical RAGForge Document representation.
    """

    def load(
        self,
        path: str | Path,
    ) -> Document:
        """
        Load a text document from disk.
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

        if path.suffix.lower() != ".txt":
            raise ValueError(
                "TextDocumentLoader only supports .txt files."
            )

        text = path.read_text(
            encoding="utf-8"
        )

        return Document(
            text=text,
            source=str(path),
            metadata={
                "file_type": "txt",
            },
        )
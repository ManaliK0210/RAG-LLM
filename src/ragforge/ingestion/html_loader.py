from pathlib import Path

from bs4 import BeautifulSoup

from ragforge.ingestion.document import Document


class HTMLDocumentLoader:
    """
    Loader for HTML documents.

    Extracts visible text from HTML markup and converts
    the result into the canonical RAGForge Document.
    """

    def load(
        self,
        path: str | Path,
    ) -> Document:
        """
        Load an HTML document from disk.
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

        if path.suffix.lower() not in {
            ".html",
            ".htm",
        }:
            raise ValueError(
                "HTMLDocumentLoader only supports "
                ".html and .htm files."
            )

        html = path.read_text(
            encoding="utf-8"
        )

        soup = BeautifulSoup(
            html,
            "html.parser",
        )

        # Remove elements that do not contain useful
        # document content.
        for element in soup(
            ["script", "style", "noscript"]
        ):
            element.decompose()

        text = soup.get_text(
            separator="\n",
            strip=True,
        )

        if not text:
            raise ValueError(
                f"HTML contains no extractable text: {path}"
            )

        title = None

        if soup.title is not None:
            title = soup.title.get_text(
                strip=True
            )

        metadata = {
            "file_type": "html",
        }

        if title:
            metadata["title"] = title

        return Document(
            text=text,
            source=str(path),
            metadata=metadata,
        )
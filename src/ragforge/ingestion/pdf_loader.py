from pathlib import Path

import pymupdf

from ragforge.ingestion.document import Document


class PDFDocumentLoader:
    """
    Loader for PDF documents.

    Extracts text from every page and converts the PDF
    into the canonical RAGForge Document representation.
    """

    def load(
        self,
        path: str | Path,
    ) -> Document:
        """
        Load a PDF document from disk.
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

        if path.suffix.lower() != ".pdf":
            raise ValueError(
                "PDFDocumentLoader only supports .pdf files."
            )

        pdf = pymupdf.open(str(path))

        try:
            page_texts = []

            for page in pdf:
                page_texts.append(
                    page.get_text()
                )

            text = "\n\n".join(
                page_texts
            ).strip()

            if not text:
                raise ValueError(
                    f"PDF contains no extractable text: {path}"
                )

            metadata = {
                "file_type": "pdf",
                "page_count": len(pdf),
            }

            if pdf.metadata:
                if pdf.metadata.get("title"):
                    metadata["title"] = (
                        pdf.metadata["title"]
                    )

                if pdf.metadata.get("author"):
                    metadata["author"] = (
                        pdf.metadata["author"]
                    )

                if pdf.metadata.get("subject"):
                    metadata["subject"] = (
                        pdf.metadata["subject"]
                    )

        finally:
            pdf.close()

        return Document(
            text=text,
            source=str(path),
            metadata=metadata,
        )
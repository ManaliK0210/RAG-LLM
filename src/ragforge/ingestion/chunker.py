from ragforge.ingestion.document import Document


class RecursiveTextChunker:
    """
    Splits documents into overlapping text chunks.

    The chunker prefers natural boundaries in this order:

        paragraph → line → sentence → word

    Each resulting chunk remains a Document so that
    source information and metadata are preserved.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than 0."
            )

        if chunk_overlap < 0:
            raise ValueError(
                "chunk_overlap cannot be negative."
            )

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than "
                "chunk_size."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(
        self,
        document: Document,
    ) -> list[Document]:
        """
        Split one document into overlapping chunks.
        """

        if not isinstance(document, Document):
            raise TypeError(
                "document must be a Document instance."
            )

        text = document.text.strip()

        if len(text) <= self.chunk_size:
            return [
                Document(
                    text=text,
                    source=document.source,
                    metadata={
                        **document.metadata,
                        "chunk_index": 0,
                    },
                )
            ]

        raw_chunks = self._recursive_split(
            text=text,
        )

        chunks = self._add_overlap(
            raw_chunks,
        )

        return [
            Document(
                text=chunk,
                source=document.source,
                metadata={
                    **document.metadata,
                    "chunk_index": index,
                },
            )
            for index, chunk in enumerate(chunks)
        ]

    def _recursive_split(
        self,
        text: str,
    ) -> list[str]:
        """
        Recursively split text using natural separators.
        """

        separators = [
            "\n\n",
            "\n",
            ". ",
            " ",
        ]

        return self._split_with_separators(
            text=text,
            separators=separators,
        )

    def _split_with_separators(
        self,
        text: str,
        separators: list[str],
    ) -> list[str]:
        """
        Split text recursively until pieces fit
        within the configured chunk size.
        """

        if len(text) <= self.chunk_size:
            return [text.strip()]

        if not separators:
            return self._split_by_length(text)

        separator = separators[0]
        parts = text.split(separator)

        if len(parts) == 1:
            return self._split_with_separators(
                text=text,
                separators=separators[1:],
            )

        chunks: list[str] = []
        current = ""

        for part in parts:
            part = part.strip()

            if not part:
                continue

            candidate = (
                part
                if not current
                else current
                + separator
                + part
            )

            if len(candidate) <= self.chunk_size:
                current = candidate
                continue

            if current:
                chunks.append(current.strip())

            if len(part) <= self.chunk_size:
                current = part
            else:
                chunks.extend(
                    self._split_with_separators(
                        text=part,
                        separators=separators[1:],
                    )
                )
                current = ""

        if current:
            chunks.append(current.strip())

        return chunks

    def _split_by_length(
        self,
        text: str,
    ) -> list[str]:
        """
        Final fallback for text containing no useful
        separators.
        """

        return [
            text[index:index + self.chunk_size]
            for index in range(
                0,
                len(text),
                self.chunk_size,
            )
        ]

    def _add_overlap(
        self,
        chunks: list[str],
    ) -> list[str]:
        """
        Add character overlap between consecutive chunks.
        """

        if self.chunk_overlap == 0:
            return chunks

        overlapped: list[str] = []

        for index, chunk in enumerate(chunks):
            if index == 0:
                overlapped.append(chunk)
                continue

            previous = chunks[index - 1]

            overlap = previous[
                -self.chunk_overlap:
            ]

            overlapped.append(
                overlap + " " + chunk
            )

        return overlapped
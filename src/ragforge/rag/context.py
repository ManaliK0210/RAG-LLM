from dataclasses import dataclass, field
from typing import Any

from ragforge.retrieval.result import SearchResult


@dataclass
class ContextChunk:
    """
    A retrieved chunk prepared for use as LLM context.
    """

    text: str
    source: str
    score: float
    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class RetrievedContext:
    """
    Collection of retrieved chunks used to construct
    the final RAG context.
    """

    chunks: list[ContextChunk] = field(
        default_factory=list
    )

    @property
    def text(self) -> str:
        """
        Return all chunks formatted as context.
        """

        sections = []

        for index, chunk in enumerate(
            self.chunks,
            start=1,
        ):
            sections.append(
                (
                    f"[Source {index}]\n"
                    f"Source: {chunk.source}\n"
                    f"Content: {chunk.text}"
                )
            )

        return "\n\n".join(sections)

    @property
    def sources(self) -> list[str]:
        """
        Return unique source names in retrieval order.
        """

        sources = []

        for chunk in self.chunks:
            if chunk.source not in sources:
                sources.append(chunk.source)

        return sources


class ContextBuilder:
    """
    Converts retrieval results into structured RAG context.
    """

    def build(
        self,
        results: list[SearchResult],
        max_chunks: int | None = None,
    ) -> RetrievedContext:
        """
        Build context from search results.
        """

        if not isinstance(results, list):
            raise TypeError(
                "results must be a list."
            )

        if max_chunks is not None:
            if max_chunks <= 0:
                raise ValueError(
                    "max_chunks must be greater than 0."
                )

            results = results[:max_chunks]

        chunks = []

        for result in results:
            source = result.metadata.get(
                "source",
                "unknown",
            )

            chunks.append(
                ContextChunk(
                    text=result.text,
                    source=str(source),
                    score=result.score,
                    metadata=result.metadata.copy(),
                )
            )

        return RetrievedContext(
            chunks=chunks
        )
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RAGResponse:
    """
    Final response returned by the RAG pipeline.
    """

    answer: str
    sources: list[str] = field(
        default_factory=list
    )
    retrieved_chunks: int = 0
    metadata: dict[str, Any] = field(
        default_factory=dict
    )
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgenticRAGResponse:
    """Final response produced by the agentic RAG system."""

    answer: str
    sources: list[str] = field(
        default_factory=list
    )
    steps: int = 0
    used_retrieval: bool = False
    used_memory: bool = False
    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not isinstance(self.answer, str):
            raise TypeError(
                "answer must be a string."
            )

        if not self.answer.strip():
            raise ValueError(
                "answer cannot be empty."
            )

        if self.steps < 0:
            raise ValueError(
                "steps cannot be negative."
            )

        if not isinstance(self.sources, list):
            raise TypeError(
                "sources must be a list."
            )

        if not isinstance(self.metadata, dict):
            raise TypeError(
                "metadata must be a dictionary."
            )
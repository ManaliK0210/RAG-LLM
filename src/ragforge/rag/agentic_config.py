from dataclasses import dataclass


@dataclass(frozen=True)
class AgenticRAGConfig:
    """Configuration for the agentic RAG system."""

    max_steps: int = 5
    retrieval_top_k: int = 5
    memory_top_k: int = 3

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError(
                "max_steps must be greater than 0."
            )

        if self.retrieval_top_k <= 0:
            raise ValueError(
                "retrieval_top_k must be greater than 0."
            )

        if self.memory_top_k <= 0:
            raise ValueError(
                "memory_top_k must be greater than 0."
            )
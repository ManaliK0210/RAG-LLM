from dataclasses import dataclass


@dataclass
class RAGConfig:
    """
    Configuration for the RAG pipeline.
    """

    top_k: int = 5
    max_context_chunks: int = 5

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        if self.max_context_chunks <= 0:
            raise ValueError(
                "max_context_chunks must be greater than 0."
            )
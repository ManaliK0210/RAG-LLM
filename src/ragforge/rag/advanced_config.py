from dataclasses import dataclass


@dataclass
class AdvancedRAGConfig:
    """
    Configuration for advanced RAG retrieval.
    """

    initial_top_k: int = 10
    final_top_k: int = 5

    use_query_rewrite: bool = False
    use_multi_query: bool = False
    use_reranking: bool = True
    use_hybrid_search: bool = False

    num_queries: int = 3

    def __post_init__(self) -> None:
        if self.initial_top_k <= 0:
            raise ValueError(
                "initial_top_k must be greater than 0."
            )

        if self.final_top_k <= 0:
            raise ValueError(
                "final_top_k must be greater than 0."
            )

        if self.final_top_k > self.initial_top_k:
            raise ValueError(
                "final_top_k cannot exceed initial_top_k."
            )

        if self.num_queries <= 0:
            raise ValueError(
                "num_queries must be greater than 0."
            )
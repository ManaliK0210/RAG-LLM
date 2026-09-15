from dataclasses import dataclass


@dataclass
class RetrievalConfig:
    """
    Configuration for semantic retrieval.
    """

    top_k: int = 5
    score_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        if self.score_threshold is not None:
            if not -1.0 <= self.score_threshold <= 1.0:
                raise ValueError(
                    "score_threshold must be between "
                    "-1 and 1."
                )
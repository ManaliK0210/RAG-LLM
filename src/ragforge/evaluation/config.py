from dataclasses import dataclass


@dataclass
class EvaluationConfig:
    """
    Configuration for RAG evaluation.
    """

    k_values: tuple[int, ...] = (1, 3, 5)

    def __post_init__(self) -> None:
        if not self.k_values:
            raise ValueError(
                "k_values cannot be empty."
            )

        if any(k <= 0 for k in self.k_values):
            raise ValueError(
                "All k values must be greater than 0."
            )

        if tuple(sorted(self.k_values)) != self.k_values:
            raise ValueError(
                "k_values must be sorted in ascending order."
            )
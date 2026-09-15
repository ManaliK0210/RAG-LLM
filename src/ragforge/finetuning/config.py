from dataclasses import dataclass


@dataclass
class LoRAConfig:
    """
    Configuration for Low-Rank Adaptation.
    """

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05

    learning_rate: float = 1e-4
    epochs: int = 1
    batch_size: int = 4

    target_modules: tuple[str, ...] = (
        "c_attn",
    )

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError(
                "rank must be greater than 0."
            )

        if self.alpha <= 0:
            raise ValueError(
                "alpha must be greater than 0."
            )

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(
                "dropout must be between 0 and 1."
            )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be greater than 0."
            )

        if self.epochs <= 0:
            raise ValueError(
                "epochs must be greater than 0."
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than 0."
            )

        if not self.target_modules:
            raise ValueError(
                "target_modules cannot be empty."
            )

        scaling = self.alpha / self.rank

        if scaling <= 0:
            raise ValueError(
                "LoRA scaling must be greater than 0."
            )
from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryConfig:
    """Configuration for the RAGForge memory subsystem."""

    short_term_max_turns: int = 10
    long_term_max_items: int = 100

    def __post_init__(self) -> None:
        if self.short_term_max_turns <= 0:
            raise ValueError(
                "short_term_max_turns must be greater than 0."
            )

        if self.long_term_max_items <= 0:
            raise ValueError(
                "long_term_max_items must be greater than 0."
            )
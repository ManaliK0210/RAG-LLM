from dataclasses import dataclass


@dataclass
class AgentConfig:
    """
    Configuration for the agent execution loop.
    """

    max_steps: int = 5

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError(
                "max_steps must be greater than 0."
            )
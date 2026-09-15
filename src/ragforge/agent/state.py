from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentStep:
    """
    Represents one action taken by the agent.
    """

    action: str
    tool_name: str | None = None
    tool_input: dict[str, Any] = field(
        default_factory=dict
    )
    observation: Any = None


@dataclass
class AgentState:
    """
    State maintained during agent execution.
    """

    query: str
    steps: list[AgentStep] = field(
        default_factory=list
    )
    final_answer: str | None = None
    completed: bool = False

    def add_step(
        self,
        step: AgentStep,
    ) -> None:
        self.steps.append(step)

    @property
    def step_count(self) -> int:
        return len(self.steps)
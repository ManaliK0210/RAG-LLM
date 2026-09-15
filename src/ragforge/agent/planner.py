from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AgentAction:
    """
    Action proposed by the planner.
    """

    action: str
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    final_answer: str | None = None


class Planner:
    """
    Planner interface for deciding the next agent action.
    """

    def plan(
        self,
        query: str,
        observations: list[Any],
        available_tools: list[dict[str, Any]],
    ) -> AgentAction:
        """
        Decide the next action.

        This baseline planner is intentionally deterministic.
        An LLM-backed planner will be added later.
        """

        if not query.strip():
            raise ValueError(
                "query cannot be empty."
            )

        if observations:
            return AgentAction(
                action="finish",
                final_answer=str(
                    observations[-1]
                ),
            )

        return AgentAction(
            action="finish",
            final_answer=(
                "No tool action was required."
            ),
        )
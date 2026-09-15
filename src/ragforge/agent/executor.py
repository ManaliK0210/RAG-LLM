from typing import Any

from ragforge.agent.registry import (
    ToolRegistry,
)
from ragforge.agent.tool import (
    ToolResult,
)


class ToolExecutor:
    """
    Executes tools selected by the agent.
    """

    def __init__(
        self,
        registry: ToolRegistry,
    ) -> None:
        self.registry = registry

    def execute(
        self,
        tool_name: str,
        tool_input: dict[str, Any] | None = None,
    ) -> ToolResult:
        if not isinstance(
            tool_name,
            str,
        ):
            raise TypeError(
                "tool_name must be a string."
            )

        if not tool_name.strip():
            raise ValueError(
                "tool_name cannot be empty."
            )

        if tool_input is None:
            tool_input = {}

        if not isinstance(
            tool_input,
            dict,
        ):
            raise TypeError(
                "tool_input must be a dictionary."
            )

        tool = self.registry.get(
            tool_name
        )

        try:
            return tool.run(
                **tool_input
            )

        except Exception as exc:
            return ToolResult(
                tool_name=tool_name,
                output=None,
                success=False,
                error=str(exc),
            )
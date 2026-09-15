from typing import Any

from ragforge.agent.tool import (
    Tool,
)


class ToolRegistry:
    """
    Registry of tools available to an agent.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(
        self,
        tool: Tool,
    ) -> None:
        if not isinstance(
            tool,
            Tool,
        ):
            raise TypeError(
                "tool must inherit from Tool."
            )

        if tool.name in self._tools:
            raise ValueError(
                f"Tool already registered: "
                f"{tool.name}"
            )

        if not tool.name.strip():
            raise ValueError(
                "Tool name cannot be empty."
            )

        self._tools[tool.name] = tool

    def get(
        self,
        name: str,
    ) -> Tool:
        if name not in self._tools:
            raise KeyError(
                f"Unknown tool: {name}"
            )

        return self._tools[name]

    def has(
        self,
        name: str,
    ) -> bool:
        return name in self._tools

    def list_tools(self) -> list[Tool]:
        return list(
            self._tools.values()
        )

    def schemas(self) -> list[dict[str, Any]]:
        return [
            tool.schema()
            for tool in self._tools.values()
        ]
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolResult:
    """
    Result returned by a tool execution.
    """

    tool_name: str
    output: Any
    success: bool = True
    error: str | None = None


class Tool(ABC):
    """
    Abstract interface for agent tools.
    """

    name: str
    description: str

    @abstractmethod
    def run(
        self,
        **kwargs: Any,
    ) -> ToolResult:
        """
        Execute the tool.
        """
        raise NotImplementedError

    def schema(self) -> dict[str, Any]:
        """
        Return an LLM-compatible tool description.
        """

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {},
        }
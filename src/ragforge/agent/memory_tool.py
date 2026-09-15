from typing import Any

from ragforge.agent.tool import Tool, ToolResult
from ragforge.memory.manager import MemoryManager


class MemoryRecallTool(Tool):
    """Agent tool for recalling relevant long-term memories."""

    name = "memory_recall"
    description = (
        "Recall relevant information stored in long-term memory."
    )

    def __init__(
        self,
        memory_manager: MemoryManager,
        top_k: int = 3,
    ) -> None:
        if not isinstance(
            memory_manager,
            MemoryManager,
        ):
            raise TypeError(
                "memory_manager must be a MemoryManager."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        self.memory_manager = memory_manager
        self.top_k = top_k

    def run(
        self,
        query: str,
        **kwargs: Any,
    ) -> ToolResult:
        if not isinstance(query, str):
            raise TypeError(
                "query must be a string."
            )

        if not query.strip():
            raise ValueError(
                "query cannot be empty."
            )

        try:
            memories = self.memory_manager.recall(
                query=query,
                top_k=self.top_k,
            )

            return ToolResult(
                tool_name=self.name,
                output=[
                    {
                        "content": memory.content,
                        "memory_type": memory.memory_type,
                        "metadata": memory.metadata,
                    }
                    for memory in memories
                ],
                success=True,
            )

        except Exception as exc:
            return ToolResult(
                tool_name=self.name,
                output=None,
                success=False,
                error=str(exc),
            )

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Information to recall from "
                            "long-term memory."
                        ),
                    }
                },
                "required": ["query"],
            },
        }
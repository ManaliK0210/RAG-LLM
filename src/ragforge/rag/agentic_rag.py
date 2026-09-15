from typing import Any

from ragforge.agent.executor import ToolExecutor
from ragforge.agent.registry import ToolRegistry
from ragforge.memory.manager import MemoryManager
from ragforge.rag.agentic_config import AgenticRAGConfig
from ragforge.rag.agentic_planner import (
    AgenticRAGPlanner,
)
from ragforge.rag.agentic_response import (
    AgenticRAGResponse,
)


class AgenticRAG:
    """
    Agentic RAG orchestration system.

    The planner decides whether to use retrieval, memory,
    or finish with a final answer.
    """

    def __init__(
        self,
        planner: AgenticRAGPlanner,
        registry: ToolRegistry,
        executor: ToolExecutor,
        memory_manager: MemoryManager | None = None,
        config: AgenticRAGConfig | None = None,
    ) -> None:
        if not isinstance(
            planner,
            AgenticRAGPlanner,
        ):
            raise TypeError(
                "planner must be an AgenticRAGPlanner."
            )

        if not isinstance(
            registry,
            ToolRegistry,
        ):
            raise TypeError(
                "registry must be a ToolRegistry."
            )

        if not isinstance(
            executor,
            ToolExecutor,
        ):
            raise TypeError(
                "executor must be a ToolExecutor."
            )

        if memory_manager is not None and not isinstance(
            memory_manager,
            MemoryManager,
        ):
            raise TypeError(
                "memory_manager must be a MemoryManager."
            )

        self.planner = planner
        self.registry = registry
        self.executor = executor
        self.memory_manager = memory_manager
        self.config = (
            config
            if config is not None
            else AgenticRAGConfig()
        )

    def run(
        self,
        query: str,
    ) -> AgenticRAGResponse:
        if not isinstance(query, str):
            raise TypeError(
                "query must be a string."
            )

        if not query.strip():
            raise ValueError(
                "query cannot be empty."
            )

        observations: list[Any] = []

        sources: list[str] = []

        used_retrieval = False
        used_memory = False

        for step_number in range(
            1,
            self.config.max_steps + 1,
        ):
            action = self.planner.plan(
                query=query,
                observations=observations,
            )

            if action.action == "finish":
                return AgenticRAGResponse(
                    answer=action.final_answer,
                    sources=sources,
                    steps=step_number - 1,
                    used_retrieval=used_retrieval,
                    used_memory=used_memory,
                    metadata={
                        "completed": True,
                    },
                )

            result = self.executor.execute(
                tool_name=action.tool_name,
                tool_input=action.tool_input or {},
            )

            if not result.success:
                observation = {
                    "tool": result.tool_name,
                    "success": False,
                    "error": result.error,
                }
            else:
                observation = {
                    "tool": result.tool_name,
                    "success": True,
                    "output": result.output,
                }

            observations.append(observation)

            if action.action == "retrieve":
                used_retrieval = True

                if result.success:
                    output = result.output

                    if isinstance(
                        output,
                        dict,
                    ):
                        tool_sources = output.get(
                            "sources",
                            [],
                        )

                        if isinstance(
                            tool_sources,
                            list,
                        ):
                            sources.extend(
                                str(source)
                                for source in tool_sources
                            )

            elif action.action == "memory":
                used_memory = True

        return AgenticRAGResponse(
            answer=(
                "The agent reached its maximum number "
                "of reasoning steps without producing "
                "a final answer."
            ),
            sources=list(
                dict.fromkeys(sources)
            ),
            steps=self.config.max_steps,
            used_retrieval=used_retrieval,
            used_memory=used_memory,
            metadata={
                "completed": False,
                "max_steps_reached": True,
            },
        )
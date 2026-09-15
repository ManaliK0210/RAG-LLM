from ragforge.agent.config import (
    AgentConfig,
)
from ragforge.agent.executor import (
    ToolExecutor,
)
from ragforge.agent.planner import (
    AgentAction,
    Planner,
)
from ragforge.agent.state import (
    AgentState,
    AgentStep,
)


class Agent:
    """
    Agent that follows a plan → execute → observe loop.
    """

    def __init__(
        self,
        planner: Planner,
        executor: ToolExecutor,
        config: AgentConfig | None = None,
    ) -> None:
        self.planner = planner
        self.executor = executor

        self.config = (
            config
            if config is not None
            else AgentConfig()
        )

    def run(
        self,
        query: str,
    ) -> AgentState:
        if not isinstance(
            query,
            str,
        ):
            raise TypeError(
                "query must be a string."
            )

        if not query.strip():
            raise ValueError(
                "query cannot be empty."
            )

        state = AgentState(
            query=query
        )

        observations = []

        for _ in range(
            self.config.max_steps
        ):
            action = self.planner.plan(
                query=query,
                observations=observations,
                available_tools=(
                    self.executor.registry.schemas()
                ),
            )

            if not isinstance(
                action,
                AgentAction,
            ):
                raise TypeError(
                    "Planner must return "
                    "an AgentAction."
                )

            if action.action == "finish":
                state.final_answer = (
                    action.final_answer
                    or ""
                )

                state.completed = True

                return state

            if action.action != "tool":
                raise ValueError(
                    f"Unknown agent action: "
                    f"{action.action}"
                )

            if not action.tool_name:
                raise ValueError(
                    "Tool action requires "
                    "a tool_name."
                )

            result = self.executor.execute(
                action.tool_name,
                action.tool_input,
            )

            observation = (
                result.output
                if result.success
                else f"Tool error: {result.error}"
            )

            observations.append(
                observation
            )

            state.add_step(
                AgentStep(
                    action="tool",
                    tool_name=action.tool_name,
                    tool_input=(
                        action.tool_input
                        or {}
                    ),
                    observation=observation,
                )
            )

        state.final_answer = (
            "Agent stopped because the "
            "maximum number of steps was reached."
        )

        state.completed = False

        return state
import json
from typing import Any

from ragforge.agent.planner import (
    AgentAction,
)
from ragforge.genai.structured import (
    StructuredOutputError,
    extract_json,
)


class LLMPlanner:
    """
    LLM-backed planner.

    The LLM is instructed to return a structured
    action describing either a tool call or a final answer.
    """

    def __init__(
        self,
        llm,
    ) -> None:
        self.llm = llm

    def plan(
        self,
        query: str,
        observations: list[Any],
        available_tools: list[dict[str, Any]],
    ) -> AgentAction:
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

        prompt = self._build_prompt(
            query,
            observations,
            available_tools,
        )

        output = self.llm.generate(
            prompt
        )

        try:
            data = extract_json(
                output
            )

        except (
            json.JSONDecodeError,
            StructuredOutputError,
        ) as exc:
            raise ValueError(
                "LLM planner returned invalid "
                "structured output."
            ) from exc

        return self._parse_action(
            data
        )

    @staticmethod
    def _build_prompt(
        query: str,
        observations: list[Any],
        available_tools: list[dict[str, Any]],
    ) -> str:
        tools_json = json.dumps(
            available_tools,
            indent=2,
        )

        observations_json = json.dumps(
            observations,
            default=str,
            indent=2,
        )

        return (
            "You are an agent planner.\n\n"
            "Decide whether to call a tool or "
            "return a final answer.\n\n"
            "Available tools:\n"
            f"{tools_json}\n\n"
            "Previous observations:\n"
            f"{observations_json}\n\n"
            "User query:\n"
            f"{query}\n\n"
            "Return JSON only.\n"
            "For a tool call:\n"
            '{"action":"tool",'
            '"tool_name":"name",'
            '"tool_input":{}}\n\n'
            "For a final answer:\n"
            '{"action":"finish",'
            '"final_answer":"answer"}'
        )

    @staticmethod
    def _parse_action(
        data: dict[str, Any],
    ) -> AgentAction:
        if not isinstance(
            data,
            dict,
        ):
            raise ValueError(
                "Planner output must be a JSON object."
            )

        action = data.get(
            "action"
        )

        if action == "tool":
            tool_name = data.get(
                "tool_name"
            )

            tool_input = data.get(
                "tool_input",
                {},
            )

            if not isinstance(
                tool_name,
                str,
            ) or not tool_name.strip():
                raise ValueError(
                    "Tool action requires "
                    "a valid tool_name."
                )

            if not isinstance(
                tool_input,
                dict,
            ):
                raise ValueError(
                    "tool_input must be an object."
                )

            return AgentAction(
                action="tool",
                tool_name=tool_name,
                tool_input=tool_input,
            )

        if action == "finish":
            final_answer = data.get(
                "final_answer"
            )

            if not isinstance(
                final_answer,
                str,
            ):
                raise ValueError(
                    "Finish action requires "
                    "a final_answer."
                )

            return AgentAction(
                action="finish",
                final_answer=final_answer,
            )

        raise ValueError(
            f"Unknown planner action: {action}"
        )
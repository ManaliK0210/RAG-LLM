from dataclasses import dataclass
from typing import Any

from ragforge.genai.structured import extract_json


@dataclass(frozen=True)
class AgenticAction:
    """Action selected by the agentic RAG planner."""

    action: str
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    final_answer: str | None = None


class AgenticRAGPlanner:
    """
    LLM-driven planner for agentic RAG.

    The planner decides whether to use retrieval, memory,
    or produce the final answer.
    """

    VALID_ACTIONS = {
        "retrieve",
        "memory",
        "finish",
    }

    def __init__(self, llm: Any) -> None:
        if llm is None:
            raise ValueError(
                "llm cannot be None."
            )

        self.llm = llm

    def _build_prompt(
        self,
        query: str,
        observations: list[Any],
    ) -> str:
        observation_text = (
            "\n".join(
                f"{index + 1}. {observation}"
                for index, observation
                in enumerate(observations)
            )
            if observations
            else "No observations yet."
        )

        return (
            "You are an agentic RAG planner.\n\n"
            "Choose exactly one action:\n"
            '- {"action": "retrieve", '
            '"tool_input": {"query": "..."}}\n'
            '- {"action": "memory", '
            '"tool_input": {"query": "..."}}\n'
            '- {"action": "finish", '
            '"final_answer": "..."}\n\n'
            "Use retrieval when external document knowledge "
            "is needed.\n"
            "Use memory when previous user information may "
            "be relevant.\n"
            "Finish when enough information is available.\n\n"
            f"User query:\n{query}\n\n"
            f"Observations:\n{observation_text}\n\n"
            "Return JSON only."
        )

    def plan(
        self,
        query: str,
        observations: list[Any],
    ) -> AgenticAction:
        if not isinstance(query, str):
            raise TypeError(
                "query must be a string."
            )

        if not query.strip():
            raise ValueError(
                "query cannot be empty."
            )

        if not isinstance(observations, list):
            raise TypeError(
                "observations must be a list."
            )

        output = self.llm.generate(
            self._build_prompt(
                query=query,
                observations=observations,
            )
        )

        try:
            data = extract_json(output)
        except Exception as exc:
            raise ValueError(
                "LLM planner returned invalid JSON."
            ) from exc

        if not isinstance(data, dict):
            raise ValueError(
                "Planner output must be a JSON object."
            )

        action = data.get("action")

        if action not in self.VALID_ACTIONS:
            raise ValueError(
                f"Invalid action: {action}"
            )

        if action in {
            "retrieve",
            "memory",
        }:
            tool_input = data.get(
                "tool_input"
            )

            if not isinstance(
                tool_input,
                dict,
            ):
                raise ValueError(
                    "tool_input must be a dictionary."
                )

            tool_query = tool_input.get(
                "query"
            )

            if not isinstance(
                tool_query,
                str,
            ):
                raise ValueError(
                    "tool_input.query must be a string."
                )

            if not tool_query.strip():
                raise ValueError(
                    "tool_input.query cannot be empty."
                )

            tool_name = (
                "rag_retrieval"
                if action == "retrieve"
                else "memory_recall"
            )

            return AgenticAction(
                action=action,
                tool_name=tool_name,
                tool_input=tool_input,
            )

        final_answer = data.get(
            "final_answer"
        )

        if not isinstance(
            final_answer,
            str,
        ):
            raise ValueError(
                "final_answer must be a string."
            )

        if not final_answer.strip():
            raise ValueError(
                "final_answer cannot be empty."
            )

        return AgenticAction(
            action="finish",
            final_answer=final_answer,
        )
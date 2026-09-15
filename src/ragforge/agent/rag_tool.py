from typing import Any

from ragforge.agent.tool import Tool, ToolResult
from ragforge.rag.context import ContextBuilder
from ragforge.retrieval.retriever import Retriever


class RAGRetrievalTool(Tool):
    """
    Agent tool for retrieving relevant knowledge.

    The tool performs retrieval only. Final answer generation
    remains the responsibility of the agent/LLM.
    """

    name = "rag_retrieval"
    description = (
        "Retrieve relevant knowledge from the indexed "
        "document collection."
    )

    def __init__(
        self,
        retriever: Retriever,
        context_builder: ContextBuilder | None = None,
        top_k: int = 5,
    ) -> None:
        if not isinstance(
            retriever,
            Retriever,
        ):
            raise TypeError(
                "retriever must be a Retriever."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        self.retriever = retriever
        self.context_builder = (
            context_builder
            if context_builder is not None
            else ContextBuilder()
        )
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
            results = self.retriever.retrieve(
                query=query,
                top_k=self.top_k,
            )

            context = self.context_builder.build(
                results
            )

            sources = [
                result.metadata.get(
                    "source",
                    result.id,
                )
                for result in results
            ]

            return ToolResult(
                tool_name=self.name,
                output={
                    "context": context.text,
                    "sources": sources,
                    "results": results,
                },
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
                            "Question or information need "
                            "to search for."
                        ),
                    }
                },
                "required": ["query"],
            },
        }
from typing import Any

from ragforge.api.dependencies import (
    APIContainer,
)
from ragforge.api.schemas import (
    AgentResponseSchema,
    GenerateResponse,
    MemoryRecallResponse,
    MemoryResponse,
    RAGResponseSchema,
)


class APIService:
    """Application service layer used by FastAPI routes."""

    def __init__(
        self,
        container: APIContainer,
    ) -> None:
        self.container = container

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
    ) -> GenerateResponse:
        if self.container.llm is None:
            raise RuntimeError(
                "LLM service is not configured."
            )

        text = self.container.llm.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

        return GenerateResponse(
            text=text
        )

    def rag(
        self,
        question: str,
    ) -> RAGResponseSchema:
        if self.container.rag_pipeline is None:
            raise RuntimeError(
                "RAG pipeline is not configured."
            )

        response = self.container.rag_pipeline.query(
            question
        )

        return RAGResponseSchema(
            answer=response.answer,
            sources=response.sources,
            retrieved_chunks=(
                response.retrieved_chunks
            ),
            metadata=response.metadata,
        )

    def agent(
        self,
        query: str,
    ) -> AgentResponseSchema:
        if self.container.agent is None:
            raise RuntimeError(
                "Agent service is not configured."
            )

        state = self.container.agent.run(
            query
        )

        return AgentResponseSchema(
            answer=state.final_answer or "",
            completed=state.completed,
            steps=state.step_count,
        )

    def remember(
        self,
        content: str,
        memory_type: str,
        metadata: dict[str, Any],
    ) -> MemoryResponse:
        memory = (
            self.container.memory_manager.remember(
                content=content,
                memory_type=memory_type,
                metadata=metadata,
            )
        )

        return MemoryResponse(
            content=memory.content,
            memory_type=memory.memory_type,
            metadata=memory.metadata,
        )

    def recall(
        self,
        query: str,
        top_k: int,
    ) -> MemoryRecallResponse:
        memories = (
            self.container.memory_manager.recall(
                query=query,
                top_k=top_k,
            )
        )

        return MemoryRecallResponse(
            memories=[
                MemoryResponse(
                    content=memory.content,
                    memory_type=memory.memory_type,
                    metadata=memory.metadata,
                )
                for memory in memories
            ]
        )

    def clear_memory(self) -> None:
        self.container.memory_manager.clear_all()
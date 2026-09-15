from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class GenerateRequest(BaseModel):
    prompt: str = Field(
        min_length=1,
        description="Prompt to send to the language model.",
    )
    max_new_tokens: int = Field(
        default=50,
        gt=0,
        le=512,
    )
    temperature: float = Field(
        default=1.0,
        gt=0.0,
    )
    top_k: int | None = Field(
        default=None,
        gt=0,
    )


class GenerateResponse(BaseModel):
    text: str


class RAGRequest(BaseModel):
    question: str = Field(
        min_length=1,
        description="Question to answer using RAG.",
    )


class RAGResponseSchema(BaseModel):
    answer: str
    sources: list[str]
    retrieved_chunks: int
    metadata: dict[str, Any]


class AgentRequest(BaseModel):
    query: str = Field(
        min_length=1,
        description="Task for the agent.",
    )


class AgentResponseSchema(BaseModel):
    answer: str
    completed: bool
    steps: int


class MemoryRequest(BaseModel):
    content: str = Field(
        min_length=1,
    )
    memory_type: str = Field(
        default="fact",
        min_length=1,
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class MemoryResponse(BaseModel):
    content: str
    memory_type: str
    metadata: dict[str, Any]


class MemoryRecallRequest(BaseModel):
    query: str = Field(
        min_length=1,
    )
    top_k: int = Field(
        default=5,
        gt=0,
    )


class MemoryRecallResponse(BaseModel):
    memories: list[MemoryResponse]
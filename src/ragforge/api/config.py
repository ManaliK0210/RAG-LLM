from dataclasses import dataclass


@dataclass(frozen=True)
class APIConfig:
    """Configuration for the RAGForge API."""

    title: str = "RAGForge API"
    version: str = "0.1.0"
    description: str = (
        "API for the RAGForge Generative AI, "
        "RAG, Agentic AI, and LLM platform."
    )
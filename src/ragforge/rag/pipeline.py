from ragforge.genai.llm import PretrainedLLM
from ragforge.rag.config import RAGConfig
from ragforge.rag.context import ContextBuilder
from ragforge.rag.prompt import build_rag_prompt
from ragforge.rag.response import RAGResponse
from ragforge.retrieval.retriever import Retriever


class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation pipeline.

    Flow:
        question
        -> retrieval
        -> context construction
        -> prompt construction
        -> LLM generation
        -> response with sources
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: PretrainedLLM,
        config: RAGConfig | None = None,
        context_builder: ContextBuilder | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm

        self.config = (
            config
            if config is not None
            else RAGConfig()
        )

        self.context_builder = (
            context_builder
            if context_builder is not None
            else ContextBuilder()
        )

    def query(
        self,
        question: str,
    ) -> RAGResponse:
        """
        Answer a question using retrieved context.
        """

        if not isinstance(question, str):
            raise TypeError(
                "question must be a string."
            )

        if not question.strip():
            raise ValueError(
                "question cannot be empty."
            )

        results = self.retriever.retrieve(
            question,
            top_k=self.config.top_k,
        )

        context = self.context_builder.build(
            results,
            max_chunks=self.config.max_context_chunks,
        )

        prompt = build_rag_prompt(
            question=question,
            context=context.text,
        )

        answer = self.llm.generate(
            prompt
        )

        return RAGResponse(
            answer=answer,
            sources=context.sources,
            retrieved_chunks=len(
                context.chunks
            ),
            metadata={
                "top_k": self.config.top_k,
            },
        )
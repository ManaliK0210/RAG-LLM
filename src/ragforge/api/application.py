from pathlib import Path

from ragforge.api.dependencies import (
    APIContainer,
    set_container,
)
from ragforge.embeddings.config import (
    EmbeddingConfig,
)
from ragforge.embeddings.model import (
    SentenceTransformerEmbedding,
)
from ragforge.embeddings.pipeline import (
    EmbeddingPipeline,
)
from ragforge.genai.llm import (
    LLMConfig,
    PretrainedLLM,
)
from ragforge.rag.config import (
    RAGConfig,
)
from ragforge.rag.pipeline import (
    RAGPipeline,
)
from ragforge.retrieval.config import (
    RetrievalConfig,
)
from ragforge.retrieval.numpy_store import (
    NumpyVectorStore,
)
from ragforge.retrieval.retriever import (
    Retriever,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

VECTOR_STORE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "vector_store.pkl"
)


def create_api_container() -> APIContainer:
    """
    Build the production API dependency container.

    Services are created when the container is requested,
    rather than at module import time.
    """

    llm = PretrainedLLM(
        LLMConfig(
            model_name="distilgpt2",
            device="cpu",
        )
    )

    embedding_model = (
        SentenceTransformerEmbedding(
            EmbeddingConfig(
                device="cpu",
            )
        )
    )

    embedding_pipeline = (
        EmbeddingPipeline(
            model=embedding_model,
        )
    )

    if VECTOR_STORE_PATH.exists():
        vector_store = (
            NumpyVectorStore.load(
                VECTOR_STORE_PATH
            )
        )
    else:
        vector_store = (
            NumpyVectorStore(
                dimension=embedding_model.dimension
            )
        )

    retriever = Retriever(
        embedding_pipeline=embedding_pipeline,
        vector_store=vector_store,
        config=RetrievalConfig(
            top_k=5,
        ),
    )

    rag_pipeline = RAGPipeline(
        retriever=retriever,
        llm=llm,
        config=RAGConfig(
            top_k=5,
            max_context_chunks=5,
        ),
    )

    return APIContainer(
        llm=llm,
        rag_pipeline=rag_pipeline,
    )


def configure_production_services() -> APIContainer:
    """
    Create and register production API services.
    """

    container = create_api_container()

    set_container(container)

    return container
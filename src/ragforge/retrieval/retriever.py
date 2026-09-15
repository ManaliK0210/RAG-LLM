from ragforge.embeddings.pipeline import EmbeddingPipeline
from ragforge.retrieval.config import RetrievalConfig
from ragforge.retrieval.result import SearchResult
from ragforge.retrieval.base import VectorStore


class Retriever:
    """
    Performs semantic retrieval over a vector store.
    """

    def __init__(
        self,
        embedding_pipeline: EmbeddingPipeline,
        vector_store: VectorStore,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.embedding_pipeline = embedding_pipeline
        self.vector_store = vector_store
        self.config = (
            config
            if config is not None
            else RetrievalConfig()
        )

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """
        Retrieve the most relevant documents for a query.
        """

        k = (
            top_k
            if top_k is not None
            else self.config.top_k
        )

        if k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        query_vector = (
            self.embedding_pipeline.embed_query(
                query
            )
        )

        results = self.vector_store.search(
            query_vector=query_vector,
            top_k=k,
        )

        if (
            self.config.score_threshold
            is not None
        ):
            results = [
                result
                for result in results
                if result.score
                >= self.config.score_threshold
            ]

        return results
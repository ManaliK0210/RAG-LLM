from ragforge.rag.advanced_config import (
    AdvancedRAGConfig,
)
from ragforge.rag.hybrid import HybridRetriever
from ragforge.rag.multi_query import (
    MultiQueryGenerator,
)
from ragforge.rag.query_transform import (
    QueryRewriter,
)
from ragforge.rag.reranker import KeywordReranker
from ragforge.retrieval.result import SearchResult
from ragforge.retrieval.retriever import Retriever


class AdvancedRetriever:
    """
    Advanced retrieval orchestration layer.

    Supports:
        - query rewriting
        - multi-query retrieval
        - reranking
        - hybrid scoring
    """

    def __init__(
        self,
        retriever: Retriever,
        config: AdvancedRAGConfig | None = None,
        query_rewriter: QueryRewriter | None = None,
        multi_query_generator: MultiQueryGenerator | None = None,
        reranker: KeywordReranker | None = None,
        hybrid_retriever: HybridRetriever | None = None,
    ) -> None:
        self.retriever = retriever

        self.config = (
            config
            if config is not None
            else AdvancedRAGConfig()
        )

        self.query_rewriter = query_rewriter
        self.multi_query_generator = (
            multi_query_generator
        )
        self.reranker = (
            reranker
            if reranker is not None
            else KeywordReranker()
        )

        self.hybrid_retriever = (
            hybrid_retriever
            if hybrid_retriever is not None
            else HybridRetriever()
        )

    def retrieve(
        self,
        query: str,
    ) -> list[SearchResult]:
        if not isinstance(query, str):
            raise TypeError(
                "query must be a string."
            )

        if not query.strip():
            raise ValueError(
                "query cannot be empty."
            )

        search_query = query

        if self.config.use_query_rewrite:
            if self.query_rewriter is None:
                raise RuntimeError(
                    "Query rewriting is enabled but "
                    "no QueryRewriter was provided."
                )

            search_query = (
                self.query_rewriter.rewrite(
                    query
                )
            )

        queries = [search_query]

        if self.config.use_multi_query:
            if self.multi_query_generator is None:
                raise RuntimeError(
                    "Multi-query retrieval is enabled "
                    "but no MultiQueryGenerator was provided."
                )

            queries = (
                self.multi_query_generator.generate(
                    search_query
                )
            )

        all_results: list[SearchResult] = []

        for current_query in queries:
            results = self.retriever.retrieve(
                current_query,
                top_k=self.config.initial_top_k,
            )

            all_results.extend(results)

        unique_results = self._deduplicate(
            all_results
        )

        if self.config.use_hybrid_search:
            unique_results = (
                self.hybrid_retriever.combine(
                    query=search_query,
                    semantic_results=unique_results,
                    top_k=self.config.initial_top_k,
                )
            )

        if self.config.use_reranking:
            unique_results = (
                self.reranker.rerank(
                    query=search_query,
                    results=unique_results,
                    top_k=self.config.final_top_k,
                )
            )
        else:
            unique_results = unique_results[
                : self.config.final_top_k
            ]

        return unique_results

    @staticmethod
    def _deduplicate(
        results: list[SearchResult],
    ) -> list[SearchResult]:
        seen: set[str] = set()
        unique: list[SearchResult] = []

        for result in results:
            if result.id in seen:
                continue

            seen.add(result.id)
            unique.append(result)

        return unique